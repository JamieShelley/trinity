"""Production-wide local analytic boundary authority for NSAMDR V11.4.

This contract patches the CURRENT V11 production model in-place.  It does not
replace FidelityResidualNetV9, the seam path, the physical-detail decoder, the
boundary specialist, the selector, or the public ``model(inputs)`` graph.

The only semantic change is structural representation:

    old: independently learned local scalar/analytic cells
    new: hard-connectivity marching-squares graph with shared movable nodes

Topology is a bounded 2x control field. Every crossed control edge owns one
shared node/tangent, neighbouring cells reuse that exact node, and deterministic
cubic-Hermite spans are the sole rendered zero-set authority.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from . import model as _model
from .parametric_boundary import LocalParametricBoundaryDecoder, make_query_grid
from .spline_graph import ConnectedSplineGraph

SCHEMA = "NSAMDR_RAVEN_PRODUCTION_BASELINE_RESIDUAL_SPLINE_GRAPH_4X_V11_8_0"

_INSTALLED = False
_ORIGINAL_GEOMETRY_INIT: Callable[..., None] | None = None
_ORIGINAL_SET_PHASE: Callable[..., None] | None = None
_ORIGINAL_ARCHITECTURE_CONTRACT: Callable[..., dict[str, object]] | None = None
_ORIGINAL_COMPUTE_LOSSES: Callable[..., dict[str, torch.Tensor]] | None = None

EVOLUTION_GENOME_NAMES = (
    "feature_gain", "evidence_gain", "distance_scale", "curvature_scale",
    "ribbon_scale", "extra_branch_gain", "csg_logit_scale", "correction_scale",
)
_ACTIVE_EVOLUTION_GENOME: dict[str, float] = {name: 1.0 for name in EVOLUTION_GENOME_NAMES}


class LocalBoundaryProductionContract:
    # Purpose: Implement set active evolution genome for LocalBoundaryProductionContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def set_active_evolution_genome(self, value: dict[str, float] | None) -> None:
        """Select the genome used by subsequently constructed production models.

        This is process-local orchestration state. Every constructed model copies the
        values into a persistent state-dict buffer, so the final checkpoint is
        self-contained and inference never consults this global.
        """
        global _ACTIVE_EVOLUTION_GENOME
        source = value or {}
        _ACTIVE_EVOLUTION_GENOME = {
            name: float(source.get(name, 1.0)) for name in EVOLUTION_GENOME_NAMES
        }

    # Purpose: Implement active evolution genome for LocalBoundaryProductionContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def active_evolution_genome(self) -> dict[str, float]:
        return dict(_ACTIVE_EVOLUTION_GENOME)

    # Purpose: Implement require current v11 instance for LocalBoundaryProductionContract.
    # Called by: _geometry_init
    # Calls: No same-class helper methods.
    def _require_current_v11_instance(self, instance: Any) -> None:
        """Fail clearly if the stale V10.1 replacement from the previous attempt remains."""
        if not hasattr(instance, "parametric_primitive_field"):
            raise RuntimeError(
                "NSAMDR V11 local-boundary override requires the current V11 model.py. "
                "The stale V10.1 model.py from NSAMDR_LOCAL_BOUNDARY_GEOMETRY_SWAP_OVERRIDE "
                "is still present. Run APPLY_OVERRIDE.ps1 to restore model.py from HEAD first."
            )

    # Purpose: Implement geometry init for LocalBoundaryProductionContract.
    # Called by: External callers and the owning workflow.
    # Calls: _require_current_v11_instance
    def _geometry_init(self: Any, config: Any) -> None:
        assert _ORIGINAL_GEOMETRY_INIT is not None
        _ORIGINAL_GEOMETRY_INIT(self, config)
        self._require_current_v11_instance(self)

        widths = config.widths
        # One state-dict path, one production component and one auditable forward.
        # Do not alias this module under a second attribute: that would duplicate
        # state-dict paths and break strict final checkpoint semantics.
        self.production_structure = LocalBoundaryProductionStructure(
            config, int(widths[0])
        )

    # Purpose: Implement geometry encode for LocalBoundaryProductionContract.
    # Called by: _geometry_forward
    # Calls: No same-class helper methods.
    def _geometry_encode(self: Any, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        value = self.stem(inputs)
        skips: list[torch.Tensor] = []
        for index, encoder in enumerate(self.encoders):
            value = encoder(value)
            skips.append(value)
            if index < len(self.downsamples):
                value = self.downsamples[index](value)
        for decoder, skip in zip(self.decoders, reversed(skips[:-1])):
            value = decoder(value, skip)

        source_prior_lr = inputs[:, 16:17].float().clamp(-1.0, 1.0)
        guidance = inputs[:, 9:14].float()
        aux = self.aux_project(value)
        aux = F.interpolate(
            aux, scale_factor=_model.UPSCALE_FACTOR, mode="bilinear", align_corners=False
        )
        prior = self.prior_project(guidance)
        prior = F.interpolate(
            prior, scale_factor=_model.UPSCALE_FACTOR, mode="bilinear", align_corners=False
        )
        aux = self.aux_refine(aux + prior.to(aux.dtype))
        return {
            "source_sdf_prior_lr": source_prior_lr,
            "decoded_feature": value,
            "aux": aux,
        }

    # Purpose: Implement geometry forward for LocalBoundaryProductionContract.
    # Called by: External callers and the owning workflow.
    # Calls: _geometry_encode
    def _geometry_forward(self: Any, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self._geometry_encode(self, inputs)
        aux = context["aux"]
        hr_h = int(inputs.shape[-2]) * int(_model.UPSCALE_FACTOR)
        hr_w = int(inputs.shape[-1]) * int(_model.UPSCALE_FACTOR)
        query_grid = make_query_grid(
            int(inputs.shape[0]), hr_h, hr_w,
            device=inputs.device, dtype=torch.float32,
        )
        # IMPORTANT: invoke the declared structural module through __call__/forward.
        # The production architecture preflight installs a forward hook here.
        structure = self.production_structure(
            context["decoded_feature"], inputs, context["source_sdf_prior_lr"], query_grid
        )
        feature_grid = structure["feature_grid"]
        local_context = structure["context"]
        spline_graph = structure["spline_graph"]
        field = structure["field"]
        structural_residual_gain = structure["structural_residual_gain"].float()

        final_pixels = field["phi_pixels"].float()
        max_distance = float(self.config.contour_sdf_max_distance_pixels)
        source_prior_pixels = field["warped_source_pixels"].float()
        sdf_raw = final_pixels / max(max_distance, 1.0e-6)
        sdf = sdf_raw.clamp(-1.0, 1.0)
        source_prior_hr = source_prior_pixels / max(max_distance, 1.0e-6)

        primitive_normal = field["primitive_normal"].float()
        curvature = field["primitive_curvature"].float()
        zeros = torch.zeros_like(final_pixels)
        zero_source = torch.zeros_like(context["source_sdf_prior_lr"])
        zero_control2 = torch.zeros(
            (inputs.shape[0], 2, inputs.shape[-2] * 2, inputs.shape[-1] * 2),
            device=inputs.device, dtype=aux.dtype,
        )
        zero_hr2 = torch.zeros(
            (inputs.shape[0], 2, final_pixels.shape[-2], final_pixels.shape[-1]),
            device=inputs.device, dtype=aux.dtype,
        )
        zero_gate = torch.full_like(sdf, -8.0)

        # Compatibility telemetry for the retired whole-tile classifier.  It is
        # deliberately zero-authority and is not used by the V11.4 curriculum.
        class_count = int(getattr(_model, "PRIMITIVE_COUNT", 7))
        param_dim = int(getattr(_model, "PARAM_DIM", 12))
        class_logits = final_pixels.new_zeros((inputs.shape[0], class_count))
        class_index = torch.zeros(
            (inputs.shape[0],), device=inputs.device, dtype=torch.int64
        )
        params = final_pixels.new_zeros((inputs.shape[0], param_dim))
        params_by_class = final_pixels.new_zeros(
            (inputs.shape[0], class_count, param_dim)
        )
        spline_keys = (
            "spline_graph_control_phi_pixels",
            "spline_graph_source_control_phi_pixels",
            "spline_control_point_h_lr",
            "spline_control_point_v_lr",
            "spline_source_control_point_h_lr",
            "spline_source_control_point_v_lr",
            "spline_control_tangent_h",
            "spline_control_tangent_v",
            "spline_control_displacement_h_lr",
            "spline_control_displacement_v_lr",
            "spline_graph_mask_h",
            "spline_graph_mask_v",
        )
        spline_outputs = {
            key: spline_graph[key].to(aux.dtype) for key in spline_keys
        }

        return {
            "sdf": sdf.to(aux.dtype),
            "sdf_raw": sdf_raw.to(aux.dtype),
            "source_sdf_prior": source_prior_hr.to(aux.dtype),
            "source_sdf_prior_pixels": source_prior_pixels.to(aux.dtype),
            "implicit_feature_grid": feature_grid,
            "implicit_source_sdf_prior_lr": context["source_sdf_prior_lr"],
            **spline_outputs,
            "primitive_normal": primitive_normal.to(aux.dtype),
            "primitive_curvature_hr": curvature.to(aux.dtype),
            "primitive_phi_pixels": field["primitive_phi_pixels"].to(aux.dtype),
            "parametric_anchor_distance_pixels": local_context["anchor_distance_pixels"].to(aux.dtype),
            "parametric_distance_delta_pixels": local_context["distance_delta_pixels"].to(aux.dtype),
            "local_branch_anchor_distance_pixels": local_context["branch_anchor_distance_pixels"].to(aux.dtype),
            "local_branch_normal_x": local_context["branch_normal_x"].to(aux.dtype),
            "local_branch_normal_y": local_context["branch_normal_y"].to(aux.dtype),
            "local_branch_curvature_per_pixel": local_context["branch_curvature_per_pixel"].to(aux.dtype),
            "local_branch_half_width_pixels": local_context["branch_half_width_pixels"].to(aux.dtype),
            "local_branch_ribbon_mode": local_context["branch_ribbon_mode"].to(aux.dtype),
            "local_branch_activation": local_context["branch_activation"].to(aux.dtype),
            "local_csg_logits": local_context["csg_logits"].to(aux.dtype),
            "local_parametric_confidence": local_context["confidence"].to(aux.dtype),
            "local_implicit_authority": field["implicit_authority"].to(aux.dtype),
            "structural_residual_gain": structural_residual_gain.to(aux.dtype),
            "contour_transport_control_pixels": zero_control2,
            "contour_transport_pixels": zero_hr2,
            "contour_dilation_control_pixels": zero_control2[:, 0:1],
            "contour_dilation_pixels": zeros.to(aux.dtype),
            "implicit_residual_pixels": field["residual_pixels"].to(aux.dtype),
            "implicit_direct_delta_pixels": field["direct_delta_pixels"].to(aux.dtype),
            "contour_normal_offset_source_pixels": zero_source.to(aux.dtype),
            "contour_normal_offset_coarse_pixels": zeros.to(aux.dtype),
            "contour_phase_offset_pixels": zeros.to(aux.dtype),
            "contour_normal_offset_pixels": zeros.to(aux.dtype),
            "coarse_sdf": sdf.to(aux.dtype),
            "coarse_sdf_pixels": final_pixels.to(aux.dtype),
            "coarse_sdf_delta_pixels": (final_pixels - source_prior_pixels).to(aux.dtype),
            "sdf_delta_pixels": (final_pixels - source_prior_pixels).to(aux.dtype),
            "sdf_residual_pixels": (final_pixels - source_prior_pixels).to(aux.dtype),
            "parametric_primitive_active": final_pixels.new_tensor(0.0).to(aux.dtype),
            "primitive_class_logits": class_logits.to(aux.dtype),
            "primitive_class_index": class_index,
            "primitive_params": params.to(aux.dtype),
            "primitive_params_by_class": params_by_class.to(aux.dtype),
            "primitive_confidence": F.interpolate(
                local_context["confidence"].float(), size=final_pixels.shape[-2:],
                mode="bilinear", align_corners=False,
            ).to(aux.dtype),
            "orientation_raw": self.orientation_head(aux),
            "edge_logits": self.edge_head(aux),
            "hardness_logits": self.hardness_head(aux),
            "boundary_gate_logits": zero_gate.to(aux.dtype),
        }

    # Purpose: Query the exact checkpointed connected spline graph.
    # Called by: Boundary supersampling and production redraw.
    # Calls: ConnectedSplineGraph.query.
    def _geometry_query_from_outputs(
        self: Any,
        outputs: dict[str, torch.Tensor],
        query_grid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        graph = {
            "spline_graph_control_phi_pixels": outputs["spline_graph_control_phi_pixels"],
            "spline_graph_source_control_phi_pixels": outputs["spline_graph_source_control_phi_pixels"],
            "spline_control_point_h_lr": outputs["spline_control_point_h_lr"],
            "spline_control_point_v_lr": outputs["spline_control_point_v_lr"],
            "spline_source_control_point_h_lr": outputs["spline_source_control_point_h_lr"],
            "spline_source_control_point_v_lr": outputs["spline_source_control_point_v_lr"],
            "spline_control_tangent_h": outputs["spline_control_tangent_h"],
            "spline_control_tangent_v": outputs["spline_control_tangent_v"],
            "spline_control_displacement_h_lr": outputs["spline_control_displacement_h_lr"],
            "spline_control_displacement_v_lr": outputs["spline_control_displacement_v_lr"],
            "spline_graph_mask_h": outputs["spline_graph_mask_h"],
            "spline_graph_mask_v": outputs["spline_graph_mask_v"],
            "source_sdf_prior_lr": outputs["implicit_source_sdf_prior_lr"],
        }
        return self.production_structure.spline_graph.query(graph, query_grid)

    # Purpose: Freeze the B1a topology producer while retaining geometric refinement.
    # Called by: _set_phase.
    # Calls: LocalBoundaryProductionStructure.lock_topology_for_proof().
    def _lock_proof_topology(self, model: Any) -> None:
        structure = model.geometry_net.production_structure
        structure.lock_topology_for_proof()
        for module in (
            model.geometry_net.stem,
            model.geometry_net.encoders,
            model.geometry_net.downsamples,
            model.geometry_net.decoders,
        ):
            model._set_trainable(module, False)

    # Purpose: Implement set phase for LocalBoundaryProductionContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _set_phase(self: Any, phase: str) -> None:
        assert _ORIGINAL_SET_PHASE is not None
        _ORIGINAL_SET_PHASE(self, phase)
        # The retired whole-tile primitive field remains in the state_dict only for
        # source compatibility. It never has optimiser authority in V11.4.
        self._set_trainable(self.geometry_net.parametric_primitive_field, False)
        self._set_trainable(self.geometry_net.production_structure, False)
        if phase in {"sdf-bootstrap", "sdf-proof"}:
            self._set_trainable(self.geometry_net.production_structure, True)
            # Geometry context is learned jointly with the local analytic actuator
            # during B1a. B1b keeps the qualified topology-producing feature path
            # fixed and refines only continuous geometric output rows.
            self._set_trainable(self.geometry_net.stem, True)
            self._set_trainable(self.geometry_net.encoders, True)
            self._set_trainable(self.geometry_net.downsamples, True)
            self._set_trainable(self.geometry_net.decoders, True)
            self._set_trainable(self.geometry_net.aux_project, True)
            self._set_trainable(self.geometry_net.prior_project, True)
            self._set_trainable(self.geometry_net.aux_refine, True)
            self._set_trainable(self.geometry_net.orientation_head, True)
            self._set_trainable(self.geometry_net.edge_head, True)
            self._set_trainable(self.geometry_net.hardness_head, True)
            if phase == "sdf-bootstrap":
                self.geometry_net.production_structure.unlock_topology_for_bootstrap()
            else:
                _local_boundary_production_contract._lock_proof_topology(self)

    # Purpose: Implement set parametric substage for LocalBoundaryProductionContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _set_parametric_substage(self: Any, substage: str) -> None:
        # V11.4 has no whole-tile classifier/regressor substage. Keep the method for
        # old callers, but point any structural request at the same production local
        # boundary actuator.
        if substage not in {"classifier", "parameters", "integration", "local"}:
            raise ValueError(f"unsupported local-boundary substage: {substage}")
        self._set_trainable(self.geometry_net.parametric_primitive_field, False)
        if not self.geometry_net.production_structure.topology_locked():
            self._set_trainable(self.geometry_net.production_structure, True)

    # Purpose: Implement architecture contract for LocalBoundaryProductionContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _architecture_contract(self: Any) -> dict[str, object]:
        assert _ORIGINAL_ARCHITECTURE_CONTRACT is not None
        contract = dict(_ORIGINAL_ARCHITECTURE_CONTRACT(self))
        contract.update({
            "schema": SCHEMA,
            "geometryPrediction": (
                "bounded 2x topology field -> shared edge-crossing nodes -> "
                "connected cubic-Hermite contour graph -> metric SDF"
            ),
            "reconstructionPrimitive": (
                "deterministic B + zero-initialized bounded structural residual gain "
                "* (connected marching-squares cubic-Hermite redraw - B)"
            ),
            "b1bObjective": "shared graph-node, tangent, span-smoothness, metric-SDF and same-renderer reconstruction",
            "geometryOutputs": (
                "source_sdf_prior", "connected_spline_graph", "structural_residual_gain",
                "edge", "orientation", "hardness"
            ),
            "baselineRelativeStructuralIdentity": True,
            "topologyGeometryFeatureSplit": True,
            "finiteWidthStrokeRepresentation": (
                "paired connected material-boundary spline branches; topology is structural"
            ),
            "connectedSplineGraphAuthority": True,
            "splineGraphControlScale": int(getattr(self.config, "spline_graph_control_scale", 2)),
            "wholeTilePrimitiveClassifierAuthority": False,
            "localBoundaryControlScale": int(getattr(self.config, "parametric_boundary_control_scale", 1)),
            "evolutionaryRecovery": "training-only bounded genome over one fixed production supernet",
            "evolutionaryGenomeFields": EVOLUTION_GENOME_NAMES,
            "evolutionaryGenomeCheckpointed": True,
            "evolutionaryControllerInferenceAuthority": False,
        })
        production = dict(contract.get("productionComponents") or {})
        production["structural representation"] = "geometry_net.production_structure"
        contract["productionComponents"] = production
        # Replace the obsolete primitive-class proof labels without touching the
        # downstream B3/B4/detail/selector stages.
        stages = list(contract.get("stagedProofs") or ())
        stages = [
            "B1-connected-spline-graph" if str(item) == "B1b-parametric-primitive" else item
            for item in stages
        ]
        contract["stagedProofs"] = tuple(stages)
        return contract

    # Purpose: Implement production component modules for LocalBoundaryProductionContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _production_component_modules(
        self,
        model: Any,
    ) -> dict[str, tuple[str, nn.Module]]:
        """Return the exact component-map shape expected by the canonical trainer.

        training.py consumes each value as ``(module_path, module)`` so it can
        register forward hooks, attribute gradients/weight deltas to the correct
        state-dict prefix, and persist architecture participation evidence.
        V11.2 returned bare modules here, which caused the training process to abort
        before its first epoch even though the standalone architecture hook test
        passed.
        """
        return {
            "geometry": ("geometry_net", model.geometry_net),
            "structural representation": (
                "geometry_net.production_structure",
                model.geometry_net.production_structure,
            ),
            "boundary renderer": ("boundary_renderer", model.boundary_renderer),
            "boundary/profile": ("boundary_specialist", model.boundary_specialist),
            "PhaseAwareSeamSR": ("seam_restorer.phase_sr", model.seam_restorer.phase_sr),
            "seam authority": ("seam_restorer.authority", model.seam_restorer.authority),
            "conditioned detail": ("detail_net", model.detail_net),
            "albedo physical head": ("detail_net.albedo_head", model.detail_net.albedo_head),
            "normal physical head": ("detail_net.normal_head", model.detail_net.normal_head),
            "material physical head": (
                "detail_net.material_head",
                model.detail_net.material_head,
            ),
            "confidence": ("detail_net.confidence_head", model.detail_net.confidence_head),
            "regret": ("detail_net.regret_head", model.detail_net.regret_head),
            "BenefitSelector": ("benefit_selector", model.benefit_selector),
        }

    # Purpose: Implement validate local architecture for LocalBoundaryProductionContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _validate_local_architecture(self, contract: dict[str, object]) -> None:
        production = contract.get("productionComponents")
        if not isinstance(production, dict):
            raise RuntimeError("V11.4 local-boundary architecture contract has no productionComponents map")
        required = {
            "geometry": "geometry_net",
            "structural representation": "geometry_net.production_structure",
            "boundary renderer": "boundary_renderer",
            "boundary/profile": "boundary_specialist",
            "PhaseAwareSeamSR": "seam_restorer.phase_sr",
            "seam authority": "seam_restorer.authority",
            "conditioned detail": "detail_net",
            "albedo physical head": "detail_net.albedo_head",
            "normal physical head": "detail_net.normal_head",
            "material physical head": "detail_net.material_head",
            "confidence": "detail_net.confidence_head",
            "regret": "detail_net.regret_head",
            "BenefitSelector": "benefit_selector",
        }
        for label, path in required.items():
            if production.get(label) != path:
                raise RuntimeError(
                    f"V11.4 local-boundary architecture contract mismatch for {label}: "
                    f"{production.get(label)!r} != {path!r}"
                )
        if contract.get("schema") != SCHEMA:
            raise RuntimeError(f"V11.4 local-boundary schema mismatch: {contract.get('schema')!r}")
        if bool(contract.get("geometryCanPaintRgb")):
            raise RuntimeError("V11.4 geometry must not paint RGB")
        if not bool(contract.get("directionalSeamEnabled")):
            raise RuntimeError("V11.4 directional seam path must remain active")
        if not bool(contract.get("detailReconstructionEnabled")):
            raise RuntimeError("V11.4 full-resolution detail path must remain active")
        if production.get("structural representation") != "geometry_net.production_structure":
            raise RuntimeError("V11.4 local structural authority must be the auditable production wrapper")
        if not bool(contract.get("evolutionaryGenomeCheckpointed")):
            raise RuntimeError("V11.4 production checkpoint must contain the locked evolution genome")
        if bool(contract.get("evolutionaryControllerInferenceAuthority")):
            raise RuntimeError("V11.4 evolutionary controller must have no inference authority")

    # Purpose: Implement local compute losses for LocalBoundaryProductionContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def _local_compute_losses(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        config: Any,
        phase: str,
    ) -> dict[str, torch.Tensor]:
        assert _ORIGINAL_COMPUTE_LOSSES is not None
        if phase in {"sdf-bootstrap", "sdf-proof"}:
            required_spline_outputs = (
                "spline_graph_control_phi_pixels",
                "spline_graph_source_control_phi_pixels",
                "spline_control_point_h_lr",
                "spline_control_point_v_lr",
                "spline_source_control_point_h_lr",
                "spline_source_control_point_v_lr",
                "spline_control_tangent_h",
                "spline_control_tangent_v",
                "spline_control_displacement_h_lr",
                "spline_control_displacement_v_lr",
                "spline_graph_mask_h",
                "spline_graph_mask_v",
            )
            missing_spline_outputs = [
                key for key in required_spline_outputs if key not in outputs
            ]
            if missing_spline_outputs:
                raise RuntimeError(
                    "V11.5 connected-spline B1 supervision is disconnected from "
                    "the production forward; missing outputs="
                    f"{missing_spline_outputs}"
                )

        losses = _ORIGINAL_COMPUTE_LOSSES(outputs, batch, config, phase)
        if phase not in {"sdf-bootstrap", "sdf-proof"}:
            return losses

        if phase == "sdf-proof":
            anchor = outputs.get("parametric_anchor_distance_pixels")
            source_prior_pixels = outputs.get("source_sdf_prior_pixels")
            target_sdf = batch.get("target_sdf")
            if anchor is None or source_prior_pixels is None or target_sdf is None:
                raise RuntimeError(
                    "V11.4 sdf-proof requires analytic anchor, source prior and target SDF"
                )
            max_distance = float(config.contour_sdf_max_distance_pixels)
            raw_target_pixels = target_sdf.float() * max_distance
            source_prior_pixels = source_prior_pixels.float()
            if bool(config.sdf_sign_gauge_invariant):
                band = max(float(config.sdf_metric_band_pixels), 1.0e-3)
                gauge_weight = torch.exp(
                    -raw_target_pixels.abs() / max(band * 0.55, 1.0e-3)
                )
                gauge_denom = gauge_weight.sum(
                    dim=(1, 2, 3), keepdim=True
                ).clamp_min(1.0)
                positive = (
                    (source_prior_pixels - raw_target_pixels).abs() * gauge_weight
                ).sum(dim=(1, 2, 3), keepdim=True) / gauge_denom
                negative = (
                    (source_prior_pixels + raw_target_pixels).abs() * gauge_weight
                ).sum(dim=(1, 2, 3), keepdim=True) / gauge_denom
                polarity = torch.where(
                    positive <= negative,
                    torch.ones_like(positive),
                    -torch.ones_like(negative),
                ).detach()
            else:
                polarity = torch.ones(
                    (raw_target_pixels.shape[0], 1, 1, 1),
                    device=raw_target_pixels.device, dtype=raw_target_pixels.dtype,
                )
            target_control = F.interpolate(
                (raw_target_pixels * polarity).detach(),
                size=anchor.shape[-2:], mode="bilinear", align_corners=False,
            )
            anchor_error = anchor.float() - target_control
            error_dx = anchor_error[:, :, :, 1:] - anchor_error[:, :, :, :-1]
            error_dy = anchor_error[:, :, 1:, :] - anchor_error[:, :, :-1, :]
            target_x = 0.5 * (
                target_control[:, :, :, 1:].abs()
                + target_control[:, :, :, :-1].abs()
            )
            target_y = 0.5 * (
                target_control[:, :, 1:, :].abs()
                + target_control[:, :, :-1, :].abs()
            )
            weight_x = (0.15 + 1.85 * torch.exp(-target_x / 4.0)).detach()
            weight_y = (0.15 + 1.85 * torch.exp(-target_y / 4.0)).detach()
            smooth_x = F.smooth_l1_loss(
                error_dx, torch.zeros_like(error_dx), beta=0.12, reduction="none"
            )
            smooth_y = F.smooth_l1_loss(
                error_dy, torch.zeros_like(error_dy), beta=0.12, reduction="none"
            )
            losses["parametric_offset_smoothness"] = 0.5 * (
                (smooth_x * weight_x).sum() / weight_x.sum().clamp_min(1.0)
                + (smooth_y * weight_y).sum() / weight_y.sum().clamp_min(1.0)
            )

        # These terms already exist in the canonical loss implementation. V11.4
        # makes them authoritative because the active structural representation is
        # now the local analytic actuator that those losses were written for.
        if phase == "sdf-bootstrap":
            # B1a trains only the differentiable topology/control field plus the
            # final sign/surface checks needed by its topology gate. Node motion,
            # tangent fitting, span regularity and metric refinement belong to B1b.
            total = (
                losses["sdf_surface"] * float(config.sdf_surface_weight)
                + losses["sdf_sign"] * float(config.sdf_sign_weight)
                + losses["sdf_topology_sign"] * float(config.sdf_topology_weight)
                + losses["spline_graph_topology_control"] * float(config.spline_graph_topology_control_weight)
                + losses["spline_graph_topology_sign"] * float(config.spline_graph_topology_sign_weight)
                + losses["edge"] * float(config.edge_weight)
                + losses["edge_sdf_consistency"] * float(config.boundary_edge_sdf_consistency_weight)
                + losses["orientation"] * float(config.orientation_weight)
                + losses["hardness"] * float(config.boundary_hardness_weight)
            )
        else:
            total = (
                losses["sdf_surface"] * float(config.sdf_surface_weight)
                + losses["sdf_sign"] * float(config.sdf_sign_weight)
                + losses["sdf_topology_sign"] * float(config.sdf_topology_weight)
                + losses["spline_graph_topology_control"] * float(config.spline_graph_topology_control_weight)
                + losses["spline_graph_topology_sign"] * float(config.spline_graph_topology_sign_weight)
                + losses["spline_graph_point"] * float(config.spline_graph_point_weight)
                + losses["spline_graph_tangent"] * float(config.spline_graph_tangent_weight)
                + losses["spline_graph_span_smoothness"] * float(config.spline_graph_span_smoothness_weight)
                + losses["spline_graph_span_tangent"] * float(config.spline_graph_span_tangent_weight)
                + losses["spline_graph_span_separation"] * float(config.spline_graph_span_separation_weight)
                + losses["spline_graph_sdf"] * float(config.spline_graph_sdf_weight)
                + losses["spline_graph_gradient"] * float(config.spline_graph_gradient_weight)
                + losses["spline_graph_eikonal"] * float(config.spline_graph_eikonal_weight)
                + losses["spline_graph_curvature"] * float(config.spline_graph_curvature_weight)
                + losses["spline_metric_offset"] * float(config.spline_metric_offset_weight)
                + losses["spline_metric_eikonal_near"] * float(config.spline_metric_eikonal_near_weight)
                + losses["edge"] * float(config.edge_weight)
                + losses["edge_sdf_consistency"] * float(config.boundary_edge_sdf_consistency_weight)
                + losses["orientation"] * float(config.orientation_weight)
                + losses["hardness"] * float(config.boundary_hardness_weight)
            )

        # A structural candidate is useful only when it improves on the observed
        # source/baseline. The canonical loss already computes these differentiable
        # regret terms on both authored Raven and analytic examples; keep them as
        # training authority in both B1a and B1b instead of telemetry-only values.
        baseline_relative_supervision = (
            losses["sdf_improvement_regret"] * float(config.sdf_improvement_regret_weight)
            + losses["geometry_regret"] * float(config.geometry_regret_weight)
            + losses["boundary_pixel_regret"] * float(config.boundary_pixel_regret_weight)
        )
        total = total + baseline_relative_supervision
        losses["baseline_relative_supervision"] = baseline_relative_supervision.detach()

        # B1a establishes topology. B1b/sdf-proof must then spend explicit
        # authority on the subpixel defects that the promotion gate measures.
        # These losses already exist in the canonical objective; V11.4 previously
        # computed them only as telemetry, leaving low-angle stair steps and radial
        # scalloping without a direct differentiable training signal.
        if phase == "sdf-proof":
            total = total + (
                losses["implicit_subpixel_surface"] * float(config.sdf_surface_weight)
                + losses["implicit_subpixel_gradient"] * float(config.sdf_metric_gradient_weight)
                + losses["implicit_subpixel_eikonal"] * float(config.sdf_eikonal_weight)
                + losses["sdf_curvature"] * float(config.sdf_curvature_weight)
            )
            if "sdf_teacher_gradient" in losses:
                total = total + losses["sdf_teacher_gradient"] * float(
                    config.sdf_teacher_gradient_weight
                )
            if "sdf_teacher_profile" in losses:
                total = total + losses["sdf_teacher_profile"] * float(
                    config.sdf_teacher_profile_weight
                )

        # Same-renderer teacher is a direct structural signal, not a separate
        # candidate path. Keep its configured proof authority bounded.
        if "sdf_teacher_render" in losses:
            total = total + losses["sdf_teacher_render"] * float(config.sdf_proof_renderer_weight)
        losses["total"] = total.float()
        return losses

    # Purpose: Implement install local boundary model contract for LocalBoundaryProductionContract.
    # Called by: install_local_boundary_training_contract
    # Calls: No same-class helper methods.
    def install_local_boundary_model_contract(self) -> None:
        global _INSTALLED, _ORIGINAL_GEOMETRY_INIT, _ORIGINAL_SET_PHASE
        global _ORIGINAL_ARCHITECTURE_CONTRACT
        if _INSTALLED:
            return

        _ORIGINAL_GEOMETRY_INIT = _model.GeometryNet.__init__
        _model.GeometryNet.__init__ = _geometry_init
        _model.GeometryNet.encode = _geometry_encode
        _model.GeometryNet.forward = _geometry_forward
        _model.GeometryNet.query_from_outputs = _geometry_query_from_outputs

        _ORIGINAL_SET_PHASE = _model.FidelityResidualNetV9.set_phase
        _model.FidelityResidualNetV9.set_phase = _set_phase
        _model.FidelityResidualNetV9.set_parametric_substage = _set_parametric_substage

        _ORIGINAL_ARCHITECTURE_CONTRACT = _model.FidelityResidualNetV9.architecture_contract
        _model.FidelityResidualNetV9.architecture_contract = _architecture_contract
        _model.MODEL_SCHEMA = SCHEMA
        _INSTALLED = True

    # Purpose: Implement local representation microproof for LocalBoundaryProductionContract.
    # Called by: install_local_boundary_training_contract and TrainingService.train_v9
    # Calls: TrainingService._parametric_structure_microproof through the training module.
    def _local_representation_microproof(
        self,
        device: torch.device,
        config: Any,
    ) -> tuple[float, float, float, float]:
        """Run the existing deterministic local analytic capacity proof.

        This must remain a class/module-level callable. Windows DataLoader workers
        use spawn/pickle; installing a nested closure on the TrainingService singleton
        makes its bound worker initializer unpickleable.
        """
        _ = config
        from . import training as training_module

        return training_module._parametric_structure_microproof(device)

    # Purpose: Implement install local boundary training contract for LocalBoundaryProductionContract.
    # Called by: External callers and the owning workflow.
    # Calls: _local_representation_microproof, install_local_boundary_model_contract
    def install_local_boundary_training_contract(self, training_module: ModuleType) -> None:
        """Redirect current trainer bookkeeping/losses to the local production field."""
        global _ORIGINAL_COMPUTE_LOSSES
        self.install_local_boundary_model_contract()

        if bool(getattr(training_module, "_nsamdr_local_boundary_v114_installed", False)):
            return
        _ORIGINAL_COMPUTE_LOSSES = training_module.compute_losses
        training_module.compute_losses = _local_compute_losses
        training_module.MODEL_SCHEMA = SCHEMA
        training_module._validate_v992_architecture_contract = _validate_local_architecture
        training_module._validate_v991_architecture_contract = _validate_local_architecture
        training_module._validate_v990_architecture_contract = _validate_local_architecture

        # The source tree already contains a deterministic local analytic capacity
        # proof. Reuse it instead of the retired whole-tile primitive proof. Keep
        # the callback pickle-safe because TrainingService also owns the Windows
        # multiprocessing DataLoader worker initializer.
        training_module._explicit_primitive_structure_microproof = (
            _local_representation_microproof
        )
        training_module._production_component_modules = _production_component_modules
        training_module._nsamdr_local_boundary_v114_installed = True

_local_boundary_production_contract = LocalBoundaryProductionContract()
set_active_evolution_genome = _local_boundary_production_contract.set_active_evolution_genome
active_evolution_genome = _local_boundary_production_contract.active_evolution_genome
_require_current_v11_instance = _local_boundary_production_contract._require_current_v11_instance
_geometry_init = _local_boundary_production_contract._geometry_init
_geometry_encode = _local_boundary_production_contract._geometry_encode
_geometry_forward = _local_boundary_production_contract._geometry_forward
_geometry_query_from_outputs = _local_boundary_production_contract._geometry_query_from_outputs
_set_phase = _local_boundary_production_contract._set_phase
_set_parametric_substage = _local_boundary_production_contract._set_parametric_substage
_architecture_contract = _local_boundary_production_contract._architecture_contract
_production_component_modules = _local_boundary_production_contract._production_component_modules
_validate_local_architecture = _local_boundary_production_contract._validate_local_architecture
_local_compute_losses = _local_boundary_production_contract._local_compute_losses
_local_representation_microproof = (
    _local_boundary_production_contract._local_representation_microproof
)
install_local_boundary_model_contract = _local_boundary_production_contract.install_local_boundary_model_contract
install_local_boundary_training_contract = _local_boundary_production_contract.install_local_boundary_training_contract


class LocalBoundaryProductionStructure(nn.Module):
    """Auditable production wrapper around the existing local analytic decoder.

    The architecture auditor observes modules through ``forward`` hooks.  V11.1
    incorrectly called ``LocalParametricBoundaryDecoder.query`` directly, which
    performed the math but bypassed the module's ``forward`` hook and therefore
    made the production structural component look inactive.  This wrapper is
    the one declared production structure and is invoked through ``__call__``.
    """

    # Purpose: Build one independent local structural feature projector.
    # Called by: __init__.
    # Calls: torch.nn.Conv2d(), torch.nn.GELU(), ResidualBlock().
    @staticmethod
    def _make_feature_project(decoded_channels: int, feature_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(int(decoded_channels) + 16, feature_channels, 3, padding=1),
            nn.GELU(),
            _model.ResidualBlock(feature_channels),
            _model.ResidualBlock(feature_channels, dilation=2),
        )

    # Purpose: Implement init for LocalBoundaryProductionStructure.
    # Called by: External callers and the owning workflow.
    # Calls: _make_feature_project.
    def __init__(self, config: Any, decoded_channels: int) -> None:
        super().__init__()
        feature_channels = int(getattr(config, "topology_field_feature_channels", 64))
        self.geometry_feature_project = self._make_feature_project(
            int(decoded_channels), feature_channels
        )
        self.topology_feature_project = self._make_feature_project(
            int(decoded_channels), feature_channels
        )
        # V11.8 structural output is a residual correction to deterministic B.
        # A zero-initialized final layer gives exact C == B at construction while
        # tanh keeps the learned signed correction bounded and differentiable.
        self.structural_residual_gain_head = nn.Sequential(
            nn.Conv2d(int(decoded_channels) + 16, feature_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feature_channels, 1, 1),
        )
        nn.init.zeros_(self.structural_residual_gain_head[-1].weight)
        nn.init.zeros_(self.structural_residual_gain_head[-1].bias)
        initial_genome = torch.tensor(
            [float(_ACTIVE_EVOLUTION_GENOME[name]) for name in EVOLUTION_GENOME_NAMES],
            dtype=torch.float32,
        )
        # Persistent so strict=True checkpoints contain the exact locked genome.
        self.register_buffer("evolution_genome", initial_genome, persistent=True)
        self.decoder = LocalParametricBoundaryDecoder(
            feature_channels,
            int(getattr(config, "topology_field_hidden_channels", 96)),
            max_distance_pixels=float(config.contour_sdf_max_distance_pixels),
            max_offset_pixels=float(getattr(config, "parametric_boundary_max_offset_pixels", 6.0)),
            max_normal_correction=float(getattr(config, "parametric_boundary_max_normal_correction", 1.5)),
            max_curvature_per_pixel=float(getattr(config, "parametric_boundary_max_curvature_per_pixel", 0.35)),
            max_ribbon_half_width_pixels=float(
                getattr(config, "parametric_boundary_max_ribbon_half_width_pixels", 6.0)
            ),
            control_scale=int(getattr(config, "parametric_boundary_control_scale", 1)),
            output_scale=int(getattr(config, "target_scale", _model.UPSCALE_FACTOR)),
        )
        self.spline_graph = ConnectedSplineGraph(feature_channels, config)
        self._topology_bootstrap_only = False

    # Purpose: Report whether B1a topology is locked for proof.
    # Called by: LocalBoundaryProductionContract._set_parametric_substage.
    # Calls: No same-class helper methods.
    def topology_locked(self) -> bool:
        return bool(self.spline_graph._topology_locked)

    # Purpose: Restore full structural trainability for B1a topology bootstrap.
    # Called by: LocalBoundaryProductionContract._set_phase.
    # Calls: PrimitiveParameterHead.unlock_topology().
    def unlock_topology_for_bootstrap(self) -> None:
        # B1a is a topology proof, not an early geometry-refinement phase. Keep
        # continuous node/tangent degrees of freedom exactly at the source crossing
        # geometry until topology is accepted; B1b receives those degrees of freedom.
        self._topology_bootstrap_only = True
        self.spline_graph.unlock_topology()
        for parameter in self.topology_feature_project.parameters():
            parameter.requires_grad_(True)
        for parameter in self.geometry_feature_project.parameters():
            parameter.requires_grad_(False)
        for parameter in self.decoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.spline_graph.geometry_head.parameters():
            parameter.requires_grad_(False)

    # Purpose: Preserve B1a topology while exposing continuous B1b geometry rows.
    # Called by: LocalBoundaryProductionContract._lock_proof_topology.
    # Calls: PrimitiveParameterHead.lock_topology().
    def lock_topology_for_proof(self) -> None:
        self._topology_bootstrap_only = False
        head = self.decoder.parameter_head
        head.lock_topology()
        self.spline_graph.lock_topology()
        for parameter in self.topology_feature_project.parameters():
            parameter.requires_grad_(False)
        for parameter in self.geometry_feature_project.parameters():
            parameter.requires_grad_(True)
        for parameter in head.geometry_net.parameters():
            parameter.requires_grad_(True)
        for parameter in self.spline_graph.geometry_head.parameters():
            parameter.requires_grad_(True)

    # Purpose: Persist the phase-local topology lock into actual model parameters.
    # Called by: TrainingService.train_v9 after each sdf-proof optimizer step.
    # Calls: PrimitiveParameterHead.restore_locked_topology_parameters().
    def restore_locked_topology_parameters(self) -> None:
        self.decoder.parameter_head.restore_locked_topology_parameters()
        self.spline_graph.restore_locked_topology_parameters()

    # Purpose: Implement genome dict for LocalBoundaryProductionStructure.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def genome_dict(self) -> dict[str, float]:
        values = self.evolution_genome.detach().float().cpu().tolist()
        return {name: float(values[index]) for index, name in enumerate(EVOLUTION_GENOME_NAMES)}

    # Purpose: Implement set genome for LocalBoundaryProductionStructure.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def set_genome(self, value: dict[str, float]) -> None:
        with torch.no_grad():
            for index, name in enumerate(EVOLUTION_GENOME_NAMES):
                self.evolution_genome[index] = float(value.get(name, 1.0))

    # Purpose: Implement genome value for LocalBoundaryProductionStructure.
    # Called by: forward
    # Calls: No same-class helper methods.
    def _genome_value(self, name: str) -> torch.Tensor:
        index = EVOLUTION_GENOME_NAMES.index(name)
        return self.evolution_genome[index].float()

    # Purpose: Apply the checkpointed evolution genome to one queried structural field.
    # Called by: forward, LocalBoundaryProductionContract._geometry_query_from_outputs
    # Calls: _genome_value
    def _apply_query_genome(
        self,
        field: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply legacy field correction only to the retired scalar query path."""
        if "spline_graph_authority" in field:
            return field
        correction_scale = self._genome_value("correction_scale").to(
            field["phi_pixels"].device
        )
        sampled_source = field["warped_source_pixels"].float()
        evolved_phi = sampled_source + (
            field["phi_pixels"].float() - sampled_source
        ) * correction_scale
        field["phi_pixels"] = evolved_phi
        field["primitive_phi_pixels"] = evolved_phi
        field["direct_delta_pixels"] = evolved_phi - sampled_source
        field["residual_pixels"] = evolved_phi - sampled_source
        return field

    # Purpose: Build and query the connected production spline graph.
    # Called by: GeometryNet.forward.
    # Calls: ConnectedSplineGraph.forward, LocalParametricBoundaryDecoder.build_context.
    def forward(
        self,
        decoded_feature: torch.Tensor,
        inputs: torch.Tensor,
        source_prior_lr: torch.Tensor,
        query_grid: torch.Tensor,
    ) -> dict[str, Any]:
        feature_gain = self._genome_value("feature_gain").to(decoded_feature.device)
        evidence_gain = self._genome_value("evidence_gain").to(decoded_feature.device)
        direct_evidence = (
            inputs[:, 0:16].to(decoded_feature.dtype)
            * evidence_gain.to(decoded_feature.dtype)
        )
        projected_input = torch.cat(
            (decoded_feature * feature_gain.to(decoded_feature.dtype), direct_evidence),
            dim=1,
        )
        geometry_feature_grid = self.geometry_feature_project(projected_input)
        topology_feature_grid = self.topology_feature_project(projected_input)
        structural_residual_gain = torch.tanh(
            self.structural_residual_gain_head(projected_input).float()
        )
        structural_residual_gain = F.interpolate(
            structural_residual_gain, size=query_grid.shape[1:3],
            mode="bilinear", align_corners=False,
        )

        # Retain the V11.4 local context only for checkpoint/public telemetry. It has
        # no query-time authority after V11.5.
        context = self.decoder.build_context(
            geometry_feature_grid,
            source_prior_lr,
            topology_feature_grid=topology_feature_grid,
        )
        max_distance = float(self.decoder.max_distance_pixels)
        control_size = context["branch_anchor_distance_pixels"].shape[-2:]
        source_control = F.interpolate(
            source_prior_lr.float() * max_distance,
            size=control_size, mode="bilinear", align_corners=False,
        )
        distance_scale = self._genome_value("distance_scale").to(source_control.device)
        context["branch_anchor_distance_pixels"] = source_control + (
            context["branch_anchor_distance_pixels"] - source_control
        ) * distance_scale
        context["anchor_distance_pixels"] = source_control + (
            context["anchor_distance_pixels"] - source_control
        ) * distance_scale
        context["distance_delta_pixels"] = context["anchor_distance_pixels"] - source_control

        geometry_scale = (
            source_control.new_zeros(())
            if self._topology_bootstrap_only
            else self._genome_value("correction_scale").to(source_control.device)
        )
        spline = self.spline_graph(
            topology_feature_grid,
            geometry_feature_grid,
            source_prior_lr,
            query_grid,
            topology_scale=distance_scale,
            displacement_scale=geometry_scale,
        )
        field = self._apply_query_genome(spline["field"])
        return {
            "feature_grid": geometry_feature_grid,
            "context": context,
            "spline_graph": spline["graph"],
            "field": field,
            "structural_residual_gain": structural_residual_gain,
            "genome": self.evolution_genome,
        }
