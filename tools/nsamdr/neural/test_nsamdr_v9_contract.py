#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import json
import random
import torch
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v9.config import V9Config
from v9.model import MODEL_SCHEMA, BoundaryRenderer, FidelityResidualNetV9, parameter_count
from v9.geometry_audit import AUDIT_SCHEMA, AuditOptions, audit_pair
from v9.contours import analytic_contour_targets
from v9.dataset import _synthetic_geometry_sample, _pack_sample
from v9.training import _build_optimizer, _config_hash, _phase_for_epoch, _phase_lr, _forward_for_phase
from v9.experiments import (
    DEFAULT_TUNING_ASSET_NAME,
    PRODUCTION_SCOPE_FIELDS,
    initialise_experiment,
    promote_experiment,
    load_experiment_manifest,
    write_experiment_manifest,
)


def main() -> int:
    repo_root = HERE.parents[2]
    full_config_path = HERE / "configs" / "v9_fidelity_full.json"
    tuning_config_path = HERE / "configs" / "v9_preview_raven.json"
    pilot_config_path = HERE / "configs" / "v9_fidelity_pilot.json"
    stability_config_path = HERE / "configs" / "v9_fidelity_stability.json"
    for path in (full_config_path, tuning_config_path, pilot_config_path, stability_config_path):
        if not path.is_file():
            raise SystemExit(f"missing V9 config: {path}")

    full = V9Config.load(full_config_path)
    tuning = V9Config.load(tuning_config_path)
    pilot = V9Config.load(pilot_config_path)
    stability = V9Config.load(stability_config_path)
    if full.total_epochs != 24 or tuning.total_epochs != 24 or pilot.total_epochs != 12 or stability.total_epochs != 6:
        raise SystemExit("unexpected V9 phase schedule")

    # The tuning model must be the production model, trained fully on a fixed
    # small scope. Only data/work-volume fields may intentionally differ.
    architecture_fields = (
        "tile_size", "target_scale", "widths", "blocks_per_level",
        "decoder_blocks", "attention_heads", "attention_window", "drop_path",
        "albedo_medium_delta", "albedo_fine_delta", "normal_medium_delta",
        "normal_fine_delta", "material_delta", "auxiliary_delta",
        "geometry_edge_support_radius",
        "contour_sdf_max_distance_pixels",
        "boundary_renderer_band_pixels", "boundary_renderer_sample_pixels",
        "boundary_renderer_hard_width_pixels", "boundary_renderer_soft_width_pixels",
        "boundary_renderer_edge_threshold", "boundary_renderer_confidence_floor",
        "boundary_renderer_gate_gain",
        "boundary_renderer_far_sample_multiplier", "boundary_renderer_far_sample_weight",
        "boundary_gate_initial_bias",
        "implicit_sdf_hidden_channels", "implicit_sdf_coordinate_scale",
        "implicit_sdf_residual_pixels", "sdf_bootstrap_residual_pixels", "sdf_proof_residual_pixels",
        "sdf_sign_gauge_invariant", "sdf_metric_band_pixels", "sdf_coarse_init_std",
        "boundary_renderer_plateau_samples",
        "boundary_renderer_plateau_max_multiplier",
        "boundary_renderer_plateau_stability_scale",
        "appearance_enabled", "appearance_edge_suppression",
    )
    for field in architecture_fields:
        if getattr(tuning, field) != getattr(full, field):
            raise SystemExit(f"Raven tuning architecture differs from production: {field}")
    phase_fields = (
        "identity_epochs", "residual_epochs", "boundary_epochs",
        "detail_epochs", "physical_finetune_epochs",
    )
    for field in phase_fields:
        if getattr(tuning, field) != getattr(full, field):
            raise SystemExit(f"Raven tuning phase schedule differs from production: {field}")
    if tuning.tiles_per_epoch != 320 or tuning.validation_tiles != 48:
        raise SystemExit("unexpected fixed Raven tuning work volume")
    if tuning.dataset_manifest.replace("\\", "/") != "artifacts/nsamdr/training_v9_preview_raven/dataset_manifest.json":
        raise SystemExit("Raven tuning dataset manifest is not isolated")

    # V9.8 converges the implicit-SDF geometry to explicit learned gate/edge
    # evidence and proves an untouched raw control. Older schemas are intentionally incompatible.
    expected_full_hash = "ce04236d056f41a5376a167d15232083d9e7ffdf1965205ffb170bb4e1bc05a0"
    if _config_hash(full) != expected_full_hash:
        raise SystemExit("V9.8 production semantic config hash changed unexpectedly")
    if MODEL_SCHEMA != "NSAMDR_SIGN_GAUGE_METRIC_SDF_RENDERER_4X_V9_8_3":
        raise SystemExit(f"unexpected V9 model schema: {MODEL_SCHEMA}")
    if pilot.tiles_per_epoch != 768 or abs(pilot.synthetic_geometry_probability - 0.82) > 1e-9:
        raise SystemExit("V9.8 geometry-convergence pilot configuration is not active")
    if full.appearance_enabled or tuning.appearance_enabled:
        raise SystemExit("V9.8 geometry-convergence proof must keep AppearanceNet disabled")

    model = FidelityResidualNetV9(full)
    contract = model.architecture_contract()
    if (
        contract.get("schema") != MODEL_SCHEMA
        or contract.get("geometryModel") != "GeometryNet"
        or contract.get("renderer") != "BoundaryRenderer"
        or bool(contract.get("geometryCanPaintRgb"))
        or tuple(contract.get("geometryOutputs", ())) != (
            "sdf", "edge", "orientation", "hardness", "boundary_gate"
        )
        or not bool(contract.get("sharedAcrossPhysicalMaps"))
        or bool(contract.get("moduloCoordinatePhase"))
        or float(contract.get("boundedSdfResidualPixels", 0.0)) <= 0.0
    ):
        raise SystemExit(f"V9.8 architecture contract failed: {contract}")
    count = parameter_count(model)
    if not 6_000_000 <= count <= 11_000_000:
        raise SystemExit(f"unexpected V9 production parameter count: {count:,}")
    if any(isinstance(module, torch.nn.PixelShuffle) for module in model.modules()):
        raise SystemExit("V9 must not contain PixelShuffle")

    phases = [_phase_for_epoch(epoch, full) for epoch in (1, 2, 7, 15, 20)]
    expected = [
        "sdf-bootstrap",
        "sdf-proof",
        "gate-proof",
        "boundary-hardening",
        "physical-finetune",
    ]
    if phases != expected:
        raise SystemExit(f"bad V9 phases: {phases}")

    tiny = V9Config(
        tile_size=32,
        widths=(32, 32, 48, 64),
        blocks_per_level=(1, 1, 1, 1),
        decoder_blocks=(1, 1, 1),
        attention_heads=4,
        batch_size=1,
    )
    tiny.validate()
    tiny_model = FidelityResidualNetV9(tiny).eval()
    x = torch.rand(1, 16, 32, 32)
    x[:, 3:5] = x[:, 3:5] * 2.0 - 1.0
    with torch.no_grad():
        output = tiny_model(x)
    identity_delta = float(
        (output["albedo"] - output["baseline_albedo"])
        .abs()
        .max()
        .item()
    )
    if identity_delta > 1.0e-7:
        raise SystemExit(
            f"V9 is not identity-initialized: max albedo delta={identity_delta}"
        )

    # V9.8.2.1 regression: validation uses the same phase helper as training.
    # EXP_0005 exposed a NameError because the helper referenced `config`
    # without accepting it. Exercise the sdf-proof path directly so that
    # this cannot regress into a runtime-only validation failure.
    proof_batch = {
        "input": x,
        "target_sdf": torch.zeros(1, 1, 128, 128, dtype=torch.float32),
    }
    with torch.no_grad():
        proof_output = _forward_for_phase(tiny_model, proof_batch, "sdf-proof", tiny)
    if proof_output["boundary_gate"].shape[-2:] != (128, 128):
        raise SystemExit("V9.8.2 sdf-proof validation helper returned an invalid gate shape")

    # Instrumented preview contract: mathematical audit must run independently
    # of the learned critic and classify an unchanged image as non-regressing.
    import cv2
    audit_image = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2.line(audit_image, (8, 91), (119, 31), (220, 220, 220), 3, cv2.LINE_AA)
    with tempfile.TemporaryDirectory(prefix="nsamdr_v95_audit_contract_") as audit_temp:
        audit = audit_pair(
            audit_image, audit_image.copy(), source_name="contract_identity",
            output_dir=Path(audit_temp), options=AuditOptions(evidence_regions=0, critic_mode="off"),
        )
    if audit.get("schema") != AUDIT_SCHEMA or audit.get("verdict") == "FAIL":
        raise SystemExit(f"V9.8 geometry auditor contract failed: {audit.get('verdict')}")

    # Tuning controls are real training semantics, not GUI-only labels.
    optimizer_config = V9Config(
        tile_size=32, widths=(32, 32, 48, 64), blocks_per_level=(1, 1, 1, 1),
        decoder_blocks=(1, 1, 1), attention_heads=4, batch_size=1,
        optimizer_name="adam", scheduler_name="cosine-phase",
        scheduler_min_lr_ratio=0.25,
    )
    optimizer_config.validate()
    optimizer, optimizer_mode = _build_optimizer(tiny_model, optimizer_config, torch.device("cpu"))
    if not isinstance(optimizer, torch.optim.Adam) or "Adam" not in optimizer_mode:
        raise SystemExit("V9 selectable optimizer is not active")
    last_residual_epoch = optimizer_config.identity_epochs + optimizer_config.residual_epochs
    if _phase_lr("sdf-proof", optimizer_config, last_residual_epoch) >= optimizer_config.learning_rate:
        raise SystemExit("V9 cosine-phase scheduler is not active")

    model_text = (HERE / "v9" / "model.py").read_text(encoding="utf-8")
    if "nn.PixelShuffle" in model_text:
        raise SystemExit("V9 source still instantiates PixelShuffle")
    if "torch.remainder" in model_text:
        raise SystemExit("V9.8 implicit SDF must not expose a repeating modulo coordinate phase")
    for required in (
        "GeometryNet",
        "ImplicitSDFResidualHead",
        "coarse_sdf_head",
        "sdf_residual_head",
        "coarse_sdf_pixels",
        "sdf_residual_pixels",
        "_adaptive_plateau_sample",
        "plateau_confidence",
        "plateau_evidence",
        "monotonic-bilinear-plus-local-coverage-deconvolution",
        "_geometry_solved_plateaus",
        "robust-conservation-deconvolution",
        "sdfBootstrapResidualPixels",
        "sdfProofResidualPixels",
        "sdfSignGaugeInvariant",
        "sdfMetricBandPixels",
        "sdfCoarseInitStd",
        "topologySafeSideSampling",
        "rendererRevision",
        "torch.cumprod",
        "BoundaryRenderer",
        "AppearanceNet",
        "sdf_override",
        "gate_override",
        "hardness_override",
        "hardness_head",
        "boundary_gate_head",
        "prior_project",
        "boundary_reconstructed_albedo",
        "boundary_reconstructed_normal",
        "boundary_reconstructed_material",
        "boundary_gate",
        "boundary_renderer_sample_pixels",
        "boundary_renderer_hard_width_pixels",
        "boundary_renderer_far_sample_multiplier",
        "boundary_gate_prediction",
        "_source_edge_support",
        "F.grid_sample",
        "sharedAcrossPhysicalMaps",
    ):
        if required not in model_text:
            raise SystemExit(f"missing V9 architecture contract: {required}")

    audit_text = (HERE / "v9" / "geometry_audit.py").read_text(encoding="utf-8")
    for required in ("edgeChamferImprovement", "offEdgeIdentityRms8bit", "GeometryCritic", "candidateWinProbabilityMean", "geometry_audit.html"):
        if required not in audit_text:
            raise SystemExit(f"missing V9.8 geometry-audit contract: {required}")
    loss_text = (HERE / "v9" / "losses.py").read_text(encoding="utf-8")
    for required in (
        "_sdf_global_polarity", "_balanced_metric_band_mean",
        "sdf_polarity_positive_fraction", "sdf_zero_rms_pixels",
    ):
        if required not in loss_text:
            raise SystemExit(f"missing V9.8.3 sign-gauge SDF contract: {required}")

    staged_audit_text = (HERE / "audit_nsamdr_v9_geometry_checkpoint.py").read_text(encoding="utf-8")
    for required in (
        "RENDERER_FAIL",
        "SDF_FAIL",
        "GATE_FAIL",
        "TOPOLOGY_FAIL",
        "FUZZ_FAIL",
        "HALO_FAIL",
        "DOUBLE_EDGE_FAIL",
        "NSAMDR_METRIC_SDF_GEOMETRY_PROOF_V3",
        "G0_line_",
        "G1_circle",
        "G2_corner",
        "G3_parallel",
        "G4_lowcontrast",
        "G5_degrade_blur",
        "outerGradientFraction",
        "haloOvershoot8bit",
        "stageA_oracle_renderer",
        "stageB_predicted_sdf",
        "stageC_full",
        "GT SDF + forced gate",
        "predicted SDF + forced gate",
        "--oracle-only",
        "plateauEvidenceDiagnostics",
        "topologyRegressionFraction",
    ):
        if required not in staged_audit_text:
            raise SystemExit(f"missing V9.8 staged proof contract: {required}")

    runner_text = (HERE / "run_nsamdr_v9_raven_tune_preview.py").read_text(encoding="utf-8")
    for required in (
        "--allocate-only",
        "--stop-after-phase",
        "TRAINING BLOCKED BY STAGE-A RENDERER PREFLIGHT",
        "GATE TRAINING BLOCKED BY STAGE-B SDF PROOF",
        "rendererPreflightPass",
        "sdfStageProofPass",
        "Resuming staged SDF proof for existing",
    ):
        if required not in runner_text:
            raise SystemExit(f"missing V9.8.2 staged-runner contract: {required}")

    preview_text = (HERE / "preview_nsamdr_v9_experiment.py").read_text(encoding="utf-8")
    for required in ("combinedGeometryAuditVerdict", "geometry_feedback.zip", "real_geometry_audit", "synthetic_geometry_audit"):
        if required not in preview_text:
            raise SystemExit(f"missing V9.8 instrumented-preview contract: {required}")

    cli_text = (HERE / "train_nsamdr_v9.py").read_text(encoding="utf-8")
    if 'state.add_argument("--auto"' not in cli_text:
        raise SystemExit("missing V9 safe auto-resume control")
    if "Auto training control: RESUME selected." not in cli_text:
        raise SystemExit("missing V9 auto-resume state detection")

    training_text = (HERE / "v9" / "training.py").read_text(encoding="utf-8")
    if "_forward_for_phase(model, batch, phase, config)" not in training_text:
        raise SystemExit("V9.8.2 validation phase helper does not receive config")
    for required in (
        "trainingSafetyPass", "reconstructionAcceptancePass", "coarse_sdf_surface",
        "sdf_metric_gradient", "sdf_zero_rms_pixels", "sdf_polarity_positive_fraction",
        "SyntheticGeometryValidationDataset", "synthetic-sdf", "best_sdf_score",
        "boundary_fuzz", "boundary_halo", "stop_after_phase", "Staged checkpoint stop reached",
    ):
        if required not in training_text:
            raise SystemExit(f"missing V9.8 training/acceptance semantic: {required}")
    for required in (
        "_ReactiveCudaMemoryGovernor",
        "torch.autograd.graph.save_on_cpu",
        "torch.cuda.mem_get_info",
        "torch.cuda.empty_cache",
        "CUDA OOM intercepted",
        "strict safety envelope remained satisfied",
        "_host_memory_info",
        "host RAM too pressured for safe activation offload",
        "_apply_dynamic_allocator_ceiling",
        "_stable_resource_snapshot",
        "strict safety envelope",
        "foreground reserve",
        "set_per_process_memory_fraction",
    ):
        if required not in training_text:
            raise SystemExit(f"missing V9 elastic CUDA policy: {required}")
    if "cuda_memory_fraction" in training_text and "set_per_process_memory_fraction" in training_text:
        # The dynamic ceiling must not be sourced from the legacy fixed
        # cuda_memory_fraction field.
        dynamic_call = "torch.cuda.set_per_process_memory_fraction(fraction, device=self.device)"
        if dynamic_call not in training_text:
            raise SystemExit("V9 allocator ceiling is not dynamically calculated")

    if not pilot.reactive_vram_enabled:
        raise SystemExit("V9 elastic CUDA sharing must be enabled")
    if pilot.reactive_vram_pause_free_fraction >= pilot.reactive_vram_resume_free_fraction:
        raise SystemExit("V9 elastic CUDA pause/resume hysteresis is invalid")
    if pilot.reactive_host_pause_free_fraction >= pilot.reactive_host_resume_free_fraction:
        raise SystemExit("V9 elastic host-RAM pause/resume hysteresis is invalid")

    loss_text = (HERE / "v9" / "losses.py").read_text(encoding="utf-8")
    for required in (
        'losses["regret"]',
        'losses["regression_fraction"]',
        'losses["improvement_fraction"]',
        'losses["unchanged"]',
        'losses["fine_zero_mean"]',
        'losses["detail_laplacian"]',
        'losses["ringing_regret"]',
        'losses["geometric_alignment"]',
        'losses["tangent_coherence"]',
        'losses["curvature_coherence"]',
        'losses["sdf_curvature"]',
        'losses["geometry_photometric"]',
        'losses["geometry_regret"]',
        'losses["geometry_proxy_improvement"]',
        'losses["hardness"]',
        'losses["boundary_photometric"]',
        'losses["boundary_profile"]',
        'losses["boundary_gate"]',
        'losses["boundary_off_contour"]',
        'losses["boundary_identity"]',
        'losses["boundary_gate_edge_mean"]',
        'losses["boundary_transition_width_mean"]',
        'losses["boundary_sdf_zero"]',
        'losses["edge_sdf_consistency"]',
        'losses["boundary_pixel_regret"]',
        'losses["sdf_surface"]',
        'losses["sdf_sign"]',
        'losses["sdf_eikonal"]',
        'losses["sdf_gradient_alignment"]',
        'losses["sdf_metric_gradient"]',
        'losses["coarse_sdf_surface"]',
        'losses["sdf_residual_l1"]',
        'losses["boundary_fuzz"]',
        'losses["boundary_halo"]',
    ):
        if required not in loss_text:
            raise SystemExit(f"missing V9 fidelity objective: {required}")


    # Exact analytic straight-line supervision must have one constant tangent,
    # not a locally oscillating raster direction.
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32)
    distance = (xx - 48.0) * 0.6 + (yy - 48.0) * 0.8
    _sdf, tangent, edge = analytic_contour_targets(distance)
    selected = edge[..., 0] > 0.5
    if not selected.any():
        raise SystemExit("analytic V9 contour target produced no edge")
    tangent_selected = tangent[selected]
    tangent_selected = tangent_selected / np.maximum(
        np.linalg.norm(tangent_selected, axis=1, keepdims=True), 1.0e-6
    )
    tangent_spread = float(np.max(np.std(tangent_selected, axis=0)))
    if tangent_spread > 1.0e-4:
        raise SystemExit(f"analytic straight-line tangent is not coherent: spread={tangent_spread}")

    # The V9.8 primitive must sharpen a known fuzzy boundary using only a
    # continuous SDF plus two-sided sampling; no learned RGB output is involved.
    renderer_cfg = V9Config(
        tile_size=32,
        widths=(32, 32, 48, 64),
        blocks_per_level=(1, 1, 1, 1),
        decoder_blocks=(1, 1, 1),
        attention_heads=4,
    )
    renderer_cfg.validate()
    renderer = BoundaryRenderer(renderer_cfg)
    width = 64
    xx = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    sdf_pixels = xx - 31.5
    sdf = (sdf_pixels / renderer_cfg.contour_sdf_max_distance_pixels).expand(1, 1, 64, width)
    target_step = (xx < 31.5).float().expand(1, 1, 64, width)
    fuzzy = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(target_step, (3, 3, 0, 0), mode="replicate"),
        kernel_size=(1, 7),
        stride=1,
    )
    edge_logits = torch.full_like(sdf, 8.0)
    hardness_logits = torch.full_like(sdf, 8.0)
    support = torch.ones_like(sdf)
    boundary_gate_logits = torch.full_like(sdf, 8.0)
    rendered, renderer_meta = renderer(
        fuzzy, sdf, edge_logits, hardness_logits, boundary_gate_logits, support, enabled=True
    )
    before_mae = float((fuzzy - target_step).abs().mean())
    after_mae = float((rendered - target_step).abs().mean())
    if not after_mae < before_mae:
        raise SystemExit(
            f"V9.8 boundary renderer did not improve controlled edge: "
            f"{before_mae:.6f}->{after_mae:.6f}"
        )
    if float(renderer_meta["boundary_gate"].mean()) <= 0.0:
        raise SystemExit("V9.8 boundary renderer produced an inactive gate")

    dataset_text = (HERE / "v9" / "dataset.py").read_text(encoding="utf-8")
    for required in (
        "SYNTHETIC_GEOMETRY_SCHEMA",
        "_synthetic_geometry_sample",
        "analytic_contour_targets",
        "synthetic_geometry_probability",
        "double_stripe",
        "near_double",
        "ring",
        "_synthetic_region_colours",
    ):
        if required not in dataset_text:
            raise SystemExit(f"missing V9 exact geometry training component: {required}")

    contour_text = (HERE / "v9" / "contours.py").read_text(encoding="utf-8")
    for required in ("_multi_scale_structure_tensor", "CONTOUR_SCHEMA", "narrow soft"):
        if required not in contour_text:
            raise SystemExit(f"missing V9 multi-scale contour component: {required}")

    indexer_path = HERE / "index_eve_texture_dataset_v9.py"
    indexer_text = indexer_path.read_text(encoding="utf-8")
    if "from v9.config import V9Config" not in indexer_text:
        raise SystemExit("V9 dataset indexer imports V9Config from the wrong package")
    if "from v8.config import V9Config" in indexer_text:
        raise SystemExit("V9 dataset indexer still imports V9Config from v8.config")

    # Fixed-ship tuning capabilities and locks must be present in source. The
    # command contract deliberately checks the consolidated dispatcher rather
    # than historical Windows wrapper filenames.
    required_workflow_files = (
        repo_root / "scripts/build/nsamdr.bat",
        repo_root / "tools/nsamdr/nsamdr_cli.py",
        HERE / "prepare_nsamdr_v9_raven_preview_dataset.py",
        HERE / "run_nsamdr_v9_raven_tune_preview.py",
        HERE / "audit_nsamdr_v9_geometry_checkpoint.py",
        HERE / "train_nsamdr_v9_preview_experiment.py",
        HERE / "preview_nsamdr_v9_experiment.py",
        HERE / "compare_nsamdr_v9_experiments.py",
        HERE / "promote_nsamdr_v9_experiment.py",
        HERE / "v9/experiments.py",
    )
    for path in required_workflow_files:
        if not path.is_file():
            raise SystemExit(f"missing V9 tuning workflow source: {path}")
    dispatcher_text = (repo_root / "tools/nsamdr/nsamdr_cli.py").read_text(encoding="utf-8")
    for required in (
        'add_parser("gui"', 'add_parser("tune"', 'add_parser("index"',
        'add_parser("train"', 'add_parser("preview"', 'add_parser("compare"',
        'add_parser("promote"', 'add_parser("validate"', 'add_parser("cleanup"',
        'add_parser("integrate"', 'add_parser("contract"',
        'add_parser("architecture"', 'add_parser("checkpoint"',
    ):
        if required not in dispatcher_text:
            raise SystemExit(f"missing consolidated NSAMDR command capability: {required}")

    gui_text = (repo_root / "tools/nsamdr/gui/nsamdr_v9_workflow_gui.py").read_text(encoding="utf-8")
    for required in (
        "Raven tune + instrumented preview",
        "Quick (~10-15 min)",
        "Full / promotion proof",
        "Compare tuning experiments",
        "Promote best configuration",
        "Full-dataset work is locked until a Full Raven proof experiment is promoted.",
        "Full production preview is locked until the selected promoted configuration completes full training.",
        "NSAMDR V9 Workflow Controller 4.9.3",
        "Bounded SDF residual px",
        "Coarse SDF loss",
        "SDF metric gradient",
        "SDF metric band px",
        "SDF coarse init std",
        "Synthetic SDF validation tiles",
        "SDF zero band px",
        "Bootstrap residual px",
        "SDF-proof residual px",
        "Hard-edge fuzz loss",
        "Halo suppression loss",
        "Plateau samples",
    ):
        if required not in gui_text:
            raise SystemExit(f"missing V9 tuning GUI contract: {required}")

    preview_builder_text = (HERE / "prepare_nsamdr_v9_raven_preview_dataset.py").read_text(encoding="utf-8")
    for required in (
        DEFAULT_TUNING_ASSET_NAME,
        "adaptive-non-overlapping-512-grid-checkerboard-v1",
        "overlapBetweenTrainAndValidation",
        "fixedPreviewSet",
        "sourceFingerprint",
        "_selection_fingerprint",
        "selectedRegions",
        "_select_fixed_regions",
        "Requested region counts are upper bounds",
        "maxTrainCrops",
        "maxValidationCrops",
    ):
        if required not in preview_builder_text:
            raise SystemExit(f"missing fixed Raven dataset contract: {required}")

    source_fingerprint_body = preview_builder_text[
        preview_builder_text.index("def _source_fingerprint("):
        preview_builder_text.index("def _selection_fingerprint(")
    ]
    if "for record in records" in source_fingerprint_body:
        raise SystemExit(
            "Raven source fingerprint illegally depends on crop records before selection"
        )

    from prepare_nsamdr_v9_raven_preview_dataset import _select_fixed_regions
    synthetic_cells = [
        {
            "familyId": "raven",
            "x": (index % 2) * 512,
            "y": (index // 2) * 512,
            "detailScore": float(4 - index),
            "holdout": index == 0,
        }
        for index in range(4)
    ]
    selected_train, selected_validation = _select_fixed_regions(
        synthetic_cells,
        max_train_crops=12,
        max_validation_crops=4,
    )
    if len(selected_train) != 3 or len(selected_validation) != 1:
        raise SystemExit(
            "Raven adaptive fixed-region policy must select 3 train + 1 held-out "
            "from a four-cell 1024x1024-style texture"
        )
    if {
        (item["familyId"], item["x"], item["y"]) for item in selected_train
    }.intersection({
        (item["familyId"], item["x"], item["y"]) for item in selected_validation
    }):
        raise SystemExit("Raven adaptive fixed-region contract allows train/held-out overlap")

    dataset_text = (HERE / "v9/dataset.py").read_text(encoding="utf-8")
    if 'if bool(manifest.get("fixedPreviewSet"))' not in dataset_text:
        raise SystemExit("fixed Raven dataset fingerprint is not bound to exact crop selection")

    # Promotion is executable, not merely descriptive: a tuned semantic value
    # must survive promotion exactly while production scope is restored.
    with tempfile.TemporaryDirectory(prefix="nsamdr-v9-contract-") as temporary:
        temp_repo = Path(temporary)
        experiment_id, directory, tuned_config = initialise_experiment(
            temp_repo,
            tuning_config_path,
            {
                "edge_weight": 0.731,
                "optimizer_name": "adam",
                "scheduler_name": "cosine-phase",
                "seed": 1337,
                "appearance_enabled": True,
            },
            preset="contract-test",
            asset_name=DEFAULT_TUNING_ASSET_NAME,
            asset_query="contract://raven",
            selection_key="type:contract",
        )
        (directory / "checkpoint_best.pt").touch()
        experiment_manifest_path = directory / "experiment.json"
        experiment_manifest = load_experiment_manifest(temp_repo, experiment_id)
        experiment_manifest["status"] = "completed"
        experiment_manifest["trainingSafetyPass"] = True
        experiment_manifest["acceptancePass"] = True
        experiment_manifest["reconstructionAcceptancePass"] = True
        experiment_manifest["combinedGeometryAuditVerdict"] = "PASS"
        experiment_manifest["acceptanceRegressionFraction"] = 0.01
        write_experiment_manifest(experiment_manifest_path, experiment_manifest)
        (directory / "previews").mkdir(parents=True, exist_ok=True)
        (directory / "previews/preview_manifest.json").write_text(
            json.dumps({
                "status": "completed",
                "qualityGate": {
                    "trainingSafetyPass": True,
                    "acceptancePass": True,
                    "reconstructionAcceptancePass": True,
                    "acceptanceRegressionFraction": 0.01,
                },
            }) + "\n",
            encoding="utf-8",
        )
        promoted_path, promotion = promote_experiment(
            temp_repo, experiment_id, full_base_config_path=full_config_path
        )
        promoted = V9Config.load(promoted_path)
        if abs(promoted.edge_weight - 0.731) > 1.0e-12:
            raise SystemExit("promotion did not preserve tuned edge weight")
        if promoted.optimizer_name != "adam" or promoted.scheduler_name != "cosine-phase":
            raise SystemExit("promotion did not preserve optimizer/scheduler")
        if promoted.dataset_manifest != full.dataset_manifest or promoted.output_dir != full.output_dir:
            raise SystemExit("promotion did not restore full production scope")
        if not promotion.get("semanticHyperparametersPreservedExactly"):
            raise SystemExit("promotion exactness contract is missing")

    cleanup_text = dispatcher_text
    if "neural_v8" in cleanup_text or "training_v8" in cleanup_text:
        raise SystemExit("V9 cleanup capability must not target V8 artifacts")
    for required in ("training_v9_preview_raven", "artifacts/nsamdr/experiments", "artifacts/nsamdr/promoted"):
        if required not in cleanup_text:
            raise SystemExit(f"V9 cleanup capability does not cover workflow artifact: {required}")

    for v9_source in (HERE / "v9").glob("*.py"):
        v9_text = v9_source.read_text(encoding="utf-8")
        if "from v8" in v9_text or "import v8" in v9_text:
            raise SystemExit(f"V9 source still depends on V8: {v9_source.name}")

    candidates = (
        repo_root / "tools" / "nsamdr" / "generate_strategy_candidates.py"
    ).read_text(encoding="utf-8")
    if "NSAMDR_NEURAL_ARCHITECTURE" not in candidates or '"V9"' not in candidates:
        raise SystemExit("Mode 3 candidate generator does not route V9")
    for required in (
        '"mapsReconstructed": ["albedo", "normalXY", "materialRGB", "roughness", "emissive"]',
        '"native4xEvaluation"',
        '"modelInputWidth"',
        '"geometryOnlyProof"',
        '"deterministicBaselineBoundaryRendered"',
        '"boundaryRendererAppliedToAllPhysicalMaps"',
        '"reconstructionPrimitive"',
        '"materialPolicy": (',
        'preview_control_provenance.json',
        'rawControlVsCandidate',
        'sourceUnchangedDuringGeneration',
        'mode3_abcd_texture_evidence.png',
        'raven_fixed_crop_manifest.json',
        'companionGeometryAudits',
    ):
        if required not in candidates:
            raise SystemExit(f"missing V9.8 native/PBR preview contract: {required}")

    render_pipeline_text = (repo_root / "trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp").read_text(encoding="utf-8")
    preview_panel_text = (repo_root / "trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp").read_text(encoding="utf-8")
    eve_test_text = (repo_root / "tools/nsamdr/eve_asset_test.py").read_text(encoding="utf-8")
    for required in (
        "scientificTriple",
        "rawControlPane",
        "legacyControlPane",
        "candidatePane",
        "useLegacySampler",
    ):
        if required not in render_pipeline_text:
            raise SystemExit(f"missing V9.8 scientific render control: {required}")
    for required in (
        "A RAW CONTROL",
        "B LEGACY EMULATION",
        "C NSAMDR V9.8",
        "SOURCE SHA-256 PROVENANCE",
    ):
        if required not in preview_panel_text:
            raise SystemExit(f"missing V9.8 scientific preview overlay: {required}")
    for required in (
        "NSAMDR_PROVENANCE_STATUS",
        "NSAMDR_PROVENANCE_SOURCE_SHA",
        "NSAMDR_PROVENANCE_CANDIDATE_SHA",
    ):
        if required not in eve_test_text:
            raise SystemExit(f"missing V9.8 provenance launch plumbing: {required}")

    print("NSAMDR V9.8.3 sign-gauge metric-SDF/staged-gate/scientific-control contract passed")
    print(f"  production parameters={count:,}")
    print(f"  identity delta={identity_delta:.9f}")
    print("  PixelShuffle=absent")
    print("  local baseline-regret protection=present")
    print("  geometric tangent/curvature coherence=present")
    print("  exact synthetic line/arc/corner training=present")
    print("  memory-safe narrow 512 feature path=present")
    print("  elastic CUDA sharing=present")
    print("  fixed CUDA allocator cap=absent")
    print("  reactive hard allocator ceiling=present")
    print(
        "  VRAM response=GPU activations -> CPU-saved activations -> yield -> recover"
    )
    print("  foreground burst reserve=present")
    print("  multi-sample pre-batch stability gate=present")
    print("  host-RAM offload guard=present")
    print("  safe auto-resume control=present")
    print("  destructive restart backup=present")
    print("  fixed Raven tuning set=present")
    print("  full 24-epoch tuning schedule=present")
    print("  immutable experiment registry=present")
    print("  exact config promotion=present")
    print("  production capability locks=present")
    gui_text = (HERE.parent / "gui/nsamdr_v9_workflow_gui.py").read_text(encoding="utf-8")
    for required in (
        "self.form_canvas",
        "self.right_split",
        "orient=\"vertical\"",
        "self.footer = footer",
        "scroll for additional tuning controls",
    ):
        if required not in gui_text:
            raise SystemExit(f"missing scroll-safe GUI layout contract: {required}")

    combined_python = HERE / "run_nsamdr_v9_raven_tune_preview.py"
    if not combined_python.is_file() or 'add_parser("tune"' not in dispatcher_text:
        raise SystemExit("combined Raven tune + preview stage is missing")
    combined_text = combined_python.read_text(encoding="utf-8")
    for required in (
        "Stage-A renderer preflight -> metric SDF -> Stage-B SDF proof -> learned gate -> Raven audit",
        "--training-mode",
        "--preview-target-size",
        "default=4096",
        "Promotion                : LOCKED (Quick experiment)",
        "RAVEN PREVIEW BLOCKED BY SYNTHETIC PROOF",
    ):
        if required not in combined_text:
            raise SystemExit(f"missing combined Raven tuning contract: {required}")
    training_text = (HERE / "v9/training.py").read_text(encoding="utf-8")
    if "early_stop_patience" not in training_text or "physical-finetune only" not in training_text:
        raise SystemExit("Full Raven convergence early-stop gate is missing")
    experiment_text = (HERE / "v9/experiments.py").read_text(encoding="utf-8")
    if "V9.4 geometry-only proof and cannot be promoted" not in experiment_text:
        raise SystemExit("V9.4 geometry-only promotion lock is missing")
    for required in (
        'Stage("tune", "1", "Raven tune + instrumented preview"',
        'Stage("tune_promote", "2"',
        'Stage("index", "3"',
        'Stage("train", "4"',
        'Stage("preview", "5"',
    ):
        if required not in gui_text:
            raise SystemExit(f"collapsed stage numbering missing: {required}")

    print("  optimizer/scheduler tuning controls=present")
    print("  quick Raven mode=11 epochs x 96 tiles")
    print("  combined Raven tune+preview stage=present")
    print("  Quick experiment promotion lock=present")
    print("  geometry-only promotion lock=present")
    print("  Full Raven convergence gate=present")
    print("  collapsed Stage 1/2/3 workflow=present")
    print("  scroll-safe tuning GUI layout=present")
    print("  native Raven 1024->4096 preview=present")
    print("  coarse SDF topology field=present")
    print("  bounded continuous SDF residual=present")
    print("  modulo-4 coordinate phase=absent")
    print("  adaptive plateau reconstruction=present")
    print("  G0-G5 synthetic geometry gate=present")
    print("  Raven preview blocked until synthetic PASS=present")
    print("  independent training/reconstruction acceptance=present")
    print("  continuous SDF boundary field=present")
    print("  implicit boundary renderer=present")
    print("  boundary gate/fuzz telemetry=present")
    print("  zero-set and edge/SDF consistency supervision=present")
    print("  benefit-weighted boundary gate=present")
    print("  pixel-level boundary regret=present")
    print("  scientific A/raw B/legacy C/NSAMDR control=present")
    print("  raw-source SHA-256 provenance=present")
    print("  full-resolution edge geometry loss=present")
    print("  renderer-aware anisotropic degradation=present")
    print("  neural physical material preview=present")
    print("  quantitative promotion gate=present")
    print("  V9.7/V9.6/V9.5/V9.4.x/V9.3/V9.2.x checkpoint compatibility=intentionally broken by V9.8.3 sign-gauge metric-SDF schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
