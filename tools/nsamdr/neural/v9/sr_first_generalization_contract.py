from __future__ import annotations

"""V13.2 representative Raven qualification for current SR-first Quick training.

The canonical Quick workflow has one learned reconstruction candidate followed by
one authority selector:

    B -> SR candidate C -> BenefitSelector -> F

Legacy geometry/profile/seam modules remain in the checkpoint graph only where
required for compatibility/evidence. They do not own Quick training stages.
"""

import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any

import torch
from torch.utils.data import DataLoader

from .application.configuration import (
    QUICK_WORK_BUDGET,
    is_sr_first_quick_config,
)
from .baseline_relative_specialist_contract import (
    PROTECTED_DRIFT_TOLERANCE,
    PROTECTED_PRESERVATION_REQUIRED,
    protected_mask,
)
from .config import V9Config
from .contours import sobel_tensor
from .dataset import PhysicalTileDatasetV9, load_dataset_manifest
from .inference import resolve_device
from .model import FidelityResidualNetV9
from . import sr_first_contract as sr


SR_GENERALIZATION_REVISION = "V13.2"
QUICK_SR_WORK_BUDGET = QUICK_WORK_BUDGET

REPRESENTATIVE_PATCHES = 32
REPRESENTATIVE_MEDIAN_EDGE_REQUIRED = 0.60
REPRESENTATIVE_MEDIAN_GLOBAL_REQUIRED = 0.45
REPRESENTATIVE_MEDIAN_GRADIENT_REQUIRED = 0.35
REPRESENTATIVE_POSITIVE_FRACTION_REQUIRED = 0.75
REPRESENTATIVE_SELECTOR_RETENTION_REQUIRED = 0.90
REPRESENTATIVE_WORST_RECOVERY_FLOOR = -0.10

_INSTALLED = False
_ORIGINAL_BACKEND_RUN: Any = None
_ORIGINAL_ARCHITECTURE_CONTRACT: Any = None


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    w = weight.to(device=value.device, dtype=value.dtype, non_blocking=True)
    if w.shape[1] == 1 and value.shape[1] != 1:
        w = w.expand(-1, value.shape[1], -1, -1)
    return (value.float() * w.float()).sum() / w.float().sum().clamp_min(1.0)


def _recovery(before: torch.Tensor, after: torch.Tensor) -> float:
    b = float(before.float().mean().item())
    a = float(after.float().mean().item())
    return (b - a) / max(b, 1.0e-8)


def _gradient_recovery(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
) -> float:
    target_gray = target.float().mean(dim=1, keepdim=True)
    tx, ty = sobel_tensor(target_gray)

    def error(value: torch.Tensor) -> torch.Tensor:
        gx, gy = sobel_tensor(value.float().mean(dim=1, keepdim=True))
        return (gx - tx).abs().mean() + (gy - ty).abs().mean()

    before = float(error(baseline).item())
    after = float(error(candidate).item())
    return (before - after) / max(before, 1.0e-8)


def _map_recovery(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
) -> float:
    before = float((baseline.float() - target.float()).abs().mean().item())
    after = float((candidate.float() - target.float()).abs().mean().item())
    return (before - after) / max(before, 1.0e-8)


def _albedo_metrics(
    baseline: torch.Tensor,
    value: torch.Tensor,
    target: torch.Tensor,
    target_edge: torch.Tensor,
) -> dict[str, float]:
    before = (baseline.float() - target.float()).abs().mean(dim=1, keepdim=True)
    after = (value.float() - target.float()).abs().mean(dim=1, keepdim=True)
    edge_weight = (0.20 + target_edge.float().clamp(0.0, 1.0) * 3.80).detach()
    global_recovery = _recovery(before, after)
    edge_before = _weighted_mean(before, edge_weight)
    edge_after = _weighted_mean(after, edge_weight)
    edge_recovery = (
        float(edge_before.item()) - float(edge_after.item())
    ) / max(float(edge_before.item()), 1.0e-8)
    return {
        "edgeRecovery": edge_recovery,
        "globalRecovery": global_recovery,
        "gradientRecovery": _gradient_recovery(baseline, value, target),
    }


def _patch_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: V9Config,
) -> tuple[dict[str, float], int, int]:
    target = batch["target_albedo"].float()
    target_edge = batch["target_edge"].float().clamp(0.0, 1.0)
    target_normal = batch["target_normal"].float()
    target_material = sr._target_material(batch, config)

    baseline = outputs["baseline_albedo"].float()
    candidate = outputs["sr_candidate_albedo"].float()
    final = outputs["albedo"].float()
    candidate_metrics = _albedo_metrics(baseline, candidate, target, target_edge)
    final_metrics = _albedo_metrics(baseline, final, target, target_edge)

    baseline_normal = outputs["baseline_normal"].float()
    baseline_material = outputs["baseline_material"].float()
    candidate_normal = outputs["sr_candidate_normal"].float()
    candidate_material = outputs["sr_candidate_material"].float()

    protected = protected_mask(baseline, target)
    protected_count = int(protected.sum().item())
    drift = (final - baseline).abs().amax(dim=1, keepdim=True)
    protected_safe = int(
        ((drift <= float(PROTECTED_DRIFT_TOLERANCE)) & protected).sum().item()
    )

    return (
        {
            "candidateEdgeRecovery": float(candidate_metrics["edgeRecovery"]),
            "candidateGlobalRecovery": float(candidate_metrics["globalRecovery"]),
            "candidateGradientRecovery": float(candidate_metrics["gradientRecovery"]),
            "candidateNormalRecovery": _map_recovery(
                baseline_normal, candidate_normal, target_normal
            ),
            "candidateMaterialRecovery": _map_recovery(
                baseline_material, candidate_material, target_material
            ),
            "finalEdgeRecovery": float(final_metrics["edgeRecovery"]),
            "finalGlobalRecovery": float(final_metrics["globalRecovery"]),
            "finalGradientRecovery": float(final_metrics["gradientRecovery"]),
            "selectorMean": float(
                outputs["benefit_selector_probability"].float().mean().item()
            ),
        },
        protected_safe,
        protected_count,
    )


def _finite_median(rows: list[dict[str, float]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return float(median(values)) if values else -float("inf")


def summarize_representative_validation(
    rows: list[dict[str, float]],
    *,
    protected_safe: int,
    protected_count: int,
) -> dict[str, Any]:
    """Summarise the held-out Raven bank and apply the V13.2 quality contract."""
    if not rows:
        return {
            "pass": False,
            "reasons": ["representative validation produced no patches"],
            "patchCount": 0,
        }

    candidate_edge = _finite_median(rows, "candidateEdgeRecovery")
    candidate_global = _finite_median(rows, "candidateGlobalRecovery")
    candidate_gradient = _finite_median(rows, "candidateGradientRecovery")
    final_edge = _finite_median(rows, "finalEdgeRecovery")
    final_global = _finite_median(rows, "finalGlobalRecovery")
    normal = _finite_median(rows, "candidateNormalRecovery")
    material = _finite_median(rows, "candidateMaterialRecovery")
    edge_positive = sum(row["candidateEdgeRecovery"] > 0.0 for row in rows) / len(rows)
    global_positive = sum(row["candidateGlobalRecovery"] > 0.0 for row in rows) / len(rows)
    edge_retention = final_edge / max(candidate_edge, 1.0e-8)
    global_retention = final_global / max(candidate_global, 1.0e-8)
    preservation = (
        float(protected_safe) / float(max(protected_count, 1))
        if protected_count > 0
        else 1.0
    )
    worst_candidate = min(
        min(float(row["candidateEdgeRecovery"]), float(row["candidateGlobalRecovery"]))
        for row in rows
    )
    worst_final = min(
        min(float(row["finalEdgeRecovery"]), float(row["finalGlobalRecovery"]))
        for row in rows
    )

    reasons: list[str] = []
    checks = (
        (
            candidate_edge >= REPRESENTATIVE_MEDIAN_EDGE_REQUIRED,
            f"median candidate edge recovery {candidate_edge:.1%} < "
            f"{REPRESENTATIVE_MEDIAN_EDGE_REQUIRED:.0%}",
        ),
        (
            candidate_global >= REPRESENTATIVE_MEDIAN_GLOBAL_REQUIRED,
            f"median candidate global recovery {candidate_global:.1%} < "
            f"{REPRESENTATIVE_MEDIAN_GLOBAL_REQUIRED:.0%}",
        ),
        (
            candidate_gradient >= REPRESENTATIVE_MEDIAN_GRADIENT_REQUIRED,
            f"median candidate gradient recovery {candidate_gradient:.1%} < "
            f"{REPRESENTATIVE_MEDIAN_GRADIENT_REQUIRED:.0%}",
        ),
        (
            edge_positive >= REPRESENTATIVE_POSITIVE_FRACTION_REQUIRED,
            f"positive-edge patch fraction {edge_positive:.1%} < "
            f"{REPRESENTATIVE_POSITIVE_FRACTION_REQUIRED:.0%}",
        ),
        (
            global_positive >= REPRESENTATIVE_POSITIVE_FRACTION_REQUIRED,
            f"positive-global patch fraction {global_positive:.1%} < "
            f"{REPRESENTATIVE_POSITIVE_FRACTION_REQUIRED:.0%}",
        ),
        (normal >= 0.0, f"median normal recovery is regressive ({normal:.1%})"),
        (material >= 0.0, f"median material recovery is regressive ({material:.1%})"),
        (
            worst_candidate >= REPRESENTATIVE_WORST_RECOVERY_FLOOR,
            f"candidate catastrophic patch recovery {worst_candidate:.1%} < "
            f"{REPRESENTATIVE_WORST_RECOVERY_FLOOR:.0%}",
        ),
        (
            worst_final >= REPRESENTATIVE_WORST_RECOVERY_FLOOR,
            f"final catastrophic patch recovery {worst_final:.1%} < "
            f"{REPRESENTATIVE_WORST_RECOVERY_FLOOR:.0%}",
        ),
        (
            edge_retention >= REPRESENTATIVE_SELECTOR_RETENTION_REQUIRED,
            f"median selector edge retention {edge_retention:.1%} < "
            f"{REPRESENTATIVE_SELECTOR_RETENTION_REQUIRED:.0%}",
        ),
        (
            global_retention >= REPRESENTATIVE_SELECTOR_RETENTION_REQUIRED,
            f"median selector global retention {global_retention:.1%} < "
            f"{REPRESENTATIVE_SELECTOR_RETENTION_REQUIRED:.0%}",
        ),
        (
            preservation >= float(PROTECTED_PRESERVATION_REQUIRED),
            f"protected-B preservation {preservation:.2%} < "
            f"{float(PROTECTED_PRESERVATION_REQUIRED):.0%}",
        ),
    )
    for passed, reason in checks:
        if not passed:
            reasons.append(reason)

    return {
        "pass": not reasons,
        "reasons": reasons,
        "patchCount": len(rows),
        "medianCandidateEdgeRecovery": candidate_edge,
        "medianCandidateGlobalRecovery": candidate_global,
        "medianCandidateGradientRecovery": candidate_gradient,
        "medianCandidateNormalRecovery": normal,
        "medianCandidateMaterialRecovery": material,
        "medianFinalEdgeRecovery": final_edge,
        "medianFinalGlobalRecovery": final_global,
        "positiveEdgePatchFraction": edge_positive,
        "positiveGlobalPatchFraction": global_positive,
        "worstCandidateRecovery": worst_candidate,
        "worstFinalRecovery": worst_final,
        "selectorEdgeRetention": edge_retention,
        "selectorGlobalRetention": global_retention,
        "protectedPreservation": preservation,
        "protectedPixelCount": int(protected_count),
        "requirements": {
            "medianCandidateEdgeRecovery": REPRESENTATIVE_MEDIAN_EDGE_REQUIRED,
            "medianCandidateGlobalRecovery": REPRESENTATIVE_MEDIAN_GLOBAL_REQUIRED,
            "medianCandidateGradientRecovery": REPRESENTATIVE_MEDIAN_GRADIENT_REQUIRED,
            "positivePatchFraction": REPRESENTATIVE_POSITIVE_FRACTION_REQUIRED,
            "selectorRetention": REPRESENTATIVE_SELECTOR_RETENTION_REQUIRED,
            "worstRecoveryFloor": REPRESENTATIVE_WORST_RECOVERY_FLOOR,
            "protectedPreservation": float(PROTECTED_PRESERVATION_REQUIRED),
        },
    }


def _run_representative_validation(
    config: V9Config,
    repo_root: Path,
    device_name: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Strict-load the selected Quick checkpoint and score held-out Raven crops."""
    import v9.training as training

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"V13.2 representative validator found no state_dict: {checkpoint_path}"
        )

    device = resolve_device(config, device_name)
    service = training._training_service
    service._configure_cuda(config, device)
    amp_dtype = service._resolve_amp_dtype(config, device)
    model = FidelityResidualNetV9(config)
    strict = model.load_state_dict(state_dict, strict=True)
    if strict.missing_keys or strict.unexpected_keys:
        raise RuntimeError(
            "V13.2 strict reload failed: "
            f"missing={strict.missing_keys} unexpected={strict.unexpected_keys}"
        )
    model = model.to(device)
    if hasattr(model, "set_inference_mode"):
        model.set_inference_mode()
    model.eval()

    manifest = load_dataset_manifest(repo_root, config)
    patch_count = max(REPRESENTATIVE_PATCHES, int(config.validation_tiles))
    dataset = PhysicalTileDatasetV9(
        manifest,
        config,
        "validation",
        patch_count,
        seed=int(config.seed) + 77,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows: list[dict[str, float]] = []
    protected_safe = 0
    protected_count = 0
    use_amp = device.type == "cuda"

    for index, raw_batch in enumerate(loader, start=1):
        batch = service._move_batch(raw_batch, device, channels_last=False)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            outputs = model(batch["input"])
        row, safe_count, total_count = _patch_metrics(outputs, batch, config)
        row["patch"] = float(index)
        rows.append(row)
        protected_safe += safe_count
        protected_count += total_count
        print(
            f"[v13.2/heldout] {index:02d}/{len(dataset):02d} "
            f"C edge={row['candidateEdgeRecovery']:+.1%} "
            f"global={row['candidateGlobalRecovery']:+.1%} "
            f"grad={row['candidateGradientRecovery']:+.1%} "
            f"F edge={row['finalEdgeRecovery']:+.1%}",
            flush=True,
        )

    summary = summarize_representative_validation(
        rows,
        protected_safe=protected_safe,
        protected_count=protected_count,
    )
    report = {
        "schema": "NSAMDR_V13_2_REPRESENTATIVE_RAVEN_V1",
        "revision": SR_GENERALIZATION_REVISION,
        **summary,
        "patches": rows,
    }
    evidence = checkpoint_path.parent.parent / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    report_path = evidence / "v13_2_representative_validation.json"
    report["reportPath"] = str(report_path.resolve())
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.v132.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.v132.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _promote_v132_quick(
    metadata: dict[str, Any],
    config: V9Config,
    repo_root: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Promote only the active V13.2 authority set after representative validation."""
    output_dir = (repo_root / config.output_dir).resolve()
    checkpoint_path = output_dir / config.checkpoint_name
    metadata_path = output_dir / config.metadata_name
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    source_selection = str(
        checkpoint.get("source_selection_kind")
        or metadata.get("sourceSelectionKind")
        or ""
    )
    final_qualification = checkpoint.get("final_qualification")
    cache_equivalence = checkpoint.get("cache_equivalence")
    prerequisites = bool(
        report.get("pass") is True
        and metadata.get("detailQualified") is True
        and source_selection == "production-final-selector-qualified"
        and isinstance(final_qualification, dict)
        and final_qualification.get("passed") is True
        and isinstance(cache_equivalence, dict)
        and cache_equivalence.get("passed") is True
    )
    if not prerequisites:
        metadata["srFirstGeneralizationRevision"] = SR_GENERALIZATION_REVISION
        metadata["srRepresentativeValidation"] = report
        return metadata

    checkpoint["selection_kind"] = "production-final"
    checkpoint["training_safety_pass"] = True
    checkpoint["acceptance_pass"] = True
    checkpoint["component_training_complete"] = True
    checkpoint["sr_first_generalization_revision"] = SR_GENERALIZATION_REVISION
    checkpoint["sr_active_training_authority"] = (
        "DetailNet multi-map SR + confidence/regret + BenefitSelector; "
        "legacy geometry/profile/seam modules are frozen non-pixel-authority evidence"
    )
    checkpoint["sr_representative_validation"] = report
    _atomic_torch(checkpoint_path, checkpoint)
    checkpoint_sha = _sha256(checkpoint_path)

    persisted = json.loads(metadata_path.read_text(encoding="utf-8"))
    persisted.update(
        {
            "selectionKind": "production-final",
            "trainingSafetyPass": True,
            "acceptancePass": True,
            "componentTrainingComplete": True,
            "checkpointSha256": checkpoint_sha,
            "srFirstGeneralizationRevision": SR_GENERALIZATION_REVISION,
            "srActiveTrainingAuthority": checkpoint["sr_active_training_authority"],
            "srRepresentativeValidation": report,
        }
    )
    _atomic_json(metadata_path, persisted)
    metadata.update(persisted)
    print(
        "[v13.2] REPRESENTATIVE RAVEN PASS: promoted SR-first Quick checkpoint "
        "to production-final authority.",
        flush=True,
    )
    return metadata


def _backend_run_v132(
    self: Any,
    config: V9Config,
    repo_root: Path,
    device: str,
    *,
    resume: bool,
    early_stop_patience: int,
    early_stop_min_delta: float,
    stop_after_phase: str | None,
) -> dict[str, Any]:
    if _ORIGINAL_BACKEND_RUN is None:
        raise RuntimeError("V13.2 installed without TrainingBackend base method")
    metadata = _ORIGINAL_BACKEND_RUN(
        self,
        config,
        repo_root,
        device,
        resume=resume,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        stop_after_phase=stop_after_phase,
    )
    if not is_sr_first_quick_config(config) or stop_after_phase is not None:
        return metadata

    output_dir = (repo_root / config.output_dir).resolve()
    checkpoint_path = output_dir / config.checkpoint_name
    print(
        f"[v13.2] running {max(REPRESENTATIVE_PATCHES, int(config.validation_tiles))} "
        "deterministic held-out Raven patches before promotion...",
        flush=True,
    )
    report = _run_representative_validation(
        config,
        repo_root,
        device,
        checkpoint_path,
    )
    print(
        "[v13.2] held-out median: "
        f"edge={float(report.get('medianCandidateEdgeRecovery', -1.0)):+.1%} "
        f"global={float(report.get('medianCandidateGlobalRecovery', -1.0)):+.1%} "
        f"gradient={float(report.get('medianCandidateGradientRecovery', -1.0)):+.1%} "
        f"retention={float(report.get('selectorEdgeRetention', 0.0)):.1%}/"
        f"{float(report.get('selectorGlobalRetention', 0.0)):.1%} "
        f"protect={float(report.get('protectedPreservation', 0.0)):.2%}",
        flush=True,
    )
    if report.get("pass") is not True:
        print("[v13.2] representative validation FAILED:", flush=True)
        for reason in report.get("reasons", []):
            print(f"  - {reason}", flush=True)
        metadata["srFirstGeneralizationRevision"] = SR_GENERALIZATION_REVISION
        metadata["srRepresentativeValidation"] = report
        return metadata

    return _promote_v132_quick(metadata, config, repo_root, report)


def _architecture_contract_v132(
    self: FidelityResidualNetV9,
) -> dict[str, object]:
    if _ORIGINAL_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError("V13.2 installed without architecture contract")
    contract = dict(_ORIGINAL_ARCHITECTURE_CONTRACT(self))
    contract["srGeneralizationRevision"] = SR_GENERALIZATION_REVISION
    contract["quickTrainingAuthority"] = (
        "DetailNet multi-map SR candidate C followed by BenefitSelector; "
        "geometry/profile/seam are frozen compatibility/evidence modules "
        "with no pixel authority"
    )
    contract["representativeRavenQualification"] = {
        "patches": REPRESENTATIVE_PATCHES,
        "medianEdgeRecovery": REPRESENTATIVE_MEDIAN_EDGE_REQUIRED,
        "medianGlobalRecovery": REPRESENTATIVE_MEDIAN_GLOBAL_REQUIRED,
        "medianGradientRecovery": REPRESENTATIVE_MEDIAN_GRADIENT_REQUIRED,
        "positivePatchFraction": REPRESENTATIVE_POSITIVE_FRACTION_REQUIRED,
        "selectorRetention": REPRESENTATIVE_SELECTOR_RETENTION_REQUIRED,
        "protectedPreservation": float(PROTECTED_PRESERVATION_REQUIRED),
        "worstRecoveryFloor": REPRESENTATIVE_WORST_RECOVERY_FLOOR,
    }
    return contract


def install_sr_first_generalization_contract() -> None:
    """Install representative qualification on the current SR-first backend."""
    global _INSTALLED, _ORIGINAL_BACKEND_RUN, _ORIGINAL_ARCHITECTURE_CONTRACT
    if _INSTALLED:
        return

    from .application.backend import TrainingBackend

    _ORIGINAL_BACKEND_RUN = TrainingBackend.run
    _ORIGINAL_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract
    TrainingBackend.run = _backend_run_v132
    FidelityResidualNetV9.architecture_contract = _architecture_contract_v132
    _INSTALLED = True
