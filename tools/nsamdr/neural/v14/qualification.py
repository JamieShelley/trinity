from __future__ import annotations

import statistics

import torch
from torch.nn import functional as F

from .config import V14Config


def _gradient_map(value: torch.Tensor) -> torch.Tensor:
    gray = value.float().mean(dim=1, keepdim=True)
    dx = F.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    return dx + dy


def _recovery(baseline_error: torch.Tensor, candidate_error: torch.Tensor) -> float:
    b = float(baseline_error.mean().item())
    c = float(candidate_error.mean().item())
    return (b - c) / max(b, 1.0e-8)


def _phase_cell_projection_fraction(residual: torch.Tensor, scale: int, oy: int, ox: int) -> float:
    h, w = residual.shape[-2:]
    usable_h = ((h - oy) // scale) * scale
    usable_w = ((w - ox) // scale) * scale
    if usable_h < scale or usable_w < scale:
        return 0.0
    value = residual[..., oy:oy + usable_h, ox:ox + usable_w].float()
    magnitude = float(value.abs().mean().item())
    if magnitude < 1.0e-5:
        return 0.0
    projected = F.interpolate(
        F.avg_pool2d(value, scale, scale),
        size=value.shape[-2:],
        mode="nearest",
    )
    return float(projected.abs().mean().item() / max(magnitude, 1.0e-8))


def _phase_independent_lattice(
    residual: torch.Tensor,
    target_residual: torch.Tensor,
    scale: int,
) -> tuple[float, float, float]:
    candidate_fractions: list[float] = []
    target_fractions: list[float] = []
    excesses: list[float] = []
    for oy in range(scale):
        for ox in range(scale):
            candidate = _phase_cell_projection_fraction(residual, scale, oy, ox)
            target = _phase_cell_projection_fraction(target_residual, scale, oy, ox)
            candidate_fractions.append(candidate)
            target_fractions.append(target)
            excesses.append(candidate - target)
    return max(excesses), max(candidate_fractions), max(target_fractions)


def sample_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *, final: bool = False) -> dict[str, float]:
    target = batch["target_albedo"].float()
    target_n = batch["target_normal"].float()
    target_m = batch["target_material"].float()
    baseline = outputs["baseline_albedo"].float()
    baseline_n = outputs["baseline_normal"].float()
    baseline_m = outputs["baseline_material"].float()
    value = outputs["albedo" if final else "candidate_albedo"].float()
    value_n = outputs["normal" if final else "candidate_normal"].float()
    value_m = outputs["material" if final else "candidate_material"].float()

    b_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    v_error = (value - target).abs().mean(dim=1, keepdim=True)
    global_recovery = _recovery(b_error, v_error)

    target_grad = _gradient_map(target)
    baseline_grad_error = (_gradient_map(baseline) - target_grad).abs()
    value_grad_error = (_gradient_map(value) - target_grad).abs()
    gradient_recovery = _recovery(baseline_grad_error, value_grad_error)

    edge_weight = 1.0 + 4.0 * target_grad / target_grad.mean().clamp_min(1.0e-6)
    b_edge = (b_error * edge_weight).mean()
    v_edge = (v_error * edge_weight).mean()
    edge_recovery = float((b_edge - v_edge).item() / max(float(b_edge.item()), 1.0e-8))

    normal_recovery = _recovery((baseline_n - target_n).abs(), (value_n - target_n).abs())
    material_recovery = _recovery((baseline_m - target_m).abs(), (value_m - target_m).abs())
    residual = value - baseline
    target_residual = target - baseline
    lattice_excess, lattice_candidate, lattice_target = _phase_independent_lattice(
        residual, target_residual, 4
    )

    protected = (baseline - target).abs().amax(dim=1, keepdim=True) <= (2.0 / 255.0)
    if bool(protected.any().item()):
        preserved = (value - baseline).abs().amax(dim=1, keepdim=True) <= (1.0 / 255.0)
        protected_rate = float(preserved[protected].float().mean().item())
    else:
        protected_rate = 1.0

    return {
        "global_recovery": global_recovery,
        "edge_recovery": edge_recovery,
        "gradient_recovery": gradient_recovery,
        "normal_recovery": normal_recovery,
        "material_recovery": material_recovery,
        "lattice_cell_excess": lattice_excess,
        "lattice_candidate_fraction": lattice_candidate,
        "lattice_target_fraction": lattice_target,
        "protected_preservation": protected_rate,
    }


def _median(items: list[float]) -> float:
    return float(statistics.median(items)) if items else float("nan")


def aggregate_candidate(metrics: list[dict[str, float]], config: V14Config) -> dict[str, object]:
    global_values = [m["global_recovery"] for m in metrics]
    edge_values = [m["edge_recovery"] for m in metrics]
    grad_values = [m["gradient_recovery"] for m in metrics]
    normal_values = [m["normal_recovery"] for m in metrics]
    material_values = [m["material_recovery"] for m in metrics]
    lattice_values = [m["lattice_cell_excess"] for m in metrics]
    protected_values = [m["protected_preservation"] for m in metrics]
    enough_heldout = len(metrics) >= int(config.minimum_heldout_samples)
    result: dict[str, object] = {
        "medianGlobalRecovery": _median(global_values),
        "medianEdgeRecovery": _median(edge_values),
        "medianGradientRecovery": _median(grad_values),
        "medianNormalRecovery": _median(normal_values),
        "medianMaterialRecovery": _median(material_values),
        "positiveGlobalFraction": sum(v > 0 for v in global_values) / max(1, len(global_values)),
        "positiveEdgeFraction": sum(v > 0 for v in edge_values) / max(1, len(edge_values)),
        "worstGlobalRecovery": min(global_values) if global_values else float("nan"),
        "maxLatticeCellExcess": max(lattice_values) if lattice_values else float("nan"),
        "medianProtectedPreservation": _median(protected_values),
        "sampleCount": len(metrics),
        "minimumHeldOutSamplesRequired": int(config.minimum_heldout_samples),
        "heldOutCoveragePass": enough_heldout,
    }
    result["passed"] = bool(
        enough_heldout
        and result["medianEdgeRecovery"] >= config.candidate_edge_recovery_required
        and result["medianGlobalRecovery"] >= config.candidate_global_recovery_required
        and result["medianGradientRecovery"] >= config.candidate_gradient_recovery_required
        and result["positiveEdgeFraction"] >= config.candidate_positive_edge_fraction_required
        and result["positiveGlobalFraction"] >= config.candidate_positive_global_fraction_required
        and result["medianNormalRecovery"] >= 0.0
        and result["medianMaterialRecovery"] >= 0.0
        and result["worstGlobalRecovery"] >= config.candidate_worst_recovery_min
        and result["maxLatticeCellExcess"] <= config.candidate_lattice_cell_excess_max
    )
    return result


def aggregate_final(
    candidate: dict[str, object],
    metrics: list[dict[str, float]],
    config: V14Config,
) -> dict[str, object]:
    final = aggregate_candidate(metrics, config)
    candidate_edge = float(candidate["medianEdgeRecovery"])
    candidate_global = float(candidate["medianGlobalRecovery"])
    edge = float(final["medianEdgeRecovery"])
    glob = float(final["medianGlobalRecovery"])
    protected = float(final["medianProtectedPreservation"])
    final["selectorEdgeRetention"] = edge / max(candidate_edge, 1.0e-8)
    final["selectorGlobalRetention"] = glob / max(candidate_global, 1.0e-8)
    final["passed"] = bool(
        final["heldOutCoveragePass"]
        and edge >= 0.0
        and glob >= 0.0
        and float(final["medianNormalRecovery"]) >= 0.0
        and float(final["medianMaterialRecovery"]) >= 0.0
        and final["selectorEdgeRetention"] >= config.selector_edge_retention_required
        and final["selectorGlobalRetention"] >= config.selector_global_retention_required
        and protected >= config.protected_preservation_required
        and float(final["worstGlobalRecovery"]) >= config.candidate_worst_recovery_min
        and float(final["maxLatticeCellExcess"]) <= config.candidate_lattice_cell_excess_max
    )
    return final
