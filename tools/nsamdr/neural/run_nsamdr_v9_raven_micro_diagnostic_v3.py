#!/usr/bin/env python3
"""V12.5 staged Micro view for independent G/S/D fusion and final authority."""
from __future__ import annotations

from typing import Any

import torch

import _run_nsamdr_v9_raven_micro_diagnostic_impl as _implementation
from v9.diagnostics_layout import run_consolidated_diagnostic


REPORT_SCHEMA = "NSAMDR_RAVEN_MICRO_PARALLEL_SPECIALIST_DIAGNOSTIC_V3"

_ORIGINAL_STAGE_METRICS = _implementation._stage_metrics


# Keep the historical labels consumed by the V2 pass logic, but map them onto the
# actual V12.5 independent specialist candidates. Add U as an extra observable stage.
CANDIDATE_KEYS = (
    ("rawBoundary", "boundary_initial_candidate_albedo"),
    ("profiledBoundary", "boundary_candidate_albedo"),
    ("preSeam", "structure_candidate_albedo"),
    ("postSeam", "seam_candidate_albedo"),
    ("detailCandidate", "detail_candidate_albedo"),
    ("fusedCandidate", "fused_candidate_albedo"),
    ("final", "albedo"),
)


# Purpose: Extend Micro telemetry with the V12.5 specialist support/fusion fields.
# Called by: reused Micro evaluation loop.
# Calls: the original metric implementation and scalar tensor means.
def _stage_metrics_v125(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    result = _ORIGINAL_STAGE_METRICS(outputs, batch)
    authority = result.setdefault("authority", {})
    for label, key in (
        ("parallelStructureSupportMean", "parallel_structure_support"),
        ("parallelSeamSupportMean", "parallel_seam_support"),
        ("parallelDetailSupportMean", "parallel_detail_support"),
        ("parallelFusionSupportMean", "parallel_fusion_support"),
        ("parallelFusionConflictMean", "parallel_fusion_conflict"),
    ):
        value = outputs.get(key)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            authority[label] = float(value.detach().float().mean().item())
        else:
            authority[label] = None
    return result


# Purpose: Render the actual V12.5 A/B/G/S/D/U/F production chain.
# Called by: reused Micro initial/current/final probe writer calls.
# Calls: PIL and the implementation's tensor-to-RGB helper.
def _write_probe_sheet_v125(
    path: Any,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    metrics: dict[str, Any],
    title: str,
) -> None:
    from PIL import Image, ImageDraw

    target = _implementation._rgb(batch["target_albedo"])
    baseline = _implementation._rgb(outputs["baseline_albedo"])
    structure = outputs.get(
        "structure_candidate_albedo",
        outputs.get("boundary_pre_seam_albedo", outputs["baseline_albedo"]),
    )
    seam = outputs.get(
        "seam_candidate_albedo",
        outputs.get("boundary_reconstructed_albedo", outputs["baseline_albedo"]),
    )
    detail = outputs.get("detail_candidate_albedo", outputs["baseline_albedo"])
    fused = outputs.get("fused_candidate_albedo", detail)
    final = outputs["albedo"]

    chosen = (
        ("A TARGET", target, None),
        ("B BASELINE", baseline, None),
        ("G STRUCTURE", _implementation._rgb(structure), "preSeam"),
        ("S SEAM", _implementation._rgb(seam), "postSeam"),
        ("D DETAIL", _implementation._rgb(detail), "detailCandidate"),
        ("U FUSED", _implementation._rgb(fused), "fusedCandidate"),
        ("F FINAL", _implementation._rgb(final), "final"),
    )
    h, w = target.shape[:2]
    header = 64
    canvas = Image.new("RGB", (w * len(chosen), h + header), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    candidate_metrics = metrics.get("candidates", {})
    for index, (label, panel, metric_key) in enumerate(chosen):
        x = index * w
        canvas.paste(Image.fromarray(panel, mode="RGB"), (x, header))
        draw.text((x + 5, 6), label, fill=(245, 245, 245))
        if metric_key and metric_key in candidate_metrics:
            item = candidate_metrics[metric_key]
            draw.text(
                (x + 5, 25),
                f"edge {item['edgeRecovery']:+.1%}",
                fill=(225, 225, 225),
            )
            draw.text(
                (x + 5, 42),
                f"global {item['globalRecovery']:+.1%}",
                fill=(225, 225, 225),
            )
    draw.text((5, h + header - 17), title, fill=(210, 210, 210))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


# Purpose: Run the staged Micro proof using V12.5 candidate naming/visualisation.
# Called by: compatibility launcher / GUI.
# Calls: unchanged Micro optimiser loop through the canonical diagnostic wrapper.
def main(argv: list[str] | None = None) -> int:
    _implementation.REPORT_SCHEMA = REPORT_SCHEMA
    _implementation.CANDIDATE_KEYS = CANDIDATE_KEYS
    _implementation._stage_metrics = _stage_metrics_v125
    _implementation._write_probe_sheet = _write_probe_sheet_v125
    return run_consolidated_diagnostic(
        _implementation.main,
        argv,
        legacy_folder="micro_diagnostics",
        category="micro",
    )


if __name__ == "__main__":
    raise SystemExit(main())
