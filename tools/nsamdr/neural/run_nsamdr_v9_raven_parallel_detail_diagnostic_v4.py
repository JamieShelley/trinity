#!/usr/bin/env python3
"""V12.6 direct-detail integration proof with calibrated fusion evidence.

Structure and seam are forced to zero authority so U must equal D exactly. The
selector evidence now matches V12.6: continuous detail confidence*(1-regret)
plus zero fusion conflict.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

import _run_nsamdr_v9_raven_parallel_detail_diagnostic_v2_impl as _implementation
from v9.diagnostics_layout import run_consolidated_diagnostic
from v9.model import UPSCALE_FACTOR


REPORT_SCHEMA = "NSAMDR_RAVEN_PARALLEL_DETAIL_INTEGRATION_V4"


# Purpose: Build the exact V12.6 selector evidence for a D-only fused candidate.
# Called by: the reused fast selector optimiser.
# Calls: observable LR support and the production selector feature builder.
def _selector_evidence_v126(
    model: Any,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    detail: dict[str, torch.Tensor],
) -> torch.Tensor:
    source_support = model._source_edge_support(
        inputs, int(model.config.geometry_edge_support_radius)
    )
    observed_support = F.interpolate(
        source_support,
        scale_factor=UPSCALE_FACTOR,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    one = torch.zeros_like(observed_support)
    two = torch.zeros(
        (
            inputs.shape[0],
            2,
            observed_support.shape[-2],
            observed_support.shape[-1],
        ),
        device=inputs.device,
        dtype=observed_support.dtype,
    )
    confidence = torch.sigmoid(detail["confidence_logits"].float())
    regret = torch.sigmoid(detail["regret_logits"].float())
    detail_support = confidence * (1.0 - regret)
    conflict = torch.zeros_like(detail_support)
    return model._selector_features(
        baseline,
        candidate,
        one,
        two,
        one,
        one,
        observed_support,
        one,
        detail_support,
        conflict,
    )


# Purpose: Execute production V12.6 with G/S authority disabled for exact D parity.
# Called by: the reused parity/final checks.
# Calls: installed production _forward_impl.
def _full_production_v126_detail_only(
    model: Any,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    inputs = batch["input"]
    height = int(inputs.shape[-2]) * UPSCALE_FACTOR
    width = int(inputs.shape[-1]) * UPSCALE_FACTOR
    zero_authority = torch.zeros(
        (inputs.shape[0], 1, height, width),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=inputs.device.type,
        enabled=False,
    ):
        return model._forward_impl(
            inputs,
            gate_override=zero_authority,
            seam_authority_override=zero_authority,
        )


# Purpose: Run the V12.6-corrected parallel-detail proof under canonical storage.
# Called by: GUI compatibility launcher / CLI.
# Calls: reused V2 optimiser with V12.6 production parity helpers.
def main(argv: list[str] | None = None) -> int:
    _implementation.REPORT_SCHEMA = REPORT_SCHEMA
    _implementation._selector_evidence = _selector_evidence_v126
    _implementation._full_production = _full_production_v126_detail_only
    return run_consolidated_diagnostic(
        _implementation.main,
        argv,
        legacy_folder="parallel_detail_diagnostics",
        category="parallel_detail",
    )


if __name__ == "__main__":
    raise SystemExit(main())
