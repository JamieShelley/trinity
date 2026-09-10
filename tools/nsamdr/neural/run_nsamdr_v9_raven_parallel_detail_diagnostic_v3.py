#!/usr/bin/env python3
"""V12.5 integration proof for the baseline-relative direct-detail specialist.

The V2 proof predates parallel G/S/D fusion and therefore compared its isolated
B->D selector path with an unrestricted production B->U->F path. V3 keeps the
same fast detail/selector optimisation but explicitly gives structure and seam
zero support during the production parity check. Under the V12.5 fusion contract
that makes U exactly D, so the diagnostic again tests one thing at a time:

    direct detail capacity -> exact D integration -> BenefitSelector retention.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

import _run_nsamdr_v9_raven_parallel_detail_diagnostic_v2_impl as _implementation
from v9.diagnostics_layout import run_consolidated_diagnostic
from v9.model import UPSCALE_FACTOR


REPORT_SCHEMA = "NSAMDR_RAVEN_PARALLEL_DETAIL_INTEGRATION_V3"


# Purpose: Build the exact selector evidence used when V12.5 fusion contains D only.
# Called by: the reused V2 selector optimiser.
# Calls: observable LR support plus V12.5 selector feature construction.
def _selector_evidence_v125(
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
    detail_evidence = (
        ((confidence - 0.50) / 0.35).clamp(0.0, 1.0)
        * ((0.50 - regret) / 0.35).clamp(0.0, 1.0)
    )
    conflict = torch.zeros_like(detail_evidence)
    return model._selector_features(
        baseline,
        candidate,
        one,
        two,
        one,
        one,
        observed_support,
        one,
        detail_evidence,
        conflict,
    )


# Purpose: Run production V12.5 with G/S support forced to identity for D parity proof.
# Called by: the reused V2 production parity/final checks.
# Calls: the installed FidelityResidualNetV9._forward_impl V12.5 composition.
def _full_production_v125_detail_only(
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


# Purpose: Run the V12.5-corrected parallel-detail proof under canonical storage.
# Called by: GUI compatibility launcher / CLI.
# Calls: the V2 optimiser implementation with V12.5 parity helpers installed.
def main(argv: list[str] | None = None) -> int:
    _implementation.REPORT_SCHEMA = REPORT_SCHEMA
    _implementation._selector_evidence = _selector_evidence_v125
    _implementation._full_production = _full_production_v125_detail_only
    return run_consolidated_diagnostic(
        _implementation.main,
        argv,
        legacy_folder="parallel_detail_diagnostics",
        category="parallel_detail",
    )


if __name__ == "__main__":
    raise SystemExit(main())
