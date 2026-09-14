#!/usr/bin/env python3
"""V16 GUI entry point with a lower-impact Stage 2 GPU duty profile.

The low-impact profile changes execution pacing only. It does not change the V16
model, optimizer, learning rate, dataset, qualification gates, or checkpoint
compatibility, so an interrupted Stage 2 run can resume under this profile.
"""
from __future__ import annotations

import nsamdr_v16_multifamily_workflow_gui as multifamily


base = multifamily.base
_original_select_multiregion = multifamily._select_multiregion
_original_args = base.App._args

# Deliberately conservative interactive-desktop profile. Training still runs the
# same optimizer steps, but shorter CUDA bursts and a lower thermal ceiling leave
# more idle time for the desktop and other applications.
LOW_IMPACT_PROFILE = {
    "cooldownEvery": "4",
    "cooldownSeconds": "1.5",
    "maxGpuTemperature": "75",
    "temperatureResumeMargin": "7",
    "thermalPollSeconds": "3",
    "validationCooldownSeconds": "5",
}


def _select_multiregion(self: base.App, stage: base.Stage, status: str) -> None:
    _original_select_multiregion(self, stage, status)
    self._label_row(
        "Interactive GPU profile",
        "Low-impact mode shortens sustained CUDA bursts and caps Stage 2 at 75 C so other desktop programs have more headroom.",
    )
    self._check(
        "Low-impact GPU mode (recommended while using this PC)",
        "stage2_low_impact_gpu",
        True,
    )
    self._label_row(
        "Low-impact pacing",
        "1.5 s synchronized idle every 4 optimizer steps; thermal pause at 75 C and resume at 68 C.",
    )
    self._label_row(
        "Training contract",
        "Pacing only: model, optimizer, LR, samples, gates, and resume checkpoint remain unchanged.",
    )


def _args(self: base.App, stage_id: str) -> list[str]:
    values = _original_args(self, stage_id)
    if stage_id != "multiregion":
        return values

    low_impact = self.vars.get("stage2_low_impact_gpu")
    if low_impact is None or not bool(low_impact.get()):
        return values

    if "--safe-mode" not in values:
        values.append("--safe-mode")
    values.extend(
        (
            "--cooldown-every",
            LOW_IMPACT_PROFILE["cooldownEvery"],
            "--cooldown-seconds",
            LOW_IMPACT_PROFILE["cooldownSeconds"],
            "--max-gpu-temperature",
            LOW_IMPACT_PROFILE["maxGpuTemperature"],
            "--temperature-resume-margin",
            LOW_IMPACT_PROFILE["temperatureResumeMargin"],
            "--thermal-poll-seconds",
            LOW_IMPACT_PROFILE["thermalPollSeconds"],
            "--validation-cooldown-seconds",
            LOW_IMPACT_PROFILE["validationCooldownSeconds"],
        )
    )
    return values


base.App._args = _args
multifamily.v16.legacy._select_multiregion = _select_multiregion


if __name__ == "__main__":
    raise SystemExit(base.main())
