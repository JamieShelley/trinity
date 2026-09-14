#!/usr/bin/env python3
"""Low-impact V16 Stage 2 runtime with truthful telemetry and leaner VRAM use.

This wrapper preserves the V16 model/data/optimizer contract while improving the
execution surface used by the GUI:

- rolling nvidia-smi telemetry so progress lines report sustained load rather than
  a single post-cooldown sample,
- memory-lean Adam execution (foreach=False),
- cached CUDA blocks are released at synchronized low-impact cooldown points,
- a lightweight resume sidecar records the exact durable checkpoint step,
- live probes move tensors to CPU before image composition and provide both compact
  and all-map contact sheets.
"""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NEURAL_ROOT = HERE.parent
if str(NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURAL_ROOT))

from v14 import safe_live_resume_fourfamily_multiregion_diagnostic as resume

live = resume.live
safe = resume.safe
base = safe.base

_ORIGINAL_SAFE_VALUES = safe.SafeFourFamilyDiagnostic._safe_values
_ORIGINAL_GPU_TELEMETRY = safe.SafeFourFamilyDiagnostic._gpu_telemetry
_ORIGINAL_GPU_TEXT = safe.SafeFourFamilyDiagnostic._gpu_text
_ORIGINAL_WRITE_RESUME = safe.SafeFourFamilyDiagnostic._write_resume_checkpoint
_ORIGINAL_ADAM = safe.torch.optim.Adam
_ORIGINAL_SYNCHRONIZE = safe.torch.cuda.synchronize


def _device_index(diagnostic: safe.SafeFourFamilyDiagnostic) -> int:
    index = diagnostic.device.index
    if index is None:
        index = torch.cuda.current_device()
    return int(index)


def _smi_sample(device_index: int) -> dict[str, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device_index),
                "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        if len(parts) < 5:
            return None
        values = [float(value) for value in parts[:5]]
        return {
            "temperatureC": values[0],
            "powerW": values[1],
            "utilizationPercent": values[2],
            "vramUsedMiB": values[3],
            "vramTotalMiB": values[4],
            "sampleUnix": time.time(),
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _ensure_sampler(diagnostic: safe.SafeFourFamilyDiagnostic) -> None:
    if diagnostic.device.type != "cuda" or not torch.cuda.is_available():
        return
    if getattr(diagnostic, "_stage2_gpu_sampler_started", False):
        return

    diagnostic._stage2_gpu_sampler_started = True  # type: ignore[attr-defined]
    diagnostic._stage2_gpu_samples = deque(maxlen=16)  # type: ignore[attr-defined]
    diagnostic._stage2_gpu_samples_lock = threading.Lock()  # type: ignore[attr-defined]
    index = _device_index(diagnostic)

    def worker() -> None:
        while True:
            sample = _smi_sample(index)
            if sample is not None:
                lock = diagnostic._stage2_gpu_samples_lock  # type: ignore[attr-defined]
                with lock:
                    diagnostic._stage2_gpu_samples.append(sample)  # type: ignore[attr-defined]
            time.sleep(0.75)

    threading.Thread(
        target=worker,
        name="nsamdr-stage2-gpu-sampler",
        daemon=True,
    ).start()


def _safe_values(self: safe.SafeFourFamilyDiagnostic) -> dict[str, Any]:
    values = _ORIGINAL_SAFE_VALUES(self)
    _ensure_sampler(self)
    values["telemetryWindowSeconds"] = 12.0
    values["adamForeach"] = False
    values["releaseCudaCacheAtCooldown"] = True
    return values


def _gpu_telemetry(self: safe.SafeFourFamilyDiagnostic) -> dict[str, Any]:
    current = _ORIGINAL_GPU_TELEMETRY(self)
    _ensure_sampler(self)
    samples: list[dict[str, float]] = []
    lock = getattr(self, "_stage2_gpu_samples_lock", None)
    stored = getattr(self, "_stage2_gpu_samples", None)
    if lock is not None and stored is not None:
        with lock:
            samples = list(stored)
    if not samples:
        return current

    def values(name: str) -> list[float]:
        return [float(sample[name]) for sample in samples if name in sample]

    util = values("utilizationPercent")
    power = values("powerW")
    temp = values("temperatureC")
    vram = values("vramUsedMiB")
    result = dict(current)
    result.update(
        {
            "rollingSampleCount": len(samples),
            "rollingWindowSeconds": 12.0,
            "utilizationAveragePercent": sum(util) / len(util) if util else None,
            "utilizationPeakPercent": max(util) if util else None,
            "powerAverageW": sum(power) / len(power) if power else None,
            "powerPeakW": max(power) if power else None,
            "temperaturePeakC": max(temp) if temp else None,
            "vramPeakMiB": max(vram) if vram else None,
        }
    )
    return result


def _gpu_text(telemetry: dict[str, Any]) -> str:
    if not telemetry:
        return "gpu-telemetry=unavailable"

    pieces: list[str] = []
    temp_now = telemetry.get("temperatureC")
    temp_peak = telemetry.get("temperaturePeakC")
    if temp_now is not None:
        if temp_peak is not None:
            pieces.append(f"temp={float(temp_now):.0f}C peak={float(temp_peak):.0f}C")
        else:
            pieces.append(f"temp={float(temp_now):.0f}C")

    power_now = telemetry.get("powerW")
    power_avg = telemetry.get("powerAverageW")
    power_limit = telemetry.get("powerLimitW")
    if power_avg is not None:
        text = f"power~{float(power_avg):.0f}W"
        if power_now is not None:
            text += f" now={float(power_now):.0f}W"
        if power_limit is not None:
            text += f"/{float(power_limit):.0f}W"
        pieces.append(text)
    elif power_now is not None:
        pieces.append(f"power={float(power_now):.0f}W")

    util_now = telemetry.get("utilizationPercent")
    util_avg = telemetry.get("utilizationAveragePercent")
    util_peak = telemetry.get("utilizationPeakPercent")
    if util_avg is not None:
        text = f"gpu~{float(util_avg):.0f}%avg"
        if util_now is not None:
            text += f" now={float(util_now):.0f}%"
        if util_peak is not None:
            text += f" peak={float(util_peak):.0f}%"
        pieces.append(text)
    elif util_now is not None:
        pieces.append(f"gpu={float(util_now):.0f}%")

    used = telemetry.get("vramUsedMiB")
    total = telemetry.get("vramTotalMiB")
    peak = telemetry.get("vramPeakMiB")
    if used is not None and total is not None:
        text = f"vram={float(used)/1024.0:.1f}/{float(total)/1024.0:.1f}GiB"
        if peak is not None:
            text += f" peak~{float(peak)/1024.0:.1f}GiB"
        pieces.append(text)
    return " ".join(pieces) if pieces else _ORIGINAL_GPU_TEXT(telemetry)


def _memory_lean_adam(*args: Any, **kwargs: Any) -> torch.optim.Optimizer:
    # PyTorch's foreach Adam path can require an additional tensor-list-sized
    # allocation. The scalar loop is slower but materially lowers peak allocator
    # pressure while preserving Adam's state and checkpoint compatibility.
    kwargs.setdefault("foreach", False)
    return _ORIGINAL_ADAM(*args, **kwargs)


def _synchronize_and_release(*args: Any, **kwargs: Any) -> None:
    _ORIGINAL_SYNCHRONIZE(*args, **kwargs)
    try:
        safe.torch.cuda.empty_cache()
    except RuntimeError:
        pass


def _write_resume_checkpoint(
    self: safe.SafeFourFamilyDiagnostic,
    path: Path,
    **kwargs: Any,
) -> None:
    _ORIGINAL_WRITE_RESUME(self, path, **kwargs)
    safe._atomic_json(  # noqa: SLF001 - companion metadata for the same checkpoint
        path.with_name("resume_checkpoint.json"),
        {
            "schema": "NSAMDR_V16_STAGE2_RESUME_SIDECAR_V1",
            "checkpoint": str(path.resolve()),
            "runDir": str(Path(kwargs["run_dir"]).resolve()),
            "step": int(kwargs["step"]),
            "maximumSteps": int(kwargs["max_steps"]),
            "elapsedSeconds": float(kwargs["elapsed_seconds"]),
            "updatedUnix": time.time(),
        },
    )


def _write_live_probes_low_vram(
    diagnostic: safe.SafeFourFamilyDiagnostic,
    model: Any,
    records: list[dict[str, Any]],
    config: Any,
    report: dict[str, object],
) -> None:
    context = live._probe_context(diagnostic)  # noqa: SLF001
    if context is None:
        return
    run_dir, step = context
    step_dir = run_dir / "probes" / f"step_{step:04d}"
    complete_marker = step_dir / "probe_index.json"
    if complete_marker.is_file():
        return

    latest_dir = run_dir / "probes" / "latest"
    per_family = dict(report.get("perFamily") or {})
    overviews: list[np.ndarray] = []
    detailed_overviews: list[np.ndarray] = []
    files: list[dict[str, str]] = []

    model.eval()
    with torch.no_grad():
        for record in records:
            family_id = str(record.get("family_id") or "unknown")
            family_name = str(record.get("source_asset_name") or family_id)
            batch = base.dataset_sample(record, config, diagnostic.device)
            with base.autocast_context(diagnostic.device, diagnostic.args.amp_precision):
                outputs = model(
                    batch["lr_albedo"],
                    batch["lr_normal"],
                    batch["lr_material"],
                )

            # Copy only the six maps used by the visualizer, then relinquish all
            # CUDA references before OpenCV composition/encoding.
            cpu_batch = {
                "target_albedo": batch["target_albedo"].detach().float().cpu(),
                "target_normal": batch["target_normal"].detach().float().cpu(),
                "target_material": batch["target_material"].detach().float().cpu(),
            }
            cpu_outputs = {
                "baseline_albedo": outputs["baseline_albedo"].detach().float().cpu(),
                "candidate_albedo": outputs["candidate_albedo"].detach().float().cpu(),
                "baseline_normal": outputs["baseline_normal"].detach().float().cpu(),
                "candidate_normal": outputs["candidate_normal"].detach().float().cpu(),
                "baseline_material": outputs["baseline_material"].detach().float().cpu(),
                "candidate_material": outputs["candidate_material"].detach().float().cpu(),
            }
            del batch, outputs
            if diagnostic.device.type == "cuda":
                safe.torch.cuda.synchronize(diagnostic.device)
                time.sleep(0.35)

            metrics = per_family.get(family_id)
            detailed, overview = live._family_visual(  # noqa: SLF001
                cpu_batch,
                cpu_outputs,
                family_name,
                metrics if isinstance(metrics, dict) else None,
                step,
            )
            filename = f"{live._slug(family_name)}.png"  # noqa: SLF001
            live._atomic_png(step_dir / filename, detailed)  # noqa: SLF001
            live._atomic_png(latest_dir / filename, detailed)  # noqa: SLF001
            overviews.append(overview)

            scale = min(1.0, 960.0 / float(detailed.shape[1]))
            if scale < 1.0:
                reduced = cv2.resize(
                    detailed,
                    (int(detailed.shape[1] * scale), int(detailed.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                reduced = detailed
            detailed_overviews.append(reduced)
            files.append({"familyId": family_id, "family": family_name, "file": filename})

    if overviews:
        contact = np.concatenate(overviews, axis=0)
        gate_banner = live._banner(  # noqa: SLF001
            contact.shape[1],
            "QUALIFICATION GATES: global >=45%   edge >=60%   gradient >=35%   lattice <=15%",
        )
        contact = np.concatenate((gate_banner, contact), axis=0)
        live._atomic_png(step_dir / "ALL_FAMILIES.png", contact)  # noqa: SLF001
        live._atomic_png(latest_dir / "ALL_FAMILIES.png", contact)  # noqa: SLF001

    if detailed_overviews:
        detail_contact = np.concatenate(detailed_overviews, axis=0)
        live._atomic_png(step_dir / "ALL_FAMILIES_DETAIL.png", detail_contact)  # noqa: SLF001
        live._atomic_png(latest_dir / "ALL_FAMILIES_DETAIL.png", detail_contact)  # noqa: SLF001

    index = {
        "schema": "NSAMDR_V16_STAGE2_LIVE_PROBE_V2",
        "step": step,
        "runDir": str(run_dir.resolve()),
        "overview": "ALL_FAMILIES.png",
        "detailOverview": "ALL_FAMILIES_DETAIL.png",
        "qualificationGates": {
            "globalRecoveryMin": 0.45,
            "edgeRecoveryMin": 0.60,
            "gradientRecoveryMin": 0.35,
            "latticeCellExcessMax": 0.15,
        },
        "files": files,
    }
    safe._atomic_json(complete_marker, index)  # noqa: SLF001
    safe._atomic_json(latest_dir / "probe_index.json", index)  # noqa: SLF001
    print(
        f"[v16.0-stage2-live] visual probe step {step}: "
        f"{latest_dir / 'ALL_FAMILIES.png'}",
        flush=True,
    )


safe.SafeFourFamilyDiagnostic._safe_values = _safe_values
safe.SafeFourFamilyDiagnostic._gpu_telemetry = _gpu_telemetry
safe.SafeFourFamilyDiagnostic._gpu_text = staticmethod(_gpu_text)
safe.SafeFourFamilyDiagnostic._write_resume_checkpoint = _write_resume_checkpoint
safe.torch.optim.Adam = _memory_lean_adam
safe.torch.cuda.synchronize = _synchronize_and_release
live._write_live_probes = _write_live_probes_low_vram


def main(argv: list[str] | None = None) -> int:
    return resume.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
