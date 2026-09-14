#!/usr/bin/env python3
"""Safe V16 Stage 2 runtime with live four-family visual probes.

This wrapper leaves the safe Stage 2 optimizer/model/data contract unchanged and adds
visual held-out probes after each validation point.  Probe generation is deliberately
best-effort: an image-writing failure is reported but never aborts training.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
NEURAL_ROOT = HERE.parent
if str(NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURAL_ROOT))

from v14 import safe_fourfamily_multiregion_diagnostic as safe

base = safe.base
_original_evaluate_records = safe.SafeFourFamilyDiagnostic._evaluate_records


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return text.strip("._-") or "family"


def _rgb_u8(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().float().clamp(0.0, 1.0)[0]
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    tensor = tensor[:3]
    return np.round(tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def _normal_rgb_u8(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach().float()[0]
    x = tensor[0].clamp(-1.0, 1.0)
    y = tensor[1].clamp(-1.0, 1.0)
    z = torch.sqrt(torch.clamp(1.0 - x * x - y * y, min=0.0, max=1.0))
    rgb = torch.stack((x, y, z), dim=0) * 0.5 + 0.5
    return np.round(rgb.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def _panel(rgb: np.ndarray, label: str) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    panel = cv2.copyMakeBorder(
        bgr,
        42,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    cv2.putText(
        panel,
        label,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def _banner(width: int, text: str, height: int = 52) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        image,
        text,
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _metric_text(family: dict[str, object] | None) -> str:
    if not family:
        return ""

    def percent(name: str) -> str:
        value = family.get(name)
        if value is None:
            return "?"
        return f"{float(value) * 100:+.1f}%"

    return (
        f" global {percent('medianGlobalRecovery')}"
        f" edge {percent('medianEdgeRecovery')}"
        f" grad {percent('medianGradientRecovery')}"
    )


def _atomic_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"OpenCV could not encode live probe {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(encoded.tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _probe_context(diagnostic: safe.SafeFourFamilyDiagnostic) -> tuple[Path, int] | None:
    pointer = diagnostic._resume_pointer()  # noqa: SLF001 - same Stage 2 runtime contract
    if not pointer.is_file():
        return None
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
        run_dir = Path(str(payload.get("runDir") or ""))
        step = int(payload.get("step") or 0)
    except (OSError, ValueError, TypeError):
        return None
    if step < 1 or not run_dir.is_dir():
        return None
    return run_dir, step


def _family_visual(
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    family_name: str,
    metrics: dict[str, object] | None,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []

    albedo_panels = [
        _panel(_rgb_u8(batch["target_albedo"]), "ALBEDO / A AUTHORED"),
        _panel(_rgb_u8(outputs["baseline_albedo"]), "ALBEDO / B BASELINE"),
        _panel(_rgb_u8(outputs["candidate_albedo"]), "ALBEDO / C CURRENT V16"),
    ]
    albedo_row = np.concatenate(albedo_panels, axis=1)
    rows.append(albedo_row)

    normal_panels = [
        _panel(_normal_rgb_u8(batch["target_normal"]), "NORMAL / A AUTHORED"),
        _panel(_normal_rgb_u8(outputs["baseline_normal"]), "NORMAL / B BASELINE"),
        _panel(_normal_rgb_u8(outputs["candidate_normal"]), "NORMAL / C CURRENT V16"),
    ]
    rows.append(np.concatenate(normal_panels, axis=1))

    material_panels = [
        _panel(_rgb_u8(batch["target_material"]), "MATERIAL / A AUTHORED"),
        _panel(_rgb_u8(outputs["baseline_material"]), "MATERIAL / B BASELINE"),
        _panel(_rgb_u8(outputs["candidate_material"]), "MATERIAL / C CURRENT V16"),
    ]
    rows.append(np.concatenate(material_panels, axis=1))

    content = np.concatenate(rows, axis=0)
    title = f"STEP {step} / {family_name}{_metric_text(metrics)}"
    detailed = np.concatenate((_banner(content.shape[1], title), content), axis=0)

    overview_title = _banner(albedo_row.shape[1], title)
    overview = np.concatenate((overview_title, albedo_row), axis=0)
    return detailed, overview


def _write_live_probes(
    diagnostic: safe.SafeFourFamilyDiagnostic,
    model: Any,
    records: list[dict[str, Any]],
    config: Any,
    report: dict[str, object],
) -> None:
    context = _probe_context(diagnostic)
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

            metrics = per_family.get(family_id)
            detailed, overview = _family_visual(
                batch,
                outputs,
                family_name,
                metrics if isinstance(metrics, dict) else None,
                step,
            )
            filename = f"{_slug(family_name)}.png"
            step_path = step_dir / filename
            latest_path = latest_dir / filename
            _atomic_png(step_path, detailed)
            _atomic_png(latest_path, detailed)
            overviews.append(overview)
            files.append({"familyId": family_id, "family": family_name, "file": filename})

            if diagnostic.device.type == "cuda":
                torch.cuda.synchronize(diagnostic.device)
                torch.cuda.empty_cache()
                time.sleep(0.25)

    if overviews:
        contact = np.concatenate(overviews, axis=0)
        _atomic_png(step_dir / "ALL_FAMILIES.png", contact)
        _atomic_png(latest_dir / "ALL_FAMILIES.png", contact)

    index = {
        "schema": "NSAMDR_V16_STAGE2_LIVE_PROBE_V1",
        "step": step,
        "runDir": str(run_dir.resolve()),
        "overview": "ALL_FAMILIES.png",
        "files": files,
    }
    safe._atomic_json(complete_marker, index)  # noqa: SLF001
    safe._atomic_json(latest_dir / "probe_index.json", index)  # noqa: SLF001
    print(
        f"[v16.0-stage2-live] visual probe step {step}: "
        f"{latest_dir / 'ALL_FAMILIES.png'}",
        flush=True,
    )


def _evaluate_records_with_live_probe(
    self: safe.SafeFourFamilyDiagnostic,
    model: Any,
    records: list[dict[str, Any]],
    config: Any,
) -> dict[str, object]:
    report = _original_evaluate_records(self, model, records, config)
    is_validation = bool(records) and all(
        str(record.get("split") or "") == "validation" for record in records
    )
    if is_validation:
        try:
            _write_live_probes(self, model, records, config, report)
        except Exception as exc:  # visual telemetry must never invalidate training
            print(
                f"[v16.0-stage2-live] WARNING: live probe generation failed: {exc}",
                flush=True,
            )
    return report


safe.SafeFourFamilyDiagnostic._evaluate_records = _evaluate_records_with_live_probe


def main(argv: list[str] | None = None) -> int:
    return safe.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
