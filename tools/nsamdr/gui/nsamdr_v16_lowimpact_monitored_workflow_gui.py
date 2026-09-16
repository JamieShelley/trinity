#!/usr/bin/env python3
"""V16 low-impact GUI with monitored Stage 2 runtime and V16.1 loss lab."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import nsamdr_v16_lowimpact_workflow_gui as low

base = low.base
multifamily = low.multifamily
_original_dispatcher_argv = base.App._dispatcher_argv
_original_args = base.App._args
_original_build = base.App._build
_original_preview_refresh_tick = base.App._preview_refresh_tick
_original_select_multiregion = multifamily.v16.legacy._select_multiregion


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _dispatcher_argv(self: base.App, command: tuple[str, ...], args: list[str]) -> list[str]:
    if command == ("v14-mini-multiregion",):
        return [
            sys.executable,
            str(
                self.repo
                / "tools/nsamdr/neural/v14/safe_live_resume_monitored_v161_fourfamily_multiregion_diagnostic.py"
            ),
            "--repo-root",
            str(self.repo),
            *args,
        ]
    return _original_dispatcher_argv(self, command, args)


def _select_multiregion(self: base.App, stage: base.Stage, status: str) -> None:
    _original_select_multiregion(self, stage, status)

    # V16.1A is deliberately shorter than the completed V16.0 four-family run.
    # The purpose is to test optimisation balance before another long experiment.
    maximum = self.vars.get("mini_max_steps")
    if maximum is not None:
        maximum.set("2048")

    self._label_row(
        "V16.1 loss laboratory",
        "Frozen V16.0 architecture/data/gates; only the candidate loss contract changes. Default controls are the V16.1A ablation.",
    )
    self._row(
        "Family/map loss normalization",
        "v161_loss_normalization",
        "baseline-relative-bounded",
        ("baseline-relative-bounded", "off"),
    )
    self._label_row(
        "Normalization references",
        "Detached baseline MAE references: albedo 0.040, normal 0.060, material 0.020; each multiplier is bounded to 0.50..2.50.",
    )
    self._row(
        "Frequency auxiliary",
        "v161_frequency_loss",
        "off",
        ("off", "wavelet", "focal-fourier"),
    )
    self._row(
        "Frequency loss weight",
        "v161_frequency_weight",
        "0.05",
        ("0.02", "0.05", "0.10"),
    )
    self._row(
        "Normal supervision",
        "v161_normal_loss",
        "xy-l1",
        ("xy-l1", "angular+l1"),
    )
    self._row(
        "Angular normal weight",
        "v161_angular_normal_weight",
        "0.25",
        ("0.10", "0.25", "0.50"),
    )
    self._check(
        "Record per-family gradient-conflict telemetry (diagnostic only)",
        "v161_gradient_telemetry",
        True,
    )
    self._label_row(
        "V16.1A default",
        "Normalization ON; frequency OFF; existing XY-L1 normal objective; gradient telemetry ON; 2048-step maximum.",
    )
    self._label_row(
        "V16.1B/C",
        "B: select wavelet frequency. C: wavelet + angular+l1 normal supervision. Keep other controls fixed for clean ablations.",
    )
    self._label_row(
        "Gradient surgery",
        "PCGrad/CAGrad are not applied yet. Telemetry measures whether cross-family gradient directions actually conflict before paying for extra backward passes.",
    )
    self._label_row(
        "Loss telemetry",
        "Each run writes v161_loss_telemetry.jsonl, v161_gradient_summary.json, and v161_loss_contract.json beside report.json.",
    )


def _args(self: base.App, stage_id: str) -> list[str]:
    values = _original_args(self, stage_id)
    if stage_id != "multiregion":
        return values

    values.extend(
        (
            "--loss-normalization",
            self._value("v161_loss_normalization", "baseline-relative-bounded"),
            "--frequency-loss",
            self._value("v161_frequency_loss", "off"),
            "--frequency-loss-weight",
            self._value("v161_frequency_weight", "0.05"),
            "--normal-loss",
            self._value("v161_normal_loss", "xy-l1"),
            "--angular-normal-weight",
            self._value("v161_angular_normal_weight", "0.25"),
            "--normalization-min-weight",
            "0.50",
            "--normalization-max-weight",
            "2.50",
            "--normalization-reference-albedo",
            "0.040",
            "--normalization-reference-normal",
            "0.060",
            "--normalization-reference-material",
            "0.020",
            "--gradient-conflict",
            (
                "telemetry-only"
                if bool(self.vars.get("v161_gradient_telemetry") and self.vars["v161_gradient_telemetry"].get())
                else "off"
            ),
        )
    )
    return values


def _stage2_resume_choices(self: base.App) -> list[str]:
    root = self.repo / "artifacts/nsamdr/diagnostics/v16_mini"
    mapping: dict[str, Path] = {}
    if not root.is_dir():
        self._stage2_resume_paths = mapping
        return []
    pointer = _read_json(root / "stage2_resume_pointer.json") or {}
    pointer_run = str(pointer.get("runDir") or "")
    labels: list[str] = []
    for run_dir in sorted((path for path in root.glob("multiregion_*") if path.is_dir()), key=lambda path: path.name, reverse=True):
        checkpoint = run_dir / "resume_checkpoint.pt"
        if not checkpoint.is_file():
            continue
        sidecar = _read_json(run_dir / "resume_checkpoint.json") or {}
        heartbeat = _read_json(run_dir / "stage2_heartbeat.json") or {}
        contract = _read_json(run_dir / "v161_loss_contract.json") or {}
        step = int(sidecar.get("step") or 0)
        maximum = int(sidecar.get("maximumSteps") or 0)
        exact = bool(step and maximum)
        if not exact and pointer_run:
            try:
                if Path(pointer_run).resolve() == run_dir.resolve():
                    step = int(pointer.get("step") or 0)
                    maximum = int(pointer.get("maximumSteps") or 0)
                    exact = bool(step and maximum)
            except OSError:
                pass
        if not step:
            step = int(heartbeat.get("step") or 0)
            maximum = int(heartbeat.get("maximumSteps") or 0)
        status = str(heartbeat.get("status") or "saved")
        short_name = run_dir.name.removeprefix("multiregion_")
        if step > 0 and maximum > 0:
            progress = f"{'' if exact else '~'}{step}/{maximum}"
        else:
            progress = "checkpoint"
        revision = "V16.1" if contract.get("schema") == "NSAMDR_V16_1_LOSS_LAB_V1" else "V16.0"
        label = f"{short_name} | {revision} | {progress} | {status}"
        mapping[label] = checkpoint.resolve()
        labels.append(label)
    self._stage2_resume_paths = mapping
    return labels


def _latest_stage2_paths(self: base.App) -> tuple[Path | None, dict[str, object], dict[str, object]]:
    root = self.repo / "artifacts/nsamdr/diagnostics/v16_mini"
    pointer = _read_json(root / "stage2_resume_pointer.json") or {}
    run_dir: Path | None = None
    if pointer.get("runDir"):
        candidate = Path(str(pointer["runDir"]))
        if candidate.is_dir():
            run_dir = candidate
    if run_dir is None and root.is_dir():
        candidates = [path for path in root.glob("multiregion_*") if path.is_dir()]
        if candidates:
            run_dir = max(candidates, key=lambda path: path.stat().st_mtime)
    heartbeat = _read_json(run_dir / "stage2_heartbeat.json") if run_dir else None
    return run_dir, pointer, heartbeat or {}


def _fmt_eta(seconds: object) -> str:
    try:
        minutes = max(0.0, float(seconds)) / 60.0
    except (TypeError, ValueError):
        return "?"
    return f"{minutes / 60.0:.1f}h" if minutes >= 60.0 else f"{minutes:.0f}m"


def _refresh_stage2_status(self: base.App) -> None:
    if not hasattr(self, "stage2_live_status_var"):
        return
    run_dir, pointer, heartbeat = _latest_stage2_paths(self)
    if run_dir is None or not heartbeat:
        self.stage2_live_status_var.set("Stage 2: no durable run")
        return
    step = heartbeat.get("step", "?")
    maximum = heartbeat.get("maximumSteps", "?")
    status = str(heartbeat.get("status") or "unknown")
    eta = _fmt_eta(heartbeat.get("etaSeconds"))
    resume_step = pointer.get("step") if pointer else None
    gpu = heartbeat.get("gpu") if isinstance(heartbeat.get("gpu"), dict) else {}
    temp = gpu.get("temperatureC") if gpu else None
    util_avg = gpu.get("utilizationAveragePercent") if gpu else None
    util_now = gpu.get("utilizationPercent") if gpu else None
    vram = gpu.get("vramUsedMiB") if gpu else None
    vram_total = gpu.get("vramTotalMiB") if gpu else None
    contract = _read_json(run_dir / "v161_loss_contract.json") or {}
    revision = "V16.1" if contract.get("schema") == "NSAMDR_V16_1_LOSS_LAB_V1" else "V16.0"
    parts = [f"Stage 2 {revision} {step}/{maximum} {status}", f"ETA {eta}"]
    if temp is not None:
        parts.append(f"{float(temp):.0f}C")
    if util_avg is not None:
        parts.append(f"GPU {float(util_avg):.0f}%avg" + (f"/{float(util_now):.0f}%now" if util_now is not None else ""))
    elif util_now is not None:
        parts.append(f"GPU {float(util_now):.0f}%now")
    if vram is not None and vram_total is not None:
        parts.append(f"VRAM {float(vram)/1024.0:.1f}/{float(vram_total)/1024.0:.1f}G")
    if resume_step is not None:
        parts.append(f"resume@{resume_step}")
    self.stage2_live_status_var.set(" | ".join(parts))


def _run_stage2_audit(self: base.App) -> None:
    existing = getattr(self, "_stage2_audit_process", None)
    if existing is not None and existing.poll() is None:
        messagebox.showinfo("Stage 2 family audit", "The CPU family-difficulty audit is already running.")
        return

    script = self.repo / "tools/nsamdr/neural/audit_nsamdr_v16_stage2_family_difficulty.py"
    if not script.is_file():
        messagebox.showerror("Stage 2 family audit", f"Audit script is missing:\n{script}")
        return

    command = [
        sys.executable,
        "-u",
        str(script),
        "--repo-root",
        str(self.repo),
    ]
    kwargs: dict[str, object] = {"cwd": str(self.repo)}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    try:
        self._stage2_audit_process = subprocess.Popen(command, **kwargs)
    except OSError as exc:
        messagebox.showerror("Stage 2 family audit", f"Could not start audit:\n{exc}")
        return

    self.stage2_audit_status_var.set("Stage 2 audit: running (CPU)")


def _refresh_stage2_audit_status(self: base.App) -> None:
    if not hasattr(self, "stage2_audit_status_var"):
        return
    process = getattr(self, "_stage2_audit_process", None)
    if process is not None:
        code = process.poll()
        if code is None:
            self.stage2_audit_status_var.set("Stage 2 audit: running (CPU)")
            return
        self._stage2_audit_process = None
        self.stage2_audit_status_var.set(
            "Stage 2 audit: complete" if code == 0 else f"Stage 2 audit: failed ({code})"
        )
        return

    run_dir, _pointer, _heartbeat = _latest_stage2_paths(self)
    if run_dir is not None and (run_dir / "difficulty_audit.json").is_file():
        self.stage2_audit_status_var.set(f"Stage 2 audit: available ({run_dir.name.removeprefix('multiregion_')})")
    else:
        self.stage2_audit_status_var.set("Stage 2 audit: ready")


def _build(self: base.App) -> None:
    _original_build(self)
    self.stage2_live_status_var = tk.StringVar(value="Stage 2: checking...")
    ttk.Label(self.footer, textvariable=self.stage2_live_status_var, font=("Segoe UI", 9, "bold")).pack(side="right", padx=(6, 8))

    self.stage2_audit_status_var = tk.StringVar(value="Stage 2 audit: ready")
    ttk.Button(
        self.footer,
        text="Run Stage 2 Audit",
        command=lambda: _run_stage2_audit(self),
    ).pack(side="left", padx=(10, 4))
    ttk.Label(self.footer, textvariable=self.stage2_audit_status_var).pack(side="left", padx=(0, 8))

    _refresh_stage2_status(self)
    _refresh_stage2_audit_status(self)


def _preview_refresh_tick(self: base.App) -> None:
    try:
        _refresh_stage2_status(self)
        _refresh_stage2_audit_status(self)
    except (OSError, ValueError, TypeError, tk.TclError):
        pass
    _original_preview_refresh_tick(self)


multifamily._stage2_resume_choices = _stage2_resume_choices
base.App._dispatcher_argv = _dispatcher_argv
base.App._args = _args
base.App._build = _build
base.App._preview_refresh_tick = _preview_refresh_tick
multifamily.v16.legacy._select_multiregion = _select_multiregion


if __name__ == "__main__":
    raise SystemExit(base.main())
