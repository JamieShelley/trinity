#!/usr/bin/env python3
"""V16 GUI entry point with safe/live four-family Stage 2 controls."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import nsamdr_v16_workflow_gui as v16


base = v16.base
_original_dispatcher_argv = base.App._dispatcher_argv
_original_select_multiregion = v16._select_multiregion
_original_args = base.App._args
_original_build = base.App._build
_original_preview_refresh_tick = base.App._preview_refresh_tick


def _dispatcher_argv(
    self: base.App,
    command: tuple[str, ...],
    args: list[str],
) -> list[str]:
    if command == ("v14-mini-multiregion",):
        return [
            sys.executable,
            str(
                self.repo
                / "tools/nsamdr/neural/v14/safe_live_resume_fourfamily_multiregion_diagnostic.py"
            ),
            "--repo-root",
            str(self.repo),
            *args,
        ]
    return _original_dispatcher_argv(self, command, args)


def _select_multiregion(self: base.App, stage: base.Stage, status: str) -> None:
    _original_select_multiregion(self, stage, status)
    self._label_row(
        "Authored families",
        "Stage 2 requires at least four distinct native Caldari battleship texture families.",
    )
    self._label_row(
        "Family split",
        "Each authored family gets its own hard pixel-disjoint train/held-out spatial domains.",
    )
    self._label_row(
        "Coverage target",
        "Controlled retest uses 8 training regions: 2 regions from each of 4 authored families.",
    )
    self._label_row(
        "Held-out target",
        "Four held-out regions use 1 unseen region from each authored family.",
    )
    self._label_row(
        "Training budget",
        "5120 optimizer steps preserve about 640 visits per training region.",
    )
    self._label_row(
        "Per-family telemetry",
        "Every validation point reports recovery separately for each authored family.",
    )
    self._label_row(
        "Live visual probes",
        "Every validation point writes A/B/C held-out albedo, normal, and material comparisons under probes/latest.",
    )
    self._label_row(
        "Crash recovery",
        "Atomic model+optimizer state is saved every 128 steps and immediately before validation.",
    )
    self._label_row(
        "GUI resume",
        "Use the Stage 2 resume selector in the footer to choose a retained run and resume it directly.",
    )
    self._label_row(
        "Durable output",
        "Progress JSONL and an atomic heartbeat are fsynced during training, so a hard reboot leaves usable evidence.",
    )
    self._label_row(
        "Safe GPU profile",
        "Adds short synchronized cooldowns, reports temperature/power/VRAM, and pauses at 80 C until below 75 C.",
    )
    self._check(
        "Use safer paced GPU runtime (recommended after PC crash)",
        "stage2_safe_mode",
        True,
    )
    self._check(
        "Resume latest compatible interrupted Stage 2 checkpoint automatically",
        "stage2_resume",
        True,
    )


def _args(self: base.App, stage_id: str) -> list[str]:
    values = _original_args(self, stage_id)
    if stage_id != "multiregion":
        return values
    safe_mode = self.vars.get("stage2_safe_mode")
    if safe_mode is not None and bool(safe_mode.get()):
        values.append("--safe-mode")
    resume = self.vars.get("stage2_resume")
    if resume is not None and bool(resume.get()):
        values.append("--resume-stage2")
    forced_checkpoint = getattr(self, "_stage2_forced_checkpoint", None)
    if forced_checkpoint:
        values.extend(("--resume-checkpoint", str(forced_checkpoint)))
    return values


def _stage2_resume_root(self: base.App) -> Path:
    return self.repo / "artifacts/nsamdr/diagnostics/v16_mini"


def _stage2_resume_choices(self: base.App) -> list[str]:
    root = _stage2_resume_root(self)
    mapping: dict[str, Path] = {}
    if not root.is_dir():
        self._stage2_resume_paths = mapping
        return []

    pointer_run = ""
    pointer_step = 0
    pointer_max = 0
    pointer_path = root / "stage2_resume_pointer.json"
    if pointer_path.is_file():
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            pointer_run = str(pointer.get("runDir") or "")
            pointer_step = int(pointer.get("step") or 0)
            pointer_max = int(pointer.get("maximumSteps") or 0)
        except (OSError, ValueError, TypeError):
            pass

    labels: list[str] = []
    run_dirs = sorted(
        (path for path in root.glob("multiregion_*") if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for run_dir in run_dirs:
        checkpoint = run_dir / "resume_checkpoint.pt"
        if not checkpoint.is_file():
            continue

        step = 0
        maximum = 0
        status = "saved"
        try:
            if pointer_run and Path(pointer_run).resolve() == run_dir.resolve():
                step = pointer_step
                maximum = pointer_max
                status = "READY"
            else:
                heartbeat_path = run_dir / "stage2_heartbeat.json"
                if heartbeat_path.is_file():
                    heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
                    step = int(heartbeat.get("step") or 0)
                    maximum = int(heartbeat.get("maximumSteps") or 0)
                    status = str(heartbeat.get("status") or "saved")
        except (OSError, ValueError, TypeError):
            pass

        short_name = run_dir.name.removeprefix("multiregion_")
        progress = f"{step}/{maximum}" if step > 0 and maximum > 0 else "checkpoint"
        label = f"{short_name} | {progress} | {status}"
        mapping[label] = checkpoint.resolve()
        labels.append(label)

    self._stage2_resume_paths = mapping
    return labels


def _refresh_stage2_resume_selector(
    self: base.App,
    *,
    force_latest: bool = False,
) -> None:
    if not hasattr(self, "stage2_resume_combo") or not hasattr(
        self, "stage2_resume_target"
    ):
        return
    choices = _stage2_resume_choices(self)
    latest = choices[0] if choices else None
    previous_latest = getattr(self, "_stage2_resume_last_latest", None)
    new_latest = latest is not None and latest != previous_latest

    self.stage2_resume_combo.configure(values=choices)
    current = self.stage2_resume_target.get().strip()
    user_selected = bool(getattr(self, "_stage2_resume_user_selected", False))
    if (
        force_latest
        or not current
        or current not in choices
        or (new_latest and not user_selected)
    ):
        self.stage2_resume_target.set(latest or "")
    self._stage2_resume_last_latest = latest


def _stage2_resume_selection_changed(
    self: base.App,
    _event: tk.Event | None = None,
) -> None:
    self._stage2_resume_user_selected = True


def _resume_selected_stage2(self: base.App) -> None:
    if self.process is not None:
        messagebox.showwarning(base.APP_TITLE, "A stage is already running.")
        return

    _refresh_stage2_resume_selector(self)
    label = self.stage2_resume_target.get().strip()
    checkpoint = getattr(self, "_stage2_resume_paths", {}).get(label)
    if checkpoint is None or not checkpoint.is_file():
        messagebox.showerror(
            base.APP_TITLE,
            "No retained Stage 2 resume checkpoint is selected.",
        )
        return

    self.tree.selection_set("multiregion")
    self._selected()
    safe_mode = self.vars.get("stage2_safe_mode")
    if safe_mode is not None:
        safe_mode.set(True)
    resume = self.vars.get("stage2_resume")
    if resume is not None:
        resume.set(True)

    self._stage2_forced_checkpoint = checkpoint
    self._update_command()
    try:
        if self._validate_before_launch("multiregion"):
            self._run("multiregion")
    finally:
        self._stage2_forced_checkpoint = None


def _build(self: base.App) -> None:
    self._stage2_resume_paths: dict[str, Path] = {}
    self._stage2_resume_user_selected = False
    self._stage2_resume_last_latest: str | None = None
    self._stage2_forced_checkpoint: Path | None = None
    _original_build(self)

    ttk.Label(self.footer, text="Stage 2 resume:").pack(
        side="left", padx=(8, 2)
    )
    self.stage2_resume_target = tk.StringVar(value="")
    self.stage2_resume_combo = ttk.Combobox(
        self.footer,
        textvariable=self.stage2_resume_target,
        width=34,
        state="readonly",
        postcommand=lambda: _refresh_stage2_resume_selector(self),
    )
    self.stage2_resume_combo.bind(
        "<<ComboboxSelected>>",
        lambda event: _stage2_resume_selection_changed(self, event),
    )
    self.stage2_resume_combo.pack(side="left", padx=(0, 3))
    ttk.Button(
        self.footer,
        text="Resume Stage 2",
        command=lambda: _resume_selected_stage2(self),
    ).pack(side="left")
    _refresh_stage2_resume_selector(self, force_latest=True)


def _preview_refresh_tick(self: base.App) -> None:
    try:
        _refresh_stage2_resume_selector(self)
    except (OSError, ValueError, TypeError, tk.TclError):
        pass
    _original_preview_refresh_tick(self)


base.App._dispatcher_argv = _dispatcher_argv
base.App._args = _args
base.App._build = _build
base.App._preview_refresh_tick = _preview_refresh_tick
v16.legacy._select_multiregion = _select_multiregion


if __name__ == "__main__":
    raise SystemExit(base.main())
