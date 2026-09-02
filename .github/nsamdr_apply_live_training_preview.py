from pathlib import Path

GUI_PATH = Path("tools/nsamdr/gui/nsamdr_v9_workflow_gui.py")
TEST_PATH = Path("tools/nsamdr/tests/test_live_training_preview_gui_contract.py")

gui = GUI_PATH.read_text(encoding="utf-8")

footer_old = '''        ttk.Button(footer, text="Stop current process", command=self.stop).pack(side="left", padx=5)\n        ttk.Label(footer, text="Preview:").pack(side="left", padx=(6, 2))\n'''
footer_new = '''        ttk.Button(footer, text="Stop current process", command=self.stop).pack(side="left", padx=5)\n        ttk.Button(footer, text="Live training preview", command=self.show_live_training_preview).pack(side="left", padx=(0, 5))\n        ttk.Label(footer, text="Preview:").pack(side="left", padx=(6, 2))\n'''
if gui.count(footer_old) != 1:
    raise SystemExit(f"expected one footer anchor, found {gui.count(footer_old)}")
gui = gui.replace(footer_old, footer_new)

init_old = '''        self.preview_log_path: Path | None = None\n        self.preview_experiment: str | None = None\n\n        root.title(APP_TITLE)\n'''
init_new = '''        self.preview_log_path: Path | None = None\n        self.preview_experiment: str | None = None\n\n        # Structural training already overwrites same-renderer Stage-A evidence\n        # after every structural epoch. Keep a separate lightweight viewer for\n        # those files so the user can judge live geometry and stop early without\n        # waiting for a qualified final checkpoint or launching another renderer.\n        self.live_preview_window: tk.Toplevel | None = None\n        self.live_preview_canvas: tk.Canvas | None = None\n        self.live_preview_photo: tk.PhotoImage | None = None\n        self.live_preview_case_var = tk.StringVar(value="")\n        self.live_preview_case_combo: ttk.Combobox | None = None\n        self.live_preview_status_var = tk.StringVar(value="Waiting for live rendered evidence")\n        self.live_preview_fit_var = tk.BooleanVar(value=True)\n        self.live_preview_signature: tuple | None = None\n        self.live_preview_run_started_wall = 0.0\n        self.live_preview_dismissed_for_run = False\n\n        root.title(APP_TITLE)\n'''
if gui.count(init_old) != 1:
    raise SystemExit(f"expected one preview init anchor, found {gui.count(init_old)}")
gui = gui.replace(init_old, init_new)

schedule_old = '''        self._poll()\n        self._poll_preview()\n        self.root.after(1000, self._preview_refresh_tick)\n'''
schedule_new = '''        self._poll()\n        self._poll_preview()\n        self.root.after(1000, self._preview_refresh_tick)\n        self.root.after(500, self._poll_live_training_preview)\n'''
if gui.count(schedule_old) != 1:
    raise SystemExit(f"expected one poll schedule anchor, found {gui.count(schedule_old)}")
gui = gui.replace(schedule_old, schedule_new)

started_old = '''                if kind == "started":\n                    self.output.insert("end", f"[GUI] Process started: PID {item[2]}\\n")\n'''
started_new = '''                if kind == "started":\n                    if stage_id in {"quick", "train"}:\n                        self.live_preview_run_started_wall = time.time()\n                        self.live_preview_dismissed_for_run = False\n                        self.live_preview_signature = None\n                        if self.live_preview_window is not None:\n                            self.live_preview_status_var.set(\n                                "Training started — waiting for this run's first rendered structural epoch"\n                            )\n                    self.output.insert("end", f"[GUI] Process started: PID {item[2]}\\n")\n'''
if gui.count(started_old) != 1:
    raise SystemExit(f"expected one process-start anchor, found {gui.count(started_old)}")
gui = gui.replace(started_old, started_new)

methods_anchor = '''    # Purpose: Implement training artifact paths for App.\n'''
methods = r'''    # Purpose: Find the newest structural same-renderer evidence for App.
    # Called by: _poll_live_training_preview, _refresh_live_training_preview
    # Calls: _experiment_ids, _experiments_root
    def _latest_live_training_evidence(
        self,
    ) -> tuple[str, Path, Path, int, dict] | None:
        candidates: list[tuple[int, str, Path, Path]] = []
        for experiment in self._experiment_ids():
            evidence_root = (
                self._experiments_root()
                / experiment
                / "previews/oracle_renderer_preflight"
            )
            progress_path = evidence_root / "live_progress.json"
            stage_dir = evidence_root / "staged_evidence"
            if not progress_path.is_file() or not stage_dir.is_dir():
                continue
            try:
                if not any(stage_dir.glob("*_stages.png")):
                    continue
                modified_ns = progress_path.stat().st_mtime_ns
            except OSError:
                continue
            candidates.append((modified_ns, experiment, progress_path, stage_dir))
        if not candidates:
            return None
        modified_ns, experiment, progress_path, stage_dir = max(candidates)
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return experiment, progress_path, stage_dir, modified_ns, payload

    # Purpose: Close the persistent live structural preview for App.
    # Called by: show_live_training_preview window protocol
    # Calls: No same-class helper methods.
    def _close_live_training_preview(self) -> None:
        if self.process is not None and self.active_stage in {"quick", "train"}:
            self.live_preview_dismissed_for_run = True
        window = self.live_preview_window
        self.live_preview_window = None
        self.live_preview_canvas = None
        self.live_preview_photo = None
        self.live_preview_case_combo = None
        self.live_preview_signature = None
        if window is not None:
            try:
                window.destroy()
            except tk.TclError:
                pass

    # Purpose: Open or raise the persistent live structural preview for App.
    # Called by: footer button, _poll_live_training_preview
    # Calls: _refresh_live_training_preview
    def show_live_training_preview(self) -> None:
        self.live_preview_dismissed_for_run = False
        if self.live_preview_window is not None:
            try:
                self.live_preview_window.deiconify()
                self.live_preview_window.lift()
                self._refresh_live_training_preview(force=True)
                return
            except tk.TclError:
                self.live_preview_window = None

        window = tk.Toplevel(self.root)
        self.live_preview_window = window
        window.title("NSAMDR Live Training Render")
        window.geometry("1420x760")
        window.minsize(900, 520)
        window.protocol("WM_DELETE_WINDOW", self._close_live_training_preview)
        window.columnconfigure(0, weight=1)
        window.rowconfigure(2, weight=1)

        controls = ttk.Frame(window, padding=(8, 8, 8, 4))
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Case:").grid(row=0, column=0, sticky="w")
        combo = ttk.Combobox(
            controls,
            textvariable=self.live_preview_case_var,
            state="readonly",
            width=46,
        )
        self.live_preview_case_combo = combo
        combo.grid(row=0, column=1, sticky="ew", padx=(4, 8))
        combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._refresh_live_training_preview(force=True),
        )
        ttk.Checkbutton(
            controls,
            text="Fit",
            variable=self.live_preview_fit_var,
            command=lambda: self._refresh_live_training_preview(force=True),
        ).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(
            controls,
            text="Refresh",
            command=lambda: self._refresh_live_training_preview(force=True),
        ).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(controls, text="Stop training", command=self.stop).grid(row=0, column=4)

        status_frame = ttk.Frame(window, padding=(8, 0, 8, 4))
        status_frame.grid(row=1, column=0, sticky="ew")
        ttk.Label(
            status_frame,
            textvariable=self.live_preview_status_var,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            status_frame,
            text=(
                "Panel 1: damaged LR | Panel 2: GT geometry + deterministic renderer | "
                "Panel 3: LIVE model geometry + same renderer | Panel 4: |P3-P2| x8"
            ),
        ).pack(anchor="w")

        image_host = ttk.Frame(window, padding=(8, 0, 8, 8))
        image_host.grid(row=2, column=0, sticky="nsew")
        image_host.columnconfigure(0, weight=1)
        image_host.rowconfigure(0, weight=1)
        canvas = tk.Canvas(image_host, background="#1e1e1e", highlightthickness=0)
        self.live_preview_canvas = canvas
        y_scroll = ttk.Scrollbar(image_host, orient="vertical", command=canvas.yview)
        x_scroll = ttk.Scrollbar(image_host, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        window.update_idletasks()
        self._refresh_live_training_preview(force=True)

    # Purpose: Refresh the same live window when structural epoch evidence changes.
    # Called by: show_live_training_preview, _poll_live_training_preview, UI controls
    # Calls: _latest_live_training_evidence
    def _refresh_live_training_preview(self, *, force: bool = False) -> None:
        window = self.live_preview_window
        canvas = self.live_preview_canvas
        combo = self.live_preview_case_combo
        if window is None or canvas is None or combo is None:
            return
        try:
            if not window.winfo_exists():
                self.live_preview_window = None
                return
        except tk.TclError:
            self.live_preview_window = None
            return

        live = self._latest_live_training_evidence()
        if live is None:
            self.live_preview_status_var.set(
                "Waiting for live rendered evidence — the window will update after the first structural epoch"
            )
            return
        experiment, _progress_path, stage_dir, progress_mtime_ns, payload = live

        # Do not label stale evidence from an earlier run as current. A manually
        # opened window may remain visible while the new run is still in epoch 1;
        # it switches automatically as soon as live_progress.json is overwritten.
        if (
            self.process is not None
            and self.active_stage in {"quick", "train"}
            and self.live_preview_run_started_wall > 0.0
            and progress_mtime_ns < int((self.live_preview_run_started_wall - 0.5) * 1.0e9)
        ):
            self.live_preview_status_var.set(
                "Waiting for this run's first rendered structural epoch — no stale result is being shown"
            )
            return

        cases = sorted(path.name for path in stage_dir.glob("*_stages.png") if path.is_file())
        if not cases:
            self.live_preview_status_var.set("Live progress exists, but no rendered Stage-A case sheet is ready yet")
            return
        combo.configure(values=cases)
        selected = self.live_preview_case_var.get().strip()
        if selected not in cases:
            selected = cases[0]
            self.live_preview_case_var.set(selected)
        image_path = stage_dir / selected
        try:
            image_mtime_ns = image_path.stat().st_mtime_ns
        except OSError:
            return

        fit = bool(self.live_preview_fit_var.get())
        signature = (experiment, progress_mtime_ns, selected, image_mtime_ns, fit)
        if force or signature != self.live_preview_signature:
            try:
                raw = tk.PhotoImage(file=str(image_path))
                if fit:
                    canvas.update_idletasks()
                    available_w = max(640, canvas.winfo_width() - 24)
                    available_h = max(320, canvas.winfo_height() - 24)
                    divisor = max(
                        1,
                        int(math.ceil(raw.width() / float(available_w))),
                        int(math.ceil(raw.height() / float(available_h))),
                    )
                    photo = raw.subsample(divisor, divisor) if divisor > 1 else raw
                else:
                    photo = raw
            except (tk.TclError, OSError):
                # Training may be between truncate/write/rename operations. Keep
                # the previous valid frame and retry on the next poll.
                return
            self.live_preview_photo = photo
            canvas.delete("all")
            canvas.create_image(8, 8, image=photo, anchor="nw")
            canvas.configure(
                scrollregion=(0, 0, max(photo.width() + 16, canvas.winfo_width()), max(photo.height() + 16, canvas.winfo_height()))
            )
            self.live_preview_signature = signature

        def metric(name: str, digits: int) -> str | None:
            value = payload.get(name)
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return None

        epoch = payload.get("epoch", "?")
        phase = str(payload.get("phase") or "unknown")
        substage = str(payload.get("b1bSubstage") or "").strip()
        parts = [f"{experiment} | epoch {epoch} | {phase}"]
        if substage and substage.lower() != "none":
            parts.append(substage)
        jitter = metric("lineJitterPixelsMean", 3)
        roughness = metric("curveRoughnessPixelsMean", 3)
        render_mae = metric("renderBandMaeMean", 4)
        if jitter is not None:
            parts.append(f"line jitter {jitter}px")
        if roughness is not None:
            parts.append(f"curve roughness {roughness}px")
        if render_mae is not None:
            parts.append(f"render-band MAE {render_mae}")
        parts.append(
            "LIVE — refreshes after each structural epoch"
            if self.process is not None and self.active_stage in {"quick", "train"}
            else "last structural training evidence"
        )
        self.live_preview_status_var.set(" | ".join(parts))

    # Purpose: Poll for newly-overwritten Stage-A evidence without rerunning inference.
    # Called by: __init__ timer
    # Calls: _latest_live_training_evidence, _refresh_live_training_preview, show_live_training_preview
    def _poll_live_training_preview(self) -> None:
        try:
            if self.live_preview_window is not None:
                self._refresh_live_training_preview()
            elif (
                self.process is not None
                and self.active_stage in {"quick", "train"}
                and not self.live_preview_dismissed_for_run
            ):
                live = self._latest_live_training_evidence()
                if live is not None:
                    progress_mtime_ns = live[3]
                    if progress_mtime_ns >= int((self.live_preview_run_started_wall - 0.5) * 1.0e9):
                        self.show_live_training_preview()
        finally:
            try:
                self.root.after(750, self._poll_live_training_preview)
            except tk.TclError:
                pass

'''
if gui.count(methods_anchor) != 1:
    raise SystemExit(f"expected one training-artifact anchor, found {gui.count(methods_anchor)}")
gui = gui.replace(methods_anchor, methods + methods_anchor)

GUI_PATH.write_text(gui, encoding="utf-8")

TEST_PATH.write_text(
    '''from pathlib import Path\n\n\nGUI_PATH = Path("tools/nsamdr/gui/nsamdr_v9_workflow_gui.py")\n\n\ndef _source() -> str:\n    return GUI_PATH.read_text(encoding="utf-8")\n\n\ndef test_live_training_preview_consumes_existing_same_renderer_evidence() -> None:\n    source = _source()\n    assert 'def _latest_live_training_evidence(' in source\n    assert '/ "previews/oracle_renderer_preflight"' in source\n    assert 'progress_path = evidence_root / "live_progress.json"' in source\n    assert 'stage_dir = evidence_root / "staged_evidence"' in source\n    assert 'tk.PhotoImage(file=str(image_path))' in source\n\n\ndef test_live_training_preview_updates_persistent_window_without_second_renderer() -> None:\n    source = _source()\n    refresh = source.split('def _refresh_live_training_preview(', 1)[1].split(\n        'def _poll_live_training_preview(', 1\n    )[0]\n    assert 'self.live_preview_photo = photo' in refresh\n    assert 'self.live_preview_signature = signature' in refresh\n    assert 'subprocess.' not in refresh\n    assert 'preview_available_model' not in refresh\n    poll = source.split('def _poll_live_training_preview(', 1)[1].split(\n        '# Purpose: Implement training artifact paths', 1\n    )[0]\n    assert 'self.show_live_training_preview()' in poll\n    assert 'self.root.after(750, self._poll_live_training_preview)' in poll\n\n\ndef test_live_training_preview_is_actionable_during_training() -> None:\n    source = _source()\n    assert 'text="Live training preview"' in source\n    assert 'text="Stop training", command=self.stop' in source\n    assert 'line jitter {jitter}px' in source\n    assert 'curve roughness {roughness}px' in source\n    assert 'render-band MAE {render_mae}' in source\n    assert 'no stale result is being shown' in source\n''',
    encoding="utf-8",
)
