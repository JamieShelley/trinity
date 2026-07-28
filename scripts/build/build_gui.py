from __future__ import annotations

import os
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


class BuildGui(tk.Tk):
    POLL_MS = 50
    DEFAULT_RAVEN = "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2"

    def __init__(self) -> None:
        super().__init__()
        self.title("Carbon Trinity Build Launcher")
        self.geometry("1180x900")
        self.minsize(980, 700)

        self.script_dir = Path(__file__).resolve().parent
        self.repo_root = self.script_dir.parent.parent

        self.output_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.current_process: subprocess.Popen[str] | None = None
        self.stop_requested = False

        self.script_vars: dict[Path, tk.BooleanVar] = {}
        self.status_var = tk.StringVar(value="Idle")
        self.current_var = tk.StringVar(value="No script running")

        self.eve_cache_var = tk.StringVar(value="")
        self.eve_query_var = tk.StringVar(value=self.DEFAULT_RAVEN)
        self.local_model_var = tk.StringVar(value="")
        self.local_albedo_var = tk.StringVar(value="")

        self._build_ui()
        self._load_scripts()
        self.after(self.POLL_MS, self._poll_output_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)

        ttk.Label(
            header,
            text="Carbon Trinity batch launcher",
            font=("Segoe UI", 13, "bold"),
        ).pack(side=tk.LEFT)

        ttk.Label(header, textvariable=self.status_var).pack(side=tk.RIGHT)

        ttk.Label(
            outer,
            text=f"Repository: {self.repo_root}",
        ).pack(fill=tk.X, pady=(4, 8))

        self._build_nsamdr_panel(outer)

        body = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        scripts_panel = ttk.Frame(body, padding=(0, 0, 8, 0))
        console_panel = ttk.Frame(body)
        body.add(scripts_panel, weight=1)
        body.add(console_panel, weight=3)

        toolbar = ttk.Frame(scripts_panel)
        toolbar.pack(fill=tk.X, pady=(0, 6))

        ttk.Button(
            toolbar,
            text="Select NSAMDR build only",
            command=self._select_nsamdr_build_only,
        ).pack(side=tk.LEFT)

        ttk.Button(
            toolbar,
            text="Select all",
            command=lambda: self._set_all(True),
        ).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Button(
            toolbar,
            text="Clear",
            command=lambda: self._set_all(False),
        ).pack(side=tk.LEFT, padx=(6, 0))

        list_outer = ttk.Frame(scripts_panel)
        list_outer.pack(fill=tk.BOTH, expand=True)

        self.script_canvas = tk.Canvas(list_outer, highlightthickness=0)
        list_scroll = ttk.Scrollbar(
            list_outer,
            orient=tk.VERTICAL,
            command=self.script_canvas.yview,
        )
        self.script_list = ttk.Frame(self.script_canvas)

        canvas_window = self.script_canvas.create_window(
            (0, 0),
            window=self.script_list,
            anchor="nw",
        )

        self.script_list.bind(
            "<Configure>",
            lambda _event: self.script_canvas.configure(
                scrollregion=self.script_canvas.bbox("all")
            ),
        )
        self.script_canvas.bind(
            "<Configure>",
            lambda event: self.script_canvas.itemconfigure(
                canvas_window,
                width=event.width,
            ),
        )
        self.script_canvas.configure(yscrollcommand=list_scroll.set)

        self.script_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(
            console_panel,
            textvariable=self.current_var,
            font=("Segoe UI", 10, "bold"),
        ).pack(fill=tk.X, pady=(0, 6))

        console_outer = ttk.Frame(console_panel)
        console_outer.pack(fill=tk.BOTH, expand=True)

        self.console = tk.Text(
            console_outer,
            wrap=tk.NONE,
            undo=False,
            font=("Consolas", 10),
            background="#101010",
            foreground="#e6e6e6",
            insertbackground="#e6e6e6",
        )
        console_y = ttk.Scrollbar(
            console_outer,
            orient=tk.VERTICAL,
            command=self.console.yview,
        )
        console_x = ttk.Scrollbar(
            console_outer,
            orient=tk.HORIZONTAL,
            command=self.console.xview,
        )
        self.console.configure(
            yscrollcommand=console_y.set,
            xscrollcommand=console_x.set,
        )

        self.console.grid(row=0, column=0, sticky="nsew")
        console_y.grid(row=0, column=1, sticky="ns")
        console_x.grid(row=1, column=0, sticky="ew")
        console_outer.rowconfigure(0, weight=1)
        console_outer.columnconfigure(0, weight=1)

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X, pady=(10, 0))

        self.run_selected_button = ttk.Button(
            controls,
            text="Run selected",
            command=self._run_selected,
        )
        self.run_selected_button.pack(side=tk.LEFT)

        self.run_build_all_button = ttk.Button(
            controls,
            text="Full repository build (not needed for NSAMDR)",
            command=self._run_build_all,
        )
        self.run_build_all_button.pack(side=tk.LEFT, padx=(6, 0))

        self.stop_button = ttk.Button(
            controls,
            text="Stop",
            command=self._request_stop,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Button(
            controls,
            text="Clear console",
            command=lambda: self.console.delete("1.0", tk.END),
        ).pack(side=tk.RIGHT)

    def _build_nsamdr_panel(self, parent: ttk.Frame) -> None:
        panel = ttk.LabelFrame(
            parent,
            text="NSAMDR real EVE asset test (Granny SDK-free)",
            padding=8,
        )
        panel.pack(fill=tk.X)
        panel.columnconfigure(1, weight=1)

        ttk.Label(panel, text="EVE installation").grid(row=0, column=0, sticky="w")
        ttk.Label(
            panel,
            text="Automatic: Windows Installed Apps, launcher metadata, previous verified cache, and common install folders",
        ).grid(row=0, column=1, columnspan=2, sticky="w", padx=(8, 0))

        ttk.Label(panel, text="Initial logical asset query").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(panel, textvariable=self.eve_query_var).grid(
            row=1, column=1, sticky="ew", padx=(8, 6), pady=(6, 0)
        )
        ttk.Button(panel, text="List matches", command=self._list_eve_assets).grid(
            row=1, column=2, sticky="ew", pady=(6, 0)
        )

        action_row = ttk.Frame(panel)
        action_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self.run_eve_asset_button = ttk.Button(
            action_row,
            text="TEST REAL EVE SHIP: auto-detect cache + open viewer",
            command=self._run_eve_asset,
        )
        self.run_eve_asset_button.pack(side=tk.LEFT)
        ttk.Label(
            action_row,
            text="The render window shows real ship names, groups LOD variants, and lets you switch cached ships.",
        ).pack(side=tk.LEFT, padx=(10, 0))

        separator = ttk.Separator(panel, orient=tk.HORIZONTAL)
        separator.grid(row=3, column=0, columnspan=3, sticky="ew", pady=9)

        ttk.Label(panel, text="Existing OBJ/GR2").grid(row=4, column=0, sticky="w")
        ttk.Entry(panel, textvariable=self.local_model_var).grid(
            row=4, column=1, sticky="ew", padx=(8, 6)
        )
        ttk.Button(panel, text="Browse", command=self._browse_local_model).grid(
            row=4, column=2, sticky="ew"
        )

        ttk.Label(panel, text="Optional albedo image").grid(
            row=5, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Entry(panel, textvariable=self.local_albedo_var).grid(
            row=5, column=1, sticky="ew", padx=(8, 6), pady=(6, 0)
        )
        browse_albedo = ttk.Frame(panel)
        browse_albedo.grid(row=5, column=2, sticky="ew", pady=(6, 0))
        ttk.Button(browse_albedo, text="Browse", command=self._browse_local_albedo).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(browse_albedo, text="Run", command=self._run_local_asset).pack(
            side=tk.LEFT, padx=(4, 0)
        )

    def _load_scripts(self) -> None:
        ignored = {"run_build_gui.bat"}

        scripts = [
            path
            for path in sorted(self.script_dir.glob("*.bat"))
            if not path.name.startswith("_") and path.name not in ignored
        ]

        for path in scripts:
            variable = tk.BooleanVar(value=False)
            self.script_vars[path] = variable
            ttk.Checkbutton(
                self.script_list,
                text=path.name,
                variable=variable,
            ).pack(fill=tk.X, anchor="w", padx=4, pady=2)

        self._set_all(False)

    def _set_all(self, selected: bool) -> None:
        for variable in self.script_vars.values():
            variable.set(selected)

    def _select_nsamdr_build_only(self) -> None:
        for path, variable in self.script_vars.items():
            variable.set(path.name.lower() == "build_nsamdr_obj_preview_dx11.bat")

    def _run_selected(self) -> None:
        selected = [
            path
            for path, variable in self.script_vars.items()
            if variable.get()
        ]
        if not selected:
            messagebox.showwarning(
                "Nothing selected",
                "Select at least one batch file.",
            )
            return
        self._start_run(selected)

    def _run_build_all(self) -> None:
        build_all = self.script_dir / "build_all.bat"
        if not build_all.exists():
            messagebox.showerror(
                "Missing script",
                f"Could not find:\n{build_all}",
            )
            return
        proceed = messagebox.askyesno(
            "Full repository build",
            "This is not the NSAMDR real-asset test.\n\n"
            "It enables ShaderCompiler and requires external re2c/lemon tools. "
            "Use the large TEST REAL EVE SHIP button above for NSAMDR.\n\n"
            "Run build_all.bat anyway?",
        )
        if proceed:
            self._start_run([build_all])

    def _run_eve_asset(self) -> None:
        script = self.script_dir / "run_nsamdr_eve_asset_dx11.bat"
        if not script.is_file():
            messagebox.showerror("Missing script", f"Could not find:\n{script}")
            return
        query = self.eve_query_var.get().strip() or self.DEFAULT_RAVEN
        self._start_commands([(script, ["", query])])

    def _list_eve_assets(self) -> None:
        script = self.script_dir / "list_nsamdr_eve_assets.bat"
        if not script.is_file():
            messagebox.showerror("Missing script", f"Could not find:\n{script}")
            return
        query = self.eve_query_var.get().strip() or "cb1"
        self._start_commands([(script, ["", query])])

    def _run_local_asset(self) -> None:
        script = self.script_dir / "run_nsamdr_obj_preview_dx11.bat"
        model = self.local_model_var.get().strip()
        if not script.is_file():
            messagebox.showerror("Missing script", f"Could not find:\n{script}")
            return
        if not model:
            messagebox.showwarning("Missing model", "Select an OBJ or GR2 model first.")
            return
        args = [model]
        albedo = self.local_albedo_var.get().strip()
        if albedo:
            args.append(albedo)
        self._start_commands([(script, args)])

    def _browse_eve_cache(self) -> None:
        selected = filedialog.askdirectory(
            title="Select EVE game files folder (or any folder inside it)",
            initialdir=self.eve_cache_var.get() or str(Path.home()),
        )
        if selected:
            self.eve_cache_var.set(selected)

    def _browse_local_model(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select converted OBJ or EVE GR2",
            filetypes=[
                ("3D model", "*.obj *.gr2"),
                ("Wavefront OBJ", "*.obj"),
                ("EVE Granny model", "*.gr2"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.local_model_var.set(selected)

    def _browse_local_albedo(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select albedo image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.local_albedo_var.set(selected)

    @staticmethod
    def _guess_eve_cache() -> str:
        candidates: list[Path] = []
        explicit = os.environ.get("EVE_SHARED_CACHE")
        if explicit:
            candidates.append(Path(explicit))
        local = os.environ.get("LOCALAPPDATA")
        program_data = os.environ.get("PROGRAMDATA")
        if local:
            candidates.append(Path(local) / "CCP" / "EVE" / "SharedCache")
        if program_data:
            candidates.append(Path(program_data) / "CCP" / "EVE" / "SharedCache")
        candidates.append(Path.home() / "Documents" / "EVE" / "SharedCache")
        for drive in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            candidates.extend([
                Path(f"{drive}:/EVE/SharedCache"),
                Path(f"{drive}:/EVE"),
                Path(f"{drive}:/CCP/EVE/SharedCache"),
                Path(f"{drive}:/CCP/EVE"),
                Path(f"{drive}:/Games/EVE/SharedCache"),
                Path(f"{drive}:/Games/EVE"),
                Path(f"{drive}:/SteamLibrary/steamapps/common/EVE Online/SharedCache"),
                Path(f"{drive}:/SteamLibrary/steamapps/common/EVE Online"),
            ])
        for candidate in candidates:
            variants = [candidate, *list(candidate.parents)[:5]]
            for root in variants:
                if (root / "ResFiles").is_dir() and (root / "tq" / "resfileindex.txt").is_file():
                    return str(root)
        return ""

    def _start_run(self, scripts: list[Path]) -> None:
        self._start_commands([(script, []) for script in scripts])

    def _start_commands(self, commands: list[tuple[Path, list[str]]]) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showwarning(
                "Already running",
                "A build sequence is already running.",
            )
            return

        self.stop_requested = False
        self._set_running_state(True)
        self.status_var.set(f"Running {len(commands)} command(s)")
        self.console.insert(
            tk.END,
            f"\n=== Starting {len(commands)} command(s) ===\n",
        )
        self.console.see(tk.END)

        self.worker = threading.Thread(
            target=self._run_commands_worker,
            args=(commands,),
            daemon=True,
        )
        self.worker.start()

    def _run_commands_worker(self, commands: list[tuple[Path, list[str]]]) -> None:
        overall_result = 0

        for index, (script, arguments) in enumerate(commands, start=1):
            if self.stop_requested:
                overall_result = 1
                break

            display = script.name
            if arguments:
                display += " " + subprocess.list2cmdline(arguments)
            self.output_queue.put(
                ("script", f"[{index}/{len(commands)}] {display}")
            )
            self.output_queue.put(
                (
                    "text",
                    f"\n\n===== [{index}/{len(commands)}] {display} =====\n",
                )
            )

            command = [
                os.environ.get("COMSPEC", "cmd.exe"),
                "/d",
                "/c",
                "call",
                str(script),
                *arguments,
            ]

            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                if os.name == "nt"
                else 0
            )

            try:
                process = subprocess.Popen(
                    command,
                    cwd=self.repo_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                )
            except OSError as exc:
                self.output_queue.put(
                    (
                        "text",
                        f"ERROR: Could not launch {script.name}: {exc}\n",
                    )
                )
                overall_result = 1
                break

            self.current_process = process
            assert process.stdout is not None

            for line in iter(process.stdout.readline, ""):
                self.output_queue.put(("text", line))
                if self.stop_requested and process.poll() is None:
                    self._terminate_process_tree(process)
                    break

            process.stdout.close()
            return_code = process.wait()
            self.current_process = None

            self.output_queue.put(
                (
                    "text",
                    f"===== {script.name} exited with code {return_code} =====\n",
                )
            )

            if return_code != 0:
                overall_result = return_code
                break

        self.output_queue.put(("done", overall_result))

    def _request_stop(self) -> None:
        if not self.worker or not self.worker.is_alive():
            return

        self.stop_requested = True
        self.status_var.set("Stopping")
        self.output_queue.put(("text", "\n=== Stop requested ===\n"))

        process = self.current_process
        if process is not None and process.poll() is None:
            threading.Thread(
                target=self._terminate_process_tree,
                args=(process,),
                daemon=True,
            ).start()

    @staticmethod
    def _terminate_process_tree(
        process: subprocess.Popen[str],
    ) -> None:
        if process.poll() is not None:
            return

        if os.name == "nt":
            subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            process.terminate()

    def _poll_output_queue(self) -> None:
        try:
            while True:
                event, payload = self.output_queue.get_nowait()

                if event == "text":
                    self.console.insert(tk.END, str(payload))
                    self.console.see(tk.END)

                elif event == "script":
                    self.current_var.set(str(payload))

                elif event == "done":
                    result = int(payload)
                    self._set_running_state(False)
                    self.current_var.set("No script running")

                    if self.stop_requested:
                        self.status_var.set("Stopped")
                        self.console.insert(
                            tk.END,
                            "\n=== Sequence stopped ===\n",
                        )
                    elif result == 0:
                        self.status_var.set("Completed")
                        self.console.insert(
                            tk.END,
                            "\n=== All selected commands completed successfully ===\n",
                        )
                    else:
                        self.status_var.set(f"Failed ({result})")
                        self.console.insert(
                            tk.END,
                            f"\n=== Sequence failed with exit code {result} ===\n",
                        )

                    self.console.see(tk.END)

        except queue.Empty:
            pass

        self.after(self.POLL_MS, self._poll_output_queue)

    def _set_running_state(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        self.run_selected_button.configure(state=state)
        self.run_build_all_button.configure(state=state)
        self.run_eve_asset_button.configure(state=state)
        self.stop_button.configure(
            state=tk.NORMAL if running else tk.DISABLED
        )

    def _on_close(self) -> None:
        if self.worker and self.worker.is_alive():
            close = messagebox.askyesno(
                "Build running",
                "Stop the running process and close?",
            )
            if not close:
                return
            self._request_stop()

        self.destroy()


def main() -> int:
    app = BuildGui()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
