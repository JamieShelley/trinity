#!/usr/bin/env python3
"""Canonical command surface for the active NSAMDR workflow."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools/nsamdr"
NEURAL_ROOT = TOOLS_ROOT / "neural"
BUILD_ROOT = REPO_ROOT / "scripts/build"
CUDA_PYTHON = REPO_ROOT / "artifacts/nsamdr/python-env/Scripts/python.exe"
CPU_PYTHON = REPO_ROOT / "artifacts/nsamdr/python-env-cpu/Scripts/python.exe"

REQUIRED_LAYOUT = (
    "scripts/build/nsamdr.bat",
    "scripts/build/setup_nsamdr_cuda.bat",
    "scripts/build/setup_nsamdr_cpu.bat",
    "scripts/build/run_nsamdr_obj_preview_dx11.bat",
    "scripts/build/nsamdr/NSAMDROBJProjectInclude.cmake",
    "tools/nsamdr/gui/nsamdr_v16_workflow_gui.py",
    "tools/nsamdr/gui/nsamdr_v16_structure_workflow_gui.py",
    "tools/nsamdr/neural/v14/baseline.py",
    "tools/nsamdr/neural/v14/config.py",
    "tools/nsamdr/neural/v14/model.py",
    "tools/nsamdr/neural/v14/dataset.py",
    "tools/nsamdr/neural/v14/losses.py",
    "tools/nsamdr/neural/v14/qualification.py",
    "tools/nsamdr/neural/v14/checkpoint.py",
    "tools/nsamdr/neural/v14/inference.py",
    "tools/nsamdr/neural/v14/trainer.py",
    "tools/nsamdr/neural/v14/workflow.py",
    "tools/nsamdr/neural/v14/preview.py",
    "tools/nsamdr/neural/v14/safe_live_resume_monitored_fourfamily_multiregion_diagnostic.py",
    "tools/nsamdr/neural/v16/structure.py",
    "tools/nsamdr/neural/v16/profiles.py",
    "tools/nsamdr/neural/v16/stage2_data.py",
    "tools/nsamdr/neural/audit_nsamdr_v16_structure_support.py",
    "tools/nsamdr/neural/audit_nsamdr_v16_boundary_profiles.py",
    "tools/nsamdr/neural/scan_eve_authored_corpus.py",
    "tools/nsamdr/eve_asset_test.py",
    "trinityal/tests/nsamdr/NSAMDRShipPreview.cpp",
    "trinityal/tests/nsamdr/NSAMDRPreview.hlsl",
)

CURRENT_TESTS = (
    "tools.nsamdr.neural.test_nsamdr_v16_structure",
    "tools.nsamdr.neural.test_nsamdr_v16_boundary_profiles",
    "tools.nsamdr.neural.test_nsamdr_v16_dataset_split",
    "tools.nsamdr.neural.test_nsamdr_v16_multifamily_balance",
    "tools.nsamdr.neural.test_nsamdr_eve_corpus_census",
    "tools.nsamdr.gui.test_nsamdr_v16_workflow_gui",
    "tools.nsamdr.test_nsamdr_cli",
)


class NSAMDRCommandLineApplication:
    def _display(self, command: Sequence[object]) -> str:
        return subprocess.list2cmdline([os.fspath(value) for value in command])

    def _run(
        self,
        command: Sequence[object],
        *,
        env: dict[str, str] | None = None,
    ) -> int:
        values = [os.fspath(value) for value in command]
        print("[nsamdr] RUN: " + self._display(values), flush=True)
        return subprocess.run(
            values,
            cwd=REPO_ROOT,
            env=env,
            check=False,
        ).returncode

    def _cmd_quote(self, value: object) -> str:
        text = os.fspath(value)
        if '"' in text:
            raise ValueError(f"BAT arguments cannot contain a double quote: {text!r}")
        return f'"{text}"'

    def _run_batch(
        self,
        script: Path,
        arguments: Sequence[object] = (),
        *,
        env: dict[str, str] | None = None,
    ) -> int:
        command_text = "call " + " ".join(
            self._cmd_quote(value) for value in (script, *arguments)
        )
        comspec = self._cmd_quote(os.environ.get("COMSPEC", "cmd.exe"))
        command_line = f'{comspec} /d /s /c "{command_text}"'
        print("[nsamdr] RUN: " + command_line, flush=True)
        return subprocess.run(
            command_line,
            cwd=REPO_ROOT,
            env=env,
            check=False,
        ).returncode

    def _setup_script(self, kind: str) -> Path:
        return BUILD_ROOT / f"setup_nsamdr_{kind}.bat"

    def _python(self, kind: str = "cuda", *, bootstrap: bool = True) -> Path:
        override = os.environ.get("NSAMDR_PYTHON_EXE", "").strip()
        if override:
            candidate = Path(override).expanduser()
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            candidate = candidate.resolve()
            if candidate.is_file():
                return candidate
            raise RuntimeError(f"NSAMDR_PYTHON_EXE does not name a file: {candidate}")

        candidate = CUDA_PYTHON if kind == "cuda" else CPU_PYTHON
        if candidate.is_file():
            return candidate
        if not bootstrap:
            return Path(sys.executable)
        code = self._run_batch(self._setup_script(kind))
        if code:
            raise SystemExit(code)
        if not candidate.is_file():
            raise RuntimeError(f"setup completed without creating {candidate}")
        return candidate

    def _python_script(
        self,
        relative: str,
        arguments: Sequence[str],
        *,
        kind: str = "cuda",
        env: dict[str, str] | None = None,
    ) -> int:
        return self._run(
            [self._python(kind), "-u", REPO_ROOT / relative, *arguments],
            env=env,
        )

    def _repo_args(self, arguments: Sequence[str]) -> list[str]:
        return ["--repo-root", os.fspath(REPO_ROOT), *arguments]

    def _configure_cuda_allocator_env(self, env: dict[str, str]) -> None:
        # The broad V16 diagnostics are stable on Windows without allocator
        # overrides. Force the Raven workflow back to PyTorch's default Windows
        # allocator path: the tuned allocator path has produced a native abort
        # immediately after the first training tile on the RTX 50-series setup.
        if os.name == "nt":
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            return
        if "PYTORCH_CUDA_ALLOC_CONF" not in env:
            env["PYTORCH_CUDA_ALLOC_CONF"] = (
                "expandable_segments:True,garbage_collection_threshold:0.80"
            )

    def _git_output(self, *arguments: str) -> tuple[int, str]:
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        try:
            completed = subprocess.run(
                ["git", "-C", os.fspath(REPO_ROOT), *arguments],
                check=False,
                capture_output=True,
                text=True,
                env=env,
                timeout=10.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return 127, ""
        return int(completed.returncode), completed.stdout.strip()

    def _source_freshness_preflight(self) -> int:
        if not (REPO_ROOT / ".git").exists():
            print(
                "[nsamdr] WARNING: Git metadata unavailable; source freshness cannot be verified.",
                flush=True,
            )
            return 0
        head_code, head = self._git_output("rev-parse", "HEAD")
        branch_code, branch = self._git_output("rev-parse", "--abbrev-ref", "HEAD")
        status_code, status = self._git_output(
            "status", "--porcelain=v1", "--untracked-files=no"
        )
        if head_code or branch_code or status_code:
            print(
                "[nsamdr] WARNING: unable to read complete Git source provenance.",
                flush=True,
            )
            return 0
        if status.strip():
            print(
                "ERROR: NSAMDR tracked source files are modified. Commit or stash them before training.",
                file=sys.stderr,
            )
            print(status, file=sys.stderr)
            return 5
        if branch != "NSAMDR":
            print(
                f"[nsamdr] Source HEAD {head} on branch {branch}; origin/NSAMDR freshness check skipped.",
                flush=True,
            )
            return 0
        remote_code, remote = self._git_output(
            "ls-remote", "origin", "refs/heads/NSAMDR"
        )
        if remote_code or not remote:
            print(
                f"[nsamdr] WARNING: unable to query origin/NSAMDR; continuing with local HEAD {head}.",
                flush=True,
            )
            return 0
        remote_head = remote.split()[0].strip()
        if remote_head != head:
            print(
                "ERROR: local NSAMDR checkout is stale; refusing to start training.",
                file=sys.stderr,
            )
            print(f"  local HEAD    : {head}", file=sys.stderr)
            print(f"  origin/NSAMDR : {remote_head}", file=sys.stderr)
            print("  action        : git pull --ff-only", file=sys.stderr)
            return 5
        print(f"[nsamdr] Source revision verified: NSAMDR {head}", flush=True)
        return 0

    def _cuda_preflight(self, python: Path, env: dict[str, str]) -> int:
        return self._run(
            [
                python,
                "-u",
                NEURAL_ROOT / "verify_cuda.py",
                "--require-arch",
                "sm_120",
                "--quick",
            ],
            env=env,
        )

    def validate_layout(self) -> int:
        missing = [
            relative
            for relative in REQUIRED_LAYOUT
            if not (REPO_ROOT / relative).is_file()
        ]
        if missing:
            print("ERROR: Missing active NSAMDR source/layout requirement:", file=sys.stderr)
            for relative in missing:
                print(f"  {relative}", file=sys.stderr)
            return 4
        print("Active NSAMDR source layout verified.", flush=True)
        return 0

    def _command_gui(self, _args: argparse.Namespace) -> int:
        return self._python_script(
            "tools/nsamdr/gui/nsamdr_v16_workflow_gui.py",
            [],
            kind="cpu",
        )

    def _command_setup(self, args: argparse.Namespace) -> int:
        return self._run_batch(
            self._setup_script(args.kind),
            ["--force"] if args.force else [],
        )

    def _command_raven_quick(self, args: argparse.Namespace) -> int:
        source_code = self._source_freshness_preflight()
        if source_code:
            return source_code
        python = self._python("cuda")
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        self._configure_cuda_allocator_env(env)
        code = self._cuda_preflight(python, env)
        if code:
            return code

        forwarded = [
            "--shared-cache", args.shared_cache,
            "--max-train-regions", str(args.max_train_regions),
            "--max-validation-regions", str(args.max_validation_regions),
            "--experiment", args.experiment,
            "--control", args.control,
            "--preview-target-size", str(args.preview_target_size),
            "--preview-device", args.preview_device,
            "--performance-profile", args.performance_profile,
            "--workers", str(args.workers),
            "--prefetch-factor", str(args.prefetch_factor),
            "--amp-precision", args.amp_precision,
        ]
        if args.rebuild_dataset:
            forwarded.append("--rebuild-dataset")
        if args.live_preview_during_training:
            forwarded += [
                "--live-preview-during-training",
                "--live-preview-target-size",
                str(args.live_preview_target_size),
            ]

        return self._run(
            [
                python,
                "-u",
                NEURAL_ROOT / "v14/workflow.py",
                "--training-mode",
                "quick",
                *self._repo_args(forwarded),
            ],
            env=env,
        )

    def _command_index_raven(self, args: argparse.Namespace) -> int:
        forwarded = [
            "--shared-cache", args.shared_cache,
            "--train-crops", str(args.train_crops),
            "--validation-crops", str(args.validation_crops),
        ]
        if args.rebuild:
            forwarded.append("--rebuild")
        return self._python_script(
            "tools/nsamdr/neural/prepare_nsamdr_v16_raven_dataset.py",
            self._repo_args(forwarded),
        )

    def _command_preview(self, args: argparse.Namespace) -> int:
        forwarded = ["--experiment", args.subject]
        if args.shared_cache is not None:
            forwarded += ["--shared-cache", args.shared_cache]
        if args.target_size is not None:
            forwarded += ["--target-size", str(args.target_size)]
        if args.device is not None:
            forwarded += ["--device", args.device]
        return self._python_script(
            "tools/nsamdr/neural/v14/preview.py",
            self._repo_args(forwarded),
        )

    def _command_validate(self, args: argparse.Namespace) -> int:
        code = self.validate_layout()
        if code or args.layout_only:
            return code
        return self._run(
            [self._python("cpu"), "-m", "unittest", *CURRENT_TESTS]
        )

    def _safe_target(self, relative: str) -> Path:
        root = REPO_ROOT.resolve()
        target = (REPO_ROOT / relative).resolve()
        target.relative_to(root)
        if target == root:
            raise ValueError("refusing to clean repository root")
        return target

    def _command_cleanup(self, args: argparse.Namespace) -> int:
        targets = []
        if args.training_data:
            targets.append(("artifacts/nsamdr/training_v9_preview_raven", "training data"))
        if args.experiments:
            targets.append(("artifacts/nsamdr/experiments", "experiments"))
        if args.diagnostics:
            targets.append(("artifacts/nsamdr/diagnostics", "diagnostics"))
        for relative, label in targets:
            target = self._safe_target(relative)
            if not target.exists():
                continue
            print(
                f"[nsamdr] {'would remove' if args.dry_run else 'removed'}: {label}: {target}",
                flush=True,
            )
            if not args.dry_run:
                shutil.rmtree(target) if target.is_dir() else target.unlink()
        return 0

    def _command_integrate(self, args: argparse.Namespace) -> int:
        forwarded = ["--repo-root", os.fspath(REPO_ROOT)]
        if args.check:
            forwarded.append("--check")
        return self._python_script(
            "tools/nsamdr/integration/apply_trinity_nsamdr_settings.py",
            forwarded,
            kind="cpu",
        )

    def _command_native(self, args: argparse.Namespace) -> int:
        launcher = BUILD_ROOT / "run_nsamdr_obj_preview_dx11.bat"
        env = os.environ.copy()
        if args.native_name == "build":
            env["NSAMDR_BUILD_ONLY"] = "1"
            return self._run_batch(launcher, [], env=env)
        if args.native_name == "obj":
            return self._run_batch(launcher, args.arguments, env=env)
        return self._python_script(
            "tools/nsamdr/eve_asset_test.py",
            [
                "prepare-run",
                "--repo-root", os.fspath(REPO_ROOT),
                "--shared-cache", args.shared_cache,
                "--query", args.query,
                "--launcher", os.fspath(launcher),
                *args.arguments,
            ],
            kind="cpu",
            env=env,
        )

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="nsamdr",
            description="NSAMDR V16.2 development and production utilities",
        )
        commands = parser.add_subparsers(dest="command", required=True)

        gui = commands.add_parser("gui")
        gui.set_defaults(handler=self._command_gui)

        setup = commands.add_parser("setup")
        setup.add_argument("kind", choices=("cuda", "cpu"))
        setup.add_argument("--force", action="store_true")
        setup.set_defaults(handler=self._command_setup)

        quick = commands.add_parser("raven-quick")
        quick.add_argument("--shared-cache", default=r"C:\CCP\EVE")
        quick.add_argument("--max-train-regions", type=int, default=16)
        quick.add_argument("--max-validation-regions", type=int, default=4)
        quick.add_argument("--experiment", default="new")
        quick.add_argument("--control", default="auto")
        quick.add_argument("--preview-target-size", type=int, default=4096)
        quick.add_argument(
            "--preview-device",
            choices=("cuda", "cpu", "auto"),
            default="cuda",
        )
        quick.add_argument("--performance-profile", default="fast")
        quick.add_argument("--workers", type=int, default=(0 if os.name == "nt" else 4))
        quick.add_argument("--prefetch-factor", type=int, default=2)
        quick.add_argument(
            "--amp-precision",
            choices=("auto", "bf16", "fp16"),
            default="auto",
        )
        quick.add_argument("--rebuild-dataset", action="store_true")
        quick.add_argument("--live-preview-during-training", action="store_true")
        quick.add_argument("--live-preview-target-size", type=int, default=1024)
        quick.set_defaults(handler=self._command_raven_quick)

        index = commands.add_parser("index")
        index_commands = index.add_subparsers(dest="index_name", required=True)
        raven = index_commands.add_parser("raven")
        raven.add_argument("--shared-cache", default=r"C:\CCP\EVE")
        raven.add_argument("--train-crops", type=int, default=16)
        raven.add_argument("--validation-crops", type=int, default=4)
        raven.add_argument("--rebuild", action="store_true")
        raven.set_defaults(handler=self._command_index_raven)

        preview = commands.add_parser("preview")
        preview.add_argument("subject")
        preview.add_argument("--shared-cache")
        preview.add_argument("--target-size", type=int)
        preview.add_argument("--device", choices=("cuda", "cpu", "auto"))
        preview.set_defaults(handler=self._command_preview)

        validate = commands.add_parser("validate")
        validate.add_argument("--layout-only", action="store_true")
        validate.set_defaults(handler=self._command_validate)

        cleanup = commands.add_parser("cleanup")
        cleanup.add_argument("--training-data", action="store_true")
        cleanup.add_argument("--experiments", action="store_true")
        cleanup.add_argument("--diagnostics", action="store_true")
        cleanup.add_argument("--dry-run", action="store_true")
        cleanup.set_defaults(handler=self._command_cleanup)

        integrate = commands.add_parser("integrate")
        integrate.add_argument("--check", action="store_true")
        integrate.set_defaults(handler=self._command_integrate)

        native = commands.add_parser("native")
        native_commands = native.add_subparsers(dest="native_name", required=True)
        build = native_commands.add_parser("build")
        build.set_defaults(handler=self._command_native)
        obj = native_commands.add_parser("obj")
        obj.add_argument("arguments", nargs=argparse.REMAINDER)
        obj.set_defaults(handler=self._command_native)
        eve = native_commands.add_parser("eve")
        eve.add_argument("--shared-cache", default="")
        eve.add_argument(
            "--query",
            default="res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2",
        )
        eve.add_argument("arguments", nargs=argparse.REMAINDER)
        eve.set_defaults(handler=self._command_native)
        return parser

    def main(self, argv: Sequence[str] | None = None) -> int:
        parser = self.build_parser()
        args = parser.parse_args(argv)
        try:
            return int(args.handler(args))
        except KeyboardInterrupt:
            return 130
        except SystemExit:
            raise
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1


_app = NSAMDRCommandLineApplication()


if __name__ == "__main__":
    raise SystemExit(_app.main())
