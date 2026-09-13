#!/usr/bin/env python3
"""Canonical NSAMDR V14 command surface."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools" / "nsamdr"
NEURAL_ROOT = TOOLS_ROOT / "neural"
CUDA_PYTHON = REPO_ROOT / "artifacts" / "nsamdr" / "python-env" / "Scripts" / "python.exe"
CPU_PYTHON = REPO_ROOT / "artifacts" / "nsamdr" / "python-env-cpu" / "Scripts" / "python.exe"
BUILD_ROOT = REPO_ROOT / "scripts" / "build"

REQUIRED_LAYOUT = (
    "scripts/build/nsamdr.bat",
    "scripts/build/setup_nsamdr_cuda.bat",
    "scripts/build/setup_nsamdr_cpu.bat",
    "scripts/build/run_nsamdr_obj_preview_dx11.bat",
    "scripts/build/nsamdr/NSAMDROBJProjectInclude.cmake",
    "tools/nsamdr/nsamdr_cli.py",
    "tools/nsamdr/gui/nsamdr_v14_workflow_gui.py",
    "tools/nsamdr/neural/prepare_nsamdr_v9_raven_preview_dataset.py",
    "tools/nsamdr/neural/v14/config.py",
    "tools/nsamdr/neural/v14/baseline.py",
    "tools/nsamdr/neural/v14/model.py",
    "tools/nsamdr/neural/v14/dataset.py",
    "tools/nsamdr/neural/v14/losses.py",
    "tools/nsamdr/neural/v14/qualification.py",
    "tools/nsamdr/neural/v14/checkpoint.py",
    "tools/nsamdr/neural/v14/inference.py",
    "tools/nsamdr/neural/v14/trainer.py",
    "tools/nsamdr/neural/v14/workflow.py",
    "tools/nsamdr/neural/v14/preview.py",
    "tools/nsamdr/neural/v14/live_view.py",
    "tools/nsamdr/neural/test_nsamdr_v14.py",
    "tools/nsamdr/neural/test_nsamdr_v14_checkpoint.py",
    "tools/nsamdr/eve_asset_test.py",
    "trinityal/tests/nsamdr/NSAMDRShipPreview.cpp",
    "trinityal/tests/nsamdr/NSAMDRPreview.hlsl",
)


class NSAMDRCommandLineApplication:
    def _display(self, command: Sequence[object]) -> str:
        return subprocess.list2cmdline([os.fspath(value) for value in command])

    def _run(self, command: Sequence[object], *, env: dict[str, str] | None = None) -> int:
        values = [os.fspath(value) for value in command]
        print("[nsamdr] RUN: " + self._display(values), flush=True)
        return subprocess.run(values, cwd=REPO_ROOT, env=env, check=False).returncode

    def _cmd_quote(self, value: object) -> str:
        text = os.fspath(value)
        if '"' in text:
            raise ValueError(f"BAT arguments cannot contain a double quote: {text!r}")
        return f'"{text}"'

    def _run_batch(self, script: Path, arguments: Sequence[object] = (), *, env: dict[str, str] | None = None) -> int:
        command_text = "call " + " ".join(self._cmd_quote(value) for value in (script, *arguments))
        command_line = f'{self._cmd_quote(os.environ.get("COMSPEC", "cmd.exe"))} /d /s /c "{command_text}"'
        print("[nsamdr] RUN: " + command_line, flush=True)
        return subprocess.run(command_line, cwd=REPO_ROOT, env=env, check=False).returncode

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

    def _python_script(self, relative: str, arguments: Sequence[str], *, kind: str = "cuda", env: dict[str, str] | None = None) -> int:
        return self._run([self._python(kind), "-u", REPO_ROOT / relative, *arguments], env=env)

    def _repo_args(self, arguments: Sequence[str]) -> list[str]:
        return ["--repo-root", os.fspath(REPO_ROOT), *arguments]

    def _reject_arguments(self, args: argparse.Namespace, command: str) -> int:
        if not args.arguments:
            return 0
        print(f"ERROR: unrecognized arguments for 'nsamdr {command}': {self._display(args.arguments)}", file=sys.stderr)
        return 2

    def _configure_cuda_allocator_env(self, env: dict[str, str]) -> None:
        if "PYTORCH_CUDA_ALLOC_CONF" in env:
            return
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            "garbage_collection_threshold:0.80"
            if os.name == "nt"
            else "expandable_segments:True,garbage_collection_threshold:0.80"
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
            print("[nsamdr] WARNING: Git metadata unavailable; source freshness cannot be verified.", flush=True)
            return 0
        head_code, head = self._git_output("rev-parse", "HEAD")
        branch_code, branch = self._git_output("rev-parse", "--abbrev-ref", "HEAD")
        status_code, status = self._git_output("status", "--porcelain=v1", "--untracked-files=no")
        if head_code or branch_code or status_code:
            print("[nsamdr] WARNING: unable to read complete Git source provenance.", flush=True)
            return 0
        if status.strip():
            print("ERROR: NSAMDR tracked source files are modified. Commit or stash them before training.", file=sys.stderr)
            print(status, file=sys.stderr)
            return 5
        if branch != "NSAMDR":
            print(f"[nsamdr] Source HEAD {head} on branch {branch}; origin/NSAMDR freshness check skipped.", flush=True)
            return 0
        remote_code, remote = self._git_output("ls-remote", "origin", "refs/heads/NSAMDR")
        if remote_code or not remote:
            print(f"[nsamdr] WARNING: unable to query origin/NSAMDR; continuing with local HEAD {head}.", flush=True)
            return 0
        remote_head = remote.split()[0].strip()
        if remote_head != head:
            print("ERROR: local NSAMDR checkout is stale; refusing to start training.", file=sys.stderr)
            print(f"  local HEAD    : {head}", file=sys.stderr)
            print(f"  origin/NSAMDR : {remote_head}", file=sys.stderr)
            print("  action        : git pull --ff-only", file=sys.stderr)
            return 5
        print(f"[nsamdr] Source revision verified: NSAMDR {head}", flush=True)
        return 0

    def _cuda_preflight(self, python: Path, env: dict[str, str]) -> int:
        return self._run([python, "-u", NEURAL_ROOT / "verify_cuda.py", "--require-arch", "sm_120", "--quick"], env=env)

    def _command_gui(self, args: argparse.Namespace) -> int:
        code = self._reject_arguments(args, "gui")
        return code or self._python_script("tools/nsamdr/gui/nsamdr_v14_workflow_gui.py", [])

    def _command_setup(self, args: argparse.Namespace) -> int:
        code = self._reject_arguments(args, "setup")
        return code or self._run_batch(self._setup_script(args.kind), ["--force"] if args.force else [])

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
        return self._run(
            [python, "-u", NEURAL_ROOT / "v14" / "workflow.py", "--training-mode", "quick", *self._repo_args(args.arguments)],
            env=env,
        )

    def _command_full_train(self, args: argparse.Namespace) -> int:
        print("ERROR: V14 Full Training is disabled until Raven Quick qualifies the HR-first architecture.", file=sys.stderr)
        return 2

    def _command_index_raven(self, args: argparse.Namespace) -> int:
        forwarded = [
            "--shared-cache", args.shared_cache,
            "--train-crops", str(args.train_crops),
            "--validation-crops", str(args.validation_crops),
        ]
        if args.rebuild:
            forwarded.append("--rebuild")
        return self._python_script("tools/nsamdr/neural/prepare_nsamdr_v9_raven_preview_dataset.py", self._repo_args(forwarded))

    def _command_preview(self, args: argparse.Namespace) -> int:
        forwarded = ["--experiment", args.subject]
        if args.shared_cache is not None:
            forwarded += ["--shared-cache", args.shared_cache]
        if args.target_size is not None:
            forwarded += ["--target-size", str(args.target_size)]
        if args.device is not None:
            forwarded += ["--device", args.device]
        return self._python_script("tools/nsamdr/neural/v14/preview.py", self._repo_args(forwarded))

    def validate_layout(self) -> int:
        missing = [relative for relative in REQUIRED_LAYOUT if not (REPO_ROOT / relative).is_file()]
        if missing:
            print("ERROR: Missing canonical NSAMDR V14 source/layout requirement:", file=sys.stderr)
            for relative in missing:
                print(f"  {relative}", file=sys.stderr)
            return 4
        print("Canonical NSAMDR V14 source layout verified.", flush=True)
        return 0

    def _command_test(self, args: argparse.Namespace) -> int:
        if args.test_name in {"contract", "architecture"}:
            code = self._reject_arguments(args, f"test {args.test_name}")
            if code:
                return code
            python = self._python("cpu" if args.device == "cpu" else "cuda")
            env = os.environ.copy()
            if args.device == "cuda":
                code = self._cuda_preflight(python, env)
                if code:
                    return code
            return self._run([python, "-u", NEURAL_ROOT / "test_nsamdr_v14.py"], env=env)
        forwarded = ["--checkpoint", args.checkpoint, "--device", args.device]
        return self._python_script("tools/nsamdr/neural/test_nsamdr_v14_checkpoint.py", self._repo_args(forwarded), kind="cpu" if args.device == "cpu" else "cuda")

    def _command_validate(self, args: argparse.Namespace) -> int:
        code = self._reject_arguments(args, "validate")
        if code:
            return code
        code = self.validate_layout()
        if code or args.layout_only:
            return code
        return self._command_test(argparse.Namespace(test_name="architecture", device=args.device, arguments=[]))

    def _safe_target(self, relative: str) -> Path:
        root = REPO_ROOT.resolve()
        target = (REPO_ROOT / relative).resolve()
        target.relative_to(root)
        if target == root:
            raise ValueError("refusing to clean repository root")
        return target

    def _command_cleanup(self, args: argparse.Namespace) -> int:
        code = self._reject_arguments(args, "cleanup")
        if code:
            return code
        if args.all_artifacts:
            args.tuning_dataset = args.experiments = args.production = True
        mapping = (
            (args.tuning_dataset, "artifacts/nsamdr/training_v9_preview_raven", "Raven authored dataset"),
            (args.experiments, "artifacts/nsamdr/experiments", "experiments"),
            (args.production, "artifacts/nsamdr/neural_v14", "V14 production state"),
        )
        for selected, relative, label in mapping:
            if not selected:
                continue
            target = self._safe_target(relative)
            if not target.exists():
                continue
            print(f"[nsamdr] {'would remove' if args.dry_run else 'removed'}: {label}: {target}", flush=True)
            if not args.dry_run:
                shutil.rmtree(target) if target.is_dir() else target.unlink()
        return 0

    def _command_integrate(self, args: argparse.Namespace) -> int:
        forwarded = ["--repo-root", os.fspath(REPO_ROOT)]
        if args.check:
            forwarded.append("--check")
        forwarded += args.arguments
        return self._python_script("tools/nsamdr/integration/apply_trinity_nsamdr_settings.py", forwarded)

    def _command_native(self, args: argparse.Namespace) -> int:
        launcher = BUILD_ROOT / "run_nsamdr_obj_preview_dx11.bat"
        env = os.environ.copy()
        if args.native_name == "build":
            code = self._reject_arguments(args, "native build")
            if code:
                return code
            env["NSAMDR_BUILD_ONLY"] = "1"
            return self._run_batch(launcher, [], env=env)
        if args.native_name == "obj":
            return self._run_batch(launcher, args.arguments, env=env)
        return self._python_script(
            "tools/nsamdr/eve_asset_test.py",
            ["prepare-run", "--repo-root", os.fspath(REPO_ROOT), "--shared-cache", args.shared_cache, "--query", args.query, "--launcher", os.fspath(launcher), *args.arguments],
            env=env,
        )

    def _set_handler(self, parser: argparse.ArgumentParser, handler: Callable[[argparse.Namespace], int]) -> None:
        parser.set_defaults(handler=handler)

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="nsamdr", description="NSAMDR V14 HR-first training, qualification and preview")
        commands = parser.add_subparsers(dest="command", required=True)

        gui = commands.add_parser("gui")
        self._set_handler(gui, self._command_gui)

        setup = commands.add_parser("setup")
        setup.add_argument("kind", choices=("cuda", "cpu"))
        setup.add_argument("--force", action="store_true")
        self._set_handler(setup, self._command_setup)

        quick = commands.add_parser("raven-quick")
        self._set_handler(quick, self._command_raven_quick)

        full = commands.add_parser("full-train")
        self._set_handler(full, self._command_full_train)

        index = commands.add_parser("index")
        index_commands = index.add_subparsers(dest="index_name", required=True)
        raven = index_commands.add_parser("raven")
        raven.add_argument("--shared-cache", default=r"C:\CCP\EVE")
        raven.add_argument("--train-crops", type=int, default=16)
        raven.add_argument("--validation-crops", type=int, default=4)
        raven.add_argument("--rebuild", action="store_true")
        self._set_handler(raven, self._command_index_raven)

        preview = commands.add_parser("preview")
        preview.add_argument("subject")
        preview.add_argument("--shared-cache")
        preview.add_argument("--target-size", type=int)
        preview.add_argument("--device", choices=("cuda", "cpu", "auto"))
        self._set_handler(preview, self._command_preview)

        validate = commands.add_parser("validate")
        validate.add_argument("--layout-only", action="store_true")
        validate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        self._set_handler(validate, self._command_validate)

        test = commands.add_parser("test")
        tests = test.add_subparsers(dest="test_name", required=True)
        for name in ("contract", "architecture"):
            item = tests.add_parser(name)
            item.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
            self._set_handler(item, self._command_test)
        checkpoint = tests.add_parser("checkpoint")
        checkpoint.add_argument("--checkpoint", required=True)
        checkpoint.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        self._set_handler(checkpoint, self._command_test)

        cleanup = commands.add_parser("cleanup")
        cleanup.add_argument("--tuning-dataset", action="store_true")
        cleanup.add_argument("--experiments", action="store_true")
        cleanup.add_argument("--production", action="store_true")
        cleanup.add_argument("--all-artifacts", action="store_true")
        cleanup.add_argument("--dry-run", action="store_true")
        self._set_handler(cleanup, self._command_cleanup)

        integrate = commands.add_parser("integrate")
        integrate.add_argument("--check", action="store_true")
        self._set_handler(integrate, self._command_integrate)

        native = commands.add_parser("native")
        native_commands = native.add_subparsers(dest="native_name", required=True)
        build = native_commands.add_parser("build")
        self._set_handler(build, self._command_native)
        obj = native_commands.add_parser("obj")
        obj.add_argument("arguments", nargs=argparse.REMAINDER)
        self._set_handler(obj, self._command_native)
        eve = native_commands.add_parser("eve")
        eve.add_argument("--shared-cache", default="")
        eve.add_argument("--query", default="res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2")
        self._set_handler(eve, self._command_native)
        return parser

    def main(self, argv: Sequence[str] | None = None) -> int:
        parser = self.build_parser()
        args, unknown = parser.parse_known_args(argv)
        args.arguments = [*getattr(args, "arguments", []), *unknown]
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
