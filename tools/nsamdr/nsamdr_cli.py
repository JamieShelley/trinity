#!/usr/bin/env python3
"""Unified command surface for the canonical NSAMDR workflow."""
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

DEFAULT_RAVEN_CONFIG = "tools/nsamdr/neural/configs/v9_preview_raven.json"
DEFAULT_FULL_CONFIG = "tools/nsamdr/neural/configs/v9_fidelity_full.json"

REQUIRED_LAYOUT = (
    "scripts/build/nsamdr.bat",
    "scripts/build/run_nsamdr_v9_gui.bat",
    "scripts/build/setup_nsamdr_cuda.bat",
    "scripts/build/setup_nsamdr_cpu.bat",
    "scripts/build/run_nsamdr_obj_preview_dx11.bat",
    "scripts/build/nsamdr/NSAMDROBJProjectInclude.cmake",
    "tools/nsamdr/nsamdr_cli.py",
    "tools/nsamdr/gui/nsamdr_v9_workflow_gui.py",
    "tools/nsamdr/neural/run_nsamdr_v9_raven_tune_preview.py",
    "tools/nsamdr/neural/prepare_nsamdr_v9_raven_preview_dataset.py",
    "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py",
    "tools/nsamdr/neural/preview_nsamdr_v9_experiment.py",
    "tools/nsamdr/neural/train_nsamdr_v9.py",
    "tools/nsamdr/neural/test_nsamdr_v9_contract.py",
    "tools/nsamdr/neural/test_nsamdr_v9_checkpoint.py",
    "tools/nsamdr/neural/smoke_test_nsamdr_v9.py",
    "tools/nsamdr/neural/v9/config.py",
    "tools/nsamdr/neural/v9/model.py",
    "tools/nsamdr/neural/v9/dataset.py",
    "tools/nsamdr/neural/v9/contours.py",
    "tools/nsamdr/neural/configs/v9_preview_raven.json",
    "tools/nsamdr/neural/configs/v9_fidelity_full.json",
    "tools/nsamdr/eve_asset_test.py",
    "tools/nsamdr/generate_strategy_candidates.py",
    "trinityal/tests/nsamdr/NSAMDRShipPreview.cpp",
    "trinityal/tests/nsamdr/NSAMDRPreview.hlsl",
)


def _display(command: Sequence[object]) -> str:
    return subprocess.list2cmdline([os.fspath(value) for value in command])


def _run(
    command: Sequence[object],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    values = [os.fspath(value) for value in command]
    print("[nsamdr] RUN: " + _display(values), flush=True)
    return subprocess.run(
        values,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
    )


def _run_code(command: Sequence[object], **kwargs: object) -> int:
    return _run(command, **kwargs).returncode


def _cmd_quote(value: object) -> str:
    """Quote one argument for the cmd.exe command string used to call a BAT."""
    text = os.fspath(value)
    if '"' in text:
        raise ValueError(f'BAT arguments cannot contain a double quote: {text!r}')
    return f'"{text}"'


def _run_batch(
    script: Path,
    arguments: Sequence[object] = (),
    *,
    env: dict[str, str] | None = None,
) -> int:
    command_text = "call " + " ".join(_cmd_quote(value) for value in (script, *arguments))
    command_line = (
        f'{_cmd_quote(os.environ.get("COMSPEC", "cmd.exe"))} '
        f'/d /s /c "{command_text}"'
    )
    print("[nsamdr] RUN: " + command_line, flush=True)
    return subprocess.run(
        command_line,
        cwd=REPO_ROOT,
        env=env,
        check=False,
    ).returncode


def _setup_script(kind: str) -> Path:
    return BUILD_ROOT / f"setup_nsamdr_{kind}.bat"


def _python(kind: str = "cuda", *, bootstrap: bool = True) -> Path:
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
    setup = _setup_script(kind)
    print(f"[nsamdr] Preparing the missing {kind.upper()} Python environment...", flush=True)
    code = _run_batch(setup)
    if code:
        raise SystemExit(code)
    if not candidate.is_file():
        raise RuntimeError(f"setup completed without creating {candidate}")
    return candidate


def _python_script(relative: str, arguments: Sequence[str], *, kind: str = "cuda") -> int:
    return _run_code([_python(kind), "-u", REPO_ROOT / relative, *arguments])


def _repo_args(arguments: Sequence[str]) -> list[str]:
    return ["--repo-root", os.fspath(REPO_ROOT), *arguments]


def _reject_arguments(args: argparse.Namespace, command: str) -> int:
    if not args.arguments:
        return 0
    print(
        f"ERROR: unrecognized arguments for 'nsamdr {command}': {_display(args.arguments)}",
        file=sys.stderr,
    )
    return 2


def _command_gui(args: argparse.Namespace) -> int:
    code = _reject_arguments(args, "gui")
    if code:
        return code
    return _python_script("tools/nsamdr/gui/nsamdr_v9_workflow_gui.py", [])


def _command_setup(args: argparse.Namespace) -> int:
    code = _reject_arguments(args, "setup")
    if code:
        return code
    forwarded = ["--force"] if args.force else []
    return _run_batch(_setup_script(args.kind), forwarded)


def _configure_cuda_allocator_env(env: dict[str, str]) -> None:
    """Set allocator defaults that are supported on the current platform.

    PyTorch 2.11 on native Windows warns that expandable_segments is not
    supported. Preserve the useful garbage-collection threshold there and only
    request expandable segments on platforms that support it. Explicit user
    configuration always wins.
    """
    if "PYTORCH_CUDA_ALLOC_CONF" in env:
        return
    if os.name == "nt":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.80"
    else:
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True,garbage_collection_threshold:0.80"
        )


def _command_workflow(args: argparse.Namespace, training_mode: str) -> int:
    python = _python("cuda")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    _configure_cuda_allocator_env(env)
    code = _run_code(
        [python, "-u", NEURAL_ROOT / "verify_cuda.py", "--require-arch", "sm_120", "--quick"],
        env=env,
    )
    if code:
        return code
    return _run_code(
        [
            python,
            "-u",
            NEURAL_ROOT / "run_nsamdr_v9_raven_tune_preview.py",
            "--training-mode",
            training_mode,
            *_repo_args(args.arguments),
        ],
        env=env,
    )


def _command_raven_quick(args: argparse.Namespace) -> int:
    return _command_workflow(args, "quick")


def _command_full_train(args: argparse.Namespace) -> int:
    return _command_workflow(args, "full")


def _command_index_eve(args: argparse.Namespace) -> int:
    forwarded = ["--config", args.config]
    if args.shared_cache:
        forwarded += ["--shared-cache", args.shared_cache]
    if args.source_root:
        forwarded += ["--source-root", args.source_root]
    if args.rebuild:
        forwarded.append("--rebuild")
    if args.max_families is not None:
        forwarded += ["--max-families", str(args.max_families)]
    if args.crops_per_family is not None:
        forwarded += ["--crops-per-family", str(args.crops_per_family)]
    forwarded += args.arguments
    return _python_script(
        "tools/nsamdr/neural/index_eve_texture_dataset_v9.py",
        _repo_args(forwarded),
    )


def _command_index_raven(args: argparse.Namespace) -> int:
    forwarded = [
        "--config", args.config,
        "--shared-cache", args.shared_cache,
        "--train-crops", str(args.train_crops),
        "--validation-crops", str(args.validation_crops),
    ]
    if args.rebuild:
        forwarded.append("--rebuild")
    forwarded += args.arguments
    return _python_script(
        "tools/nsamdr/neural/prepare_nsamdr_v9_raven_preview_dataset.py",
        _repo_args(forwarded),
    )


def _cuda_preflight(python: Path, env: dict[str, str]) -> int:
    return _run_code(
        [python, "-u", NEURAL_ROOT / "verify_cuda.py", "--require-arch", "sm_120", "--quick"],
        env=env,
    )


def _command_preview_experiment(args: argparse.Namespace) -> int:
    forwarded = ["--experiment", args.experiment, *args.arguments]
    return _python_script(
        "tools/nsamdr/neural/preview_nsamdr_v9_experiment.py",
        _repo_args(forwarded),
    )


def _python_script_with_env(relative: str, arguments: Sequence[str], *, env: dict[str, str]) -> int:
    return _run_code([_python("cuda"), "-u", REPO_ROOT / relative, *arguments], env=env)


def _command_preview(args: argparse.Namespace) -> int:
    forwarded = list(args.arguments)
    for option, attribute in (
        ("--shared-cache", "shared_cache"),
        ("--target-size", "target_size"),
        ("--device", "device"),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            forwarded += [option, str(value)]
    args.arguments = forwarded

    experiment = str(args.subject or "").strip()
    if not experiment:
        raise ValueError("preview requires a completed EXP_####")
    args.experiment = experiment
    return _command_preview_experiment(args)


def validate_layout() -> int:
    missing = [relative for relative in REQUIRED_LAYOUT if not (REPO_ROOT / relative).is_file()]
    if missing:
        print("ERROR: Missing canonical NSAMDR source/layout requirement:", file=sys.stderr)
        for relative in missing:
            print(f"  {relative}", file=sys.stderr)
        return 4
    print("Canonical NSAMDR source layout verified successfully.", flush=True)
    print("  unified dispatcher        : present", flush=True)
    print("  staged Raven workflow     : present", flush=True)
    print("  immutable experiments     : present", flush=True)
    print("  native DX11 OBJ preview   : present", flush=True)
    return 0


def _command_validate(args: argparse.Namespace) -> int:
    code = _reject_arguments(args, "validate")
    if code:
        return code
    code = validate_layout()
    if code or args.layout_only:
        return code
    code = _command_test(argparse.Namespace(test_name="contract", config=None, arguments=[]))
    if code:
        return code
    return _command_test(
        argparse.Namespace(test_name="architecture", device=args.device, config=None, arguments=[])
    )


def _command_test(args: argparse.Namespace) -> int:
    if args.test_name == "contract":
        code = _reject_arguments(args, "test contract")
        if code:
            return code
        return _python_script("tools/nsamdr/neural/test_nsamdr_v9_contract.py", [])
    if args.test_name == "architecture":
        if args.device == "cpu" and not os.environ.get("NSAMDR_PYTHON_EXE"):
            if CPU_PYTHON.is_file():
                python = CPU_PYTHON
            elif CUDA_PYTHON.is_file():
                python = CUDA_PYTHON
            else:
                python = _python("cpu")
        else:
            python = _python("cuda")
        env = os.environ.copy()
        if args.device == "cuda":
            code = _cuda_preflight(python, env)
            if code:
                return code
        return _run_code(
            [python, NEURAL_ROOT / "smoke_test_nsamdr_v9.py", "--device", args.device, *args.arguments],
            env=env,
        )
    forwarded = _repo_args((["--config", args.config] if args.config else []) + args.arguments)
    return _python_script("tools/nsamdr/neural/test_nsamdr_v9_checkpoint.py", forwarded)


def _safe_resolved_target(target: Path) -> Path:
    root = REPO_ROOT.resolve()
    resolved = target.resolve()
    resolved.relative_to(root)
    if resolved == root:
        raise ValueError("refusing to clean the repository root")
    return resolved


def _safe_target(relative: str) -> Path:
    return _safe_resolved_target(REPO_ROOT / relative)


def _command_cleanup(args: argparse.Namespace) -> int:
    code = _reject_arguments(args, "cleanup")
    if code:
        return code
    targets: list[tuple[Path, str]] = []
    if args.all_artifacts:
        args.tuning_dataset = args.experiments = True
        args.dataset = args.production = True
    mapping = (
        (args.tuning_dataset, "artifacts/nsamdr/training_v9_preview_raven", "Raven tuning dataset"),
        (args.experiments, "artifacts/nsamdr/experiments", "experiments"),
        (args.dataset, "artifacts/nsamdr/training_v9", "production dataset"),
        (args.production, "artifacts/nsamdr/neural_v9", "production checkpoint/state"),
    )
    targets.extend((_safe_target(relative), label) for selected, relative, label in mapping if selected)
    for cache_root in (TOOLS_ROOT, REPO_ROOT / "trinityal" / "tests" / "nsamdr"):
        if cache_root.is_dir():
            targets.extend(
                (_safe_resolved_target(path), "Python cache")
                for path in cache_root.rglob("__pycache__")
            )
            targets.extend(
                (_safe_resolved_target(path), "Python bytecode")
                for path in cache_root.rglob("*.pyc")
            )
    seen: set[Path] = set()
    for target, label in targets:
        if target in seen:
            continue
        seen.add(target)
        if not target.exists():
            continue
        action = "would remove" if args.dry_run else "removed"
        print(f"[nsamdr] {action}: {label}: {target}", flush=True)
        if args.dry_run:
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    return 0


def _command_integrate(args: argparse.Namespace) -> int:
    forwarded = ["--repo-root", os.fspath(REPO_ROOT)]
    if args.check:
        forwarded.append("--check")
    forwarded += args.arguments
    return _python_script("tools/nsamdr/integration/apply_trinity_nsamdr_settings.py", forwarded)


def _command_native(args: argparse.Namespace) -> int:
    launcher = BUILD_ROOT / "run_nsamdr_obj_preview_dx11.bat"
    env = os.environ.copy()
    if args.native_name == "build":
        code = _reject_arguments(args, "native build")
        if code:
            return code
        env["NSAMDR_BUILD_ONLY"] = "1"
        command_args: list[str] = []
    elif args.native_name == "obj":
        command_args = args.arguments
    else:
        return _python_script_with_env(
            "tools/nsamdr/eve_asset_test.py",
            [
                "prepare-run",
                "--repo-root", os.fspath(REPO_ROOT),
                "--shared-cache", args.shared_cache,
                "--query", args.query,
                "--launcher", os.fspath(launcher),
                *args.arguments,
            ],
            env=env,
        )
    return _run_batch(launcher, command_args, env=env)


def _set_handler(parser: argparse.ArgumentParser, handler: Callable[[argparse.Namespace], int]) -> None:
    parser.set_defaults(handler=handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nsamdr",
        description="Canonical NSAMDR training, qualification, and preview commands",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    gui = commands.add_parser("gui", help="open the production workflow controller")
    _set_handler(gui, _command_gui)

    setup = commands.add_parser("setup", help="create or verify a Python environment")
    setup.add_argument("kind", choices=("cuda", "cpu"))
    setup.add_argument("--force", action="store_true")
    _set_handler(setup, _command_setup)

    raven_quick = commands.add_parser(
        "raven-quick",
        help="train and qualify the complete production model on the small Raven dataset",
    )
    _set_handler(raven_quick, _command_raven_quick)

    full_train = commands.add_parser(
        "full-train",
        help="train and qualify the same production model on the full production dataset",
    )
    _set_handler(full_train, _command_full_train)

    index = commands.add_parser("index", help="prepare an NSAMDR dataset")
    index_commands = index.add_subparsers(dest="index_name", required=True)
    eve = index_commands.add_parser("eve", help="index the full authored EVE dataset")
    eve.add_argument("--config", default=DEFAULT_FULL_CONFIG)
    source = eve.add_mutually_exclusive_group()
    source.add_argument("--shared-cache")
    source.add_argument("--source-root")
    eve.add_argument("--rebuild", action="store_true")
    eve.add_argument("--max-families", type=int)
    eve.add_argument("--crops-per-family", type=int)
    _set_handler(eve, _command_index_eve)
    raven = index_commands.add_parser("raven", help="build the fixed Raven tuning dataset")
    raven.add_argument("--config", default=DEFAULT_RAVEN_CONFIG)
    raven.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    raven.add_argument("--train-crops", type=int, default=16)
    raven.add_argument("--validation-crops", type=int, default=4)
    raven.add_argument("--rebuild", action="store_true")
    _set_handler(raven, _command_index_raven)

    preview = commands.add_parser("preview", help="preview one completed, qualified EXP_####")
    preview.add_argument("subject", help="completed EXP_####")
    preview.add_argument("--shared-cache")
    preview.add_argument("--target-size", type=int)
    preview.add_argument("--device", choices=("cuda", "cpu", "auto"))
    _set_handler(preview, _command_preview)

    validate = commands.add_parser("validate", help="validate layout, contract and architecture")
    validate.add_argument("--layout-only", action="store_true")
    validate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    _set_handler(validate, _command_validate)

    test = commands.add_parser("test", help="run an NSAMDR verification suite")
    test_commands = test.add_subparsers(dest="test_name", required=True)
    contract = test_commands.add_parser("contract", help="run the production semantic contract")
    _set_handler(contract, _command_test)
    architecture = test_commands.add_parser("architecture", help="run the production architecture smoke test")
    architecture.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    _set_handler(architecture, _command_test)
    checkpoint = test_commands.add_parser("checkpoint", help="validate a production checkpoint")
    checkpoint.add_argument("--config")
    _set_handler(checkpoint, _command_test)

    cleanup = commands.add_parser("cleanup", help="remove caches or selected NSAMDR artifacts")
    cleanup.add_argument("--tuning-dataset", action="store_true")
    cleanup.add_argument("--experiments", action="store_true")
    cleanup.add_argument("--dataset", action="store_true")
    cleanup.add_argument("--production", action="store_true")
    cleanup.add_argument("--all-artifacts", action="store_true")
    cleanup.add_argument("--dry-run", action="store_true")
    _set_handler(cleanup, _command_cleanup)

    integrate = commands.add_parser("integrate", help="apply or verify the Trinity graphics-setting override")
    integrate.add_argument("--check", action="store_true")
    _set_handler(integrate, _command_integrate)

    native = commands.add_parser("native", help="build or run the specialized native DX11 preview")
    native_commands = native.add_subparsers(dest="native_name", required=True)
    native_build = native_commands.add_parser("build", help="build the native preview only")
    _set_handler(native_build, _command_native)
    native_obj = native_commands.add_parser("obj", help="preview a local OBJ or GR2 asset")
    native_obj.add_argument("arguments", nargs=argparse.REMAINDER, help="model and optional texture arguments")
    _set_handler(native_obj, _command_native)
    native_eve = native_commands.add_parser("eve", help="extract and preview an EVE asset")
    native_eve.add_argument("--shared-cache", default="")
    native_eve.add_argument("--query", default="res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2")
    _set_handler(native_eve, _command_native)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    forwarded = list(getattr(args, "arguments", []))
    forwarded.extend(unknown)
    args.arguments = forwarded
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - command boundary reports actionable failure
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
