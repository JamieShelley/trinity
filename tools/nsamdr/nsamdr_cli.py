#!/usr/bin/env python3
"""Unified command surface for the active NSAMDR V9.8.3 workflow."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
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
    "tools/nsamdr/neural/compare_nsamdr_v9_experiments.py",
    "tools/nsamdr/neural/promote_nsamdr_v9_experiment.py",
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
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    values = [os.fspath(value) for value in command]
    print("[nsamdr] RUN: " + _display(values), flush=True)
    return subprocess.run(
        values,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
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


def _command_tune(args: argparse.Namespace) -> int:
    python = _python("cuda")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.80")
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
            *_repo_args(args.arguments),
        ],
        env=env,
    )


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


def _command_train_preview(args: argparse.Namespace) -> int:
    python = _python("cuda")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.80")
    code = _cuda_preflight(python, env)
    if code:
        return code
    return _run_code(
        [python, "-u", NEURAL_ROOT / "train_nsamdr_v9_preview_experiment.py", *_repo_args(args.arguments)],
        env=env,
    )


def _command_train_full(args: argparse.Namespace) -> int:
    python = _python("cuda")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.80")
    code = _cuda_preflight(python, env)
    if code:
        return 5

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        print(f"ERROR: Missing V9 config: {config_path}", file=sys.stderr)
        return 3

    if not args.skip_dataset:
        index_args = argparse.Namespace(
            config=args.config,
            shared_cache=args.shared_cache,
            source_root=args.source_root,
            rebuild=args.rebuild_dataset,
            max_families=None,
            crops_per_family=None,
            arguments=[],
        )
        code = _command_index_eve(index_args)
        if code:
            return 6

    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"ERROR: Cannot read V9 config {config_path}: {exc}", file=sys.stderr)
        return 3
    manifest_value = str(
        config_payload.get("datasetManifest")
        or config_payload.get("dataset_manifest")
        or "artifacts/nsamdr/training_v9/dataset_manifest.json"
    )
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not manifest_path.is_file():
        print(f"ERROR: V9 dataset manifest missing: {manifest_path}", file=sys.stderr)
        return 7

    forwarded = ["--config", args.config, "--device", "cuda"]
    forwarded.append("--" + args.control)
    if args.performance_profile:
        forwarded += ["--performance-profile", args.performance_profile]
    if args.tuning_file:
        forwarded += ["--tuning-file", args.tuning_file]
    if args.loss_precision:
        forwarded += ["--loss-precision", args.loss_precision]
    if args.torch_compile:
        forwarded += ["--torch-compile", args.torch_compile]
    if args.channels_last:
        forwarded += ["--channels-last", args.channels_last]
    if args.optimizer_kernel:
        forwarded += ["--optimizer-kernel", args.optimizer_kernel]
    if args.workers is not None:
        forwarded += ["--workers", str(args.workers)]
    if args.prefetch_factor is not None:
        forwarded += ["--prefetch-factor", str(args.prefetch_factor)]
    if args.amp_precision:
        forwarded += ["--amp-precision", args.amp_precision]
    forwarded += args.arguments
    code = _run_code(
        [python, "-u", NEURAL_ROOT / "train_nsamdr_v9.py", *_repo_args(forwarded)],
        env=env,
    )
    if code:
        return 20
    code = _run_code(
        [python, NEURAL_ROOT / "test_nsamdr_v9_checkpoint.py", *_repo_args(["--config", args.config])],
        env=env,
    )
    if code:
        return 21
    for candidate_dir in (REPO_ROOT / "artifacts" / "nsamdr" / "eve_assets").glob(
        "**/strategy_candidates_4096"
    ):
        if candidate_dir.is_dir():
            shutil.rmtree(candidate_dir)
    print("[nsamdr] V9 checkpoint ready; candidate caches invalidated.", flush=True)
    return 0


def _command_preview_experiment(args: argparse.Namespace) -> int:
    forwarded = ["--experiment", args.experiment, *args.arguments]
    return _python_script(
        "tools/nsamdr/neural/preview_nsamdr_v9_experiment.py",
        _repo_args(forwarded),
    )


def _preview_production(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-dir", default="artifacts/nsamdr/neural_v9")
    parser.add_argument("--shared-cache", default="")
    parser.add_argument("--preview-strength", default="1.0")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--target-size", type=int, default=4096)
    parser.add_argument("--force-candidate", action="store_true")
    known, remainder = parser.parse_known_args(list(arguments))
    checkpoint_dir = (REPO_ROOT / known.checkpoint_dir).resolve()
    checkpoint = checkpoint_dir / "nsamdr_v9_fidelity.pt"
    if not checkpoint.is_file():
        print(f"ERROR: Missing V9 checkpoint: {checkpoint}", file=sys.stderr)
        return 2
    env = os.environ.copy()
    env.update(
        {
            "NSAMDR_V9_PREVIEW_STRENGTH": known.preview_strength,
            "NSAMDR_NEURAL_ARCHITECTURE": "V9",
            "NSAMDR_NEURAL_CHECKPOINT_DIR": os.fspath(checkpoint_dir),
            "NSAMDR_INFERENCE_DEVICE": known.device,
            "NSAMDR_MODE3_TARGET_SIZE": str(known.target_size),
            "NSAMDR_MODE3_CANDIDATE_TAG": "v9",
        }
    )
    if known.force_candidate:
        env["NSAMDR_FORCE_CANDIDATE"] = "1"
    return _python_script_with_env(
        "tools/nsamdr/eve_asset_test.py",
        [
            "prepare-run",
            "--repo-root", os.fspath(REPO_ROOT),
            "--shared-cache", known.shared_cache,
            "--query", "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2",
            "--neural-checkpoint-dir", os.fspath(checkpoint_dir),
            "--launcher", os.fspath(BUILD_ROOT / "run_nsamdr_obj_preview_dx11.bat"),
            *remainder,
        ],
        env=env,
    )


def _python_script_with_env(relative: str, arguments: Sequence[str], *, env: dict[str, str]) -> int:
    return _run_code([_python("cuda"), "-u", REPO_ROOT / relative, *arguments], env=env)


def _command_preview(args: argparse.Namespace) -> int:
    forwarded = list(args.arguments)
    for option, attribute in (
        ("--shared-cache", "shared_cache"),
        ("--target-size", "target_size"),
        ("--device", "device"),
        ("--checkpoint-dir", "checkpoint_dir"),
        ("--preview-strength", "preview_strength"),
        ("--geometry-critic", "geometry_critic"),
        ("--geometry-audit-policy", "geometry_audit_policy"),
        ("--geometry-evidence-regions", "geometry_evidence_regions"),
        ("--critic-checkpoint", "critic_checkpoint"),
    ):
        value = getattr(args, attribute, None)
        if value is not None:
            forwarded += [option, str(value)]
    if getattr(args, "force_candidate", False):
        forwarded.append("--force-candidate")
    geometry_audit = getattr(args, "geometry_audit", None)
    if geometry_audit is not None:
        forwarded.append("--geometry-audit" if geometry_audit else "--no-geometry-audit")
    args.arguments = forwarded

    subject = str(args.subject or "").strip()
    if subject.lower() == "production":
        if args.experiment:
            raise ValueError("preview production does not accept an experiment id")
        return _preview_production(args.arguments)
    if subject.lower() == "experiment":
        experiment = args.experiment
        if not experiment and args.arguments and not args.arguments[0].startswith("-"):
            experiment = args.arguments.pop(0)
    else:
        experiment = subject
        if args.experiment:
            raise ValueError("use either 'preview EXP_####' or 'preview experiment EXP_####'")
    if not experiment:
        raise ValueError("preview requires EXP_#### or the 'production' target")
    args.experiment = experiment
    return _command_preview_experiment(args)


def _command_candidate(args: argparse.Namespace) -> int:
    asset_dir = REPO_ROOT / "artifacts" / "nsamdr" / "eve_assets" / args.asset_id
    forwarded = [
        "--obj", os.fspath(asset_dir / f"{args.asset_id}.obj"),
        "--materials", os.fspath(asset_dir / "ship.materials.tsv"),
        "--asset-manifest", os.fspath(asset_dir / "asset_manifest.json"),
        "--output-root", os.fspath(asset_dir / f"strategy_candidates_{args.target_size}"),
        "--target-size", str(args.target_size),
        "--super-resolution-backend", args.super_resolution_backend,
        "--inference-device", args.inference_device,
    ]
    if args.install_dependencies:
        forwarded.append("--install-dependencies")
    if args.force:
        forwarded.append("--force")
    if args.checkpoint_dir:
        forwarded += ["--checkpoint-dir", args.checkpoint_dir]
    forwarded += args.arguments
    return _python_script("tools/nsamdr/generate_strategy_candidates.py", forwarded)


def _command_compare(args: argparse.Namespace) -> int:
    forwarded = ["--experiments", *args.experiments]
    if args.no_open:
        forwarded.append("--no-open")
    forwarded += args.arguments
    return _python_script(
        "tools/nsamdr/neural/compare_nsamdr_v9_experiments.py",
        _repo_args(forwarded),
    )


def _command_promote(args: argparse.Namespace) -> int:
    forwarded = ["--experiment", args.experiment]
    if args.full_base_config:
        forwarded += ["--full-base-config", args.full_base_config]
    forwarded += args.arguments
    return _python_script(
        "tools/nsamdr/neural/promote_nsamdr_v9_experiment.py",
        _repo_args(forwarded),
    )


def validate_layout() -> int:
    missing = [relative for relative in REQUIRED_LAYOUT if not (REPO_ROOT / relative).is_file()]
    if missing:
        print("ERROR: Missing NSAMDR V9 source/layout capability:", file=sys.stderr)
        for relative in missing:
            print(f"  {relative}", file=sys.stderr)
        return 4
    print("NSAMDR V9 source layout verified successfully.", flush=True)
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
        args.tuning_dataset = args.experiments = args.promotion = True
        args.dataset = args.pilot = args.stability = args.production = True
    mapping = (
        (args.tuning_dataset, "artifacts/nsamdr/training_v9_preview_raven", "Raven tuning dataset"),
        (args.experiments, "artifacts/nsamdr/experiments", "tuning experiments"),
        (args.promotion, "artifacts/nsamdr/promoted", "promoted configuration"),
        (args.pilot, "artifacts/nsamdr/neural_v9_pilot", "pilot checkpoint/state"),
        (args.dataset, "artifacts/nsamdr/training_v9", "production dataset"),
        (args.stability, "artifacts/nsamdr/neural_v9_stability", "stability checkpoint/state"),
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


def _promoted_config() -> str | None:
    completed = _run(
        [
            _python("cuda"),
            "-u",
            NEURAL_ROOT / "promote_nsamdr_v9_experiment.py",
            "--repo-root", REPO_ROOT,
            "--print-selected-config",
        ],
        capture=True,
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)
    lines = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
    return lines[-1] if lines else None


def _command_run(args: argparse.Namespace) -> int:
    code = _reject_arguments(args, "run")
    if code:
        return code
    code = _command_validate(
        argparse.Namespace(layout_only=False, device="cuda", arguments=[])
    )
    if code:
        return code
    config = _promoted_config()
    config_path = Path(config) if config else None
    if config_path is not None and not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if config_path is None or not config_path.is_file():
        print(
            "ERROR: No promoted V9 tuning configuration is selected. "
            "Complete and promote a Full Raven proof first.",
            file=sys.stderr,
        )
        return 1
    config = os.fspath(config_path.resolve())
    index_args = argparse.Namespace(
        config=config,
        shared_cache=args.shared_cache,
        source_root=None,
        rebuild=False,
        max_families=None,
        crops_per_family=None,
        arguments=[],
    )
    code = _command_index_eve(index_args)
    if code:
        return code
    train_args = argparse.Namespace(
        config=config,
        shared_cache=args.shared_cache,
        source_root=None,
        rebuild_dataset=False,
        skip_dataset=True,
        control="auto",
        performance_profile="fast",
        tuning_file=None,
        loss_precision=None,
        torch_compile=None,
        channels_last=None,
        optimizer_kernel=None,
        workers=8,
        prefetch_factor=2,
        amp_precision="auto",
        arguments=[],
    )
    code = _command_train_full(train_args)
    if code:
        return code
    return _preview_production(["--checkpoint-dir", "artifacts/nsamdr/neural_v9", "--force-candidate"])


def _command_retrain_preview(args: argparse.Namespace) -> int:
    code = _reject_arguments(args, "retrain-preview")
    if code:
        return code
    if args.wait_pid:
        while _pid_exists(args.wait_pid):
            time.sleep(0.5)
    train_args = argparse.Namespace(
        config=args.config,
        shared_cache=args.shared_cache,
        source_root=None,
        rebuild_dataset=False,
        skip_dataset=True,
        control="restart",
        performance_profile="fast",
        tuning_file=None,
        loss_precision=None,
        torch_compile=None,
        channels_last=None,
        optimizer_kernel=None,
        workers=8,
        prefetch_factor=2,
        amp_precision="auto",
        arguments=[],
    )
    code = _command_train_full(train_args)
    if code:
        return code
    return _preview_production(["--checkpoint-dir", "artifacts/nsamdr/neural_v9", "--force-candidate"])


def _pid_exists(pid: int) -> bool:
    completed = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return False
    return any(
        len(row) >= 2 and row[1].strip() == str(pid)
        for row in csv.reader(completed.stdout.splitlines())
    )


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
        description="NSAMDR V9.8.3 build, training, experiment and preview commands",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    gui = commands.add_parser("gui", help="open the V9 workflow controller")
    _set_handler(gui, _command_gui)

    setup = commands.add_parser("setup", help="create or verify a Python environment")
    setup.add_argument("kind", choices=("cuda", "cpu"))
    setup.add_argument("--force", action="store_true")
    _set_handler(setup, _command_setup)

    tune = commands.add_parser("tune", help="run the staged Raven tune and preview workflow")
    _set_handler(tune, _command_tune)

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
    raven.add_argument("--train-crops", type=int, default=12)
    raven.add_argument("--validation-crops", type=int, default=4)
    raven.add_argument("--rebuild", action="store_true")
    _set_handler(raven, _command_index_raven)

    train = commands.add_parser("train", help="train an NSAMDR model")
    train_commands = train.add_subparsers(dest="train_name", required=True)
    preview_train = train_commands.add_parser("preview", help="train one fixed-Raven experiment")
    _set_handler(preview_train, _command_train_preview)
    full_train = train_commands.add_parser("full", help="index and train the promoted production model")
    full_train.add_argument("--config", default=DEFAULT_FULL_CONFIG)
    source = full_train.add_mutually_exclusive_group()
    source.add_argument("--shared-cache")
    source.add_argument("--source-root")
    full_train.add_argument("--rebuild-dataset", action="store_true")
    full_train.add_argument("--skip-dataset", action="store_true")
    full_train.add_argument("--control", choices=("auto", "resume", "restart"), default="auto")
    full_train.add_argument("--auto", dest="control", action="store_const", const="auto")
    full_train.add_argument("--resume", dest="control", action="store_const", const="resume")
    full_train.add_argument("--restart", dest="control", action="store_const", const="restart")
    full_train.add_argument("--performance-profile")
    full_train.add_argument("--tuning-file")
    full_train.add_argument("--loss-precision")
    full_train.add_argument("--torch-compile")
    full_train.add_argument("--channels-last")
    full_train.add_argument("--optimizer-kernel")
    full_train.add_argument("--workers", type=int)
    full_train.add_argument("--prefetch-factor", type=int)
    full_train.add_argument("--amp-precision")
    _set_handler(full_train, _command_train_full)

    preview = commands.add_parser("preview", help="launch an experiment or production preview")
    preview.add_argument("subject", help="EXP_####, 'experiment', or 'production'")
    preview.add_argument("experiment", nargs="?", help="EXP_#### after the optional 'experiment' word")
    preview.add_argument("--shared-cache")
    preview.add_argument("--target-size", type=int)
    preview.add_argument("--device", choices=("cuda", "cpu", "auto"))
    preview.add_argument("--checkpoint-dir")
    preview.add_argument("--preview-strength")
    preview.add_argument("--force-candidate", action="store_true")
    preview.add_argument("--geometry-audit", dest="geometry_audit", action="store_true", default=None)
    preview.add_argument("--no-geometry-audit", dest="geometry_audit", action="store_false")
    preview.add_argument("--geometry-critic", choices=("off", "auto", "required"))
    preview.add_argument("--geometry-audit-policy", choices=("report", "strict"))
    preview.add_argument("--geometry-evidence-regions", type=int)
    preview.add_argument("--critic-checkpoint")
    _set_handler(preview, _command_preview)

    candidate = commands.add_parser("candidate", help="generate a V9 strategy candidate for an extracted asset")
    candidate.add_argument("asset_id", nargs="?", default="cb1_t1")
    candidate.add_argument("--target-size", type=int, default=4096)
    candidate.add_argument("--super-resolution-backend", choices=("auto", "classic", "realesrgan"), default="auto")
    candidate.add_argument("--inference-device", choices=("auto", "cuda", "cpu"), default="auto")
    candidate.add_argument("--install-dependencies", action="store_true")
    candidate.add_argument("--force", action="store_true")
    candidate.add_argument("--checkpoint-dir")
    _set_handler(candidate, _command_candidate)

    compare = commands.add_parser("compare", help="compare two or three completed experiments")
    compare.add_argument("experiments", nargs="+", metavar="EXP_####")
    compare.add_argument("--no-open", action="store_true")
    _set_handler(compare, _command_compare)

    promote = commands.add_parser("promote", help="promote a completed Full experiment")
    promote.add_argument("experiment")
    promote.add_argument("--full-base-config")
    _set_handler(promote, _command_promote)

    validate = commands.add_parser("validate", help="validate layout, contract and architecture")
    validate.add_argument("--layout-only", action="store_true")
    validate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    _set_handler(validate, _command_validate)

    test = commands.add_parser("test", help="run an NSAMDR test capability")
    test_commands = test.add_subparsers(dest="test_name", required=True)
    contract = test_commands.add_parser("contract", help="run the V9.8.3 semantic contract")
    _set_handler(contract, _command_test)
    architecture = test_commands.add_parser("architecture", help="run the V9 architecture smoke test")
    architecture.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    _set_handler(architecture, _command_test)
    checkpoint = test_commands.add_parser("checkpoint", help="validate a V9 checkpoint")
    checkpoint.add_argument("--config")
    _set_handler(checkpoint, _command_test)

    cleanup = commands.add_parser("cleanup", help="remove caches or selected V9 artifacts")
    cleanup.add_argument("--tuning-dataset", action="store_true")
    cleanup.add_argument("--experiments", action="store_true")
    cleanup.add_argument("--promotion", action="store_true")
    cleanup.add_argument("--pilot", action="store_true")
    cleanup.add_argument("--dataset", action="store_true")
    cleanup.add_argument("--stability", action="store_true")
    cleanup.add_argument("--production", action="store_true")
    cleanup.add_argument("--all-artifacts", action="store_true")
    cleanup.add_argument("--dry-run", action="store_true")
    _set_handler(cleanup, _command_cleanup)

    integrate = commands.add_parser("integrate", help="apply or verify the Trinity graphics-setting override")
    integrate.add_argument("--check", action="store_true")
    _set_handler(integrate, _command_integrate)

    run = commands.add_parser("run", help="run the promoted production pipeline")
    run.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    _set_handler(run, _command_run)

    retrain = commands.add_parser("retrain-preview", help="restart full training, then open the production preview")
    retrain.add_argument("--config", default=DEFAULT_FULL_CONFIG)
    retrain.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    retrain.add_argument("--wait-pid", type=int)
    _set_handler(retrain, _command_retrain_preview)

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
