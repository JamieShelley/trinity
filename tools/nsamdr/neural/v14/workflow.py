from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch

from .config import V14Config
from .trainer import V14Trainer


def _next_experiment(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    values = []
    for path in root.glob("EXP_*"):
        try:
            values.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return f"EXP_{max(values, default=0)+1:04d}"


def _prepare_dataset(args: argparse.Namespace, repo_root: Path) -> None:
    script = repo_root / "tools/nsamdr/neural/prepare_nsamdr_v9_raven_preview_dataset.py"
    command = [
        sys.executable, "-u", str(script),
        "--repo-root", str(repo_root),
        "--shared-cache", args.shared_cache,
        "--train-crops", str(args.max_train_regions),
        "--validation-crops", str(args.max_validation_regions),
    ]
    if args.rebuild_dataset:
        command.append("--rebuild")
    print("[v14-workflow] prepare authored Raven dataset: " + subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=repo_root, check=False)
    if completed.returncode:
        raise RuntimeError(f"Raven dataset preparation failed with exit code {completed.returncode}")


def _diagnostics(experiment: Path, root: Path) -> Path:
    diagnostics = root / "artifacts/nsamdr/diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    base = diagnostics / f"{experiment.name}_V14_DIAGNOSTICS"
    archive = Path(shutil.make_archive(str(base), "zip", root_dir=experiment))
    return archive


def _start_live_view(repo_root: Path, experiment: Path) -> subprocess.Popen[bytes]:
    script = repo_root / "tools/nsamdr/neural/v14/live_view.py"
    command = [sys.executable, "-u", str(script), "--experiment-dir", str(experiment)]
    print("[v14-workflow] live A/B/C/F viewer: " + subprocess.list2cmdline(command), flush=True)
    return subprocess.Popen(command, cwd=repo_root)


def _stop_live_view(experiment: Path, process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    stop = experiment / "previews" / "live" / "viewer.stop"
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_text("stop\n", encoding="utf-8")
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.terminate()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSAMDR V14 HR-first Raven workflow")
    p.add_argument("--training-mode", choices=("quick", "full"), default="quick")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    p.add_argument("--max-train-regions", type=int, default=16)
    p.add_argument("--max-validation-regions", type=int, default=4)
    p.add_argument("--experiment", default="new")
    p.add_argument("--control", default="auto")
    p.add_argument("--preview-target-size", type=int, default=4096)
    p.add_argument("--preview-device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--performance-profile", default="fast")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    p.add_argument("--live-preview-during-training", action="store_true")
    p.add_argument("--live-preview-target-size", type=int, default=1024)
    p.add_argument("--rebuild-dataset", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.training_mode != "quick":
        print("ERROR: V14 Full Training remains disabled until Raven HR-first candidate qualifies.", file=sys.stderr)
        return 2
    _prepare_dataset(args, repo_root)

    experiments = repo_root / "artifacts/nsamdr/experiments"
    experiment_id = _next_experiment(experiments) if args.experiment.strip().lower() == "new" else args.experiment.strip().upper()
    experiment = experiments / experiment_id
    if experiment.exists() and any(experiment.iterdir()):
        raise RuntimeError(f"V14 experiments are immutable; choose new instead of reusing {experiment_id}")
    experiment.mkdir(parents=True, exist_ok=True)

    config = V14Config()
    config.validate()
    config.save(experiment / "resolved_config.json")
    manifest = {
        "schema": "NSAMDR_V14_EXPERIMENT_V1",
        "experiment": experiment_id,
        "status": "running",
        "qualified": False,
        "modelSchema": config.schema,
        "trainingMode": args.training_mode,
        "createdUnix": time.time(),
        "source": "V14 HR-first clean architecture",
    }
    manifest_path = experiment / "experiment.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    device = torch.device("cuda" if args.preview_device in {"cuda", "auto"} and torch.cuda.is_available() else "cpu")
    if args.preview_device == "cuda" and device.type != "cuda":
        raise RuntimeError("V14 Raven Quick requested CUDA but CUDA is unavailable")

    viewer: subprocess.Popen[bytes] | None = None
    if args.live_preview_during_training:
        viewer = _start_live_view(repo_root, experiment)

    try:
        trainer = V14Trainer(
            repo_root,
            experiment,
            config,
            device=device,
            workers=args.workers,
            prefetch_factor=args.prefetch_factor,
            amp_precision=args.amp_precision,
            live_preview_target_size=args.live_preview_target_size if args.live_preview_during_training else 0,
        )
        result = trainer.run()
        manifest.update(result)
        manifest["status"] = result["status"]
        manifest["qualified"] = bool(result.get("qualified"))
        manifest["finishedUnix"] = time.time()
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["qualified"] = False
        manifest["finishedUnix"] = time.time()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (experiment / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        archive = _diagnostics(experiment, repo_root)
        print(f"[v14-workflow] FAILED diagnostics: {archive}", flush=True)
        raise
    finally:
        _stop_live_view(experiment, viewer)

    archive = _diagnostics(experiment, repo_root)
    print(f"[v14-workflow] diagnostics: {archive}", flush=True)
    if result["qualified"]:
        print(f"[v14-workflow] QUALIFIED {experiment_id}", flush=True)
        return 0
    print(f"[v14-workflow] REJECTED {experiment_id}: {result.get('failedStage')}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
