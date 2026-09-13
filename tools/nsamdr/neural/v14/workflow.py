from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

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
    (experiment / "experiment.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    device = torch.device("cuda" if args.preview_device in {"cuda", "auto"} and torch.cuda.is_available() else "cpu")
    if args.preview_device == "cuda" and device.type != "cuda":
        raise RuntimeError("V14 Raven Quick requested CUDA but CUDA is unavailable")

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
    (experiment / "experiment.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    archive = _diagnostics(experiment, repo_root)
    print(f"[v14-workflow] diagnostics: {archive}", flush=True)
    if result["qualified"]:
        print(f"[v14-workflow] QUALIFIED {experiment_id}", flush=True)
        return 0
    print(f"[v14-workflow] REJECTED {experiment_id}: {result.get('failedStage')}", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
