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

if __package__ in {None, ""}:
    NEURAL_ROOT = Path(__file__).resolve().parent.parent
    if str(NEURAL_ROOT) not in sys.path:
        sys.path.insert(0, str(NEURAL_ROOT))
    from v14.config import V16Config
    from v14.model import MODEL_SCHEMA
    from v14.trainer import V14Trainer
else:
    from .config import V16Config
    from .model import MODEL_SCHEMA
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
        sys.executable,
        "-u",
        str(script),
        "--repo-root",
        str(repo_root),
        "--shared-cache",
        args.shared_cache,
        "--train-crops",
        str(args.max_train_regions),
        "--validation-crops",
        str(args.max_validation_regions),
    ]
    if args.rebuild_dataset:
        command.append("--rebuild")
    print(
        "[v16.0-workflow] prepare authored Raven dataset: "
        + subprocess.list2cmdline(command),
        flush=True,
    )
    completed = subprocess.run(command, cwd=repo_root, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"Raven dataset preparation failed with exit code {completed.returncode}"
        )


def _diagnostics(experiment: Path, root: Path) -> Path:
    diagnostics = root / "artifacts/nsamdr/diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    base = diagnostics / f"{experiment.name}_V16_0_DIAGNOSTICS"
    return Path(shutil.make_archive(str(base), "zip", root_dir=experiment))


def _start_live_view(
    repo_root: Path,
    experiment: Path,
) -> subprocess.Popen[bytes]:
    script = repo_root / "tools/nsamdr/neural/v14/live_view.py"
    command = [
        sys.executable,
        "-u",
        str(script),
        "--experiment-dir",
        str(experiment),
    ]
    print(
        "[v16.0-workflow] live A/B/C/F viewer: "
        + subprocess.list2cmdline(command),
        flush=True,
    )
    return subprocess.Popen(command, cwd=repo_root)


def _stop_live_view(
    experiment: Path,
    process: subprocess.Popen[bytes] | None,
) -> None:
    if process is None:
        return
    stop = experiment / "previews" / "live" / "viewer.stop"
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_text("stop\n", encoding="utf-8")
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.terminate()


def _write_architecture_participation(
    experiment: Path,
    trainer: V14Trainer,
) -> None:
    contract = trainer.model.architecture_contract()
    active = tuple(contract.get("activeComponents", ()))
    retired = tuple(contract.get("retiredComponents", ()))
    expected_active = (
        "baseline",
        "context_encoder",
        "context_adapter",
        "swinir_hr_refiner",
        "albedo_tail",
        "normal_tail",
        "material_tail",
        "selector",
    )
    passed = bool(
        contract.get("schema") == MODEL_SCHEMA
        and contract.get("revision") == "V16.0"
        and contract.get("backbone") == "SwinIR-style-fixed-HR-RSTB"
        and active == expected_active
        and retired == ()
        and contract.get("geometryPixelAuthority") is False
        and contract.get("seamPixelAuthority") is False
        and contract.get("profilePixelAuthority") is False
        and contract.get("lrPhaseGridPixelAuthority") is False
        and contract.get("multiscaleFeatureHierarchy") is False
        and contract.get("batchNormalizationUsed") is False
        and contract.get("windowAttentionUsed") is True
        and contract.get("shiftedWindowAttentionUsed") is True
        and contract.get("pixelShuffleUsed") is False
        and contract.get("transposedConvolutionUsed") is False
        and contract.get("contextUpsampling")
        == "bilinear-phase-neutral + HR 3x3 adapter"
        and contract.get("decoderUpsampling") == "none"
        and contract.get("residualBounding") == "tanh"
        and contract.get("residualSupervision")
        == "bounded-pre-physical-projection"
        and int(contract.get("attentionWindowSize", -1)) == 8
        and int(contract.get("swinGroups", -1)) == 6
        and int(contract.get("swinLayersPerGroup", -1)) == 6
        and contract.get("selectorUsesPhysicalMaps") is True
    )
    payload = {
        "schema": "NSAMDR_V16_ARCHITECTURE_PARTICIPATION_V1",
        "pass": passed,
        "modelSchema": contract.get("schema"),
        "activeComponents": active,
        "retiredComponents": retired,
        "historicalRejectedComponents": contract.get(
            "historicalRejectedComponents",
            (),
        ),
        "parameterCount": sum(
            parameter.numel() for parameter in trainer.model.parameters()
        ),
        "contract": contract,
    }
    (experiment / "architecture_participation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("V16.0 architecture participation contract failed")


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NSAMDR V16.0 SwinIR-style phase-neutral HR-first Raven workflow"
    )
    parser.add_argument("--training-mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    parser.add_argument("--max-train-regions", type=int, default=16)
    parser.add_argument("--max-validation-regions", type=int, default=4)
    parser.add_argument("--experiment", default="new")
    parser.add_argument("--control", default="auto")
    parser.add_argument("--preview-target-size", type=int, default=4096)
    parser.add_argument(
        "--preview-device",
        choices=("cuda", "cpu", "auto"),
        default="cuda",
    )
    parser.add_argument("--performance-profile", default="fast")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--amp-precision",
        choices=("auto", "bf16", "fp16"),
        default="auto",
    )
    parser.add_argument("--live-preview-during-training", action="store_true")
    parser.add_argument("--live-preview-target-size", type=int, default=1024)
    parser.add_argument("--rebuild-dataset", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.training_mode != "quick":
        print(
            "ERROR: V16.0 Full Training remains disabled until Raven HR-first candidate qualifies.",
            file=sys.stderr,
        )
        return 2

    _prepare_dataset(args, repo_root)

    experiments = repo_root / "artifacts/nsamdr/experiments"
    experiment_id = (
        _next_experiment(experiments)
        if args.experiment.strip().lower() == "new"
        else args.experiment.strip().upper()
    )
    experiment = experiments / experiment_id
    if experiment.exists() and any(experiment.iterdir()):
        raise RuntimeError(
            f"V16.0 experiments are immutable; choose new instead of reusing {experiment_id}"
        )
    experiment.mkdir(parents=True, exist_ok=True)

    config = V16Config()
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
        "source": "V16.0 SwinIR-style fixed-HR phase-neutral architecture",
    }
    manifest_path = experiment / "experiment.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    use_cuda = args.preview_device in {"cuda", "auto"} and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if args.preview_device == "cuda" and device.type != "cuda":
        raise RuntimeError("V16.0 Raven Quick requested CUDA but CUDA is unavailable")

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
            live_preview_target_size=(
                args.live_preview_target_size if args.live_preview_during_training else 0
            ),
        )
        _write_architecture_participation(experiment, trainer)
        result = trainer.run()
        manifest.update(result)
        manifest["status"] = result["status"]
        manifest["qualified"] = bool(result.get("qualified"))
        manifest["finishedUnix"] = time.time()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["qualified"] = False
        manifest["finishedUnix"] = time.time()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (experiment / "failure_traceback.txt").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        archive = _diagnostics(experiment, repo_root)
        print(f"[v16.0-workflow] FAILED diagnostics: {archive}", flush=True)
        raise
    finally:
        _stop_live_view(experiment, viewer)

    archive = _diagnostics(experiment, repo_root)
    print(f"[v16.0-workflow] diagnostics: {archive}", flush=True)
    if result["qualified"]:
        print(f"[v16.0-workflow] QUALIFIED {experiment_id}", flush=True)
        return 0
    print(
        f"[v16.0-workflow] REJECTED {experiment_id}: {result.get('failedStage')}",
        flush=True,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
