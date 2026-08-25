#!/usr/bin/env python3
"""Canonical Quick/Full NSAMDR experiment, qualification, and preview workflow."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v9.experiments import (
    ensure_experiment_layout,
    experiment_dir,
    freeze_final_checkpoint,
    load_experiment_manifest,
    load_final_manifest,
    qualify_final_manifest,
)


PRODUCTION_CONFIG = "tools/nsamdr/neural/configs/v9_fidelity_full.json"
RAVEN_DATA_CONFIG = "tools/nsamdr/neural/configs/v9_preview_raven.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(command: list[str], *, cwd: Path, label: str) -> None:
    print(f"[workflow] {label}: " + subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"required workflow record is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"workflow record is not an object: {path}")
    return payload


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one complete production NSAMDR experiment (Quick or Full)"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--training-mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--base-config", default=PRODUCTION_CONFIG)
    parser.add_argument("--raven-data-config", default=RAVEN_DATA_CONFIG)
    parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--max-train-regions", type=int, default=16)
    parser.add_argument("--max-validation-regions", type=int, default=4)
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--experiment", default="new")
    parser.add_argument("--control", choices=("auto", "resume"), default="auto")
    parser.add_argument("--tiles-per-epoch", type=int)
    parser.add_argument("--validation-tiles", type=int)
    parser.add_argument("--performance-profile", default="optimized")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--amp-precision", default="auto")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0005)
    parser.add_argument("--preview-target-size", type=int, default=4096)
    parser.add_argument("--preview-device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--skip-preview", action="store_true", help="qualify without opening the native viewer")
    return parser


def _prepare_dataset(args: argparse.Namespace, root: Path, python: str, base_config: Path, raven_config: Path) -> None:
    if args.training_mode == "quick":
        command = [
            python,
            "-u",
            str(root / "tools/nsamdr/neural/prepare_nsamdr_v9_raven_preview_dataset.py"),
            "--repo-root",
            str(root),
            "--config",
            str(raven_config),
            "--shared-cache",
            args.shared_cache,
            "--train-crops",
            str(max(1, int(args.max_train_regions))),
            "--validation-crops",
            str(max(1, int(args.max_validation_regions))),
        ]
    else:
        command = [
            python,
            "-u",
            str(root / "tools/nsamdr/neural/index_eve_texture_dataset_v9.py"),
            "--repo-root",
            str(root),
            "--config",
            str(base_config),
        ]
        if args.source_root is not None:
            command += ["--source-root", str(args.source_root.resolve())]
        else:
            command += ["--shared-cache", args.shared_cache]
    if args.rebuild_dataset:
        command.append("--rebuild")
    _run(command, cwd=root, label=f"prepare {args.training_mode} dataset")


def _training_command(
    args: argparse.Namespace,
    root: Path,
    python: str,
    base_config: Path,
    dataset_config: Path,
    result_file: Path,
) -> list[str]:
    command = [
        python,
        "-u",
        str(root / "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py"),
        "--repo-root",
        str(root),
        "--base-config",
        str(base_config),
        "--dataset-config",
        str(dataset_config),
        "--experiment",
        args.experiment,
        "--control",
        args.control,
        "--training-mode",
        args.training_mode,
        "--performance-profile",
        args.performance_profile,
        "--workers",
        str(args.workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--amp-precision",
        args.amp_precision,
        "--device",
        args.device,
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--early-stop-min-delta",
        str(args.early_stop_min_delta),
        "--result-file",
        str(result_file),
    ]
    if args.tiles_per_epoch is not None:
        command += ["--tiles-per-epoch", str(args.tiles_per_epoch)]
    if args.validation_tiles is not None:
        command += ["--validation-tiles", str(args.validation_tiles)]
    return command


def _diagnostics(experiment_directory: Path) -> Path:
    target = experiment_directory.parent / f"{experiment_directory.name}_DIAGNOSTICS.zip"
    roots = (
        experiment_directory / "config.json",
        experiment_directory / "resolved_config.json",
        experiment_directory / "experiment.json",
        experiment_directory / "final_manifest.json",
        experiment_directory / "architecture_participation.json",
        experiment_directory / "training_log.csv",
    )
    folders = (
        experiment_directory / "metrics",
        experiment_directory / "checkpoints",
        experiment_directory / "evidence",
        experiment_directory / "previews",
    )
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in roots:
            if source.is_file():
                archive.write(source, source.relative_to(experiment_directory))
        for folder in folders:
            if not folder.is_dir():
                continue
            for source in sorted(folder.rglob("*")):
                if source.is_file() and source.suffix.lower() not in {".pt", ".pth"}:
                    archive.write(source, source.relative_to(experiment_directory))
    return target


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    python = sys.executable
    base_config = _resolve(root, args.base_config)
    raven_config = _resolve(root, args.raven_data_config)
    dataset_config = raven_config if args.training_mode == "quick" else base_config
    result_file = root / "artifacts/nsamdr/gui/last_nsamdr_workflow_result.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.unlink(missing_ok=True)

    print("=" * 78, flush=True)
    print(f"NSAMDR CANONICAL {args.training_mode.upper()} WORKFLOW", flush=True)
    print("Model/config semantics   : production", flush=True)
    print("Mode difference          : dataset + work budget only", flush=True)
    print("Alternate Raven networks : forbidden", flush=True)
    print("=" * 78, flush=True)

    train_command = _training_command(
        args, root, python, base_config, dataset_config, result_file
    )

    requested_new = str(args.experiment).strip().upper() in {"", "NEW"}
    if requested_new:
        _run(train_command + ["--allocate-only"], cwd=root, label="allocate experiment")
        allocation = _read_json(result_file)
        experiment = str(allocation.get("experiment") or "").strip().upper()
        if not experiment:
            raise RuntimeError("trainer allocation did not return an experiment id")
        experiment_directory = experiment_dir(root, experiment)
        index = train_command.index("--experiment")
        train_command[index + 1] = experiment
        index = train_command.index("--control")
        train_command[index + 1] = "auto"
    else:
        experiment = str(args.experiment).strip().upper()
        experiment_directory = experiment_dir(root, experiment)
    ensure_experiment_layout(experiment_directory)

    diagnostics_path: Path | None = None
    try:
        evidence = experiment_directory / "evidence"
        preflight = evidence / "architecture_preflight.json"
        contract = root / "tools/nsamdr/neural/raven_architecture_contract.py"
        _run(
            [
                python, "-u", str(contract), "pre", "--repo-root", str(root),
                "--config", str(experiment_directory / "resolved_config.json"),
                "--output", str(preflight),
            ],
            cwd=root,
            label="architecture preflight",
        )

        manifest = load_experiment_manifest(root, experiment)
        status = str(manifest.get("status") or "").lower()
        if status == "completed":
            raise RuntimeError(
                f"{experiment} is an immutable completed final; use `nsamdr preview {experiment}`"
            )

        _prepare_dataset(args, root, python, base_config, raven_config)
        if status != "trained-pending-qualification":
            result_file.unlink(missing_ok=True)
            _run(train_command, cwd=root, label="train complete production model")
            trained = _read_json(result_file)
            if str(trained.get("experiment") or "").upper() != experiment:
                raise RuntimeError("trainer changed the allocated experiment identity")

        final_manifest_path = experiment_directory / "final_manifest.json"
        if not final_manifest_path.is_file():
            freeze_final_checkpoint(
                root,
                experiment,
                source_checkpoint=experiment_directory / "nsamdr_v9_fidelity.pt",
                source_metadata=experiment_directory / "nsamdr_v9_fidelity.json",
                preflight_path=preflight,
            )

        participation = experiment_directory / "architecture_participation.json"
        _run(
            [
                python, "-u", str(contract), "post", "--repo-root", str(root),
                "--experiment-dir", str(experiment_directory),
                "--output", str(participation),
            ],
            cwd=root,
            label="strict-load and uncached production-forward qualification",
        )
        final = qualify_final_manifest(root, experiment, participation)
        load_final_manifest(root, experiment, require_qualified=True)

        if not args.skip_preview:
            preview = [
                python,
                "-u",
                str(root / "tools/nsamdr/neural/preview_nsamdr_v9_experiment.py"),
                "--repo-root",
                str(root),
                "--experiment",
                experiment,
                "--shared-cache",
                args.shared_cache,
                "--target-size",
                str(args.preview_target_size),
                "--device",
                args.preview_device,
            ]
            _run(preview, cwd=root, label="verify candidate provenance and launch native renderer")

        diagnostics_path = _diagnostics(experiment_directory)
        result_file.write_text(
            json.dumps(
                {
                    "experiment": experiment,
                    "directory": str(experiment_directory),
                    "trainingMode": args.training_mode,
                    "status": "completed",
                    "checkpoint": final["checkpoint"],
                    "diagnostics": str(diagnostics_path),
                    "completedUtc": _utc_now(),
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    finally:
        if experiment_directory.is_dir() and diagnostics_path is None:
            try:
                diagnostics_path = _diagnostics(experiment_directory)
            except Exception as diagnostics_error:  # noqa: BLE001 - preserve the workflow failure
                print(
                    f"[workflow] WARNING: diagnostics ZIP creation failed: {diagnostics_error}",
                    file=sys.stderr,
                    flush=True,
                )

    final = load_final_manifest(root, experiment, require_qualified=True)
    print("=" * 78, flush=True)
    print("NSAMDR WORKFLOW COMPLETE", flush=True)
    print(f"Experiment               : {experiment}", flush=True)
    print(f"Immutable checkpoint     : {final['_checkpointPath']}", flush=True)
    print(f"Checkpoint SHA-256       : {final['checkpoint']['sha256']}", flush=True)
    print(f"Diagnostics              : {diagnostics_path}", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
