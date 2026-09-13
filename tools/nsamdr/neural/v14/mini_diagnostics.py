#!/usr/bin/env python3
"""Focused, non-promotable V14 diagnostics.

These restore the useful diagnostic progression that existed before the V14 cleanup,
without restoring any V9-V13 model/trainer code:

* capacity    - overfit one deterministic edge-dense Raven patch with production V14 C;
* multiregion - train V14 C on a small disjoint Raven subset and test held-out regions;
* selector    - train only BenefitSelector from the latest passing multiregion candidate.

Nothing in this module can create or promote a production final checkpoint.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch

if __package__ in {None, ""}:
    NEURAL_ROOT = Path(__file__).resolve().parent.parent
    if str(NEURAL_ROOT) not in sys.path:
        sys.path.insert(0, str(NEURAL_ROOT))
    from v14.checkpoint import load_checkpoint, save_checkpoint
    from v14.config import V14Config
    from v14.dataset import RavenSRDataset, load_manifest
    from v14.losses import candidate_loss, selector_loss
    from v14.model import MODEL_SCHEMA, NSAMDRV14
    from v14.qualification import aggregate_candidate, aggregate_final, sample_metrics
else:
    from .checkpoint import load_checkpoint, save_checkpoint
    from .config import V14Config
    from .dataset import RavenSRDataset, load_manifest
    from .losses import candidate_loss, selector_loss
    from .model import MODEL_SCHEMA, NSAMDRV14
    from .qualification import aggregate_candidate, aggregate_final, sample_metrics


DIAGNOSTIC_SCHEMA = "NSAMDR_V14_MINI_DIAGNOSTIC_V1"


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V14 mini diagnostic requested CUDA but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision in {"auto", "bf16"} else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


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
        str(args.prepare_train_regions),
        "--validation-crops",
        str(args.prepare_validation_regions),
    ]
    if args.rebuild_dataset:
        command.append("--rebuild")
    print("[v14-mini] prepare authored Raven dataset: " + subprocess.list2cmdline(command), flush=True)
    result = subprocess.run(command, cwd=repo_root, check=False)
    if result.returncode:
        raise RuntimeError(f"Raven dataset preparation failed with exit code {result.returncode}")


def _run_directory(repo_root: Path, mode: str) -> Path:
    root = repo_root / "artifacts/nsamdr/diagnostics/v14_mini"
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    candidate = root / f"{mode}_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{mode}_{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _archive(run_dir: Path) -> Path:
    return Path(shutil.make_archive(str(run_dir) + "_DIAGNOSTICS", "zip", root_dir=run_dir))


def _batchify(sample: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if not isinstance(value, torch.Tensor) or key == "record_index":
            continue
        tensor = value.unsqueeze(0) if value.ndim == 3 else value
        result[key] = tensor.to(device, non_blocking=True)
    return result


def _record_key(record: dict[str, Any]) -> str:
    return str(record.get("cropId") or record.get("crop_id") or record.get("path") or "unknown")


def _detail_score(record: dict[str, Any]) -> float:
    try:
        return float(record.get("detailScore", record.get("detail_score", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _pseudo_manifest(records: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    copied = []
    for record in records:
        item = dict(record)
        item["split"] = split
        copied.append(item)
    return {"crops": copied}


def _dataset_sample(record: dict[str, Any], config: V14Config, device: torch.device) -> dict[str, torch.Tensor]:
    manifest = _pseudo_manifest([record], split="validation")
    dataset = RavenSRDataset(
        manifest,
        config,
        "validation",
        1,
        seed=config.seed,
        degradation="clean",
    )
    return _batchify(dataset[0], device)


def _iter_train_batches(
    manifest: dict[str, Any],
    config: V14Config,
    count: int,
    *,
    seed: int,
    degradation: str,
    device: torch.device,
):
    dataset = RavenSRDataset(
        manifest,
        config,
        "train",
        count,
        seed=seed,
        degradation=degradation,
    )
    for index in range(len(dataset)):
        yield _batchify(dataset[index], device)


def _validation_metrics(
    model: NSAMDRV14,
    manifest: dict[str, Any],
    config: V14Config,
    device: torch.device,
    precision: str,
    *,
    final: bool,
) -> list[dict[str, float]]:
    records = [record for record in manifest["crops"] if record.get("split") == "validation"]
    dataset = RavenSRDataset(
        manifest,
        config,
        "validation",
        len(records),
        seed=config.seed + 701,
        degradation="clean",
    )
    metrics: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = _batchify(dataset[index], device)
            with _autocast(device, precision):
                outputs = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
            metrics.append(sample_metrics(outputs, batch, final=final))
    return metrics


def _u8(value: torch.Tensor) -> np.ndarray:
    image = value.detach().float().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    return np.round(image * 255.0).astype(np.uint8)


def _save_probe(run_dir: Path, batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor], *, include_final: bool) -> Path:
    panels: list[tuple[str, np.ndarray]] = [
        ("A AUTHORED", _u8(batch["target_albedo"])),
        ("B BASELINE", _u8(outputs["baseline_albedo"])),
        ("C V14 SR", _u8(outputs["candidate_albedo"])),
    ]
    if include_final:
        panels.append(("F SELECTED", _u8(outputs["albedo"])))
    rendered: list[np.ndarray] = []
    for label, image in panels:
        panel = image.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(panel, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        rendered.append(panel)
    contact = np.concatenate(rendered, axis=1)
    path = run_dir / "ABCF_probe.png"
    cv2.imwrite(str(path), contact[:, :, ::-1])
    return path


def _write_report(run_dir: Path, report: dict[str, Any]) -> Path:
    path = run_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _threshold_config(args: argparse.Namespace, *, minimum_heldout: int) -> V14Config:
    config = V14Config(
        minimum_heldout_samples=minimum_heldout,
        candidate_edge_recovery_required=float(args.required_edge_recovery),
        candidate_global_recovery_required=float(args.required_global_recovery),
        candidate_gradient_recovery_required=float(args.required_gradient_recovery),
    )
    config.validate()
    return config


def _capacity(args: argparse.Namespace, repo_root: Path, device: torch.device) -> tuple[int, Path]:
    _prepare_dataset(args, repo_root)
    config = _threshold_config(args, minimum_heldout=1)
    manifest = load_manifest(repo_root, config)
    records = [record for record in manifest["crops"] if record.get("split") == "train"]
    if not records:
        raise RuntimeError("V14 capacity diagnostic found no Raven training regions")
    record = max(records, key=_detail_score)
    batch = _dataset_sample(record, config, device)
    model = NSAMDRV14(config).to(device)
    model.set_candidate_training()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(args.learning_rate), weight_decay=config.weight_decay)
    run_dir = _run_directory(repo_root, "capacity")

    print("=" * 72, flush=True)
    print("V14 HR RESIDUAL CAPACITY — DIAGNOSTIC ONLY", flush=True)
    print(f"Region        : {_record_key(record)}", flush=True)
    print(f"Geometry      : {config.train_lr_size} -> {config.train_hr_size}", flush=True)
    print(f"Steps         : {args.steps}", flush=True)
    print(f"Learning rate : {args.learning_rate}", flush=True)
    print("=" * 72, flush=True)

    last_metrics: dict[str, float] = {}
    for step in range(1, int(args.steps) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, args.amp_precision):
            outputs = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
            losses = candidate_loss(outputs, batch, config)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        if step == 1 or step % int(args.report_every) == 0 or step == int(args.steps):
            model.eval()
            with torch.no_grad(), _autocast(device, args.amp_precision):
                outputs = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
            last_metrics = sample_metrics(outputs, batch, final=False)
            print(
                f"  step {step:4d}/{int(args.steps):4d} "
                f"global={last_metrics['global_recovery']*100:+.2f}% "
                f"edge={last_metrics['edge_recovery']*100:+.2f}% "
                f"grad={last_metrics['gradient_recovery']*100:+.2f}% "
                f"lattice={last_metrics['lattice_cell_excess']*100:+.1f}%",
                flush=True,
            )

    passed = bool(
        last_metrics.get("edge_recovery", -1.0) >= float(args.required_edge_recovery)
        and last_metrics.get("global_recovery", -1.0) >= float(args.required_global_recovery)
        and last_metrics.get("gradient_recovery", -1.0) >= float(args.required_gradient_recovery)
        and last_metrics.get("lattice_cell_excess", 1.0) <= config.candidate_lattice_cell_excess_max
    )
    checkpoint = run_dir / "candidate_checkpoint.pt"
    save_checkpoint(checkpoint, model, config, epoch=0, phase="mini-capacity", metrics=last_metrics)
    model.eval()
    with torch.no_grad(), _autocast(device, args.amp_precision):
        outputs = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
    probe = _save_probe(run_dir, batch, outputs, include_final=False)
    report = {
        "schema": DIAGNOSTIC_SCHEMA,
        "mode": "capacity",
        "passed": passed,
        "promotable": False,
        "modelSchema": MODEL_SCHEMA,
        "record": _record_key(record),
        "metrics": last_metrics,
        "thresholds": {
            "edgeRecovery": float(args.required_edge_recovery),
            "globalRecovery": float(args.required_global_recovery),
            "gradientRecovery": float(args.required_gradient_recovery),
            "maxLatticeCellExcess": config.candidate_lattice_cell_excess_max,
        },
        "candidateCheckpoint": str(checkpoint.resolve()),
        "probe": str(probe.resolve()),
        "datasetFingerprint": manifest.get("fingerprint"),
    }
    _write_report(run_dir, report)
    return (0 if passed else 2), run_dir


def _multiregion(args: argparse.Namespace, repo_root: Path, device: torch.device) -> tuple[int, Path]:
    args.prepare_train_regions = max(int(args.prepare_train_regions), int(args.train_regions))
    args.prepare_validation_regions = max(int(args.prepare_validation_regions), int(args.validation_regions))
    _prepare_dataset(args, repo_root)
    config = _threshold_config(args, minimum_heldout=2)
    config.tiles_per_epoch = int(args.tiles_per_epoch)
    config.clean_epochs = int(args.epochs)
    config.robust_epochs = 0
    config.validate()
    manifest = load_manifest(repo_root, config)
    train_records = sorted(
        [record for record in manifest["crops"] if record.get("split") == "train"],
        key=_detail_score,
        reverse=True,
    )[: int(args.train_regions)]
    validation_records = sorted(
        [record for record in manifest["crops"] if record.get("split") == "validation"],
        key=_detail_score,
        reverse=True,
    )[: int(args.validation_regions)]
    run_dir = _run_directory(repo_root, "multiregion")
    if len(train_records) < 2 or len(validation_records) < 2:
        report = {
            "schema": DIAGNOSTIC_SCHEMA,
            "mode": "multiregion",
            "passed": False,
            "promotable": False,
            "reason": "insufficient-disjoint-regions",
            "trainRegions": len(train_records),
            "validationRegions": len(validation_records),
            "requiredTrainRegions": 2,
            "requiredValidationRegions": 2,
            "datasetFingerprint": manifest.get("fingerprint"),
        }
        _write_report(run_dir, report)
        print(
            f"[v14-mini] multiregion rejected: need >=2 train and >=2 validation regions; "
            f"found {len(train_records)} / {len(validation_records)}",
            flush=True,
        )
        return 2, run_dir

    pseudo = {"crops": [*train_records, *validation_records]}
    model = NSAMDRV14(config).to(device)
    model.set_candidate_training()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(args.learning_rate), weight_decay=config.weight_decay)
    best_score = -1.0e9
    best_path: Path | None = None
    best_report: dict[str, object] | None = None

    print("=" * 72, flush=True)
    print("V14 MULTI-REGION SR MINI — DIAGNOSTIC ONLY", flush=True)
    print(f"Train/held-out : {len(train_records)} / {len(validation_records)} disjoint regions", flush=True)
    print(f"Epochs         : {args.epochs} clean SR", flush=True)
    print(f"Tiles/epoch    : {args.tiles_per_epoch}", flush=True)
    print("=" * 72, flush=True)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running = 0.0
        for index, batch in enumerate(
            _iter_train_batches(
                pseudo,
                config,
                int(args.tiles_per_epoch),
                seed=config.seed + epoch * 31,
                degradation="clean",
                device=device,
            ),
            start=1,
        ):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, args.amp_precision):
                outputs = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
                losses = candidate_loss(outputs, batch, config)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            running += float(losses["total"].detach().item())
            if index == 1 or index % 16 == 0 or index == int(args.tiles_per_epoch):
                print(f"  epoch {epoch}/{args.epochs} tile {index:3d}/{args.tiles_per_epoch} loss={running/index:.6f}", flush=True)

        metrics = _validation_metrics(model, pseudo, config, device, args.amp_precision, final=False)
        report = aggregate_candidate(metrics, config)
        report["epoch"] = epoch
        score = (
            float(report["medianGlobalRecovery"])
            + float(report["medianEdgeRecovery"])
            + float(report["medianGradientRecovery"])
        )
        checkpoint = run_dir / "checkpoints" / f"candidate_epoch_{epoch:02d}.pt"
        save_checkpoint(checkpoint, model, config, epoch=epoch, phase="mini-multiregion", metrics=report)
        print(
            f"  held-out global={float(report['medianGlobalRecovery'])*100:+.2f}% "
            f"edge={float(report['medianEdgeRecovery'])*100:+.2f}% "
            f"grad={float(report['medianGradientRecovery'])*100:+.2f}% "
            f"wins={float(report['positiveGlobalFraction'])*100:.1f}% "
            f"qualified={'YES' if report['passed'] else 'NO'}",
            flush=True,
        )
        selection = (1 if bool(report["passed"]) else 0, score)
        current = (1 if best_report and bool(best_report.get("passed")) else 0, best_score)
        if best_report is None or selection > current:
            best_score = score
            best_path = checkpoint
            best_report = report

    if best_path is None or best_report is None:
        raise RuntimeError("V14 multiregion mini produced no candidate checkpoint")
    model, _payload = load_checkpoint(best_path, device)
    metrics = _validation_metrics(model, pseudo, config, device, args.amp_precision, final=False)
    selected_report = aggregate_candidate(metrics, config)
    selected_report["selectedCheckpoint"] = str(best_path.resolve())
    selected_report["selectedEpoch"] = int(best_report["epoch"])

    preview_record = validation_records[0]
    batch = _dataset_sample(preview_record, config, device)
    model.eval()
    with torch.no_grad(), _autocast(device, args.amp_precision):
        outputs = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
    probe = _save_probe(run_dir, batch, outputs, include_final=False)
    report = {
        "schema": DIAGNOSTIC_SCHEMA,
        "mode": "multiregion",
        "passed": bool(selected_report["passed"]),
        "promotable": False,
        "modelSchema": MODEL_SCHEMA,
        "candidate": selected_report,
        "candidateCheckpoint": str(best_path.resolve()),
        "trainRecords": [str(record["path"]) for record in train_records],
        "validationRecords": [str(record["path"]) for record in validation_records],
        "probe": str(probe.resolve()),
        "datasetFingerprint": manifest.get("fingerprint"),
    }
    _write_report(run_dir, report)
    return (0 if bool(selected_report["passed"]) else 2), run_dir


def _latest_passing_multiregion(repo_root: Path) -> tuple[Path, dict[str, Any]]:
    root = repo_root / "artifacts/nsamdr/diagnostics/v14_mini"
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    if root.is_dir():
        for path in root.glob("multiregion_*/report.json"):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                checkpoint = Path(str(report.get("candidateCheckpoint") or ""))
                if report.get("schema") == DIAGNOSTIC_SCHEMA and report.get("passed") is True and checkpoint.is_file():
                    candidates.append((path.stat().st_mtime, path, report))
            except (OSError, ValueError, TypeError):
                continue
    if not candidates:
        raise RuntimeError("Selector Retention Mini requires a passing V14 Multi-Region SR Mini first")
    _mtime, path, report = max(candidates, key=lambda item: item[0])
    return path, report


def _selector(args: argparse.Namespace, repo_root: Path, device: torch.device) -> tuple[int, Path]:
    source_report_path, source_report = _latest_passing_multiregion(repo_root)
    checkpoint = Path(str(source_report["candidateCheckpoint"]))
    model, _payload = load_checkpoint(checkpoint, device)
    config = model.config
    config.selector_edge_retention_required = float(args.required_retention)
    config.selector_global_retention_required = float(args.required_retention)
    config.protected_preservation_required = float(args.protected_preservation)
    config.validate()
    manifest = load_manifest(repo_root, config)
    if source_report.get("datasetFingerprint") != manifest.get("fingerprint"):
        raise RuntimeError("Raven dataset changed since the passing multiregion mini; rerun that diagnostic first")

    train_paths = set(map(str, source_report.get("trainRecords") or []))
    validation_paths = set(map(str, source_report.get("validationRecords") or []))
    train_records = [record for record in manifest["crops"] if str(record.get("path")) in train_paths]
    validation_records = [record for record in manifest["crops"] if str(record.get("path")) in validation_paths]
    if len(train_records) < 2 or len(validation_records) < 2:
        raise RuntimeError("Passing multiregion mini no longer resolves its original disjoint Raven regions")
    pseudo = {"crops": [*train_records, *validation_records]}

    source_candidate = dict(source_report.get("candidate") or {})
    if not source_candidate or source_candidate.get("passed") is not True:
        raise RuntimeError("Passing multiregion report does not contain a qualified candidate block")

    model.set_selector_training()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(args.selector_learning_rate), weight_decay=config.weight_decay)
    run_dir = _run_directory(repo_root, "selector")
    best_path: Path | None = None
    best_report: dict[str, object] | None = None
    best_key = (-1, -1.0e9)

    print("=" * 72, flush=True)
    print("V14 SELECTOR RETENTION MINI — DIAGNOSTIC ONLY", flush=True)
    print(f"Candidate source : {source_report_path.parent}", flush=True)
    print(f"Selector epochs  : {args.selector_epochs}", flush=True)
    print(f"Tiles/epoch      : {args.tiles_per_epoch}", flush=True)
    print("=" * 72, flush=True)

    for epoch in range(1, int(args.selector_epochs) + 1):
        model.train()
        for index, batch in enumerate(
            _iter_train_batches(
                pseudo,
                config,
                int(args.tiles_per_epoch),
                seed=config.seed + 900 + epoch,
                degradation="clean",
                device=device,
            ),
            start=1,
        ):
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, args.amp_precision):
                outputs = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
                losses = selector_loss(outputs, batch)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            if index == 1 or index % 16 == 0 or index == int(args.tiles_per_epoch):
                print(
                    f"  epoch {epoch}/{args.selector_epochs} tile {index:3d}/{args.tiles_per_epoch} "
                    f"loss={float(losses['total'].detach().item()):.6f}",
                    flush=True,
                )

        final_metrics = _validation_metrics(model, pseudo, config, device, args.amp_precision, final=True)
        report = aggregate_final(source_candidate, final_metrics, config)
        checkpoint_out = run_dir / "checkpoints" / f"selector_epoch_{epoch:02d}.pt"
        save_checkpoint(checkpoint_out, model, config, epoch=epoch, phase="mini-selector", metrics=report)
        score = float(report["medianGlobalRecovery"]) + float(report["medianEdgeRecovery"])
        selection = (1 if bool(report["passed"]) else 0, score)
        if selection > best_key:
            best_key = selection
            best_path = checkpoint_out
            best_report = report
        print(
            f"  final global={float(report['medianGlobalRecovery'])*100:+.2f}% "
            f"edge={float(report['medianEdgeRecovery'])*100:+.2f}% "
            f"edgeRetention={float(report['selectorEdgeRetention'])*100:.1f}% "
            f"globalRetention={float(report['selectorGlobalRetention'])*100:.1f}% "
            f"protected={float(report['medianProtectedPreservation'])*100:.2f}% "
            f"qualified={'YES' if report['passed'] else 'NO'}",
            flush=True,
        )

    if best_path is None or best_report is None:
        raise RuntimeError("V14 selector mini produced no checkpoint")
    model, _payload = load_checkpoint(best_path, device)
    final_metrics = _validation_metrics(model, pseudo, config, device, args.amp_precision, final=True)
    selected_report = aggregate_final(source_candidate, final_metrics, config)
    selected_report["selectedCheckpoint"] = str(best_path.resolve())

    batch = _dataset_sample(validation_records[0], config, device)
    model.eval()
    with torch.no_grad(), _autocast(device, args.amp_precision):
        outputs = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
    probe = _save_probe(run_dir, batch, outputs, include_final=True)
    report = {
        "schema": DIAGNOSTIC_SCHEMA,
        "mode": "selector",
        "passed": bool(selected_report["passed"]),
        "promotable": False,
        "modelSchema": MODEL_SCHEMA,
        "sourceMultiregionReport": str(source_report_path.resolve()),
        "candidate": source_candidate,
        "final": selected_report,
        "selectorCheckpoint": str(best_path.resolve()),
        "probe": str(probe.resolve()),
        "datasetFingerprint": manifest.get("fingerprint"),
    }
    _write_report(run_dir, report)
    return (0 if bool(selected_report["passed"]) else 2), run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSAMDR V14 focused Raven mini diagnostics")
    p.add_argument("--mode", choices=("capacity", "multiregion", "selector"), required=True)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    p.add_argument("--rebuild-dataset", action="store_true")
    p.add_argument("--prepare-train-regions", type=int, default=16)
    p.add_argument("--prepare-validation-regions", type=int, default=4)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")

    p.add_argument("--steps", type=int, default=1024)
    p.add_argument("--report-every", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=2.0e-4)
    p.add_argument("--required-edge-recovery", type=float, default=0.60)
    p.add_argument("--required-global-recovery", type=float, default=0.45)
    p.add_argument("--required-gradient-recovery", type=float, default=0.35)

    p.add_argument("--train-regions", type=int, default=4)
    p.add_argument("--validation-regions", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--tiles-per-epoch", type=int, default=64)

    p.add_argument("--selector-epochs", type=int, default=2)
    p.add_argument("--selector-learning-rate", type=float, default=2.0e-4)
    p.add_argument("--required-retention", type=float, default=0.90)
    p.add_argument("--protected-preservation", type=float, default=0.99)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    device = _device(args.device)
    if args.mode == "capacity":
        code, run_dir = _capacity(args, repo_root, device)
    elif args.mode == "multiregion":
        code, run_dir = _multiregion(args, repo_root, device)
    else:
        code, run_dir = _selector(args, repo_root, device)
    archive = _archive(run_dir)
    print(f"[v14-mini] report      : {run_dir / 'report.json'}", flush=True)
    print(f"[v14-mini] diagnostics : {archive}", flush=True)
    print(f"[v14-mini] result      : {'PASS' if code == 0 else 'FAIL'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
