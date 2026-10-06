#!/usr/bin/env python3
"""V17.2 train-fit-first unseen-sibling architecture proof.

Proof order is enforced:
1. Train exactly one authored crop.
2. The trained crop must pass the unchanged candidate fit gates.
3. Only then evaluate a different never-trained crop from the same authority.
4. Require visual A/B/C review before any broader training.

If the trained crop does not pass by the final bounded stage, the architecture
is rejected at train fit and sibling transfer is not interpreted.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from probe_nsamdr_v16_full_broad import (
    DEFAULT_MANIFEST,
    _autocast,
    _device,
    _loss_values,
    _proof_loss_terms,
    _residual_diagnostics,
    _to_device,
)
from probe_nsamdr_v16_memorization import _gate_snapshot
from v14.config import V16Config
from v14.qualification import sample_metrics
from v16.broad_prior import AuthorityBalancedSRDataset
from v17.model import NSAMDRV17


SCHEMA = "NSAMDR_V17_SIBLING_ARCHITECTURE_PROOF_V3"
CHECKPOINT_SCHEMA = "NSAMDR_V17_SIBLING_ARCHITECTURE_CHECKPOINT_V3"
DEFAULT_AUTHORITY = "13006d2b807f89ac"
EARLY_TRANSFER_GATE = {
    "global_recovery": 0.30,
    "edge_recovery": 0.35,
    "gradient_recovery": 0.25,
    "detail_recovery_1px": 0.05,
    "lattice_cell_excess_max": 0.15,
}


def _parse_stages(values: list[str]) -> list[int]:
    result: list[int] = []
    for raw in values:
        for token in str(raw).replace(";", ",").split(","):
            token = token.strip()
            if token:
                result.append(int(token))
    result = sorted(set(result))
    if not result or result[0] < 1:
        raise ValueError("stages must contain positive update counts")
    return result


def _authority_records(
    manifest: dict[str, Any],
    authority_id: str,
) -> list[dict[str, Any]]:
    rows = [
        dict(record)
        for record in manifest.get("crops", [])
        if str(record.get("split")) == "train"
        and str(record.get("family_id") or record.get("familyId") or "")
        == str(authority_id)
    ]
    rows.sort(
        key=lambda record: (
            str(record.get("crop_id") or record.get("cropId") or ""),
            str(record.get("path") or ""),
        )
    )
    if len(rows) < 2:
        raise RuntimeError(
            f"authority {authority_id} has only {len(rows)} authored train crop(s); "
            "V17 sibling proof requires at least two"
        )
    return rows


def _batch_for_record(
    record: dict[str, Any],
    config: V16Config,
    *,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    selected = dict(record)
    selected["split"] = "validation"
    dataset = AuthorityBalancedSRDataset(
        {"crops": [selected]},
        config,
        "validation",
        1,
        seed=seed,
        degradation="clean",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    batch = _to_device(next(iter(loader)), device)
    return batch, {
        "authorityId": str(selected.get("family_id") or selected.get("familyId") or ""),
        "cropId": str(selected.get("crop_id") or selected.get("cropId") or ""),
        "recordPath": str(selected.get("path") or ""),
    }


def _evaluate(
    model: NSAMDRV17,
    batch: dict[str, torch.Tensor],
    config: V16Config,
    *,
    device: torch.device,
    precision: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model.eval()
    with torch.no_grad():
        with _autocast(device, precision):
            outputs = model(
                batch["lr_albedo"],
                batch["lr_normal"],
                batch["lr_material"],
            )
            terms = _proof_loss_terms(outputs, batch, config)
    result = {
        "lossTerms": _loss_values(terms),
        "metrics": sample_metrics(outputs, batch, final=False),
        "residualDiagnostics": _residual_diagnostics(outputs, batch, config),
    }
    result["gateChecks"] = _gate_snapshot(result, config)
    return result, outputs


def _target_albedo_residual_magnitude(
    model: NSAMDRV17,
    batch: dict[str, torch.Tensor],
) -> float:
    with torch.no_grad():
        baseline, _normal, _material = model.baseline(
            batch["lr_albedo"],
            batch["lr_normal"],
            batch["lr_material"],
        )
    return float(
        (batch["target_albedo"].float() - baseline.float())
        .abs()
        .mean()
        .detach()
        .cpu()
    )


def _tensor_rgb(value: torch.Tensor) -> np.ndarray:
    return (
        value.detach()
        .float()
        .cpu()[0]
        .permute(1, 2, 0)
        .numpy()
        .clip(0.0, 1.0)
    )


def _preview_panel(image: np.ndarray, label: str) -> np.ndarray:
    bgr = cv2.cvtColor(
        np.uint8(np.rint(np.clip(image, 0.0, 1.0) * 255.0)),
        cv2.COLOR_RGB2BGR,
    )
    bar = np.zeros((42, bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        bar,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate((bar, bgr), axis=0)


def _write_preview(
    path: Path,
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    *,
    role: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    authored = _tensor_rgb(batch["target_albedo"])
    baseline = _tensor_rgb(outputs["baseline_albedo"])
    candidate = _tensor_rgb(outputs["candidate_albedo"])
    sheet = np.concatenate(
        (
            _preview_panel(authored, "A  AUTHORED"),
            _preview_panel(baseline, "B  DETERMINISTIC 4X"),
            _preview_panel(candidate, f"C  V17.2 {role.upper()}"),
        ),
        axis=1,
    )
    if not cv2.imwrite(str(path), sheet, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"could not write V17.2 preview: {path}")


def _train_one(
    model: NSAMDRV17,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    config: V16Config,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, precision):
        outputs = model(
            batch["lr_albedo"],
            batch["lr_normal"],
            batch["lr_material"],
        )
        terms = _proof_loss_terms(outputs, batch, config)
        loss = terms["total"]
    loss.backward()
    parameters = model.trainable_parameters()
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    return _loss_values(terms)


def _transfer_gate(result: dict[str, Any]) -> dict[str, bool]:
    metrics = result["metrics"]
    return {
        "global": float(metrics["global_recovery"])
        >= EARLY_TRANSFER_GATE["global_recovery"],
        "edge": float(metrics["edge_recovery"])
        >= EARLY_TRANSFER_GATE["edge_recovery"],
        "gradient": float(metrics["gradient_recovery"])
        >= EARLY_TRANSFER_GATE["gradient_recovery"],
        "detail1px": float(metrics["detail_recovery_1px"])
        >= EARLY_TRANSFER_GATE["detail_recovery_1px"],
        "lattice": float(metrics["lattice_cell_excess"])
        <= EARLY_TRANSFER_GATE["lattice_cell_excess_max"],
    }


def _summary_line(label: str, result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return (
        f"{label}: "
        f"global={float(metrics['global_recovery'])*100:+.2f}% "
        f"edge={float(metrics['edge_recovery'])*100:+.2f}% "
        f"grad={float(metrics['gradient_recovery'])*100:+.2f}% "
        f"detail1={float(metrics['detail_recovery_1px'])*100:+.2f}% "
        f"lattice={float(metrics['lattice_cell_excess'])*100:+.2f}%"
    )


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest is missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = _device(str(args.device))
    stages = _parse_stages(list(args.stages))
    authority_id = str(args.authority_id).strip()
    records = _authority_records(manifest, authority_id)

    manifest_relative = (
        str(manifest_path.relative_to(repo_root))
        if manifest_path.is_relative_to(repo_root)
        else str(manifest_path)
    )
    config = V16Config(
        dataset_manifest=manifest_relative,
        train_hr_size=int(args.hr_size),
        train_lr_size=int(args.hr_size) // 4,
        validation_hr_size=int(args.hr_size),
        validation_lr_size=int(args.hr_size) // 4,
        minimum_heldout_samples=4,
        tiles_per_epoch=1,
        validation_tiles=1,
        clean_epochs=1,
        robust_epochs=1,
        selector_epochs=1,
        use_gradient_checkpointing=False,
    )
    config.validate()

    train_batch, train_meta = _batch_for_record(
        records[0],
        config,
        seed=int(args.seed) + 1,
        device=device,
    )
    sibling_batch, sibling_meta = _batch_for_record(
        records[1],
        config,
        seed=int(args.seed) + 2,
        device=device,
    )

    torch.manual_seed(int(args.seed))
    model = NSAMDRV17(
        config,
        encoder_channels=int(args.encoder_channels),
        mid_channels=int(args.mid_channels),
        detail_channels=int(args.detail_channels),
        decoder_blocks=int(args.decoder_blocks),
    ).to(device)
    model.set_candidate_training()
    optimizer = torch.optim.Adam(
        model.trainable_parameters(),
        lr=float(args.learning_rate),
        betas=(0.9, 0.999),
        eps=1.0e-8,
        foreach=False,
    )

    train_target_residual = _target_albedo_residual_magnitude(model, train_batch)
    sibling_target_residual = _target_albedo_residual_magnitude(model, sibling_batch)
    if train_target_residual < float(args.minimum_target_residual):
        raise RuntimeError(
            "selected trained crop is too close to deterministic B; choose another authority"
        )
    if sibling_target_residual < float(args.minimum_target_residual):
        raise RuntimeError(
            "selected sibling crop is too close to deterministic B; choose another authority"
        )

    initial_train, initial_train_outputs = _evaluate(
        model,
        train_batch,
        config,
        device=device,
        precision=str(args.amp_precision),
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (
        repo_root
        / "artifacts/nsamdr/diagnostics/v17_sibling_architecture"
        / f"proof_{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_preview(
        run_dir / "stage_000000" / "trained_ABC.png",
        train_batch,
        initial_train_outputs,
        role="trained",
    )

    print("NSAMDR V17.2 TRAIN-FIT-FIRST SIBLING ARCHITECTURE PROOF", flush=True)
    print(f"Authority        : {authority_id}", flush=True)
    print(f"Trained crop     : {train_meta['cropId']}", flush=True)
    print(f"Sibling crop     : {sibling_meta['cropId']} (held until train-fit gate passes)", flush=True)
    print(f"Stages           : {stages}", flush=True)
    print("Sampling         : fixed authored 512 crop; clean 128 LR; no augmentation", flush=True)
    print("Architecture     : LR encoder + 2x mid-band + 4x high-detail residual decoder", flush=True)
    print("Upsampling       : bilinear resize-convolution; no PixelShuffle/ConvTranspose/phase tensor", flush=True)
    print(_summary_line("Initial trained ", initial_train), flush=True)

    snapshots: list[dict[str, Any]] = [
        {
            "update": 0,
            "trained": initial_train,
            "trainedCandidateGatePass": bool(all(initial_train["gateChecks"].values())),
            "siblingEvaluated": False,
            "previews": {
                "trained": str(
                    (run_dir / "stage_000000" / "trained_ABC.png").resolve()
                ),
            },
        }
    ]
    started = time.monotonic()
    stage_set = set(stages)
    latest_train = initial_train
    latest_sibling: dict[str, Any] | None = None
    trained_gate_pass = False
    stop_update = 0
    decision = "reject-trained-fit"

    for update in range(1, stages[-1] + 1):
        train_loss = _train_one(
            model,
            optimizer,
            train_batch,
            config,
            device=device,
            precision=str(args.amp_precision),
        )
        if update == 1 or update % 32 == 0:
            elapsed = max(time.monotonic() - started, 1.0e-6)
            rate = update / elapsed
            eta = (stages[-1] - update) / max(rate, 1.0e-6)
            print(
                f"[v17-sibling] update {update:4d}/{stages[-1]} "
                f"loss={train_loss['total']:.6f} "
                f"elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m",
                flush=True,
            )
        if update not in stage_set:
            continue

        latest_train, train_outputs = _evaluate(
            model,
            train_batch,
            config,
            device=device,
            precision=str(args.amp_precision),
        )
        stage_dir = run_dir / f"stage_{update:06d}"
        trained_preview = stage_dir / "trained_ABC.png"
        _write_preview(
            trained_preview,
            train_batch,
            train_outputs,
            role="trained",
        )
        trained_gate = dict(latest_train["gateChecks"])
        trained_gate_pass = bool(all(trained_gate.values()))
        snapshot: dict[str, Any] = {
            "update": update,
            "trainLossBeforeUpdate": train_loss,
            "trained": latest_train,
            "trainedCandidateGatePass": trained_gate_pass,
            "siblingEvaluated": False,
            "previews": {
                "trained": str(trained_preview.resolve()),
            },
        }
        print(_summary_line(f"Stage {update:4d} train  ", latest_train), flush=True)
        print(f"Stage {update:4d} train gates: {trained_gate}", flush=True)

        if trained_gate_pass:
            latest_sibling, sibling_outputs = _evaluate(
                model,
                sibling_batch,
                config,
                device=device,
                precision=str(args.amp_precision),
            )
            sibling_preview = stage_dir / "sibling_ABC.png"
            _write_preview(
                sibling_preview,
                sibling_batch,
                sibling_outputs,
                role="sibling",
            )
            sibling_gate = _transfer_gate(latest_sibling)
            snapshot["sibling"] = latest_sibling
            snapshot["siblingEvaluated"] = True
            snapshot["siblingEarlyTransferGate"] = sibling_gate
            snapshot["previews"]["sibling"] = str(sibling_preview.resolve())
            print(
                "TRAIN-FIT GATE PASSED: evaluating unseen sibling exactly once.",
                flush=True,
            )
            print(_summary_line(f"Stage {update:4d} sibling", latest_sibling), flush=True)
            print(f"Stage {update:4d} sibling early gate: {sibling_gate}", flush=True)
            decision = "sibling-visual-review-required"
            stop_update = update
            snapshots.append(snapshot)
            break

        snapshots.append(snapshot)
        stop_update = update

    checkpoint_path = run_dir / "checkpoint.pt"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "authorityId": authority_id,
            "trainCropId": train_meta["cropId"],
            "siblingCropId": sibling_meta["cropId"],
            "manifest": str(manifest_path.resolve()),
            "config": config.to_dict(),
            "architecture": model.architecture_contract(),
            "update": int(stop_update),
            "modelState": model.state_dict(),
            "optimizerState": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    trained_gate = dict(latest_train["gateChecks"])
    sibling_gate = (
        _transfer_gate(latest_sibling)
        if latest_sibling is not None
        else None
    )
    report = {
        "schema": SCHEMA,
        "architecture": model.architecture_contract(),
        "authorityId": authority_id,
        "trainedCrop": train_meta,
        "siblingCrop": sibling_meta,
        "stages": stages,
        "stopUpdate": int(stop_update),
        "decision": decision,
        "minimumTargetResidual": float(args.minimum_target_residual),
        "trainTargetAlbedoResidualMagnitude": train_target_residual,
        "siblingTargetAlbedoResidualMagnitude": sibling_target_residual,
        "earlySiblingTransferThresholds": EARLY_TRANSFER_GATE,
        "finalTrainedCandidateGate": trained_gate,
        "finalSiblingEarlyTransferGate": sibling_gate,
        "trainedCandidateGatePass": bool(trained_gate_pass),
        "siblingEvaluated": latest_sibling is not None,
        "siblingNumericalGatePass": (
            bool(all(sibling_gate.values())) if sibling_gate is not None else None
        ),
        "visualGateRequired": latest_sibling is not None,
        "visualGateInstruction": (
            "Inspect sibling_ABC.png. Thin authored seams/manufactured boundaries "
            "must visibly reappear in C; a sharpened/pixelated B-like result rejects V17.2."
            if latest_sibling is not None
            else "No sibling visual gate: trained crop failed the candidate fit gate."
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "snapshots": snapshots,
        "elapsedSeconds": time.monotonic() - started,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 84, flush=True)
    print("V17.2 ARCHITECTURE DECISION", flush=True)
    print(_summary_line("Final trained ", latest_train), flush=True)
    print(f"Trained candidate gates : {trained_gate}", flush=True)
    if latest_sibling is None:
        print(
            "Decision                : REJECT AT TRAIN FIT - sibling was not evaluated",
            flush=True,
        )
        print(
            f"Trained preview         : "
            f"{run_dir / f'stage_{stop_update:06d}' / 'trained_ABC.png'}",
            flush=True,
        )
    else:
        print(_summary_line("Final sibling ", latest_sibling), flush=True)
        print(f"Sibling early gate      : {sibling_gate}", flush=True)
        print(
            "Decision                : TRAIN FIT PASSED - visual sibling review required",
            flush=True,
        )
        print(
            f"Sibling preview         : "
            f"{run_dir / f'stage_{stop_update:06d}' / 'sibling_ABC.png'}",
            flush=True,
        )
    print(f"Report                  : {report_path}", flush=True)
    return 0, report_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Train one V17.2 crop; evaluate unseen sibling only after train-fit gates pass"
        )
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--manifest", default=DEFAULT_MANIFEST)
    value.add_argument("--authority-id", default=DEFAULT_AUTHORITY)
    value.add_argument("--hr-size", type=int, default=512)
    value.add_argument(
        "--stages",
        nargs="+",
        default=["128,256,384,512,768"],
        help="cumulative exact-crop updates",
    )
    value.add_argument("--minimum-target-residual", type=float, default=0.01)
    value.add_argument("--seed", type=int, default=17001)
    value.add_argument("--learning-rate", type=float, default=2.0e-4)
    value.add_argument("--encoder-channels", type=int, default=96)
    value.add_argument("--mid-channels", type=int, default=64)
    value.add_argument("--detail-channels", type=int, default=48)
    value.add_argument("--decoder-blocks", type=int, default=4)
    value.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    value.add_argument(
        "--amp-precision",
        choices=("auto", "bf16", "fp16"),
        default="auto",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    code, _ = run(parser().parse_args(argv))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
