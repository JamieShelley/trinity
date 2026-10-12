#!/usr/bin/env python3
"""Bounded V17.2 multi-authority spatial-transfer diagnostic.

One crop per selected authority is trained using deterministic D4 augmentation.
Each authority's second crop is an untouched sibling holdout. This is a research
proof only: it cannot qualify production or native EVE rendering.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from probe_nsamdr_v16_full_broad import DEFAULT_MANIFEST, _device
from probe_nsamdr_v17_sibling_architecture import (
    EARLY_TRANSFER_GATE,
    _batch_for_record,
    _evaluate,
    _parse_stages,
    _target_albedo_residual_magnitude,
    _train_one,
    _transfer_gate,
    _write_preview,
)
from v14.config import V16Config
from v17.model import NSAMDRV17

SCHEMA = "NSAMDR_V17_GENERALIZATION_PROOF_V1"
CHECKPOINT_SCHEMA = "NSAMDR_V17_GENERALIZATION_CHECKPOINT_V1"
ROOT = "artifacts/nsamdr/diagnostics/v17_generalization"


def _select_pairs(
    manifest: dict[str, Any], anchor: str, count: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Select independent train/sibling pairs; never train the sibling."""
    if count < 2:
        raise ValueError("generalization proof needs at least two authorities")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest.get("crops", []):
        if str(row.get("split")) != "train":
            continue
        authority = str(row.get("family_id") or row.get("familyId") or "")
        if authority:
            groups[authority].append(dict(row))
    pairs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for authority, records in groups.items():
        records.sort(key=lambda r: (str(r.get("crop_id") or r.get("cropId") or ""), str(r.get("path") or "")))
        if len(records) < 2:
            continue
        first, sibling = records[:2]
        first_id = str(first.get("crop_id") or first.get("cropId") or "")
        sibling_id = str(sibling.get("crop_id") or sibling.get("cropId") or "")
        if not first_id or first_id == sibling_id or first.get("path") == sibling.get("path"):
            continue
        pairs[authority] = first, sibling
    if anchor not in pairs:
        raise RuntimeError(f"authority {anchor} has no distinct train/sibling pair")
    chosen = [anchor, *sorted(authority for authority in pairs if authority != anchor)[:count - 1]]
    if len(chosen) != count:
        raise RuntimeError(f"requested {count} authorities, but only {len(chosen)} have pairs")
    return [pairs[authority] for authority in chosen]


def _augment_batch(batch: dict[str, torch.Tensor], variant: int) -> dict[str, torch.Tensor]:
    """D4 spatial augmentation with correct normal-XY rotations/reflections."""
    if not 0 <= variant < 8:
        raise ValueError("D4 variant must be between 0 and 7")
    turns, mirror = variant % 4, variant >= 4
    result = dict(batch)
    for key in ("lr_albedo", "lr_material", "target_albedo", "target_material"):
        value = torch.rot90(batch[key], turns, (-2, -1))
        result[key] = torch.flip(value, (-1,)).contiguous() if mirror else value.contiguous()
    for key in ("lr_normal", "target_normal"):
        value = torch.rot90(batch[key], turns, (-2, -1))
        x, y = value[:, 0:1], value[:, 1:2]
        for _ in range(turns):
            x, y = -y, x
        value = torch.cat((x, y), dim=1)
        if mirror:
            value = torch.flip(value, (-1,)).contiguous()
            value[:, 0:1] = -value[:, 0:1]
        result[key] = value.contiguous()
    return result


def _checkpoint(
    path: Path, model: NSAMDRV17, optimizer: torch.optim.Optimizer,
    manifest: Path, crop_ids: list[str], update: int,
    snapshots: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": CHECKPOINT_SCHEMA,
        "architecture": model.architecture_contract(),
        "manifest": str(manifest),
        "trainCropIds": crop_ids,
        "update": update,
        "modelState": model.state_dict(),
        "optimizerState": optimizer.state_dict(),
        "snapshots": snapshots,
    }, path)


def _resume_path(root: Path, raw: str) -> Path | None:
    if raw == "fresh":
        return None
    if raw != "auto":
        path = Path(raw)
        return (root / path).resolve() if not path.is_absolute() else path
    runs = root / ROOT
    candidates = [
        path for path in runs.glob("proof_*/stage_*/checkpoint.pt")
        if not (path.parents[1] / "report.json").exists()
    ]
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def _load_checkpoint(
    path: Path, model: NSAMDRV17, optimizer: torch.optim.Optimizer,
    manifest: Path, crop_ids: list[str], device: torch.device,
) -> tuple[int, list[dict[str, Any]]]:
    if not path.is_file():
        raise RuntimeError(f"generalization checkpoint missing: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("architecture") != model.architecture_contract()
        or payload.get("manifest") != str(manifest)
        or payload.get("trainCropIds") != crop_ids
    ):
        raise RuntimeError("incompatible generalization checkpoint")
    model.load_state_dict(payload["modelState"], strict=True)
    optimizer.load_state_dict(payload["optimizerState"])
    return int(payload["update"]), list(payload.get("snapshots") or [])


def _score(results: list[dict[str, Any]], *, sibling: bool) -> dict[str, Any]:
    gates = [_transfer_gate(item) if sibling else item["gateChecks"] for item in results]
    metric_keys = ("global_recovery", "edge_recovery", "gradient_recovery", "detail_recovery_1px", "lattice_cell_excess")
    return {
        "allPass": all(all(checks.values()) for checks in gates),
        "gateChecks": gates,
        "mean": {
            name: sum(float(item["metrics"][name]) for item in results) / len(results)
            for name in metric_keys
        },
        "perCrop": results,
    }


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo = args.repo_root.resolve()
    manifest_path = Path(args.manifest)
    manifest_path = (repo / manifest_path).resolve() if not manifest_path.is_absolute() else manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = _select_pairs(manifest, str(args.authority_id), int(args.authorities))
    stages = _parse_stages(list(args.stages))
    train_ids = [str(a.get("crop_id") or a.get("cropId")) for a, _ in pairs]
    holdout_ids = [str(b.get("crop_id") or b.get("cropId")) for _, b in pairs]
    if set(train_ids) & set(holdout_ids):
        raise RuntimeError("train and sibling crop IDs overlap")
    device = _device(args.device)
    config = V16Config(
        dataset_manifest=str(manifest_path),
        train_hr_size=args.hr_size,
        train_lr_size=args.hr_size // 4,
        validation_hr_size=args.hr_size,
        validation_lr_size=args.hr_size // 4,
        minimum_heldout_samples=4,
        tiles_per_epoch=1,
        validation_tiles=1,
        clean_epochs=1,
        robust_epochs=1,
        selector_epochs=1,
        use_gradient_checkpointing=False,
    )
    config.validate()
    train = [_batch_for_record(a, config, seed=args.seed + i * 2, device=device)[0]
             for i, (a, _) in enumerate(pairs)]
    holdout = [_batch_for_record(b, config, seed=args.seed + i * 2 + 1, device=device)[0]
               for i, (_, b) in enumerate(pairs)]
    torch.manual_seed(args.seed)
    model = NSAMDRV17(
        config, encoder_channels=args.encoder_channels,
        mid_channels=args.mid_channels, detail_channels=args.detail_channels,
        decoder_blocks=args.decoder_blocks,
    ).to(device)
    model.set_candidate_training()
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, foreach=False)
    for i, batch in enumerate(train):
        if _target_albedo_residual_magnitude(model, batch) < args.minimum_target_residual:
            raise RuntimeError(f"train crop {train_ids[i]} has insufficient B residual")
    for i, batch in enumerate(holdout):
        if _target_albedo_residual_magnitude(model, batch) < args.minimum_target_residual:
            raise RuntimeError(f"holdout crop {holdout_ids[i]} has insufficient B residual")

    resume = _resume_path(repo, str(args.resume))
    start, snapshots = (0, [])
    if resume is not None:
        start, snapshots = _load_checkpoint(resume, model, optimizer, manifest_path,
                                            train_ids, device)
    future = [stage for stage in stages if stage > start]
    if not future:
        raise RuntimeError(f"no stages above resumed update {start}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = repo / ROOT / f"proof_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"V17.2 MULTI-AUTHORITY GENERALIZATION: {train_ids}", flush=True)
    print(f"Untouched siblings: {holdout_ids}", flush=True)
    print(f"Resume: {resume or '<fresh>'} | Start update: {start} | Stages: {future}", flush=True)
    print("Training: one crop/authority; deterministic D4; never optimize on siblings", flush=True)
    started = time.monotonic()
    stage_set = set(future)
    stop = start
    final_train: dict[str, Any] | None = None
    final_siblings: dict[str, Any] | None = None
    sibling_previews: dict[str, str] = {}
    for update in range(start + 1, future[-1] + 1):
        index = (update - 1) % len(train)
        variant = ((update - 1) // len(train)) % 8
        _train_one(model, optimizer, _augment_batch(train[index], variant),
                   config, device=device, precision=args.amp_precision)
        if update == start + 1 or update % 64 == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            eta = (future[-1] - update) * elapsed / max(update - start, 1)
            print(f"[v17-generalization] update {update}/{future[-1]} "
                  f"variant={variant} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                  flush=True)
        if update not in stage_set:
            continue
        trained = [_evaluate(model, sample, config, device=device,
                             precision=args.amp_precision)[0] for sample in train]
        final_train = _score(trained, sibling=False)
        stage = run_dir / f"stage_{update:06d}"
        stage.mkdir(parents=True, exist_ok=True)
        snapshots.append({
            "update": update, "trainedAllPass": final_train["allPass"],
            "trainMean": final_train["mean"], "trainGateChecks": final_train["gateChecks"],
        })
        print(f"Stage {update} train gates all pass: {final_train['allPass']} "
              f"mean={final_train['mean']}", flush=True)
        stop = update
        if final_train["allPass"] or update == future[-1]:
            siblings = []
            for i, sample in enumerate(holdout):
                metrics, output = _evaluate(model, sample, config, device=device,
                                            precision=args.amp_precision)
                siblings.append(metrics)
                sibling_preview = stage / f"sibling_{i:02d}_ABC.png"
                _write_preview(sibling_preview, sample, output, role="sibling")
                sibling_previews[holdout_ids[i]] = str(sibling_preview.resolve())
                if i == 0:
                    _write_preview(stage / "sibling_ABC.png", sample, output, role="sibling")
            final_siblings = _score(siblings, sibling=True)
            print(f"Unseen siblings all pass: {final_siblings['allPass']} "
                  f"mean={final_siblings['mean']}", flush=True)
            break
        _checkpoint(stage / "checkpoint.pt", model, optimizer,
                    manifest_path, train_ids, update, snapshots)

    decision = (
        "sibling-visual-review-required"
        if final_train and final_train["allPass"] and final_siblings and final_siblings["allPass"]
        else "reject-generalization"
    )
    _checkpoint(run_dir / f"stage_{stop:06d}" / "checkpoint.pt",
                model, optimizer, manifest_path, train_ids, stop, snapshots)
    report = {
        "schema": SCHEMA, "architecture": model.architecture_contract(),
        "decision": decision, "experimental": True,
        "trainCropIds": train_ids, "unseenSiblingCropIds": holdout_ids,
        "d4Variants": 8, "startUpdate": start, "stopUpdate": stop,
        "requestedStages": future, "resumeSource": str(resume or ""),
        "candidateThresholds": {
            "globalRecovery": config.candidate_global_recovery_required,
            "edgeRecovery": config.candidate_edge_recovery_required,
            "gradientRecovery": config.candidate_gradient_recovery_required,
        },
        "earlySiblingTransferThresholds": EARLY_TRANSFER_GATE,
        "trained": final_train, "siblings": final_siblings,
        "snapshots": snapshots,
        "visualGateRequired": decision == "sibling-visual-review-required",
        "siblingPreviews": sibling_previews,
        "preview": str((run_dir / f"stage_{stop:06d}" / "sibling_ABC.png").resolve()),
        "elapsedSeconds": time.monotonic() - started,
    }
    path = run_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Decision: {decision} | Report: {path}", flush=True)
    return 0, path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--manifest", default=DEFAULT_MANIFEST)
    value.add_argument("--authority-id", default="13006d2b807f89ac")
    value.add_argument("--authorities", type=int, default=4)
    value.add_argument("--hr-size", type=int, default=512)
    value.add_argument("--stages", nargs="+", default=["2048,4096,6144,8192"])
    value.add_argument("--resume", default="auto")
    value.add_argument("--seed", type=int, default=17001)
    value.add_argument("--learning-rate", type=float, default=2e-4)
    value.add_argument("--minimum-target-residual", type=float, default=0.01)
    value.add_argument("--encoder-channels", type=int, default=96)
    value.add_argument("--mid-channels", type=int, default=64)
    value.add_argument("--detail-channels", type=int, default=48)
    value.add_argument("--decoder-blocks", type=int, default=4)
    value.add_argument("--device", choices=("cpu", "auto", "cuda"), default="cuda")
    value.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    return value


def main(argv: list[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
