#!/usr/bin/env python3
"""Augmented two-crop fit and held-out transfer diagnostic for NSAMDR V16.

The fixed two-crop diagnostic shows that repeated fitting of 32 exact crops
continues to improve train recovery while held-out recovery plateaus/regresses.
This probe keeps the same 16 authorities, the same two authored crops per
authority, and the same optimizer/update budget, but cycles every crop through
all eight D4 orientation transforms during training.

The transform bank is applied in memory to the already loaded clean LR/HR map
tensors. Normal XY vectors are rotated/reflected with the image so map semantics
remain valid.

This is a diagnostic only and is not production qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from probe_nsamdr_v16_full_broad import (
    DEFAULT_MANIFEST,
    _device,
    _evaluate,
    _load_checkpoint,
    _optimizer,
)
from probe_nsamdr_v16_interference import (
    CHECKPOINT_SCHEMA as INTERFERENCE_CHECKPOINT_SCHEMA,
)
from probe_nsamdr_v16_memorization import _parse_stages, _train_one
from probe_nsamdr_v16_sibling_crop_transfer import _batch_for_record
from probe_nsamdr_v16_two_crop_fit import (
    _authority_records,
    _evaluate_train_samples,
    _heldout_delta,
    _select_fixed_crops,
)


SCHEMA = "NSAMDR_V16_AUGMENTED_TWO_CROP_FIT_TRANSFER_PROBE_V1"
CHECKPOINT_SCHEMA = "NSAMDR_V16_AUGMENTED_TWO_CROP_FIT_TRANSFER_CHECKPOINT_V1"
PROBE_REVISION = "augmented-two-crop-r1"
AUGMENTATION_VARIANTS = 8


def _spatial_transform(value: torch.Tensor, turns: int, mirror_x: bool) -> torch.Tensor:
    result = torch.rot90(value, int(turns), dims=(-2, -1)) if int(turns) else value
    if mirror_x:
        result = torch.flip(result, dims=(-1,))
    return result.contiguous()


def _normal_transform(value: torch.Tensor, turns: int, mirror_x: bool) -> torch.Tensor:
    result = torch.rot90(value, int(turns), dims=(-2, -1)) if int(turns) else value
    x = result[:, 0:1].clone()
    y = result[:, 1:2].clone()
    for _ in range(int(turns) % 4):
        x, y = -y, x
    result = torch.cat((x, y), dim=1)
    if mirror_x:
        result = torch.flip(result, dims=(-1,)).contiguous()
        result[:, 0:1] = -result[:, 0:1]
    return result.contiguous()


def _augment_batch(
    batch: dict[str, torch.Tensor],
    variant: int,
) -> dict[str, torch.Tensor]:
    value = int(variant)
    if value < 0 or value >= AUGMENTATION_VARIANTS:
        raise ValueError(f"augmentation variant must be 0..{AUGMENTATION_VARIANTS - 1}")
    turns = value % 4
    mirror_x = value >= 4
    result: dict[str, torch.Tensor] = {}
    for key, tensor in batch.items():
        if key in {"lr_albedo", "lr_material", "target_albedo", "target_material"}:
            result[key] = _spatial_transform(tensor, turns, mirror_x)
        elif key in {"lr_normal", "target_normal"}:
            result[key] = _normal_transform(tensor, turns, mirror_x)
        else:
            result[key] = tensor
    return result


def _load_authority_reference(
    path: Path,
    *,
    source_checkpoint: Path,
    source_step: int,
    manifest_path: Path,
) -> list[str]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != INTERFERENCE_CHECKPOINT_SCHEMA:
        raise RuntimeError("authority reference is not an interference checkpoint")
    if Path(str(payload.get("sourceCheckpoint", ""))).resolve() != source_checkpoint.resolve():
        raise RuntimeError("authority reference belongs to a different source checkpoint")
    if int(payload.get("sourceCheckpointStep", -1)) != int(source_step):
        raise RuntimeError("authority reference source step mismatch")
    if Path(str(payload.get("manifest", ""))).resolve() != manifest_path.resolve():
        raise RuntimeError("authority reference belongs to a different manifest")
    values = [str(value) for value in payload.get("authorityIds") or []]
    if not values:
        raise RuntimeError("authority reference has no authorityIds")
    return values


def _attach_base_batches(
    selected: list[dict[str, Any]],
    manifest: dict[str, Any],
    config,
    *,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, item in enumerate(selected):
        batch, sample = _batch_for_record(
            item["record"],
            manifest,
            config,
            seed=seed + index,
            device=device,
        )
        result.append({**item, "batch": batch, "sample": sample})
    return result


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer: torch.optim.Optimizer,
    source_checkpoint: Path,
    source_step: int,
    manifest_path: Path,
    authority_reference: Path,
    authority_ids: list[str],
    crop_ids: list[str],
    seed: int,
    total_update: int,
    updates_per_crop: int,
    snapshots: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "probeRevision": PROBE_REVISION,
            "sourceCheckpoint": str(source_checkpoint.resolve()),
            "sourceCheckpointStep": int(source_step),
            "manifest": str(manifest_path.resolve()),
            "authorityReference": str(authority_reference.resolve()),
            "authorityIds": list(authority_ids),
            "cropIds": list(crop_ids),
            "seed": int(seed),
            "augmentationVariants": int(AUGMENTATION_VARIANTS),
            "totalUpdate": int(total_update),
            "updatesPerCrop": int(updates_per_crop),
            "modelState": model.state_dict(),
            "optimizerState": optimizer.state_dict(),
            "snapshots": snapshots,
        },
        path,
    )


def _load_continuation(
    path: Path,
    *,
    model,
    optimizer: torch.optim.Optimizer,
    source_checkpoint: Path,
    source_step: int,
    manifest_path: Path,
    authority_reference: Path,
    authority_ids: list[str],
    crop_ids: list[str],
    device: torch.device,
) -> tuple[int, int, list[dict[str, Any]]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"augmented continuation checkpoint schema mismatch: {path}")
    checks = (
        (Path(str(payload.get("sourceCheckpoint", ""))).resolve(), source_checkpoint.resolve(), "source checkpoint"),
        (int(payload.get("sourceCheckpointStep", -1)), int(source_step), "source step"),
        (Path(str(payload.get("manifest", ""))).resolve(), manifest_path.resolve(), "manifest"),
        (Path(str(payload.get("authorityReference", ""))).resolve(), authority_reference.resolve(), "authority reference"),
        (list(payload.get("authorityIds") or []), list(authority_ids), "authority ids"),
        (list(payload.get("cropIds") or []), list(crop_ids), "crop ids"),
        (int(payload.get("augmentationVariants", -1)), int(AUGMENTATION_VARIANTS), "augmentation variants"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise RuntimeError(f"continuation checkpoint {label} mismatch")
    model.load_state_dict(payload["modelState"], strict=True)
    optimizer.load_state_dict(payload["optimizerState"])
    return (
        int(payload.get("totalUpdate", 0)),
        int(payload.get("updatesPerCrop", 0)),
        list(payload.get("snapshots") or []),
    )


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    source_checkpoint = Path(args.resume)
    if not source_checkpoint.is_absolute():
        source_checkpoint = (repo_root / source_checkpoint).resolve()
    authority_reference = Path(args.authority_reference)
    if not authority_reference.is_absolute():
        authority_reference = (repo_root / authority_reference).resolve()

    for label, path in (
        ("manifest", manifest_path),
        ("source checkpoint", source_checkpoint),
        ("authority reference", authority_reference),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = _device(args.device)
    model, _loaded_optimizer, config, source_step, seed, _curve = _load_checkpoint(
        source_checkpoint,
        device=device,
        manifest_path=manifest_path,
    )
    authority_ids = _load_authority_reference(
        authority_reference,
        source_checkpoint=source_checkpoint,
        source_step=source_step,
        manifest_path=manifest_path,
    )
    if int(args.authority_count) > 0:
        authority_ids = authority_ids[: int(args.authority_count)]
    authority_count = len(authority_ids)
    crops_per_authority = int(args.crops_per_authority)

    selected_records = _select_fixed_crops(
        manifest,
        authority_ids,
        crops_per_authority=crops_per_authority,
    )
    selected = _attach_base_batches(
        selected_records,
        manifest,
        config,
        device=device,
        seed=seed + 17001,
    )
    base_sample_count = len(selected)
    crop_ids = [item["sample"]["cropId"] for item in selected]

    optimizer = _optimizer(model, config)
    stages_per_crop = _parse_stages(args.stages_per_crop)
    max_per_crop = stages_per_crop[-1]
    max_total_update = max_per_crop * base_sample_count
    stage_totals = {
        int(stage) * base_sample_count: int(stage)
        for stage in stages_per_crop
    }

    validation_families = len(
        {
            str(record.get("family_id") or record.get("familyId") or "")
            for record in manifest.get("crops", [])
            if str(record.get("split")) == "validation"
        }
    )
    source_heldout = _evaluate(
        model,
        manifest,
        config,
        split="validation",
        samples=validation_families,
        device=device,
        seed=seed,
        precision=args.amp_precision,
    )

    output_dir = source_checkpoint.parent / "augmented_two_crop_fit"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{authority_count:02d}_authorities_{crops_per_authority:02d}_crops_"
        f"d4_step_{source_step:06d}"
    )
    output_path = output_dir / f"{stem}.json"
    continuation_path = output_dir / f"{stem}_checkpoint.pt"

    snapshots: list[dict[str, Any]] = []
    start_total_update = 0
    if args.continue_from:
        continue_from = Path(args.continue_from)
        if not continue_from.is_absolute():
            continue_from = (repo_root / continue_from).resolve()
        if not continue_from.is_file():
            raise RuntimeError(f"continuation checkpoint is missing: {continue_from}")
        start_total_update, resumed_per_crop, snapshots = _load_continuation(
            continue_from,
            model=model,
            optimizer=optimizer,
            source_checkpoint=source_checkpoint,
            source_step=source_step,
            manifest_path=manifest_path,
            authority_reference=authority_reference,
            authority_ids=authority_ids,
            crop_ids=crop_ids,
            device=device,
        )
        if start_total_update > max_total_update:
            raise RuntimeError("continuation checkpoint is beyond requested final stage")
        print(
            f"Resume             : total={start_total_update} "
            f"per-crop={resumed_per_crop}",
            flush=True,
        )

    print("NSAMDR V16 AUGMENTED TWO-CROP FIT + HELD-OUT TRANSFER", flush=True)
    print(f"Probe revision      : {PROBE_REVISION}", flush=True)
    print(f"Source step         : {source_step}", flush=True)
    print(f"Authorities         : {authority_count}", flush=True)
    print(f"Crops / authority   : {crops_per_authority}", flush=True)
    print(f"Base train samples  : {base_sample_count}", flush=True)
    print(f"D4 variants / crop  : {AUGMENTATION_VARIANTS}", flush=True)
    print(f"Stages / crop       : {stages_per_crop}", flush=True)
    print(f"Max total updates   : {max_total_update}", flush=True)
    print("Optimizer            : fresh Adam state unless resumed", flush=True)
    print("Degradation          : clean", flush=True)

    started = time.monotonic()
    for total_update in range(start_total_update + 1, max_total_update + 1):
        base_index = (total_update - 1) % base_sample_count
        crop_cycle = (total_update - 1) // base_sample_count
        variant = crop_cycle % AUGMENTATION_VARIANTS
        batch = _augment_batch(selected[base_index]["batch"], variant)
        _train_one(
            model,
            optimizer,
            batch,
            config,
            device=device,
            precision=args.amp_precision,
        )

        if total_update == start_total_update + 1 or total_update % int(args.progress_every) == 0:
            elapsed = max(time.monotonic() - started, 1.0e-6)
            segment_done = total_update - start_total_update
            rate = segment_done / elapsed
            eta = (max_total_update - total_update) / max(rate, 1.0e-6)
            print(
                f"Progress            : total={total_update}/{max_total_update} "
                f"~per-crop={total_update // base_sample_count}/{max_per_crop} "
                f"variant={variant}/7 elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m",
                flush=True,
            )

        should_save = (
            total_update % max(1, int(args.checkpoint_every)) == 0
            or total_update in stage_totals
            or total_update == max_total_update
        )
        if should_save:
            _save_checkpoint(
                continuation_path,
                model=model,
                optimizer=optimizer,
                source_checkpoint=source_checkpoint,
                source_step=source_step,
                manifest_path=manifest_path,
                authority_reference=authority_reference,
                authority_ids=authority_ids,
                crop_ids=crop_ids,
                seed=seed,
                total_update=total_update,
                updates_per_crop=total_update // base_sample_count,
                snapshots=snapshots,
            )

        if total_update not in stage_totals:
            continue

        updates_per_crop = stage_totals[total_update]
        train_eval = _evaluate_train_samples(
            model,
            selected,
            config,
            device=device,
            precision=args.amp_precision,
            authority_count=authority_count,
        )
        heldout_eval = _evaluate(
            model,
            manifest,
            config,
            split="validation",
            samples=validation_families,
            device=device,
            seed=seed,
            precision=args.amp_precision,
        )
        delta = _heldout_delta(source_heldout, heldout_eval)
        snapshot = {
            "updatesPerCrop": int(updates_per_crop),
            "updatesPerAuthority": int(updates_per_crop * crops_per_authority),
            "totalUpdate": int(total_update),
            "augmentationVariants": int(AUGMENTATION_VARIANTS),
            "trainFixedOrientationEvaluation": train_eval,
            "heldout": heldout_eval,
            "heldoutDeltaVsSource": delta,
        }
        snapshots.append(snapshot)

        tm = train_eval["metricDistributions"]
        print(
            f"Per-crop {updates_per_crop:4d} train : "
            f"global={tm['global_recovery']['median']*100:+.2f}% "
            f"edge={tm['edge_recovery']['median']*100:+.2f}% "
            f"grad={tm['gradient_recovery']['median']*100:+.2f}% "
            f"1px={tm['detail_recovery_1px']['median']*100:+.2f}% "
            f"lattice={tm['lattice_cell_excess']['median']*100:+.2f}%",
            flush=True,
        )
        print(
            f"Per-crop {updates_per_crop:4d} held  : "
            f"global={heldout_eval['median_global_recovery']*100:+.2f}% "
            f"edge={heldout_eval['median_edge_recovery']*100:+.2f}% "
            f"grad={heldout_eval['median_gradient_recovery']*100:+.2f}% "
            f"1px={heldout_eval['metricDistributions']['detail_recovery_1px']['median']*100:+.2f}% "
            f"lattice={heldout_eval['median_lattice_cell_excess']*100:+.2f}%",
            flush=True,
        )

        _save_checkpoint(
            continuation_path,
            model=model,
            optimizer=optimizer,
            source_checkpoint=source_checkpoint,
            source_step=source_step,
            manifest_path=manifest_path,
            authority_reference=authority_reference,
            authority_ids=authority_ids,
            crop_ids=crop_ids,
            seed=seed,
            total_update=total_update,
            updates_per_crop=updates_per_crop,
            snapshots=snapshots,
        )

    report = {
        "schema": SCHEMA,
        "probeRevision": PROBE_REVISION,
        "diagnosticRole": "d4-augmented-two-crop-fit-and-heldout-transfer",
        "qualificationEligible": False,
        "qualificationExclusionReasons": [
            "bounded-train-fit-diagnostic",
            "restricted-authority-set",
            "not-production-broad-training-schedule",
            "material-semantics-unresolved",
            "selector-not-qualified",
        ],
        "manifest": str(manifest_path),
        "sourceCheckpoint": str(source_checkpoint),
        "sourceCheckpointStep": int(source_step),
        "authorityReference": str(authority_reference),
        "authorityIds": authority_ids,
        "cropsPerAuthority": crops_per_authority,
        "cropIds": crop_ids,
        "augmentationVariants": int(AUGMENTATION_VARIANTS),
        "augmentationPolicy": "deterministic-D4-rotations-and-mirror-with-normal-vector-transform",
        "trainingPolicy": "round-robin-two-authored-crops-with-D4-variant-cycling",
        "optimizerPolicy": "fresh-adam-state",
        "stagesPerCrop": stages_per_crop,
        "startTotalUpdate": int(start_total_update),
        "sourceHeldout": source_heldout,
        "snapshots": snapshots,
        "final": snapshots[-1],
        "continuationCheckpoint": str(continuation_path),
        "sourceCheckpointModified": False,
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Report              : {output_path}", flush=True)
    print(f"Checkpoint          : {continuation_path}", flush=True)
    return 0, output_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Train the same 16 authorities and two crops through all eight D4 "
            "orientation transforms, then evaluate complete held-out authorities."
        )
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--manifest", default=DEFAULT_MANIFEST)
    value.add_argument("--resume", required=True, help="source full-broad checkpoint")
    value.add_argument("--authority-reference", required=True)
    value.add_argument("--authority-count", type=int, default=16)
    value.add_argument("--crops-per-authority", type=int, default=2)
    value.add_argument(
        "--stages-per-crop",
        nargs="+",
        default=["112", "224"],
    )
    value.add_argument("--progress-every", type=int, default=32)
    value.add_argument("--checkpoint-every", type=int, default=512)
    value.add_argument("--continue-from", default="")
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
