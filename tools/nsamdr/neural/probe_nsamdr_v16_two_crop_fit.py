#!/usr/bin/env python3
"""Two-crop-per-authority fit and held-out transfer diagnostic for NSAMDR V16.

The 16-authority one-crop fit passes train-fit gates but drops sharply on unseen
sibling crops. This probe keeps the same 16 authority identities, trains two
fixed authored crops per authority from the original broad source checkpoint,
and evaluates all 38 complete held-out authorities at bounded stages.

Stage units are updates per crop. With 16 authorities and 2 crops each,
224 updates per crop is 7168 total optimizer updates: the same total update
budget as the prior 16-authority one-crop run at 448 updates per authority.

This is a diagnostic only and is not production qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import numpy as np
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
from probe_nsamdr_v16_memorization import (
    _evaluate_exact,
    _gate_snapshot,
    _parse_stages,
    _train_one,
)
from probe_nsamdr_v16_sibling_crop_transfer import (
    _authority_records,
    _batch_for_record,
)


SCHEMA = "NSAMDR_V16_TWO_CROP_FIT_TRANSFER_PROBE_V1"
CHECKPOINT_SCHEMA = "NSAMDR_V16_TWO_CROP_FIT_TRANSFER_CHECKPOINT_V1"
METRIC_KEYS = (
    "global_recovery",
    "edge_recovery",
    "gradient_recovery",
    "normal_recovery",
    "lattice_cell_excess",
    "protected_preservation",
    "detail_recovery_1px",
    "detail_recovery_2px",
    "detail_recovery_4px",
)
DIAGNOSTIC_KEYS = (
    "candidate_to_target_residual_ratio",
    "residual_cosine_similarity",
    "target_weighted_sign_agreement",
    "least_squares_residual_gain",
)


def _distribution(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray(
        [float(value) for value in values if np.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


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


def _select_fixed_crops(
    manifest: dict[str, Any],
    authority_ids: list[str],
    *,
    crops_per_authority: int,
) -> list[dict[str, Any]]:
    if crops_per_authority < 1:
        raise ValueError("crops-per-authority must be positive")
    selected: list[dict[str, Any]] = []
    for authority_id in authority_ids:
        records = _authority_records(manifest, authority_id)
        if len(records) < crops_per_authority:
            raise RuntimeError(
                f"authority {authority_id} has {len(records)} crop(s); "
                f"{crops_per_authority} required"
            )
        for crop_index, record in enumerate(records[:crops_per_authority]):
            selected.append(
                {
                    "authorityId": authority_id,
                    "cropIndex": int(crop_index),
                    "record": record,
                }
            )
    return selected


def _attach_batches(
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


def _evaluate_train_samples(
    model,
    selected: list[dict[str, Any]],
    config,
    *,
    device: torch.device,
    precision: str,
    authority_count: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in selected:
        evaluated = _evaluate_exact(
            model,
            item["batch"],
            config,
            device=device,
            precision=precision,
        )
        metrics = dict(evaluated["metrics"])
        metrics["heldout_sample"] = 0.0
        evaluated = {**evaluated, "metrics": metrics}
        rows.append(
            {
                "authorityId": item["authorityId"],
                "cropId": item["sample"]["cropId"],
                "cropIndex": item["cropIndex"],
                **evaluated,
                "gateChecks": _gate_snapshot(evaluated, config),
            }
        )

    metric_distributions = {
        key: _distribution([float(row["metrics"][key]) for row in rows])
        for key in METRIC_KEYS
    }
    diagnostic_distributions = {
        key: _distribution(
            [float(row["residualDiagnostics"][key]) for row in rows]
        )
        for key in DIAGNOSTIC_KEYS
    }
    gate_pass_counts = {
        key: sum(bool(row["gateChecks"][key]) for row in rows)
        for key in ("global", "edge", "gradient", "lattice")
    }
    median_gate_checks = {
        "global": float(metric_distributions["global_recovery"]["median"])
        >= float(config.candidate_global_recovery_required),
        "edge": float(metric_distributions["edge_recovery"]["median"])
        >= float(config.candidate_edge_recovery_required),
        "gradient": float(metric_distributions["gradient_recovery"]["median"])
        >= float(config.candidate_gradient_recovery_required),
        "lattice": float(metric_distributions["lattice_cell_excess"]["median"])
        <= float(config.candidate_lattice_cell_excess_max),
    }
    return {
        "sampleCount": len(rows),
        "authorityCount": int(authority_count),
        "metricDistributions": metric_distributions,
        "residualDiagnosticDistributions": diagnostic_distributions,
        "sampleGatePassCounts": gate_pass_counts,
        "medianGateChecks": median_gate_checks,
        "allMedianGatesPass": bool(all(median_gate_checks.values())),
        "perSample": rows,
    }


def _summary_median(summary: dict[str, Any], key: str) -> float:
    direct = f"median_{key}"
    if direct in summary:
        return float(summary[direct])
    distributions = dict(summary.get("metricDistributions") or {})
    item = distributions.get(key)
    if isinstance(item, dict) and "median" in item:
        return float(item["median"])
    raise KeyError(f"summary has no median for metric {key!r}")


def _heldout_delta(
    source: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    keys = (
        "global_recovery",
        "edge_recovery",
        "gradient_recovery",
        "normal_recovery",
        "lattice_cell_excess",
        "detail_recovery_1px",
        "detail_recovery_2px",
        "detail_recovery_4px",
    )
    return {
        key: _summary_median(candidate, key) - _summary_median(source, key)
        for key in keys
    }


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
            "sourceCheckpoint": str(source_checkpoint.resolve()),
            "sourceCheckpointStep": int(source_step),
            "manifest": str(manifest_path.resolve()),
            "authorityReference": str(authority_reference.resolve()),
            "authorityIds": list(authority_ids),
            "cropIds": list(crop_ids),
            "seed": int(seed),
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
        raise RuntimeError(f"two-crop continuation checkpoint schema mismatch: {path}")
    checks = (
        (Path(str(payload.get("sourceCheckpoint", ""))).resolve(), source_checkpoint.resolve(), "source checkpoint"),
        (int(payload.get("sourceCheckpointStep", -1)), int(source_step), "source step"),
        (Path(str(payload.get("manifest", ""))).resolve(), manifest_path.resolve(), "manifest"),
        (Path(str(payload.get("authorityReference", ""))).resolve(), authority_reference.resolve(), "authority reference"),
        (list(payload.get("authorityIds") or []), list(authority_ids), "authority ids"),
        (list(payload.get("cropIds") or []), list(crop_ids), "crop ids"),
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
    selected = _attach_batches(
        selected_records,
        manifest,
        config,
        device=device,
        seed=seed + 15001,
    )
    sample_count = len(selected)
    crop_ids = [item["sample"]["cropId"] for item in selected]

    optimizer = _optimizer(model, config)
    stages_per_crop = _parse_stages(args.stages_per_crop)
    max_per_crop = stages_per_crop[-1]
    max_total_update = max_per_crop * sample_count
    stage_totals = {
        int(stage) * sample_count: int(stage)
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

    output_dir = source_checkpoint.parent / "two_crop_fit"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{authority_count:02d}_authorities_{crops_per_authority:02d}_crops_step_{source_step:06d}"
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

    print("NSAMDR V16 TWO-CROP FIT + HELD-OUT TRANSFER PROBE", flush=True)
    print(f"Source step        : {source_step}", flush=True)
    print(f"Authorities        : {authority_count}", flush=True)
    print(f"Crops / authority  : {crops_per_authority}", flush=True)
    print(f"Fixed train samples: {sample_count}", flush=True)
    print(f"Stages / crop      : {stages_per_crop}", flush=True)
    print(f"Max total updates  : {max_total_update}", flush=True)
    print("Optimizer           : fresh Adam state", flush=True)
    print("Sampling            : deterministic round-robin fixed clean crops", flush=True)

    started = time.monotonic()
    segment_total = max_total_update - start_total_update
    for total_update in range(start_total_update + 1, max_total_update + 1):
        sample_index = (total_update - 1) % sample_count
        _train_one(
            model,
            optimizer,
            selected[sample_index]["batch"],
            config,
            device=device,
            precision=args.amp_precision,
        )

        if total_update == 1 or total_update % int(args.progress_every) == 0:
            elapsed = max(time.monotonic() - started, 1.0e-6)
            segment_done = total_update - start_total_update
            rate = segment_done / elapsed
            eta = (max_total_update - total_update) / max(rate, 1.0e-6)
            print(
                f"Progress           : total={total_update}/{max_total_update} "
                f"~per-crop={total_update // sample_count}/{max_per_crop} "
                f"elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m",
                flush=True,
            )

        should_recovery_save = (
            total_update % max(1, int(args.checkpoint_every)) == 0
            or total_update in stage_totals
            or total_update == max_total_update
        )
        if should_recovery_save:
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
                updates_per_crop=total_update // sample_count,
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
            "trainFixedCrops": train_eval,
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
            f"1px={_summary_median(heldout_eval, 'detail_recovery_1px')*100:+.2f}% "
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
        "diagnosticRole": "two-fixed-crops-per-authority-fit-and-heldout-transfer",
        "qualificationEligible": False,
        "qualificationExclusionReasons": [
            "bounded-train-fit-diagnostic",
            "fixed-crop-schedule",
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
        "trainingPolicy": "deterministic-round-robin-two-fixed-clean-crops-per-authority",
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
    print(f"Report             : {output_path}", flush=True)
    print(f"Checkpoint         : {continuation_path}", flush=True)
    return 0, output_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Train two fixed crops per selected authority at a matched update "
            "budget and evaluate all complete held-out authorities."
        )
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--manifest", default=DEFAULT_MANIFEST)
    value.add_argument("--resume", required=True, help="source full-broad checkpoint")
    value.add_argument(
        "--authority-reference",
        required=True,
        help="16-authority interference checkpoint providing the controlled authority set",
    )
    value.add_argument("--authority-count", type=int, default=16)
    value.add_argument("--crops-per-authority", type=int, default=2)
    value.add_argument(
        "--stages-per-crop",
        nargs="+",
        default=["112", "224"],
        help="cumulative updates per crop; 224 with 16x2 crops matches 7168 prior updates",
    )
    value.add_argument("--progress-every", type=int, default=32)
    value.add_argument(
        "--checkpoint-every",
        type=int,
        default=512,
        help="overwrite the recovery checkpoint every N total optimizer updates",
    )
    value.add_argument(
        "--continue-from",
        default="",
        help="resume model/optimizer/update state from a two-crop checkpoint",
    )
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
