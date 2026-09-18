#!/usr/bin/env python3
"""Fixed-crop multi-authority interference diagnostic for NSAMDR V16.

This probe starts from an existing full-broad checkpoint, chooses a deterministic
set of authored train authorities, fixes one center crop per authority with no
augmentation, resets Adam state, and trains them round-robin.

The stage unit is updates per authority. For four authorities, stage 128 means
512 total optimizer updates. Evaluation always covers every selected authority.

Purpose:
- compare fixed single-authority memorization with fixed multi-authority fitting;
- separate cross-authority interference/optimization from representation capacity;
- keep the experiment deterministic, bounded, and resumable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
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
    _load_checkpoint,
    _optimizer,
)
from probe_nsamdr_v16_memorization import (
    _evaluate_exact,
    _fixed_batch,
    _gate_snapshot,
    _parse_stages,
    _train_one,
)


SCHEMA = "NSAMDR_V16_AUTHORITY_INTERFERENCE_PROBE_V1"
CHECKPOINT_SCHEMA = "NSAMDR_V16_AUTHORITY_INTERFERENCE_CHECKPOINT_V1"
METRIC_KEYS = (
    "global_recovery",
    "edge_recovery",
    "gradient_recovery",
    "normal_recovery",
    "lattice_cell_excess",
    "protected_preservation",
)
DIAGNOSTIC_KEYS = (
    "candidate_to_target_residual_ratio",
    "residual_cosine_similarity",
    "target_weighted_sign_agreement",
    "least_squares_residual_gain",
)


def _train_authority_ids(manifest: dict[str, Any]) -> list[str]:
    values = {
        str(record.get("family_id") or record.get("familyId") or "").strip()
        for record in manifest.get("crops", [])
        if str(record.get("split")) == "train"
    }
    values.discard("")
    return sorted(values)


def _candidate_authority_order(
    manifest: dict[str, Any],
    *,
    anchor_authority: str,
    seed: int,
) -> list[str]:
    authorities = _train_authority_ids(manifest)
    anchor = str(anchor_authority).strip()
    if anchor and anchor not in authorities:
        raise RuntimeError(f"anchor authority not found in train split: {anchor}")
    remainder = [value for value in authorities if value != anchor]
    random.Random(int(seed) + 9201).shuffle(remainder)
    return ([anchor] if anchor else []) + remainder


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


def _aggregate(
    rows: list[dict[str, Any]],
    config,
) -> dict[str, Any]:
    metrics = {
        key: _distribution([float(row["metrics"][key]) for row in rows])
        for key in METRIC_KEYS
    }
    diagnostics = {
        key: _distribution(
            [float(row["residualDiagnostics"][key]) for row in rows]
        )
        for key in DIAGNOSTIC_KEYS
    }
    pass_counts = {
        key: sum(bool(row["gateChecks"][key]) for row in rows)
        for key in ("global", "edge", "gradient", "lattice")
    }
    count = len(rows)
    median_gate_checks = {
        "global": float(metrics["global_recovery"]["median"])
        >= float(config.candidate_global_recovery_required),
        "edge": float(metrics["edge_recovery"]["median"])
        >= float(config.candidate_edge_recovery_required),
        "gradient": float(metrics["gradient_recovery"]["median"])
        >= float(config.candidate_gradient_recovery_required),
        "lattice": float(metrics["lattice_cell_excess"]["median"])
        <= float(config.candidate_lattice_cell_excess_max),
    }
    return {
        "authorityCount": count,
        "metricDistributions": metrics,
        "residualDiagnosticDistributions": diagnostics,
        "authorityGatePassCounts": pass_counts,
        "authorityGatePassFractions": {
            key: (float(value) / float(count) if count else 0.0)
            for key, value in pass_counts.items()
        },
        "medianGateChecks": median_gate_checks,
        "allMedianGatesPass": bool(all(median_gate_checks.values())),
        "allAuthoritiesPassAllCandidateGates": bool(
            rows and all(all(row["gateChecks"].values()) for row in rows)
        ),
    }


def _evaluate_all(
    model,
    selected: list[dict[str, Any]],
    config,
    *,
    device: torch.device,
    precision: str,
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
        rows.append(
            {
                "authorityId": item["sample"]["authorityId"],
                "cropId": item["sample"]["cropId"],
                **evaluated,
                "gateChecks": _gate_snapshot(evaluated, config),
            }
        )
    return {
        "perAuthority": rows,
        "aggregate": _aggregate(rows, config),
    }


def _select_authorities(
    model,
    manifest: dict[str, Any],
    config,
    *,
    authority_count: int,
    anchor_authority: str,
    seed: int,
    minimum_target_residual: float,
    device: torch.device,
    precision: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if authority_count < 1:
        raise ValueError("authority-count must be positive")

    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for authority_id in _candidate_authority_order(
        manifest,
        anchor_authority=anchor_authority,
        seed=seed,
    ):
        batch, sample = _fixed_batch(
            manifest,
            config,
            authority_id,
            seed=seed + 9001,
            device=device,
        )
        evaluated = _evaluate_exact(
            model,
            batch,
            config,
            device=device,
            precision=precision,
        )
        target_residual = float(
            evaluated["residualDiagnostics"]["target_residual_magnitude"]
        )
        if target_residual < float(minimum_target_residual):
            rejected.append(
                {
                    "authorityId": authority_id,
                    "cropId": sample["cropId"],
                    "targetResidualMagnitude": target_residual,
                    "reason": "target-residual-below-minimum",
                }
            )
            continue
        selected.append(
            {
                "authorityId": authority_id,
                "batch": batch,
                "sample": sample,
                "initial": evaluated,
            }
        )
        if len(selected) >= authority_count:
            break

    if len(selected) != authority_count:
        raise RuntimeError(
            f"only {len(selected)} non-trivial authorities available; "
            f"{authority_count} requested"
        )
    return selected, rejected


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer: torch.optim.Optimizer,
    source_checkpoint: Path,
    source_step: int,
    manifest_path: Path,
    authority_ids: list[str],
    seed: int,
    total_update: int,
    updates_per_authority: int,
    snapshots: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "sourceCheckpoint": str(source_checkpoint.resolve()),
            "sourceCheckpointStep": int(source_step),
            "manifest": str(manifest_path.resolve()),
            "authorityIds": list(authority_ids),
            "seed": int(seed),
            "totalUpdate": int(total_update),
            "updatesPerAuthority": int(updates_per_authority),
            "modelState": model.state_dict(),
            "optimizerState": optimizer.state_dict(),
            "snapshots": snapshots,
        },
        path,
    )


def _load_interference_checkpoint(
    path: Path,
    *,
    model,
    optimizer: torch.optim.Optimizer,
    source_checkpoint: Path,
    source_step: int,
    manifest_path: Path,
    authority_ids: list[str],
    device: torch.device,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"interference checkpoint schema mismatch: {path}")
    if Path(str(payload.get("sourceCheckpoint", ""))).resolve() != source_checkpoint.resolve():
        raise RuntimeError("interference checkpoint belongs to a different source checkpoint")
    if int(payload.get("sourceCheckpointStep", -1)) != int(source_step):
        raise RuntimeError("interference checkpoint source step mismatch")
    if Path(str(payload.get("manifest", ""))).resolve() != manifest_path.resolve():
        raise RuntimeError("interference checkpoint belongs to a different manifest")
    if list(payload.get("authorityIds") or []) != list(authority_ids):
        raise RuntimeError("interference checkpoint authority set/order mismatch")
    model.load_state_dict(payload["modelState"], strict=True)
    optimizer.load_state_dict(payload["optimizerState"])
    return (
        int(payload["totalUpdate"]),
        int(payload["updatesPerAuthority"]),
        int(payload["seed"]),
        list(payload.get("snapshots") or []),
    )


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest is missing: {manifest_path}")

    checkpoint_path = Path(args.resume)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (repo_root / checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(f"checkpoint is missing: {checkpoint_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = _device(args.device)
    (
        model,
        _loaded_optimizer,
        config,
        source_step,
        seed,
        _curve,
    ) = _load_checkpoint(
        checkpoint_path,
        device=device,
        manifest_path=manifest_path,
    )

    selected, rejected = _select_authorities(
        model,
        manifest,
        config,
        authority_count=int(args.authority_count),
        anchor_authority=args.anchor_authority,
        seed=seed,
        minimum_target_residual=float(args.minimum_target_residual),
        device=device,
        precision=args.amp_precision,
    )
    authority_ids = [item["authorityId"] for item in selected]
    authority_count = len(selected)

    optimizer = _optimizer(model, config)
    stages_per_authority = _parse_stages(args.stages_per_authority)
    max_per_authority = stages_per_authority[-1]
    max_total_update = max_per_authority * authority_count
    start_total_update = 0
    start_per_authority = 0
    snapshots: list[dict[str, Any]] = []

    output_dir = checkpoint_path.parent / "interference"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{authority_count:02d}_authorities_step_{source_step:06d}"
    report_path = output_dir / f"{stem}.json"
    continuation_path = output_dir / f"{stem}_checkpoint.pt"

    if args.continue_from:
        resume_path = Path(args.continue_from)
        if not resume_path.is_absolute():
            resume_path = (repo_root / resume_path).resolve()
        if not resume_path.is_file():
            raise RuntimeError(
                f"interference continuation checkpoint is missing: {resume_path}"
            )
        (
            start_total_update,
            start_per_authority,
            continuation_seed,
            snapshots,
        ) = _load_interference_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            source_checkpoint=checkpoint_path,
            source_step=source_step,
            manifest_path=manifest_path,
            authority_ids=authority_ids,
            device=device,
        )
        if continuation_seed != seed:
            raise RuntimeError("interference checkpoint seed mismatch")
        if max_total_update <= start_total_update:
            raise RuntimeError(
                "no requested stage exceeds the continuation checkpoint"
            )

    torch.manual_seed(seed + 9301)

    initial = _evaluate_all(
        model,
        selected,
        config,
        device=device,
        precision=args.amp_precision,
    )
    if not snapshots:
        snapshots = [
            {
                "updatesPerAuthority": 0,
                "totalUpdate": 0,
                **initial,
            }
        ]

    print("NSAMDR V16 FIXED MULTI-AUTHORITY INTERFERENCE PROBE", flush=True)
    print(f"Checkpoint step   : {source_step}", flush=True)
    print(f"Authorities       : {authority_count}", flush=True)
    print(f"Authority IDs     : {', '.join(authority_ids)}", flush=True)
    print("Sampling          : one fixed center crop each; no augmentation; clean LR", flush=True)
    print("Schedule          : deterministic round-robin", flush=True)
    print(
        "Optimizer         : "
        + ("restored interference Adam state" if args.continue_from else "fresh Adam state"),
        flush=True,
    )
    print(f"Start / authority : {start_per_authority}", flush=True)
    print(f"Stages / authority: {stages_per_authority}", flush=True)

    stage_totals = {
        int(stage) * authority_count: int(stage)
        for stage in stages_per_authority
    }
    started = time.monotonic()

    for total_update in range(start_total_update + 1, max_total_update + 1):
        authority_index = (total_update - 1) % authority_count
        _train_one(
            model,
            optimizer,
            selected[authority_index]["batch"],
            config,
            device=device,
            precision=args.amp_precision,
        )
        if total_update not in stage_totals:
            continue

        updates_per_authority = stage_totals[total_update]
        evaluated = _evaluate_all(
            model,
            selected,
            config,
            device=device,
            precision=args.amp_precision,
        )
        snapshot = {
            "updatesPerAuthority": updates_per_authority,
            "totalUpdate": total_update,
            **evaluated,
        }
        snapshots.append(snapshot)

        aggregate = evaluated["aggregate"]
        metrics = aggregate["metricDistributions"]
        print(
            f"Per-auth {updates_per_authority:4d}  : "
            f"median-global={metrics['global_recovery']['median']*100:+.2f}% "
            f"median-edge={metrics['edge_recovery']['median']*100:+.2f}% "
            f"median-grad={metrics['gradient_recovery']['median']*100:+.2f}% "
            f"median-lattice={metrics['lattice_cell_excess']['median']*100:+.2f}% "
            f"all-median-gates={aggregate['allMedianGatesPass']}",
            flush=True,
        )

        _save_checkpoint(
            continuation_path,
            model=model,
            optimizer=optimizer,
            source_checkpoint=checkpoint_path,
            source_step=source_step,
            manifest_path=manifest_path,
            authority_ids=authority_ids,
            seed=seed,
            total_update=total_update,
            updates_per_authority=updates_per_authority,
            snapshots=snapshots,
        )

    elapsed = time.monotonic() - started
    final = snapshots[-1]
    report = {
        "schema": SCHEMA,
        "sourceCheckpoint": str(checkpoint_path),
        "sourceCheckpointStep": int(source_step),
        "manifest": str(manifest_path),
        "authorityCount": authority_count,
        "authorityIds": authority_ids,
        "anchorAuthority": str(args.anchor_authority),
        "selectionPolicy": "anchor-then-seeded-shuffle-nontrivial-fixed-center-crops",
        "rejectedAuthorities": rejected,
        "minimumTargetResidual": float(args.minimum_target_residual),
        "optimizerPolicy": (
            "restored-interference-adam-state"
            if args.continue_from
            else "fresh-adam-state"
        ),
        "trainingPolicy": "deterministic-round-robin-fixed-exact-samples",
        "stagesPerAuthority": stages_per_authority,
        "startUpdatesPerAuthority": int(start_per_authority),
        "elapsedSeconds": elapsed,
        "snapshots": snapshots,
        "final": final,
        "sourceCheckpointModified": False,
        "continuationCheckpoint": str(continuation_path),
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Report            : {report_path}", flush=True)
    print(f"Continuation      : {continuation_path}", flush=True)
    return 0, report_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Train deterministic fixed crops from multiple authored authorities "
            "round-robin to measure cross-authority interference."
        )
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--manifest", default=DEFAULT_MANIFEST)
    value.add_argument("--resume", required=True)
    value.add_argument(
        "--continue-from",
        default="",
        help="interference checkpoint from a previous bounded run",
    )
    value.add_argument("--authority-count", type=int, default=4)
    value.add_argument(
        "--anchor-authority",
        default="13006d2b807f89ac",
        help="known single-authority reference included first",
    )
    value.add_argument(
        "--stages-per-authority",
        nargs="+",
        default=["64", "128"],
        help=(
            "cumulative updates per authority; accepts comma-separated or "
            "space-separated values"
        ),
    )
    value.add_argument(
        "--minimum-target-residual",
        type=float,
        default=0.01,
        help="skip trivial authorities whose authored target is too close to B",
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
