#!/usr/bin/env python3
"""Same-authority unseen-crop transfer diagnostic for NSAMDR V16.

The fixed-authority interference probe trains one deterministic crop per selected
train authority. This diagnostic evaluates the resulting checkpoint on both that
trained crop and a different authored crop from the same authorities, with no
training or augmentation.

Purpose:
- separate exact-crop memorization from same-authority spatial transfer;
- determine whether the remaining held-out gap is primarily crop overfit or
  cross-authority generalisation;
- keep the next decision CPU/GPU-cheap before scaling to more authorities.

This is a diagnostic only and is never production qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from probe_nsamdr_v16_full_broad import (
    DEFAULT_MANIFEST,
    _device,
    _load_checkpoint,
    _to_device,
)
from probe_nsamdr_v16_interference import (
    CHECKPOINT_SCHEMA as INTERFERENCE_CHECKPOINT_SCHEMA,
    _aggregate,
)
from probe_nsamdr_v16_memorization import (
    _evaluate_exact,
    _gate_snapshot,
)
from v16.broad_prior import AuthorityBalancedSRDataset


SCHEMA = "NSAMDR_V16_SIBLING_CROP_TRANSFER_PROBE_V1"


def _authority_records(
    manifest: dict[str, Any],
    authority_id: str,
) -> list[dict[str, Any]]:
    rows = [
        dict(record)
        for record in manifest.get("crops", [])
        if str(record.get("split")) == "train"
        and str(record.get("family_id") or record.get("familyId") or "") == str(authority_id)
    ]
    rows.sort(
        key=lambda record: (
            str(record.get("crop_id") or record.get("cropId") or ""),
            str(record.get("path") or ""),
        )
    )
    return rows


def _authority_crop_pair_records(
    manifest: dict[str, Any],
    authority_ids: list[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for authority_id in authority_ids:
        records = _authority_records(manifest, authority_id)
        if len(records) < 2:
            raise RuntimeError(
                f"authority {authority_id} has only {len(records)} authored train crop(s); "
                "sibling-crop transfer requires at least two"
            )
        result.append(
            {
                "authorityId": str(authority_id),
                "trained": records[0],
                "sibling": records[1],
                "availableCrops": len(records),
            }
        )
    return result


def _batch_for_record(
    record: dict[str, Any],
    manifest: dict[str, Any],
    config,
    *,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
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


def _evaluate_pairs(
    model,
    pairs: list[dict[str, Any]],
    manifest: dict[str, Any],
    config,
    *,
    role: str,
    device: torch.device,
    precision: str,
    seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        record = pair[role]
        batch, sample = _batch_for_record(
            record,
            manifest,
            config,
            seed=seed + index,
            device=device,
        )
        evaluated = _evaluate_exact(
            model,
            batch,
            config,
            device=device,
            precision=precision,
        )
        metrics = dict(evaluated["metrics"])
        metrics["heldout_sample"] = 0.0
        evaluated = {**evaluated, "metrics": metrics}
        rows.append(
            {
                "authorityId": pair["authorityId"],
                "cropId": sample["cropId"],
                "evaluationRole": (
                    "train-authority-trained-fixed-crop"
                    if role == "trained"
                    else "train-authority-unseen-sibling-crop"
                ),
                **evaluated,
                "gateChecks": _gate_snapshot(evaluated, config),
            }
        )
    return {
        "perAuthority": rows,
        "aggregate": _aggregate(rows, config),
    }


def _median_metric(summary: dict[str, Any], key: str) -> float:
    return float(summary["aggregate"]["metricDistributions"][key]["median"])


def _comparison(
    source: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    keys = (
        "global_recovery",
        "edge_recovery",
        "gradient_recovery",
        "normal_recovery",
        "lattice_cell_excess",
        "protected_preservation",
    )
    return {
        key: _median_metric(candidate, key) - _median_metric(source, key)
        for key in keys
    }


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    source_checkpoint = Path(args.resume)
    if not source_checkpoint.is_absolute():
        source_checkpoint = (repo_root / source_checkpoint).resolve()
    candidate_checkpoint = Path(args.candidate_checkpoint)
    if not candidate_checkpoint.is_absolute():
        candidate_checkpoint = (repo_root / candidate_checkpoint).resolve()

    for label, path in (
        ("manifest", manifest_path),
        ("source checkpoint", source_checkpoint),
        ("candidate checkpoint", candidate_checkpoint),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = _device(args.device)
    model, loaded_optimizer, config, source_step, seed, _curve = _load_checkpoint(
        source_checkpoint,
        device=device,
        manifest_path=manifest_path,
    )
    del loaded_optimizer

    try:
        candidate_payload = torch.load(
            candidate_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        candidate_payload = torch.load(candidate_checkpoint, map_location="cpu")
    if not isinstance(candidate_payload, dict):
        raise RuntimeError("candidate checkpoint payload is not a dictionary")
    if candidate_payload.get("schema") != INTERFERENCE_CHECKPOINT_SCHEMA:
        raise RuntimeError("candidate checkpoint is not an interference checkpoint")
    if Path(str(candidate_payload.get("sourceCheckpoint", ""))).resolve() != source_checkpoint.resolve():
        raise RuntimeError("candidate checkpoint belongs to a different source checkpoint")
    if int(candidate_payload.get("sourceCheckpointStep", -1)) != int(source_step):
        raise RuntimeError("candidate checkpoint source step mismatch")
    if Path(str(candidate_payload.get("manifest", ""))).resolve() != manifest_path.resolve():
        raise RuntimeError("candidate checkpoint belongs to a different manifest")

    authority_ids = [str(value) for value in candidate_payload.get("authorityIds") or []]
    if not authority_ids:
        raise RuntimeError("candidate checkpoint has no authorityIds")
    pairs = _authority_crop_pair_records(manifest, authority_ids)

    print("NSAMDR V16 SAME-AUTHORITY SIBLING-CROP TRANSFER PROBE", flush=True)
    print(f"Source step        : {source_step}", flush=True)
    print(f"Authorities        : {len(authority_ids)}", flush=True)
    print("Training           : none", flush=True)
    print("Comparison         : trained first crop vs unseen second crop", flush=True)

    source_trained = _evaluate_pairs(
        model,
        pairs,
        manifest,
        config,
        role="trained",
        device=device,
        precision=args.amp_precision,
        seed=seed + 12001,
    )
    source_sibling = _evaluate_pairs(
        model,
        pairs,
        manifest,
        config,
        role="sibling",
        device=device,
        precision=args.amp_precision,
        seed=seed + 13001,
    )

    model.load_state_dict(candidate_payload["modelState"], strict=True)
    candidate_trained = _evaluate_pairs(
        model,
        pairs,
        manifest,
        config,
        role="trained",
        device=device,
        precision=args.amp_precision,
        seed=seed + 12001,
    )
    candidate_sibling = _evaluate_pairs(
        model,
        pairs,
        manifest,
        config,
        role="sibling",
        device=device,
        precision=args.amp_precision,
        seed=seed + 13001,
    )

    trained_vs_source = _comparison(source_trained, candidate_trained)
    sibling_vs_source = _comparison(source_sibling, candidate_sibling)
    sibling_vs_trained_candidate = _comparison(candidate_trained, candidate_sibling)

    output_dir = source_checkpoint.parent / "sibling_crop_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{candidate_checkpoint.stem}.json"
    report = {
        "schema": SCHEMA,
        "diagnosticRole": "same-train-authority-unseen-sibling-crop-transfer",
        "qualificationEligible": False,
        "qualificationExclusionReasons": [
            "train-authority-diagnostic",
            "same-authority-crops-not-independent-heldout-authorities",
            "not-production-broad-training-schedule",
            "material-semantics-unresolved",
            "selector-not-qualified",
        ],
        "manifest": str(manifest_path),
        "sourceCheckpoint": str(source_checkpoint),
        "sourceCheckpointStep": int(source_step),
        "candidateCheckpoint": str(candidate_checkpoint),
        "candidateUpdatesPerAuthority": int(candidate_payload.get("updatesPerAuthority", 0)),
        "authorityIds": authority_ids,
        "cropPolicy": "first-sorted-crop-trained/second-sorted-crop-unseen",
        "sourceTrainedCrop": source_trained,
        "candidateTrainedCrop": candidate_trained,
        "sourceSiblingCrop": source_sibling,
        "candidateSiblingCrop": candidate_sibling,
        "medianDeltaCandidateVsSourceOnTrainedCrop": trained_vs_source,
        "medianDeltaCandidateVsSourceOnSiblingCrop": sibling_vs_source,
        "medianDeltaSiblingVsTrainedCandidate": sibling_vs_trained_candidate,
        "interpretation": {
            "candidateImprovesSiblingGlobal": sibling_vs_source["global_recovery"] > 0.0,
            "candidateImprovesSiblingEdge": sibling_vs_source["edge_recovery"] > 0.0,
            "candidateImprovesSiblingGradient": sibling_vs_source["gradient_recovery"] > 0.0,
            "candidateReducesSiblingLattice": sibling_vs_source["lattice_cell_excess"] < 0.0,
            "candidateSiblingMedianGatesPass": bool(
                candidate_sibling["aggregate"]["allMedianGatesPass"]
            ),
        },
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for name, summary in (
        ("candidate trained", candidate_trained),
        ("candidate sibling", candidate_sibling),
    ):
        metrics = summary["aggregate"]["metricDistributions"]
        print(
            f"{name:18s}: "
            f"global={metrics['global_recovery']['median']*100:+.2f}% "
            f"edge={metrics['edge_recovery']['median']*100:+.2f}% "
            f"grad={metrics['gradient_recovery']['median']*100:+.2f}% "
            f"lattice={metrics['lattice_cell_excess']['median']*100:+.2f}%",
            flush=True,
        )
    print(
        "Sibling vs source : "
        f"global={sibling_vs_source['global_recovery']*100:+.2f}pp "
        f"edge={sibling_vs_source['edge_recovery']*100:+.2f}pp "
        f"grad={sibling_vs_source['gradient_recovery']*100:+.2f}pp "
        f"lattice={sibling_vs_source['lattice_cell_excess']*100:+.2f}pp",
        flush=True,
    )
    print(f"Report             : {output_path}", flush=True)
    return 0, output_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Evaluate a fixed-authority checkpoint on an unseen sibling crop "
            "from each of the same train authorities."
        )
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--manifest", default=DEFAULT_MANIFEST)
    value.add_argument("--resume", required=True, help="source full-broad checkpoint")
    value.add_argument("--candidate-checkpoint", required=True)
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
