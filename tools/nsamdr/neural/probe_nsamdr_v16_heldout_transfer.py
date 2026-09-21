#!/usr/bin/env python3
"""Held-out transfer diagnostic for an NSAMDR V16 train-fit checkpoint.

This probe compares the original full-broad source checkpoint with a derived
fixed-authority interference checkpoint on the exact same complete held-out
validation authorities. It performs no training.

Purpose:
- determine whether reconstruction learned during fixed train-authority fitting
  transfers to authorities that were never used for candidate training;
- distinguish useful transferable structure learning from pure memorization;
- export matched source/candidate held-out previews for visual inspection.

This is a generalisation diagnostic, not final production qualification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from probe_nsamdr_v16_full_broad import (
    DEFAULT_MANIFEST,
    _device,
    _evaluate,
    _gate_status,
    _load_checkpoint,
)
from probe_nsamdr_v16_interference import (
    CHECKPOINT_SCHEMA as INTERFERENCE_CHECKPOINT_SCHEMA,
)


SCHEMA = "NSAMDR_V16_HELDOUT_TRANSFER_PROBE_V1"
MEDIAN_KEYS = (
    "median_global_recovery",
    "median_edge_recovery",
    "median_gradient_recovery",
    "median_normal_recovery",
    "median_lattice_cell_excess",
)
RESIDUAL_KEYS = (
    "candidate_to_target_residual_ratio",
    "residual_cosine_similarity",
    "target_weighted_sign_agreement",
    "least_squares_residual_gain",
)


def _validation_authority_count(manifest: dict[str, Any]) -> int:
    values = {
        str(record.get("family_id") or record.get("familyId") or "").strip()
        for record in manifest.get("crops", [])
        if str(record.get("split")) == "validation"
    }
    values.discard("")
    return len(values)


def _validate_candidate_payload(
    payload: dict[str, Any],
    *,
    source_checkpoint: Path,
    source_step: int,
    manifest_path: Path,
) -> None:
    if payload.get("schema") != INTERFERENCE_CHECKPOINT_SCHEMA:
        raise RuntimeError("candidate checkpoint is not an interference checkpoint")
    if Path(str(payload.get("sourceCheckpoint", ""))).resolve() != source_checkpoint.resolve():
        raise RuntimeError("candidate checkpoint belongs to a different source checkpoint")
    if int(payload.get("sourceCheckpointStep", -1)) != int(source_step):
        raise RuntimeError("candidate checkpoint source step mismatch")
    if Path(str(payload.get("manifest", ""))).resolve() != manifest_path.resolve():
        raise RuntimeError("candidate checkpoint belongs to a different manifest")
    if not isinstance(payload.get("modelState"), dict):
        raise RuntimeError("candidate checkpoint is missing modelState")


def _median_deltas(
    source: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(source[key])
        for key in MEDIAN_KEYS
    }


def _residual_medians(summary: dict[str, Any]) -> dict[str, float]:
    distributions = dict(summary.get("residualDiagnosticDistributions") or {})
    result: dict[str, float] = {}
    for key in RESIDUAL_KEYS:
        item = distributions.get(key)
        if isinstance(item, dict) and "median" in item:
            result[key] = float(item["median"])
    return result


def _residual_deltas(
    source: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(source[key])
        for key in RESIDUAL_KEYS
        if key in source and key in candidate
    }


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"manifest is missing: {manifest_path}")

    source_checkpoint = Path(args.resume)
    if not source_checkpoint.is_absolute():
        source_checkpoint = (repo_root / source_checkpoint).resolve()
    if not source_checkpoint.is_file():
        raise RuntimeError(f"source checkpoint is missing: {source_checkpoint}")

    candidate_checkpoint = Path(args.candidate_checkpoint)
    if not candidate_checkpoint.is_absolute():
        candidate_checkpoint = (repo_root / candidate_checkpoint).resolve()
    if not candidate_checkpoint.is_file():
        raise RuntimeError(f"candidate checkpoint is missing: {candidate_checkpoint}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    available_validation = _validation_authority_count(manifest)
    if available_validation < 1:
        raise RuntimeError("manifest has no held-out validation authorities")
    validation_samples = (
        int(args.validation_samples)
        if int(args.validation_samples) > 0
        else available_validation
    )
    if validation_samples > available_validation:
        raise RuntimeError(
            f"validation-samples {validation_samples} exceeds "
            f"{available_validation} held-out authorities"
        )

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
    _validate_candidate_payload(
        candidate_payload,
        source_checkpoint=source_checkpoint,
        source_step=source_step,
        manifest_path=manifest_path,
    )

    output_dir = source_checkpoint.parent / "heldout_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{candidate_checkpoint.stem}.json"
    source_preview_root = output_dir / f"{candidate_checkpoint.stem}_source_previews"
    candidate_preview_root = output_dir / f"{candidate_checkpoint.stem}_candidate_previews"

    print("NSAMDR V16 HELD-OUT TRANSFER PROBE", flush=True)
    print(f"Source step        : {source_step}", flush=True)
    print(f"Candidate          : {candidate_checkpoint}", flush=True)
    print(f"Held-out authorities: {validation_samples}", flush=True)
    print("Training           : none", flush=True)

    source = _evaluate(
        model,
        manifest,
        config,
        split="validation",
        samples=validation_samples,
        device=device,
        seed=seed,
        precision=args.amp_precision,
        preview_root=source_preview_root if int(args.preview_samples) > 0 else None,
        preview_samples=int(args.preview_samples),
    )

    model.load_state_dict(candidate_payload["modelState"], strict=True)
    candidate = _evaluate(
        model,
        manifest,
        config,
        split="validation",
        samples=validation_samples,
        device=device,
        seed=seed,
        precision=args.amp_precision,
        preview_root=candidate_preview_root if int(args.preview_samples) > 0 else None,
        preview_samples=int(args.preview_samples),
    )

    source_residual = _residual_medians(source)
    candidate_residual = _residual_medians(candidate)
    median_deltas = _median_deltas(source, candidate)
    residual_deltas = _residual_deltas(source_residual, candidate_residual)

    candidate_status = _gate_status(candidate, config)
    report = {
        "schema": SCHEMA,
        "diagnosticRole": "independent-heldout-transfer",
        "qualificationEligible": False,
        "qualificationExclusionReasons": [
            "derived-from-fixed-train-authority-fit-checkpoint",
            "not-production-broad-training-schedule",
            "material-semantics-unresolved",
            "selector-not-qualified",
        ],
        "manifest": str(manifest_path),
        "sourceCheckpoint": str(source_checkpoint),
        "sourceCheckpointStep": int(source_step),
        "candidateCheckpoint": str(candidate_checkpoint),
        "candidateCheckpointSchema": str(candidate_payload.get("schema")),
        "candidateUpdatesPerAuthority": int(candidate_payload.get("updatesPerAuthority", 0)),
        "candidateAuthorityIds": list(candidate_payload.get("authorityIds") or []),
        "heldOutAuthorityCount": int(validation_samples),
        "availableHeldOutAuthorityCount": int(available_validation),
        "evaluationPolicy": "same-complete-heldout-authorities-clean-validation-no-training",
        "source": source,
        "candidate": candidate,
        "medianDeltaCandidateVsSource": median_deltas,
        "sourceResidualMedians": source_residual,
        "candidateResidualMedians": candidate_residual,
        "residualMedianDeltaCandidateVsSource": residual_deltas,
        "candidateGateStatus": candidate_status,
        "previewSamples": int(args.preview_samples),
        "interpretation": {
            "candidateImprovesGlobalMedian": median_deltas["median_global_recovery"] > 0.0,
            "candidateImprovesEdgeMedian": median_deltas["median_edge_recovery"] > 0.0,
            "candidateImprovesGradientMedian": median_deltas["median_gradient_recovery"] > 0.0,
            "candidateReducesLatticeMedian": median_deltas["median_lattice_cell_excess"] < 0.0,
            "candidatePassesNumericalCandidateGates": bool(
                candidate_status["albedoNormalNumericalGatesPass"]
            ),
        },
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "Source held-out    : "
        f"global={source['median_global_recovery']*100:+.2f}% "
        f"edge={source['median_edge_recovery']*100:+.2f}% "
        f"grad={source['median_gradient_recovery']*100:+.2f}% "
        f"lattice={source['median_lattice_cell_excess']*100:+.2f}%",
        flush=True,
    )
    print(
        "Candidate held-out : "
        f"global={candidate['median_global_recovery']*100:+.2f}% "
        f"edge={candidate['median_edge_recovery']*100:+.2f}% "
        f"grad={candidate['median_gradient_recovery']*100:+.2f}% "
        f"lattice={candidate['median_lattice_cell_excess']*100:+.2f}%",
        flush=True,
    )
    print(
        "Delta              : "
        f"global={median_deltas['median_global_recovery']*100:+.2f}pp "
        f"edge={median_deltas['median_edge_recovery']*100:+.2f}pp "
        f"grad={median_deltas['median_gradient_recovery']*100:+.2f}pp "
        f"lattice={median_deltas['median_lattice_cell_excess']*100:+.2f}pp",
        flush=True,
    )
    print(
        "Candidate gates    : "
        f"{candidate_status['checks']} "
        f"all={candidate_status['albedoNormalNumericalGatesPass']}",
        flush=True,
    )
    print(f"Report             : {output_path}", flush=True)
    return 0, output_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Compare a fixed-authority train-fit checkpoint against its source "
            "checkpoint on independent complete held-out authorities."
        )
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--manifest", default=DEFAULT_MANIFEST)
    value.add_argument("--resume", required=True, help="source full-broad checkpoint")
    value.add_argument(
        "--candidate-checkpoint",
        required=True,
        help="derived interference checkpoint to evaluate",
    )
    value.add_argument(
        "--validation-samples",
        type=int,
        default=0,
        help="held-out authorities to evaluate; 0 means all available",
    )
    value.add_argument(
        "--preview-samples",
        type=int,
        default=4,
        help="write matched source/candidate previews for the first N held-out authorities",
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
