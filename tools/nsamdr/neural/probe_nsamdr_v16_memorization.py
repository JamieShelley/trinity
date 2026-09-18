#!/usr/bin/env python3
"""Exact single-authority memorization diagnostic for NSAMDR V16.

This probe starts from an existing full-broad checkpoint, selects one authored
train authority/crop, disables augmentation by evaluating that crop through the
validation path, resets Adam state, and repeatedly trains on the exact same
sample. It does not modify the source checkpoint.

Purpose:
- pass: current model/objective can fit one exact target; broad training failure
  is then more likely interference/sampling/generalisation related.
- fail: current representation/objective cannot even fit one exact target from
  the step-596 state; do not spend another broad authority cycle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from probe_nsamdr_v16_full_broad import (
    DEFAULT_MANIFEST,
    _autocast,
    _device,
    _load_checkpoint,
    _loss_values,
    _optimizer,
    _proof_loss_terms,
    _residual_diagnostics,
    _to_device,
)
from v14.qualification import sample_metrics
from v16.broad_prior import AuthorityBalancedSRDataset


SCHEMA = "NSAMDR_V16_EXACT_MEMORIZATION_PROBE_V1"


def _parse_stages(raw: str) -> list[int]:
    values: list[int] = []
    for token in str(raw).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError("memorization stages must be positive")
        values.append(value)
    result = sorted(set(values))
    if not result:
        raise ValueError("at least one memorization stage is required")
    return result


def _authority_records(
    manifest: dict[str, Any],
    authority_id: str,
) -> list[dict[str, Any]]:
    authority_id = str(authority_id).strip()
    if not authority_id:
        raise ValueError("--authority-id is required")
    records = [
        dict(record)
        for record in manifest.get("crops", [])
        if str(record.get("split")) == "train"
        and str(record.get("family_id") or record.get("familyId") or "") == authority_id
    ]
    records.sort(
        key=lambda record: (
            str(record.get("crop_id") or record.get("cropId") or ""),
            str(record.get("path") or ""),
        )
    )
    if not records:
        raise RuntimeError(f"train authority not found in manifest: {authority_id}")
    return records


def _fixed_batch(
    manifest: dict[str, Any],
    config,
    authority_id: str,
    *,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    records = _authority_records(manifest, authority_id)
    selected = dict(records[0])
    selected["split"] = "validation"
    fixed_manifest = {"crops": [selected]}
    dataset = AuthorityBalancedSRDataset(
        fixed_manifest,
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
    metadata = {
        "authorityId": authority_id,
        "cropId": str(selected.get("crop_id") or selected.get("cropId") or ""),
        "recordPath": str(selected.get("path") or ""),
        "availableAuthorityCrops": len(records),
        "fixedCropPolicy": "first-sorted-authority-crop",
        "spatialPolicy": "validation-center-crop-no-augmentation",
        "degradation": "clean",
    }
    return batch, metadata


def _evaluate_exact(
    model,
    batch: dict[str, torch.Tensor],
    config,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        with _autocast(device, precision):
            outputs = model(
                batch["lr_albedo"],
                batch["lr_normal"],
                batch["lr_material"],
            )
            terms = _proof_loss_terms(outputs, batch, config)
    return {
        "lossTerms": _loss_values(terms),
        "metrics": sample_metrics(outputs, batch, final=False),
        "residualDiagnostics": _residual_diagnostics(outputs, batch, config),
    }


def _train_one(
    model,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    config,
    *,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.train()
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
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
    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    return _loss_values(terms)


def _gate_snapshot(item: dict[str, Any], config) -> dict[str, bool]:
    metrics = item["metrics"]
    return {
        "global": float(metrics["global_recovery"])
        >= float(config.candidate_global_recovery_required),
        "edge": float(metrics["edge_recovery"])
        >= float(config.candidate_edge_recovery_required),
        "gradient": float(metrics["gradient_recovery"])
        >= float(config.candidate_gradient_recovery_required),
        "lattice": float(metrics["lattice_cell_excess"])
        <= float(config.candidate_lattice_cell_excess_max),
    }


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

    batch, sample = _fixed_batch(
        manifest,
        config,
        args.authority_id,
        seed=seed + 9001,
        device=device,
    )

    # Reset optimizer state on purpose. This probe asks whether the current
    # representation can fit one exact sample without broad-corpus Adam history.
    optimizer = _optimizer(model, config)
    stages = _parse_stages(args.stages)
    max_step = stages[-1]
    torch.manual_seed(seed + 9101)

    initial = _evaluate_exact(
        model,
        batch,
        config,
        device=device,
        precision=args.amp_precision,
    )
    target_residual = float(
        initial["residualDiagnostics"]["target_residual_magnitude"]
    )
    if target_residual < float(args.minimum_target_residual):
        raise RuntimeError(
            "selected exact sample is too close to the deterministic baseline "
            f"(target residual {target_residual:.6f} < "
            f"{float(args.minimum_target_residual):.6f}); choose another authority"
        )

    print("NSAMDR V16 EXACT SINGLE-AUTHORITY MEMORIZATION PROBE", flush=True)
    print(f"Checkpoint step   : {source_step}", flush=True)
    print(f"Authority         : {sample['authorityId']}", flush=True)
    print(f"Crop              : {sample['cropId']}", flush=True)
    print("Sampling          : fixed center crop; no augmentation; clean LR", flush=True)
    print("Optimizer         : fresh Adam state", flush=True)
    print(f"Stages            : {stages}", flush=True)
    print(
        "Initial           : "
        f"global={initial['metrics']['global_recovery']*100:+.2f}% "
        f"edge={initial['metrics']['edge_recovery']*100:+.2f}% "
        f"grad={initial['metrics']['gradient_recovery']*100:+.2f}% "
        f"cos={initial['residualDiagnostics']['residual_cosine_similarity']:+.3f}",
        flush=True,
    )

    snapshots: list[dict[str, Any]] = [
        {
            "update": 0,
            **initial,
            "gateChecks": _gate_snapshot(initial, config),
        }
    ]
    started = time.monotonic()
    stage_set = set(stages)

    for update in range(1, max_step + 1):
        train_loss = _train_one(
            model,
            optimizer,
            batch,
            config,
            device=device,
            precision=args.amp_precision,
        )
        if update not in stage_set:
            continue
        evaluated = _evaluate_exact(
            model,
            batch,
            config,
            device=device,
            precision=args.amp_precision,
        )
        item = {
            "update": update,
            "trainLossBeforeUpdate": train_loss,
            **evaluated,
            "gateChecks": _gate_snapshot(evaluated, config),
        }
        snapshots.append(item)
        print(
            f"Update {update:4d}       : "
            f"global={evaluated['metrics']['global_recovery']*100:+.2f}% "
            f"edge={evaluated['metrics']['edge_recovery']*100:+.2f}% "
            f"grad={evaluated['metrics']['gradient_recovery']*100:+.2f}% "
            f"lattice={evaluated['metrics']['lattice_cell_excess']*100:+.2f}% "
            f"cos={evaluated['residualDiagnostics']['residual_cosine_similarity']:+.3f} "
            f"sign={evaluated['residualDiagnostics']['target_weighted_sign_agreement']*100:.1f}%",
            flush=True,
        )

    elapsed = time.monotonic() - started
    final = snapshots[-1]
    report = {
        "schema": SCHEMA,
        "sourceCheckpoint": str(checkpoint_path),
        "sourceCheckpointStep": int(source_step),
        "manifest": str(manifest_path),
        "sample": sample,
        "optimizerPolicy": "fresh-adam-state",
        "trainingPolicy": "same-exact-sample-repeated",
        "stages": stages,
        "elapsedSeconds": elapsed,
        "initial": snapshots[0],
        "snapshots": snapshots,
        "final": final,
        "interpretation": {
            "allCandidateGatesPass": bool(all(final["gateChecks"].values())),
            "globalGatePass": bool(final["gateChecks"]["global"]),
            "edgeGatePass": bool(final["gateChecks"]["edge"]),
            "gradientGatePass": bool(final["gateChecks"]["gradient"]),
            "latticeGatePass": bool(final["gateChecks"]["lattice"]),
        },
        "sourceCheckpointModified": False,
    }

    output_dir = checkpoint_path.parent / "memorization"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"single_authority_{sample['authorityId']}_step_{source_step:06d}.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Report            : {output_path}", flush=True)
    return 0, output_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Fine-tune one fixed authored train sample from a full-broad "
            "checkpoint to test exact memorization capacity."
        )
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--manifest", default=DEFAULT_MANIFEST)
    value.add_argument("--resume", required=True)
    value.add_argument("--authority-id", required=True)
    value.add_argument(
        "--stages",
        default="1,2,4,8,16,32,64",
        help="cumulative exact-sample update counts",
    )
    value.add_argument(
        "--minimum-target-residual",
        type=float,
        default=0.01,
        help="reject trivial samples whose authored target is too close to B",
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
