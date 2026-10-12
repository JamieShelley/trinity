#!/usr/bin/env python3
"""V17.2 V2 research: validation-selected 12-authority fit, sealed sibling test."""
from __future__ import annotations

import argparse
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
from probe_nsamdr_v17_generalization import _augment_batch, _score, _select_pairs
from probe_nsamdr_v17_sibling_architecture import (
    EARLY_TRANSFER_GATE, _batch_for_record, _evaluate, _parse_stages,
    _target_albedo_residual_magnitude, _train_one, _write_preview,
)
from v14.config import V16Config
from v17.model import NSAMDRV17

SCHEMA = "NSAMDR_V17_GENERALIZATION_V2_PROOF_V1"
CHECKPOINT_SCHEMA = "NSAMDR_V17_GENERALIZATION_V2_CHECKPOINT_V1"
ROOT = "artifacts/nsamdr/diagnostics/v17_generalization_v2"
PREVIOUS_VALIDATION_AUTHORITIES = (
    "13006d2b807f89ac", "00ea41d462e3d7b6",
    "0157a9c8f5e7c74f", "021b97ee95e2119a",
)
DEFAULT_STAGES = "6144,12288,18432,24576,30720"


def _split_pairs(
    manifest: dict[str, Any], *, anchor: str, authorities: int,
) -> dict[str, list[dict[str, Any]]]:
    """One crop/authority for learning; V1 siblings for validation; others sealed."""
    if authorities < 12:
        raise ValueError("V2 requires at least 12 training authorities")
    if anchor != PREVIOUS_VALIDATION_AUTHORITIES[0]:
        raise ValueError("V2 anchor must be the existing V1 anchor")
    pairs = _select_pairs(manifest, anchor, authorities)
    index = {
        str(a.get("family_id") or a.get("familyId")): b for a, b in pairs
    }
    if not set(PREVIOUS_VALIDATION_AUTHORITIES).issubset(index):
        raise RuntimeError("all four V1 development authorities are required")
    train = [a for a, _ in pairs]
    validation = [index[authority] for authority in PREVIOUS_VALIDATION_AUTHORITIES]
    final_test = [
        b for a, b in pairs
        if str(a.get("family_id") or a.get("familyId"))
        not in PREVIOUS_VALIDATION_AUTHORITIES
    ]
    if len(final_test) < 8:
        raise RuntimeError("at least eight untouched test siblings required")
    records = [*train, *validation, *final_test]
    ids = [str(r.get("crop_id") or r.get("cropId") or "") for r in records]
    paths = [str(r.get("path") or "") for r in records]
    if not all(ids) or not all(paths) or len(set(ids)) != len(ids) or len(set(paths)) != len(paths):
        raise RuntimeError("V2 train/validation/test crop identities overlap")
    return {"training": train, "validation": validation, "final_test": final_test}


def _ids(records: list[dict[str, Any]]) -> list[str]:
    return [str(r.get("crop_id") or r.get("cropId")) for r in records]


def _validation_score(result: dict[str, Any]) -> float:
    """Fixed validation objective; prioritizes edge, penalizes excess lattice."""
    v = result["mean"]
    return (
        float(v["global_recovery"]) + 2 * float(v["edge_recovery"])
        + float(v["gradient_recovery"]) + float(v["detail_recovery_1px"])
        - 2 * max(0.0, float(v["lattice_cell_excess"]) - EARLY_TRANSFER_GATE["lattice_cell_excess_max"])
    )


def _promote_checkpoint(
    *, qualified: bool, previous_qualified: bool,
    score: float, best_score: float, min_improvement: float,
    no_best: bool,
) -> bool:
    """A passing checkpoint must supersede any checkpoint that fails gates."""
    if no_best or (qualified and not previous_qualified):
        return True
    if previous_qualified and not qualified:
        return False
    return score > best_score + min_improvement


def _summarize(results: list[dict[str, Any]], *, sibling: bool) -> dict[str, Any]:
    value = _score(results, sibling=sibling)
    return {
        "allPass": value["allPass"], "mean": value["mean"], "gateChecks": value["gateChecks"],
        "perCrop": [
            {"metrics": item["metrics"], "gateChecks": gates,
             "residualDiagnostics": item["residualDiagnostics"]}
            for item, gates in zip(results, value["gateChecks"])
        ],
    }


def _evaluate_set(
    model: NSAMDRV17, batches: list[dict[str, torch.Tensor]], config: V16Config,
    *, device: torch.device, precision: str, sibling: bool,
) -> dict[str, Any]:
    results = [
        _evaluate(model, batch, config, device=device, precision=precision)[0]
        for batch in batches
    ]
    return _summarize(results, sibling=sibling)


def _save(path: Path, model: NSAMDRV17, optimizer: torch.optim.Optimizer,
          *, manifest: Path, partitions: dict[str, list[dict[str, Any]]],
          seed: int, learning_rate: float, update: int,
          history: list[dict[str, Any]], best_update: int,
          best_score: float, stale: int, best_qualified: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CHECKPOINT_SCHEMA, "architecture": model.architecture_contract(),
        "manifest": str(manifest), "cropIds": {k: _ids(v) for k, v in partitions.items()},
        "seed": int(seed), "learningRate": float(learning_rate),
        "update": int(update), "bestUpdate": int(best_update),
        "bestScore": float(best_score), "stale": int(stale),
        "bestQualified": bool(best_qualified),
        "history": history, "modelState": model.state_dict(),
        "optimizerState": optimizer.state_dict(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _resume_path(repo: Path, mode: str) -> Path | None:
    if mode == "fresh":
        return None
    if mode == "auto":
        candidates = [
            p for p in (repo / ROOT).glob("proof_*/stage_*/checkpoint.pt")
            if not (p.parents[1] / "report.json").is_file()
        ]
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    path = Path(mode)
    return (repo / path).resolve() if not path.is_absolute() else path.resolve()


def _restore(
    path: Path, *, model: NSAMDRV17, optimizer: torch.optim.Optimizer,
    manifest: Path, partitions: dict[str, list[dict[str, Any]]],
    seed: int, learning_rate: float, device: torch.device,
) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"V2 checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("architecture") != model.architecture_contract()
        or payload.get("manifest") != str(manifest)
        or payload.get("cropIds") != {k: _ids(v) for k, v in partitions.items()}
        or payload.get("seed") != int(seed)
        or payload.get("learningRate") != float(learning_rate)
    ):
        raise RuntimeError("incompatible V2 checkpoint; V1 checkpoints are not accepted")
    model.load_state_dict(payload["modelState"], strict=True)
    optimizer.load_state_dict(payload["optimizerState"])
    return payload


def _sheets(folder: Path, model: NSAMDRV17,
            batches: list[dict[str, torch.Tensor]], crop_ids: list[str],
            config: V16Config, *, device: torch.device, precision: str,
            role: str) -> dict[str, str]:
    paths = {}
    for i, (batch, crop_id) in enumerate(zip(batches, crop_ids)):
        _result, output = _evaluate(model, batch, config, device=device, precision=precision)
        path = folder / f"{role}_{i:02d}_ABC.png"
        _write_preview(path, batch, output, role=role)
        paths[crop_id] = str(path.resolve())
    return paths


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo = args.repo_root.resolve()
    manifest_path = Path(args.manifest)
    manifest_path = (repo / manifest_path).resolve() if not manifest_path.is_absolute() else manifest_path.resolve()
    splits = _split_pairs(json.loads(manifest_path.read_text(encoding="utf-8")),
                          anchor=args.authority_id, authorities=int(args.authorities))
    stages = _parse_stages(list(args.stages))
    if args.patience < 1 or args.min_improvement < 0:
        raise ValueError("patience must be positive and minimum improvement nonnegative")
    device = _device(args.device)
    config = V16Config(
        dataset_manifest=str(manifest_path), train_hr_size=args.hr_size,
        train_lr_size=args.hr_size // 4, validation_hr_size=args.hr_size,
        validation_lr_size=args.hr_size // 4, minimum_heldout_samples=4,
        tiles_per_epoch=1, validation_tiles=1, clean_epochs=1,
        robust_epochs=1, selector_epochs=1, use_gradient_checkpointing=False,
    )
    config.validate()

    def load(records: list[dict[str, Any]], offset: int) -> list[dict[str, torch.Tensor]]:
        return [
            _batch_for_record(record, config, seed=args.seed + offset + index, device=device)[0]
            for index, record in enumerate(records)
        ]

    training = load(splits["training"], 1)
    validation = load(splits["validation"], 1001)
    # Final-test records must not be loaded or evaluated during training/selection.
    torch.manual_seed(args.seed)
    model = NSAMDRV17(
        config, encoder_channels=args.encoder_channels, mid_channels=args.mid_channels,
        detail_channels=args.detail_channels, decoder_blocks=args.decoder_blocks,
    ).to(device)
    model.set_candidate_training()
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, foreach=False)
    for records, batches in ((splits["training"], training), (splits["validation"], validation)):
        for record, batch in zip(records, batches):
            if _target_albedo_residual_magnitude(model, batch) < args.minimum_target_residual:
                raise RuntimeError(f"insufficient target residual: {record.get('crop_id')}")
    resume = _resume_path(repo, args.resume)
    start, history, best_update, best_score, stale = 0, [], 0, float("-inf"), 0
    best_qualified = False
    best_checkpoint: Path | None = None
    if resume is not None:
        data = _restore(
            resume, model=model, optimizer=optimizer, manifest=manifest_path,
            partitions=splits, seed=args.seed, learning_rate=args.learning_rate, device=device,
        )
        start, history = int(data["update"]), list(data["history"])
        best_update, best_score, stale = (
            int(data["bestUpdate"]), float(data["bestScore"]), int(data["stale"])
        )
        best_qualified = bool(data.get("bestQualified", False))
        if best_update:
            best_checkpoint = resume.parent.parent / f"stage_{best_update:06d}" / "checkpoint.pt"
            if not best_checkpoint.is_file():
                raise RuntimeError(f"previous best checkpoint unavailable: {best_checkpoint}")
    future = [stage for stage in stages if stage > start]
    if not future:
        raise RuntimeError(f"no V2 stages above resumed update {start}")
    run_dir = repo / ROOT / f"proof_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"V17.2 GENERALIZATION V2 train={len(training)} validation={len(validation)} sealedFinalTest={len(splits['final_test'])}", flush=True)
    print(f"Resume={resume or '<fresh>'} startUpdate={start} stages={future}", flush=True)
    clock = time.monotonic()
    stop, stop_reason = start, "bounded-final-stage"
    stage_set = set(future)
    for update in range(start + 1, future[-1] + 1):
        index = (update - 1) % len(training)
        variant = ((update - 1) // len(training)) % 8
        _train_one(model, optimizer, _augment_batch(training[index], variant),
                   config, device=device, precision=args.amp_precision)
        if update == start + 1 or update % 256 == 0:
            elapsed = max(time.monotonic() - clock, 1e-6)
            print(f"[v17-generalization-v2] update {update}/{future[-1]} variant={variant} "
                  f"elapsed={elapsed/60:.1f}m eta={(future[-1]-update)*elapsed/(update-start)/60:.1f}m", flush=True)
        if update not in stage_set:
            continue
        train_result = _evaluate_set(model, training, config, device=device,
                                     precision=args.amp_precision, sibling=False)
        validation_result = _evaluate_set(model, validation, config, device=device,
                                          precision=args.amp_precision, sibling=True)
        score = _validation_score(validation_result)
        qualifies = bool(train_result["allPass"] and validation_result["allPass"])
        improved = _promote_checkpoint(
            qualified=qualifies, previous_qualified=best_qualified,
            score=score, best_score=best_score,
            min_improvement=args.min_improvement, no_best=(best_update == 0),
        )
        if improved:
            best_update, best_score, stale = update, score, 0
            best_qualified = qualifies
        else:
            stale += 1
        snapshot = {
            "update": update, "trained": train_result, "validation": validation_result,
            "validationScore": score, "newBest": improved,
            "bestUpdate": best_update, "staleStages": stale,
        }
        history.append(snapshot)
        checkpoint = run_dir / f"stage_{update:06d}" / "checkpoint.pt"
        _save(checkpoint, model, optimizer, manifest=manifest_path, partitions=splits,
              seed=args.seed, learning_rate=args.learning_rate, update=update,
              history=history, best_update=best_update, best_score=best_score,
              stale=stale, best_qualified=best_qualified)
        if improved:
            best_checkpoint = checkpoint
        stop = update
        print(f"Stage {update} trainedPass={train_result['allPass']} "
              f"validationPass={validation_result['allPass']} score={score:.5f} "
              f"best={best_update} stale={stale}", flush=True)
        if qualifies:
            stop_reason = "training-and-validation-qualified"
            break
        if stale >= args.patience:
            stop_reason = "validation-plateau"
            break
    if best_checkpoint is None:
        raise RuntimeError("V2 did not produce a best validation checkpoint")
    _restore(best_checkpoint, model=model, optimizer=optimizer, manifest=manifest_path,
             partitions=splits, seed=args.seed, learning_rate=args.learning_rate, device=device)
    best = next(s for s in history if int(s["update"]) == best_update)
    selected_train, selected_validation = best["trained"], best["validation"]
    selected_dir = run_dir / f"selected_{best_update:06d}"
    validation_previews = _sheets(
        selected_dir, model, validation, _ids(splits["validation"]),
        config, device=device, precision=args.amp_precision, role="validation",
    )
    test_pass = bool(selected_train["allPass"] and selected_validation["allPass"])
    final_result, final_previews = None, {}
    if test_pass:
        final_test = load(splits["final_test"], 2001)
        final_metrics: list[dict[str, Any]] = []
        for index, (batch, crop_id) in enumerate(
            zip(final_test, _ids(splits["final_test"]))
        ):
            metrics, output = _evaluate(
                model, batch, config, device=device, precision=args.amp_precision,
            )
            final_metrics.append(metrics)
            image = selected_dir / f"final_test_{index:02d}_ABC.png"
            _write_preview(image, batch, output, role="final_test")
            final_previews[crop_id] = str(image.resolve())
        final_result = _summarize(final_metrics, sibling=True)
    decision = (
        "final-test-visual-review-required" if final_result and final_result["allPass"]
        else "reject-final-transfer" if final_result is not None
        else "reject-validation-transfer"
    )
    report = {
        "schema": SCHEMA, "experimental": True, "architecture": model.architecture_contract(),
        "decision": decision, "stopReason": stop_reason,
        "startUpdate": start, "stopUpdate": stop, "requestedStages": future,
        "resumeSource": str(resume or ""), "bestValidationUpdate": best_update,
        "bestQualified": best_qualified,
        "bestValidationScore": best_score, "selectedCheckpoint": str(best_checkpoint.resolve()),
        "trainingCropIds": _ids(splits["training"]),
        "validationCropIds": _ids(splits["validation"]),
        "finalTestCropIds": _ids(splits["final_test"]),
        "finalTestSealedUntilSelection": True, "finalTestEvaluated": test_pass,
        "finalTestEvaluationCount": int(test_pass),
        "earlySiblingTransferThresholds": EARLY_TRANSFER_GATE,
        "trainedGateThresholds": {
            "global": config.candidate_global_recovery_required,
            "edge": config.candidate_edge_recovery_required,
            "gradient": config.candidate_gradient_recovery_required,
            "lattice": config.candidate_lattice_cell_excess_max,
        },
        "history": history, "selectedTrained": selected_train,
        "selectedValidation": selected_validation, "selectedFinalTest": final_result,
        "validationPreviews": validation_previews, "finalTestPreviews": final_previews,
        "patience": args.patience, "minImprovement": args.min_improvement,
        "visualGateRequired": decision == "final-test-visual-review-required",
        "elapsedSeconds": time.monotonic() - clock,
    }
    path = run_dir / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"V2 decision={decision} best={best_update} stop={stop} finalTestEvaluated={test_pass} report={path}", flush=True)
    return 0, path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--authority-id", default=PREVIOUS_VALIDATION_AUTHORITIES[0])
    p.add_argument("--authorities", type=int, default=12)
    p.add_argument("--hr-size", type=int, default=512)
    p.add_argument("--stages", nargs="+", default=[DEFAULT_STAGES])
    p.add_argument("--resume", default="auto", help="fresh, auto interrupted V2 only, or explicit V2 checkpoint")
    p.add_argument("--seed", type=int, default=17001)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--minimum-target-residual", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--min-improvement", type=float, default=0.002)
    p.add_argument("--encoder-channels", type=int, default=96)
    p.add_argument("--mid-channels", type=int, default=64)
    p.add_argument("--detail-channels", type=int, default=48)
    p.add_argument("--decoder-blocks", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    p.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
