#!/usr/bin/env python3
"""V17.3 controlled native-spatial vs fixed-source learning; independent holdouts.

Separate research experiment. Retains V17.2 model, loss and unchanged gates.
Never opens the V2 sealed sibling authority source and never evaluates V17.3
final-test pixels unless a chosen arm passes training AND validation gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from probe_nsamdr_v16_full_broad import DEFAULT_MANIFEST, _device, _to_device
from probe_nsamdr_v17_generalization import _augment_batch
from probe_nsamdr_v17_generalization_v2 import (
    _evaluate_set, _ids, _promote_checkpoint, _sheets, _summarize,
    _validation_score,
)
from probe_nsamdr_v17_sibling_architecture import (
    EARLY_TRANSFER_GATE, _batch_for_record, _evaluate,
    _parse_stages, _target_albedo_residual_magnitude,
    _train_one, _write_preview,
)
from v14.config import V16Config
from v17.model import NSAMDRV17
from v17.spatial_sampling import (
    NativeAuthoritySampler, _crop_id, _family_ids, _id, _partitions,
)

SCHEMA = "NSAMDR_V17_SPATIAL_GENERALIZATION_PROOF_V1"
CHECKPOINT_SCHEMA = "NSAMDR_V17_SPATIAL_GENERALIZATION_CHECKPOINT_V1"
ROOT = "artifacts/nsamdr/diagnostics/v17_spatial_generalization"
STAGES = "12288,24576,36864,49152"


def _latest_v2(repo: Path) -> tuple[Path, dict[str, Any]]:
    root = repo / "artifacts/nsamdr/diagnostics/v17_generalization_v2"
    eligible = []
    for path in root.glob("proof_*/report.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (data.get("schema") == "NSAMDR_V17_GENERALIZATION_V2_PROOF_V1"
                    and data.get("finalTestCropIds")):
                eligible.append((path.stat().st_mtime, path.resolve(), data))
        except (OSError, ValueError, TypeError):
            pass
    if not eligible:
        raise RuntimeError("V17.3 requires a completed V2 proof for leakage exclusions")
    _time, path, data = max(eligible)
    return path, data


def _contract(args: argparse.Namespace, manifest_path: Path, v2_path: Path,
              subsets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    return {
        "schema": SCHEMA, "manifest": str(manifest_path),
        "manifestSha256": sha(manifest_path), "v2Report": str(v2_path),
        "v2ReportSha256": sha(v2_path),
        "partitionCropIds": {k: _ids(v) for k, v in subsets.items()},
        "arms": ["fixed", "spatial"] if args.arm == "both" else [args.arm],
        "stages": _parse_stages(list(args.stages)),
        "seed": int(args.seed), "learningRate": float(args.learning_rate),
        "minimumTargetResidual": float(args.minimum_target_residual),
        "patience": int(args.patience), "minImprovement": float(args.min_improvement),
        "architecture": {
            "encoderChannels": args.encoder_channels, "decoderMidChannels": args.mid_channels,
            "decoderDetailChannels": args.detail_channels,
            "decoderBlocksPerStage": args.decoder_blocks,
        },
        "sourcePolicy": "native-authored-1024-min; no V2 authority reused",
        "spatialPolicy": "50-percent-uniform-50-percent-detail-by-authority-exposure",
        "controlPolicy": "one fixed native 512 region per authority with identical D4",
    }


def _run_dir(repo: Path, contract: dict[str, Any], mode: str) -> Path:
    root = repo / ROOT
    if mode == "fresh":
        path = root / f"proof_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        path.mkdir(parents=True, exist_ok=False)
        (path / "run_manifest.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True)+"\n", encoding="utf-8",
        )
        return path
    if mode != "auto":
        raise ValueError("V17.3 resume must be fresh or auto")
    candidates = [p for p in root.glob("proof_*")
                  if (p / "run_manifest.json").is_file()
                  and not (p / "report.json").is_file()]
    if not candidates:
        raise RuntimeError("no incomplete V17.3 run for auto resume; choose fresh")
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    stored = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    if stored != contract:
        raise RuntimeError("incomplete V17.3 proof contract differs (manifest/split/arms/"
                           "stages/seed/optimizer); use fresh, never mix checkpoints")
    return path


def _model(config: V16Config, args: argparse.Namespace,
           device: torch.device) -> tuple[NSAMDRV17, torch.optim.Optimizer]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = NSAMDRV17(
        config, encoder_channels=args.encoder_channels,
        mid_channels=args.mid_channels, detail_channels=args.detail_channels,
        decoder_blocks=args.decoder_blocks,
    ).to(device)
    model.set_candidate_training()
    optimizer = torch.optim.Adam(
        model.trainable_parameters(), lr=args.learning_rate,
        betas=(0.9, 0.999), eps=1e-8, foreach=False,
    )
    return model, optimizer


def _save(path: Path, model: NSAMDRV17, optimizer: torch.optim.Optimizer,
          contract: dict[str, Any], arm: str, *, update: int,
          history: list[dict[str, Any]], best_update: int,
          best_score: float, stale: int, qualified: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schema": CHECKPOINT_SCHEMA, "contract": contract,
        "arm": arm, "update": update, "history": history,
        "bestUpdate": best_update, "bestScore": best_score,
        "stale": stale, "bestQualified": qualified,
        "architecture": model.architecture_contract(),
        "modelState": model.state_dict(), "optimizerState": optimizer.state_dict(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def _restore(path: Path, model: NSAMDRV17, optimizer: torch.optim.Optimizer,
             contract: dict[str, Any], arm: str, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if (not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA
            or payload.get("contract") != contract or payload.get("arm") != arm
            or payload.get("architecture") != model.architecture_contract()):
        raise RuntimeError("V17.3 checkpoint provenance/architecture mismatch")
    model.load_state_dict(payload["modelState"], strict=True)
    optimizer.load_state_dict(payload["optimizerState"])
    return payload


def _train_arm(
    arm: str, args: argparse.Namespace, run_dir: Path,
    contract: dict[str, Any], sampler: NativeAuthoritySampler,
    training: list[dict[str, torch.Tensor]], authority_ids: list[str],
    validation: list[dict[str, torch.Tensor]], config: V16Config,
    device: torch.device, initial_residuals: dict[str, float],
) -> dict[str, Any]:
    arm_dir = run_dir / arm
    done_path = arm_dir / "arm_report.json"
    if done_path.is_file():
        return json.loads(done_path.read_text(encoding="utf-8"))
    model, optimizer = _model(config, args, device)
    checkpoints = sorted(arm_dir.glob("stage_*/checkpoint.pt"))
    stages: list[int] = contract["stages"]
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.parent.name.split("_")[1]))
        state = _restore(latest, model, optimizer, contract, arm, device)
        start = int(state["update"])
        history = list(state["history"])
        best_update = int(state["bestUpdate"])
        best_score = float(state["bestScore"])
        stale = int(state["stale"])
        best_qualified = bool(state["bestQualified"])
    else:
        start, history, best_update, best_score, stale = 0, [], 0, float("-inf"), 0
        best_qualified = False
    if start and start not in stages:
        raise RuntimeError("V17.3 resume update is not a bounded checkpoint")
    selected_checkpoint = (
        arm_dir / f"stage_{best_update:06d}" / "checkpoint.pt"
        if best_update else None
    )
    clock = time.monotonic()
    stop_reason = "bounded-final-stage"
    if stale >= args.patience:
        stop_reason = "validation-plateau"
    elif best_qualified:
        stop_reason = "training-and-validation-qualified"
    else:
        for update in range(start + 1, stages[-1] + 1):
            index = (update - 1) % len(training)
            authority = authority_ids[index]
            exposure = (update - 1) // len(training)
            variant = exposure % 8
            if arm == "fixed":
                batch = training[index]
            else:
                raw, _meta = sampler.sample(authority, exposure, mode="spatial")
                batch = _to_device(raw, device)
            _train_one(model, optimizer, _augment_batch(batch, variant),
                       config, device=device, precision=args.amp_precision)
            if update == start+1 or update % 256 == 0:
                elapsed = max(time.monotonic()-clock, 1e-6)
                progress = (update-start) / max(1, stages[-1]-start)
                print(
                    f"[v17-spatial] arm={arm} update {update}/{stages[-1]} "
                    f"variant={variant} elapsed={elapsed/60:.1f}m "
                    f"eta={(1-progress)*elapsed/max(progress,1e-6)/60:.1f}m",
                    flush=True,
                )
            if update not in stages:
                continue
            # The same frozen 12 fixed-location probes are evaluated for both arms.
            train_result = _evaluate_set(
                model, training, config, device=device,
                precision=args.amp_precision, sibling=False,
            )
            val_result = _evaluate_set(
                model, validation, config, device=device,
                precision=args.amp_precision, sibling=True,
            )
            score = _validation_score(val_result)
            qualifies = bool(train_result["allPass"] and val_result["allPass"])
            improved = _promote_checkpoint(
                qualified=qualifies, previous_qualified=best_qualified,
                score=score, best_score=best_score,
                min_improvement=args.min_improvement, no_best=not bool(best_update),
            )
            if improved:
                best_update, best_score, stale, best_qualified = update, score, 0, qualifies
            else:
                stale += 1
            history.append({
                "update": update, "trained": train_result, "validation": val_result,
                "validationScore": score, "newBest": improved,
                "bestUpdate": best_update, "staleStages": stale,
            })
            checkpoint = arm_dir / f"stage_{update:06d}" / "checkpoint.pt"
            _save(checkpoint, model, optimizer, contract, arm, update=update,
                  history=history, best_update=best_update, best_score=best_score,
                  stale=stale, qualified=best_qualified)
            if improved:
                selected_checkpoint = checkpoint
            print(f"[v17-spatial] arm={arm} stage={update} "
                  f"trainedPass={train_result['allPass']} "
                  f"validationPass={val_result['allPass']} "
                  f"score={score:.5f} best={best_update} stale={stale}", flush=True)
            if qualifies:
                stop_reason = "training-and-validation-qualified"
                break
            if stale >= args.patience:
                stop_reason = "validation-plateau"
                break
    if selected_checkpoint is None or not selected_checkpoint.is_file():
        raise RuntimeError(f"V17.3 {arm} has no selected checkpoint")
    _restore(selected_checkpoint, model, optimizer, contract, arm, device)
    selected = next(row for row in history if int(row["update"]) == best_update)
    output = {
        "arm": arm, "stopUpdate": int(history[-1]["update"]),
        "stopReason": stop_reason, "bestValidationUpdate": best_update,
        "bestValidationScore": best_score, "bestQualified": best_qualified,
        "selectedCheckpoint": str(selected_checkpoint.resolve()),
        "selectedTraining": selected["trained"],
        "selectedValidation": selected["validation"],
        "trainingProbeType": "same frozen fixed authored region for both arms",
        "trainingResiduals": initial_residuals,
        "history": history,
        "validationPreviews": _sheets(
            arm_dir / f"selected_{best_update:06d}", model, validation,
            contract["partitionCropIds"]["validation"],
            config, device=device, precision=args.amp_precision, role="validation",
        ),
        "elapsedSeconds": time.monotonic()-clock,
    }
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.write_text(json.dumps(output, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return output


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo = args.repo_root.resolve()
    manifest_path = Path(args.manifest)
    manifest_path = (repo / manifest_path).resolve() if not manifest_path.is_absolute() else manifest_path.resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"authored census corpus manifest missing: {manifest_path}")
    v2_path, v2 = _latest_v2(repo)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.patience < 1 or args.min_improvement < 0:
        raise ValueError("patience >=1 and nonnegative minimum validation improvement required")
    if args.learning_rate != 0.0002 or args.hr_size != 512:
        raise ValueError("V17.3 first controlled experiment locks V17.2 loss and 512 HR / Adam 0.0002")
    device = _device(args.device)
    config = V16Config(
        dataset_manifest=str(manifest_path),
        train_hr_size=512, train_lr_size=128, validation_hr_size=512,
        validation_lr_size=128, minimum_heldout_samples=4,
        tiles_per_epoch=1, validation_tiles=1, clean_epochs=1,
        robust_epochs=1, selector_epochs=1, use_gradient_checkpointing=False,
    )
    config.validate()
    probe_model, probe_optimizer = _model(config, args, device)
    training_screen: dict[str, float] = {}
    rejected_screen: list[dict[str, Any]] = []

    def eligible_training(record: dict[str, Any]) -> bool:
        batch = _batch_for_record(
            record, config, seed=args.seed+1, device=device,
        )[0]
        magnitude = _target_albedo_residual_magnitude(probe_model, batch)
        crop = _crop_id(record)
        training_screen[crop] = magnitude
        if magnitude < args.minimum_target_residual:
            rejected_screen.append({"cropId": crop, "targetResidual": magnitude})
            print(
                f"[v17-spatial] reject training crop={crop} "
                f"targetResidual={magnitude:.6f} below {args.minimum_target_residual:.6f}",
                flush=True,
            )
            return False
        return True

    subsets = _partitions(
        manifest, v2, train_count=args.authorities,
        dev_count=4, test_count=8, min_size=1024,
        eligible_training=eligible_training,
    )
    del probe_model, probe_optimizer
    contract = _contract(args, manifest_path, v2_path, subsets)
    arms = contract["arms"]
    run_dir = _run_dir(repo, contract, args.resume)
    train_ids = [_id(row) for row in subsets["training"]]
    print(
        f"V17.3 SPATIAL GENERALIZATION arms={','.join(arms)} "
        f"train={len(train_ids)} independentValidation=4 "
        f"sealedFinalTest=8 V2AuthoritiesExcluded={len(_family_ids([*(v2['trainingCropIds']),*(v2['validationCropIds']),*(v2['finalTestCropIds'])]))} "
        f"resume={args.resume} run={run_dir}", flush=True,
    )
    validation = [
        _batch_for_record(row, config, seed=args.seed+1001+i, device=device)[0]
        for i, row in enumerate(subsets["validation"])
    ]
    sampler = NativeAuthoritySampler(
        repo, manifest, subsets["training"], hr_size=512, seed=args.seed,
    )
    try:
        training = [
            _to_device(sampler.sample(authority, 0, mode="fixed")[0], device)
            for authority in train_ids
        ]
        torch.manual_seed(args.seed)
        initial, _optimizer = _model(config, args, device)
        residuals = {}
        for record, batch in zip(subsets["training"], training):
            magnitude = _target_albedo_residual_magnitude(initial, batch)
            residuals[_crop_id(record)] = magnitude
            if magnitude < args.minimum_target_residual:
                raise RuntimeError(
                    f"V17.3 training authority has weak frozen baseline-residual "
                    f"{_crop_id(record)}={magnitude:.6f} "
                    f"<{args.minimum_target_residual}; no reranking after selection"
                )
        for record, batch in zip(subsets["validation"], validation):
            magnitude = _target_albedo_residual_magnitude(initial, batch)
            if magnitude < args.minimum_target_residual:
                raise RuntimeError(
                    f"V17.3 independent validation crop has insufficient residual "
                    f"{_crop_id(record)}={magnitude:.6f}; no adaptive heldout selection"
                )
        del initial, _optimizer
        audit = {
            "trainingNativeSources": sampler.audit,
            "trainingCandidateResiduals": training_screen,
            "rejectedTrainingCandidates": rejected_screen,
            "trainAuthorities": train_ids,
            "independentValidationAuthorities": [_id(r) for r in subsets["validation"]],
            "sealedFinalTestAuthorities": [_id(r) for r in subsets["final_test"]],
            "v2ProtectedAuthorities": sorted(_family_ids([
                *(v2["trainingCropIds"]), *(v2["validationCropIds"]), *(v2["finalTestCropIds"]),
            ])),
            "materialPolicy": "unsupervised neutral when missing; material loss unchanged",
            "normalPolicy": "alpha-green if supported, otherwise explicitly reported red-green fallback",
        }
        (run_dir / "physical_map_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True)+"\n", encoding="utf-8",
        )
        results = {
            arm: _train_arm(
                arm, args, run_dir, contract, sampler, training,
                train_ids, validation, config, device, residuals,
            )
            for arm in arms
        }
        candidates = [result for result in results.values() if result["bestQualified"]]
        selected_arm = max(candidates, key=lambda item: item["bestValidationScore"]) if candidates else None
        final_result, final_previews = None, {}
        if selected_arm is not None:
            # Fail closed on interrupted final-test evaluation. The sealed set
            # must never be replayed when resuming a partially completed proof.
            sentinel = run_dir / "final_test_started.json"
            if sentinel.exists():
                raise RuntimeError(
                    "V17.3 independent final test was already opened in this "
                    "incomplete run; cannot evaluate it a second time"
                )
            sentinel.write_text(
                json.dumps({
                    "schema": SCHEMA,
                    "selectedArm": selected_arm["arm"],
                    "finalTestCropIds": contract["partitionCropIds"]["final_test"],
                    "oneShot": True,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
            model, optimizer = _model(config, args, device)
            _restore(
                Path(selected_arm["selectedCheckpoint"]), model, optimizer,
                contract, str(selected_arm["arm"]), device,
            )
            # ONLY here is the eight-authority independent final test unsealed.
            test_batches = [
                _batch_for_record(record, config, seed=args.seed+5000+i, device=device)[0]
                for i, record in enumerate(subsets["final_test"])
            ]
            metrics = []
            final_dir = run_dir / "final_test_once"
            for i, (record, batch) in enumerate(zip(subsets["final_test"], test_batches)):
                item, outputs = _evaluate(model, batch, config,
                                          device=device, precision=args.amp_precision)
                metrics.append(item)
                image = final_dir / f"final_test_{i:02d}_ABC.png"
                _write_preview(image, batch, outputs, role="final_test")
                final_previews[_crop_id(record)] = str(image.resolve())
            final_result = _summarize(metrics, sibling=True)
        decision = (
            "final-test-visual-review-required" if final_result and final_result["allPass"]
            else "reject-final-transfer" if final_result is not None
            else "reject-independent-validation"
        )
        report = {
            "schema": SCHEMA, "researchOnly": True, "productionPromotionAllowed": False,
            "decision": decision, "qualificationGatesModified": False,
            "architectureRevision": "V17.2-unchanged",
            "contract": contract, "physicalMapAudit": str((run_dir/"physical_map_audit.json").resolve()),
            "armResults": results,
            "selectedArm": selected_arm["arm"] if selected_arm else None,
            "trainingCropIds": _ids(subsets["training"]),
            "validationCropIds": _ids(subsets["validation"]),
            "finalTestCropIds": _ids(subsets["final_test"]),
            "finalTestSealedUntilSelection": True,
            "finalTestEvaluated": final_result is not None,
            "finalTestEvaluationCount": 1 if final_result is not None else 0,
            "selectedFinalTest": final_result,
            "finalTestPreviews": final_previews,
            "trainGateThresholds": {
                "global": config.candidate_global_recovery_required,
                "edge": config.candidate_edge_recovery_required,
                "gradient": config.candidate_gradient_recovery_required,
                "lattice": config.candidate_lattice_cell_excess_max,
            },
            "independentTransferGateThresholds": EARLY_TRANSFER_GATE,
            "visualGateRequired": decision == "final-test-visual-review-required",
        }
        path = run_dir / "report.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        print(f"V17.3 decision={decision} selectedArm={report['selectedArm']} "
              f"finalTestEvaluated={report['finalTestEvaluated']} report={path}", flush=True)
        return 0, path
    finally:
        sampler.close()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--arm", choices=("both", "fixed", "spatial"), default="both")
    p.add_argument("--authorities", type=int, default=12)
    p.add_argument("--hr-size", type=int, default=512)
    p.add_argument("--stages", nargs="+", default=[STAGES])
    p.add_argument("--resume", choices=("fresh", "auto"), default="fresh")
    p.add_argument("--seed", type=int, default=17301)
    p.add_argument("--learning-rate", type=float, default=0.0002)
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
