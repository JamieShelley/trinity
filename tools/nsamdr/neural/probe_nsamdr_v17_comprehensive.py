#!/usr/bin/env python3
"""V17.4 A-F controlled proof: physical sources, sampling, degradation, training,
loss and architecture. Research-only, source-sealed and resumable per arm.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from probe_nsamdr_v16_full_broad import _autocast, _device, _proof_loss_terms, _to_device
from probe_nsamdr_v17_generalization import _augment_batch
from probe_nsamdr_v17_generalization_v2 import (
    _evaluate_set, _promote_checkpoint, _summarize, _validation_score, _sheets,
)
from probe_nsamdr_v17_sibling_architecture import (
    EARLY_TRANSFER_GATE, _batch_for_record, _evaluate, _train_one, _write_preview,
)
from probe_nsamdr_v17_spatial_generalization import (
    SCHEMA as V3_SCHEMA, _latest_v2, _model, _restore as _restore_v3,
)
from v14.baseline import normalize_xy
from v14.config import V16Config
from v14.dataset import _normalise_xy
from v14.qualification import sample_metrics
from v16.broad_prior import _detail_map
from v17.spatial_sampling import (
    NativeAuthoritySampler, _detail_integral, _family_ids, _id, _score_box,
)

SCHEMA = "NSAMDR_V17_COMPREHENSIVE_AF_PROOF_V1"
CHECKPOINT_SCHEMA = "NSAMDR_V17_COMPREHENSIVE_AF_CHECKPOINT_V1"
ROOT = "artifacts/nsamdr/diagnostics/v17_comprehensive_proof"
STAGES = (61440, 73728, 86016)
INITIAL_UPDATE = 49152
FREQ_WEIGHTS = {"detailBand3": 0.10, "detailBand7": 0.05}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _v3_report(repo: Path, explicit: str) -> tuple[Path, dict[str, Any]]:
    if explicit:
        path = Path(explicit)
        path = (repo / path).resolve() if not path.is_absolute() else path.resolve()
        value = _json(path)
        if value.get("schema") != V3_SCHEMA:
            raise RuntimeError("V17.4 requires an authentic completed V17.3 proof")
        return path, value
    root = repo / "artifacts/nsamdr/diagnostics/v17_spatial_generalization"
    reports = []
    for path in root.glob("proof_*/report.json"):
        try:
            value = _json(path)
            if value.get("schema") == V3_SCHEMA and "spatial" in value.get("armResults", {}):
                reports.append((path.stat().st_mtime, path.resolve(), value))
        except (ValueError, OSError, TypeError):
            continue
    if not reports:
        raise RuntimeError("No completed V17.3 spatial proof report is available")
    _stamp, path, value = max(reports)
    return path, value


def _origin(repo: Path, explicit: str) -> dict[str, Any]:
    path, report = _v3_report(repo, explicit)
    contract = report.get("contract") or {}
    spatial = report.get("armResults", {}).get("spatial") or {}
    if (report.get("finalTestEvaluated") or report.get("finalTestEvaluationCount")
            or int(spatial.get("bestValidationUpdate") or 0) != INITIAL_UPDATE
            or report.get("decision") != "reject-independent-validation"):
        raise RuntimeError("V17.4 expects the sealed, nonqualified V17.3 49,152-update proof")
    manifest = Path(contract.get("manifest") or "")
    v2 = Path(contract.get("v2Report") or "")
    checkpoint = Path(spatial.get("selectedCheckpoint") or "")
    for label, source in (("corpus manifest", manifest), ("V2 proof", v2),
                          ("spatial checkpoint", checkpoint)):
        if not source.is_file():
            raise RuntimeError(f"V17.4 missing original {label}: {source}")
    if _sha(manifest) != contract.get("manifestSha256") or _sha(v2) != contract.get("v2ReportSha256"):
        raise RuntimeError("V17.4 source corpus/V2 proof fingerprint changed since V17.3")
    if contract.get("seed") != 17301 or float(contract.get("learningRate", -1)) != 0.0002:
        raise RuntimeError("Unsupported initial V17.3 training contract")
    splits = contract.get("partitionCropIds") or {}
    if any(len(splits.get(k) or []) != n for k, n in (
        ("training", 12), ("validation", 4), ("final_test", 8)
    )):
        raise RuntimeError("V17.3 authoritative training/validation/sealed split changed")
    if contract.get("architecture") != {
        "encoderChannels": 96, "decoderMidChannels": 64,
        "decoderDetailChannels": 48, "decoderBlocksPerStage": 4,
    }:
        raise RuntimeError("V17.3 architecture differs from V17.4 frozen continuation")
    return {
        "v3Report": str(path), "v3ReportSha256": _sha(path),
        "manifest": str(manifest), "v2Report": str(v2),
        "checkpoint": str(checkpoint), "checkpointSha256": _sha(checkpoint),
        "v3Contract": contract, "v3BestScore": float(spatial["bestValidationScore"]),
        "splits": splits,
    }


def _records(source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest = _json(Path(source["manifest"]))
    indexed = {str(r.get("crop_id") or r.get("cropId")): r
               for r in manifest.get("crops", [])}
    subsets: dict[str, list[dict[str, Any]]] = {}
    all_ids = []
    for role in ("training", "validation", "final_test"):
        expected_split = "train" if role == "training" else "validation"
        subset = []
        for crop in source["splits"][role]:
            if crop not in indexed:
                raise RuntimeError(f"V17.3 frozen authority crop missing: {crop}")
            row = dict(indexed[crop])
            if row.get("split") != expected_split or not crop.endswith("_000"):
                raise RuntimeError(f"V17.4 invalid frozen {role} provenance: {crop}")
            subset.append(row)
            all_ids.append(_id(row))
        subsets[role] = subset
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("V17.4 split violates whole-authority independence")
    v2 = _json(Path(source["v2Report"]))
    if v2.get("schema") != "NSAMDR_V17_GENERALIZATION_V2_PROOF_V1":
        raise RuntimeError("V2 source schema changed")
    protected = _family_ids([*(v2.get("trainingCropIds") or []),
                              *(v2.get("validationCropIds") or []),
                              *(v2.get("finalTestCropIds") or [])])
    if protected.intersection(all_ids):
        raise RuntimeError("V17.4 would expose a V2 protected authority")
    return manifest, subsets


def _contract(source: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "source": source,
        "stages": list(STAGES), "frequencyLoss": FREQ_WEIGHTS,
        "seed": 17301, "learningRate": 0.0002, "hrSize": 512,
        "patience": int(args.patience), "minImprovement": float(args.min_improvement),
        "freezeSplits": True, "maxFinalTestEvaluations": 1,
        "samplerVersion": "V17.3-native-spatial-unchanged",
        "arms": ["D_continuation", "E_frequency_loss"],
    }


def _run_dir(repo: Path, contract: dict[str, Any], resume: str) -> Path:
    root = repo / ROOT
    if resume == "fresh":
        path = root / f"proof_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        path.mkdir(parents=True, exist_ok=False)
        (path / "run_manifest.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        return path
    if resume != "auto":
        raise ValueError("resume must be fresh or auto")
    folders = [p for p in root.glob("proof_*")
               if (p / "run_manifest.json").is_file() and not (p / "report.json").is_file()]
    if not folders:
        raise RuntimeError("No incomplete unified A-F proof to resume; use fresh")
    path = max(folders, key=lambda p: p.stat().st_mtime)
    if _json(path / "run_manifest.json") != contract:
        raise RuntimeError("Unified A-F resume contract mismatch: choose fresh, do not mix runs")
    return path


def _audit_a(manifest: dict[str, Any], parts: dict[str, list[dict[str, Any]]],
             sampler: NativeAuthoritySampler) -> dict[str, Any]:
    families = {str(row.get("familyId") or row.get("family_id")): row
                for row in manifest.get("families", [])}
    issues = []
    warnings = []
    locations: dict[str, list[str]] = {}
    for role, rows in parts.items():
        for row in rows:
            authority = _id(row)
            family = families.get(authority)
            if not family:
                issues.append(f"missing authority metadata: {authority}")
                continue
            for channel in ("albedo", "normal", "material"):
                meta = family.get(channel)
                if not isinstance(meta, dict) or not meta.get("path"):
                    if channel != "material":
                        issues.append(f"missing mandatory {channel} metadata: {authority}")
                    continue
                identity = str(meta["path"]).replace("\\", "/").casefold()
                locations.setdefault(f"{channel}|{identity}", []).append(authority)
    duplicates = {k: sorted(set(v)) for k, v in locations.items()
                  if len(set(v)) > 1}
    if duplicates:
        issues.append("duplicate authored source paths across independent authorities")
    for authority, audit in sampler.audit.items():
        if audit["normalEncoding"] != "alpha-green":
            warnings.append(f"{authority}: normal uses {audit['normalEncoding']}")
        if not audit["materialSupervised"]:
            warnings.append(f"{authority}: no supervised packed-material map")
        if min(*audit["albedoSize"], *audit["normalSize"]) < 1024:
            issues.append(f"{authority}: native input smaller than 1024")
    return {
        "status": "FAIL" if issues else "WARN" if warnings else "PASS",
        "hardFailures": issues, "warnings": warnings, "sourcePathDuplicates": duplicates,
        "readAuthorityRoles": ["training"],
        "untouchedSealedFinalAuthorities": [_id(r) for r in parts["final_test"]],
        "normalEncodingCounts": {
            key: sum(x["normalEncoding"] == key for x in sampler.audit.values())
            for key in sorted({x["normalEncoding"] for x in sampler.audit.values()})
        },
        "materialSupervisedAuthorities": sum(
            bool(x["materialSupervised"]) for x in sampler.audit.values()
        ),
        "perTrainingAuthority": sampler.audit,
        "v2AuthorityPixelsOpened": 0, "finalTestPixelsOpened": 0,
    }


def _antialias_integral(family: Any) -> np.ndarray:
    a, n = family.albedo, family.normal
    side_w = max(1, int(round(a.shape[1] / max(1, min(a.shape[:2]) // 256))))
    side_h = max(1, int(round(a.shape[0] / max(1, min(a.shape[:2]) // 256))))
    low_a = cv2.resize(a.astype(np.float32), (side_w, side_h),
                       interpolation=cv2.INTER_AREA) / 255.0
    low_n = cv2.resize(n.astype(np.float32), (side_w, side_h),
                       interpolation=cv2.INTER_AREA)
    low_n = _normalise_xy(low_n / 127.5 - 1.0)
    return cv2.integral(_detail_map(low_a, low_n).astype(np.float32),
                        sdepth=cv2.CV_64F)


def _audit_b(sampler: NativeAuthoritySampler, *, seed: int) -> dict[str, Any]:
    entries = []
    for authority, family in sampler.families.items():
        rng = random.Random(seed + int(authority[:8], 16))
        h, w = family.albedo.shape[:2]
        positions = [(rng.randint(0, w-512), rng.randint(0, h-512))
                     for _ in range(24)]
        original = family.detail_integral
        corrected = _antialias_integral(family)
        score = lambda integral, pos: _score_box(integral, pos[0], pos[1], 512, w, h)
        old_scores = [score(original, pos) for pos in positions]
        new_scores = [score(corrected, pos) for pos in positions]
        old_index = int(np.argmax(old_scores))
        new_index = int(np.argmax(new_scores))
        entries.append({
            "authorityId": authority, "candidateCount": len(positions),
            "legacyChosenOrigin": list(positions[old_index]),
            "antialiasedChosenOrigin": list(positions[new_index]),
            "sameChosenOrigin": old_index == new_index,
            "legacyMaxScore": old_scores[old_index],
            "antialiasedMaxScore": new_scores[new_index],
            "meanAbsoluteCandidateScoreDifference": float(
                np.abs(np.asarray(old_scores)-np.asarray(new_scores)).mean()
            ),
        })
    agree = sum(x["sameChosenOrigin"] for x in entries)
    return {
        "status": "WARN" if agree != len(entries) else "PASS",
        "samplingChangedInTraining": False,
        "samplingChangedAuthorities": len(entries) - agree,
        "trainingAuthoritiesAudited": len(entries),
        "method": "24 identical candidate regions scored by strided versus INTER_AREA anti-aliased importance maps",
        "perAuthority": entries,
        "interpretation": "Diagnostic only: D/E retain the original V17.3 sampler",
    }


def _audit_c(manifest: dict[str, Any], parts: dict[str, list[dict[str, Any]]],
             sampler: NativeAuthoritySampler) -> dict[str, Any]:
    source_by_id = {str(f.get("familyId")): f for f in manifest.get("families", [])}
    entries = []
    for row in parts["training"]:
        authority = _id(row)
        fam = sampler.families[authority]
        x, y = fam.fixed_origin
        crop = fam.albedo[y:y+512, x:x+512]
        area = cv2.resize(crop, (128,128), interpolation=cv2.INTER_AREA)
        linear = cv2.resize(crop, (128,128), interpolation=cv2.INTER_LINEAR)
        cubic = cv2.resize(crop, (128,128), interpolation=cv2.INTER_CUBIC)
        path = Path((source_by_id[authority].get("albedo") or {}).get("path") or "")
        direct_mip = None
        if path.suffix.lower() == ".dds" and path.is_file():
            try:
                with Image.open(path) as im:
                    original_w, original_h = im.size
                    for level in range(1, min(12, getattr(im, "n_frames", 1))):
                        im.seek(level)
                        if im.size == (original_w // 4, original_h // 4):
                            candidate = np.asarray(im.convert("RGB"))
                            sample = candidate[y//4:(y+512)//4, x//4:(x+512)//4]
                            if sample.shape == (128,128,3) and not (x%4 or y%4):
                                direct_mip = float(np.abs(
                                    sample.astype(np.float32) -
                                    area.astype(np.float32)).mean() / 255.0)
                            break
            except (OSError, EOFError, ValueError, RuntimeError):
                pass
        entries.append({
            "authorityId": authority, "source": path.suffix.lower(),
            "actualAuthoredMipComparison": (
                "available" if direct_mip is not None else "unavailable"
            ),
            "actualMipVsSyntheticAreaMAE": direct_mip,
            "syntheticLinearVsAreaMAE": float(
                np.abs(linear.astype(np.float32)-area.astype(np.float32)).mean()/255),
            "syntheticCubicVsAreaMAE": float(
                np.abs(cubic.astype(np.float32)-area.astype(np.float32)).mean()/255),
        })
    real = [x["actualMipVsSyntheticAreaMAE"] for x in entries
            if x["actualMipVsSyntheticAreaMAE"] is not None]
    return {
        "status": "WARN" if real and max(real) > 0.03
                  else "PASS" if real else "UNAVAILABLE",
        "actualMipComparisons": len(real),
        "syntheticFilterComparisons": len(entries),
        "maximumRealMipMismatch": max(real) if real else None,
        "perAuthority": entries,
        "limitation": "Pillow may not expose EVE DDS mip chains; synthetic filter tests cannot prove a runtime mip mismatch",
        "trainingDegradationChanged": False,
    }


def _batch_config(source: dict[str, Any]) -> V16Config:
    config = V16Config(
        dataset_manifest=source["manifest"], train_hr_size=512, train_lr_size=128,
        validation_hr_size=512, validation_lr_size=128, minimum_heldout_samples=4,
        tiles_per_epoch=1, validation_tiles=1, clean_epochs=1,
        robust_epochs=1, selector_epochs=1, use_gradient_checkpointing=False,
    )
    config.validate()
    return config


def _model_args() -> argparse.Namespace:
    return argparse.Namespace(seed=17301, learning_rate=0.0002,
                              encoder_channels=96, mid_channels=64,
                              detail_channels=48, decoder_blocks=4)


def _load_v3_origin(source: dict[str, Any], config: V16Config,
                    device: torch.device):
    model, opt = _model(config, _model_args(), device)
    payload = _restore_v3(Path(source["checkpoint"]), model, opt,
                          source["v3Contract"], "spatial", device)
    if payload.get("update") != INITIAL_UPDATE:
        raise RuntimeError("V17.3 selected source checkpoint does not contain update 49,152")
    return model, opt


def _frequency_loss(outputs: dict[str, torch.Tensor],
                    batch: dict[str, torch.Tensor]) -> torch.Tensor:
    prediction = outputs["candidate_albedo"].float()
    target = batch["target_albedo"].float()
    result = prediction.new_zeros(())
    for width, weight in ((3, FREQ_WEIGHTS["detailBand3"]),
                          (7, FREQ_WEIGHTS["detailBand7"])):
        p = prediction-F.avg_pool2d(prediction, width, stride=1, padding=width//2)
        t = target-F.avg_pool2d(target, width, stride=1, padding=width//2)
        result = result + float(weight)*F.l1_loss(p, t)
    return result


def _train_update(model, opt, batch, config, device, precision: str,
                  *, frequency: bool) -> None:
    if not frequency:
        _train_one(model, opt, batch, config, device=device, precision=precision)
        return
    model.train()
    opt.zero_grad(set_to_none=True)
    with _autocast(device, precision):
        outputs = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
        base = _proof_loss_terms(outputs, batch, config)["total"]
        total = base + _frequency_loss(outputs, batch)
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
    opt.step()


def _save_stage(path: Path, model, optimizer, contract: dict[str, Any],
                arm: str, update: int, history: list[dict[str, Any]],
                best_update: int, best_score: float, stale: int,
                best_qualified: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema": CHECKPOINT_SCHEMA, "contract": contract, "arm": arm,
        "update": update, "history": history, "bestUpdate": best_update,
        "bestScore": best_score, "stale": stale, "bestQualified": best_qualified,
        "architecture": model.architecture_contract(),
        "modelState": model.state_dict(), "optimizerState": optimizer.state_dict(),
        "torchRngState": torch.get_rng_state(),
        "cudaRngState": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _load_stage(path: Path, model, opt, contract, arm, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if (state.get("schema") != CHECKPOINT_SCHEMA
            or state.get("contract") != contract or state.get("arm") != arm
            or state.get("architecture") != model.architecture_contract()):
        raise RuntimeError("A-F checkpoint provenance mismatch; resume refused")
    model.load_state_dict(state["modelState"], strict=True)
    opt.load_state_dict(state["optimizerState"])
    torch.set_rng_state(state["torchRngState"].cpu())
    if device.type == "cuda" and state.get("cudaRngState"):
        torch.cuda.set_rng_state_all(state["cudaRngState"])
    return state


def _train_arm(arm: str, path: Path, contract: dict[str, Any],
               source: dict[str, Any], config: V16Config,
               sampler: NativeAuthoritySampler, train_ids: list[str],
               frozen_train: list[dict[str, torch.Tensor]],
               validation: list[dict[str, torch.Tensor]],
               device: torch.device, precision: str) -> dict[str, Any]:
    root = path / arm
    done = root / "arm_report.json"
    if done.is_file():
        return _json(done)
    model, opt = _load_v3_origin(source, config, device)
    stage_files = list(root.glob("stage_*/checkpoint.pt"))
    history = []
    best_update = start = INITIAL_UPDATE
    best_score = float(source["v3BestScore"])
    best_qualified = False
    stale = 0
    if stage_files:
        latest = max(stage_files, key=lambda p: int(p.parent.name.split("_")[1]))
        state = _load_stage(latest, model, opt, contract, arm, device)
        start = int(state["update"])
        history = list(state["history"])
        best_update = int(state["bestUpdate"])
        best_score = float(state["bestScore"])
        stale = int(state["stale"])
        best_qualified = bool(state["bestQualified"])
        if start not in STAGES:
            raise RuntimeError("Unexpected A-F resume checkpoint update")
    selected = (Path(source["checkpoint"]) if best_update == INITIAL_UPDATE
                else root / f"stage_{best_update:06d}" / "checkpoint.pt")
    t0 = time.monotonic()
    stop_reason = "bounded-stage-limit"
    if stale >= contract["patience"]:
        stop_reason = "validation-plateau"
    elif best_qualified:
        stop_reason = "train-and-validation-qualified"
    else:
        for update in range(start+1, STAGES[-1]+1):
            i = (update-1) % len(train_ids)
            exposure = (update-1)//len(train_ids)
            raw, _ = sampler.sample(train_ids[i], exposure, mode="spatial")
            batch = _to_device(raw, device)
            batch = _augment_batch(batch, exposure % 8)
            _train_update(model, opt, batch, config, device, precision,
                          frequency=(arm == "E_frequency_loss"))
            if update == start+1 or update%256 == 0:
                elapsed = max(time.monotonic()-t0, 1e-6)
                progress = (update-start)/(STAGES[-1]-start)
                eta = (1-progress)*elapsed/max(progress,1e-6)/60
                print(f"[v17-af] arm={arm} update {update}/{STAGES[-1]} "
                      f"elapsed={elapsed/60:.1f}m eta={eta:.1f}m", flush=True)
            if update not in STAGES:
                continue
            trained = _evaluate_set(model, frozen_train, config, device=device,
                                    precision=precision, sibling=False)
            val = _evaluate_set(model, validation, config, device=device,
                                precision=precision, sibling=True)
            score = _validation_score(val)
            qualifies = bool(trained["allPass"] and val["allPass"])
            improved = _promote_checkpoint(
                qualified=qualifies, previous_qualified=best_qualified,
                score=score, best_score=best_score,
                min_improvement=contract["minImprovement"], no_best=False,
            )
            if improved:
                best_update, best_score, best_qualified, stale = (
                    update, score, qualifies, 0
                )
            else:
                stale += 1
            history.append({
                "update": update, "trained": trained, "validation": val,
                "validationScore": score, "newBest": improved,
                "bestUpdate": best_update, "staleStages": stale,
            })
            checkpoint = root / f"stage_{update:06d}" / "checkpoint.pt"
            _save_stage(checkpoint, model, opt, contract, arm, update, history,
                        best_update, best_score, stale, best_qualified)
            if improved:
                selected = checkpoint
            print(f"[v17-af] arm={arm} stage={update} "
                  f"trainPass={trained['allPass']} validationPass={val['allPass']} "
                  f"score={score:.6f} best={best_update} stale={stale}", flush=True)
            if qualifies:
                stop_reason = "train-and-validation-qualified"
                break
            if stale >= contract["patience"]:
                stop_reason = "validation-plateau"
                break
    if best_update == INITIAL_UPDATE:
        model, opt = _load_v3_origin(source, config, device)
        selected_train = _evaluate_set(model, frozen_train, config,
                                       device=device, precision=precision, sibling=False)
        selected_val = _evaluate_set(model, validation, config,
                                     device=device, precision=precision, sibling=True)
    else:
        _load_stage(selected, model, opt, contract, arm, device)
        record = next(h for h in history if h["update"] == best_update)
        selected_train, selected_val = record["trained"], record["validation"]
    preview = _sheets(root/f"selected_{best_update:06d}", model, validation,
                      source["splits"]["validation"], config,
                      device=device, precision=precision, role="validation")
    result = {
        "arm": arm, "startUpdate": INITIAL_UPDATE,
        "stopUpdate": history[-1]["update"] if history else INITIAL_UPDATE,
        "stopReason": stop_reason, "bestValidationUpdate": best_update,
        "bestValidationScore": best_score, "bestQualified": best_qualified,
        "selectedCheckpoint": str(selected.resolve()),
        "selectedTrained": selected_train, "selectedValidation": selected_val,
        "history": history, "validationPreviews": preview,
        "lossModification": FREQ_WEIGHTS if arm == "E_frequency_loss" else {},
        "elapsedSeconds": time.monotonic()-t0,
    }
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result


def _ablate(model, batch: dict[str, torch.Tensor], config: V16Config,
            device: torch.device, precision: str) -> dict[str, Any]:
    model.eval()
    with torch.no_grad(), _autocast(device, precision):
        output = model(batch["lr_albedo"],batch["lr_normal"],batch["lr_material"])
        lr_maps = torch.cat((
            batch["lr_albedo"].float(), batch["lr_normal"].float(),
            batch["lr_material"].float(),
        ), dim=1)
        latent = model.encoder(lr_maps)
        without_latent = model.decoder(torch.zeros_like(latent), lr_maps)
        branches = {
            "full": output["decoder_mid_residual_hr"] + output["decoder_detail_residual"],
            "mid_only": output["decoder_mid_residual_hr"],
            "detail_only": output["decoder_detail_residual"],
            "encoder_zeroed": (
                without_latent["mid_residual_hr"] +
                without_latent["detail_residual"]
            ),
            "no_learned_residual": torch.zeros_like(output["decoder_detail_residual"]),
        }
        results = {}
        for name, raw in branches.items():
            o = dict(output)
            o["candidate_albedo"] = (output["baseline_albedo"] +
                torch.tanh(raw[:,0:3].float())*config.albedo_residual_cap).clamp(0,1)
            o["candidate_normal"] = normalize_xy(
                output["baseline_normal"]+
                torch.tanh(raw[:,3:5].float())*config.normal_residual_cap)
            o["candidate_material"] = (output["baseline_material"] +
                torch.tanh(raw[:,5:8].float())*config.material_residual_cap).clamp(0,1)
            results[name] = sample_metrics(o,batch,final=False)
    return results


def _audit_f(arm_results, source, contract, config, validation, device,
             precision) -> dict[str, Any]:
    # No speculative new model: show what the already-trained mid and detail
    # residual branches contribute on exactly the frozen development crops.
    winner = max(arm_results.values(),key=lambda x:x["bestValidationScore"])
    model, optimizer = _load_v3_origin(source,config,device)
    if winner["bestValidationUpdate"] != INITIAL_UPDATE:
        _load_stage(Path(winner["selectedCheckpoint"]),model,optimizer,
                    contract,winner["arm"],device)
    samples = [_ablate(model,batch,config,device,precision) for batch in validation]
    keys = ("full","mid_only","detail_only","encoder_zeroed","no_learned_residual")
    means = {}
    for branch in keys:
        means[branch] = {
            key: sum(row[branch][key] for row in samples)/len(samples)
            for key in ("global_recovery","edge_recovery","gradient_recovery",
                        "detail_recovery_1px","lattice_cell_excess")
        }
    return {
        "status": "PASS", "selectedResearchArm": winner["arm"],
        "selectedResearchUpdate": winner["bestValidationUpdate"],
        "branchAblationMeans": means,
        "perDevelopmentCrop": samples,
        "architectureChanged": False, "architectureUpgradeProven": False,
        "encoderTheoreticalLRReceptiveFieldPixels": 97,
        "encoderReceptiveFieldNote": "97 LR pixels for encoder stem + 8 two-conv dilation blocks + tail; decoder adds contextual extent",
        "warning": "Removing a trained branch is an inference ablation, not a trained alternative architecture",
    }


def _final_once(path, winner, source, contract, config, parts, device, precision):
    # Final test must be opened at most once, only after training + validation
    # qualification. On interruption after marker creation, fail closed.
    marker = path/"final_test_started.json"
    if marker.exists():
        raise RuntimeError("Final test previously opened in this proof; replay refused")
    model, opt = _load_v3_origin(source,config,device)
    if winner["bestValidationUpdate"] != INITIAL_UPDATE:
        _load_stage(Path(winner["selectedCheckpoint"]),model,opt,
                    contract,winner["arm"],device)
    marker.write_text(json.dumps({
        "selectedArm":winner["arm"],"sealedIds":source["splits"]["final_test"],
        "oneShot":True},indent=2)+"\n",encoding="utf-8")
    batches = [
        _batch_for_record(row,config,seed=17301+5000+i,device=device)[0]
        for i,row in enumerate(parts["final_test"])
    ]
    items=[]; images={}
    folder=path/"final_test_once"
    for i,(row,batch) in enumerate(zip(parts["final_test"],batches)):
        metric,output=_evaluate(model,batch,config,device=device,precision=precision)
        items.append(metric)
        target=folder/f"final_test_{i:02d}_ABC.png"
        _write_preview(target,batch,output,role="final_test")
        images[source["splits"]["final_test"][i]]=str(target.resolve())
    return _summarize(items,sibling=True),images


def _matched_comparison(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Unbiased same-update D/E paired development comparison, not best-vs-best."""
    if not all(k in results for k in ("D_continuation", "E_frequency_loss")):
        return []
    d = {int(row["update"]): row for row in results["D_continuation"]["history"]}
    e = {int(row["update"]): row for row in results["E_frequency_loss"]["history"]}
    paired = []
    for update in sorted(d.keys() & e.keys()):
        a, b = d[update], e[update]
        ma, mb = a["validation"]["mean"], b["validation"]["mean"]
        row = {"update": update,
               "D_validationScore": a["validationScore"],
               "E_validationScore": b["validationScore"],
               "deltaValidationScore": b["validationScore"]-a["validationScore"]}
        for k in ("global_recovery","edge_recovery","gradient_recovery",
                  "detail_recovery_1px","lattice_cell_excess"):
            row["D_"+k] = ma[k]
            row["E_"+k] = mb[k]
            row["delta_"+k] = mb[k]-ma[k]
        paired.append(row)
    return paired


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    phases=report["phases"]
    lines=[
        "# NSAMDR V17.4 — Unified A–F proof",
        "",
        f"Decision: **{report['decision']}**",
        f"Original V17.3 checkpoint: {INITIAL_UPDATE:,} updates (read-only)",
        "",
        "| Phase | Status | Evidence |",
        "|---|---|---|",
    ]
    for key,phase in phases.items():
        message=str(phase.get("summary") or phase.get("limitation") or
                    phase.get("warning") or "").replace("|","/")
        lines.append(f"| {key} | {phase.get('status','UNKNOWN')} | {message[:180]} |")
    lines+=["","## Training comparison","",
            "| Arm | Best update | Validation score | Mean global | Mean edge | Mean 1px detail | Qualified |",
            "|---|---:|---:|---:|---:|---:|---|"]
    for arm,item in report.get("armResults",{}).items():
        m=item["selectedValidation"]["mean"]
        lines.append("| "+ " | ".join([
            arm,str(item["bestValidationUpdate"]),f"{item['bestValidationScore']:.4f}",
            f"{100*m['global_recovery']:.2f}%",
            f"{100*m['edge_recovery']:.2f}%",
            f"{100*m['detail_recovery_1px']:.2f}%",
            str(bool(item["bestQualified"])),
        ])+" |")
    if report.get("matchedComparison"):
        lines += [
            "", "## Same-update D versus E development comparison", "",
            "| Update | D edge | E edge | E − D edge (pp) | D 1px | E 1px | Score E − D |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for entry in report["matchedComparison"]:
            lines.append("| "+" | ".join([
                str(entry["update"]),
                f"{100*entry['D_edge_recovery']:.2f}%",
                f"{100*entry['E_edge_recovery']:.2f}%",
                f"{100*entry['delta_edge_recovery']:+.2f}",
                f"{100*entry['D_detail_recovery_1px']:.2f}%",
                f"{100*entry['E_detail_recovery_1px']:.2f}%",
                f"{entry['deltaValidationScore']:+.5f}",
            ])+" |")
    lines+=["","Final-test evaluated: **"+str(report["finalTestEvaluated"])+"**",
            "Sealed authorities: "+str(len(report["finalTestCropIds"])),
            "",
            "The qualification thresholds remain unchanged. A/B/C validation previews ",
            "are in the D/E selected-checkpoint folders. Source and sampling phases ",
            "inspect training sources only; Phase F examines development validation ",
            "only. UNAVAILABLE is never interpreted as PASS.",
            ""]
    (path/"report.md").write_text("\n".join(lines),encoding="utf-8")


def run(args: argparse.Namespace) -> tuple[int,Path]:
    if args.patience < 1 or args.min_improvement < 0:
        raise ValueError("patience must be positive and min improvement nonnegative")
    repo=args.repo_root.resolve()
    source=_origin(repo,args.v3_report)
    manifest,parts=_records(source)
    contract=_contract(source,args)
    folder=_run_dir(repo,contract,args.resume)
    device=_device("cpu" if args.audit_only else args.device)
    config=_batch_config(source)
    print(f"[v17-af] A-F proof={folder} auditOnly={args.audit_only} "
          f"train={len(parts['training'])} validation={len(parts['validation'])} "
          f"sealedFinalTest={len(parts['final_test'])}",flush=True)
    phases: dict[str,dict[str,Any]]={}
    audit_path=folder/"ABC_audits.json"
    try:
        sampler=NativeAuthoritySampler(
            repo,manifest,parts["training"],hr_size=512,seed=17301,
        )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        # Decode/missing-source errors are part of audit A; emit a durable
        # blocked proof rather than leaving a misleading incomplete train run.
        cause=f"{type(exc).__name__}: {exc}"
        phases={
            "A_source":{"status":"FAIL","hardFailures":[cause],
                        "summary":"Mandatory training source could not be decoded"},
            "B_sampling":{"status":"SKIPPED","summary":"A source failed"},
            "C_degradation":{"status":"SKIPPED","summary":"A source failed"},
            "D_continuation":{"status":"SKIPPED","summary":"A source failed"},
            "E_frequency_loss":{"status":"SKIPPED","summary":"A source failed"},
            "F_architecture":{"status":"SKIPPED","summary":"A source failed"},
        }
        report={
            "schema":SCHEMA,"researchOnly":True,
            "productionPromotionAllowed":False,
            "qualificationGatesModified":False,
            "decision":"reject-source-audit",
            "phases":phases,"armResults":{},
            "sourceV3Report":source["v3Report"],
            "sourceCheckpointSha256":source["checkpointSha256"],
            "trainingCropIds":source["splits"]["training"],
            "validationCropIds":source["splits"]["validation"],
            "finalTestCropIds":source["splits"]["final_test"],
            "finalTestSealedUntilQualification":True,
            "finalTestEvaluated":False,"finalTestEvaluationCount":0,
            "selectedFinalTest":None,"finalTestPreviews":{},
            "visualGateRequired":False,
        }
        output=folder/"report.json"
        output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                          encoding="utf-8")
        _write_markdown(folder,report)
        print(f"[v17-af] BLOCKED source audit: {cause}; report={output}",
              flush=True)
        return 0,output
    try:
        if audit_path.is_file():
            audits=_json(audit_path)
        else:
            audits={
                "A_source":_audit_a(manifest,parts,sampler),
                "B_sampling":_audit_b(sampler,seed=17301),
                "C_degradation":_audit_c(manifest,parts,sampler),
            }
            audit_path.write_text(
                json.dumps(audits,indent=2,sort_keys=True)+"\n",encoding="utf-8",
            )
        phases.update(audits)
        for key,audit in audits.items():
            print(f"[v17-af] {key} status={audit['status']}",flush=True)
        if audits["A_source"]["status"] == "FAIL":
            for key in ("D_continuation","E_frequency_loss","F_architecture"):
                phases[key]={"status":"SKIPPED","summary":"Hard source audit failure"}
            result_arms={}
        elif args.audit_only:
            for key in ("D_continuation","E_frequency_loss","F_architecture"):
                phases[key]={"status":"SKIPPED","summary":"Audit-only requested"}
            result_arms={}
        else:
            train_ids=[_id(row) for row in parts["training"]]
            fixed=[_to_device(sampler.sample(authority,0,mode="fixed")[0],device)
                   for authority in train_ids]
            development=[
                _batch_for_record(row,config,seed=17301+1001+i,device=device)[0]
                for i,row in enumerate(parts["validation"])
            ]
            result_arms={}
            for key in ("D_continuation","E_frequency_loss"):
                result_arms[key]=_train_arm(
                    key,folder,contract,source,config,sampler,train_ids,fixed,
                    development,device,args.amp_precision,
                )
                arm=result_arms[key]
                phases[key]={
                    "status":"PASS" if arm["bestQualified"] else "WARN",
                    "summary":f"Best validation update {arm['bestValidationUpdate']}, "
                              f"score={arm['bestValidationScore']:.5f}; "
                              f"qualified={arm['bestQualified']}",
                }
            try:
                phases["F_architecture"]=_audit_f(
                    result_arms,source,contract,config,development,
                    device,args.amp_precision,
                )
            except Exception as exc:
                phases["F_architecture"]={
                    "status":"FAIL","summary":f"Architecture branch ablation failed: {exc}",
                }
        choices=[item for item in result_arms.values() if item["bestQualified"]]
        winner=max(choices,key=lambda x:x["bestValidationScore"]) if choices else None
        final,images=None,{}
        if winner is not None:
            final,images=_final_once(
                folder,winner,source,contract,config,parts,device,args.amp_precision,
            )
        decision=("audit-only" if args.audit_only
                  else "reject-source-audit" if phases["A_source"]["status"]=="FAIL"
                  else "reject-independent-validation" if not winner
                  else "final-test-visual-review-required" if final and final["allPass"]
                  else "reject-final-transfer")
        report={
            "schema":SCHEMA,"researchOnly":True,"productionPromotionAllowed":False,
            "qualificationGatesModified":False,"decision":decision,
            "sourceV3Report":source["v3Report"],"sourceCheckpointSha256":source["checkpointSha256"],
            "phases":phases,"armResults":result_arms,
            "matchedComparison":_matched_comparison(result_arms),
            "selectedQualifiedArm":winner["arm"] if winner else None,
            "trainingCropIds":source["splits"]["training"],
            "validationCropIds":source["splits"]["validation"],
            "finalTestCropIds":source["splits"]["final_test"],
            "finalTestSealedUntilQualification":True,
            "finalTestEvaluated":final is not None,
            "finalTestEvaluationCount":int(final is not None),
            "selectedFinalTest":final,"finalTestPreviews":images,
            "visualGateRequired":decision=="final-test-visual-review-required",
            "existingTrainGateThresholds":{
                "global":config.candidate_global_recovery_required,
                "edge":config.candidate_edge_recovery_required,
                "gradient":config.candidate_gradient_recovery_required,
                "lattice":config.candidate_lattice_cell_excess_max,
            },
            "existingTransferGateThresholds":EARLY_TRANSFER_GATE,
        }
        file=folder/"report.json"
        file.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        _write_markdown(folder,report)
        print(f"[v17-af] decision={decision} sealedTestOpened={report['finalTestEvaluated']} "
              f"report={file}",flush=True)
        return 0,file
    finally:
        sampler.close()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=Path.cwd())
    p.add_argument("--v3-report",default="")
    p.add_argument("--resume",choices=("fresh","auto"),default="fresh")
    p.add_argument("--audit-only",action="store_true")
    p.add_argument("--patience",type=int,default=2)
    p.add_argument("--min-improvement",type=float,default=0.002)
    p.add_argument("--device",choices=("auto","cuda","cpu"),default="cuda")
    p.add_argument("--amp-precision",choices=("auto","bf16","fp16"),default="auto")
    return p


if __name__=="__main__":
    run(parser().parse_args())
