#!/usr/bin/env python3
"""Stage 9: exact matched A12/B64 continuation from Stage 8, research only.

Starts from the *last trained* 24,576-update optimizer state of each arm,
NOT an independently initialized model, selected earlier checkpoint, or a
new authority roster. No final-test pixels, no change to V17.2 or loss.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback
from typing import Any

import torch

from probe_nsamdr_v16_full_broad import _device, _to_device
from probe_nsamdr_v17_comprehensive import _batch_config, _origin, _records
from probe_nsamdr_v17_generalization import _augment_batch
from probe_nsamdr_v17_generalization_v2 import (
    _evaluate_set, _promote_checkpoint, _sheets, _validation_score,
)
from probe_nsamdr_v17_sibling_architecture import _batch_for_record, _train_one
from probe_nsamdr_v17_5_diversity import (
    _assert_unchanged_architecture, _atomic_json, _checkpoint_load as load_stage8,
    _make_model, _tensor_sha,
)
from v17.diversity_corpus import (
    SELECTION_SEED, selected_rows, sha256_file, sha256_json,
)
from v17.diversity_continuation import (
    ARMS, SCHEMA as PROVENANCE_SCHEMA, UPDATES, decide,
    diagnostic_summary, find_stage8_report, probe_groups, verify_stage8,
)
from v17.spatial_sampling import NativeAuthoritySampler, _id

SCHEMA = "NSAMDR_V17_5_MATCHED_CONTINUATION_PROOF_V1"
CHECKPOINT_SCHEMA = "NSAMDR_V17_5_MATCHED_CONTINUATION_CHECKPOINT_V1"
ROOT = "artifacts/nsamdr/diagnostics/v17_5_matched_continuation"
START_UPDATE = 24576


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _new_directory(repo: Path, resume: str) -> Path:
    root = repo / ROOT
    if resume == "fresh":
        folder = root / ("proof_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        folder.mkdir(parents=True, exist_ok=False)
        return folder
    found = [p for p in root.glob("proof_*")
             if (p / "run_manifest.json").is_file() and
             not (p / "report.json").is_file()]
    if len(found) != 1:
        raise RuntimeError(f"Stage 9 auto-resume expects one unfinished proof, found {len(found)}")
    return found[0]


def _stage_save(path: Path, model, optimizer, contract: dict,
                arm: str, update: int, history: list, best: int,
                score: float, qualified: bool) -> None:
    if update not in UPDATES or arm not in ARMS:
        raise RuntimeError("invalid Stage 9 checkpoint stage or arm")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema":CHECKPOINT_SCHEMA, "contract":contract, "arm":arm,
        "update":update,"history":history,"bestUpdate":best,
        "bestScore":score,"bestQualified":qualified,
        "architecture":model.architecture_contract(),
        "modelState":model.state_dict(),"optimizerState":optimizer.state_dict(),
        "torchRngState":torch.get_rng_state().cpu().contiguous(),
        "cudaRngState":([x.cpu().contiguous() for x in torch.cuda.get_rng_state_all()]
                        if torch.cuda.is_available() else None),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload,temporary)
    temporary.replace(path)


def _stage_load(path: Path, model, optimizer, contract: dict,
                arm: str, device: torch.device) -> dict:
    payload = torch.load(path,map_location="cpu",weights_only=False)
    if (not isinstance(payload,dict) or payload.get("schema") != CHECKPOINT_SCHEMA or
        payload.get("contract") != contract or payload.get("arm") != arm or
        payload.get("update") not in UPDATES or
        payload.get("architecture") != model.architecture_contract()):
        raise RuntimeError("Stage 9 checkpoint contract/arm/architecture mismatch")
    model.load_state_dict(payload["modelState"],strict=True)
    optimizer.load_state_dict(payload["optimizerState"])
    torch.set_rng_state(payload["torchRngState"].cpu().contiguous())
    if device.type == "cuda" and payload.get("cudaRngState") is not None:
        torch.cuda.set_rng_state_all([x.cpu().contiguous() for x in payload["cudaRngState"]])
    return payload


def _contract(audit: dict, stage8: dict, *, device: str, amp: str) -> dict:
    return {
        "schema":SCHEMA, "parentStage8ReportSha256":audit["stage8ReportSha256"],
        "parentStage8ManifestSha256":audit["stage8RunManifestSha256"],
        "trainedSelectionSha256":audit["trainedSelectionSha256"],
        "parentCheckpointsSha256":audit["stage8CheckpointSha256"],
        "parentCheckpoints":audit["stage8CheckpointPaths"],
        "sourceManifestSha256":stage8["sourceManifestSha256"],
        "trainingSeed":stage8["trainingSeed"],"selectionSeed":17301,
        "initialModelSha256":stage8["initialModelSha256"],
        "architecture":stage8["architecture"],"learningRate":stage8["learningRate"],
        "startUpdate":START_UPDATE,"stages":list(UPDATES),
        "spatialPolicy":stage8["sampling"],"augmentation":stage8["augmentation"],
        "loss":stage8["loss"],"ampPrecision":amp,"device":device,
        "sealedFinalEnabled":False,"qualificationThresholdChanges":False,
    }


def _snapshot(model, fixed: list, development: list, config,
              device: torch.device, amp: str, arm: str, update: int) -> dict:
    train = _evaluate_set(model,fixed,config,device=device,precision=amp,sibling=False)
    dev = _evaluate_set(model,development,config,device=device,precision=amp,sibling=True)
    groups = probe_groups(train,arm)
    return {
        "update":update,"trained":train,"validation":dev,
        "trainProbeGroups":groups,
        "trainResidualDiagnostics":diagnostic_summary(train),
        "developmentResidualDiagnostics":diagnostic_summary(dev),
        "validationScore":_validation_score(dev),
        "qualified":bool(train["allPass"] and dev["allPass"]),
        "exposuresPerAuthority":update//len(fixed),
    }


def _run_arm(arm: str, folder: Path, contract: dict,
             repo: Path, selection: dict, manifest: dict, parts: dict,
             source: dict, stage8_report: dict, stage8_contract: dict,
             config, device: torch.device, amp: str) -> dict:
    root = folder / arm
    completed = root / "arm_report.json"
    if completed.is_file():
        result = _read(completed)
        if (result.get("arm") != arm or
            result.get("runContractSha256") != sha256_json(contract) or
            result.get("stopUpdate") != UPDATES[-1]):
            raise RuntimeError(f"completed Stage 9 arm mismatch: {arm}")
        return result
    rows = selected_rows(selection,manifest,arm)
    ids = [_id(row) for row in rows]
    if ids != stage8_report["armResults"][arm]["trainingAuthorities"]:
        raise RuntimeError(f"Stage 9 would change the trained {arm} roster")
    # The only native sources decoded here are training sources. The four
    # frozen development crops are separately loaded from the original V3.
    sampler = NativeAuthoritySampler(repo,manifest,rows,hr_size=512,seed=SELECTION_SEED)
    try:
        fixed = [_to_device(sampler.sample(fid,0,mode="fixed")[0],device) for fid in ids]
        development = [
            _batch_for_record(row,config,seed=SELECTION_SEED+1001+i,device=device)[0]
            for i,row in enumerate(parts["validation"])
        ]
        model,opt = _make_model(config,contract["trainingSeed"],device)
        _assert_unchanged_architecture(model,source)
        if _tensor_sha(model) != contract["initialModelSha256"]:
            raise RuntimeError("original Stage 8 initialization changed")
        stage9_paths = sorted(root.glob("stage_*/checkpoint.pt"))
        if stage9_paths:
            last = stage9_paths[-1]
            state = _stage_load(last,model,opt,contract,arm,device)
            start = int(state["update"])
            history = list(state["history"])
            best = int(state["bestUpdate"])
            best_score = float(state["bestScore"])
            qualified = bool(state["bestQualified"])
            if len(history) != len([x for x in UPDATES if x <= start]):
                raise RuntimeError("Stage 9 checkpoint history is incomplete")
        else:
            parent = Path(contract["parentCheckpoints"][arm])
            if sha256_file(parent) != contract["parentCheckpointsSha256"][arm]:
                raise RuntimeError("Stage 8 starting checkpoint bytes changed")
            state = load_stage8(parent,model,opt,stage8_contract,arm,device)
            if int(state["update"]) != START_UPDATE:
                raise RuntimeError("must continue final Stage 8 update, not best earlier update")
            if [x["update"] for x in state["history"]] != [6144,12288,24576]:
                raise RuntimeError("Stage 8 stage history changed")
            start = START_UPDATE
            history = []
            best = int(state["bestUpdate"])
            best_score = float(state["bestScore"])
            qualified = bool(state["bestQualified"])
        began = time.monotonic()
        for update in range(start+1,UPDATES[-1]+1):
            idx=(update-1)%len(ids)
            exposure=(update-1)//len(ids)
            raw,_ = sampler.sample(ids[idx],exposure,mode="spatial")
            augmented = _augment_batch(_to_device(raw,device),exposure%8)
            _train_one(model,opt,augmented,config,device=device,precision=amp)
            if update == start+1 or update%256==0:
                elapsed=max(time.monotonic()-began,1e-6)
                progress=(update-start)/max(1,UPDATES[-1]-start)
                print(f"[v17-continuation] arm={arm} update {update}/{UPDATES[-1]} "
                      f"elapsed={elapsed/60:.1f}m "
                      f"eta={(1-progress)*elapsed/max(progress,1e-6)/60:.1f}m",
                      flush=True)
            if update not in UPDATES:
                continue
            result = _snapshot(model,fixed,development,config,device,amp,arm,update)
            improved = _promote_checkpoint(
                qualified=result["qualified"],previous_qualified=qualified,
                score=result["validationScore"],best_score=best_score,
                min_improvement=.002,no_best=(best == 0))
            if improved:
                best,best_score,qualified=(update,result["validationScore"],
                                           result["qualified"])
            result["newBest"] = bool(improved)
            result["bestUpdate"] = best
            history.append(result)
            _stage_save(root/f"stage_{update:06d}"/"checkpoint.pt",
                        model,opt,contract,arm,update,history,best,best_score,qualified)
            print(f"[v17-continuation] arm={arm} stage={update} "
                  f"trainPass={result['trained']['allPass']} "
                  f"devPass={result['validation']['allPass']} "
                  f"score={result['validationScore']:.6f}",flush=True)
        if len(history) != len(UPDATES):
            raise RuntimeError(f"incomplete Stage 9 checkpoints for {arm}")
        latest = history[-1]
        # The raw final-stage comparison is essential: never compare an earlier
        # cherry-picked A12 checkpoint against a later B64 checkpoint.
        chosen_path = (root/f"stage_{best:06d}"/"checkpoint.pt" if best>START_UPDATE
                       else Path(contract["parentCheckpoints"][arm]) if best == START_UPDATE
                       else Path(stage8_report["armResults"][arm]["selectedCheckpoint"]))
        if best>START_UPDATE:
            _stage_load(chosen_path,model,opt,contract,arm,device)
        else:
            load_stage8(chosen_path,model,opt,stage8_contract,arm,device)
        previews = _sheets(root/f"selected_{best:06d}",model,development,
                           selection["developmentCropIds"],config,
                           device=device,precision=amp,role="validation")
        output={
            "arm":arm,"runContractSha256":sha256_json(contract),
            "startUpdate":START_UPDATE,"stopUpdate":UPDATES[-1],
            "bestValidationUpdate":best,"bestValidationScore":best_score,
            "selectedCheckpoint":str(chosen_path.resolve()),
            "selectedTrained":latest["trained"],
            "selectedValidation":latest["validation"],
            "selectedTrainedGroups":latest["trainProbeGroups"],
            "bestCheckpointWasUsedForPreviews":True,
            "atFinalUpdate":latest,
            "history":history,"validationPreviews":previews,
            "exposuresPerAuthority":UPDATES[-1]//len(ids),
            "trainingAuthorities":ids,
            "elapsedSeconds":time.monotonic()-began,
        }
        _atomic_json(completed,output)
        return output
    finally:
        sampler.close()


def _values(evaluation: dict, metric: str) -> list[float]:
    return [float(row["metrics"][metric]) for row in evaluation["perCrop"]]


def _compare(a: dict, b: dict, baseline: dict) -> list[dict]:
    result = []
    for update in (START_UPDATE,*UPDATES):
        if update == START_UPDATE:
            left = baseline["armResults"]["A12"]["history"][-1]
            right = baseline["armResults"]["B64"]["history"][-1]
        else:
            left = next(x for x in a["history"] if x["update"]==update)
            right = next(x for x in b["history"] if x["update"]==update)
        x,y = left["validation"],right["validation"]
        ae,be = _values(x,"edge_recovery"),_values(y,"edge_recovery")
        ag,bg = _values(x,"global_recovery"),_values(y,"global_recovery")
        al,bl = _values(x,"lattice_cell_excess"),_values(y,"lattice_cell_excess")
        result.append({
            "update":update,
            "A12MeanEdge":sum(ae)/4,"B64MeanEdge":sum(be)/4,
            "A12WorstEdge":min(ae),"B64WorstEdge":min(be),
            "A12MaxLattice":max(al),"B64MaxLattice":max(bl),
            "A12EdgeByCrop":ae,"B64EdgeByCrop":be,
            "edgeDeltaByCrop":[v-u for u,v in zip(ae,be)],
            "globalDeltaByCrop":[v-u for u,v in zip(ag,bg)],
            "A12TrainPass":left["trained"]["allPass"],
            "B64TrainPass":right["trained"]["allPass"],
            "A12DevelopmentPass":x["allPass"],"B64DevelopmentPass":y["allPass"],
            "A12ExposuresPerAuthority":update//12,
            "B64ExposuresPerAuthority":update//64,
        })
    return result


def _markdown(report: dict) -> str:
    lines=["# Stage 9 — matched A12/B64 continuation",
           "",f"Decision: **{report['decision']}**","",
           "Research-only; unchanged V17.2 architecture/loss and original gates.",
           "No eight-authority sealed final-test pixels opened.","",
           "| Update | A12 mean edge | B64 mean edge | A12 worst edge | B64 worst edge | B64 max lattice |",
           "|---:|---:|---:|---:|---:|---:|"]
    for item in report.get("matchedComparison",[]):
        lines.append(f"| {item['update']} | {item['A12MeanEdge']:.4f} | "
                     f"{item['B64MeanEdge']:.4f} | {item['A12WorstEdge']:.4f} | "
                     f"{item['B64WorstEdge']:.4f} | {item['B64MaxLattice']:.4f} |")
    lines += ["","Report includes common-A12/new52 probe groups and per-crop residual direction and phase diagnostics.",
              "Qualification and BenefitSelector/native rendering remain independently gated."]
    return "\n".join(lines)+"\n"


def run(args: argparse.Namespace) -> tuple[int,Path]:
    repo=args.repo_root.resolve()
    stage8_path=find_stage8_report(repo,args.stage8_report)
    source=_origin(repo,args.v3_report)
    manifest,parts=_records(source)
    selection,parent_contract,audit=verify_stage8(
        stage8_path,source,manifest,parts,preview_path=args.selection_only_report)
    if str(args.device) != parent_contract["device"] or args.amp_precision != parent_contract["ampPrecision"]:
        raise RuntimeError("Stage 9 requires exact Stage 8 CUDA device and AMP precision")
    device=_device(args.device)
    config=_batch_config(source)
    folder=_new_directory(repo,args.resume)
    try:
        contract=_contract(audit,parent_contract,device=str(device),amp=args.amp_precision)
        provenance=folder/"provenance.json"
        manifest_path=folder/"run_manifest.json"
        if args.resume=="auto":
            if _read(manifest_path)!=contract or _read(provenance)!=audit:
                raise RuntimeError("Stage 9 resume contract or trained roster provenance differs")
        else:
            _atomic_json(provenance,audit)
            _atomic_json(manifest_path,contract)
        report={"schema":SCHEMA,"decision":"RUNNING","researchOnly":True,
                "productionPromotionAllowed":False,"qualificationGatesModified":False,
                "finalTestEvaluated":False,"finalTestEvaluationCount":0,
                "selectedFinalTest":None,"startUpdate":START_UPDATE,
                "targetUpdate":UPDATES[-1],"runContract":contract,
                "stage8SourceReport":str(stage8_path),
                "trainedSelectionSha256":audit["trainedSelectionSha256"],
                "selectionPreviewComparison":audit["previewComparison"],
                "reportPath":str(folder/"report.json"),"armResults":{}}
        if args.audit_only:
            report["decision"]="PROVENANCE_VERIFIED_RESEARCH_ONLY"
            report["auditOnly"]=True
            _atomic_json(folder/"report.json",report)
            (folder/"report.md").write_text(_markdown(report),encoding="utf-8")
            print(f"[v17-continuation] source verified, no optimizer updates: {folder/'report.json'}",flush=True)
            return 0,folder/"report.json"
        parent = _read(stage8_path)
        results = {arm:_run_arm(arm,folder,contract,repo,selection,manifest,parts,
                                source,parent,parent_contract,config,device,args.amp_precision)
                   for arm in ARMS}
        report["armResults"]=results
        report["matchedComparison"]=_compare(results["A12"],results["B64"],parent)
        report["decision"]=decide(report["matchedComparison"],results)
        report["additionalOptimizerUpdates"]=2*(UPDATES[-1]-START_UPDATE)
        report["decisionBasis"]="Matched last-update 49152 results; prior best checkpoint retained separately"
        _atomic_json(folder/"report.json",report)
        (folder/"report.md").write_text(_markdown(report),encoding="utf-8")
        print(f"[v17-continuation] decision={report['decision']} "
              f"finalTestOpened=False report={folder/'report.json'}",flush=True)
        return 0,folder/"report.json"
    except BaseException as exc:
        failure={"schema":SCHEMA,"decision":"BLOCKED",
                 "reason":f"{type(exc).__name__}: {exc}",
                 "traceback":traceback.format_exc(),
                 "finalTestEvaluated":False,
                 "productionPromotionAllowed":False}
        _atomic_json(folder/"failure.json",failure)
        if not any(folder.glob("*/stage_*/checkpoint.pt")) and not any(folder.glob("*/arm_report.json")):
            _atomic_json(folder/"report.json",failure)
        print(f"[v17-continuation] BLOCKED {type(exc).__name__}: {exc}",flush=True)
        raise


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=Path.cwd())
    p.add_argument("--stage8-report",default="")
    p.add_argument("--v3-report",default="")
    p.add_argument("--selection-only-report",default="")
    p.add_argument("--audit-only",action="store_true")
    p.add_argument("--resume",choices=("fresh","auto"),default="fresh")
    p.add_argument("--device",choices=("cuda",),default="cuda")
    p.add_argument("--amp-precision",choices=("auto","bf16","fp16"),default="auto")
    return p


if __name__=="__main__":
    run(parser().parse_args())
