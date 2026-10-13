#!/usr/bin/env python3
"""V17.6 Stage 10A.1: diagnostic and hard-authority feature-capacity proof.

Research-only. No validation or sealed final-test pixels in this proof. Uses
the frozen Stage 9 A12 backbone, the original authored-training authority
indices 1 and 10, and unchanged Stage 10A losses/physical caps.

Phases:
  diagnose:          inspect original Stage 10A 0/3072/6144 checkpoints
  stability:         short authority-1 trial through the known failure interval
  capacity:          2 isolated 10-channel specialist fits (one per authority)
  compare-features:  2 matched pairs, 10 vs 58 frozen-HR-feature channels
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

HERE=Path(__file__).resolve().parent
REPO_ROOT=HERE.parents[2]
for folder in (REPO_ROOT,HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0,str(folder))

from probe_nsamdr_v16_full_broad import _autocast, _device, _to_device
from probe_nsamdr_v17_comprehensive import _origin, _records, _batch_config
from probe_nsamdr_v17_generalization import _augment_batch
from probe_nsamdr_v17_sibling_architecture import _evaluate
from probe_nsamdr_v176_detail import (
    _atomic_json, _parent_report, _read, _tensor_sha, verify_parent,
    SCHEMA as STAGE10A_SCHEMA, CHECKPOINT_SCHEMA as STAGE10A_CHECKPOINT,
    DETAIL_SEED,
)
from tools.nsamdr.neural.v17.diversity_corpus import (
    selected_rows, sha256_file, sha256_json,
)
from tools.nsamdr.neural.v17.model import NSAMDRV17
from tools.nsamdr.neural.v17.spatial_sampling import NativeAuthoritySampler, _id
from tools.nsamdr.neural.v176.fine_detail import NSAMDRV176Detail
from tools.nsamdr.neural.v176.detail_capacity import (
    HARD_TRAINING_INDICES, MODES, HRFeatureDetail, create_matched_variants,
    diagnostic_batch,
)
from tools.nsamdr.neural.v176.stability import (
    STABILITY_POLICY, NumericalStabilityError, guarded_train_step, save_failure,
)

SCHEMA="NSAMDR_V176_STAGE10A1_CAPACITY_ISOLATION_V1"
CHECKPOINT_SCHEMA="NSAMDR_V176_STAGE10A1_CAPACITY_CHECKPOINT_V1"
ROOT="artifacts/nsamdr/diagnostics/v176_stage10a1_capacity"
STAGES=(0,1536,3072,6144)
STABILITY_STAGES=(0,256,512,1024,1536,1792,2048,2560,2816,3072)
ALT_EXPOSURES=(20000,20001,20002,20003)
SEED=17611
LEARNING_RATE=.0002
PHASES=("diagnose","stability","capacity","compare-features")


def _folder(repo:Path,resume:str,phase:str)->Path:
    root=repo/ROOT
    if resume=="fresh":
        path=root/("proof_"+datetime.now().strftime("%Y%m%d-%H%M%S"))
        path.mkdir(parents=True,exist_ok=False)
        return path
    possible=[p for p in root.glob("proof_*")
              if (p/"run_manifest.json").is_file()
              and not (p/"report.json").is_file()
              and not ((p/"failure.json").is_file() and
                       _read(p/"failure.json").get("decision")=="BLOCKED_NUMERICAL_STABILITY")
              and _read(p/"run_manifest.json").get("phase")==phase]
    if len(possible)!=1:
        raise RuntimeError(f"Stage 10A.1 {phase} resume requires one unfinished proof; got {len(possible)}")
    return possible[0]


def _parent_stage10a(repo:Path,explicit:str)->Path:
    if explicit:
        p=Path(explicit).resolve()
        if not p.is_file():
            raise RuntimeError(f"specified Stage 10A report missing: {p}")
        return p
    root=repo/"artifacts/nsamdr/diagnostics/v176_stage10a_detail"
    reports=[]
    for p in root.glob("proof_*/report.json"):
        try:
            r=_read(p)
        except (OSError,ValueError):
            continue
        if (r.get("schema")==STAGE10A_SCHEMA and
            r.get("specialistUpdates")==6144 and
            r.get("finalTestEvaluated") is False):
            reports.append(p)
    if not reports:
        raise RuntimeError("completed Stage 10A training report unavailable")
    return max(reports,key=lambda p:p.stat().st_mtime).resolve()


def verify_stage10a_report(path:Path,backbone_audit:dict)->tuple[dict,dict]:
    report=_read(path)
    contract=_read(path.parent/"run_manifest.json")
    provenance=_read(path.parent/"provenance.json")
    if (report.get("schema")!=STAGE10A_SCHEMA or
        report.get("researchOnly") is not True or
        report.get("finalTestEvaluated") is not False or
        report.get("finalTestEvaluationCount")!=0 or
        report.get("productionPromotionAllowed") is not False or
        report.get("qualificationGatesModified") is not False or
        report.get("specialistUpdates")!=6144 or
        report.get("runContract")!=contract or
        [row.get("update") for row in report.get("history",[])]!=list(STAGES) or
        contract.get("parentBackboneTensorSha256")!=backbone_audit["stage9BackboneTensorSha256"] or
        contract.get("parentA12CheckpointSha256")!=backbone_audit["stage9A12CheckpointSha256"] or
        contract.get("trainingIds")!=backbone_audit["trainingAuthorityIds"] or
        contract.get("sealedFinalCropIds")!=backbone_audit["sealedFinalCropIds"] or
        contract.get("detailSeed")!=DETAIL_SEED or
        provenance!=backbone_audit):
        raise RuntimeError("Stage 10A completed report or frozen-backbone lineage differs")
    checkpoint_hashes={}
    for update in (0,3072,6144):
        path_ckpt=path.parent/f"stage_{update:06d}"/"checkpoint.pt"
        if not path_ckpt.is_file():
            raise RuntimeError(f"missing original Stage 10A checkpoint: {update}")
        data=torch.load(path_ckpt,map_location="cpu",weights_only=True)
        if (data.get("schema")!=STAGE10A_CHECKPOINT or
            data.get("contract")!=contract or
            data.get("update")!=update or
            [row.get("update") for row in data.get("history",[])]!=
            [stage for stage in STAGES if stage<=update] or
            not isinstance(data.get("detailState"),dict) or
            not isinstance(data.get("optimizerState"),dict) or
            not isinstance(data.get("torchRngState"),torch.Tensor)):
            raise RuntimeError(f"Stage 10A {update} checkpoint identity differs")
        checkpoint_hashes[str(update)]=sha256_file(path_ckpt)
    return contract,{"stage10aReport":str(path.resolve()),
                     "stage10aReportSha256":sha256_file(path),
                     "stage10aCheckpointSha256":checkpoint_hashes,
                     "stage10aRunManifestSha256":sha256_file(path.parent/"run_manifest.json")}


def _samples(sampler:NativeAuthoritySampler,authority:str,device:torch.device):
    fixed,_=sampler.sample(authority,0,mode="fixed")
    alternatives=[]
    metadata=[]
    for exposure in ALT_EXPOSURES:
        sample,meta=sampler.sample(authority,exposure,mode="spatial")
        alternatives.append(_to_device(sample,device))
        metadata.append(meta)
    return _to_device(fixed,device),alternatives,metadata


def _evaluation(model, batch, config,device,amp)->dict:
    result,_=_evaluate(model,batch,config,device=device,precision=amp)
    return {"metrics":result["metrics"],"gateChecks":result["gateChecks"],
            "residualDiagnostics":result["residualDiagnostics"]}


def _snapshot(model, fixed,alternates,config,device,amp,update:int)->dict:
    try:
        fixed_result=_evaluation(model,fixed,config,device,amp)
        stats=diagnostic_batch(model,fixed)
    except NumericalStabilityError as exc:
        exc.context.update({"evaluationProbe":"fixed"})
        raise
    alt=[]
    for index,batch in enumerate(alternates):
        try:
            alt.append(_evaluation(model,batch,config,device,amp))
        except NumericalStabilityError as exc:
            exc.context.update({"evaluationProbe":"alternate","alternateIndex":index})
            raise
    return {
        "update":update,"fixed":fixed_result,"alternateRegions":alt,
        "alternateMeanEdge":sum(row["metrics"]["edge_recovery"] for row in alt)/len(alt),
        "alternateWorstEdge":min(row["metrics"]["edge_recovery"] for row in alt),
        "alternateMaxLattice":max(row["metrics"]["lattice_cell_excess"] for row in alt),
        "diagnostic":stats,
    }


def _train_step(model,opt,batch,device,amp,collect:bool,*,failure_path=None,context=None)->dict|None:
    try:
        metrics=guarded_train_step(model,opt,batch,_autocast(device,amp),collect=collect)
        return metrics if collect else None
    except NumericalStabilityError as exc:
        if failure_path is not None:
            save_failure(failure_path,exc,model,opt,context or {})
        raise


def _save(path,model,opt,contract,arm,update,history):
    path.parent.mkdir(parents=True,exist_ok=True)
    obj={"schema":CHECKPOINT_SCHEMA,"contract":contract,"arm":arm,
         "update":update,"history":history,
         "architecture":model.architecture_contract(),
         "detailState":model.specialist.state_dict(),
         "optimizerState":opt.state_dict(),
         "torchRngState":torch.get_rng_state().cpu().contiguous(),
         "cudaRngState":[x.cpu().contiguous() for x in torch.cuda.get_rng_state_all()]
                         if torch.cuda.is_available() else None}
    tmp=path.with_suffix(".tmp")
    torch.save(obj,tmp)
    tmp.replace(path)


def _restore(path,model,opt,contract,arm,device):
    obj=torch.load(path,map_location="cpu",weights_only=True)
    stages=tuple(contract.get("stages",STAGES))
    if (obj.get("schema")!=CHECKPOINT_SCHEMA or obj.get("contract")!=contract or
        obj.get("arm")!=arm or obj.get("architecture")!=model.architecture_contract() or
        obj.get("update") not in stages or
        [x.get("update") for x in obj.get("history",[])]!=
        [x for x in stages if x<=obj["update"]]):
        raise RuntimeError("Stage 10A.1 resume contract/arm/history mismatch")
    model.specialist.load_state_dict(obj["detailState"],strict=True)
    opt.load_state_dict(obj["optimizerState"])
    torch.set_rng_state(obj["torchRngState"].cpu().contiguous())
    if device.type=="cuda":
        state=obj.get("cudaRngState")
        if not isinstance(state,list) or not state:
            raise RuntimeError("Stage 10A.1 CUDA RNG state missing")
        torch.cuda.set_rng_state_all([x.cpu().contiguous() for x in state])
    return obj


def _make_backbone(config,checkpoint,device):
    model=NSAMDRV17(config,encoder_channels=96,mid_channels=64,
                   detail_channels=48,decoder_blocks=4).to(device)
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (state.get("schema")!="NSAMDR_V17_5_MATCHED_CONTINUATION_CHECKPOINT_V1" or
        state.get("update")!=49152 or state.get("arm")!="A12" or
        state.get("architecture")!=model.architecture_contract()):
        raise RuntimeError("Stage 9 source checkpoint invalid")
    model.load_state_dict(state["modelState"],strict=True)
    model.eval()
    return model


def _arm_name(index:int,mode:str)->str:
    if index not in HARD_TRAINING_INDICES or mode not in MODES:
        raise ValueError("only the two preregistered training indices and two modes")
    return f"train_{index:02d}_{mode}"


def _run_arm(root,model,contract,authority,index,mode,sampler,config,device,amp):
    arm=_arm_name(index,mode)
    folder=root/arm
    stages=tuple(contract.get("stages",STAGES))
    diagnostic_interval=16 if contract.get("phase")=="stability" else 256
    if (folder/"arm_report.json").is_file():
        r=_read(folder/"arm_report.json")
        if r.get("runContractSha256")!=sha256_json(contract) or r.get("arm")!=arm:
            raise RuntimeError("completed arm contract differs")
        return r
    fixed,alternate,meta=_samples(sampler,authority,device)
    if mode=="output10":
        candidate=model[0]
    else:
        candidate=model[1]
    candidate=candidate.to(device)
    expected_sha=contract["parentBackboneTensorSha256"]
    if _tensor_sha(candidate.backbone)!=expected_sha:
        raise RuntimeError("matched Stage 9 backbone weights differ")
    optimizer=torch.optim.Adam(candidate.trainable_parameters(),lr=LEARNING_RATE)
    checkpoints=sorted(folder.glob("stage_*/checkpoint.pt"))
    if checkpoints:
        state=_restore(checkpoints[-1],candidate,optimizer,contract,arm,device)
        start=int(state["update"])
        history=list(state["history"])
    else:
        start=0
        try:
            history=[_snapshot(candidate,fixed,alternate,config,device,amp,0)]
        except NumericalStabilityError as exc:
            probe=exc.context.get("evaluationProbe")
            origin=meta[exc.context["alternateIndex"]] if probe=="alternate" else {"samplingPolicy":"fixed"}
            save_failure(folder/"failure_tensors.pt",exc,candidate,optimizer,
                         {"arm":arm,"authorityIndex":index,"authorityId":authority,"update":0,
                          "cropId":contract.get("trainingAuthorities",{}).get(str(index),{}).get("cropId"),
                          "sourceRegion":origin,**exc.context})
            raise
        _save(folder/"stage_000000"/"checkpoint.pt",candidate,optimizer,
              contract,arm,0,history)
    began=time.monotonic()
    recent=[]
    for update in range(start+1,stages[-1]+1):
        raw,origin=sampler.sample(authority,update-1,mode="spatial")
        batch=_augment_batch(_to_device(raw,device),(update-1)%8)
        context={"arm":arm,"authorityIndex":index,"authorityId":authority,
                 "cropId":contract.get("trainingAuthorities",{}).get(str(index),{}).get("cropId"),
                 "update":update,"exposure":update-1,"augmentation":(update-1)%8,
                 "sourceRegion":origin}
        metrics=_train_step(candidate,optimizer,batch,device,amp,
                            collect=(update==start+1 or update%diagnostic_interval==0),
                            failure_path=folder/"failure_tensors.pt",context=context)
        if metrics is not None:
            row={**context,**metrics}
            recent.append(row)
            recent=recent[-16:]
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/"telemetry.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
        if update not in stages[1:]:
            if update%256==0:
                print(f"[v176-capacity] {arm} {update}/{stages[-1]}",flush=True)
            continue
        try:
            snap=_snapshot(candidate,fixed,alternate,config,device,amp,update)
        except NumericalStabilityError as exc:
            probe=exc.context.get("evaluationProbe")
            origin=meta[exc.context["alternateIndex"]] if probe=="alternate" else {"samplingPolicy":"fixed"}
            save_failure(folder/"failure_tensors.pt",exc,candidate,optimizer,
                         {**context,"sourceRegion":origin,**exc.context})
            raise
        snap["recentTrainGradients"]=list(recent)
        snap["backboneSha256"]=_tensor_sha(candidate.backbone)
        if snap["backboneSha256"]!=expected_sha:
            raise RuntimeError("frozen Stage 9 backbone changed")
        history.append(snap)
        _save(folder/f"stage_{update:06d}"/"checkpoint.pt",
              candidate,optimizer,contract,arm,update,history)
        _atomic_json(folder/"progress.json",{"arm":arm,"update":update,
                                              "fixedEdge":snap["fixed"]["metrics"]["edge_recovery"],
                                              "alternateMeanEdge":snap["alternateMeanEdge"]})
    if [x["update"] for x in history]!=list(stages):
        raise RuntimeError("bounded authority specialization history incomplete")
    result={"arm":arm,"authorityIndex":index,"authorityId":authority,"mode":mode,
            "startUpdate":0,"stopUpdate":stages[-1],"history":history,
            "alternateProbeOrigins":meta,
            "runContractSha256":sha256_json(contract),
            "elapsedSeconds":time.monotonic()-began,
            "qualification":("TRAIN_REGIONS_PASS" if
                history[-1]["fixed"]["gateChecks"]["edge"] and
                history[-1]["fixed"]["gateChecks"]["lattice"] and
                all(row["gateChecks"]["edge"] and row["gateChecks"]["lattice"]
                    for row in history[-1]["alternateRegions"])
                else "TRAIN_REGIONS_UNQUALIFIED")}
    _atomic_json(folder/"arm_report.json",result)
    return result


def _decision(arms:dict, phase:str)->dict:
    if phase=="stability":
        return {"decision":"STABILITY_TRIAL_COMPLETED_TRAIN_ONLY",
                "numericalStability":"WITHIN_REGISTERED_LIMITS",
                "comparisons":{}}
    comparisons={}
    for index in HARD_TRAINING_INDICES:
        plain=arms.get(_arm_name(index,"output10"))
        rich=arms.get(_arm_name(index,"hr_features58"))
        if plain:
            last=plain["history"][-1]
            entry={"plainFixedEdge":last["fixed"]["metrics"]["edge_recovery"],
                   "plainAlternateEdge":last["alternateMeanEdge"],
                   "plainFixedGlobal":last["fixed"]["metrics"]["global_recovery"],
                   "plainFixedLattice":last["fixed"]["metrics"]["lattice_cell_excess"],
                   "plainLattice":last["alternateMaxLattice"],
                   "plainTrainRegionsQualify":plain["qualification"]=="TRAIN_REGIONS_PASS"}
            if rich:
                later=rich["history"][-1]
                entry.update({
                    "richFixedEdge":later["fixed"]["metrics"]["edge_recovery"],
                    "richAlternateEdge":later["alternateMeanEdge"],
                    "richFixedGlobal":later["fixed"]["metrics"]["global_recovery"],
                    "richFixedLattice":later["fixed"]["metrics"]["lattice_cell_excess"],
                    "richLattice":later["alternateMaxLattice"],
                    "richMinusPlainFixedEdge":later["fixed"]["metrics"]["edge_recovery"]
                        -entry["plainFixedEdge"],
                    "richMinusPlainAlternateEdge":later["alternateMeanEdge"]
                        -entry["plainAlternateEdge"],
                    "richTrainRegionsQualify":rich["qualification"]=="TRAIN_REGIONS_PASS",
                })
            comparisons[str(index)]=entry
    if phase=="capacity":
        status=("CAPACITY_SIGNAL_TRAIN_ONLY" if
                all(comparisons[str(i)]["plainTrainRegionsQualify"] for i in HARD_TRAINING_INDICES)
                else "ISOLATED_STANDARD_SPECIALIST_UNQUALIFIED")
    else:
        status=("RICH_HR_FEATURE_SIGNAL_TRAIN_ONLY" if
                all(comparisons[str(i)].get("richMinusPlainFixedEdge",0) >= .02 and
                    comparisons[str(i)].get("richMinusPlainAlternateEdge",0) >= .02 and
                    comparisons[str(i)]["richLattice"] <= .15 and
                    comparisons[str(i)]["richFixedLattice"] <= .15 and
                    comparisons[str(i)]["richFixedGlobal"] >=
                        comparisons[str(i)]["plainFixedGlobal"]-.01
                    for i in HARD_TRAINING_INDICES)
                else "NO_PROVEN_HR_FEATURE_BENEFIT")
    return {"decision":status,"comparisons":comparisons}


def run(args):
    repo=args.repo_root.resolve()
    folder=_folder(repo,args.resume,args.phase)
    try:
        detail_path=_parent_stage10a(repo,args.stage10a_report)
        original=_read(detail_path)
        stage9path=Path(original["stage9SourceReport"]).resolve()
        source=_origin(repo,args.v3_report)
        manifest,parts=_records(source)
        config=_batch_config(source)
        # Authenticate both completed ancestor proofs before source pixel access.
        backbone=NSAMDRV17(config,encoder_channels=96,mid_channels=64,
                           detail_channels=48,decoder_blocks=4)
        selection,stage9audit=verify_parent(stage9path,source,manifest,parts,backbone)
        detail_contract,detail_audit=verify_stage10a_report(detail_path,stage9audit)
        rows=selected_rows(selection,manifest,"A12")
        if len(rows)!=12 or [_id(r) for r in rows]!=stage9audit["trainingAuthorityIds"]:
            raise RuntimeError("hard-authority indices are not the original frozen A12")
        indices=(1,) if args.phase=="stability" else HARD_TRAINING_INDICES
        stages=STABILITY_STAGES if args.phase=="stability" else STAGES
        identities={str(i):{"authorityId":_id(rows[i]),
                            "cropId":rows[i].get("crop_id") or rows[i].get("cropId")}
                    for i in indices}
        contract={"schema":SCHEMA,"phase":args.phase,
                  "parentStage9CheckpointSha256":stage9audit["stage9A12CheckpointSha256"],
                  "parentBackboneTensorSha256":stage9audit["stage9BackboneTensorSha256"],
                  "parentStage10AReportSha256":detail_audit["stage10aReportSha256"],
                  "parentStage10AStagesSha256":detail_audit["stage10aCheckpointSha256"],
                  "sourceManifestSha256":stage9audit["sourceManifestSha256"],
                  "selectionSha256":stage9audit["stage8SelectionSha256"],
                  "trainingAuthorities":identities,
                  "seed":SEED,"stages":list(stages),
                  "alternateExposures":list(ALT_EXPOSURES),
                  "learningRate":LEARNING_RATE,"loss":"unchanged Stage10A stationary Haar",
                  "frozenBackbone":True,"ampPrecision":args.amp_precision,
                  "numericalStability":dict(STABILITY_POLICY),
                  "device":args.device,"finalTestEnabled":False,
                  "developmentPixelsOpened":0,"finalTestPixelsOpened":0}
        manifest_path=folder/"run_manifest.json"
        if args.resume=="auto":
            if _read(manifest_path)!=contract:
                raise RuntimeError("Stage10A.1 resume manifest/source selection changed")
        else:
            _atomic_json(manifest_path,contract)
            _atomic_json(folder/"provenance.json",{**stage9audit,**detail_audit})
        report={"schema":SCHEMA,"phase":args.phase,"decision":"RUNNING",
                "researchOnly":True,"productionPromotionAllowed":False,
                "qualificationGatesModified":False,
                "developmentPixelsOpened":0,"finalTestEvaluated":False,
                "finalTestEvaluationCount":0,
                "runContract":contract,"armResults":{}}
        if args.audit_only:
            report["decision"]="PARENT_PROVENANCE_VERIFIED_ONLY"
            report["auditOnly"]=True
            _atomic_json(folder/"report.json",report)
            return 0,folder/"report.json"
        device=_device(args.device)
        backbone=_make_backbone(config,Path(stage9audit["stage9A12Checkpoint"]),device)
        if _tensor_sha(backbone)!=stage9audit["stage9BackboneTensorSha256"]:
            raise RuntimeError("Stage9 frozen backbone tensor SHA mismatch")
        sampler=NativeAuthoritySampler(
            repo,manifest,[rows[i] for i in indices],
            hr_size=512,seed=17301,
        )
        try:
            if args.phase=="diagnose":
                results={}
                for update in (0,3072,6144):
                    torch.manual_seed(DETAIL_SEED)
                    model=NSAMDRV176Detail(backbone,detail_channels=32).to(device)
                    ckpt=detail_path.parent/f"stage_{update:06d}"/"checkpoint.pt"
                    data=torch.load(ckpt,map_location="cpu",weights_only=True)
                    if data["contract"]!=detail_contract:
                        raise RuntimeError("original specialist checkpoint source differs")
                    model.specialist.load_state_dict(data["detailState"],strict=True)
                    measured={}
                    for i in HARD_TRAINING_INDICES:
                        fid=_id(rows[i])
                        fixed,alternate,origins=_samples(sampler,fid,device)
                        measured[str(i)]={"metrics":_snapshot(
                            model,fixed,alternate,config,device,args.amp_precision,update),
                            "alternateProbeOrigins":origins}
                    results[str(update)]=measured
                report["diagnostics"]=results
                report["decision"]="CORRECTION_DIAGNOSTICS_RESEARCH_ONLY"
            else:
                armresults={}
                for i in indices:
                    torch.manual_seed(SEED+i)
                    plain,rich=create_matched_variants(backbone,seed=SEED+i)
                    pairs=(("output10",plain),) if args.phase in ("capacity","stability") else (
                        ("output10",plain),("hr_features58",rich))
                    for mode,_unused in pairs:
                        result=_run_arm(folder,(plain,rich),contract,_id(rows[i]),i,
                                        mode,sampler,config,device,args.amp_precision)
                        armresults[result["arm"]]=result
                report["armResults"]=armresults
                report.update(_decision(armresults,args.phase))
        finally:
            sampler.close()
        _atomic_json(folder/"report.json",report)
        print(f"[v176-capacity] {report['decision']} report={folder/'report.json'}",flush=True)
        return 0,folder/"report.json"
    except BaseException as exc:
        error={"schema":SCHEMA,"decision":"BLOCKED","phase":args.phase,
               "reason":f"{type(exc).__name__}: {exc}",
               "traceback":traceback.format_exc(),
               "researchOnly":True,"developmentPixelsOpened":0,
               "finalTestEvaluated":False,
               "productionPromotionAllowed":False}
        if isinstance(exc,NumericalStabilityError):
            error.update({"decision":"BLOCKED_NUMERICAL_STABILITY",
                          "numericalDiagnostic":exc.diagnostic,
                          "failingContext":exc.context})
        _atomic_json(folder/"failure.json",error)
        if not any(folder.glob("*/stage_*/checkpoint.pt")):
            _atomic_json(folder/"report.json",error)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=Path.cwd())
    p.add_argument("--stage10a-report",default="")
    p.add_argument("--v3-report",default="")
    p.add_argument("--phase",choices=PHASES,default="diagnose")
    p.add_argument("--resume",choices=("fresh","auto"),default="fresh")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--amp-precision",choices=("auto","bf16","fp16"),default="auto")
    p.add_argument("--audit-only",action="store_true")
    return p


if __name__=="__main__":
    run(parser().parse_args())
