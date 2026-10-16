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
  attainable:        target-assisted scalar fits on those training regions
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
from probe_nsamdr_v17_sibling_architecture import _evaluate, _gate_snapshot, sample_metrics
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
    tensor_statistics,
)
from tools.nsamdr.neural.v176.attainable_recovery import (
    ATTAINABLE_POLICY, scalar_candidate, rgb_projection_relaxation,
    fit_scalar_field, band_relative_l1,
)
from tools.nsamdr.neural.v176.wavelet_supervision import detail_supervision_loss

SCHEMA="NSAMDR_V176_STAGE10A1_CAPACITY_ISOLATION_V1"
CHECKPOINT_SCHEMA="NSAMDR_V176_STAGE10A1_CAPACITY_CHECKPOINT_V1"
ROOT="artifacts/nsamdr/diagnostics/v176_stage10a1_capacity"
STAGES=(0,1536,3072,6144)
STABILITY_STAGES=(0,256,512,1024,1536,1792,2048,2560,2816,3072)
ALT_EXPOSURES=(20000,20001,20002,20003)
SEED=17611
LEARNING_RATE=.0002
PHASES=("diagnose","stability","capacity","compare-features","attainable")


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


def _verify_comparison(path:Path, lineage:dict)->tuple[dict,dict]:
    """Authenticate completed matched fits and their final weights before pixels."""
    report=_read(path)
    contract=_read(path.parent/"run_manifest.json")
    keys=("parentStage9CheckpointSha256","parentBackboneTensorSha256",
          "parentStage10AReportSha256","parentStage10AStagesSha256",
          "sourceManifestSha256","selectionSha256","trainingAuthorities",
          "seed","alternateExposures","learningRate","loss","numericalStability")
    if (report.get("schema")!=SCHEMA or report.get("phase")!="compare-features" or
        contract.get("phase")!="compare-features" or
        report.get("runContract")!=contract or
        contract.get("stages")!=list(STAGES) or contract.get("frozenBackbone") is not True or
        contract.get("developmentPixelsOpened")!=0 or contract.get("finalTestPixelsOpened")!=0 or
        contract.get("finalTestEnabled") is not False or
        report.get("decision") not in ("NO_PROVEN_HR_FEATURE_BENEFIT","RICH_HR_FEATURE_SIGNAL_TRAIN_ONLY") or
        report.get("researchOnly") is not True or report.get("developmentPixelsOpened")!=0 or
        report.get("finalTestEvaluated") is not False or report.get("finalTestEvaluationCount")!=0 or
        report.get("productionPromotionAllowed") is not False or
        report.get("qualificationGatesModified") is not False or
        any(contract.get(k)!=lineage[k] for k in keys)):
        raise RuntimeError("completed feature comparison or source lineage differs")
    expected={_arm_name(i,m) for i in HARD_TRAINING_INDICES for m in MODES}
    if set(report.get("armResults",{}))!=expected:
        raise RuntimeError("completed comparison requires all four matched arms")
    hashes={}
    for i in HARD_TRAINING_INDICES:
        for mode in MODES:
            arm=_arm_name(i,mode)
            result=report["armResults"][arm]
            history=result.get("history",[])
            ckpt=path.parent/arm/f"stage_{STAGES[-1]:06d}"/"checkpoint.pt"
            state=torch.load(ckpt,map_location="cpu",weights_only=True)
            if (result.get("arm")!=arm or result.get("mode")!=mode or
                result.get("authorityIndex")!=i or
                result.get("authorityId")!=lineage["trainingAuthorities"][str(i)]["authorityId"] or
                result.get("startUpdate")!=0 or result.get("stopUpdate")!=STAGES[-1] or
                result.get("runContractSha256")!=sha256_json(contract) or
                [row.get("update") for row in history]!=list(STAGES) or
                any(row.get("backboneSha256")!=lineage["parentBackboneTensorSha256"] for row in history[1:]) or
                state.get("schema")!=CHECKPOINT_SCHEMA or state.get("contract")!=contract or
                state.get("arm")!=arm or state.get("update")!=STAGES[-1] or
                state.get("history")!=history or not isinstance(state.get("detailState"),dict)):
                raise RuntimeError(f"completed feature comparison checkpoint differs: {arm}")
            hashes[arm]=sha256_file(ckpt)
    return report,{"comparisonReport":str(path.resolve()),
                   "comparisonReportSha256":sha256_file(path),
                   "comparisonCheckpointSha256":hashes}


def _comparison_source(repo:Path,explicit:str,lineage:dict)->tuple[Path,dict,dict]:
    candidates=([Path(explicit).resolve()] if explicit else
                sorted((repo/ROOT).glob("proof_*/report.json"),
                       key=lambda p:p.stat().st_mtime,reverse=True))
    for path in candidates:
        try:
            report,audit=_verify_comparison(path,lineage)
            return path,report,audit
        except (OSError,ValueError,KeyError,RuntimeError):
            if explicit:
                raise
    raise RuntimeError("completed matching compare-features proof with all four final checkpoints unavailable")


def _measure_attainable(outputs:dict,batch:dict,config,*,with_loss=True)->dict:
    with torch.no_grad():
        result={"metrics":sample_metrics(outputs,batch,final=False)}
        result["gateChecks"]=_gate_snapshot(result,config)
        if with_loss:
            result.update({
                "lossTerms":{k:float(v) for k,v in detail_supervision_loss(outputs,batch).items()},
                "bandRelativeL1":band_relative_l1(outputs,batch),
                "rawDetail":tensor_statistics(outputs["fine_detail_raw"]),
                "highpassDetail":tensor_statistics(outputs["fine_detail_highpass"]),
            })
    return result


def _attainable_evidence(region:dict)->str:
    # Only the independent-channel relaxation is an upper bound, and only
    # for the two fixed target-weighted pixel-error recovery metrics.
    bound=region["rgbProjectionRelaxation"]["gateChecks"]
    if not bound["edge"] or not bound["global"]:
        return "PROJECTION_RANGE_LIMIT_ON_EDGE_OR_GLOBAL"
    if all(region["fits"]["stage10a"]["measurement"]["gateChecks"].values()):
        return "FEASIBLE_SAME_LOSS_FIT_NETWORK_LEARNING_UNRESOLVED"
    if all(region["fits"]["edge"]["measurement"]["gateChecks"].values()):
        return "FEASIBLE_EDGE_FIT_OBJECTIVE_OR_OPTIMIZATION_UNRESOLVED"
    return "UNRESOLVED_FINITE_BUDGET_FITS_DO_NOT_PROVE_A_CEILING"


def _run_attainable(folder,backbone,comparison_path,comparison,contract,
                    sampler,rows,config,device)->dict:
    """Read model checkpoints; optimize disposable training-pixel fields only."""
    results={}
    cap=config.albedo_residual_cap
    for i in HARD_TRAINING_INDICES:
        authority=_id(rows[i])
        fixed,alternates,origins=_samples(sampler,authority,device)
        trained={}
        for mode,model in zip(MODES,create_matched_variants(backbone,seed=SEED+i)):
            arm=_arm_name(i,mode)
            if comparison["armResults"][arm]["alternateProbeOrigins"]!=origins:
                raise RuntimeError("attainable and comparison spatial probe origins differ")
            ckpt=comparison_path.parent/arm/f"stage_{STAGES[-1]:06d}"/"checkpoint.pt"
            if sha256_file(ckpt)!=contract["comparisonCheckpointSha256"][arm]:
                raise RuntimeError("comparison checkpoint changed after provenance audit")
            state=torch.load(ckpt,map_location="cpu",weights_only=True)
            if state.get("architecture")!=model.architecture_contract():
                raise RuntimeError("comparison specialist architecture differs")
            model.specialist.load_state_dict(state["detailState"],strict=True)
            trained[mode]=model.to(device).eval()
        regions=[]
        probes=[("fixed",fixed,{"samplingPolicy":"fixed","exposure":0})]+[
            (f"alternate_{n:02d}",batch,{**meta,"exposure":exposure})
            for n,(batch,meta,exposure) in enumerate(zip(alternates,origins,ALT_EXPOSURES))]
        for probe,batch,origin in probes:
            path=folder/f"authority_{i:02d}"/f"{probe}.json"
            identity={"runContractSha256":sha256_json(contract),"authorityIndex":i,
                      "authorityId":authority,"cropId":contract["trainingAuthorities"][str(i)]["cropId"],
                      "probe":probe,"sourceRegion":origin}
            if path.is_file():
                cached=_read(path)
                if any(cached.get(k)!=v for k,v in identity.items()):
                    raise RuntimeError("attainable region resume identity differs")
                regions.append(cached)
                continue
            with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False), torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic,allow_tf32=False,
            ):
                original=backbone(batch["lr_albedo"],batch["lr_normal"],batch["lr_material"])
                zero=torch.zeros_like(original["baseline_albedo"][:,:1])
                control=scalar_candidate(original,zero,cap)
                if not torch.equal(control["candidate_albedo"],original["candidate_albedo"]):
                    raise RuntimeError("zero-field control changed the frozen candidate")
                region={**identity,"targetAssisted":True,"deployable":False,
                        "frozenCandidate":_measure_attainable(control,batch,config),
                        "trained":{},"fits":{}}
                relaxed={**original,"candidate_albedo":rgb_projection_relaxation(
                    original,batch["target_albedo"],cap)}
                region["rgbProjectionRelaxation"]=_measure_attainable(relaxed,batch,config,with_loss=False)
                for mode,model in trained.items():
                    outputs=model(batch["lr_albedo"],batch["lr_normal"],batch["lr_material"])
                    region["trained"][mode]=_measure_attainable(outputs,batch,config)
            edge_seed=None
            for objective in ATTAINABLE_POLICY["objectives"]:
                def progress(row):
                    print(f"[v176-attainable] authority={i} probe={probe} objective={objective} "
                          f"update={row['update']}/{ATTAINABLE_POLICY['stages'][-1]} "
                          f"best={row['bestObjective']:.6f}",flush=True)
                try:
                    outputs,fit=fit_scalar_field(original,batch,cap,objective,
                                                initial_raw=edge_seed,progress=progress)
                except NumericalStabilityError as exc:
                    exc.context.update({**identity,"objective":objective})
                    path.parent.mkdir(parents=True,exist_ok=True)
                    torch.save({"context":exc.context,"diagnostic":exc.diagnostic,
                                "tensors":exc.tensors},path.with_suffix(".failure.pt"))
                    raise
                fit["measurement"]=_measure_attainable(outputs,batch,config)
                fit["normalUnchanged"]=bool(torch.equal(outputs["candidate_normal"],original["candidate_normal"]))
                fit["materialUnchanged"]=bool(torch.equal(outputs["candidate_material"],original["candidate_material"]))
                if not fit["normalUnchanged"] or not fit["materialUnchanged"]:
                    raise RuntimeError("attainable fit altered normal or material")
                region["fits"][objective]=fit
                if objective=="edge":
                    edge_seed=outputs["fine_detail_raw"].detach()
            region["evidence"]=_attainable_evidence(region)
            _atomic_json(path,region)
            regions.append(region)
        results[str(i)]={"authorityId":authority,"regions":regions}
        del trained
    if _tensor_sha(backbone)!=contract["parentBackboneTensorSha256"]:
        raise RuntimeError("attainable probe changed the frozen backbone")
    return {"decision":"ATTAINABLE_RECOVERY_DIAGNOSTICS_TRAIN_ONLY",
            "attainableRecovery":results,"networkWeightsUpdated":False,
            "globalOptimalityProven":False}


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
    if getattr(args,"comparison_report","") and args.phase!="attainable":
        raise ValueError("--comparison-report is only used by --phase attainable")
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
        comparison_audit={}
        if args.phase=="attainable":
            comparison_path,comparison,comparison_audit=_comparison_source(
                repo,getattr(args,"comparison_report",""),contract)
            contract.update({"attainablePolicy":dict(ATTAINABLE_POLICY),"stages":[],
                             "specialistUpdates":0,
                             "comparisonReportSha256":comparison_audit["comparisonReportSha256"],
                             "comparisonCheckpointSha256":comparison_audit["comparisonCheckpointSha256"]})
        manifest_path=folder/"run_manifest.json"
        if args.resume=="auto":
            if _read(manifest_path)!=contract:
                raise RuntimeError("Stage10A.1 resume manifest/source selection changed")
        else:
            _atomic_json(manifest_path,contract)
            _atomic_json(folder/"provenance.json",{**stage9audit,**detail_audit,**comparison_audit})
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
            if args.phase=="attainable":
                report.update(_run_attainable(folder,backbone,comparison_path,comparison,
                                             contract,sampler,rows,config,device))
            elif args.phase=="diagnose":
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
        if not any(folder.glob("*/stage_*/checkpoint.pt")) and (
            args.phase!="attainable" or isinstance(exc,NumericalStabilityError)
        ):
            _atomic_json(folder/"report.json",error)
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=Path.cwd())
    p.add_argument("--stage10a-report",default="")
    p.add_argument("--comparison-report",default="",
                   help="completed compare-features report for --phase attainable; defaults to latest matching proof")
    p.add_argument("--v3-report",default="")
    p.add_argument("--phase",choices=PHASES,default="diagnose")
    p.add_argument("--resume",choices=("fresh","auto"),default="fresh")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--amp-precision",choices=("auto","bf16","fp16"),default="auto")
    p.add_argument("--audit-only",action="store_true")
    return p


if __name__=="__main__":
    run(parser().parse_args())
