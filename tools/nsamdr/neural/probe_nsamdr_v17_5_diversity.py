#!/usr/bin/env python3
"""V17.5 research-only matched fresh A12/B64 native-authority diversity proof.

Never opens the frozen eight-authority final-test pixels, changes V17.2 model
or loss, or promotes an output to the production pipeline.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for folder in (REPO_ROOT, HERE):
    if str(folder) not in sys.path:
        sys.path.insert(0,str(folder))

from tools.nsamdr.neural.v17.diversity_corpus import (
    SELECTION_SCHEMA, SELECTION_SEED, freeze_authorities, selected_rows,
    sha256_file, sha256_json, verify_frozen_selection,
)
from probe_nsamdr_v16_full_broad import _device, _to_device
from probe_nsamdr_v17_comprehensive import _origin, _records, _batch_config, _json
from probe_nsamdr_v17_generalization import _augment_batch
from probe_nsamdr_v17_generalization_v2 import (
    _evaluate_set, _validation_score, _promote_checkpoint, _sheets,
)
from probe_nsamdr_v17_sibling_architecture import (
    _batch_for_record, _train_one, _target_albedo_residual_magnitude,
)
from probe_nsamdr_v17_spatial_generalization import _model
from v17.spatial_sampling import NativeAuthoritySampler, _id, _crop_id

SCHEMA = "NSAMDR_V17_5_NATIVE_AUTHORITY_DIVERSITY_PROOF_V1"
CHECKPOINT_SCHEMA = "NSAMDR_V17_5_DIVERSITY_CHECKPOINT_V1"
ROOT = "artifacts/nsamdr/diagnostics/v17_5_diversity"
ARMS = ("A12","B64")
STAGES = (6144,12288,24576)
DEFAULT_SEED = 17301
LEARNING_RATE = .0002
ARCHITECTURE = {"encoderChannels":96,"decoderMidChannels":64,
                "decoderDetailChannels":48,"decoderBlocksPerStage":4}

def _atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w",encoding="utf-8") as handle:
        json.dump(obj,handle,indent=2,sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)

def _tensor_sha(model:torch.nn.Module)->str:
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        tensor=value.detach().cpu().contiguous()
        h.update(name.encode("utf-8")+b"\0")
        h.update(str(tensor.dtype).encode()+b"\0")
        h.update(json.dumps(list(tensor.shape)).encode()+b"\0")
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()

def _make_model(config,seed:int,device:torch.device):
    opts=argparse.Namespace(seed=seed,learning_rate=LEARNING_RATE,
                            encoder_channels=96,mid_channels=64,
                            detail_channels=48,decoder_blocks=4)
    return _model(config,opts,device)

def _assert_unchanged_architecture(model,source:dict)->None:
    original=source["v3Contract"].get("architecture")
    if original!=ARCHITECTURE:
        raise RuntimeError("V17.5 cannot change historical V17.3 model architecture")
    existing=model.architecture_contract()
    if (existing.get("encoderChannels")!=96 or existing.get("decoderMidChannels")!=64
        or existing.get("decoderDetailChannels")!=48
        or existing.get("decoderBlocksPerStage")!=4
        or existing.get("revision")!="V17.2-proof"):
        raise RuntimeError("V17.5 model architecture drift")

def _checkpoint_save(path:Path,model,opt,contract:dict,arm:str,
                     update:int,history:list,best:int,score:float,qualified:bool)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    data={"schema":CHECKPOINT_SCHEMA,"contract":contract,"arm":arm,"update":update,
          "history":history,"bestUpdate":best,"bestScore":score,
          "bestQualified":qualified,"architecture":model.architecture_contract(),
          "modelState":model.state_dict(),"optimizerState":opt.state_dict(),
          "torchRngState":torch.get_rng_state().cpu().contiguous(),
          "cudaRngState":[x.cpu().contiguous() for x in torch.cuda.get_rng_state_all()]
                          if torch.cuda.is_available() else None}
    temp=path.with_suffix(".tmp")
    torch.save(data,temp)
    temp.replace(path)

def _checkpoint_load(path:Path,model,opt,contract:dict,arm:str,
                     device:torch.device)->dict:
    state=torch.load(path,map_location="cpu",weights_only=False)
    if (not isinstance(state,dict) or state.get("schema")!=CHECKPOINT_SCHEMA
        or state.get("contract")!=contract or state.get("arm")!=arm
        or state.get("architecture")!=model.architecture_contract()
        or state.get("update") not in STAGES):
        raise RuntimeError("V17.5 checkpoint provenance/architecture/stage mismatch")
    model.load_state_dict(state["modelState"],strict=True)
    opt.load_state_dict(state["optimizerState"])
    torch.set_rng_state(state["torchRngState"].cpu().contiguous())
    if device.type=="cuda" and state.get("cudaRngState"):
        torch.cuda.set_rng_state_all([x.cpu().contiguous()
                                      for x in state["cudaRngState"]])
    return state

def _new_directory(repo:Path)->Path:
    path=repo/ROOT/("proof_"+datetime.now().strftime("%Y%m%d-%H%M%S"))
    path.mkdir(parents=True,exist_ok=False)
    return path

def _resume_directory(repo:Path)->Path:
    root=repo/ROOT
    found=[p for p in root.glob("proof_*")
           if (p/"run_manifest.json").is_file() and not (p/"report.json").is_file()]
    if len(found)!=1:
        raise RuntimeError(f"auto resume requires exactly one unfinished V17.5 proof, got {len(found)}")
    return found[0]

def _initial_report(source:dict,selection:dict,contract:dict,folder:Path)->dict:
    return {"schema":SCHEMA,"decision":"RUNNING","researchOnly":True,
            "productionPromotionAllowed":False,"qualificationGatesModified":False,
            "finalTestEvaluated":False,"finalTestEvaluationCount":0,
            "selectedFinalTest":None,"trainingUpdates":0,"armResults":{},
            "sourceV3Report":source["v3Report"],"runContract":contract,
            "frozenSelectionSha256":sha256_json(selection),
            "trainingSourcePolicy":selection["selectionPolicy"],
            "dataContract":"native 512 authored HR -> 128 LR clean 4x",
            "reportPath":str(folder/"report.json")}

def _training_suitability(config,repo:Path,device:torch.device)->Any:
    model,_optimizer=_make_model(config,17301,device)
    def measure(row:dict)->float:
        batch,_=_batch_for_record(row,config,seed=SELECTION_SEED+1,device=device)
        return _target_albedo_residual_magnitude(model,batch)
    return measure

def _start_source(repo:Path,explicit:str):
    source=_origin(repo,explicit)
    manifest,parts=_records(source)
    v2=_json(Path(source["v2Report"]))
    return source,manifest,parts,v2

def _preflight_selection(path:Path,source:dict,manifest:dict,parts:dict)->dict:
    selection=_json(path)
    verify_frozen_selection(selection,source,manifest,parts,verify_bytes=True)
    return selection

def _selection_for_run(args,repo,source,manifest,parts,v2,folder,config,device):
    path=folder/"frozen_selection.json"
    if args.resume=="auto":
        if not path.is_file():
            raise RuntimeError("incomplete proof has no frozen selection; cannot rerank on resume")
        selection=_preflight_selection(path,source,manifest,parts)
        if args.selection and sha256_file(Path(args.selection).resolve())!=sha256_file(path):
            raise RuntimeError("operator selection differs from unfinished proof selection")
        return selection
    if args.selection:
        if args.seed==17301 or args.seed==17302:
            selection=_preflight_selection(Path(args.selection).resolve(),source,manifest,parts)
        else:
            raise RuntimeError("non-standard training seed is outside preregistered study")
        audit={"status":"REUSED","sourceSelection":str(Path(args.selection).resolve()),
               "developmentPixelsOpened":0,"sealedPixelsOpened":0}
    else:
        if args.seed!=17301:
            raise RuntimeError("replication requires --selection from original seed-17301 proof")
        selection,audit=freeze_authorities(
            repo,source,manifest,parts,v2,
            suitability=_training_suitability(config,repo,device))
    verify_frozen_selection(selection,source,manifest,parts)
    _atomic_json(path,selection)
    _atomic_json(folder/"selection_audit.json",audit)
    return selection

def _train_arm(arm:str,folder:Path,contract:dict,selection:dict,
               repo:Path,manifest:dict,parts:dict,source:dict,
               config,device:torch.device,precision:str)->dict:
    arm_root=folder/arm
    completed=arm_root/"arm_report.json"
    if completed.is_file():
        result=_json(completed)
        if result.get("runContractSha256")!=sha256_json(contract) or result.get("arm")!=arm:
            raise RuntimeError(f"completed {arm} report contract mismatch")
        return result
    rows=selected_rows(selection,manifest,arm)
    train_ids=[_id(row) for row in rows]
    protected=set(selection["protection"]["v2FamilyIds"]
                  +selection["protection"]["v16ValidationFamilyIds"]
                  +selection["protection"]["frozenDevFamilyIds"]
                  +selection["protection"]["frozenFinalFamilyIds"])
    if protected.intersection(train_ids):
        raise RuntimeError(f"protected source selected by {arm}")
    sampler=NativeAuthoritySampler(repo,manifest,rows,hr_size=512,seed=SELECTION_SEED)
    try:
        if set(sampler.audit)!=set(train_ids):
            raise RuntimeError("native sampler exposed unexpected authority")
        fixed=[_to_device(sampler.sample(fid,0,mode="fixed")[0],device) for fid in train_ids]
        # The only held-out pixel reads: exactly 4 frozen development crops.
        development=[
            _batch_for_record(r,config,seed=SELECTION_SEED+1001+i,device=device)[0]
            for i,r in enumerate(parts["validation"])]
        model,opt=_make_model(config,contract["trainingSeed"],device)
        _assert_unchanged_architecture(model,source)
        init_sha=_tensor_sha(model)
        if init_sha!=contract["initialModelSha256"]:
            raise RuntimeError("matched fresh initialization mismatch")
        files=sorted(arm_root.glob("stage_*/checkpoint.pt"))
        if files:
            latest=files[-1]
            state=_checkpoint_load(latest,model,opt,contract,arm,device)
            start=int(state["update"])
            history=list(state["history"])
            best=int(state["bestUpdate"])
            best_score=float(state["bestScore"])
            qualified=bool(state["bestQualified"])
        else:
            start,best,best_score,qualified=0,0,float("-inf"),False
            history=[]
        begin=time.monotonic()
        for update in range(start+1,STAGES[-1]+1):
            idx=(update-1)%len(train_ids)
            exposure=(update-1)//len(train_ids)
            raw,_sample=sampler.sample(train_ids[idx],exposure,mode="spatial")
            batch=_augment_batch(_to_device(raw,device),exposure%8)
            _train_one(model,opt,batch,config,device=device,precision=precision)
            if update==start+1 or update%256==0:
                elapsed=max(time.monotonic()-begin,1e-6)
                progress=(update-start)/max(1,STAGES[-1]-start)
                print(f"[v17-diversity] arm={arm} update {update}/{STAGES[-1]} "
                      f"elapsed={elapsed/60:.1f}m eta={(1-progress)*elapsed/max(progress,1e-6)/60:.1f}m",
                      flush=True)
            if update not in STAGES:
                continue
            train=_evaluate_set(model,fixed,config,device=device,precision=precision,sibling=False)
            validation=_evaluate_set(model,development,config,device=device,precision=precision,sibling=True)
            score=_validation_score(validation)
            valid=bool(train["allPass"] and validation["allPass"])
            improved=_promote_checkpoint(
                qualified=valid,previous_qualified=qualified,score=score,best_score=best_score,
                min_improvement=.002,no_best=(best==0))
            if improved:
                best,best_score,qualified=update,score,valid
            history.append({"update":update,"trained":train,"validation":validation,
                            "validationScore":score,"newBest":improved,"bestUpdate":best,
                            "exposuresPerAuthority":update//len(train_ids)})
            _checkpoint_save(arm_root/f"stage_{update:06d}"/"checkpoint.pt",
                             model,opt,contract,arm,update,history,best,best_score,qualified)
            print(f"[v17-diversity] arm={arm} stage={update} trainPass={train['allPass']} "
                  f"devPass={validation['allPass']} score={score:.6f} best={best}",flush=True)
        if len(history)!=len(STAGES):
            raise RuntimeError(f"{arm} checkpoints incomplete")
        selected_path=arm_root/f"stage_{best:06d}"/"checkpoint.pt"
        _checkpoint_load(selected_path,model,opt,contract,arm,device)
        selected=next(row for row in history if row["update"]==best)
        previews=_sheets(arm_root/f"selected_{best:06d}",model,development,
                         selection["developmentCropIds"],config,
                         device=device,precision=precision,role="validation")
        result={"arm":arm,"runContractSha256":sha256_json(contract),
                "initialModelSha256":init_sha,"bestValidationUpdate":best,
                "bestValidationScore":best_score,"bestQualified":qualified,
                "selectedCheckpoint":str(selected_path.resolve()),
                "selectedTrained":selected["trained"],"selectedValidation":selected["validation"],
                "trainingAuthorities":train_ids,"trainingProbeType":"fixed native 512 each",
                "history":history,"validationPreviews":previews,
                "optimizerUpdates":STAGES[-1],"imagesPerUpdate":1,
                "exposuresPerAuthority":STAGES[-1]//len(train_ids),
                "samplerAudit":sampler.audit,"elapsedSeconds":time.monotonic()-begin}
        _atomic_json(completed,result)
        return result
    finally:
        sampler.close()

def _metric(result:dict,name:str)->list[float]:
    return [float(item["metrics"][name]) for item in result["perCrop"]]

def _compare(results:dict)->list[dict]:
    out=[]
    a,b=results["A12"],results["B64"]
    for update in STAGES:
        x=next(h for h in a["history"] if h["update"]==update)
        y=next(h for h in b["history"] if h["update"]==update)
        xa,ya=_metric(x["validation"],"edge_recovery"),_metric(y["validation"],"edge_recovery")
        ga,gb=_metric(x["validation"],"global_recovery"),_metric(y["validation"],"global_recovery")
        la,lb=_metric(x["validation"],"lattice_cell_excess"),_metric(y["validation"],"lattice_cell_excess")
        out.append({"update":update,"A12ExposuresPerAuthority":update//12,
                    "B64ExposuresPerAuthority":update//64,
                    "A12TrainPass":x["trained"]["allPass"],"B64TrainPass":y["trained"]["allPass"],
                    "A12DevelopmentPass":x["validation"]["allPass"],
                    "B64DevelopmentPass":y["validation"]["allPass"],
                    "A12EdgeByCrop":xa,"B64EdgeByCrop":ya,
                    "edgeDeltaByCrop":[v-u for u,v in zip(xa,ya)],
                    "A12WorstEdge":min(xa),"B64WorstEdge":min(ya),
                    "A12MeanEdge":sum(xa)/4,"B64MeanEdge":sum(ya)/4,
                    "A12WorstGlobal":min(ga),"B64WorstGlobal":min(gb),
                    "A12MaxLattice":max(la),"B64MaxLattice":max(lb)})
    return out

def _decision(results:dict)->tuple[str,bool]:
    a,b=(results[x] for x in ARMS)
    if b["selectedTrained"]["allPass"] and b["selectedValidation"]["allPass"]:
        return "VALIDATION_QUALIFIED_RESEARCH_ONLY",False
    av,bv=a["selectedValidation"],b["selectedValidation"]
    ae,be=_metric(av,"edge_recovery"),_metric(bv,"edge_recovery")
    ag,bg=_metric(av,"global_recovery"),_metric(bv,"global_recovery")
    lattice=_metric(bv,"lattice_cell_excess")
    if (min(be)-min(ae)>=.03 and (sum(be)-sum(ae))/4>=.02
        and all(x<=.15 for x in lattice)
        and all(y-x>=-.01 for x,y in zip(ag,bg))):
        return "DIVERSITY_BENEFIT_SIGNAL",False
    ta,tb=a["selectedTrained"]["mean"]["edge_recovery"],b["selectedTrained"]["mean"]["edge_recovery"]
    return "NO_DEMONSTRATED_BENEFIT",bool(tb<ta-.05)

def _markdown(report:dict)->str:
    lines=["# NSAMDR V17.5 Native Authority Diversity A12 versus B64",
           "",f"Decision: **{report['decision']}**","",
           "Research-only, identical V17.2 architecture and original training loss.",
           "No final-test pixels accessed; no native production promotion.","",
           "| Updates | A12 worst edge | B64 worst edge | A12 mean edge | B64 mean edge |",
           "|---:|---:|---:|---:|---:|"]
    for row in report.get("matchedComparison",[]):
        lines.append(f"| {row['update']} | {row['A12WorstEdge']:.3f} | "
                     f"{row['B64WorstEdge']:.3f} | {row['A12MeanEdge']:.3f} | "
                     f"{row['B64MeanEdge']:.3f} |")
    lines+=["","Required gates unchanged; per-crop results and selected A/B/C images are in report.json."]
    return "\n".join(lines)+"\n"

def run(args:argparse.Namespace)->tuple[int,Path]:
    if args.seed not in (17301,17302):
        raise ValueError("V17.5 permits only preregistered seeds 17301 or 17302")
    repo=args.repo_root.resolve()
    source,manifest,parts,v2=_start_source(repo,args.v3_report)
    config=_batch_config(source)
    device=_device("cpu" if args.selection_only else args.device)
    folder=_new_directory(repo) if args.resume=="fresh" else _resume_directory(repo)
    try:
        selection=_selection_for_run(args,repo,source,manifest,parts,v2,folder,config,device)
        model,_opt=_make_model(config,args.seed,device)
        _assert_unchanged_architecture(model,source)
        initial_sha=_tensor_sha(model)
        del model,_opt
        contract={"schema":SCHEMA,"sourceV3Sha256":source["v3ReportSha256"],
                  "sourceManifestSha256":selection["manifestSha256"],
                  "v2ReportSha256":selection["v2ReportSha256"],
                  "selectionSha256":sha256_json(selection),
                  "trainingSeed":args.seed,"selectionSeed":SELECTION_SEED,
                  "initialModelSha256":initial_sha,
                  "architecture":ARCHITECTURE,"stages":list(STAGES),
                  "learningRate":LEARNING_RATE,
                  "scale":4,"trainHR":512,"trainLR":128,
                  "sampling":"V17.3 native-spatial 50/50 uniform/detail",
                  "augmentation":"D4 turns and mirror",
                  "loss":"V17.2 unchanged","minimumValidationImprovement":.002,
                  "device":str(device),"ampPrecision":args.amp_precision,
                  "sealedFinalEnabled":False}
        manifest_path=folder/"run_manifest.json"
        if args.resume=="auto":
            if _json(manifest_path)!=contract:
                raise RuntimeError("unfinished V17.5 contract mismatch; resume refused")
        else:
            _atomic_json(manifest_path,contract)
        report=_initial_report(source,selection,contract,folder)
        _atomic_json(folder/"source_compatibility.json",
                     {"sourceManifestSha256":selection["manifestSha256"],
                      "sourceV3ReportSha256":source["v3ReportSha256"],
                      "frozenSelectionSha256":sha256_json(selection),
                      "verified":True,"sealedFinalPixelsOpened":0})
        if args.selection_only:
            report["decision"]="SELECTION_VERIFIED_RESEARCH_ONLY"
            report["selectionOnly"]=True
            _atomic_json(folder/"report.json",report)
            (folder/"report.md").write_text(_markdown(report),encoding="utf-8")
            print(f"[v17-diversity] selection verified A12=12 B64=64 report={folder/'report.json'}",flush=True)
            return 0,folder/"report.json"
        results={arm:_train_arm(arm,folder,contract,selection,repo,manifest,parts,
                                source,config,device,args.amp_precision) for arm in ARMS}
        report["armResults"]=results
        report["trainingUpdates"]=len(ARMS)*STAGES[-1]
        report["matchedComparison"]=_compare(results)
        report["decision"],report["underexposurePossible"]=_decision(results)
        _atomic_json(folder/"report.json",report)
        (folder/"report.md").write_text(_markdown(report),encoding="utf-8")
        print(f"[v17-diversity] decision={report['decision']} finalTestOpened=False "
              f"report={folder/'report.json'}",flush=True)
        return 0,folder/"report.json"
    except BaseException as exc:
        # An interrupted run remains resumable; no misleading completed proof.
        _atomic_json(folder/"failure.json",{"schema":SCHEMA,"decision":"BLOCKED",
                     "reason":f"{type(exc).__name__}: {exc}",
                     "traceback":traceback.format_exc(),
                     "finalTestEvaluated":False,"productionPromotionAllowed":False})
        print(f"[v17-diversity] BLOCKED {type(exc).__name__}: {exc} folder={folder}",flush=True)
        raise

def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=Path.cwd())
    p.add_argument("--v3-report",default="")
    p.add_argument("--device",choices=("auto","cpu","cuda"),default="cuda")
    p.add_argument("--amp-precision",choices=("auto","bf16","fp16"),default="auto")
    p.add_argument("--resume",choices=("fresh","auto"),default="fresh")
    p.add_argument("--seed",type=int,choices=(17301,17302),default=17301)
    p.add_argument("--selection-only",action="store_true")
    p.add_argument("--selection",default="")
    return p

if __name__=="__main__":
    run(parser().parse_args())
