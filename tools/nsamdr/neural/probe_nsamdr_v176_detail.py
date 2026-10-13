#!/usr/bin/env python3
"""V17.6 Stage 10A: frozen Stage 9 A12 + trainable grayscale detail specialist.

Isolated, research-only. Never writes Stage 9 checkpoints, opens sealed final
pixels, changes V17.2 loss/model, or promotes a production texture.
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

from probe_nsamdr_v16_full_broad import _device, _to_device
from probe_nsamdr_v17_comprehensive import _batch_config, _origin, _records
from probe_nsamdr_v17_generalization import _augment_batch
from probe_nsamdr_v17_generalization_v2 import (
    _evaluate_set, _promote_checkpoint, _sheets, _validation_score,
)
from probe_nsamdr_v17_sibling_architecture import _batch_for_record
from probe_nsamdr_v17_5_diversity import _atomic_json, _make_model, _tensor_sha
from tools.nsamdr.neural.v17.diversity_corpus import (
    selected_rows, sha256_file, sha256_json,
)
from tools.nsamdr.neural.v17.diversity_continuation import verify_stage8
from tools.nsamdr.neural.v17.spatial_sampling import NativeAuthoritySampler, _id
from tools.nsamdr.neural.v176.fine_detail import NSAMDRV176Detail
from tools.nsamdr.neural.v176.wavelet_supervision import (
    LOSS_WEIGHTS, detail_supervision_loss,
)

SCHEMA="NSAMDR_V176_STAGE10A_DETAIL_SPECIALIST_PROOF_V1"
CHECKPOINT_SCHEMA="NSAMDR_V176_STAGE10A_CHECKPOINT_V1"
PARENT_SCHEMA="NSAMDR_V17_5_MATCHED_CONTINUATION_PROOF_V1"
PARENT_CHECKPOINT_SCHEMA="NSAMDR_V17_5_MATCHED_CONTINUATION_CHECKPOINT_V1"
ROOT="artifacts/nsamdr/diagnostics/v176_stage10a_detail"
STAGES=(0,1536,3072,6144)
SELECTION_SEED=17301
DETAIL_SEED=17610
LR=.0002


def _read(path:Path)->dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parent_report(repo:Path, explicit:str)->Path:
    if explicit:
        path=Path(explicit).resolve()
        if not path.is_file():
            raise RuntimeError(f"Stage 9 proof report missing: {path}")
        return path
    root=repo/"artifacts/nsamdr/diagnostics/v17_5_matched_continuation"
    paths=[]
    for path in root.glob("proof_*/report.json"):
        try:
            report=_read(path)
        except (OSError,ValueError):
            continue
        if (report.get("schema")==PARENT_SCHEMA and
            report.get("additionalOptimizerUpdates")==49152 and
            set(report.get("armResults") or {})=={"A12","B64"} and
            report.get("finalTestEvaluated") is False):
            paths.append(path)
    if not paths:
        raise RuntimeError("completed Stage 9 A12/B64 parent not found")
    return max(paths,key=lambda p:p.stat().st_mtime).resolve()


def _run_folder(repo:Path,resume:str)->Path:
    root=repo/ROOT
    if resume=="fresh":
        path=root/("proof_"+datetime.now().strftime("%Y%m%d-%H%M%S"))
        path.mkdir(parents=True,exist_ok=False)
        return path
    unfinished=[p for p in root.glob("proof_*")
                if (p/"run_manifest.json").is_file()
                and not (p/"report.json").is_file()]
    if len(unfinished)!=1:
        raise RuntimeError(f"Stage 10A resume requires one unfinished proof; got {len(unfinished)}")
    return unfinished[0]


def verify_parent(
    path:Path, source:dict, manifest:dict, parts:dict, backbone:torch.nn.Module,
)->tuple[dict,dict]:
    """Metadata + authentic Stage 9 A12 model, zero final test exposure.

    Reads all authenticated parent state on CPU; no authority or validation
    texture pixels are decoded at this point.
    """
    report=_read(path)
    if (report.get("schema")!=PARENT_SCHEMA or
        report.get("researchOnly") is not True or
        report.get("finalTestEvaluated") is not False or
        report.get("finalTestEvaluationCount")!=0 or
        report.get("productionPromotionAllowed") is not False or
        report.get("qualificationGatesModified") is not False or
        report.get("additionalOptimizerUpdates")!=49152 or
        set(report.get("armResults") or {})!={"A12","B64"}):
        raise RuntimeError("Stage 10A requires completed, sealed Stage 9 A12/B64 research proof")
    root=path.parent
    parent_manifest=_read(root/"run_manifest.json")
    if (report.get("runContract")!=parent_manifest or
        parent_manifest.get("schema")!=PARENT_SCHEMA or
        parent_manifest.get("trainedSelectionSha256")!=report.get("trainedSelectionSha256") or
        parent_manifest.get("stages")!=[36864,49152] or
        parent_manifest.get("trainingSeed")!=17301 or
        parent_manifest.get("sealedFinalEnabled") is not False):
        raise RuntimeError("Stage 9 run manifest/frozen selection/seed changed")
    stage8_path=Path(report["stage8SourceReport"]).resolve()
    selection,stage8_contract,audit8=verify_stage8(
        stage8_path,source,manifest,parts
    )
    if (audit8["trainedSelectionSha256"]!=parent_manifest["trainedSelectionSha256"] or
        audit8["stage8ReportSha256"]!=parent_manifest["parentStage8ReportSha256"] or
        audit8["stage8RunManifestSha256"]!=parent_manifest["parentStage8ManifestSha256"] or
        parent_manifest["sourceManifestSha256"]!=sha256_file(Path(source["manifest"]))):
        raise RuntimeError("Stage 10A original V17.3/V17.5 source provenance mismatch")
    trained=[_id(r) for r in selected_rows(selection,manifest,"A12")]
    if trained!=report["armResults"]["A12"]["trainingAuthorities"]:
        raise RuntimeError("Stage 10A trained A12 authority roster changed")
    checkpoint=root/"A12"/"stage_049152"/"checkpoint.pt"
    if not checkpoint.is_file():
        raise RuntimeError("Stage 9 final A12 checkpoint missing")
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (not isinstance(state,dict) or
        state.get("schema")!=PARENT_CHECKPOINT_SCHEMA or
        state.get("contract")!=parent_manifest or
        state.get("architecture")!=backbone.architecture_contract() or
        state.get("arm")!="A12" or state.get("update")!=49152 or
        [item.get("update") for item in state.get("history",[])]!=[36864,49152] or
        not isinstance(state.get("modelState"),dict) or
        not isinstance(state.get("optimizerState"),dict) or
        not isinstance(state.get("torchRngState"),torch.Tensor) or
        not isinstance(state.get("cudaRngState"),list)):
        raise RuntimeError("Stage 9 A12 final model/checkpoint identity invalid")
    if report["armResults"]["A12"].get("atFinalUpdate",{}).get("update")!=49152:
        raise RuntimeError("Stage 9 A12 numerical evidence not final-update matched")
    backbone.load_state_dict(state["modelState"],strict=True)
    backbone.eval()
    signature=_tensor_sha(backbone)
    audit={
        "stage9Report":str(path.resolve()),
        "stage9ReportSha256":sha256_file(path),
        "stage9ManifestSha256":sha256_file(root/"run_manifest.json"),
        "stage9A12Checkpoint":str(checkpoint.resolve()),
        "stage9A12CheckpointSha256":sha256_file(checkpoint),
        "stage9BackboneTensorSha256":signature,
        "stage8SelectionSha256":audit8["trainedSelectionSha256"],
        "trainingAuthorityIds":trained,
        "developmentCropIds":selection["developmentCropIds"],
        "sealedFinalCropIds":selection["sealedFinalCropIds"],
        "sourceManifestSha256":sha256_file(Path(source["manifest"])),
        "architecture":backbone.architecture_contract(),
        "finalTestPixelsOpened":0,
    }
    return selection,audit


def _save(path:Path,model:NSAMDRV176Detail,opt,contract:dict,
          update:int,history:list,best:int,best_score:float,best_qualified:bool)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    payload={
        "schema":CHECKPOINT_SCHEMA,"contract":contract,"update":update,
        "history":history,"bestUpdate":best,"bestScore":best_score,
        "bestQualified":best_qualified,
        "architecture":model.architecture_contract(),
        "detailState":model.specialist.state_dict(),
        "optimizerState":opt.state_dict(),
        "torchRngState":torch.get_rng_state().cpu().contiguous(),
        "cudaRngState":[x.cpu().contiguous() for x in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_available() else None,
    }
    temp=path.with_suffix(".tmp")
    torch.save(payload,temp)
    temp.replace(path)


def _restore(path:Path,model:NSAMDRV176Detail,opt,contract:dict,
             device:torch.device)->dict:
    state=torch.load(path,map_location="cpu",weights_only=True)
    if (not isinstance(state,dict) or
        state.get("schema")!=CHECKPOINT_SCHEMA or
        state.get("contract")!=contract or
        state.get("architecture")!=model.architecture_contract() or
        state.get("update") not in STAGES or
        [x.get("update") for x in state.get("history",[])]!=
        [x for x in STAGES if x<=state["update"]]):
        raise RuntimeError("Stage 10A specialist checkpoint/resume identity changed")
    model.specialist.load_state_dict(state["detailState"],strict=True)
    opt.load_state_dict(state["optimizerState"])
    torch.set_rng_state(state["torchRngState"].cpu().contiguous())
    if device.type=="cuda" and state.get("cudaRngState") is not None:
        torch.cuda.set_rng_state_all([x.cpu().contiguous() for x in state["cudaRngState"]])
    return state


def _snapshot(model:NSAMDRV176Detail,training:list,dev:list,config,
              device:torch.device,amp:str,update:int)->dict:
    trained=_evaluate_set(model,training,config,device=device,precision=amp,sibling=False)
    validation=_evaluate_set(model,dev,config,device=device,precision=amp,sibling=True)
    return {"update":update,"trained":trained,"validation":validation,
            "validationScore":_validation_score(validation),
            "qualified":bool(trained["allPass"] and validation["allPass"])}


def _train_one(model:NSAMDRV176Detail,opt,batch,device:torch.device,precision:str):
    # Reuse the existing mixed-precision policy; only specialist gradients exist.
    from probe_nsamdr_v16_full_broad import _autocast
    model.train()
    opt.zero_grad(set_to_none=True)
    with _autocast(device,precision):
        result=model(batch["lr_albedo"],batch["lr_normal"],batch["lr_material"])
        terms=detail_supervision_loss(result,batch)
    if not bool(torch.isfinite(terms["total"]).item()):
        raise RuntimeError("non-finite wavelet specialist training objective")
    terms["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),1.0)
    opt.step()
    return {key:float(t.detach().cpu()) for key,t in terms.items()}


def _report_markdown(report:dict)->str:
    lines=["# V17.6 Stage 10A fine-detail specialist",
           "",f"Decision: **{report['decision']}**",
           "","Frozen authenticated Stage 9 A12 checkpoint; new grayscale head only.",
           "Sealed final-test authorities never evaluated. Production promotion prohibited.",
           "","| Specialist update | Train mean edge | Dev mean edge | Worst dev edge | Max dev lattice |",
           "|---:|---:|---:|---:|---:|"]
    for row in report.get("history",[]):
        tr=row["trained"]["mean"];dv=row["validation"]["mean"]
        worst=min(float(x["metrics"]["edge_recovery"]) for x in row["validation"]["perCrop"])
        mx=max(float(x["metrics"]["lattice_cell_excess"]) for x in row["validation"]["perCrop"])
        lines.append(f"| {row['update']} | {tr['edge_recovery']:.4f} | {dv['edge_recovery']:.4f} | "
                     f"{worst:.4f} | {mx:.4f} |")
    lines.append("")
    return "\n".join(lines)+"\n"


def run(args:argparse.Namespace)->tuple[int,Path]:
    repo=args.repo_root.resolve()
    folder=_run_folder(repo,args.resume)
    try:
        parent_path=_parent_report(repo,args.stage9_report)
        source=_origin(repo,args.v3_report)
        manifest,parts=_records(source)
        config=_batch_config(source)
        device=_device("cpu" if args.audit_only else "cuda")
        backbone,_unused=_make_model(config,17301,device)
        selection,audit=verify_parent(parent_path,source,manifest,parts,backbone)
        # Fresh specialist initialization, independent of the frozen parent's
        # historical model-init RNG or training optimizer.
        torch.manual_seed(DETAIL_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(DETAIL_SEED)
        model=NSAMDRV176Detail(backbone,detail_channels=32).to(device)
        contract={
            "schema":SCHEMA,"parentStage9ReportSha256":audit["stage9ReportSha256"],
            "parentStage9ManifestSha256":audit["stage9ManifestSha256"],
            "parentA12CheckpointSha256":audit["stage9A12CheckpointSha256"],
            "parentBackboneTensorSha256":audit["stage9BackboneTensorSha256"],
            "sourceManifestSha256":audit["sourceManifestSha256"],
            "frozenSelectionSha256":audit["stage8SelectionSha256"],
            "trainingIds":audit["trainingAuthorityIds"],
            "developmentCropIds":audit["developmentCropIds"],
            "sealedFinalCropIds":audit["sealedFinalCropIds"],
            "detailSeed":DETAIL_SEED,
            "learningRate":LR,"detailArchitecture":model.architecture_contract(),
            "originalBackboneFrozen":True,
            "checkpointUpdates":list(STAGES),
            "lossWeights":LOSS_WEIGHTS,"device":"cuda","ampPrecision":args.amp_precision,
            "dataSampler":"original A12 native spatial, unchanged",
            "finalTestEvaluated":False,
        }
        manifest_path=folder/"run_manifest.json"
        if args.resume=="auto":
            if _read(manifest_path)!=contract or _read(folder/"provenance.json")!=audit:
                raise RuntimeError("Stage 10A resume source/contract differs")
        else:
            _atomic_json(folder/"provenance.json",audit)
            _atomic_json(manifest_path,contract)
        report={"schema":SCHEMA,"decision":"RUNNING","researchOnly":True,
                "productionPromotionAllowed":False,"finalTestEvaluated":False,
                "finalTestEvaluationCount":0,"qualificationGatesModified":False,
                "runContract":contract,"stage9SourceReport":str(parent_path),
                "history":[]}
        if args.audit_only:
            report["auditOnly"]=True
            report["decision"]="AUTHENTICATED_PARENT_RESEARCH_ONLY"
            _atomic_json(folder/"report.json",report)
            (folder/"report.md").write_text(_report_markdown(report),encoding="utf-8")
            return 0,folder/"report.json"
        rows=selected_rows(selection,manifest,"A12")
        sampler=NativeAuthoritySampler(repo,manifest,rows,hr_size=512,seed=SELECTION_SEED)
        try:
            authority_ids=[_id(x) for x in rows]
            fixed=[_to_device(sampler.sample(fid,0,mode="fixed")[0],device)
                   for fid in authority_ids]
            development=[_batch_for_record(row,config,seed=SELECTION_SEED+1001+i,device=device)[0]
                         for i,row in enumerate(parts["validation"])]
            opt=torch.optim.Adam(model.trainable_parameters(),lr=LR)
            checkpoints=sorted(folder.glob("stage_*/checkpoint.pt"))
            if checkpoints:
                last=_restore(checkpoints[-1],model,opt,contract,device)
                start=int(last["update"]);history=list(last["history"])
                best=int(last["bestUpdate"])
                best_score=float(last["bestScore"])
                qualified=bool(last["bestQualified"])
            else:
                start=0
                with torch.no_grad():
                    original=model.backbone(
                        fixed[0]["lr_albedo"],fixed[0]["lr_normal"],fixed[0]["lr_material"])
                    fresh=model(
                        fixed[0]["lr_albedo"],fixed[0]["lr_normal"],fixed[0]["lr_material"])
                    if not all(torch.equal(original[key],fresh[key])
                               for key in ("candidate_albedo","candidate_normal","candidate_material")):
                        raise RuntimeError("Stage 10A zero-init changes original C0")
                history=[_snapshot(model,fixed,development,config,device,args.amp_precision,0)]
                best=0;best_score=history[0]["validationScore"]
                qualified=history[0]["qualified"]
                history[0]["newBest"]=True
                _atomic_json(folder/"baseline_metrics.json",history[0])
                _save(folder/"stage_000000"/"checkpoint.pt",
                      model,opt,contract,0,history,best,best_score,qualified)
            began=time.monotonic()
            for update in range(start+1,STAGES[-1]+1):
                idx=(update-1)%len(authority_ids)
                exposure=(update-1)//len(authority_ids)
                raw,_=sampler.sample(authority_ids[idx],exposure,mode="spatial")
                batch=_augment_batch(_to_device(raw,device),exposure%8)
                _train_one(model,opt,batch,device,args.amp_precision)
                if update==start+1 or update%256==0:
                    print(f"[v176-detail] update {update}/{STAGES[-1]} "
                          f"elapsed={(time.monotonic()-began)/60:.1f}m",flush=True)
                if update not in STAGES:
                    continue
                sample=_snapshot(model,fixed,development,config,device,args.amp_precision,update)
                promoted=_promote_checkpoint(
                    qualified=sample["qualified"],previous_qualified=qualified,
                    score=sample["validationScore"],best_score=best_score,
                    min_improvement=.002,no_best=False)
                if promoted:
                    best=update;best_score=sample["validationScore"]
                    qualified=sample["qualified"]
                sample["newBest"]=bool(promoted)
                sample["bestUpdate"]=best
                history.append(sample)
                if _tensor_sha(model.backbone)!=audit["stage9BackboneTensorSha256"]:
                    raise RuntimeError("original Stage 9 backbone changed during detail training")
                _save(folder/f"stage_{update:06d}"/"checkpoint.pt",
                      model,opt,contract,update,history,best,best_score,qualified)
            if [x["update"] for x in history]!=list(STAGES):
                raise RuntimeError("Stage 10A stage history incomplete")
            latest=history[-1]
            # Never promote based on mean-only improvements; gate at every crop.
            report["decision"]=(
                "VALIDATION_QUALIFIED_RESEARCH_ONLY" if
                latest["qualified"] else "UNQUALIFIED_DETAIL_RESEARCH"
            )
            report["history"]=history
            report["bestValidationUpdate"]=best
            report["bestValidationScore"]=best_score
            report["finalUpdate"]=STAGES[-1]
            report["specialistUpdates"]=STAGES[-1]
            report["frozenBackboneAfterSha256"]=_tensor_sha(model.backbone)
            if best != STAGES[-1]:
                _restore(folder/f"stage_{best:06d}"/"checkpoint.pt",model,opt,contract,device)
            report["validationPreviews"]=_sheets(
                folder/f"selected_{best:06d}",model,development,
                selection["developmentCropIds"],config,
                device=device,precision=args.amp_precision,role="validation")
            _atomic_json(folder/"report.json",report)
            (folder/"report.md").write_text(_report_markdown(report),encoding="utf-8")
            print(f"[v176-detail] {report['decision']} {folder/'report.json'}",flush=True)
            return 0,folder/"report.json"
        finally:
            sampler.close()
    except BaseException as exc:
        error={"schema":SCHEMA,"decision":"BLOCKED",
               "reason":f"{type(exc).__name__}: {exc}",
               "traceback":traceback.format_exc(),
               "researchOnly":True,"finalTestEvaluated":False,
               "productionPromotionAllowed":False}
        _atomic_json(folder/"failure.json",error)
        if not any(folder.glob("stage_*/checkpoint.pt")):
            _atomic_json(folder/"report.json",error)
        raise


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=Path.cwd())
    p.add_argument("--stage9-report",default="")
    p.add_argument("--v3-report",default="")
    p.add_argument("--audit-only",action="store_true")
    p.add_argument("--resume",choices=("fresh","auto"),default="fresh")
    p.add_argument("--amp-precision",choices=("auto","bf16","fp16"),default="auto")
    return p


if __name__=="__main__":
    run(parser().parse_args())
