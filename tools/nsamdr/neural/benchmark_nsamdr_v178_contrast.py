#!/usr/bin/env python3
"""Fixed, unaugmented contrast-transfer probes; never training or qualification.

The registered V4 observation generator is frozen here so a later training
augmentation cannot silently change the original/inverted comparison cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from tools.nsamdr.neural.v177.synthetic_geometry import make_batch
from tools.nsamdr.neural.v178.contour_model import (
    SCHEMA, StructuralContourHead, contour_supervised_loss,
    foreground_metrics, physical_edge_evidence,
)
from tools.nsamdr.neural.v178.golden_patch import sha256, validate_native_evidence
from tools.nsamdr.neural.train_nsamdr_v178_contour import TRAIN_SCHEMA, source_hash

REPORT_SCHEMA="NSAMDR_V178_FIXED_CONTRAST_DIAGNOSTIC_V1"
CASE_GROUPS={
    "reported_failure":tuple(17822+1000000+1723*i for i in range(12)),
    "independent_holdout_a":tuple(17832+2000000+1723*i for i in range(12)),
    "independent_holdout_b":tuple(17832+3000000+1723*i for i in range(12)),
}


def registered_batch(seed:int, device:torch.device, *, patch:int=32):
    a,n,_,_,_,target,_=make_batch(1,patch,seed=seed)
    a,n=a.to(device),n.to(device)
    edges=physical_edge_evidence(a,n)
    generator=torch.Generator(device="cpu").manual_seed(int(seed)^0x5A17832)
    coarse=torch.rand((1,2,4,4),generator=generator)
    auxiliary=F.interpolate(coarse,size=(patch,patch),mode="bilinear",
                            align_corners=False)
    level=torch.rand((1,2,1,1),generator=generator)
    auxiliary=(level+.25*(auxiliary-.5)).clamp(0,1)
    absent=torch.rand((1,2,1,1),generator=generator)<.5
    defaults=torch.tensor([0.,.5]).view(1,2,1,1)
    observation=torch.cat((edges,torch.where(absent,defaults,auxiliary).to(device)),dim=1)
    return a,n,observation,target[:,:1].to(device)


def authenticate(checkpoint:Path, repo:Path)->tuple[dict,dict]:
    report_path=checkpoint.parent/"report.json"
    report=json.loads(report_path.read_text(encoding="utf-8"))
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    current_source=source_hash(repo)
    if (state.get("schema")!=SCHEMA or report.get("modelSchema")!=SCHEMA
        or report.get("schema")!=TRAIN_SCHEMA
        or state.get("config",{}).get("sourceHash")!=current_source
        or report.get("sourceHash")!=current_source
        or report.get("config")!=state.get("config")
        or report.get("checkpointSha256")!=sha256(checkpoint)
        or bool(state.get("syntheticPass"))!=(
            report.get("decision")=="SYNTHETIC_CONTOUR_PASS_ONLY")):
        raise RuntimeError("contrast benchmark requires an authenticated current-source checkpoint/report")
    updates=int(state.get("steps",0))
    if updates<1:
        raise RuntimeError("contrast benchmark requires a completed checkpoint")
    trained={int(state["config"]["seed"])+8191*i for i in range(updates)}
    registered={seed for group in CASE_GROUPS.values() for seed in group}
    if trained & registered:
        raise RuntimeError("registered contrast holdout overlaps a training seed")
    return state,report


def _pixels(value:torch.Tensor)->Image.Image:
    array=value.detach().cpu()[0].clamp(0,1).numpy()
    if array.shape[0]==1:
        array=np.repeat(array,3,axis=0)
    return Image.fromarray(np.uint8(np.rint(array.transpose(1,2,0)*255)))


def _score(value:torch.Tensor,target:torch.Tensor)->dict:
    loss,parts=contour_supervised_loss(value,target)
    return {"contourLoss":float(loss),**foreground_metrics(value,target),
            "components":parts}


def _tensor_digest(value:torch.Tensor)->str:
    array=value.detach().cpu().contiguous().numpy()
    h=hashlib.sha256()
    h.update(str((array.shape,str(array.dtype))).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def _mean(rows:list[dict],key:str)->dict:
    scores=[row[key] for row in rows]
    result={name:sum(score[name] for score in scores)/len(scores)
            for name in scores[0] if name!="components"}
    result["components"]={name:sum(score["components"][name] for score in scores)/len(scores)
                          for name in scores[0]["components"]}
    return result


def native_sources(evidence:dict)->dict:
    """G0 crop validation alone does not authenticate its original source maps."""
    sources={}
    for name,asset in evidence["assets"].items():
        if "source" not in asset:
            continue
        source=Path(asset["source"])
        if not source.is_file() or sha256(source)!=asset.get("sourceSha256"):
            raise RuntimeError(f"registered native source map changed or missing: {source}")
        sources[name]={"path":str(source.resolve()),"sha256":asset["sourceSha256"]}
    return sources


@torch.no_grad()
def synthetic(model:StructuralContourHead,device:torch.device,output:Path)->dict:
    result={}
    for group,seeds in CASE_GROUPS.items():
        rows=[]
        cell=128
        sheet=Image.new("RGB",(6*cell,len(seeds)*(cell+24)),(20,23,28))
        draw=ImageDraw.Draw(sheet)
        for index,seed in enumerate(seeds):
            a,n,m,target=registered_batch(seed,device)
            inverted=1.-a
            delta=float((physical_edge_evidence(inverted,n)-m[:,:1]).abs().max())
            original=model(a,n,m)
            changed=model(inverted,n,m)
            base=original["baseline_contour"]
            pred=original["candidate_contour"]
            inv=changed["candidate_contour"]
            rows.append({"seed":seed,"edgeMagnitudeMaxDifference":delta,
                         "inputTensorSha256":{
                             name:_tensor_digest(value) for name,value in
                             (("albedo",a),("normal",n),("observation",m),("target",target))},
                         "baseline":_score(base,target),
                         "original":_score(pred,target),
                         "inverted":_score(inv,target)})
            panels=(("RGB",a),("1-RGB",inverted),("analytic target",target),
                    ("observation",base),("original prediction",pred),
                    ("inverted prediction",inv))
            y=index*(cell+24)
            for column,(label,value) in enumerate(panels):
                draw.text((column*cell+3,y+5),label,fill=(240,240,240))
                sheet.paste(_pixels(value).resize((cell,cell),Image.Resampling.NEAREST),
                            (column*cell,y+24))
        sheet.save(output/(group+".png"))
        result[group]={"caseCount":len(seeds),"cases":rows,
                       "baseline":_mean(rows,"baseline"),
                       "original":_mean(rows,"original"),
                       "inverted":_mean(rows,"inverted"),
                       "edgeMagnitudeMaxDifference":max(row["edgeMagnitudeMaxDifference"] for row in rows)}
        print(f"[v178-contrast] {group} original_recall={result[group]['original']['contourRecall']:.6f} "
              f"inverted_recall={result[group]['inverted']['contourRecall']:.6f}",flush=True)
    return result


@torch.no_grad()
def native(model:StructuralContourHead,device:torch.device,folder:Path,output:Path)->dict:
    from tools.nsamdr.neural.v178.auto_preview import _comparison, _predict_tiled
    from tools.nsamdr.neural.v178.real_fit import _input
    import cv2

    evidence=validate_native_evidence(folder)
    sources=native_sources(evidence)
    a,n,m,alignment=_input(folder,evidence,device)
    m=torch.cat((physical_edge_evidence(a,n),m[:,1:]),dim=1)
    baseline,tiled=_predict_tiled(model,a,n,m)
    dense=model(a,n,m)["candidate_contour"].cpu()
    difference=(dense-tiled).abs()
    _pixels(tiled).save(output/"native_candidate_contour.png")
    _pixels(baseline).save(output/"native_baseline_observation.png")
    image=cv2.imread(str(folder/"native_albedo.png"),cv2.IMREAD_UNCHANGED)
    _comparison(output,image,baseline[0,0].numpy(),tiled[0,0].numpy())
    np.savez_compressed(output/"native_float_maps.npz",
                        candidate=tiled[0,0].numpy(),baseline=baseline[0,0].numpy())
    return {
        "folder":str(folder),"evidenceSha256":sha256(folder/"evidence.json"),
        "sourceSha256":evidence["identity"]["sourceSha256"],
        "sourceMaps":sources,
        "nativeDimensions":list(a.shape[-2:][::-1]),
        "outputDimensions":list(tiled.shape[-2:][::-1]),
        "alignment":alignment,"albedoMean":float(a.mean()),
        "candidateMean":float(tiled.mean()),
        "candidateActiveFraction":float((tiled>=.30).float().mean()),
        "baselineMean":float(baseline.mean()),
        "denseTiledMaxDifference":float(difference.max()),
        "denseTiledMeanDifference":float(difference.mean()),
        "realEveAccuracyProven":False,
    }


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=ROOT)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--native-folder",type=Path,action="append",default=[])
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    return p


def main(argv:list[str]|None=None)->int:
    args=parser().parse_args(argv)
    repo=args.repo_root.resolve()
    checkpoint=args.checkpoint.resolve()
    output=args.output.resolve()
    state,training=authenticate(checkpoint,repo)
    if args.native_folder and not state.get("syntheticPass"):
        raise RuntimeError("failed synthetic gate blocks native Raven inference")
    folders=[folder.resolve() for folder in args.native_folder]
    if len({folder.name for folder in folders})!=len(folders):
        raise ValueError("native evidence folder names must be unique within a contrast benchmark")
    for folder in folders:
        native_sources(validate_native_evidence(folder))
    device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.set_num_threads(min(4,torch.get_num_threads()))
    model=StructuralContourHead().to(device)
    model.load_state_dict(state["model"],strict=True)
    model.eval()
    output.mkdir(parents=True,exist_ok=False)
    report={
        "schema":REPORT_SCHEMA,"modelSchema":SCHEMA,
        "checkpoint":str(checkpoint),"checkpointSha256":sha256(checkpoint),
        "trainingReportSha256":sha256(checkpoint.parent/"report.json"),
        "trainingSourceHash":state["config"]["sourceHash"],
        "benchmarkSourceSha256":sha256(Path(__file__)),
        "trainingDecision":training["decision"],"device":str(device),
        "torchVersion":str(torch.__version__),"patchSize":32,
        "geometryNormalsObservationsTargetsPreserved":True,
        "registeredCaseSeedsAbsentFromTraining":True,
        "realEveAccuracyProven":False,"productionPromotionAllowed":False,
        "synthetic":synthetic(model,device,output),
    }
    report["native"]={}
    for folder in folders:
        dest=output/"native"/folder.name
        dest.mkdir(parents=True,exist_ok=False)
        report["native"][folder.name]=native(model,device,folder,dest)
        print(f"[v178-contrast] native={folder.name} "
              f"mean={report['native'][folder.name]['candidateMean']:.6f} "
              f"dense_tiled_max={report['native'][folder.name]['denseTiledMaxDifference']:.9f}",flush=True)
    report["assets"]={str(path.relative_to(output).as_posix()):sha256(path)
                      for path in sorted(output.rglob("*")) if path.is_file()}
    (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"[v178-contrast] diagnostic only report={output/'report.json'}",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
