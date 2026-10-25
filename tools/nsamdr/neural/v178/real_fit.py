"""G2 real-EVE single Golden Patch structural *capacity* fit.

This is intentional supervised overfitting to an independently traced native
Raven patch. The target is training-only. Neither the result nor its metrics
establish improvement on unseen real EVE textures or production suitability.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .contour_model import SCHEMA, StructuralContourHead, contour_loss
from .golden_patch import sha256, validate_trace, render_contour_trace

SCHEMA_FIT = "NSAMDR_V178_G2_REAL_PATCH_FIT_V1"


def _native_map(
    folder: Path, name: str, size: int, channel: int,
    default: float, signed: bool = False, categorical: bool = False,
) -> tuple[np.ndarray, dict]:
    path = folder / name
    if not path.is_file():
        return np.full((size,size),default,dtype=np.float32), {
            "source":None,"alignedFrom":None,"policy":"missing-explicit-default",
        }
    image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] != 4:
        raise RuntimeError(f"invalid native physical-map crop: {path}")
    h,w=image.shape[:2]
    physical=(2,1,0,3)[int(channel)]
    pixel=image[:,:,physical].astype(np.float32)/255.
    if (w,h)!=(size,size):
        interp=cv2.INTER_NEAREST if categorical else (
            cv2.INTER_AREA if w>size or h>size else cv2.INTER_LINEAR
        )
        pixel=cv2.resize(pixel,(size,size),interpolation=interp)
        policy="categorical-nearest-feature-alignment" if categorical else "continuous-UV-feature-alignment"
    else:
        policy="native-direct"
    value=pixel*2-1 if signed else pixel
    return value.astype(np.float32), {
        "source":str(path.resolve()),"alignedFrom":[w,h],
        "policy":policy,"sourceSha256":sha256(path),
    }


def _input(folder:Path, evidence:dict, device:torch.device) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor,dict]:
    size=int(evidence["identity"]["nativeRoi"][2])
    photo=cv2.imread(str(folder/"native_albedo.png"),cv2.IMREAD_UNCHANGED)
    if photo is None or photo.shape!=(size,size,4):
        raise RuntimeError("frozen native albedo missing or dimensions changed")
    a=cv2.cvtColor(photo[:,:,:3],cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    layout=evidence["identity"]["channelLayout"]
    x,px=_native_map(folder,"native_normal.png",size,layout["normalX"],0.,signed=True)
    y,py=_native_map(folder,"native_normal.png",size,layout["normalY"],0.,signed=True)
    # On V5 the first material channel is an ID, not a real contour.
    # This report MUST retain the domain mismatch: that baseline is not a
    # legitimate geometrical reference and is never used as qualification.
    selector,pm=_native_map(folder,"native_material.png",size,
                            layout["material"],0.,categorical=True)
    glow,pg=_native_map(folder,"native_glow.png",size,layout["glow"],0.)
    roughness,pr=_native_map(folder,"native_roughness.png",size,
                             layout["roughness"],.5)
    normal=np.stack((x,y),axis=2)
    material=np.stack((selector,glow,roughness),axis=2)
    to_tensor=lambda data:torch.from_numpy(
        np.ascontiguousarray(data.transpose(2,0,1))
    ).unsqueeze(0).to(device=device,dtype=torch.float32)
    return to_tensor(a),to_tensor(normal),to_tensor(material),{
        "normalX":px,"normalY":py,"material":pm,
        "glow":pg,"roughness":pr,
        "semanticBaselineIsValidContour":False,
    }


def _quality(prediction: np.ndarray, target: np.ndarray,
             protected: np.ndarray) -> dict[str,float]:
    if prediction.shape!=target.shape or protected.shape!=target.shape:
        raise ValueError("real Golden Patch dimension mismatch")
    error=float(np.mean(np.abs(prediction-target)))
    reference=target>=.30
    found=prediction>=.30
    intersection=int((reference&found).sum())
    union=int((reference|found).sum())
    iou=intersection/max(1,union)
    protected_mask=protected>=.9
    false_rate=float(np.mean(found[protected_mask])) if protected_mask.any() else 0.
    hit_rate=intersection/max(1,int(reference.sum()))
    return {
        "meanAbsoluteCoverageError":error,
        "contourIoU":iou,
        "contourHitRate":hit_rate,
        "protectedFalsePositiveRate":false_rate,
    }


def _sheet(folder:Path, original:np.ndarray, baseline:np.ndarray,
           result:np.ndarray, reference:np.ndarray) ->Path:
    from PIL import Image,ImageDraw,ImageFont
    h,w=original.shape[:2]
    original_rgb=cv2.cvtColor(original[:,:,:3],cv2.COLOR_BGR2RGB)
    authored=cv2.resize(original_rgb,(w*4,h*4),interpolation=cv2.INTER_NEAREST)
    bicubic=cv2.resize(original_rgb,(w*4,h*4),interpolation=cv2.INTER_CUBIC)
    def overlay(rgb:np.ndarray,weight:np.ndarray,colour:tuple[int,int,int])->np.ndarray:
        alpha=np.clip(weight[...,None],0,1)
        color=np.array(colour,dtype=np.float32).reshape(1,1,3)
        return np.uint8(np.rint(
            rgb.astype(np.float32)*(1-.65*alpha)+color*(.65*alpha)
        ).clip(0,255))
    panels=[
        ("A - NATIVE PIXELS 4x (INSPECTION)",authored),
        ("B - INTERPOLATED COLOUR REFERENCE",bicubic),
        ("C - FITTED CONTOUR FIELD (OVERLAY)",overlay(bicubic,result,(25,245,220))),
        ("T - INDEPENDENTLY TRACED CONTOURS",overlay(bicubic,reference,(245,208,35))),
    ]
    header=34
    canvas=Image.new("RGB",(w*8, h*8+header*2),(21,24,29))
    draw=ImageDraw.Draw(canvas)
    for index,(title,panel) in enumerate(panels):
        px=(index%2)*w*4
        py=(index//2)*(h*4+header)
        draw.text((px+8,py+10),title,fill=(255,255,255))
        canvas.paste(Image.fromarray(panel),(px,py+header))
    output=folder/"golden_real_fit_comparison.png"
    canvas.save(output)
    return output


def fit(folder:Path, *, device_name:str="cuda",steps:int=300,
        patch_size:int=48,seed:int=17803,resume:str="fresh")->dict:
    if not 1<=steps<=5000 or not 16<=patch_size<=128:
        raise ValueError("invalid bounded real-patch capacity experiment")
    if resume not in ("fresh","auto"):
        raise ValueError("invalid resume")
    folder=folder.resolve()
    evidence,annotation=validate_trace(folder)
    target_file=render_contour_trace(folder)
    device=torch.device(device_name)
    if device_name=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for G2 real Golden Patch fit")
    if device_name=="cpu":
        torch.set_num_threads(min(4,torch.get_num_threads()))
    albedo,normal,material,feature_provenance=_input(folder,evidence,device)
    size=albedo.shape[-1]
    if patch_size>size:
        raise ValueError("training patch exceeds native Golden Patch size")
    target=cv2.imread(str(target_file),cv2.IMREAD_GRAYSCALE)
    protected=cv2.imread(str(folder/"protected_regions_4x.png"),cv2.IMREAD_GRAYSCALE)
    if target is None or protected is None or target.shape!=(size*4,size*4):
        raise RuntimeError("independent clean 4x reference is missing or invalid")
    target_t=torch.from_numpy(target.astype(np.float32)[None,None]/255).to(device)
    protected_t=torch.from_numpy(protected.astype(np.float32)[None,None]/255).to(device)

    source_id=evidence["identity"]["sourceSha256"]
    traced_id=sha256(folder/"trace.json")
    sample_id=hashlib.sha256((source_id+traced_id).encode()).hexdigest()[:12]
    run=folder/"fit_runs"/f"trace_{sample_id}_seed_{seed}_p{patch_size}"
    run.mkdir(parents=True,exist_ok=True)
    resume_file=run/"resume.pt"
    torch.manual_seed(seed)
    if device_name=="cuda":
        torch.cuda.manual_seed_all(seed)
    model=StructuralContourHead().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0007)
    config={
        "sourceSha256":source_id,"traceSha256":traced_id,
        "seed":seed,"patchSize":patch_size,"architectureSchema":SCHEMA,
    }
    start=0
    if resume=="fresh" and resume_file.exists():
        raise RuntimeError("Golden Patch fit already exists; use --resume auto")
    if resume=="auto" and resume_file.is_file():
        saved=torch.load(resume_file,map_location="cpu",weights_only=True)
        if saved.get("schema")!=SCHEMA_FIT or saved.get("config")!=config:
            raise RuntimeError("real patch fit resume provenance mismatch")
        model.load_state_dict(saved["model"],strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for name,value in state.items():
                if torch.is_tensor(value):
                    state[name]=value.to(device)
        start=int(saved["step"])
        if start>steps:
            raise RuntimeError("fit resume step exceeds requested horizon")
    # Every positive trace is actively sampled, with negative patches included.
    ys,xs=np.where(target[::4,::4]>.2*255)
    if len(xs)==0:
        raise RuntimeError("curated contours contain no occupied supervision pixels")
    rng=np.random.default_rng(seed)
    begin=time.monotonic()
    for index in range(start,steps):
        if rng.random()<.80:
            choice=int(rng.integers(0,len(xs)))
            px,py=int(xs[choice]),int(ys[choice])
        else:
            px=int(rng.integers(0,size)); py=int(rng.integers(0,size))
        left=max(0,min(size-patch_size,px-patch_size//2))
        top=max(0,min(size-patch_size,py-patch_size//2))
        slices=(slice(top,top+patch_size),slice(left,left+patch_size))
        rs=(slice(top*4,(top+patch_size)*4),
            slice(left*4,(left+patch_size)*4))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        outputs=model(albedo[...,slices[0],slices[1]],
                      normal[...,slices[0],slices[1]],
                      material[...,slices[0],slices[1]])
        reference=target_t[...,rs[0],rs[1]]
        protected_region=protected_t[...,rs[0],rs[1]]
        prediction=outputs["candidate_contour"]
        geometric,parts=contour_loss(prediction,reference)
        positive=(reference>.10).float()
        positive_weight=1+5*positive
        coverage=(positive_weight*torch.abs(prediction-reference)).mean()
        false_structure=(prediction*protected_region).mean()
        loss=geometric+coverage+.50*false_structure
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite single-patch contour fit loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        step=index+1
        if step==1 or step%25==0 or step==steps:
            print(f"[v178-real-fit] update={step}/{steps} "
                  f"loss={float(loss):.6f} coverage={float(coverage):.6f}",
                  flush=True)
        if step%50==0 or step==steps:
            tmp=run/"resume.tmp"
            torch.save({"schema":SCHEMA_FIT,"config":config,"step":step,
                        "model":model.state_dict(),"optimizer":optimizer.state_dict()},tmp)
            os.replace(tmp,resume_file)
    model.eval()
    with torch.no_grad():
        output=model(albedo,normal,material)
    prediction=output["candidate_contour"][0,0].detach().float().cpu().numpy()
    baseline=output["baseline_contour"][0,0].detach().float().cpu().numpy()
    reference=target.astype(np.float32)/255.
    protected_np=protected.astype(np.float32)/255.
    metrics=_quality(prediction,reference,protected_np)
    # Capacity proof is strictly same-patch supervised overfit, not inference
    # on withheld EVE observations; no real-world generalisation is claimed.
    passed=(metrics["contourIoU"]>=.65
            and metrics["contourHitRate"]>=.8
            and metrics["protectedFalsePositiveRate"]<=.02)
    checkpoint=run/"candidate.pt"
    tmp=run/"candidate.tmp"
    torch.save({
        "schema":SCHEMA_FIT,"config":config,"model":model.state_dict(),
        "singlePatchCapacityPass":bool(passed),
        "generalisationVerified":False,"productionPromotionAllowed":False,
    },tmp)
    os.replace(tmp,checkpoint)
    original=cv2.imread(str(folder/"native_albedo.png"),cv2.IMREAD_UNCHANGED)
    sheet=_sheet(run,original,baseline,prediction,reference)
    report={
        "schema":SCHEMA_FIT,
        "decision":"SINGLE_PATCH_CAPACITY_PASS_ONLY" if passed else "REAL_PATCH_FIT_INSUFFICIENT",
        "trainingOnly":True,"targetUsedInTraining":True,
        "targetUsedAtInference":False,
        "nativeEvidencePreserved":True,
        "sourceSha256":source_id,"traceSha256":traced_id,
        "config":config,"trainingUpdates":steps-start,
        "metrics":metrics,"thresholds":{
            "contourIoU":.65,"contourHitRate":.80,
            "maxProtectedFalsePositiveRate":.02,
        },
        "checkpoint":str(checkpoint.resolve()),
        "checkpointSha256":sha256(checkpoint),
        "comparisonSheet":str(sheet.resolve()),
        "featureProvenance":feature_provenance,
        "semanticDomainWarning":"V5 selector is categorical and not a contour reference",
        "realEveGeneralisationVerified":False,
        "sealedTestAccess":False,"productionPromotionAllowed":False,
        "elapsedSeconds":time.monotonic()-begin,
    }
    (run/"report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(f"[v178-real-fit] {report['decision']} "
          f"IoU={metrics['contourIoU']:.3f} "
          f"sheet={sheet}",flush=True)
    return report
