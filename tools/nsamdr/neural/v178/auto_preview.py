"""Automatically infer native-to-4x Raven structural contours: NO traced target.

This is a falsifiable visual experiment, not a clean-label generator.
Synthetic-only training precedes evaluation. Physical channels retain their
authored native values. Predictions are not promoted to the Raven shader.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from tools.nsamdr.neural.v178.contour_model import (
    SCHEMA, StructuralContourHead, physical_edge_evidence,
)
from tools.nsamdr.neural.v178.golden_patch import (
    sha256, validate_native_evidence,
)
from tools.nsamdr.neural.v178.real_fit import _input
from tools.nsamdr.neural.train_nsamdr_v178_contour import source_hash

REPORT_SCHEMA = "NSAMDR_V178_AUTOMATIC_REAL_PATCH_PREVIEW_V1"


def _predict_tiled(
    model: StructuralContourHead, albedo: torch.Tensor,
    normal: torch.Tensor, material: torch.Tensor, *,
    tile: int = 96, halo: int = 20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense-equivalent overlap inference. Never globally resize native input."""
    h,w=albedo.shape[-2:]
    if min(h,w)<16 or tile<16 or halo<12:
        raise ValueError("contour inference requires valid native patches")
    baseline=torch.empty((1,1,h*4,w*4),device="cpu")
    candidate=torch.empty_like(baseline)
    for y in range(0,h,tile):
        y1=min(h,y+tile)
        sy=max(0,y-halo); ey=min(h,y1+halo)
        for x in range(0,w,tile):
            x1=min(w,x+tile)
            sx=max(0,x-halo); ex=min(w,x1+halo)
            with torch.no_grad():
                out=model(albedo[...,sy:ey,sx:ex],
                          normal[...,sy:ey,sx:ex],
                          material[...,sy:ey,sx:ex])
            cut=(slice((y-sy)*4,(y1-sy)*4),
                 slice((x-sx)*4,(x1-sx)*4))
            region=(slice(y*4,y1*4),slice(x*4,x1*4))
            baseline[...,region[0],region[1]] = (
                out["baseline_contour"][...,cut[0],cut[1]].detach().cpu()
            )
            candidate[...,region[0],region[1]] = (
                out["candidate_contour"][...,cut[0],cut[1]].detach().cpu()
            )
    return baseline, candidate


def _overlay(image: np.ndarray, mask: np.ndarray, colour: tuple[int,int,int]) -> np.ndarray:
    weight=np.clip(mask[...,None],0,1)*.70
    rgb=np.array(colour,dtype=np.float32).reshape(1,1,3)
    return np.uint8(np.clip(np.rint(image.astype(np.float32)*(1-weight)+rgb*weight),0,255))


def _comparison(folder: Path, rgba: np.ndarray, base: np.ndarray,
                inferred: np.ndarray) -> Path:
    size=rgba.shape[0]
    rgb=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_BGR2RGB)
    authored=cv2.resize(rgb,(size*4,size*4),interpolation=cv2.INTER_NEAREST)
    filtered=cv2.resize(rgb,(size*4,size*4),interpolation=cv2.INTER_CUBIC)
    difference=np.abs(inferred-base)
    diff=np.uint8(np.clip(difference*255*3,0,255))
    panels=[
        ("A  NATIVE RAVEN (4X NEAREST)",authored),
        ("B  BICUBIC COLOUR (NOT TARGET)",filtered),
        ("C  AUTO CONTOURS - UNQUALIFIED",_overlay(filtered,inferred,(0,224,255))),
        ("D  CHANGE FROM EDGE OBSERVATION",cv2.applyColorMap(diff,cv2.COLORMAP_TURBO)[:,:,::-1]),
    ]
    gutter=30
    canvas=Image.new("RGB",(size*8,size*8+gutter*2),(20,23,28))
    label=ImageDraw.Draw(canvas)
    for index,(title,pixels) in enumerate(panels):
        col=index%2; row=index//2
        ox=col*size*4; oy=row*(size*4+gutter)
        label.text((ox+8,oy+9),title,fill=(240,240,240))
        canvas.paste(Image.fromarray(pixels),(ox,oy+gutter))
    path=folder/"automatic_contour_comparison.png"
    canvas.save(path)
    return path


def run(folder: Path, checkpoint: Path, *, device_name: str="cuda",
        tile: int=96) -> dict:
    folder=folder.expanduser().resolve()
    checkpoint=checkpoint.expanduser().resolve()
    evidence=validate_native_evidence(folder)
    repo=Path(__file__).resolve().parents[4]
    device=torch.device(device_name)
    if device.type=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Train a new V2 synthetic contour checkpoint first: {checkpoint}")
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    if (state.get("schema")!=SCHEMA
        or state.get("config",{}).get("sourceHash")!=source_hash(repo)):
        raise RuntimeError("checkpoint is from the old V1 material-ID contour model or stale training sources; retrain G2 V2")
    if not state.get("syntheticPass"):
        raise RuntimeError("synthetic contour proof failed its holdout gate; automatic Raven inference blocked")
    model=StructuralContourHead().to(device)
    model.load_state_dict(state["model"],strict=True)
    model.eval()
    albedo,normal,material,provenance=_input(folder,evidence,device)
    observation=physical_edge_evidence(albedo,normal)
    material=torch.cat((observation,material[:,1:]),dim=1)
    baseline,candidate=_predict_tiled(model,albedo,normal,material,tile=tile)
    b=baseline[0,0].numpy()
    c=candidate[0,0].numpy()
    native=observation[0,0].detach().cpu().numpy()
    image=cv2.imread(str(folder/"native_albedo.png"),cv2.IMREAD_UNCHANGED)
    dest=folder/"automatic_contour_preview"
    dest.mkdir(parents=True,exist_ok=True)
    maps={
        "native_edge_observation.png": np.rint(np.clip(native*255,0,255)).astype(np.uint8),
        "baseline_observation_4x.png": np.rint(np.clip(b*255,0,255)).astype(np.uint8),
        "candidate_contour_4x.png": np.rint(np.clip(c*255,0,255)).astype(np.uint8),
    }
    for name,pixels in maps.items():
        if not cv2.imwrite(str(dest/name),pixels):
            raise RuntimeError(f"could not write {name}")
    sheet=_comparison(dest,image,b,c)
    report={
        "schema":REPORT_SCHEMA,
        "decision":"REAL_EVE_UNQUALIFIED_AUTOMATIC_CONTOUR_PREVIEW",
        "nativeEvidenceVerified":True,
        "humanContourTracingRequired":False,
        "referenceAvailable":False,
        "referenceSynthesizedFromInput":False,
        "realEveAccuracyProven":False,
        "productionPromotionAllowed":False,
        "sourceSha256":evidence["identity"]["sourceSha256"],
        "syntheticCheckpoint":str(checkpoint),
        "syntheticCheckpointSha256":sha256(checkpoint),
        "modelSchema":SCHEMA,
        "alignment":provenance,
        "materialIdUsedAsContourInput":False,
        "nativeDimensions":list(image.shape[1::-1]),
        "outputDimensions":[image.shape[1]*4,image.shape[0]*4],
        "observationMean":float(np.mean(native)),
        "candidateMean":float(np.mean(c)),
        "candidateDifferenceFromObservation":float(np.mean(np.abs(c-b))),
        "visualComparison":str(sheet),
        "candidateMask":str(dest/"candidate_contour_4x.png"),
        "notes":"C is an inference hypothesis, not verified geometry. D is a diagnostic difference, not error against a real clean target.",
    }
    (dest/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"[v178-auto] {report['decision']} comparison={sheet} "
          f"changed={report['candidateDifferenceFromObservation']:.6f}",flush=True)
    return report
