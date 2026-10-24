#!/usr/bin/env python3
"""Bounded V17.7 synthetic native->clean-4x learning experiment.

No sealed EVE holdout access. No alteration of historical V17.6 artifacts.
Outputs are explicitly synthetic-only, unqualified for real-ship promotion.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from tools.nsamdr.neural.v177.enhancement_model import NativeEnhancer4x, SCHEMA
from tools.nsamdr.neural.v177.synthetic_geometry import make_batch

EVIDENCE_SCHEMA = "NSAMDR_V177_SYNTHETIC_ENHANCER_REPORT_V1"
DEFAULT_ROOT = ROOT / "artifacts/nsamdr/diagnostics/v177_native_enhancement"
SOURCE_FILES = (
    "tools/nsamdr/neural/v177/semantic_baseline.py",
    "tools/nsamdr/neural/v177/enhancement_model.py",
    "tools/nsamdr/neural/v177/synthetic_geometry.py",
    "tools/nsamdr/neural/train_nsamdr_v177_enhancement.py",
)


def source_digest(repo: Path = ROOT) -> str:
    h = hashlib.sha256()
    for name in SOURCE_FILES:
        path = repo / name
        h.update(name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def _loss(
    prediction: dict[str,torch.Tensor],
    reference: tuple[torch.Tensor,torch.Tensor,torch.Tensor],
    preserved: torch.Tensor,
) -> tuple[torch.Tensor,dict[str,float]]:
    a,n,m = reference
    rgb = F.l1_loss(prediction["candidate_albedo"],a)
    normal = F.l1_loss(prediction["candidate_normal"],n)
    material = F.l1_loss(prediction["candidate_material"],m)
    p=prediction["candidate_albedo"]
    def gradients(v: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        return v[:,:,:,1:]-v[:,:,:,:-1],v[:,:,1:,:]-v[:,:,:-1,:]
    dx,dy=gradients(p)
    gx,gy=gradients(a)
    edges= .5*(F.l1_loss(dx,gx)+F.l1_loss(dy,gy))
    protected = (p-a).abs().mul(preserved).mean()
    loss = rgb + .45*normal + .35*material + .4*edges + .15*protected
    return loss,{
        "rgb":float(rgb.detach()),"normal":float(normal.detach()),
        "material":float(material.detach()),"edges":float(edges.detach()),
        "protected":float(protected.detach()),
    }


def _score(model: NativeEnhancer4x, device: torch.device, *,seed:int,patch:int) -> dict:
    model.eval()
    baseline_sum=0.0
    candidate_sum=0.0
    parts={}
    with torch.no_grad():
        for case in range(8):
            tensors=[v.to(device) for v in make_batch(1,patch,seed=seed+1000000+case*1009)]
            a,n,m,ta,tn,tm,protected=tensors
            pred=model(a,n,m,material_policy="legacy_mask")
            base={
                "candidate_albedo":pred["baseline_albedo"],
                "candidate_normal":pred["baseline_normal"],
                "candidate_material":pred["baseline_material"],
            }
            b, _ = _loss(base,(ta,tn,tm),protected)
            q, detail = _loss(pred,(ta,tn,tm),protected)
            baseline_sum+=float(b)
            candidate_sum+=float(q)
            for k,v in detail.items():
                parts[k]=parts.get(k,0.0)+v/8
    return {
        "baselineLoss":baseline_sum/8, "candidateLoss":candidate_sum/8,
        "relativeRecovery":1.0-candidate_sum/max(1e-12,baseline_sum),
        "candidateLossComponents":parts,
        "syntheticHoldoutSeed":seed+1000000,
    }


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=ROOT)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--steps",type=int,default=300)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--patch-size",type=int,default=32)
    p.add_argument("--seed",type=int,default=177)
    p.add_argument("--learning-rate",type=float,default=.001)
    p.add_argument("--resume",choices=("fresh","auto"),default="fresh")
    return p


def main(argv:list[str]|None=None)->int:
    args=parser().parse_args(argv)
    if not (1<=args.steps<=20000 and 1<=args.batch_size<=16 and
            16<=args.patch_size<=128 and 0<args.learning_rate<=.01):
        raise ValueError("out-of-bounds V17.7 experimental training configuration")
    repo=args.repo_root.resolve()
    if not (repo/"tools/nsamdr/neural/v177/enhancement_model.py").is_file():
        raise ValueError("invalid repository root")
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the requested V17.7 device")
    device=torch.device(args.device)
    if device.type=="cpu":
        torch.set_num_threads(min(4,torch.get_num_threads()))
    digest=source_digest(repo)
    run=repo/"artifacts/nsamdr/diagnostics/v177_native_enhancement"/(
        f"seed_{args.seed}_patch_{args.patch_size}_batch_{args.batch_size}"
    )
    run.mkdir(parents=True,exist_ok=True)
    latest=run/"resume.pt"
    torch.manual_seed(args.seed)
    if device.type=="cuda":
        torch.cuda.manual_seed_all(args.seed)
    model=NativeEnhancer4x().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,
                                weight_decay=0.00001)
    config={
        "seed":args.seed, "patchSize":args.patch_size,"batchSize":args.batch_size,
        "learningRate":args.learning_rate,"sourceSha256":digest,
    }
    start=0
    if args.resume=="auto" and latest.is_file():
        state=torch.load(latest,map_location="cpu",weights_only=True)
        if state.get("schema")!=SCHEMA or state.get("config")!=config:
            raise RuntimeError("V17.7 resume identity mismatch; refusing to cross experiments")
        model.load_state_dict(state["model"],strict=True)
        optimizer.load_state_dict(state["optimizer"])
        # CUDA must own the Adam optimizer's loaded momenta.
        for values in optimizer.state.values():
            for key,value in values.items():
                if torch.is_tensor(value):
                    values[key]=value.to(device)
        start=int(state["step"])
        if start>args.steps:
            raise RuntimeError("resume step exceeds requested training horizon")
    elif args.resume=="fresh":
        if latest.exists():
            # Do not overwrite a previous scientific run accidentally.
            raise RuntimeError(
                f"existing V17.7 run at {run}; use --resume auto or change --seed"
            )
    original=_score(model,device,seed=args.seed,patch=args.patch_size)
    print(f"[v177-enhance] device={device} steps={args.steps} starting={start} "
          f"baseline_holdout={original['baselineLoss']:.6f}",flush=True)
    start_time=time.monotonic()
    for step in range(start,args.steps):
        model.train()
        data=[v.to(device) for v in make_batch(
            args.batch_size,args.patch_size,seed=args.seed+step*9973,
        )]
        a,n,m,ta,tn,tm,protected=data
        optimizer.zero_grad(set_to_none=True)
        pred=model(a,n,m,material_policy="legacy_mask")
        loss,parts=_loss(pred,(ta,tn,tm),protected)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"nonfinite V17.7 loss at step={step+1}")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        update=step+1
        if update%25==0 or update==args.steps or update==1:
            print(f"[v177-enhance] update={update}/{args.steps} "
                  f"loss={float(loss):.6f} grad={float(norm):.5f}",flush=True)
        if update%50==0 or update==args.steps:
            temporary=run/"resume.tmp"
            torch.save({
                "schema":SCHEMA,"config":config,"step":update,
                "model":model.cpu().state_dict(),"optimizer":optimizer.state_dict(),
            },temporary)
            os.replace(temporary,latest)
            model.to(device)
    validation=_score(model,device,seed=args.seed,patch=args.patch_size)
    # A synthetic edge task can be used as a bounded learning diagnostic,
    # but does not imply successful generalisation to real EVE assets.
    improved=(validation["relativeRecovery"]>=.05 and
              validation["candidateLoss"]<original["candidateLoss"] and
              validation["candidateLoss"]<validation["baselineLoss"])
    candidate=run/"candidate.pt"
    saved={
        "schema":SCHEMA,"architecture":"native_8_to_4x_rgb_normal_material_v1",
        "sourceSha256":digest,"config":config,"steps":args.steps,
        "model":model.cpu().state_dict(),
        "syntheticImproved":bool(improved),
        "trainingOnly":True, "productionPromotionAllowed":False,
    }
    temp=run/"candidate.tmp"
    torch.save(saved,temp)
    os.replace(temp,candidate)
    report={
        "schema":EVIDENCE_SCHEMA,"decision":(
            "SYNTHETIC_IMPROVEMENT_ONLY" if improved else "SYNTHETIC_LEARNING_UNPROVEN"
        ),
        "trainingOnly":True,"productionPromotionAllowed":False,
        "actualEveTexturesUsedForTraining":False,
        "networkWeightsUpdated":args.steps>start,
        "trainingStartStep":start,"trainingEndStep":args.steps,
        "config":config,"before":original,"after":validation,
        "checkpoint":str(candidate.resolve()),"checkpointSha256":file_digest(candidate),
        "elapsedSeconds":time.monotonic()-start_time,
        "realRavenQualityVerified":False,
    }
    (run/"report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8",
    )
    print(f"[v177-enhance] decision={report['decision']} "
          f"synthetic_recovery={validation['relativeRecovery']:.4f} "
          f"report={run/'report.json'}",flush=True)
    return 0 if improved else 2


if __name__=="__main__":
    raise SystemExit(main())
