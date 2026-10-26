#!/usr/bin/env python3
"""Bounded G2 synthetic structural-contour experiment.

No EVE sealed tests, no manual golden reference exposed to training.
Only unseen analytic clean 4× geometry is eligible for the first gate.
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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from tools.nsamdr.neural.v177.synthetic_geometry import make_batch
from tools.nsamdr.neural.v178.contour_model import (
    SCHEMA, StructuralContourHead, contour_loss, physical_edge_evidence,
)

TRAIN_SCHEMA = "NSAMDR_V178_G2_CONTOUR_REPORT_V1"
FILES = (
    "tools/nsamdr/neural/v177/synthetic_geometry.py",
    "tools/nsamdr/neural/v178/contour_model.py",
    "tools/nsamdr/neural/train_nsamdr_v178_contour.py",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def source_hash(repo: Path) -> str:
    h = hashlib.sha256()
    for name in FILES:
        h.update(name.encode())
        h.update((repo/name).read_bytes())
    return h.hexdigest()


def _batch(count:int, patch:int, seed:int, device:torch.device):
    images=make_batch(count,patch,seed=seed)
    a,n,m,_,_,target_material,_=images
    a,n,m=a.to(device),n.to(device),m.to(device)
    # Synthetic and real inference use the SAME observation semantics:
    # native RGB/normal structural edges, not synthetic coverage or V5 IDs.
    edges=physical_edge_evidence(a,n)
    observation=torch.cat((edges,m[:,1:]),dim=1)
    return (a,n,observation,target_material[:,:1].to(device))


@torch.no_grad()
def evaluate(model: StructuralContourHead, device: torch.device,
             *, patch: int, seed: int, cases: int=12) -> dict:
    model.eval()
    baseline_total=0.
    candidate_total=0.
    components={"coverage":0.,"tangent":0.,"curvature":0.}
    for index in range(cases):
        a,n,m,reference=_batch(1,patch,seed+1000000+index*1723,device)
        result=model(a,n,m)
        base,_=contour_loss(result["baseline_contour"],reference)
        pred,parts=contour_loss(result["candidate_contour"],reference)
        baseline_total+=float(base)
        candidate_total+=float(pred)
        for k,v in parts.items():
            components[k]+=v/cases
    baseline_total/=cases
    candidate_total/=cases
    return {
        "baselineContourLoss": baseline_total,
        "candidateContourLoss": candidate_total,
        "relativeContourRecovery": 1.-candidate_total/max(1e-12,baseline_total),
        "candidateComponents": components,
        "holdoutSeedStart":seed+1000000,
        "holdoutCaseCount":cases,
    }


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=ROOT)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--steps",type=int,default=400)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--patch-size",type=int,default=32)
    p.add_argument("--seed",type=int,default=17812)
    p.add_argument("--learning-rate",type=float,default=.0007)
    p.add_argument("--resume",choices=("fresh","auto"),default="fresh")
    return p


def main(argv:list[str]|None=None)->int:
    args=parser().parse_args(argv)
    if (not 1<=args.steps<=12000 or not 1<=args.batch_size<=16 or
        not 16<=args.patch_size<=128 or not 0<args.learning_rate<=.01):
        raise ValueError("out-of-bounds G2 synthetic training parameters")
    repo=args.repo_root.resolve()
    code_hash=source_hash(repo)
    device=torch.device(args.device)
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    if args.device=="cpu":
        torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(args.seed)
    if args.device=="cuda":
        torch.cuda.manual_seed_all(args.seed)
    model=StructuralContourHead().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    config={
        "seed":args.seed,"patchSize":args.patch_size,
        "batchSize":args.batch_size,"learningRate":args.learning_rate,
        "sourceHash":code_hash,
    }
    folder=repo/"artifacts/nsamdr/diagnostics/v178_golden_contour"/(
        f"seed_{args.seed}_patch_{args.patch_size}_batch_{args.batch_size}"
    )
    folder.mkdir(parents=True,exist_ok=True)
    resume_file=folder/"resume.pt"
    start=0
    if args.resume=="fresh" and resume_file.exists():
        raise RuntimeError("a checkpoint exists; use --resume auto or a new seed")
    if args.resume=="auto" and resume_file.is_file():
        saved=torch.load(resume_file,map_location="cpu",weights_only=True)
        if saved.get("schema")!=SCHEMA or saved.get("config")!=config:
            raise RuntimeError("G2 checkpoint source/configuration identity mismatch")
        model.load_state_dict(saved["model"],strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):
                    state[key]=value.to(device)
        start=int(saved["step"])
        if start>args.steps:
            raise ValueError("requested training horizon precedes resumed step")
    before=evaluate(model,device,patch=args.patch_size,seed=args.seed)
    print(f"[v178-golden] G2 synthetic structural training "
          f"start={start} target={args.steps} baseline={before['baselineContourLoss']:.6f}",
          flush=True)
    start_time=time.monotonic()
    for index in range(start,args.steps):
        model.train()
        a,n,m,reference=_batch(args.batch_size,args.patch_size,
                               args.seed+index*8191,device)
        optimizer.zero_grad(set_to_none=True)
        prediction=model(a,n,m)
        loss,_=contour_loss(prediction["candidate_contour"],reference)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite G2 contour loss at {index+1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        optimizer.step()
        step=index+1
        if step==1 or step%25==0 or step==args.steps:
            print(f"[v178-golden] update={step}/{args.steps} "
                  f"contour_loss={float(loss):.6f}",flush=True)
        if step%50==0 or step==args.steps:
            temp=folder/"resume.tmp"
            torch.save({"schema":SCHEMA,"config":config,"step":step,
                        "model":model.state_dict(),"optimizer":optimizer.state_dict()},temp)
            os.replace(temp,resume_file)
    after=evaluate(model,device,patch=args.patch_size,seed=args.seed)
    # Only synthetic demonstration gate; NEVER a real-texture promotion.
    qualified=(
        after["relativeContourRecovery"]>=.20
        and after["candidateContourLoss"]<before["candidateContourLoss"]
        and after["candidateContourLoss"]<after["baselineContourLoss"]
    )
    checkpoint=folder/"candidate.pt"
    temp=folder/"candidate.tmp"
    torch.save({
        "schema":SCHEMA,"config":config,"model":model.state_dict(),
        "syntheticPass":qualified,"steps":args.steps,
        "productionPromotionAllowed":False,
        "realEveReferenceQualified":False,
    },temp)
    os.replace(temp,checkpoint)
    report={
        "schema":TRAIN_SCHEMA,
        "decision":"SYNTHETIC_CONTOUR_PASS_ONLY" if qualified else "G2_SYNTHETIC_FAIL",
        "sourceHash":code_hash,"config":config,
        "checkpoint":str(checkpoint.resolve()),"checkpointSha256":digest(checkpoint),
        "before":before,"after":after,
        "requiredRelativeContourRecovery":.20,
        "trainingUpdates":args.steps-start,
        "elapsedSeconds":time.monotonic()-start_time,
        "realRavenReferenceAvailable":False,
        "realEveQualityVerified":False,
        "productionPromotionAllowed":False,
        "sealedTestAccess":False,
    }
    (folder/"report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(f"[v178-golden] {report['decision']} "
          f"contour_recovery={after['relativeContourRecovery']:.3f} "
          f"report={folder/'report.json'}",flush=True)
    return 0 if qualified else 2


if __name__=="__main__":
    raise SystemExit(main())
