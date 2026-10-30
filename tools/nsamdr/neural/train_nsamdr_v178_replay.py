#!/usr/bin/env python3
"""Isolated V178 V7 rehearsal: authenticated V5 replay + composed geometry.

Not a promoted checkpoint. This measures whether continuing V5 with three
historical samples per composed sample avoids the V6 fixed-holdout regressions.
It does not change the model, supervised loss, V5 code or registered tests.
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
    sys.path.insert(0, str(ROOT))

from tools.nsamdr.neural.benchmark_nsamdr_v178_contrast import authenticate
from tools.nsamdr.neural.train_nsamdr_v178_contour import _batch, evaluate
from tools.nsamdr.neural.v178.composed_geometry import make_case
from tools.nsamdr.neural.v178.contour_model import StructuralContourHead, contour_supervised_loss

SCHEMA = "NSAMDR_V178_CONTOUR_HEAD_V7_V5_REPLAY_DIAGNOSTIC"
SOURCE_FILES = (
    'tools/nsamdr/neural/v177/synthetic_geometry.py',
    'tools/nsamdr/neural/v178/contour_model.py',
    'tools/nsamdr/neural/train_nsamdr_v178_contour.py',
    'tools/nsamdr/neural/v178/composed_geometry.py',
    'tools/nsamdr/neural/train_nsamdr_v178_replay.py',
)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for piece in iter(lambda: f.read(1<<20), b''):
            h.update(piece)
    return h.hexdigest()

def source_identity(repo):
    h=hashlib.sha256()
    for path in SOURCE_FILES:
        h.update(path.encode());h.update((repo/path).read_bytes())
    return h.hexdigest()

def replay_batch(patch, seed, device, composed_slot=3):
    if not 0 <= composed_slot < 4:
        raise ValueError('composed_slot must be 0..3')
    tensors=[]
    for i in range(4):
        sid=seed+i*1723
        if i==composed_slot:
            a,n,m,t,_=make_case(patch,sid)
            rng=torch.Generator(device='cpu').manual_seed(sid^0x5A17842)
            if torch.rand((),generator=rng).item()<.5:
                a=1.-a
            tensors.append(tuple(z.to(device) for z in (a,n,m,t)))
        else:
            tensors.append(tuple(z[0] for z in _batch(1,patch,sid,device,augment_rgb=True)))
    return tuple(torch.stack([item[k] for item in tensors]) for k in range(4))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=ROOT)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--steps',type=int,default=120)
    p.add_argument('--seed',type=int,default=17862)
    p.add_argument('--learning-rate',type=float,default=.0001)
    p.add_argument('--resume',choices=('fresh','auto'),default='fresh')
    a=p.parse_args(argv)
    if not (1<=a.steps<=400 and 0<a.learning_rate<=.0007):raise ValueError('bounded fine tuning only')
    device=torch.device(a.device)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    repo=a.repo_root.resolve();sid=source_identity(repo)
    parent=repo/'artifacts/nsamdr/diagnostics/v178_golden_contour/seed_17842_patch_32_batch_4/candidate.pt'
    original,origin_report=authenticate(parent,repo)
    identity={'sourceHash':sid,'seed':a.seed,'learningRate':a.learning_rate,
              'patchSize':32,'batchSize':4,'composedPerBatch':1,
              'geometryMix':'V5_REPLAY_3_OF_4_COMPOSED_1_OF_4',
              'parentCheckpointSha256':sha(parent),'parentReportSha256':sha(parent.parent/'report.json'),
              'modelSchema':SCHEMA}
    out=repo/'artifacts/nsamdr/diagnostics/v178_replay_cpu_v7'/f'seed_{a.seed}'
    if (out/'candidate.pt').exists() or (out/'report.json').exists():
        raise RuntimeError('existing V7 candidate/report cannot be overwritten')
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(a.seed)
    model=StructuralContourHead().to(device)
    model.load_state_dict(original['model'],strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate)
    initial=evaluate(model,device,patch=32,seed=a.seed)
    resume=out/'resume.pt';start=0
    if a.resume=='fresh' and resume.exists():raise RuntimeError('existing resume requires --resume auto')
    if a.resume=='auto' and resume.exists():
        prior=torch.load(resume,weights_only=True,map_location='cpu')
        if prior['schema']!=SCHEMA or prior['identity']!=identity:raise RuntimeError('resume identity mismatch')
        model.load_state_dict(prior['model'],strict=True)
        optimizer.load_state_dict(prior['optimizer'])
        start=prior['step']
        if start>a.steps:raise RuntimeError('cannot reduce horizon')
    began=time.monotonic()
    for step in range(start,a.steps):
        model.train()
        x,n,m,t=replay_batch(32,a.seed+8191*step,device,composed_slot=step%4)
        optimizer.zero_grad(set_to_none=True)
        output=model(x,n,m)['candidate_contour']
        loss,_=contour_supervised_loss(output,t)
        if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss at {step+1}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step==0 or (step+1)%20==0 or step+1==a.steps:
            print(f'[V7 replay] {step+1}/{a.steps} loss={loss.item():.6f}',flush=True)
        if (step+1)%20==0 or step+1==a.steps:
            tmp=out/'resume.tmp'
            torch.save({'schema':SCHEMA,'identity':identity,'step':step+1,
                        'model':model.state_dict(),'optimizer':optimizer.state_dict()},tmp)
            os.replace(tmp,resume)
    after=evaluate(model,device,patch=32,seed=a.seed)
    final=out/'candidate.pt';tmp=out/'candidate.tmp'
    torch.save({'schema':SCHEMA,'identity':identity,'model':model.state_dict(),
                'steps':a.steps,'parentSha256':identity['parentCheckpointSha256'],
                'productionPromotionAllowed':False,'realEveReferenceQualified':False},tmp)
    os.replace(tmp,final)
    report={'schema':'NSAMDR_V178_V7_REPLAY_OFFLINE_REPORT_V1',
            'device':str(device),'identity':identity,'steps':a.steps,'trainingUpdates':a.steps-start,
            'checkpointSha256':sha(final),'initial':initial,'after':after,
            'elapsedSeconds':time.monotonic()-began,'decision':'DIAGNOSTIC_ONLY',
            'productionPromotionAllowed':False,'realEveQualityVerified':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(f'[V7 replay] diagnostic complete, checkpoint={final}',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
