#!/usr/bin/env python3
"""Offline V8 diagnostic: replay V5 and add independent native normal nuisance.

Preserves architecture, loss and original checkpoint; never promotes the output.
The independent Gaussian perturbation is not claimed to model real EVE normals.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.nsamdr.neural.benchmark_nsamdr_v178_contrast import authenticate
from tools.nsamdr.neural.v178.contour_model import StructuralContourHead,contour_supervised_loss
from tools.nsamdr.neural.train_nsamdr_v178_contour import evaluate
from tools.nsamdr.neural.train_nsamdr_v178_replay import replay_batch, sha
from tools.nsamdr.neural.diagnose_nsamdr_v178_normal_domain import perturb

SCHEMA='NSAMDR_V178_V8_NORMAL_VARIATION_OFFLINE_EXPERIMENT'
SOURCE_FILES=('tools/nsamdr/neural/v177/synthetic_geometry.py',
 'tools/nsamdr/neural/v178/contour_model.py',
 'tools/nsamdr/neural/train_nsamdr_v178_contour.py',
 'tools/nsamdr/neural/v178/composed_geometry.py',
 'tools/nsamdr/neural/train_nsamdr_v178_replay.py',
 'tools/nsamdr/neural/diagnose_nsamdr_v178_normal_domain.py',
 'tools/nsamdr/neural/train_nsamdr_v178_normal_variation.py')

def identity(repo):
 h=hashlib.sha256()
 for path in SOURCE_FILES:
  h.update(path.encode());h.update((repo/path).read_bytes())
 return h.hexdigest()

def variation_batch(seed,step,device):
 """Two out of four independent normal nuisance cases; retain geometric target."""
 a,n,m,t=replay_batch(32,seed,device,composed_slot=step%4)
 nn=n.clone();mm=m.clone()
 for i in range(4):
  if i%2 == (step//4)%2:
   newn,newm=perturb(a[i:i+1],n[i:i+1],m[i:i+1],seed+i*1723,kind='smooth')
   nn[i:i+1]=newn;mm[i:i+1]=newm
 return a,nn,mm,t

def main(argv=None):
 p=argparse.ArgumentParser()
 p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
 p.add_argument('--steps',type=int,default=160)
 p.add_argument('--seed',type=int,default=17872)
 p.add_argument('--lr',type=float,default=.00015)
 p.add_argument('--resume',choices=('fresh','auto'),default='fresh')
 a=p.parse_args(argv)
 if not (1<=a.steps<=400 and 0<a.lr<=.0007):raise ValueError('bounded experiment only')
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 device=torch.device(a.device);torch.set_num_threads(min(4,torch.get_num_threads()))
 parent=ROOT/'artifacts/nsamdr/diagnostics/v178_golden_contour/seed_17842_patch_32_batch_4/candidate.pt'
 old,_=authenticate(parent,ROOT)
 provenance={'sourceHash':identity(ROOT),'seed':a.seed,'lr':a.lr,'parentCheckpointSha256':sha(parent),
             'parentModelSchema':old['schema'],'trainGeometry':'3 V5 replay / 1 composed',
             'normalVariation':'2 of 4 samples independently perturbed, smooth seeded Gaussian sigma=0.2',
             'modelSchema':SCHEMA}
 directory=ROOT/'artifacts/nsamdr/diagnostics/v178_normal_variation_cpu_v8'/f'seed_{a.seed}'
 if (directory/'candidate.pt').exists() or (directory/'report.json').exists():
  raise RuntimeError('cannot overwrite completed candidate')
 directory.mkdir(parents=True,exist_ok=True)
 torch.manual_seed(a.seed)
 model=StructuralContourHead().to(device);model.load_state_dict(old['model'],strict=True)
 optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
 start=0;resume=directory/'resume.pt'
 if a.resume=='fresh' and resume.exists():raise RuntimeError('existing resumable state')
 if a.resume=='auto' and resume.exists():
  saved=torch.load(resume,map_location='cpu',weights_only=True)
  if saved['schema']!=SCHEMA or saved['provenance']!=provenance:raise RuntimeError('resume mismatch')
  model.load_state_dict(saved['model'],strict=True)
  optimizer.load_state_dict(saved['optimizer']);start=saved['step']
  if start>a.steps:raise RuntimeError('reducing training horizon')
 initial=evaluate(model,device,patch=32,seed=a.seed)
 began=time.monotonic()
 for i in range(start,a.steps):
  model.train();rgb,norm,obs,target=variation_batch(a.seed+8191*i,i,device)
  optimizer.zero_grad(set_to_none=True)
  prediction=model(rgb,norm,obs)['candidate_contour']
  loss,_=contour_supervised_loss(prediction,target)
  if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
  loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
  optimizer.step()
  if i==0 or (i+1)%20==0 or i+1==a.steps:
   print(f'[V8 variation] update={i+1}/{a.steps} loss={loss.item():.6f}',flush=True)
  if (i+1)%20==0 or i+1==a.steps:
   tmp=directory/'resume.tmp';torch.save({'schema':SCHEMA,'provenance':provenance,'step':i+1,
     'model':model.state_dict(),'optimizer':optimizer.state_dict()},tmp);os.replace(tmp,resume)
 after=evaluate(model,device,patch=32,seed=a.seed)
 checkpoint=directory/'candidate.pt';tmp=directory/'candidate.tmp'
 torch.save({'schema':SCHEMA,'provenance':provenance,'model':model.state_dict(),
  'steps':a.steps,'productionPromotionAllowed':False,'realEveQualityVerified':False},tmp)
 os.replace(tmp,checkpoint)
 report={'schema':'NSAMDR_V178_V8_NORMAL_VARIATION_REPORT_V1','device':str(device),
  'provenance':provenance,'checkpointSha256':sha(checkpoint),'steps':a.steps,'trainingUpdates':a.steps-start,
  'initial':initial,'after':after,'elapsedSeconds':time.monotonic()-began,
  'decision':'DIAGNOSTIC_ONLY','productionPromotionAllowed':False,'realEveQualityVerified':False}
 (directory/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print('[V8 variation] complete (not promoted)',flush=True)
 return 0
if __name__=='__main__':raise SystemExit(main())
