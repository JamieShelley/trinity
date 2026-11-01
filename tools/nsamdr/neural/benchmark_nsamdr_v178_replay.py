#!/usr/bin/env python3
"""Fixed, untouched synthetic controls for an isolated V5-replay diagnostic."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from artifacts.nsamdr.diagnostics.v178_contrast_transfer.probe_analytic_stress import authenticated_models
from tools.nsamdr.neural.benchmark_nsamdr_v178_contrast import CASE_GROUPS, registered_batch
from tools.nsamdr.neural.benchmark_nsamdr_v178_continuity import check_candidate, SEED_START, KINDS
from tools.nsamdr.neural.train_nsamdr_v178_replay import SCHEMA, source_identity, sha
from tools.nsamdr.neural.v178.contour_model import StructuralContourHead,foreground_metrics
from tools.nsamdr.neural.v178.composed_geometry import make_case

@torch.no_grad()
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--repo-root',type=Path,default=ROOT)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
 args=p.parse_args();repo=args.repo_root.resolve();device=torch.device(args.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 torch.set_num_threads(min(4,torch.get_num_threads()))
 models,identity=authenticated_models(device)
 v6=repo/'artifacts/nsamdr/diagnostics/v178_golden_contour/seed_17852_patch_32_batch_4/candidate.pt'
 models['v6'],identity['v6']=check_candidate(v6,repo,device)
 v7=repo/'artifacts/nsamdr/diagnostics/v178_replay_cpu_v7/seed_17862/candidate.pt'
 state=torch.load(v7,map_location='cpu',weights_only=True)
 rpt=json.loads((v7.parent/'report.json').read_text())
 if not (state['schema']==SCHEMA and state['identity']==rpt['identity']
         and state['identity']['sourceHash']==source_identity(repo)
         and state['parentSha256']==identity['v5']['checkpointSha256']
         and rpt['checkpointSha256']==sha(v7)
         and rpt['steps']==state['steps']):
  raise RuntimeError('V7 provenance mismatch')
 models['v7']=StructuralContourHead().to(device)
 models['v7'].load_state_dict(state['model'],strict=True);models['v7'].eval()
 identity['v7']={'checkpointSha256':sha(v7),'parentSha256':state['parentSha256'],
                 'sourceHash':state['identity']['sourceHash'],'trainingSteps':state['steps']}
 values=('contourRecall','contourPrecision','contourIoU','predictedActiveFraction')
 def score(rows):return {field:sum(row[field] for row in rows)/len(rows) for field in values}
 fixed={}
 for name,seeds in CASE_GROUPS.items():
  table={pol:{model:[] for model in models} for pol in ('original','inverted')}
  for seed in seeds:
   a,n,m,t=registered_batch(seed,device)
   for polarity,rgb in (('original',a),('inverted',1-a)):
    for key,model in models.items():
     prediction=model(rgb,n,m)['candidate_contour']
     table[polarity][key].append(foreground_metrics(prediction,t))
  fixed[name]={pol:{key:score(rows) for key,rows in sets.items()} for pol,sets in table.items()}
  print(name,{pol:{key:round(info['contourIoU'],4) for key,info in d.items()} for pol,d in fixed[name].items()},flush=True)
 composed={}
 for kindindex,kind in enumerate(KINDS):
  table={pol:{model:[] for model in models} for pol in ('original','inverted')}
  for i in range(4):
   seed=SEED_START+10000*kindindex+1723*i
   for sampling in ('area','point'):
    a,n,m,t,_=make_case(32,seed,kind=kind,sampling=sampling,phase=(2,1))
    a,n,m,t=[x.unsqueeze(0).to(device) for x in (a,n,m,t)]
    for polarity,rgb in (('original',a),('inverted',1-a)):
     for key,model in models.items():
      prediction=model(rgb,n,m)['candidate_contour']
      table[polarity][key].append(foreground_metrics(prediction,t))
  composed[kind]={pol:{key:score(rows) for key,rows in d.items()} for pol,d in table.items()}
  print(kind,{pol:{key:round(info['contourIoU'],4) for key,info in d.items()} for pol,d in composed[kind].items()},flush=True)
 seeds=[s for item in CASE_GROUPS.values() for s in item]
 seeds += [SEED_START+10000*i+1723*j for i in range(len(KINDS)) for j in range(4)]
 trained={17862+8191*i for i in range(state['steps'])}
 if trained.intersection(seeds):raise RuntimeError('training overlaps registered holdouts')
 output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise RuntimeError('refusing output overwrite')
 output.write_text(json.dumps({'schema':'NSAMDR_V178_REPLAY_DIAGNOSTIC_V1',
  'diagnosticOnly':True,'productionPromotionAllowed':False,'thresholdsUnchanged':True,
  'fixed':fixed,'composed':composed,'provenance':identity},indent=2,sort_keys=True)+'\n')
 return 0

if __name__=='__main__':raise SystemExit(main())
