#!/usr/bin/env python3
"""Immutable evaluation of V8 on registered cases and normal-nuisance controls."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from artifacts.nsamdr.diagnostics.v178_contrast_transfer.probe_analytic_stress import authenticated_models
from tools.nsamdr.neural.benchmark_nsamdr_v178_contrast import CASE_GROUPS,registered_batch
from tools.nsamdr.neural.v178.contour_model import StructuralContourHead,foreground_metrics
from tools.nsamdr.neural.v178.composed_geometry import make_case,KINDS
from tools.nsamdr.neural.diagnose_nsamdr_v178_normal_domain import perturb
from tools.nsamdr.neural.train_nsamdr_v178_normal_variation import SCHEMA,identity,sha

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',required=True,type=Path)
 p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
 args=p.parse_args();device=torch.device(args.device)
 torch.set_num_threads(min(4,torch.get_num_threads()))
 models,provenance=authenticated_models(device)
 models.pop('v4')
 path=ROOT/'artifacts/nsamdr/diagnostics/v178_normal_variation_cpu_v8/seed_17872/candidate.pt'
 state=torch.load(path,map_location='cpu',weights_only=True)
 report=json.loads((path.parent/'report.json').read_text())
 if not (state['schema']==SCHEMA and state['provenance']==report['provenance']
    and state['provenance']['sourceHash']==identity(ROOT)
    and state['provenance']['parentCheckpointSha256']==provenance['v5']['checkpointSha256']
    and report['checkpointSha256']==sha(path) and report['steps']==state['steps']):
  raise RuntimeError('V8 authentication failed')
 models['v8']=StructuralContourHead().to(device);models['v8'].load_state_dict(state['model'],strict=True);models['v8'].eval()
 provenance['v8']={'checkpointSha256':sha(path),'parentCheckpointSha256':state['provenance']['parentCheckpointSha256']}
 keys=('contourIoU','contourPrecision','contourRecall','predictedActiveFraction')
 def mean(vals):return {k:sum(x[k] for x in vals)/len(vals) for k in keys}
 fixed={};sample_cases={}
 with torch.no_grad():
  for name,seeds in CASE_GROUPS.items():
   rows={condition:{model:{pol:[] for pol in ('original','inverted')} for model in models}
         for condition in ('clean','smooth','iid')}
   for seed in seeds:
    a,n,m,t=registered_batch(seed,device)
    for condition in rows:
     if condition=='clean':nn,mm=n,m
     else:nn,mm=perturb(a,n,m,seed,kind='smooth' if condition=='smooth' else 'iid')
     for pol,rgb in (('original',a),('inverted',1-a)):
      for model,network in models.items():
       pred=network(rgb,nn,mm)['candidate_contour']
       rows[condition][model][pol].append(foreground_metrics(pred,t))
   fixed[name]={condition:{model:{pol:mean(v) for pol,v in sides.items()} for model,sides in row.items()}
                for condition,row in rows.items()}
   for condition in rows:
    print(name,condition,{model:{pol:round(v['contourIoU'],4) for pol,v in sides.items()}
       for model,sides in fixed[name][condition].items()},flush=True)
  composed={}
  for i,kind in enumerate(KINDS):
   rows={model:{pol:[] for pol in ('original','inverted')} for model in models}
   for j in range(4):
    seed=50000000+10000*i+1723*j
    for sampling in ('area','point'):
     a,n,m,t,_=make_case(32,seed,kind=kind,sampling=sampling,phase=(2,1))
     a,n,m,t=[v.unsqueeze(0).to(device) for v in (a,n,m,t)]
     for pol,rgb in (('original',a),('inverted',1-a)):
      for model,network in models.items():
       pred=network(rgb,n,m)['candidate_contour']
       rows[model][pol].append(foreground_metrics(pred,t))
   composed[kind]={model:{pol:mean(v) for pol,v in sides.items()} for model,sides in rows.items()}
   print('composed',kind,{model:{pol:round(v['contourIoU'],4) for pol,v in sides.items()}
      for model,sides in composed[kind].items()},flush=True)
 output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise RuntimeError('refuse overwrite')
 output.write_text(json.dumps({'schema':'NSAMDR_V178_NORMAL_VARIATION_FIXED_DIAGNOSTIC_V1',
  'diagnosticOnly':True,'realEveQualityVerified':False,'thresholdsChanged':False,
  'provenance':provenance,'fixed':fixed,'composed':composed},indent=2,sort_keys=True)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
