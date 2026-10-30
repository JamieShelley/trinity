#!/usr/bin/env python3
"""V4/V5/V6 exact registered 36-case benchmark, diagnostic only."""
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
from tools.nsamdr.neural.benchmark_nsamdr_v178_continuity import check_candidate
from tools.nsamdr.neural.v178.contour_model import foreground_metrics

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--candidate',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
 args=p.parse_args();device=torch.device(args.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 torch.set_num_threads(min(4,torch.get_num_threads()))
 models,provenance=authenticated_models(device)
 models['v6'],provenance['v6']=check_candidate(args.candidate.resolve(),ROOT,device)
 candidates=set()
 for grp in CASE_GROUPS.values():candidates.update(grp)
 training={provenance['v6']['trainingSeed']+8191*i for i in range(provenance['v6']['trainingSteps'])}
 if candidates&training:raise RuntimeError('V6 training seed overlaps registered holdouts')
 groups={}
 keys=('contourRecall','contourPrecision','contourIoU','predictedActiveFraction')
 for group,seeds in CASE_GROUPS.items():
  rows={f'{model}_{pol}':[] for model in models for pol in ('original','inverted')}
  for seed in seeds:
   a,n,m,t=registered_batch(seed,device)
   for model,network in models.items():
    for polarity,rgb in (('original',a),('inverted',1-a)):
     prediction=network(rgb,n,m)['candidate_contour']
     rows[f'{model}_{polarity}'].append(foreground_metrics(prediction,t))
  groups[group]={name:{key:sum(v[key] for v in values)/len(values) for key in keys}
                 for name,values in rows.items()}
  print(group,{name:round(groupdata['contourIoU'],4) for name,groupdata in groups[group].items()},flush=True)
 output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
 if output.exists():raise RuntimeError('refusing to overwrite registered benchmark')
 output.write_text(json.dumps({'schema':'NSAMDR_V178_V6_FIXED_HOLDOUT_DIAGNOSTIC_V1',
  'diagnosticOnly':True,'models':provenance,'registeredCases':36,
  'originalCaseSetsPreserved':True,'trainingSeedsDisjoint':True,
  'acceptanceThresholdsChanged':False,'groups':groups},indent=2,sort_keys=True)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
