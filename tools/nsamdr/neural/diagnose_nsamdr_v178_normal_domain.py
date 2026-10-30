#!/usr/bin/env python3
"""Fixed-holdout, inference-only independent normal-domain stress diagnostic.

Does not train, alter checkpoints, loosen gates or simulate exact EVE normals.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from artifacts.nsamdr.diagnostics.v178_contrast_transfer.probe_analytic_stress import authenticated_models
from tools.nsamdr.neural.benchmark_nsamdr_v178_continuity import check_candidate
from tools.nsamdr.neural.benchmark_nsamdr_v178_contrast import CASE_GROUPS,registered_batch
from tools.nsamdr.neural.v178.contour_model import physical_edge_evidence,foreground_metrics

def perturb(a,n,obs,seed,scale=.20,kind='iid'):
    rng=torch.Generator(device='cpu').manual_seed(seed^0x458178)
    z=torch.randn(tuple(n.shape),generator=rng)
    if kind=='smooth':
        z=F.avg_pool2d(F.pad(z,(2,2,2,2),mode='reflect'),5,stride=1)
        z=z*(z.square().mean().sqrt().clamp_min(1e-7).reciprocal())
    new_n=(n.cpu()+scale*z).clamp(-1,1).to(n.device)
    new_obs=torch.cat((physical_edge_evidence(a,new_n),obs[:,1:]),dim=1)
    return new_n,new_obs

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
 args=p.parse_args();device=torch.device(args.device)
 torch.set_num_threads(min(4,torch.get_num_threads()))
 models,provenance=authenticated_models(device)
 models['v6'],provenance['v6']=check_candidate(ROOT/'artifacts/nsamdr/diagnostics/v178_golden_contour/seed_17852_patch_32_batch_4/candidate.pt',ROOT,device)
 groups={}
 for group,seeds in CASE_GROUPS.items():
  table={case:{name:{pol:[] for pol in ('original','inverted')} for name in models}
         for case in ('native_synthetic','iid_noise','smooth_noise')}
  inputs={case:[] for case in table}
  for seed in seeds:
   a,n,m,t=registered_batch(seed,device)
   for case in table:
    if case=='native_synthetic':nn,mm=n,m
    else:nn,mm=perturb(a,n,m,seed,kind='iid' if case=='iid_noise' else 'smooth')
    inputs[case].append({'absNormalMean':float(nn.abs().mean()),'edgeMean':float(mm[:,:1].mean()),
                         'normalWasModified':not torch.equal(nn,n),
                         'targetUnchanged':True,'albedoUnchanged':True})
    for modelname,model in models.items():
     for pol,rgb in (('original',a),('inverted',1-a)):
      pred=model(rgb,nn,mm)['candidate_contour']
      table[case][modelname][pol].append(foreground_metrics(pred,t))
  groups[group]={case:{'normalAbs':sum(x['absNormalMean'] for x in inputs[case])/len(seeds),
                       'edgeMean':sum(x['edgeMean'] for x in inputs[case])/len(seeds),
                       'scores':{model:{pol:{key:sum(v[key] for v in vals)/len(vals)
                                               for key in ('contourIoU','contourPrecision','contourRecall','predictedActiveFraction')}
                                           for pol,vals in pols.items()}
                                 for model,pols in table[case].items()}}
                 for case in table}
  print(group)
  for case, data in groups[group].items():
   print(' ',case,'normal',round(data['normalAbs'],3),'edge',round(data['edgeMean'],3),
         'orig IoU',{model:round(scores['original']['contourIoU'],3) for model,scores in data['scores'].items()},flush=True)
 out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise RuntimeError('cannot overwrite diagnostic')
 out.write_text(json.dumps({'schema':'NSAMDR_V178_NORMAL_DOMAIN_STRESS_V1',
    'diagnosticOnly':True,'noNetworkOrLossChange':True,
    'warning':'Independent Gaussian normal nuisance is a controlled stress, not a physically realistic EVE normal map',
    'groups':groups},indent=2,sort_keys=True)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
