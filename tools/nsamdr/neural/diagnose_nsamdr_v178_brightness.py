#!/usr/bin/env python3
"""Diagnostic RGB brightness transfer on registered fixed cases; never trains."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from artifacts.nsamdr.diagnostics.v178_contrast_transfer.probe_analytic_stress import authenticated_models
from tools.nsamdr.neural.benchmark_nsamdr_v178_continuity import check_candidate
from tools.nsamdr.neural.benchmark_nsamdr_v178_contrast import CASE_GROUPS,registered_batch
from tools.nsamdr.neural.v178.contour_model import StructuralContourHead,foreground_metrics,physical_edge_evidence
from tools.nsamdr.neural.train_nsamdr_v178_replay import source_identity,sha,SCHEMA

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',required=True,type=Path)
 p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
 args=p.parse_args();device=torch.device(args.device)
 torch.set_num_threads(min(4,torch.get_num_threads()))
 models,provenance=authenticated_models(device)
 models['v6'],provenance['v6']=check_candidate(ROOT/'artifacts/nsamdr/diagnostics/v178_golden_contour/seed_17852_patch_32_batch_4/candidate.pt',ROOT,device)
 v7=ROOT/'artifacts/nsamdr/diagnostics/v178_replay_cpu_v7/seed_17862/candidate.pt'
 state=torch.load(v7,map_location='cpu',weights_only=True); report=json.loads((v7.parent/'report.json').read_text())
 if not (state['schema']==SCHEMA and report['checkpointSha256']==sha(v7) and state['identity']==report['identity'] and state['identity']['sourceHash']==source_identity(ROOT)):
  raise RuntimeError('V7 identity failure')
 models['v7']=StructuralContourHead().to(device)
 models['v7'].load_state_dict(state['model'],strict=True);models['v7'].eval()
 native=[]
 for pth in sorted((ROOT/'artifacts/nsamdr/golden_patch').glob('*/native_albedo.png')):
  arr=np.asarray(Image.open(pth).convert('RGB'),dtype=np.float32)/255
  native.append({'asset':pth.parent.name,'meanRGB':float(arr.mean())})
 groups={}
 with torch.no_grad():
  for name,seeds in CASE_GROUPS.items():
   rows=[]
   for seed in seeds:
    a,n,m,t=registered_batch(seed,device)
    bright=(a+.42).clamp(0,1)
    clipping=float((a+.42>1).float().mean())
    # Calibrated observation stays fixed to isolate absolute-RGB sensitivity;
    # independently report the changed edge evidence due to saturation.
    edge_change=float((physical_edge_evidence(bright,n)-m[:,:1]).abs().amax())
    measures={'seed':seed,'rgbMeanOriginal':float(a.mean()),'rgbMeanBright':float(bright.mean()),
              'clippedPixelFraction':clipping,'maxEdgeChange':edge_change,'models':{}}
    for modelname,model in models.items():
     measures['models'][modelname]={}
     for polarity in ('original','inverted'):
      source=a if polarity=='original' else 1-a
      modified=bright if polarity=='original' else 1-bright
      before=foreground_metrics(model(source,n,m)['candidate_contour'],t)
      after=foreground_metrics(model(modified,n,m)['candidate_contour'],t)
      measures['models'][modelname][polarity]={'before':before,'bright':after}
    rows.append(measures)
   groups[name]={
    'sourceMean':sum(v['rgbMeanOriginal'] for v in rows)/len(rows),
    'brightMean':sum(v['rgbMeanBright'] for v in rows)/len(rows),
    'maxClippedFraction':max(v['clippedPixelFraction'] for v in rows),
    'maxEdgeChange':max(v['maxEdgeChange'] for v in rows),
    'meanIoU':{model:{pol:{condition:sum(row['models'][model][pol][condition]['contourIoU'] for row in rows)/len(rows)
                     for condition in ('before','bright')} for pol in ('original','inverted')}
               for model in models}}
   print(name,{model:{pol:{k:round(v,4) for k,v in result.items()} for pol,result in p.items()} for model,p in groups[name]['meanIoU'].items()},flush=True)
 out=args.output.resolve();out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists():raise RuntimeError('refusing to overwrite diagnostic report')
 out.write_text(json.dumps({'schema':'NSAMDR_V178_BRIGHTNESS_TRANSFER_DIAGNOSTIC_V1','intervention':'+0.42 RGB clamp, all geometry, normal, native edge observation, target and auxiliary held fixed','diagnosticOnly':True,'native':native,'groups':groups},indent=2,sort_keys=True)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
