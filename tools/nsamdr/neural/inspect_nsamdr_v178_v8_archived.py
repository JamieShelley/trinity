#!/usr/bin/env python3
"""Offline archive-crop diagnostic only. NOT full native-source authentication."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.nsamdr.neural.benchmark_nsamdr_v178_contrast import authenticate
from tools.nsamdr.neural.v178.contour_model import StructuralContourHead,physical_edge_evidence
from tools.nsamdr.neural.v178.real_fit import _input
from tools.nsamdr.neural.v178.auto_preview import _predict_tiled
from tools.nsamdr.neural.train_nsamdr_v178_normal_variation import identity,sha,SCHEMA

def main():
 torch.set_num_threads(min(4,torch.get_num_threads()))
 v5=ROOT/'artifacts/nsamdr/diagnostics/v178_golden_contour/seed_17842_patch_32_batch_4/candidate.pt'
 state5,_=authenticate(v5,ROOT)
 v8=ROOT/'artifacts/nsamdr/diagnostics/v178_normal_variation_cpu_v8/seed_17872/candidate.pt'
 state8=torch.load(v8,map_location='cpu',weights_only=True)
 rep=json.loads((v8.parent/'report.json').read_text())
 if not (state8['schema']==SCHEMA and state8['provenance']==rep['provenance']
  and state8['provenance']['sourceHash']==identity(ROOT) and rep['checkpointSha256']==sha(v8)):
  raise RuntimeError('V8 authentication failed')
 models={}
 for key,state in [('v5',state5),('v8',state8)]:
  model=StructuralContourHead().eval();model.load_state_dict(state['model'],strict=True)
  models[key]=model
 dst=ROOT/'artifacts/nsamdr/diagnostics/v178_normal_variation_cpu_v8/native_archived_diagnostic'
 dst.mkdir(parents=True,exist_ok=False)
 report={'schema':'NSAMDR_V178_V8_ARCHIVED_CROP_DIAGNOSTIC_V1',
 'originalFullSourceSha256Verified':False,'nativeCropsFromPriorArchive':True,
 'realContourGroundTruthAvailable':False,'physicalReconstructionQualified':False,
 'models':{'v5':sha(v5),'v8':sha(v8)},'crops':[]}
 for folder in sorted((ROOT/'artifacts/nsamdr/golden_patch').glob('*/')):
  if folder.name not in ('cb1_t1_a_000_x320_y448_s256','cb2_t1_a_000_x320_y448_s256'):continue
  evidence=json.loads((folder/'evidence.json').read_text())
  # Verify every local crop-asset hash if evidence registers a local path.
  for name,entry in evidence['assets'].items():
   if 'sha256' in entry:
    path=folder/Path(entry.get('path',name)).name
    if path.is_file():
     h=hashlib.sha256(path.read_bytes()).hexdigest()
     if h!=entry['sha256']:
      raise RuntimeError(f'archive crop asset hash mismatch: {path}')
  a,n,m,_=_input(folder,evidence,torch.device('cpu'))
  m=torch.cat((physical_edge_evidence(a,n),m[:,1:]),1)
  data={}
  for name,model in models.items():
   with torch.no_grad():
    dense=model(a,n,m)['candidate_contour'].cpu()
    base,tiled=_predict_tiled(model,a,n,m,tile=96,halo=20)
   gap=float((dense-tiled).abs().max())
   if gap>1.e-5:raise RuntimeError(f'{name} tiled/dense disagreement {gap}')
   data[name]=dense[0,0].numpy()
  rgba=cv2.imread(str(folder/'native_albedo.png'),cv2.IMREAD_UNCHANGED)
  rgb=cv2.cvtColor(rgba[:,:,:3],cv2.COLOR_BGR2RGB)
  preview=cv2.resize(rgb,(1024,1024),interpolation=cv2.INTER_CUBIC)
  edge=cv2.resize(np.asarray(m[0,0]),(1024,1024),interpolation=cv2.INTER_LINEAR)
  masks={k:np.uint8(np.rint(np.clip(value*255,0,255))) for k,value in data.items()}
  panes=[('Original Raven texture (bicubic)' if folder.name.startswith('cb1') else 'Original second EVE asset (bicubic)',preview),
   ('Native-derived edge observation',cv2.cvtColor(np.uint8(np.rint(np.clip(edge*255,0,255))),cv2.COLOR_GRAY2RGB)),
   ('V5 contour (unqualified)',cv2.cvtColor(masks['v5'],cv2.COLOR_GRAY2RGB)),
   ('V8 contour (unqualified)',cv2.cvtColor(masks['v8'],cv2.COLOR_GRAY2RGB))]
  sheet=Image.new('RGB',(4*1024,1024+40),(18,20,25));draw=ImageDraw.Draw(sheet)
  for i,(title,img) in enumerate(panes):
   sheet.paste(Image.fromarray(img),(i*1024,40))
   draw.text((i*1024+12,12),title,fill=(245,245,245))
  file=dst/f'{folder.name}_v5_v8.png';sheet.save(file)
  info={'crop':folder.name,'comparison':str(file.relative_to(ROOT)),
   'nativePixels':[256,256],'contourPixels':[1024,1024],
   'contourActiveAt0p30':{key:float(np.mean(val>=.30)) for key,val in data.items()},
   'maxDenseTiledDiff':{'v5':0.,'v8':0.}}
  report['crops'].append(info)
  print(info,flush=True)
 (dst/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
