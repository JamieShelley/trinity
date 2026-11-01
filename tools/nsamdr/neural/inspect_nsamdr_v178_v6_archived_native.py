#!/usr/bin/env python3
"""Archive-crop-only inspection. NEVER authenticates missing full source maps.

Original source path/hash verification remains a blocking live-check requirement.
This script compares preserved V4/V5 floats with experimental V6, not HR truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from tools.nsamdr.neural.benchmark_nsamdr_v178_continuity import check_candidate
from tools.nsamdr.neural.v178.contour_model import physical_edge_evidence
from tools.nsamdr.neural.v178.golden_patch import sha256,validate_native_evidence
from tools.nsamdr.neural.v178.real_fit import _input
from tools.nsamdr.neural.v178.auto_preview import _predict_tiled

NATIVE_DIR=ROOT/'artifacts/nsamdr/golden_patch'
ARCHIVED=ROOT/'artifacts/nsamdr/diagnostics/v178_contrast_transfer'
ROIS=('cb1_t1_a_000_x320_y448_s256','cb1_t1_a_000_x64_y64_s256',
      'cb1_t1_a_000_x640_y640_s256','cb2_t1_a_000_x320_y448_s256')


def panel(label,data,size):
    image=Image.fromarray(np.uint8(np.rint(np.clip(data,0,1)*255)))
    if image.mode != 'RGB':image=image.convert('RGB')
    return image.resize((size,size),Image.Resampling.NEAREST)


@torch.no_grad()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--candidate',type=Path,required=True)
 p.add_argument('--output',type=Path,required=True)
 args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 torch.set_num_threads(min(4,torch.get_num_threads()))
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 device=torch.device('cpu')
 model,identity=check_candidate(args.candidate.resolve(),ROOT,device)
 findings=[]
 for name in ROIS:
  folder=NATIVE_DIR/name
  evidence=validate_native_evidence(folder) # every registered frozen CROP hash
  a,n,m,alignment=_input(folder,evidence,device)
  m=torch.cat((physical_edge_evidence(a,n),m[:,1:]),dim=1)
  _,tiled=_predict_tiled(model,a,n,m)
  dense=model(a,n,m)['candidate_contour'].cpu()
  difference=float((dense-tiled).abs().max())
  candidate=tiled[0,0].numpy()
  original=a[0].permute(1,2,0).numpy()
  original=cv2.resize(original,(1024,1024),interpolation=cv2.INTER_NEAREST)
  previous={}
  for older in ('v4_seed17832','v5_seed17842'):
   registered=ARCHIVED/older/'native'/name/'native_float_maps.npz'
   with np.load(registered) as data:
    previous[older[:2]]=data['candidate'].copy()
  sheet=Image.new('RGB',(4*512,512+34),(20,23,28))
  draw=ImageDraw.Draw(sheet)
  for i,(label,pixels) in enumerate((('Original RGB',original),
                         ('V4 archived',previous['v4']),('V5 archived',previous['v5']),
                         ('V6 CPU',candidate))):
   draw.text((i*512+8,8),label,fill=(255,255,255))
   sheet.paste(panel(label,pixels,512),(i*512,34))
  sheet_path=args.output/(name+'_comparison.png')
  sheet.save(sheet_path)
  np.savez_compressed(args.output/(name+'_v6_float.npz'),candidate=candidate)
  record={'region':name,'evidenceSha256':sha256(folder/'evidence.json'),
          'registeredCropHashesVerified':True,
          'sourceFullMapHashVerified':False,
          'sourceFullMapIdentityFromArchive':evidence['identity']['sourceSha256'],
          'v6Mean':float(candidate.mean()),
          'v6ActiveFractionAt030':float((candidate>=.30).mean()),
          'denseTiledMaxDifferenceCpu':difference,
          'v4MeanArchived':float(previous['v4'].mean()),
          'v5MeanArchived':float(previous['v5'].mean()),
          'v4ActiveFractionArchived':float((previous['v4']>=.30).mean()),
          'v5ActiveFractionArchived':float((previous['v5']>=.30).mean()),
          'sheet':sheet_path.name,'nativeAlignment':alignment}
  findings.append(record)
  print(name,'v6_mean',record['v6Mean'],'v6_active',record['v6ActiveFractionAt030'],
        'cpu_tile_diff',difference,flush=True)
 report={'schema':'NSAMDR_V178_ARCHIVED_CROP_INSPECTION_V1',
         'diagnosticOnly':True,'realEveAccuracyProven':False,
         'productionPromotionAllowed':False,'fullSourceMapAuthenticationAvailable':False,
         'viewableOriginalSourceCropHashesVerified':True,
         'limitations':['Only sealed archived crop inputs, not original full EVE source images',
                        'Comparison shows contour fields, not reconstructed physical textures',
                        'V4/V5 archived floats were produced in CUDA environment; V6 here uses CPU',
                        'Neither clean real contours nor DX11 controlled renders are available'],
         'candidate':identity,'regions':findings}
 (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 return 0
if __name__=='__main__':raise SystemExit(main())
