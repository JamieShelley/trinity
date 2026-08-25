#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from v9.config import V9Config
from v9.inference import load_trained_model
from v9.model import MODEL_SCHEMA,parameter_count

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument('--repo-root',type=Path,default=Path.cwd()); p.add_argument('--config',type=Path); a=p.parse_args()
    cfg=V9Config.load(a.config.resolve() if a.config else None); root=a.repo_root.resolve(); checkpoint=root/cfg.output_dir/cfg.checkpoint_name; metadata=root/cfg.output_dir/cfg.metadata_name
    if not checkpoint.is_file() or not metadata.is_file(): raise SystemExit(f'missing V9 checkpoint or metadata under {checkpoint.parent}')
    model,loaded,payload=load_trained_model(checkpoint,'cpu')
    if payload.get('schema')!=MODEL_SCHEMA: raise SystemExit('V9 schema mismatch')
    x=torch.zeros(1,17,32,32)
    with torch.no_grad(): y=model(x)
    if tuple(y['albedo'].shape[-2:])!=(128,128): raise SystemExit('V9 checkpoint does not perform 4x reconstruction')
    meta=json.loads(metadata.read_text(encoding='utf-8'))
    print('NSAMDR V9 checkpoint validation passed'); print(f'  parameters={parameter_count(model):,}'); print(f'  bestValidation={meta.get("bestValidationTotal")}'); print(f'  acceptancePass={meta.get("acceptancePass")}'); print(f'  regressionFraction={meta.get("acceptanceRegressionFraction")}'); print(f'  checkpoint={checkpoint}')
    return 0
if __name__=='__main__': raise SystemExit(main())
