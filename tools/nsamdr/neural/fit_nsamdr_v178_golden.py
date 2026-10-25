#!/usr/bin/env python3
"""G2 single authentic Raven-patch contour capacity fit (not generalisation)."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.nsamdr.neural.v178.real_fit import fit


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",required=True,type=Path)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--steps",type=int,default=300)
    p.add_argument("--patch-size",type=int,default=48)
    p.add_argument("--seed",type=int,default=17803)
    p.add_argument("--resume",choices=("fresh","auto"),default="fresh")
    return p


def main(argv:list[str]|None=None)->int:
    args=parser().parse_args(argv)
    report=fit(args.folder,device_name=args.device,steps=args.steps,
               patch_size=args.patch_size,seed=args.seed,resume=args.resume)
    return 0 if report["decision"]=="SINGLE_PATCH_CAPACITY_PASS_ONLY" else 2


if __name__=="__main__":
    raise SystemExit(main())
