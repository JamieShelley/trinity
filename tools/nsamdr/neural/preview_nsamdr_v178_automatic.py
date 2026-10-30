#!/usr/bin/env python3
"""NSAMDR real Raven automatic Golden Patch inference: no trace or pseudo-label."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from tools.nsamdr.neural.v178.auto_preview import run


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,default=ROOT/
        "artifacts/nsamdr/diagnostics/v178_golden_contour/"
        "seed_17842_patch_32_batch_4/candidate.pt")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--tile",type=int,default=96)
    return p


def main(argv:list[str]|None=None)->int:
    args=parser().parse_args(argv)
    report=run(args.folder,args.checkpoint,device_name=args.device,tile=args.tile)
    # Do not let the GUI report ordinary success for an empty all-black mask.
    return 3 if report["realContourCollapsed"] else 0


if __name__=="__main__":
    raise SystemExit(main())
