#!/usr/bin/env python3
"""NSAMDR G0–G2: prepare real Raven evidence or render curated contour reference.

Preparation captures native physical pixels and provenance. Rendering a target
requires an independent human trace.json; a template is NOT a reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for item in (ROOT, HERE):
    if str(item) not in sys.path:
        sys.path.insert(0,str(item))

from tools.nsamdr.neural.v178.golden_patch import create, render_contour_trace
from tools.nsamdr.neural.v176.raven_renderer import DEFAULT_RAVEN


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root",type=Path,default=ROOT)
    p.add_argument("--action",choices=("prepare","render-trace"),default="prepare")
    p.add_argument("--shared-cache",default=r"C:\CCP\EVE")
    p.add_argument("--source-name",default="")
    p.add_argument("--crop-x",type=int,default=320)
    p.add_argument("--crop-y",type=int,default=448)
    p.add_argument("--size",type=int,default=256)
    p.add_argument("--folder",type=Path)
    return p


def main(argv:list[str]|None=None)->int:
    args=parser().parse_args(argv)
    repo=args.repo_root.resolve()
    if args.action=="render-trace":
        if args.folder is None:
            raise ValueError("--folder is required for render-trace")
        result=render_contour_trace(args.folder.resolve())
        print(f"[v178-golden] INDEPENDENT_TRACE_RENDERED {result}",flush=True)
        return 0
    from tools.nsamdr import eve_asset_test as eve
    prepared=eve.prepare_asset(repo,args.shared_cache,DEFAULT_RAVEN,"")
    manifest=prepared[6]
    if manifest is None or not manifest.is_file():
        raise RuntimeError("prepared Raven lacks a valid physical-map manifest")
    report=create(manifest,
        repo/"artifacts/nsamdr/golden_patch",
        crop_x=args.crop_x,crop_y=args.crop_y,size=args.size,
        source_name=args.source_name,
    )
    print(f"[v178-golden] decision={report['decision']}",flush=True)
    print(f"[v178-golden] original={report['identity']['source']}",flush=True)
    print(f"[v178-golden] template={report['referenceTemplate']}",flush=True)
    print("[v178-golden] Trace independent contours in trace.json "
          "before invoking --action render-trace. "
          "Do not treat the authored raster as its own clean reference.",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
