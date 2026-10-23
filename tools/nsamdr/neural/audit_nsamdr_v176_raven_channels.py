#!/usr/bin/env python3
"""Inspect native EVE Raven physical channels for texture-edge stair-stepping.

This diagnostic never launches a model, trains, or changes the source.
It compares nearest and bilinear interpolation in selected native ROI panels.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for directory in (ROOT, HERE, HERE.parent):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))

import eve_asset_test as eve  # type: ignore
from tools.nsamdr.neural.v176.channel_audit import run
from tools.nsamdr.neural.v176.raven_renderer import DEFAULT_RAVEN


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    p.add_argument("--ship-query", default=DEFAULT_RAVEN)
    p.add_argument("--crop-size", type=int, default=128)
    p.add_argument("--max-textures", type=int, default=6)
    p.add_argument("--crop-x", type=int)
    p.add_argument("--crop-y", type=int)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (args.crop_x is None) != (args.crop_y is None):
        raise ValueError("specify both --crop-x and --crop-y, or neither")
    repo = args.repo_root.resolve()
    prepared = eve.prepare_asset(
        repo, args.shared_cache, args.ship_query, "",
    )
    materials = prepared[6]
    if materials is None or not materials.is_file():
        raise RuntimeError("prepared native EVE ship has no material manifest")
    folder = repo / "artifacts/nsamdr/diagnostics/v176_raven_channel_audit" / (
        "audit_"+datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    result = run(materials, folder, crop_size=args.crop_size,
                 max_textures=args.max_textures,
                 origin=((args.crop_x,args.crop_y)
                         if args.crop_x is not None else None))
    print("[v176-channels] Native pixels retained. No model weights updated.",flush=True)
    for entry in result["textures"]:
        print(f"[v176-channels] {Path(entry['source']).name} "
              f"native={entry['sourceNativeDimensions']} "
              f"region={entry['roiOriginNative']} "
              f"channels={len(entry['channels'])}",flush=True)
    print(f"[v176-channels] report={folder/'report.json'}",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
