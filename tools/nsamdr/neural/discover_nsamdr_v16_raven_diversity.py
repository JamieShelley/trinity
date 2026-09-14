#!/usr/bin/env python3
"""Discover usable native authored Raven texture families for V16 diagnostics.

This is a CPU-only data-authority scan. It reads the EVE resource indexes and DDS
headers directly. It deliberately does not prepare render assets, convert GR2,
convert DDS files, enumerate environments, or modify the canonical V16 dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
NSAMDR_ROOT = HERE.parent
if str(NSAMDR_ROOT) not in sys.path:
    sys.path.insert(0, str(NSAMDR_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import authored_texture_dataset as authored_dataset
import eve_asset_test as eve  # type: ignore
import prepare_nsamdr_v9_raven_preview_dataset as legacy
from v14.config import V16Config
from v14.diagnostic_support import archive_run, make_run_directory


REPORT_SCHEMA = "NSAMDR_V16_RAVEN_DIVERSITY_SCAN_V2_INDEX_ONLY"


def _asset_path(value: str | None) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def _candidate_entries(
    entries: list[eve.ShipCatalogEntry],
    anchor: eve.ShipCatalogEntry,
    limit: int,
) -> list[eve.ShipCatalogEntry]:
    anchor_asset = _asset_path(anchor.preferred_asset)
    anchor_parent = str(PurePosixPath(anchor_asset).parent) if anchor_asset else ""

    selected: list[eve.ShipCatalogEntry] = []
    seen: set[str] = set()
    for entry in entries:
        name_match = "raven" in entry.display_name.casefold()
        asset = _asset_path(entry.preferred_asset)
        hull_match = bool(
            anchor_parent
            and asset
            and str(PurePosixPath(asset).parent) == anchor_parent
        )
        if not (name_match or hull_match):
            continue
        identity = str(entry.canonical_key).casefold()
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(entry)

    if str(anchor.canonical_key).casefold() not in seen:
        selected.append(anchor)

    selected.sort(
        key=lambda entry: (
            0 if entry.canonical_key == anchor.canonical_key else 1,
            0 if "raven" in entry.display_name.casefold() else 1,
            entry.display_name.casefold(),
            str(entry.canonical_key).casefold(),
        )
    )
    return selected[: max(1, int(limit))]


def _source_record(
    row: eve.ResourceRow,
    resfiles: Path,
    repository: authored_dataset.AuthoredTextureDatasetRepository,
) -> dict[str, Any]:
    path = resfiles / Path(row.hashed)
    if not path.is_file():
        raise RuntimeError(f"native EVE resource is not local: {row.logical} -> {path}")
    width, height, mip_count, format_name = repository.parse_dds_header(path)
    return {
        "logical": row.logical,
        "hashed": row.hashed,
        "indexFile": row.index_file,
        "width": int(width),
        "height": int(height),
        "mipCount": int(mip_count),
        "format": format_name,
        "authority": "eve-full-index-native-semantic",
    }


def scan(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    run_dir = make_run_directory(repo_root, "diversity")

    builder = legacy.RavenPreviewDatasetPreparationApplication()
    cache_root, indexes, resfiles = eve.resolve_layout(
        args.shared_cache,
        allow_prompt=False,
    )
    print(f"[v16-diversity] Run: {run_dir}", flush=True)
    print(f"[v16-diversity] EVE SharedCache: {cache_root}", flush=True)
    print("[v16-diversity] Reading resource indexes once...", flush=True)
    rows = builder._authoritative_rows(indexes)
    print(f"[v16-diversity] Indexed resources: {len(rows)}", flush=True)

    entries = eve._build_sde_ship_catalog(rows, repo_root)  # noqa: SLF001
    anchor = builder._find_navy_raven(rows, repo_root)
    candidates = _candidate_entries(entries, anchor, args.limit)
    crop_size = int(V16Config().train_hr_size)
    dds_repository = authored_dataset.AuthoredTextureDatasetRepository()

    print(
        f"[v16-diversity] Candidate Raven/hull entries: {len(candidates)} "
        f"(limit={args.limit})",
        flush=True,
    )
    print(
        "[v16-diversity] Fast path: index + DDS headers only; "
        "no GR2, PNG, material, or environment conversion.",
        flush=True,
    )

    assets: list[dict[str, Any]] = []
    unique_families: dict[tuple[str, str], dict[str, Any]] = {}

    for index, entry in enumerate(candidates, start=1):
        prefix = f"[v16-diversity] {index}/{len(candidates)}"
        print(
            f"{prefix} {entry.display_name} ({entry.canonical_key})",
            flush=True,
        )
        record: dict[str, Any] = {
            "displayName": entry.display_name,
            "canonicalKey": entry.canonical_key,
            "preferredAsset": entry.preferred_asset,
            "groupName": entry.group_name,
            "factionName": entry.faction_name,
            "status": "error",
            "family": None,
        }
        try:
            model = eve.select_model(
                rows,
                entry.preferred_asset or legacy.DEFAULT_TUNING_ASSET_QUERY,
            )
            textures = eve.related_textures(rows, model)
            albedo_row = textures.get("albedo")
            normal_row = textures.get("normal")
            if albedo_row is None or normal_row is None:
                raise RuntimeError(
                    "candidate has no authoritative aligned albedo+normal texture pair"
                )

            albedo = _source_record(albedo_row, resfiles, dds_repository)
            normal = _source_record(normal_row, resfiles, dds_repository)
            if min(int(albedo["width"]), int(albedo["height"])) < crop_size:
                raise RuntimeError(
                    f"albedo is smaller than the required {crop_size}px crop: "
                    f"{albedo['width']}x{albedo['height']}"
                )
            if min(int(normal["width"]), int(normal["height"])) < crop_size:
                raise RuntimeError(
                    f"normal is smaller than the required {crop_size}px crop: "
                    f"{normal['width']}x{normal['height']}"
                )

            sources: dict[str, Any] = {
                "albedo": albedo,
                "normal": normal,
            }
            for role in ("pgs", "material"):
                row = textures.get(role)
                if row is not None:
                    try:
                        sources[role] = _source_record(row, resfiles, dds_repository)
                    except RuntimeError as exc:
                        sources[role] = {"logical": row.logical, "error": str(exc)}

            identity = (
                str(albedo["logical"]).casefold(),
                str(normal["logical"]).casefold(),
            )
            family = {
                "identity": list(identity),
                "modelLogical": model.logical,
                "sources": sources,
            }
            record["status"] = "ok"
            record["family"] = family
            is_new = identity not in unique_families
            if is_new:
                unique_families[identity] = {
                    **family,
                    "firstSeenOn": entry.display_name,
                    "canonicalKey": entry.canonical_key,
                }
            print(
                f"{prefix} OK family={'NEW' if is_new else 'duplicate'} "
                f"albedo={albedo['width']}x{albedo['height']} "
                f"normal={normal['width']}x{normal['height']} "
                f"unique={len(unique_families)}",
                flush=True,
            )
        except Exception as exc:  # diagnostic scan reports unusable variants and continues
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{prefix} SKIP {record['error']}", flush=True)
        assets.append(record)

    usable_assets = [asset for asset in assets if asset.get("status") == "ok"]
    passed = len(unique_families) >= 2
    report = {
        "schema": REPORT_SCHEMA,
        "revision": "V16.0",
        "mode": "diversity",
        "passed": passed,
        "scanMethod": "eve-full-index-and-dds-headers-only",
        "anchor": {
            "displayName": anchor.display_name,
            "canonicalKey": anchor.canonical_key,
            "preferredAsset": anchor.preferred_asset,
        },
        "candidateCount": len(candidates),
        "usableAssetCount": len(usable_assets),
        "uniqueAuthoredFamilyCount": len(unique_families),
        "minimumRecommendedFamilies": 2,
        "sufficientForBroaderStage2": passed,
        "assets": assets,
        "uniqueFamilies": list(unique_families.values()),
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    archive_path = archive_run(run_dir)

    print(f"[v16-diversity] Usable assets: {len(usable_assets)}", flush=True)
    print(
        f"[v16-diversity] Unique authored families: {len(unique_families)}",
        flush=True,
    )
    print(
        f"[v16-diversity] Broader Stage 2: "
        f"{'SUPPORTED' if passed else 'INSUFFICIENT'}",
        flush=True,
    )
    print(f"[v16-diversity] Report: {report_path}", flush=True)
    print(f"[v16-diversity] Diagnostics ZIP: {archive_path}", flush=True)
    return (0 if passed else 2), report_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Discover authored Raven family diversity for V16 Stage 2"
    )
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    p.add_argument("--limit", type=int, default=8)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    code, _report = scan(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
