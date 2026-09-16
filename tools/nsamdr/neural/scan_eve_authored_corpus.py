#!/usr/bin/env python3
"""CPU-only census of independent authored EVE ship texture authorities.

The census is intentionally index/header based.  It answers the data-authority
question without decoding thousands of textures or preparing training data:

* how many independent authored albedo+normal authorities exist;
* what native 4x supervision tiers are genuinely available;
* which ships/factions share each authority;
* which direct auxiliary texture resources are visible from the model index.

Material *semantics* are not claimed from this fast pass.  EVE material behaviour
can be resolved through SOF/material-area data rather than a single adjacent
``_pgs.dds`` file.  Therefore this report distinguishes direct auxiliary texture
presence from full semantic material resolution instead of reporting a misleading
"material complete = 0" result.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
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


SCHEMA = "NSAMDR_EVE_AUTHORED_CORPUS_CENSUS_V2"
DIRECT_AUXILIARY_ROLES = ("pgs", "material", "roughness", "glow")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _authoritative_rows(indexes: list[Path]) -> list[eve.ResourceRow]:
    ordered = sorted(indexes, key=lambda path: 0 if "prefetch" in path.name.lower() else 1)
    return eve.read_rows(ordered)


def _source_record(
    row: eve.ResourceRow,
    resfiles: Path,
    repository: authored_dataset.AuthoredTextureDatasetRepository,
) -> dict[str, Any]:
    source = resfiles / Path(row.hashed)
    if not source.is_file():
        raise RuntimeError(f"resource is not local: {row.logical} -> {source}")
    width, height, mip_count, format_name = repository.parse_dds_header(source)
    return {
        "logical": row.logical,
        "hashed": row.hashed,
        "indexFile": row.index_file,
        "width": int(width),
        "height": int(height),
        "mipCount": int(mip_count),
        "format": format_name,
    }


def _resolution_tier(albedo: dict[str, Any], normal: dict[str, Any]) -> int:
    """Return the largest square 4x supervision tier supported by both maps."""

    native_min = min(
        int(albedo["width"]),
        int(albedo["height"]),
        int(normal["width"]),
        int(normal["height"]),
    )
    if native_min >= 4096:
        return 4096
    if native_min >= 2048:
        return 2048
    if native_min >= 1024:
        return 1024
    if native_min >= 512:
        return 512
    return native_min


def _authority_id(albedo: str, normal: str) -> str:
    """Identity is authored albedo+normal authority, independent of SOF material."""

    payload = "|".join(
        value.strip().replace("\\", "/").casefold() for value in (albedo, normal)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _shape_key(record: dict[str, Any]) -> str:
    return f"{int(record['width'])}x{int(record['height'])}"


def _append_unique_source(
    target: dict[str, list[dict[str, Any]]], role: str, source: dict[str, Any]
) -> None:
    values = target.setdefault(role, [])
    logical = str(source.get("logical") or "").casefold()
    if logical and all(str(item.get("logical") or "").casefold() != logical for item in values):
        values.append(source)


def scan(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = repo_root / "artifacts/nsamdr/diagnostics/eve_census" / f"census_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    cache_root, indexes, resfiles = eve.resolve_layout(args.shared_cache, allow_prompt=False)
    print(f"[eve-census] SharedCache: {cache_root}", flush=True)
    print("[eve-census] Reading authoritative resource indexes...", flush=True)
    rows = _authoritative_rows(indexes)
    entries = eve._build_sde_ship_catalog(rows, repo_root)  # noqa: SLF001
    entries = [entry for entry in entries if str(entry.preferred_asset or "").strip()]

    # Multiple SDE entries can point at the same model. Scan each model once.
    by_asset: dict[str, list[eve.ShipCatalogEntry]] = defaultdict(list)
    for entry in entries:
        key = str(entry.preferred_asset).strip().replace("\\", "/").casefold()
        by_asset[key].append(entry)
    models = sorted(by_asset.items(), key=lambda item: item[0])
    if args.limit > 0:
        models = models[: int(args.limit)]

    dds_repository = authored_dataset.AuthoredTextureDatasetRepository()
    # Deliberately dedupe by albedo+normal only.  SOF/material appearance can vary
    # while the underlying authored geometry authority remains the same.
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[dict[str, str]] = []

    for index, (asset_key, owners) in enumerate(models, start=1):
        display = owners[0].display_name
        if index == 1 or index % 50 == 0 or index == len(models):
            print(f"[eve-census] {index}/{len(models)} {display}", flush=True)
        try:
            model = eve.select_model(rows, owners[0].preferred_asset)
            textures = eve.related_textures(rows, model)
            albedo_row = textures.get("albedo")
            normal_row = textures.get("normal")
            if albedo_row is None or normal_row is None:
                raise RuntimeError("no aligned authoritative albedo+normal pair")
            albedo = _source_record(albedo_row, resfiles, dds_repository)
            normal = _source_record(normal_row, resfiles, dds_repository)

            identity = (
                str(albedo["logical"]).casefold(),
                str(normal["logical"]).casefold(),
            )
            if identity not in unique:
                unique[identity] = {
                    "authorityId": _authority_id(*identity),
                    "modelLogical": model.logical,
                    "resolutionTier": _resolution_tier(albedo, normal),
                    "albedo": albedo,
                    "normal": normal,
                    "directAuxiliaryTextures": {},
                    "materialSemantics": {
                        "status": "not-evaluated-index-only",
                        "reason": (
                            "Full EVE material behaviour may be defined by SOF/material-area "
                            "semantics rather than one direct model-adjacent texture."
                        ),
                    },
                    "ships": [],
                }
            record = unique[identity]

            auxiliaries = record["directAuxiliaryTextures"]
            for role in DIRECT_AUXILIARY_ROLES:
                row = textures.get(role)
                if row is None:
                    continue
                try:
                    _append_unique_source(
                        auxiliaries,
                        role,
                        _source_record(row, resfiles, dds_repository),
                    )
                except RuntimeError:
                    continue

            known = {str(item.get("canonicalKey")) for item in record["ships"]}
            for owner in owners:
                if str(owner.canonical_key) in known:
                    continue
                record["ships"].append(
                    {
                        "displayName": owner.display_name,
                        "canonicalKey": owner.canonical_key,
                        "groupName": owner.group_name,
                        "factionName": owner.faction_name,
                        "preferredAsset": owner.preferred_asset,
                    }
                )
        except Exception as exc:
            failures.append(
                {
                    "asset": asset_key,
                    "displayName": display,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    authorities = sorted(
        unique.values(),
        key=lambda value: (-int(value["resolutionTier"]), str(value["authorityId"])),
    )
    tier_counts = Counter(int(value["resolutionTier"]) for value in authorities)
    albedo_shapes = Counter(_shape_key(value["albedo"]) for value in authorities)
    normal_shapes = Counter(_shape_key(value["normal"]) for value in authorities)
    faction_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    for authority in authorities:
        for ship in authority["ships"]:
            faction_counts[str(ship.get("factionName") or "unknown")] += 1
            group_counts[str(ship.get("groupName") or "unknown")] += 1

    direct_auxiliary_count = sum(
        1 for item in authorities if any(item["directAuxiliaryTextures"].values())
    )
    any_4096_axis_count = sum(
        1
        for item in authorities
        if max(
            int(item["albedo"]["width"]),
            int(item["albedo"]["height"]),
            int(item["normal"]["width"]),
            int(item["normal"]["height"]),
        )
        >= 4096
    )

    report = {
        "schema": SCHEMA,
        "scanMethod": "full-sde-ship-catalog-plus-resource-index-plus-dds-headers",
        "cpuOnly": True,
        "datasetMutation": False,
        "materialSemanticResolutionPerformed": False,
        "materialSemanticCompleteAuthorityCount": None,
        "candidateModelCount": len(models),
        "uniqueAuthoredAuthorityCount": len(authorities),
        "authorityIdentity": "aligned-authored-albedo-plus-normal-logical-resource-pair",
        "authorityCountsByResolutionTier": {
            str(key): int(value) for key, value in sorted(tier_counts.items(), reverse=True)
        },
        "albedoAuthorityShapes": dict(albedo_shapes.most_common()),
        "normalAuthorityShapes": dict(normal_shapes.most_common()),
        "native4096OrHigherCount": sum(
            1 for item in authorities if int(item["resolutionTier"]) >= 4096
        ),
        "native2048OrHigherCount": sum(
            1 for item in authorities if int(item["resolutionTier"]) >= 2048
        ),
        "native1024OrHigherCount": sum(
            1 for item in authorities if int(item["resolutionTier"]) >= 1024
        ),
        "authoritiesWithAny4096AxisCount": any_4096_axis_count,
        "directAuxiliaryTextureAuthorityCount": direct_auxiliary_count,
        "shipCountByFaction": dict(faction_counts.most_common()),
        "shipCountByGroup": dict(group_counts.most_common()),
        "authorities": authorities,
        "failures": failures,
    }
    report_path = run_dir / "eve_authored_corpus_census.json"
    _atomic_json(report_path, report)

    summary_lines = [
        "NSAMDR EVE AUTHORED TEXTURE CORPUS CENSUS",
        "=" * 76,
        f"Candidate ship models       : {len(models)}",
        f"Unique albedo+normal auth.  : {len(authorities)}",
        f"Native >=4096 authorities  : {report['native4096OrHigherCount']}",
        f"Native >=2048 authorities  : {report['native2048OrHigherCount']}",
        f"Native >=1024 authorities  : {report['native1024OrHigherCount']}",
        f"Any 4096-axis authorities  : {report['authoritiesWithAny4096AxisCount']}",
        f"Direct auxiliary textures  : {report['directAuxiliaryTextureAuthorityCount']}",
        "Material semantic complete  : NOT EVALUATED (requires SOF/material-area resolution)",
        f"Scan failures              : {len(failures)}",
        "",
        "Resolution tiers",
        "-" * 76,
    ]
    for tier, count in sorted(tier_counts.items(), reverse=True):
        summary_lines.append(f"{tier:>6}px : {count}")
    summary_lines.extend(
        (
            "",
            "Production relevance",
            "-" * 76,
            "The tier uses the minimum aligned albedo/normal dimension.",
            "A 4096x2048 texture is therefore a genuine 2048-tier authority, not 4K-square truth.",
            "Native >=4096 authorities can provide genuine 1024->4096 supervision.",
            "Native >=2048 authorities can provide genuine 512->2048 supervision.",
            "Native >=1024 authorities can provide genuine 256->1024 supervision.",
            "Direct auxiliary texture presence is telemetry only; it is not material-semantic completeness.",
            "",
            f"JSON: {report_path}",
        )
    )
    summary_path = run_dir / "eve_authored_corpus_census_summary.txt"
    _atomic_text(summary_path, "\n".join(summary_lines) + "\n")
    latest = run_dir.parent / "LATEST.txt"
    _atomic_text(latest, str(run_dir) + "\n")
    print("\n".join(summary_lines), flush=True)
    return 0, report_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Census independent authored EVE ship albedo+normal authorities"
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    value.add_argument(
        "--limit", type=int, default=0, help="0 scans the complete ship catalog"
    )
    return value


def main(argv: list[str] | None = None) -> int:
    code, _report = scan(parser().parse_args(argv))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
