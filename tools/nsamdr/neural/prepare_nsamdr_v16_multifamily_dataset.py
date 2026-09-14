#!/usr/bin/env python3
"""Build the V16 Stage 2 dataset from multiple authored Raven hull families.

This builder is intentionally Stage-2-specific.  Capacity keeps the existing
single-family Raven dataset.  Multi-Region uses at least two distinct native
albedo+normal families and applies the same hard pixel-disjoint spatial split
independently inside each family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
NSAMDR_ROOT = HERE.parent
if str(NSAMDR_ROOT) not in sys.path:
    sys.path.insert(0, str(NSAMDR_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eve_asset_test as eve  # type: ignore
import prepare_nsamdr_v16_raven_dataset as spatial
from v9.config import V9Config


DATASET_SCHEMA = "NSAMDR_RAVEN_DEVELOPMENT_DATASET_V6_MULTI_FAMILY_SPATIAL_DOMAIN_DISJOINT"
CROP_SCHEMA = "NSAMDR_RAVEN_DEVELOPMENT_CROP_V6_MULTI_FAMILY_SPATIAL_DOMAIN_DISJOINT"
BUILDER_VERSION = "raven-native-authored-multi-family-spatial-domain-v6"
MINIMUM_AUTHORED_FAMILIES = 2


def _asset_path(value: str | None) -> str:
    return str(value or "").strip().replace("\\", "/").lower()


def _candidate_entries(
    entries: list[eve.ShipCatalogEntry],
    anchor: eve.ShipCatalogEntry,
) -> list[eve.ShipCatalogEntry]:
    anchor_asset = _asset_path(anchor.preferred_asset)
    anchor_parent = str(PurePosixPath(anchor_asset).parent) if anchor_asset else ""
    selected: list[eve.ShipCatalogEntry] = []
    seen: set[str] = set()
    for entry in entries:
        asset = _asset_path(entry.preferred_asset)
        name_match = "raven" in entry.display_name.casefold()
        hull_match = bool(
            anchor_parent
            and asset
            and str(PurePosixPath(asset).parent) == anchor_parent
        )
        if not (name_match or hull_match):
            continue
        key = str(entry.canonical_key).casefold()
        if key in seen:
            continue
        seen.add(key)
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
    return selected


def _texture_pair_identity(
    rows: list[eve.ResourceRow],
    entry: eve.ShipCatalogEntry,
) -> tuple[str, str] | None:
    model = eve.select_model(rows, entry.preferred_asset)
    textures = eve.related_textures(rows, model)
    albedo = textures.get("albedo")
    normal = textures.get("normal")
    if albedo is None or normal is None:
        return None
    return (albedo.logical.casefold(), normal.logical.casefold())


def _select_distinct_assets(
    app: spatial.V16RavenDatasetPreparationApplication,
    rows: list[eve.ResourceRow],
    repo_root: Path,
) -> list[eve.ShipCatalogEntry]:
    entries = eve._build_sde_ship_catalog(rows, repo_root)  # noqa: SLF001
    anchor = app._find_navy_raven(rows, repo_root)
    selected: list[eve.ShipCatalogEntry] = []
    seen_pairs: set[tuple[str, str]] = set()
    for entry in _candidate_entries(entries, anchor):
        try:
            pair = _texture_pair_identity(rows, entry)
        except Exception:
            continue
        if pair is None or pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        selected.append(entry)
        if len(selected) >= MINIMUM_AUTHORED_FAMILIES:
            break
    if len(selected) < MINIMUM_AUTHORED_FAMILIES:
        raise RuntimeError(
            "V16 Stage 2 requires at least two distinct authored Raven hull "
            f"albedo+normal families; found {len(selected)}"
        )
    return selected


def _prepare_asset_without_environments(
    repo_root: Path,
    rows: list[eve.ResourceRow],
    resfiles: Path,
    entry: eve.ShipCatalogEntry,
) -> tuple[dict[str, Any], Path]:
    """Prepare only geometry/SOF/material data required to resolve training maps."""
    model = eve.select_model(rows, entry.preferred_asset)
    textures = eve.related_textures(rows, model)
    asset_name = Path(model.logical.rsplit("/", 1)[-1]).stem
    output_dir = repo_root / "artifacts" / "nsamdr" / "eve_assets" / asset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    gr2_path = eve.copy_resource(resfiles, model, output_dir)
    obj_path = eve.convert_gr2(repo_root, gr2_path, output_dir / f"{asset_name}.obj")
    conversion_summary = obj_path.with_suffix(".conversion.json")

    copied: dict[str, Path] = {}
    converted: dict[str, Path] = {}
    for role, row in textures.items():
        copied[role] = eve.copy_resource(resfiles, row, output_dir)
        try:
            converted[role] = eve.convert_dds(
                repo_root,
                copied[role],
                output_dir / f"{asset_name}_{role}.png",
            )
        except RuntimeError as exc:
            print(
                f"[v16-multifamily] WARNING: {entry.display_name} {role} "
                f"conversion failed: {exc}",
                flush=True,
            )

    sof_identity = eve._resolve_sof_identity(  # noqa: SLF001
        rows,
        repo_root,
        model,
        entry.canonical_key,
    )
    material_manifest: Path | None = None
    sof_manifest_path: Path | None = None
    sof_texture_metadata: dict[str, dict] = {}
    data_black_row = next(
        (row for row in rows if row.logical.lower() == eve.SOF_DATA_PATH),
        None,
    )
    fallback_reason = ""
    if data_black_row and sof_identity.get("hull") and sof_identity.get("faction"):
        try:
            data_black = eve.copy_resource(resfiles, data_black_row, output_dir / "sof")
            sof_manifest_path = eve.convert_sof(
                repo_root,
                data_black,
                output_dir / f"{asset_name}.stage2.sof-visuals.json",
                str(sof_identity["hull"]),
                str(sof_identity["faction"]),
                str(sof_identity.get("race") or ""),
            )
            material_manifest, sof_texture_metadata = eve._prepare_sof_materials(  # noqa: SLF001
                repo_root,
                rows,
                resfiles,
                output_dir,
                conversion_summary,
                sof_manifest_path,
                str(sof_identity.get("race") or ""),
            )
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            fallback_reason = f"SOF extraction failed: {exc}"
    else:
        fallback_reason = "SOF identity or data.black unavailable"

    if material_manifest is None:
        material_manifest = eve._write_tint_only_material_manifest(  # noqa: SLF001
            output_dir,
            conversion_summary,
            str(sof_identity.get("race") or ""),
            fallback_reason or "SOF visual data unavailable",
        )

    manifest = {
        "model": {"logical": model.logical, "local": str(gr2_path)},
        "textures": {
            role: {
                "logical": textures[role].logical,
                "hashed": textures[role].hashed,
                "indexFile": textures[role].index_file,
                "local": str(path),
                "converted": str(converted[role]) if role in converted else None,
            }
            for role, path in copied.items()
        },
        "obj": str(obj_path),
        "conversionSummary": str(conversion_summary),
        "sofIdentity": sof_identity,
        "sofVisualManifest": str(sof_manifest_path) if sof_manifest_path else None,
        "materialManifest": str(material_manifest),
        "materialBaselineReport": str(output_dir / "ship.materials.report.json"),
        "sofTextures": sof_texture_metadata,
        "albedoPng": str(converted.get("albedo")) if converted.get("albedo") else None,
        "normalPng": str(converted.get("normal")) if converted.get("normal") else None,
        "pgsPng": str(converted.get("pgs")) if converted.get("pgs") else None,
    }
    manifest_path = output_dir / "stage2_asset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest, manifest_path


def _family_identity(family: dict[str, Any]) -> tuple[str, str]:
    provenance = dict(family.get("sourceProvenance") or {})
    albedo = provenance.get("albedo") if isinstance(provenance, dict) else None
    normal = provenance.get("normal") if isinstance(provenance, dict) else None
    return (
        str(albedo.get("logical") or "").casefold() if isinstance(albedo, dict) else "",
        str(normal.get("logical") or "").casefold() if isinstance(normal, dict) else "",
    )


def _auxiliary_score(family: dict[str, Any]) -> int:
    provenance = dict(family.get("sourceProvenance") or {})
    return sum(
        1
        for role in ("material", "roughnessMap", "glow")
        if isinstance(provenance.get(role), dict)
        and str(provenance[role].get("logical") or "")
    )


def _dedupe_native_families(families: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    for family in families:
        identity = _family_identity(family)
        if not identity[0] or not identity[1]:
            continue
        existing = chosen.get(identity)
        if existing is None or _auxiliary_score(family) > _auxiliary_score(existing):
            chosen[identity] = family
    return list(chosen.values())


def _source_fingerprint(
    app: spatial.V16RavenDatasetPreparationApplication,
    families: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> str:
    payload: dict[str, Any] = {
        "builder": BUILDER_VERSION,
        "assets": assets,
        "families": [],
    }
    for family in families:
        item: dict[str, Any] = {
            "sourceAsset": family.get("sourceAsset"),
            "sourceProvenance": family.get("sourceProvenance", {}),
            "channels": family.get("channels", {}),
            "files": {},
        }
        for role in ("albedo", "normal", "material", "roughnessMap", "glow"):
            raw = str(family.get(role) or "")
            path = Path(raw) if raw else None
            item["files"][role] = {
                "path": raw,
                "size": path.stat().st_size if path and path.is_file() else 0,
                "sha256": app._sha256_file(path) if path and path.is_file() else "",
            }
        payload["families"].append(item)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def prepare(
    repo_root: Path,
    config: V9Config,
    *,
    shared_cache: str,
    rebuild: bool,
    train_crops: int,
    validation_crops: int,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    app = spatial.V16RavenDatasetPreparationApplication()
    output_root = (repo_root / config.dataset_root).resolve()
    manifest_path = (repo_root / config.dataset_manifest).resolve()
    crop_root = output_root / "crops"
    crop_size = int(config.source_crop_size)
    if crop_size != config.tile_size * config.target_scale:
        raise RuntimeError(
            "Stage 2 source crop size must equal the model HR target size"
        )

    cache_root, indexes, resfiles = eve.resolve_layout(shared_cache, allow_prompt=False)
    print(f"[v16-multifamily] EVE SharedCache: {cache_root}", flush=True)
    rows = app._authoritative_rows(indexes)
    selected_assets = _select_distinct_assets(app, rows, repo_root)
    print(
        "[v16-multifamily] Authored asset families: "
        + ", ".join(entry.display_name for entry in selected_assets),
        flush=True,
    )

    native_families: list[dict[str, Any]] = []
    asset_payloads: list[dict[str, Any]] = []
    for entry in selected_assets:
        print(
            f"[v16-multifamily] Preparing {entry.display_name} ({entry.canonical_key})",
            flush=True,
        )
        asset_manifest, asset_manifest_path = _prepare_asset_without_environments(
            repo_root,
            rows,
            resfiles,
            entry,
        )
        report_path = Path(str(asset_manifest.get("materialBaselineReport") or ""))
        families = app._texture_families_from_report(report_path, asset_manifest)
        families, hull_probe = app._native_family_sources(
            repo_root,
            output_root,
            asset_manifest,
            families,
            rows,
            resfiles,
            crop_size,
        )
        families = _dedupe_native_families(families)
        if not families:
            raise RuntimeError(
                f"{entry.display_name} produced no native authored albedo+normal family"
            )
        # One albedo+normal authority family per selected hull.  This prevents the
        # same authored pixels from receiving different train/validation sides via
        # multiple SOF material-area records.
        family = max(families, key=_auxiliary_score)
        family["sourceAsset"] = {
            "displayName": entry.display_name,
            "typeId": entry.type_id,
            "selectionKey": entry.canonical_key,
            "preferredAsset": entry.preferred_asset,
            "assetManifest": str(asset_manifest_path.resolve()),
        }
        native_families.append(family)
        asset_payloads.append(
            {
                **family["sourceAsset"],
                "sofIdentity": asset_manifest.get("sofIdentity"),
                "hullAuthorityProbe": hull_probe,
            }
        )

    native_families = _dedupe_native_families(native_families)
    if len(native_families) < MINIMUM_AUTHORED_FAMILIES:
        raise RuntimeError(
            "Stage 2 family preparation collapsed below two distinct native "
            f"albedo+normal authorities: {len(native_families)}"
        )

    source_fingerprint = _source_fingerprint(app, native_families, asset_payloads)
    if not rebuild and manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            crop_paths = [Path(record["path"]) for record in existing.get("crops", [])]
            if (
                existing.get("schema") == DATASET_SCHEMA
                and existing.get("sourceFingerprint") == source_fingerprint
                and int(existing.get("splitPolicy", {}).get("maxTrainCrops", -1)) == int(train_crops)
                and int(existing.get("splitPolicy", {}).get("maxValidationCrops", -1)) == int(validation_crops)
                and int(existing.get("splitPolicy", {}).get("seed", -1)) == int(config.seed)
                and crop_paths
                and all(path.is_file() for path in crop_paths)
            ):
                print(
                    f"[v16-multifamily] Stage 2 dataset is current: {manifest_path}",
                    flush=True,
                )
                return existing
        except (OSError, ValueError, KeyError, TypeError):
            pass

    shutil.rmtree(crop_root, ignore_errors=True)
    (crop_root / "train").mkdir(parents=True, exist_ok=True)
    (crop_root / "validation").mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    family_payloads: list[dict[str, Any]] = []
    for family_index, family in enumerate(native_families):
        albedo_path = Path(str(family["albedo"]))
        normal_path = Path(str(family["normal"]))
        material_path = (
            Path(str(family.get("material") or "")) if family.get("material") else None
        )
        roughness_path = (
            Path(str(family.get("roughnessMap") or ""))
            if family.get("roughnessMap")
            else None
        )
        glow_path = (
            Path(str(family.get("glow") or "")) if family.get("glow") else None
        )
        albedo = app._load_rgb(albedo_path)
        normal, normal_encoding = app._normal_rgb(normal_path)
        h, w = albedo.shape[:2]
        normal_h, normal_w = normal.shape[:2]
        if min(w, h, normal_w, normal_h) < crop_size:
            raise RuntimeError(
                f"Stage 2 authored family is smaller than {crop_size}px: "
                f"albedo={w}x{h} normal={normal_w}x{normal_h}"
            )
        if normal.shape[:2] != (h, w):
            normal = app._resize_rgb(normal, w, h)

        channels = dict(family.get("channels") or {})
        material_source = material_path if material_path and material_path.is_file() else None
        roughness_source = (
            roughness_path if roughness_path and roughness_path.is_file() else material_source
        )
        glow_source = glow_path if glow_path and glow_path.is_file() else material_source
        material_valid = bool(material_source or roughness_source or glow_source)
        if material_valid:
            material_plane = app._semantic_plane(
                material_source,
                int(channels.get("material", 0)),
                w,
                h,
                0,
            )
            emissive_plane = app._semantic_plane(
                glow_source,
                int(channels.get("glow", 1)),
                w,
                h,
                0,
            )
            roughness_plane = app._semantic_plane(
                roughness_source,
                int(channels.get("roughness", 2)),
                w,
                h,
                128,
            )
            material = np.stack(
                (material_plane, emissive_plane, roughness_plane),
                axis=-1,
            )
        else:
            material = np.stack(
                (
                    np.zeros((h, w), dtype=np.uint8),
                    np.zeros((h, w), dtype=np.uint8),
                    np.full((h, w), 128, dtype=np.uint8),
                ),
                axis=-1,
            )

        source_asset = dict(family.get("sourceAsset") or {})
        family_id = hashlib.sha1(
            (
                f"{source_asset.get('selectionKey')}|{_family_identity(family)}|"
                f"{json.dumps(channels, sort_keys=True)}"
            ).encode("utf-8")
        ).hexdigest()[:16]
        xs = app._grid_positions(w, crop_size)
        ys = app._grid_positions(h, crop_size)
        cell_count = 0
        for gy, y in enumerate(ys):
            for gx, x in enumerate(xs):
                a = np.ascontiguousarray(albedo[y:y + crop_size, x:x + crop_size])
                n = np.ascontiguousarray(normal[y:y + crop_size, x:x + crop_size])
                m = np.ascontiguousarray(material[y:y + crop_size, x:x + crop_size])
                if a.shape[:2] != (crop_size, crop_size):
                    continue
                candidates.append(
                    {
                        "familyId": family_id,
                        "familyIndex": family_index,
                        "gx": gx,
                        "gy": gy,
                        "x": x,
                        "y": y,
                        "detailScore": app._detail_score(a, n, m),
                        "holdout": False,
                        "albedo": a,
                        "normal": n,
                        "material": m,
                        "materialValid": material_valid,
                        "normalEncoding": normal_encoding,
                        "family": family,
                    }
                )
                cell_count += 1

        provenance = dict(family.get("sourceProvenance") or {})
        family_payloads.append(
            {
                "familyId": family_id,
                "sourceAsset": source_asset,
                "areaName": family.get("areaName", ""),
                "areaType": family.get("areaType", ""),
                "shaderFamily": family.get("shaderFamily", ""),
                "albedo": str(albedo_path),
                "normal": str(normal_path),
                "material": str(material_path) if material_path else "",
                "roughnessMap": str(roughness_path) if roughness_path else "",
                "glow": str(glow_path) if glow_path else "",
                "channels": channels,
                "normalEncoding": normal_encoding,
                "materialSupervision": material_valid,
                "sourceAuthority": "native-eve-full-index-semantic",
                "sourceProvenance": provenance,
                "sourceSize": [w, h],
                "normalSourceSize": [normal_w, normal_h],
                "slidingWindowCandidates": cell_count,
            }
        )
        print(
            f"[v16-multifamily] Family {family_id}: "
            f"{source_asset.get('displayName')} albedo={w}x{h} "
            f"normal={normal_w}x{normal_h} material={'yes' if material_valid else 'no'}",
            flush=True,
        )

    selected_train, selected_validation = app._select_fixed_regions(
        candidates,
        max_train_crops=train_crops,
        max_validation_crops=validation_crops,
        seed=int(config.seed),
    )

    selected_cells = [
        *(dict(item, split="train") for item in selected_train),
        *(dict(item, split="validation") for item in selected_validation),
    ]
    records: list[dict[str, Any]] = []
    for item in selected_cells:
        split = str(item["split"])
        crop_id = f"{item['familyId']}_{item['gx']:02d}_{item['gy']:02d}_{split}"
        destination = crop_root / split / f"{crop_id}.npz"
        metadata = {
            "schema": CROP_SCHEMA,
            "cropId": crop_id,
            "familyId": item["familyId"],
            "split": split,
            "pixelBox": [
                item["x"],
                item["y"],
                item["x"] + crop_size,
                item["y"] + crop_size,
            ],
            "grid": [item["gx"], item["gy"]],
            "detailScore": item["detailScore"],
            "detailStratum": int(item["detailStratum"]),
            "fixed": True,
            "overlapBetweenTrainAndValidation": False,
            "normalEncoding": item["normalEncoding"],
            "materialSupervision": item["materialValid"],
            "sourceAuthority": "native-eve-full-index-semantic",
        }
        np.savez_compressed(
            destination,
            albedo=item["albedo"],
            normal=item["normal"],
            material=item["material"],
            material_valid=np.asarray(
                [1.0 if item["materialValid"] else 0.0],
                dtype=np.float32,
            ),
            metadata=np.asarray(json.dumps(metadata), dtype=np.str_),
        )
        records.append(
            {
                "crop_id": crop_id,
                "family_id": item["familyId"],
                "split": split,
                "path": str(destination.resolve()),
                "source_box": [
                    item["x"],
                    item["y"],
                    item["x"] + crop_size,
                    item["y"] + crop_size,
                ],
                "detail_score": item["detailScore"],
                "detail_stratum": int(item["detailStratum"]),
                "source_asset_key": str(
                    item["family"].get("sourceAsset", {}).get("selectionKey") or ""
                ),
                "source_asset_name": str(
                    item["family"].get("sourceAsset", {}).get("displayName") or ""
                ),
                "albedo_logical": str(
                    item["family"].get("sourceProvenance", {})
                    .get("albedo", {})
                    .get("logical")
                    or ""
                ),
                "normal_logical": str(
                    item["family"].get("sourceProvenance", {})
                    .get("normal", {})
                    .get("logical")
                    or ""
                ),
            }
        )

    selected_by_family: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        selected_by_family.setdefault(str(record["family_id"]), []).append(
            {
                "cropId": record["crop_id"],
                "split": record["split"],
                "sourceBox": record["source_box"],
            }
        )
    for family_payload in family_payloads:
        family_payload["selectedRegions"] = sorted(
            selected_by_family.get(str(family_payload["familyId"]), []),
            key=lambda item: (str(item["split"]), str(item["cropId"])),
        )
        domain = app._last_domains.get(str(family_payload["familyId"]), None)
        if domain is not None:
            family_payload["spatialDomain"] = domain

    fingerprint = app._selection_fingerprint(
        source_fingerprint,
        records,
        seed=config.seed,
    )
    payload: dict[str, Any] = {
        "schema": DATASET_SCHEMA,
        "sourceFingerprint": source_fingerprint,
        "fingerprint": fingerprint,
        "builderVersion": BUILDER_VERSION,
        "modelScope": "stage2-multifamily-diagnostic",
        "fixedPreviewSet": True,
        "deterministic": True,
        "authoredSourceAuthority": "native-eve-full-index-semantic",
        "authoredResolutionPolicy": "preserve-native-no-pretraining-upscale",
        "minimumAuthoredDimension": crop_size,
        "assets": asset_payloads,
        "splitPolicy": {
            "type": "multi-family-feature-stratified-spatial-domain-disjoint-v6",
            "detailStrata": 4,
            "maxTrainCrops": train_crops,
            "maxValidationCrops": validation_crops,
            "trainCrops": sum(1 for record in records if record["split"] == "train"),
            "validationCrops": sum(
                1 for record in records if record["split"] == "validation"
            ),
            "seed": config.seed,
            "augmentationSeed": config.seed,
            "validationSeed": config.seed + 77,
            "trainValidationPixelOverlap": False,
            "intraSplitWindowOverlapAllowed": True,
            "domainRule": "per-family-hard-axis-bisection",
            "minimumAuthoredFamilies": MINIMUM_AUTHORED_FAMILIES,
            "authoredFamilyCount": len(family_payloads),
            "spatialDomains": list(app._last_domains.values()),
        },
        "families": family_payloads,
        "crops": records,
        "counts": {
            "families": len(family_payloads),
            "trainCrops": sum(1 for record in records if record["split"] == "train"),
            "validationCrops": sum(
                1 for record in records if record["split"] == "validation"
            ),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 68, flush=True)
    print("NSAMDR V16 STAGE 2 MULTI-FAMILY DATASET READY", flush=True)
    print(f"Authored families          : {len(family_payloads)}", flush=True)
    for family in family_payloads:
        print(
            f"  {family['familyId']}  {family['sourceAsset'].get('displayName')}  "
            f"{family['sourceSize'][0]}x{family['sourceSize'][1]}",
            flush=True,
        )
    print(f"Fixed train crops          : {payload['counts']['trainCrops']}", flush=True)
    print(f"Fixed held-out crops       : {payload['counts']['validationCrops']}", flush=True)
    print("Split rule                 : per-family hard pixel-disjoint domains", flush=True)
    print(f"Manifest                   : {manifest_path}", flush=True)
    print("=" * 68, flush=True)
    return payload


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the V16 Stage 2 multi-family authored dataset"
    )
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--config",
        type=Path,
        default=Path("tools/nsamdr/neural/configs/v9_preview_raven.json"),
    )
    p.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--train-crops", type=int, default=16)
    p.add_argument("--validation-crops", type=int, default=4)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = V9Config.load(config_path.resolve())
    prepare(
        repo_root,
        config,
        shared_cache=args.shared_cache,
        rebuild=bool(args.rebuild),
        train_crops=max(1, int(args.train_crops)),
        validation_crops=max(1, int(args.validation_crops)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
