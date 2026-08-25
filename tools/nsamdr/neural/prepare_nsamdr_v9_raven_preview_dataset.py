#!/usr/bin/env python3
"""Build the deterministic, feature-stratified Raven development dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
NSAMDR_ROOT = HERE.parent
if str(NSAMDR_ROOT) not in sys.path:
    sys.path.insert(0, str(NSAMDR_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import eve_asset_test as eve  # type: ignore
import authored_texture_dataset as authored_dataset
from v9.config import V9Config
from v9.experiments import (
    DEFAULT_TUNING_ASSET_NAME,
    DEFAULT_TUNING_ASSET_QUERY,
)

PREVIEW_DATASET_SCHEMA = "NSAMDR_RAVEN_DEVELOPMENT_DATASET_V2"
PREVIEW_CROP_SCHEMA = "NSAMDR_RAVEN_DEVELOPMENT_CROP_V2"
BUILDER_VERSION = "raven-feature-stratified-disjoint-v2"


def _find_navy_raven(rows: list[eve.ResourceRow], repo_root: Path) -> eve.ShipCatalogEntry:
    entries = eve._build_sde_ship_catalog(rows, repo_root)  # noqa: SLF001 - shared repo helper
    exact = [entry for entry in entries if entry.display_name.strip().lower() == DEFAULT_TUNING_ASSET_NAME.lower()]
    if not exact:
        near = [entry.display_name for entry in entries if "raven" in entry.display_name.lower()]
        raise RuntimeError(
            f"{DEFAULT_TUNING_ASSET_NAME!r} was not found in the EVE SDE ship catalog. "
            f"Raven matches: {near[:12]}"
        )
    return exact[0]


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def _semantic_plane(path: Path | None, channel: int, width: int, height: int, default: int) -> np.ndarray:
    if path is None or not path.is_file():
        return np.full((height, width), int(default), dtype=np.uint8)
    rgba = _load_rgba(path)
    plane = rgba[..., max(0, min(3, int(channel)))]
    if plane.shape != (height, width):
        plane = np.asarray(
            Image.fromarray(plane, mode="L").resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
    return plane


def _resize_rgb(array: np.ndarray, width: int, height: int, *, nearest: bool = False) -> np.ndarray:
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return np.asarray(Image.fromarray(array, mode="RGB").resize((width, height), mode), dtype=np.uint8)


def _normal_rgb(path: Path) -> tuple[np.ndarray, str]:
    return authored_dataset.load_normal_training_rgb(path)


def _detail_score(albedo: np.ndarray, normal: np.ndarray, material: np.ndarray) -> float:
    score = authored_dataset.detail_map(albedo, normal, material)
    return float(score.mean() + score.max(initial=0.0) * 0.35)


def _grid_positions(length: int, crop_size: int) -> list[int]:
    """Return a strictly non-overlapping fixed grid.

    Uncovered edge remainders are intentionally ignored. Appending a shifted
    last crop would make held-out cells overlap training cells and invalidate
    experiment comparisons.
    """
    if length < crop_size:
        return [0]
    return list(range(0, length - crop_size + 1, crop_size))

def _texture_families_from_report(report_path: Path, fallback_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for area in report.get("areas", []):
            textures = area.get("textures", {}) if isinstance(area, dict) else {}
            albedo = str(textures.get("albedo") or "")
            normal = str(textures.get("normal") or "")
            material = str(textures.get("material") or "")
            roughness = str(textures.get("roughnessMap") or "")
            glow = str(textures.get("glow") or "")
            channels = dict(area.get("channels") or {}) if isinstance(area, dict) else {}
            if not albedo or not normal:
                continue
            key = (albedo, normal, material, roughness, glow, json.dumps(channels, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            result.append(
                {
                    "areaName": str(area.get("areaName") or ""),
                    "areaType": str(area.get("areaType") or ""),
                    "shaderFamily": str(area.get("shaderFamily") or ""),
                    "albedo": albedo,
                    "normal": normal,
                    "material": material,
                    "roughnessMap": roughness,
                    "glow": glow,
                    "channels": channels,
                    "materialEncoding": "canonical-material-emissive-roughness",
                }
            )

    # Legacy fallback: the asset manifest still exposes the directly related
    # Raven maps even if SOF area extraction is incomplete.
    if not result:
        albedo = str(fallback_manifest.get("albedoPng") or "")
        normal = str(fallback_manifest.get("normalPng") or "")
        material = str(fallback_manifest.get("pgsPng") or "")
        if albedo and normal:
            result.append(
                {
                    "areaName": "legacy-raven",
                    "areaType": "primary",
                    "shaderFamily": "legacy",
                    "albedo": albedo,
                    "normal": normal,
                    "material": material,
                    "roughnessMap": "",
                    "glow": "",
                    "channels": {"material": 0, "glow": 1, "roughness": 2},
                    "materialEncoding": "legacy-rgb-fallback",
                }
            )
    if not result:
        raise RuntimeError("Raven asset preparation produced no aligned albedo+normal texture family")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprint(
    asset_manifest: dict[str, Any],
    families: list[dict[str, Any]],
) -> str:
    """Fingerprint only immutable source inputs available before crop selection."""
    payload: dict[str, Any] = {
        "builder": BUILDER_VERSION,
        "model": asset_manifest.get("model", {}).get("logical"),
        "sofIdentity": asset_manifest.get("sofIdentity"),
        "families": [],
    }
    for family in families:
        entry = {
            "areaName": family.get("areaName", ""),
            "areaType": family.get("areaType", ""),
            "shaderFamily": family.get("shaderFamily", ""),
        }
        files: dict[str, dict[str, Any]] = {}
        for role in ("albedo", "normal", "material", "roughnessMap", "glow"):
            raw = str(family.get(role) or "")
            path = Path(raw) if raw else None
            files[role] = {
                "path": raw,
                "size": path.stat().st_size if path and path.is_file() else 0,
                "sha256": _sha256_file(path) if path and path.is_file() else "",
            }
        entry["files"] = files
        entry["channels"] = family.get("channels", {})
        entry["materialEncoding"] = family.get("materialEncoding", "")
        payload["families"].append(entry)

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _selection_fingerprint(
    source_fingerprint: str,
    records: list[dict[str, Any]],
    *,
    seed: int,
) -> str:
    """Fingerprint the exact deterministic Raven regions selected for tuning."""
    selected_regions = sorted(
        (
            {
                "cropId": str(record["crop_id"]),
                "familyId": str(record["family_id"]),
                "split": str(record["split"]),
                "sourceBox": list(record["source_box"]),
            }
            for record in records
        ),
        key=lambda item: (item["split"], item["cropId"]),
    )
    payload = {
        "schema": PREVIEW_DATASET_SCHEMA,
        "builder": BUILDER_VERSION,
        "sourceFingerprint": source_fingerprint,
        "selectedTrainCrops": sum(1 for record in records if record["split"] == "train"),
        "selectedValidationCrops": sum(1 for record in records if record["split"] == "validation"),
        "seed": int(seed),
        "selectedRegions": selected_regions,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()



def _select_fixed_regions(
    candidates: list[dict[str, Any]],
    *,
    max_train_crops: int,
    max_validation_crops: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose deterministic, disjoint subsets balanced across authored features.

    Requested region counts are upper bounds. The authored Raven texture
    topology decides the actual number of unique spatial regions; regions are
    never duplicated or shared between train and validation merely to satisfy
    an arbitrary requested count. Selection covers the full detail distribution
    and texture families instead of sorting for only the highest-detail crops.
    """
    if len(candidates) < 2:
        raise RuntimeError(
            "Raven fixed tuning set requires at least two unique non-overlapping "
            "512x512 regions so training and held-out data remain spatially separate."
        )

    def stable_key(item: dict[str, Any]) -> tuple[str, str, int, int]:
        identity = (
            f"{seed}|{item['familyId']}|{item['x']}|{item['y']}|"
            f"{int(bool(item['materialValid']))}"
        )
        return (
            hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            str(item["familyId"]),
            int(item["y"]),
            int(item["x"]),
        )

    validation_pool = [item for item in candidates if bool(item["holdout"])]
    training_pool = [item for item in candidates if not bool(item["holdout"])]

    # Tiny layouts can theoretically put every cell on one side of the
    # checkerboard. Move one *whole unique cell* to the missing split rather
    # than copying or overlapping it.
    if not validation_pool:
        all_ranked = sorted(candidates, key=stable_key)
        validation_pool = [all_ranked[0]]
        training_pool = all_ranked[1:]
    elif not training_pool:
        all_ranked = sorted(candidates, key=stable_key)
        training_pool = [all_ranked[0]]
        validation_pool = all_ranked[1:]

    def stratified_subset(pool: list[dict[str, Any]], requested: int) -> list[dict[str, Any]]:
        requested = min(len(pool), max(1, int(requested)))
        ordered_by_detail = sorted(
            pool,
            key=lambda item: (
                float(item["detailScore"]),
                str(item["familyId"]),
                int(item["y"]),
                int(item["x"]),
            ),
        )
        strata: dict[int, list[dict[str, Any]]] = {index: [] for index in range(4)}
        denominator = max(1, len(ordered_by_detail))
        for rank, item in enumerate(ordered_by_detail):
            stratum = min(3, (rank * 4) // denominator)
            item["detailStratum"] = stratum
            strata[stratum].append(item)
        for values in strata.values():
            values.sort(key=stable_key)

        # Low/high/mid-low/mid-high ensures a four-crop validation set samples
        # every detail quartile. Within a quartile prefer an as-yet-unseen family.
        stratum_order = (0, 3, 1, 2)
        family_counts: dict[str, int] = {}
        selected: list[dict[str, Any]] = []
        while len(selected) < requested:
            made_progress = False
            for stratum in stratum_order:
                values = strata[stratum]
                if not values or len(selected) >= requested:
                    continue
                choice_index = min(
                    range(len(values)),
                    key=lambda index: (
                        family_counts.get(str(values[index]["familyId"]), 0),
                        stable_key(values[index]),
                    ),
                )
                choice = values.pop(choice_index)
                selected.append(choice)
                family = str(choice["familyId"])
                family_counts[family] = family_counts.get(family, 0) + 1
                made_progress = True
            if not made_progress:
                break
        return selected

    selected_train = stratified_subset(training_pool, max_train_crops)
    selected_validation = stratified_subset(validation_pool, max_validation_crops)
    if not selected_train or not selected_validation:
        raise RuntimeError(
            "Raven fixed tuning set could not produce at least one unique "
            "training region and one unique held-out region."
        )

    train_keys = {
        (str(item["familyId"]), int(item["x"]), int(item["y"]))
        for item in selected_train
    }
    validation_keys = {
        (str(item["familyId"]), int(item["x"]), int(item["y"]))
        for item in selected_validation
    }
    overlap = train_keys.intersection(validation_keys)
    if overlap:
        raise RuntimeError(
            f"Raven fixed tuning split contains spatial overlap: {sorted(overlap)}"
        )
    return selected_train, selected_validation

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
    output_root = (repo_root / config.dataset_root).resolve()
    manifest_path = (repo_root / config.dataset_manifest).resolve()
    crop_root = output_root / "crops"

    cache_root, indexes, _resfiles = eve.resolve_layout(shared_cache, allow_prompt=False)
    print(f"[preview-dataset] EVE SharedCache: {cache_root}", flush=True)
    rows = eve.read_rows(indexes)
    selected = _find_navy_raven(rows, repo_root)
    print(
        f"[preview-dataset] Fixed asset: {selected.display_name} "
        f"({selected.canonical_key}) {selected.preferred_asset}",
        flush=True,
    )

    # Reuse the real ship-preparation path so faction/SOF texture insertion is
    # exactly the same as the public Mode 1/2/3 preview.
    _obj, _albedo, _normal, _pgs, _env, _envs, _materials, asset_manifest_path, _catalog, _cache = eve.prepare_asset(
        repo_root,
        shared_cache,
        selected.preferred_asset or DEFAULT_TUNING_ASSET_QUERY,
        selected.canonical_key,
    )
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    report_path = Path(str(asset_manifest.get("materialBaselineReport") or ""))
    families = _texture_families_from_report(report_path, asset_manifest)
    source_fingerprint = _source_fingerprint(asset_manifest, families)

    if not rebuild and manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            crop_paths = [Path(record["path"]) for record in existing.get("crops", [])]
            if (
                existing.get("schema") == PREVIEW_DATASET_SCHEMA
                and existing.get("sourceFingerprint") == source_fingerprint
                and int(existing.get("splitPolicy", {}).get("maxTrainCrops", -1)) == int(train_crops)
                and int(existing.get("splitPolicy", {}).get("maxValidationCrops", -1)) == int(validation_crops)
                and int(existing.get("splitPolicy", {}).get("seed", -1)) == int(config.seed)
                and crop_paths
                and all(path.is_file() for path in crop_paths)
            ):
                print(f"[preview-dataset] Fixed Raven dataset is current: {manifest_path}", flush=True)
                return existing
        except (OSError, ValueError, KeyError, TypeError):
            pass

    shutil.rmtree(crop_root, ignore_errors=True)
    (crop_root / "train").mkdir(parents=True, exist_ok=True)
    (crop_root / "validation").mkdir(parents=True, exist_ok=True)

    crop_size = int(config.source_crop_size)
    if crop_size != config.tile_size * config.target_scale:
        raise RuntimeError(
            "preview sourceCropSize must equal the model HR target size so the "
            "fixed spatial regions cannot drift inside a larger bundle"
        )

    candidates: list[dict[str, Any]] = []
    family_payloads: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        albedo_path = Path(str(family["albedo"]))
        normal_path = Path(str(family["normal"]))
        material_path = Path(str(family.get("material") or "")) if family.get("material") else None
        roughness_path = Path(str(family.get("roughnessMap") or "")) if family.get("roughnessMap") else None
        glow_path = Path(str(family.get("glow") or "")) if family.get("glow") else None
        if not albedo_path.is_file() or not normal_path.is_file():
            continue

        albedo = _load_rgb(albedo_path)
        normal, normal_encoding = _normal_rgb(normal_path)
        h, w = albedo.shape[:2]
        if normal.shape[:2] != (h, w):
            normal = _resize_rgb(normal, w, h)

        channels = dict(family.get("channels") or {})
        material_source = material_path if material_path and material_path.is_file() else None
        roughness_source = roughness_path if roughness_path and roughness_path.is_file() else material_source
        glow_source = glow_path if glow_path and glow_path.is_file() else material_source
        material_valid = bool(material_source or roughness_source or glow_source)
        if material_valid:
            material_plane = _semantic_plane(material_source, int(channels.get("material", 0)), w, h, 0)
            emissive_plane = _semantic_plane(glow_source, int(channels.get("glow", 1)), w, h, 0)
            roughness_plane = _semantic_plane(roughness_source, int(channels.get("roughness", 2)), w, h, 128)
            material = np.stack((material_plane, emissive_plane, roughness_plane), axis=-1)
        else:
            material = np.stack((
                np.zeros((h, w), dtype=np.uint8),
                np.zeros((h, w), dtype=np.uint8),
                np.full((h, w), 128, dtype=np.uint8),
            ), axis=-1)

        if h < crop_size or w < crop_size:
            scale = max(crop_size / max(w, 1), crop_size / max(h, 1))
            new_w = max(crop_size, round(w * scale))
            new_h = max(crop_size, round(h * scale))
            albedo = _resize_rgb(albedo, new_w, new_h)
            normal = _resize_rgb(normal, new_w, new_h)
            material = _resize_rgb(material, new_w, new_h, nearest=True)
            h, w = albedo.shape[:2]

        family_id = hashlib.sha1(
            f"{selected.canonical_key}|{family_index}|{albedo_path}|{normal_path}|{material_path}|{roughness_path}|{glow_path}|{json.dumps(channels, sort_keys=True)}".encode("utf-8")
        ).hexdigest()[:16]
        xs = _grid_positions(w, crop_size)
        ys = _grid_positions(h, crop_size)
        cell_count = 0
        for gy, y in enumerate(ys):
            for gx, x in enumerate(xs):
                a = np.ascontiguousarray(albedo[y:y + crop_size, x:x + crop_size])
                n = np.ascontiguousarray(normal[y:y + crop_size, x:x + crop_size])
                m = np.ascontiguousarray(material[y:y + crop_size, x:x + crop_size])
                if a.shape[:2] != (crop_size, crop_size):
                    continue
                score = _detail_score(a, n, m)
                holdout = ((gx + 2 * gy + family_index) % 4) == 0
                candidates.append(
                    {
                        "familyId": family_id,
                        "familyIndex": family_index,
                        "gx": gx,
                        "gy": gy,
                        "x": x,
                        "y": y,
                        "detailScore": score,
                        "holdout": holdout,
                        "albedo": a,
                        "normal": n,
                        "material": m,
                        "materialValid": material_valid,
                        "normalEncoding": normal_encoding,
                        "family": family,
                    }
                )
                cell_count += 1
        family_payloads.append(
            {
                "familyId": family_id,
                "areaName": family.get("areaName", ""),
                "areaType": family.get("areaType", ""),
                "shaderFamily": family.get("shaderFamily", ""),
                "albedo": str(albedo_path),
                "normal": str(normal_path),
                "material": str(material_path) if material_path else "",
                "roughnessMap": str(roughness_path) if roughness_path else "",
                "glow": str(glow_path) if glow_path else "",
                "channels": channels,
                "materialEncoding": "canonical-material-emissive-roughness",
                "normalEncoding": normal_encoding,
                "materialSupervision": material_valid,
                "sourceSize": [w, h],
                "nonOverlappingGridCells": cell_count,
            }
        )

    if not candidates:
        raise RuntimeError("fixed Raven preview dataset has no usable 512x512 texture regions")

    selected_train, selected_validation = _select_fixed_regions(
        candidates,
        max_train_crops=train_crops,
        max_validation_crops=validation_crops,
        seed=int(config.seed),
    )

    available_validation = sum(1 for item in candidates if bool(item["holdout"]))
    available_training = sum(1 for item in candidates if not bool(item["holdout"]))
    print(
        f"[preview-dataset] Unique non-overlapping 512 regions: {len(candidates)} "
        f"(checkerboard train={available_training}, held-out={available_validation})",
        flush=True,
    )
    print(
        f"[preview-dataset] Requested region caps: train<={train_crops}, "
        f"held-out<={validation_crops}",
        flush=True,
    )
    print(
        f"[preview-dataset] Selected fixed set: train={len(selected_train)}, "
        f"held-out={len(selected_validation)}",
        flush=True,
    )
    if len(selected_train) < train_crops or len(selected_validation) < validation_crops:
        print(
            "[preview-dataset] NOTE: Raven source topology provides fewer unique "
            "512x512 cells than the requested caps; using all eligible cells "
            "without overlap or duplication.",
            flush=True,
        )

    selected_cells = [
        *(dict(item, split="train") for item in selected_train),
        *(dict(item, split="validation") for item in selected_validation),
    ]
    records: list[dict[str, Any]] = []
    for ordinal, item in enumerate(selected_cells):
        split = str(item["split"])
        crop_id = f"{item['familyId']}_{item['gx']:02d}_{item['gy']:02d}_{split}"
        destination = crop_root / split / f"{crop_id}.npz"
        metadata = {
            "schema": PREVIEW_CROP_SCHEMA,
            "cropId": crop_id,
            "familyId": item["familyId"],
            "split": split,
            "pixelBox": [item["x"], item["y"], item["x"] + crop_size, item["y"] + crop_size],
            "grid": [item["gx"], item["gy"]],
            "detailScore": item["detailScore"],
            "detailStratum": int(item["detailStratum"]),
            "fixed": True,
            "overlapBetweenTrainAndValidation": False,
            "normalEncoding": item["normalEncoding"],
            "materialSupervision": item["materialValid"],
        }
        np.savez_compressed(
            destination,
            albedo=item["albedo"],
            normal=item["normal"],
            material=item["material"],
            material_valid=np.asarray([1.0 if item["materialValid"] else 0.0], dtype=np.float32),
            metadata=np.asarray(json.dumps(metadata), dtype=np.str_),
        )
        records.append(
            {
                "crop_id": crop_id,
                "family_id": item["familyId"],
                "split": split,
                "path": str(destination.resolve()),
                "source_box": [item["x"], item["y"], item["x"] + crop_size, item["y"] + crop_size],
                "detail_score": item["detailScore"],
                "detail_stratum": int(item["detailStratum"]),
                "albedo_logical": str(item["family"].get("albedo") or ""),
                "normal_logical": str(item["family"].get("normal") or ""),
                "material_logical": str(item["family"].get("material") or ""),
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

    fingerprint = _selection_fingerprint(
        source_fingerprint,
        records,
        seed=config.seed,
    )

    payload: dict[str, Any] = {
        "schema": PREVIEW_DATASET_SCHEMA,
        "sourceFingerprint": source_fingerprint,
        "fingerprint": fingerprint,
        "builderVersion": BUILDER_VERSION,
        "modelScope": "tuning",
        "fixedPreviewSet": True,
        "deterministic": True,
        "asset": {
            "displayName": selected.display_name,
            "typeId": selected.type_id,
            "selectionKey": selected.canonical_key,
            "preferredAsset": selected.preferred_asset,
            "query": selected.preferred_asset or DEFAULT_TUNING_ASSET_QUERY,
            "sofIdentity": asset_manifest.get("sofIdentity"),
            "assetManifest": str(asset_manifest_path),
        },
        "splitPolicy": {
            "type": "feature-stratified-non-overlapping-512-grid-v2",
            "detailStrata": 4,
            "maxTrainCrops": train_crops,
            "maxValidationCrops": validation_crops,
            "trainCrops": sum(1 for record in records if record["split"] == "train"),
            "validationCrops": sum(1 for record in records if record["split"] == "validation"),
            "seed": config.seed,
            "augmentationSeed": config.seed,
            "validationSeed": config.seed + 77,
        },
        "families": family_payloads,
        "crops": records,
        "counts": {
            "families": len(family_payloads),
            "trainCrops": sum(1 for record in records if record["split"] == "train"),
            "validationCrops": sum(1 for record in records if record["split"] == "validation"),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("=" * 64, flush=True)
    print("NSAMDR RAVEN FEATURE-STRATIFIED DEVELOPMENT DATASET READY", flush=True)
    print(f"Asset                    : {selected.display_name}", flush=True)
    print(f"Selection                 : {selected.canonical_key}", flush=True)
    print(f"Texture families          : {len(family_payloads)}", flush=True)
    print(
        f"Fixed train crops         : {payload['counts']['trainCrops']} (cap {train_crops})",
        flush=True,
    )
    print(
        f"Fixed held-out crops      : {payload['counts']['validationCrops']} (cap {validation_crops})",
        flush=True,
    )
    print(f"Crop geometry             : {crop_size}x{crop_size}, non-overlapping", flush=True)
    print(f"Manifest                  : {manifest_path}", flush=True)
    print("=" * 64, flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the deterministic feature-stratified Raven development dataset"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("tools/nsamdr/neural/configs/v9_preview_raven.json"))
    parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--train-crops", type=int, default=16, help="Maximum unique non-overlapping training regions")
    parser.add_argument("--validation-crops", type=int, default=4, help="Maximum unique non-overlapping held-out regions")
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = V9Config.load(config_path.resolve())
    prepare(
        repo_root,
        config,
        shared_cache=args.shared_cache,
        rebuild=args.rebuild,
        train_crops=max(1, args.train_crops),
        validation_crops=max(1, args.validation_crops),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
