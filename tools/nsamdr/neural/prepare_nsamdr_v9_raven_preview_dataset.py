#!/usr/bin/env python3
"""Build the deterministic, feature-stratified Raven development dataset.

The Raven quick workflow uses authored EVE texture pixels as supervision.  The
important distinction is between *native authored texture resolution* and the
4096px render/preview target: EVE's SOF semantic maps are not guaranteed to be
4096x4096.  This builder therefore preserves the native full-index DDS pixels,
records their true dimensions, and never substitutes a differently encoded map
merely because it is larger.
"""
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

PREVIEW_DATASET_SCHEMA = "NSAMDR_RAVEN_DEVELOPMENT_DATASET_V4_NATIVE_AUTHORED"
PREVIEW_CROP_SCHEMA = "NSAMDR_RAVEN_DEVELOPMENT_CROP_V4_NATIVE_AUTHORED"
BUILDER_VERSION = "raven-native-authored-feature-stratified-disjoint-v4"


class RavenPreviewDatasetPreparationApplication:
    def _find_navy_raven(
        self,
        rows: list[eve.ResourceRow],
        repo_root: Path,
    ) -> eve.ShipCatalogEntry:
        entries = eve._build_sde_ship_catalog(rows, repo_root)  # noqa: SLF001
        exact = [
            entry
            for entry in entries
            if entry.display_name.strip().lower() == DEFAULT_TUNING_ASSET_NAME.lower()
        ]
        if not exact:
            near = [entry.display_name for entry in entries if "raven" in entry.display_name.lower()]
            raise RuntimeError(
                f"{DEFAULT_TUNING_ASSET_NAME!r} was not found in the EVE SDE ship catalog. "
                f"Raven matches: {near[:12]}"
            )
        return exact[0]

    def _authoritative_rows(self, indexes: list[Path]) -> list[eve.ResourceRow]:
        """Read prefetch first so the full index wins duplicate logical resources."""
        ordered = sorted(indexes, key=lambda path: 0 if "prefetch" in path.name.lower() else 1)
        return eve.read_rows(ordered)

    def _load_rgb(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _load_rgba(self, path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGBA"), dtype=np.uint8)

    def _semantic_plane(
        self,
        path: Path | None,
        channel: int,
        width: int,
        height: int,
        default: int,
    ) -> np.ndarray:
        if path is None or not path.is_file():
            return np.full((height, width), int(default), dtype=np.uint8)
        rgba = self._load_rgba(path)
        plane = rgba[..., max(0, min(3, int(channel)))]
        if plane.shape != (height, width):
            plane = np.asarray(
                Image.fromarray(plane, mode="L").resize(
                    (width, height), Image.Resampling.BILINEAR
                ),
                dtype=np.uint8,
            )
        return plane

    def _resize_rgb(
        self,
        array: np.ndarray,
        width: int,
        height: int,
        *,
        nearest: bool = False,
    ) -> np.ndarray:
        mode = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
        return np.asarray(Image.fromarray(array, mode="RGB").resize((width, height), mode), dtype=np.uint8)

    def _normal_rgb(self, path: Path) -> tuple[np.ndarray, str]:
        return authored_dataset.load_normal_training_rgb(path)

    def _detail_score(
        self,
        albedo: np.ndarray,
        normal: np.ndarray,
        material: np.ndarray,
    ) -> float:
        score = authored_dataset.detail_map(albedo, normal, material)
        return float(score.mean() + score.max(initial=0.0) * 0.35)

    def _grid_positions(self, length: int, crop_size: int) -> list[int]:
        if length < crop_size:
            return [0]
        return list(range(0, length - crop_size + 1, crop_size))

    def _texture_families_from_report(
        self,
        report_path: Path,
        fallback_manifest: dict[str, Any],
    ) -> list[dict[str, Any]]:
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
                key = (
                    albedo,
                    normal,
                    material,
                    roughness,
                    glow,
                    json.dumps(channels, sort_keys=True),
                )
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

    def _manifest_texture_logical_map(
        self,
        asset_manifest: dict[str, Any],
    ) -> dict[str, str]:
        """Map prepared PNG/local paths back to EVE logical resources."""
        result: dict[str, str] = {}

        def add(record: object) -> None:
            if not isinstance(record, dict):
                return
            logical = str(record.get("logical") or "").strip().replace("\\", "/")
            if not logical:
                return
            for field in ("converted", "local"):
                raw = str(record.get(field) or "").strip()
                if raw:
                    result[str(Path(raw).resolve()).casefold()] = logical

        sof_textures = asset_manifest.get("sofTextures")
        if isinstance(sof_textures, dict):
            for record in sof_textures.values():
                add(record)
        direct_textures = asset_manifest.get("textures")
        if isinstance(direct_textures, dict):
            for record in direct_textures.values():
                add(record)
        return result

    def _native_family_sources(
        self,
        repo_root: Path,
        output_root: Path,
        asset_manifest: dict[str, Any],
        families: list[dict[str, Any]],
        rows: list[eve.ResourceRow],
        resfiles: Path,
        crop_size: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Resolve exact SOF semantics to untouched native full-index DDS pixels.

        A larger texture with a different semantic encoding is *not* a valid HR
        replacement.  The old V3 code incorrectly assumed every SR authority map
        had to be 4K and therefore rejected Raven's native 1024px V5 albedo map.
        Here 4096 is treated as the render target, while the dataset authority is
        the true authored semantic texture at its native resolution.
        """
        logical_by_prepared_path = self._manifest_texture_logical_map(asset_manifest)
        rows_by_logical = {
            row.logical.strip().replace("\\", "/").lower(): row for row in rows
        }
        native_root = output_root / "native_authored_sources"
        native_root.mkdir(parents=True, exist_ok=True)
        dds_repository = authored_dataset.AuthoredTextureDatasetRepository()
        converted_cache: dict[str, tuple[Path, tuple[int, int], eve.ResourceRow]] = {}

        def decode_logical(logical: str, role: str) -> tuple[Path, dict[str, Any]]:
            row = rows_by_logical.get(logical.lower())
            if row is None:
                raise RuntimeError(
                    f"Raven {role} logical resource is absent from the authoritative full index: {logical}"
                )
            cache_key = row.logical.lower()
            cached = converted_cache.get(cache_key)
            if cached is None:
                source_path = resfiles / Path(row.hashed)
                if not source_path.is_file():
                    raise RuntimeError(
                        f"Raven {role} native EVE resource is not local: {row.logical} -> {source_path}"
                    )
                width, height, mip_count, format_name = dds_repository.parse_dds_header(source_path)
                identity = hashlib.sha256(
                    f"{row.logical.lower()}|{row.hashed}".encode("utf-8")
                ).hexdigest()[:24]
                output_path = native_root / f"{identity}_{Path(row.logical).stem}.png"
                eve.convert_dds(repo_root, source_path, output_path)
                with Image.open(output_path) as converted:
                    converted_size = (int(converted.width), int(converted.height))
                if converted_size != (width, height):
                    raise RuntimeError(
                        f"Native Raven {role} decode changed authored dimensions for {row.logical}: "
                        f"DDS={width}x{height}, PNG={converted_size[0]}x{converted_size[1]}"
                    )
                cached = (output_path, (width, height), row)
                converted_cache[cache_key] = cached
            output_path, (width, height), source_row = cached
            source_path = resfiles / Path(source_row.hashed)
            _width, _height, mip_count, format_name = dds_repository.parse_dds_header(source_path)
            return output_path, {
                "logical": source_row.logical,
                "hashed": source_row.hashed,
                "indexFile": source_row.index_file,
                "width": width,
                "height": height,
                "mipCount": mip_count,
                "format": format_name,
                "authority": "eve-full-index-native-semantic",
            }

        def resolve_prepared(raw: str, role: str) -> tuple[Path, dict[str, Any]]:
            prepared_path = Path(raw).resolve()
            logical = logical_by_prepared_path.get(str(prepared_path).casefold())
            if logical is None and raw.strip().lower().startswith("res:/"):
                logical = raw.strip().replace("\\", "/")
            if not logical:
                raise RuntimeError(
                    f"Raven {role} source could not be mapped back to an EVE logical resource: {raw}"
                )
            return decode_logical(logical, role)

        hull_probe: dict[str, Any] = {}
        direct_textures = asset_manifest.get("textures")
        if isinstance(direct_textures, dict):
            for role in ("albedo", "normal"):
                record = direct_textures.get(role)
                if not isinstance(record, dict):
                    continue
                logical = str(record.get("logical") or "").strip().replace("\\", "/")
                if not logical:
                    continue
                try:
                    _decoded, provenance = decode_logical(logical, f"hull-{role}")
                    hull_probe[role] = provenance
                except RuntimeError as exc:
                    hull_probe[role] = {"logical": logical, "error": str(exc)}

        result: list[dict[str, Any]] = []
        for family in families:
            native = dict(family)
            provenance: dict[str, Any] = {}
            shader_family = str(family.get("shaderFamily") or "").lower()
            for role in ("albedo", "normal", "material", "roughnessMap", "glow"):
                raw = str(family.get(role) or "").strip()
                if not raw:
                    continue
                try:
                    output_path, source = resolve_prepared(raw, role)
                except RuntimeError:
                    if role in ("albedo", "normal"):
                        raise
                    provenance[role] = {
                        "authority": "prepared-auxiliary-fallback",
                        "path": raw,
                    }
                    continue

                if shader_family == "legacy" and role in ("albedo", "normal"):
                    probed = hull_probe.get(role)
                    if isinstance(probed, dict) and not probed.get("error"):
                        if int(probed.get("width", 0)) * int(probed.get("height", 0)) > int(source["width"]) * int(source["height"]):
                            logical = str(probed.get("logical") or "")
                            if logical:
                                output_path, source = decode_logical(logical, f"legacy-{role}")

                if role in ("albedo", "normal") and min(int(source["width"]), int(source["height"])) < crop_size:
                    raise RuntimeError(
                        f"Raven authored {role} semantic texture is smaller than the required "
                        f"{crop_size}x{crop_size} training crop: {source['width']}x{source['height']} "
                        f"{source['logical']}"
                    )

                native[role] = str(output_path)
                provenance[role] = source
            native["sourceProvenance"] = provenance
            result.append(native)

        return result, hull_probe

    def _sha256_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _source_fingerprint(
        self,
        asset_manifest: dict[str, Any],
        families: list[dict[str, Any]],
        hull_probe: dict[str, Any],
    ) -> str:
        payload: dict[str, Any] = {
            "builder": BUILDER_VERSION,
            "model": asset_manifest.get("model", {}).get("logical"),
            "sofIdentity": asset_manifest.get("sofIdentity"),
            "hullAuthorityProbe": hull_probe,
            "families": [],
        }
        for family in families:
            entry = {
                "areaName": family.get("areaName", ""),
                "areaType": family.get("areaType", ""),
                "shaderFamily": family.get("shaderFamily", ""),
                "sourceProvenance": family.get("sourceProvenance", {}),
            }
            files: dict[str, dict[str, Any]] = {}
            for role in ("albedo", "normal", "material", "roughnessMap", "glow"):
                raw = str(family.get(role) or "")
                path = Path(raw) if raw else None
                files[role] = {
                    "path": raw,
                    "size": path.stat().st_size if path and path.is_file() else 0,
                    "sha256": self._sha256_file(path) if path and path.is_file() else "",
                }
            entry["files"] = files
            entry["channels"] = family.get("channels", {})
            entry["materialEncoding"] = family.get("materialEncoding", "")
            payload["families"].append(entry)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _selection_fingerprint(
        self,
        source_fingerprint: str,
        records: list[dict[str, Any]],
        *,
        seed: int,
    ) -> str:
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
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _select_fixed_regions(
        self,
        candidates: list[dict[str, Any]],
        *,
        max_train_crops: int,
        max_validation_crops: int,
        seed: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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

            selected: list[dict[str, Any]] = []
            family_counts: dict[str, int] = {}
            for _ in range(requested + 4):
                made_progress = False
                for stratum in (0, 3, 1, 2):
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
                if len(selected) >= requested or not made_progress:
                    break
            return selected

        selected_train = stratified_subset(training_pool, max_train_crops)
        selected_validation = stratified_subset(validation_pool, max_validation_crops)
        if not selected_train or not selected_validation:
            raise RuntimeError(
                "Raven fixed tuning set could not produce at least one unique training region "
                "and one unique held-out region."
            )
        train_keys = {(str(i["familyId"]), int(i["x"]), int(i["y"])) for i in selected_train}
        validation_keys = {(str(i["familyId"]), int(i["x"]), int(i["y"])) for i in selected_validation}
        overlap = train_keys.intersection(validation_keys)
        if overlap:
            raise RuntimeError(f"Raven fixed tuning split contains spatial overlap: {sorted(overlap)}")
        return selected_train, selected_validation

    def prepare(
        self,
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
        crop_size = int(config.source_crop_size)
        if crop_size != config.tile_size * config.target_scale:
            raise RuntimeError(
                "preview sourceCropSize must equal the model HR target size so the fixed "
                "spatial regions cannot drift inside a larger bundle"
            )

        cache_root, indexes, resfiles = eve.resolve_layout(shared_cache, allow_prompt=False)
        print(f"[preview-dataset] EVE SharedCache: {cache_root}", flush=True)
        rows = self._authoritative_rows(indexes)
        selected = self._find_navy_raven(rows, repo_root)
        print(
            f"[preview-dataset] Fixed asset: {selected.display_name} "
            f"({selected.canonical_key}) {selected.preferred_asset}",
            flush=True,
        )

        _obj, _albedo, _normal, _pgs, _env, _envs, _materials, asset_manifest_path, _catalog, _cache = eve.prepare_asset(
            repo_root,
            shared_cache,
            selected.preferred_asset or DEFAULT_TUNING_ASSET_QUERY,
            selected.canonical_key,
        )
        asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
        report_path = Path(str(asset_manifest.get("materialBaselineReport") or ""))
        families = self._texture_families_from_report(report_path, asset_manifest)
        families, hull_probe = self._native_family_sources(
            repo_root,
            output_root,
            asset_manifest,
            families,
            rows,
            resfiles,
            crop_size,
        )
        source_fingerprint = self._source_fingerprint(asset_manifest, families, hull_probe)

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

            albedo = self._load_rgb(albedo_path)
            normal, normal_encoding = self._normal_rgb(normal_path)
            h, w = albedo.shape[:2]
            normal_h, normal_w = normal.shape[:2]
            if min(w, h) < crop_size:
                raise RuntimeError(
                    f"Raven native albedo {w}x{h} is smaller than training crop {crop_size}: {albedo_path}"
                )
            if min(normal_w, normal_h) < crop_size:
                raise RuntimeError(
                    f"Raven native normal {normal_w}x{normal_h} is smaller than training crop {crop_size}: {normal_path}"
                )
            if normal.shape[:2] != (h, w):
                normal = self._resize_rgb(normal, w, h)

            channels = dict(family.get("channels") or {})
            material_source = material_path if material_path and material_path.is_file() else None
            roughness_source = roughness_path if roughness_path and roughness_path.is_file() else material_source
            glow_source = glow_path if glow_path and glow_path.is_file() else material_source
            material_valid = bool(material_source or roughness_source or glow_source)
            if material_valid:
                material_plane = self._semantic_plane(material_source, int(channels.get("material", 0)), w, h, 0)
                emissive_plane = self._semantic_plane(glow_source, int(channels.get("glow", 1)), w, h, 0)
                roughness_plane = self._semantic_plane(roughness_source, int(channels.get("roughness", 2)), w, h, 128)
                material = np.stack((material_plane, emissive_plane, roughness_plane), axis=-1)
            else:
                material = np.stack(
                    (
                        np.zeros((h, w), dtype=np.uint8),
                        np.zeros((h, w), dtype=np.uint8),
                        np.full((h, w), 128, dtype=np.uint8),
                    ),
                    axis=-1,
                )

            family_id = hashlib.sha1(
                f"{selected.canonical_key}|{family_index}|{albedo_path}|{normal_path}|"
                f"{material_path}|{roughness_path}|{glow_path}|{json.dumps(channels, sort_keys=True)}".encode("utf-8")
            ).hexdigest()[:16]
            xs = self._grid_positions(w, crop_size)
            ys = self._grid_positions(h, crop_size)
            cell_count = 0
            for gy, y in enumerate(ys):
                for gx, x in enumerate(xs):
                    a = np.ascontiguousarray(albedo[y:y + crop_size, x:x + crop_size])
                    n = np.ascontiguousarray(normal[y:y + crop_size, x:x + crop_size])
                    m = np.ascontiguousarray(material[y:y + crop_size, x:x + crop_size])
                    if a.shape[:2] != (crop_size, crop_size):
                        continue
                    score = self._detail_score(a, n, m)
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

            provenance = dict(family.get("sourceProvenance") or {})
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
                    "sourceAuthority": "native-eve-full-index-semantic",
                    "sourceProvenance": provenance,
                    "sourceSize": [w, h],
                    "normalSourceSize": [normal_w, normal_h],
                    "nonOverlappingGridCells": cell_count,
                }
            )
            print(
                f"[preview-dataset] Family {family_id} {family.get('shaderFamily') or 'unknown'}: "
                f"albedo={w}x{h} normal={normal_w}x{normal_h}",
                flush=True,
            )

        if not candidates:
            raise RuntimeError("fixed Raven preview dataset has no usable 512x512 texture regions")

        selected_train, selected_validation = self._select_fixed_regions(
            candidates,
            max_train_crops=train_crops,
            max_validation_crops=validation_crops,
            seed=int(config.seed),
        )

        available_validation = sum(1 for item in candidates if bool(item["holdout"]))
        available_training = sum(1 for item in candidates if not bool(item["holdout"]))
        print(
            f"[preview-dataset] Unique non-overlapping {crop_size} regions: {len(candidates)} "
            f"(checkerboard train={available_training}, held-out={available_validation})",
            flush=True,
        )
        print(
            f"[preview-dataset] Requested region caps: train<={train_crops}, held-out<={validation_crops}",
            flush=True,
        )
        print(
            f"[preview-dataset] Selected fixed set: train={len(selected_train)}, held-out={len(selected_validation)}",
            flush=True,
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
                "sourceAuthority": "native-eve-full-index-semantic",
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
                    "albedo_logical": str(item["family"].get("sourceProvenance", {}).get("albedo", {}).get("logical") or ""),
                    "normal_logical": str(item["family"].get("sourceProvenance", {}).get("normal", {}).get("logical") or ""),
                    "material_logical": str(item["family"].get("sourceProvenance", {}).get("material", {}).get("logical") or ""),
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

        fingerprint = self._selection_fingerprint(source_fingerprint, records, seed=config.seed)
        payload: dict[str, Any] = {
            "schema": PREVIEW_DATASET_SCHEMA,
            "sourceFingerprint": source_fingerprint,
            "fingerprint": fingerprint,
            "builderVersion": BUILDER_VERSION,
            "modelScope": "tuning",
            "fixedPreviewSet": True,
            "deterministic": True,
            "authoredSourceAuthority": "native-eve-full-index-semantic",
            "authoredResolutionPolicy": "preserve-native-no-pretraining-upscale",
            "renderTargetIndependentOfTextureResolution": True,
            "minimumAuthoredDimension": crop_size,
            "hullAuthorityProbe": hull_probe,
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
                "type": "feature-stratified-non-overlapping-native-grid-v4",
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
        print("NSAMDR RAVEN NATIVE-AUTHORED DEVELOPMENT DATASET READY", flush=True)
        print(f"Asset                    : {selected.display_name}", flush=True)
        print(f"Selection                 : {selected.canonical_key}", flush=True)
        print("Authored authority        : native EVE full-index semantic textures", flush=True)
        print("Resolution policy         : preserve native; 4096 preview target is independent", flush=True)
        for role in ("albedo", "normal"):
            probe = hull_probe.get(role)
            if isinstance(probe, dict) and probe.get("logical"):
                if probe.get("error"):
                    print(f"Hull {role:6s} probe       : {probe['logical']} ({probe['error']})", flush=True)
                else:
                    print(
                        f"Hull {role:6s} probe       : {probe['width']}x{probe['height']} {probe['logical']}",
                        flush=True,
                    )
        print(f"Texture families          : {len(family_payloads)}", flush=True)
        print(f"Fixed train crops         : {payload['counts']['trainCrops']} (cap {train_crops})", flush=True)
        print(f"Fixed held-out crops      : {payload['counts']['validationCrops']} (cap {validation_crops})", flush=True)
        print(f"Crop geometry             : {crop_size}x{crop_size}, non-overlapping", flush=True)
        print(f"Manifest                  : {manifest_path}", flush=True)
        print("=" * 64, flush=True)
        return payload

    def main(self) -> int:
        parser = argparse.ArgumentParser(
            description="Build the deterministic feature-stratified Raven development dataset"
        )
        parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        parser.add_argument(
            "--config",
            type=Path,
            default=Path("tools/nsamdr/neural/configs/v9_preview_raven.json"),
        )
        parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
        parser.add_argument("--rebuild", action="store_true")
        parser.add_argument("--train-crops", type=int, default=16)
        parser.add_argument("--validation-crops", type=int, default=4)
        args = parser.parse_args()
        repo_root = args.repo_root.resolve()
        config_path = args.config if args.config.is_absolute() else repo_root / args.config
        config = V9Config.load(config_path.resolve())
        self.prepare(
            repo_root,
            config,
            shared_cache=args.shared_cache,
            rebuild=args.rebuild,
            train_crops=max(1, args.train_crops),
            validation_crops=max(1, args.validation_crops),
        )
        return 0


_raven_preview_dataset_preparation_application = RavenPreviewDatasetPreparationApplication()
_find_navy_raven = _raven_preview_dataset_preparation_application._find_navy_raven
_authoritative_rows = _raven_preview_dataset_preparation_application._authoritative_rows
_load_rgb = _raven_preview_dataset_preparation_application._load_rgb
_load_rgba = _raven_preview_dataset_preparation_application._load_rgba
_semantic_plane = _raven_preview_dataset_preparation_application._semantic_plane
_resize_rgb = _raven_preview_dataset_preparation_application._resize_rgb
_normal_rgb = _raven_preview_dataset_preparation_application._normal_rgb
_detail_score = _raven_preview_dataset_preparation_application._detail_score
_grid_positions = _raven_preview_dataset_preparation_application._grid_positions
_texture_families_from_report = _raven_preview_dataset_preparation_application._texture_families_from_report
_manifest_texture_logical_map = _raven_preview_dataset_preparation_application._manifest_texture_logical_map
_native_family_sources = _raven_preview_dataset_preparation_application._native_family_sources
_sha256_file = _raven_preview_dataset_preparation_application._sha256_file
_source_fingerprint = _raven_preview_dataset_preparation_application._source_fingerprint
_selection_fingerprint = _raven_preview_dataset_preparation_application._selection_fingerprint
_select_fixed_regions = _raven_preview_dataset_preparation_application._select_fixed_regions
prepare = _raven_preview_dataset_preparation_application.prepare
main = _raven_preview_dataset_preparation_application.main


if __name__ == "__main__":
    raise SystemExit(main())
