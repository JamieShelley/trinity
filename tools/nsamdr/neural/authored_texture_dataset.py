"""Neutral authored EVE texture discovery and crop-bundle preparation.

The established schema and fingerprint inputs are intentionally retained so
existing NSAMDR manifests, crop bundles, and resume state remain compatible.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image


class AuthoredTextureDatasetConfig(Protocol):
    """Configuration fields required by authored texture extraction."""

    dataset_manifest: str
    dataset_root: str
    max_families: int
    crops_per_family: int
    source_crop_size: int
    min_source_dimension: int
    min_auxiliary_dimension: int
    validation_fraction: float
    seed: int

    def validate(self) -> None: ...


DATASET_SCHEMA = "NSAMDR_EVE_CACHE_ALBEDO_NORMAL_DATASET_V7_1"
CROP_SCHEMA = "NSAMDR_EVE_CACHE_PBR_CROP_V1"
ROLE_SUFFIXES = {
    "albedo": ("_d.dds", "_d.png"),
    "normal": ("_n.dds", "_n.png"),
    "material": ("_pgs.dds", "_pgs.png"),
}


@dataclass(frozen=True)
class TextureSource:
    role: str
    logical: str
    path: str
    width: int
    height: int
    mip_count: int
    format: str


@dataclass(frozen=True)
class PBRFamily:
    family_id: str
    stem: str
    split: str
    albedo: TextureSource
    normal: TextureSource
    material: TextureSource | None


@dataclass(frozen=True)
class CropRecord:
    crop_id: str
    family_id: str
    split: str
    path: str
    source_box: tuple[int, int, int, int]
    detail_score: float
    albedo_logical: str
    normal_logical: str
    material_logical: str


def _stable_fraction(value: str) -> float:
    integer = int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)
    return integer / float(0xFFFFFFFFFFFF)


def _family_split(stem: str, validation_fraction: float) -> str:
    return "validation" if _stable_fraction(stem.lower()) < validation_fraction else "train"


def _strip_role_suffix(logical: str) -> tuple[str, str] | None:
    lower = logical.lower().replace("\\", "/")
    for role, suffixes in ROLE_SUFFIXES.items():
        for suffix in suffixes:
            if lower.endswith(suffix):
                return lower[:-len(suffix)], role
    return None


def parse_dds_header(path: Path) -> tuple[int, int, int, str]:
    """Read dimensions/mips/format without decoding the DDS payload."""
    with path.open("rb") as handle:
        header = handle.read(148)
    if len(header) < 128 or header[:4] != b"DDS ":
        raise ValueError(f"not a DDS file: {path}")
    height, width = struct.unpack_from("<II", header, 12)
    mip_count = struct.unpack_from("<I", header, 28)[0] or 1
    fourcc = header[84:88]
    if fourcc == b"DX10" and len(header) >= 148:
        dxgi = struct.unpack_from("<I", header, 128)[0]
        format_name = f"DXGI_{dxgi}"
    elif fourcc.strip(b"\x00 "):
        format_name = fourcc.decode("ascii", errors="replace")
    else:
        rgb_bits = struct.unpack_from("<I", header, 88)[0]
        format_name = f"RGB{rgb_bits}"
    return int(width), int(height), int(mip_count), format_name


def _cache_fingerprint(indexes: Sequence[Path], config: AuthoredTextureDatasetConfig, source_root: str) -> str:
    digest = hashlib.sha256()
    digest.update(DATASET_SCHEMA.encode("utf-8"))
    digest.update(b"legacy-albedo-normal-material-passthrough-crops-v7.1")
    digest.update(source_root.encode("utf-8", errors="replace"))
    for path in sorted(indexes):
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8", errors="replace"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    for value in (
        config.max_families, config.crops_per_family, config.source_crop_size,
        config.min_source_dimension, config.min_auxiliary_dimension, config.validation_fraction,
    ):
        digest.update(repr(value).encode("ascii"))
    return digest.hexdigest()


def _import_eve_asset_helpers(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.nsamdr import eve_asset_test  # type: ignore
    return eve_asset_test


def _role_minimum(config: AuthoredTextureDatasetConfig, role: str) -> int:
    return config.min_source_dimension if role == "albedo" else config.min_auxiliary_dimension


def _prefer_larger(previous: TextureSource | None, candidate: TextureSource) -> TextureSource:
    if previous is None:
        return candidate
    previous_key = (previous.width * previous.height, min(previous.width, previous.height), previous.mip_count)
    candidate_key = (candidate.width * candidate.height, min(candidate.width, candidate.height), candidate.mip_count)
    return candidate if candidate_key > previous_key else previous


def _map_suffix_histogram(rows: Sequence[object]) -> dict[str, int]:
    suffixes = ("_d.dds", "_n.dds", "_pgs.dds", "_pmdg.dds", "_pgr.dds", "_ar.dds", "_ap.dds", "_no.dds")
    counts: Counter[str] = Counter()
    for row in rows:
        logical = str(getattr(row, "logical", "")).lower().replace("\\", "/")
        if "/model/ship/" not in logical and "/model/structure/" not in logical:
            continue
        for suffix in suffixes:
            if logical.endswith(suffix):
                counts[suffix] += 1
                break
    return dict(sorted(counts.items()))


def _dimension_histogram(values: Sequence[TextureSource]) -> dict[str, int]:
    counts: Counter[str] = Counter(f"{value.width}x{value.height}" for value in values)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _family_missing_roles(grouped: dict[str, dict[str, TextureSource]], limit: int = 24) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            -max((source.width * source.height for source in item[1].values()), default=0),
            item[0],
        ),
    )
    for stem, roles in ranked:
        missing = [role for role in ("albedo", "normal", "material") if role not in roles]
        if not missing:
            continue
        result.append({
            "stem": stem,
            "present": {role: f"{source.width}x{source.height}" for role, source in roles.items()},
            "missing": missing,
        })
        if len(result) >= limit:
            break
    return result


def discover_shared_cache_families(
    repo_root: Path,
    config: AuthoredTextureDatasetConfig,
    shared_cache: str | None,
) -> tuple[list[PBRFamily], dict[str, object]]:
    helpers = _import_eve_asset_helpers(repo_root)
    cache_root, indexes, resfiles = helpers.resolve_layout(shared_cache, allow_prompt=os.name == "nt")
    rows = helpers.read_rows(indexes)

    grouped_all: dict[str, dict[str, TextureSource]] = {}
    grouped_accepted: dict[str, dict[str, TextureSource]] = {}
    role_sources_all: dict[str, list[TextureSource]] = {role: [] for role in ROLE_SUFFIXES}
    role_sources_accepted: dict[str, list[TextureSource]] = {role: [] for role in ROLE_SUFFIXES}
    stats: Counter[str] = Counter()
    role_stats: dict[str, Counter[str]] = {role: Counter() for role in ROLE_SUFFIXES}
    rejected_examples: list[dict[str, object]] = []

    for row in rows:
        stats["indexRows"] += 1
        parsed = _strip_role_suffix(row.logical)
        if parsed is None:
            continue
        stem, role = parsed
        if "/model/ship/" not in stem and "/model/structure/" not in stem:
            continue
        stats["indexedRelevantTextureRows"] += 1
        role_stats[role]["indexed"] += 1
        source_path = helpers.source_path(resfiles, row)
        if not source_path.is_file():
            role_stats[role]["notLocal"] += 1
            continue
        stats["locallyAvailableTextureRows"] += 1
        role_stats[role]["local"] += 1
        try:
            width, height, mip_count, format_name = parse_dds_header(source_path)
        except (OSError, ValueError, struct.error):
            stats["headerParseFailures"] += 1
            role_stats[role]["headerParseFailure"] += 1
            continue

        candidate = TextureSource(
            role, row.logical, str(source_path.resolve()), width, height, mip_count, format_name)
        role_sources_all[role].append(candidate)
        all_bucket = grouped_all.setdefault(stem, {})
        all_bucket[role] = _prefer_larger(all_bucket.get(role), candidate)

        minimum = _role_minimum(config, role)
        if min(width, height) < minimum:
            role_stats[role]["belowMinimum"] += 1
            if len(rejected_examples) < 32:
                rejected_examples.append({
                    "logical": row.logical,
                    "role": role,
                    "dimensions": f"{width}x{height}",
                    "requiredMinimum": minimum,
                })
            continue

        role_stats[role]["accepted"] += 1
        role_sources_accepted[role].append(candidate)
        accepted_bucket = grouped_accepted.setdefault(stem, {})
        accepted_bucket[role] = _prefer_larger(accepted_bucket.get(role), candidate)

    complete_before_threshold = sum(
        1 for roles in grouped_all.values()
        if all(role in roles for role in ("albedo", "normal", "material"))
    )
    paired_before_threshold = sum(
        1 for roles in grouped_all.values()
        if all(role in roles for role in ("albedo", "normal"))
    )
    high_res_albedo_stems = {
        stem for stem, roles in grouped_accepted.items() if "albedo" in roles
    }
    families: list[PBRFamily] = []
    for stem, roles in grouped_accepted.items():
        # Current SharedCache legacy assets usually provide aligned `_d` and `_n`
        # maps but no `_pgs` resource. The real HR pair is still valid training
        # data; packed material remains a runtime passthrough map.
        if not all(role in roles for role in ("albedo", "normal")):
            continue
        family_id = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:16]
        families.append(PBRFamily(
            family_id=family_id,
            stem=stem,
            split=_family_split(stem, config.validation_fraction),
            albedo=roles["albedo"],
            normal=roles["normal"],
            material=roles.get("material"),
        ))
    families.sort(
        key=lambda family: (
            -min(family.albedo.width, family.albedo.height),
            -family.albedo.width * family.albedo.height,
            family.stem,
        )
    )
    selected = families[:config.max_families]
    audit = {
        "totalIndexRows": stats["indexRows"],
        "indexedRelevantTextureRows": stats["indexedRelevantTextureRows"],
        "locallyAvailableTextureRows": stats["locallyAvailableTextureRows"],
        "headerParseFailures": stats["headerParseFailures"],
        "roleThresholds": {
            "albedo": config.min_source_dimension,
            "normal": config.min_auxiliary_dimension,
            "material": config.min_auxiliary_dimension,
        },
        "roleCounts": {role: dict(values) for role, values in role_stats.items()},
        "roleDimensionsAll": {role: _dimension_histogram(values) for role, values in role_sources_all.items()},
        "roleDimensionsAccepted": {role: _dimension_histogram(values) for role, values in role_sources_accepted.items()},
        "mapSuffixCounts": _map_suffix_histogram(rows),
        "candidateStemsBeforeThreshold": len(grouped_all),
        "completeFamiliesBeforeThreshold": complete_before_threshold,
        "pairedAlbedoNormalBeforeThreshold": paired_before_threshold,
        "highResolutionAlbedoStems": len(high_res_albedo_stems),
        "pairedAlbedoNormalAfterThreshold": len(families),
        "materialSupervisedFamiliesAfterThreshold": sum(1 for family in families if family.material is not None),
        "completeFamiliesAfterThreshold": sum(1 for family in families if family.material is not None),
        "missingRoleExamples": _family_missing_roles(grouped_accepted),
        "dimensionRejectedExamples": rejected_examples,
    }
    metadata = {
        "sourceType": "eve-shared-cache",
        "cacheRoot": str(cache_root),
        "indexes": [str(path) for path in indexes],
        "resfiles": str(resfiles),
        "indexedTextureRows": stats["indexedRelevantTextureRows"],
        "locallyAvailableTextureRows": stats["locallyAvailableTextureRows"],
        "completePbrFamilies": sum(1 for family in families if family.material is not None),
        "completePbrFamiliesBeforeThreshold": complete_before_threshold,
        "pairedAlbedoNormalFamilies": len(families),
        "pairedAlbedoNormalFamiliesBeforeThreshold": paired_before_threshold,
        "materialSupervisedFamilies": sum(1 for family in families if family.material is not None),
        "incompleteFamiliesRejected": max(0, len(grouped_accepted) - len(families)),
        "selectedFamilies": len(selected),
        "discoveryAudit": audit,
        "fingerprint": _cache_fingerprint(indexes, config, str(cache_root)),
    }
    return selected, metadata

def discover_extracted_families(source_root: Path, config: AuthoredTextureDatasetConfig) -> tuple[list[PBRFamily], dict[str, object]]:
    grouped: dict[str, dict[str, TextureSource]] = {}
    files = [path for path in source_root.rglob("*") if path.is_file()]
    role_counts: dict[str, Counter[str]] = {role: Counter() for role in ROLE_SUFFIXES}
    for path in files:
        parsed = _strip_role_suffix(path.name.lower())
        if parsed is None:
            continue
        name_stem, role = parsed
        logical_stem = str(path.parent.resolve()).lower().replace("\\", "/") + "/" + name_stem
        role_counts[role]["indexed"] += 1
        try:
            if path.suffix.lower() == ".dds":
                width, height, mip_count, format_name = parse_dds_header(path)
            else:
                with Image.open(path) as image:
                    width, height = image.size
                mip_count, format_name = 1, path.suffix.lower().lstrip(".").upper()
        except (OSError, ValueError, struct.error):
            role_counts[role]["headerParseFailure"] += 1
            continue
        minimum = _role_minimum(config, role)
        if min(width, height) < minimum:
            role_counts[role]["belowMinimum"] += 1
            continue
        role_counts[role]["accepted"] += 1
        candidate = TextureSource(role, str(path), str(path.resolve()), width, height, mip_count, format_name)
        bucket = grouped.setdefault(logical_stem, {})
        bucket[role] = _prefer_larger(bucket.get(role), candidate)

    families: list[PBRFamily] = []
    for stem, roles in grouped.items():
        if not all(role in roles for role in ("albedo", "normal")):
            continue
        family_id = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:16]
        families.append(PBRFamily(
            family_id, stem, _family_split(stem, config.validation_fraction),
            roles["albedo"], roles["normal"], roles.get("material"),
        ))
    families.sort(key=lambda family: (-family.albedo.width * family.albedo.height, family.stem))
    families = families[:config.max_families]
    return families, {
        "sourceType": "extracted-texture-directory",
        "sourceRoot": str(source_root.resolve()),
        "completePbrFamilies": sum(1 for family in families if family.material is not None),
        "pairedAlbedoNormalFamilies": len(families),
        "materialSupervisedFamilies": sum(1 for family in families if family.material is not None),
        "selectedFamilies": len(families),
        "discoveryAudit": {
            "roleThresholds": {
                "albedo": config.min_source_dimension,
                "normal": config.min_auxiliary_dimension,
                "material": config.min_auxiliary_dimension,
            },
            "roleCounts": {role: dict(values) for role, values in role_counts.items()},
            "candidateStemsAfterThreshold": len(grouped),
            "pairedAlbedoNormalAfterThreshold": len(families),
            "completeFamiliesAfterThreshold": sum(1 for family in families if family.material is not None),
        },
        "fingerprint": hashlib.sha256(
            "\n".join(f"{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}" for path in sorted(files)).encode("utf-8")
        ).hexdigest(),
    }

def _decode_source(
    source: TextureSource,
    output: Path,
    repo_root: Path,
    converter: tuple[str, Path] | None,
) -> Path:
    path = Path(source.path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".dds" and path.name.lower().endswith((".png", ".jpg", ".jpeg", ".tga")):
        shutil.copy2(path, output)
        return output
    if converter is None:
        helpers = _import_eve_asset_helpers(repo_root)
        converter_dir = repo_root / "tools" / "nsamdr" / "gr2_converter"
        converter = helpers.ensure_converter(converter_dir)
    node, script = converter
    result = subprocess.run(
        [node, str(script), "dds-to-png", str(path), str(output)],
        cwd=script.parent,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if result.returncode != 0 or not output.is_file():
        diagnostic = (result.stderr or result.stdout or "DDS conversion failed").strip()
        raise RuntimeError(f"Could not decode {source.logical}:\n{diagnostic[-2000:]}")
    return output


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _renormalize_normal_xy(normal_xy: np.ndarray) -> np.ndarray:
    length = np.sqrt(np.maximum((normal_xy ** 2).sum(axis=-1, keepdims=True), 1.0e-8))
    return normal_xy / np.maximum(1.0, length / 0.999)


def load_normal_training_rgb(path: Path) -> tuple[np.ndarray, str]:
    """Decode an EVE legacy `_n` map into explicit X/Y/Z training channels.

    Legacy ship normal maps store tangent-space X in alpha and Y in green. Some
    extracted sources lose alpha; those fall back to red/green. The returned
    RGB array always stores encoded X in R and encoded Y in G so training and
    inference use the same semantic channel layout.
    """
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]
    # A constant opaque alpha indicates an RGB-only extraction rather than a
    # real alpha-encoded normal component.
    use_alpha = float(alpha.std()) > 0.75 and int(alpha.max()) - int(alpha.min()) > 3
    encoded_x = alpha if use_alpha else rgba[..., 0]
    encoded_y = rgba[..., 1]
    xy = np.stack((encoded_x, encoded_y), axis=-1).astype(np.float32) / 127.5 - 1.0
    xy = _renormalize_normal_xy(xy)
    z = np.sqrt(np.clip(1.0 - np.square(xy).sum(axis=-1), 0.0, 1.0))
    encoded_z = np.uint8(np.round(np.clip(z * 255.0, 0.0, 255.0)))
    training_rgb = np.stack((encoded_x, encoded_y, encoded_z), axis=-1).astype(np.uint8)
    return training_rgb, ("alpha-green" if use_alpha else "red-green-fallback")


def _resize_rgb(array: np.ndarray, size: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return np.asarray(Image.fromarray(array, mode="RGB").resize(size, mode), dtype=np.uint8)


def _crop_aligned(array: np.ndarray, normalized_box: tuple[float, float, float, float], output_size: int, *, nearest: bool) -> np.ndarray:
    height, width = array.shape[:2]
    x0 = max(0, min(width - 1, int(round(normalized_box[0] * width))))
    y0 = max(0, min(height - 1, int(round(normalized_box[1] * height))))
    x1 = max(x0 + 1, min(width, int(round(normalized_box[2] * width))))
    y1 = max(y0 + 1, min(height, int(round(normalized_box[3] * height))))
    cropped = array[y0:y1, x0:x1]
    return _resize_rgb(cropped, (output_size, output_size), nearest=nearest)


def detail_map(albedo: np.ndarray, normal: np.ndarray, material: np.ndarray) -> np.ndarray:
    # Score high-resolution regions using aligned albedo, normal and material boundaries.
    target = 512
    def plane(array: np.ndarray) -> np.ndarray:
        image = Image.fromarray(array, mode="RGB")
        image.thumbnail((target, target), Image.Resampling.BILINEAR)
        resized = np.asarray(image, dtype=np.float32) / 255.0
        return resized
    a = plane(albedo)
    n = plane(normal)
    m = plane(material)
    height = min(a.shape[0], n.shape[0], m.shape[0])
    width = min(a.shape[1], n.shape[1], m.shape[1])
    a, n, m = a[:height, :width], n[:height, :width], m[:height, :width]
    luma = a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722
    score = np.zeros((height, width), dtype=np.float32)
    for signal, weight in ((luma[..., None], 1.0), (n, 1.35), (m, 1.15)):
        gx = np.diff(signal, axis=1, append=signal[:, -1:, :])
        gy = np.diff(signal, axis=0, append=signal[-1:, :, :])
        score += weight * np.sqrt(np.square(gx) + np.square(gy)).mean(axis=-1)
    return score


def _score_box(score: np.ndarray, box: tuple[float, float, float, float]) -> float:
    height, width = score.shape
    x0 = max(0, min(width - 1, int(box[0] * width)))
    y0 = max(0, min(height - 1, int(box[1] * height)))
    x1 = max(x0 + 1, min(width, int(math.ceil(box[2] * width))))
    y1 = max(y0 + 1, min(height, int(math.ceil(box[3] * height))))
    region = score[y0:y1, x0:x1]
    return float(region.mean() + region.max(initial=0.0) * 0.35)


def _select_crop_boxes(
    albedo: np.ndarray,
    normal: np.ndarray,
    material: np.ndarray,
    crop_size: int,
    count: int,
    rng: random.Random,
) -> list[tuple[tuple[float, float, float, float], float]]:
    height, width = albedo.shape[:2]
    fraction_x = min(1.0, crop_size / max(width, 1))
    fraction_y = min(1.0, crop_size / max(height, 1))
    score = detail_map(albedo, normal, material)
    candidates: list[tuple[float, tuple[float, float, float, float]]] = []
    for _ in range(max(96, count * 16)):
        x0 = rng.uniform(0.0, max(0.0, 1.0 - fraction_x))
        y0 = rng.uniform(0.0, max(0.0, 1.0 - fraction_y))
        box = (x0, y0, min(1.0, x0 + fraction_x), min(1.0, y0 + fraction_y))
        candidates.append((_score_box(score, box), box))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[tuple[float, float, float, float], float]] = []
    detail_count = max(1, round(count * 0.75))
    for value, box in candidates:
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        if any((cx - (old[0][0] + old[0][2]) * 0.5) ** 2 + (cy - (old[0][1] + old[0][3]) * 0.5) ** 2 < 0.015 for old in selected):
            continue
        selected.append((box, value))
        if len(selected) >= detail_count:
            break
    while len(selected) < count:
        value, box = rng.choice(candidates[max(0, len(candidates) // 2):])
        selected.append((box, value))
    return selected[:count]


def _write_discovery_report(output_root: Path, source_metadata: dict[str, object]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "discovery_report.json"
    path.write_text(json.dumps(source_metadata, indent=2) + "\n", encoding="utf-8")
    return path


def _print_discovery_audit(source_metadata: dict[str, object], report_path: Path) -> None:
    audit = source_metadata.get("discoveryAudit")
    if not isinstance(audit, dict):
        return
    print("============================================================", flush=True)
    print("NSAMDR AUTHORED TEXTURE CACHE DISCOVERY AUDIT", flush=True)
    print("============================================================", flush=True)
    if "cacheRoot" in source_metadata:
        print(f"EVE SharedCache         : {source_metadata['cacheRoot']}", flush=True)
        print(f"Index files             : {len(source_metadata.get('indexes', []))}", flush=True)
    elif "sourceRoot" in source_metadata:
        print(f"Extracted source root   : {source_metadata['sourceRoot']}", flush=True)
    print(f"Relevant indexed rows   : {audit.get('indexedRelevantTextureRows', source_metadata.get('indexedTextureRows', 0))}", flush=True)
    print(f"Local texture rows      : {audit.get('locallyAvailableTextureRows', source_metadata.get('locallyAvailableTextureRows', 0))}", flush=True)
    thresholds = audit.get("roleThresholds", {})
    if isinstance(thresholds, dict):
        print(
            "Dimension thresholds    : "
            f"albedo>={thresholds.get('albedo', '?')} "
            f"normal>={thresholds.get('normal', '?')} "
            f"PGS>={thresholds.get('material', '?')}",
            flush=True,
        )
    role_counts = audit.get("roleCounts", {})
    if isinstance(role_counts, dict):
        for role in ("albedo", "normal", "material"):
            values = role_counts.get(role, {})
            if isinstance(values, dict):
                print(
                    f"{role.title():22s}: indexed={values.get('indexed', 0)} "
                    f"local={values.get('local', values.get('indexed', 0))} "
                    f"accepted={values.get('accepted', 0)} "
                    f"below-min={values.get('belowMinimum', 0)} "
                    f"not-local={values.get('notLocal', 0)}",
                    flush=True,
                )
    print(f"Complete PBR before    : {audit.get('completeFamiliesBeforeThreshold', '?')}", flush=True)
    print(f"Albedo+normal before   : {audit.get('pairedAlbedoNormalBeforeThreshold', '?')}", flush=True)
    print(f"Albedo+normal accepted : {audit.get('pairedAlbedoNormalAfterThreshold', source_metadata.get('pairedAlbedoNormalFamilies', 0))}", flush=True)
    print(f"Material-supervised    : {audit.get('materialSupervisedFamiliesAfterThreshold', source_metadata.get('materialSupervisedFamilies', 0))}", flush=True)
    suffixes = audit.get("mapSuffixCounts", {})
    if isinstance(suffixes, dict) and suffixes:
        print("Indexed map suffixes    : " + ", ".join(f"{key}={value}" for key, value in suffixes.items()), flush=True)
    print(f"Discovery report        : {report_path}", flush=True)
    print("============================================================", flush=True)


def prepare_dataset(
    repo_root: Path,
    config: AuthoredTextureDatasetConfig,
    *,
    shared_cache: str | None = None,
    source_root: Path | None = None,
    rebuild: bool = False,
    audit_only: bool = False,
) -> dict[str, object]:
    """Create aligned high-resolution PBR crop bundles and a manifest."""
    config.validate()
    output_root = Path(config.dataset_root)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    manifest_path = Path(config.dataset_manifest)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    crop_root = output_root / "crops"

    if source_root is not None:
        families, source_metadata = discover_extracted_families(source_root.resolve(), config)
    else:
        families, source_metadata = discover_shared_cache_families(repo_root, config, shared_cache)
    report_path = _write_discovery_report(output_root, source_metadata)
    _print_discovery_audit(source_metadata, report_path)
    if audit_only:
        return {
            "schema": DATASET_SCHEMA,
            "auditOnly": True,
            "source": source_metadata,
            "candidateFamilies": len(families),
            "discoveryReport": str(report_path),
        }
    fingerprint = str(source_metadata["fingerprint"])

    if not rebuild and manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("schema") == DATASET_SCHEMA and existing.get("fingerprint") == fingerprint:
                crop_paths = [Path(record["path"]) for record in existing.get("crops", [])]
                if crop_paths and all(path.is_file() for path in crop_paths):
                    print(f"NSAMDR authored texture dataset is current: {manifest_path}", flush=True)
                    return existing
        except (OSError, ValueError, TypeError):
            pass

    if not families:
        audit = source_metadata.get("discoveryAudit", {})
        paired_before = audit.get("pairedAlbedoNormalBeforeThreshold", 0) if isinstance(audit, dict) else 0
        if paired_before:
            reason = (
                "Aligned albedo+normal families exist, but the configured dimension limits rejected them. "
                f"Albedo requires >= {config.min_source_dimension}px and normal requires >= "
                f"{config.min_auxiliary_dimension}px."
            )
        else:
            reason = "No locally available legacy family had both aligned `_d` and `_n` maps."
        raise RuntimeError(
            "No usable EVE albedo+normal training families were found. " + reason + "\n"
            f"Discovery report: {report_path}\n"
            "Verify the SharedCache or pass --source-root with extracted matching `_d` and `_n` maps."
        )

    if crop_root.exists():
        shutil.rmtree(crop_root)
    crop_root.mkdir(parents=True, exist_ok=True)
    converter: tuple[str, Path] | None = None
    if any(
        Path(source.path).suffix.lower() == ".dds" or not Path(source.path).suffix
        for family in families
        for source in (family.albedo, family.normal, family.material)
        if source is not None
    ):
        helpers = _import_eve_asset_helpers(repo_root)
        converter = helpers.ensure_converter(repo_root / "tools" / "nsamdr" / "gr2_converter")

    print("============================================================", flush=True)
    print("NSAMDR AUTHORED TEXTURE DATASET PREPARATION", flush=True)
    print("============================================================", flush=True)
    print(f"Source type             : {source_metadata['sourceType']}", flush=True)
    if "cacheRoot" in source_metadata:
        print(f"EVE SharedCache         : {source_metadata['cacheRoot']}", flush=True)
        print(f"Indexed texture rows    : {source_metadata.get('indexedTextureRows', 0)}", flush=True)
        print(f"Local texture rows      : {source_metadata.get('locallyAvailableTextureRows', 0)}", flush=True)
    elif "sourceRoot" in source_metadata:
        print(f"Extracted source root   : {source_metadata['sourceRoot']}", flush=True)
    print(f"Albedo+normal families  : {source_metadata.get('pairedAlbedoNormalFamilies', len(families))}", flush=True)
    print(f"Material-supervised     : {source_metadata.get('materialSupervisedFamilies', 0)}", flush=True)
    print(f"Selected families       : {len(families)}", flush=True)
    print(f"Crops per family        : {config.crops_per_family}", flush=True)
    print(f"Stored HR crop          : {config.source_crop_size}x{config.source_crop_size}", flush=True)
    print("Training target         : highest authored albedo + aligned authored normal", flush=True)
    print("Training input          : generated soft client-like LR from those real HR targets", flush=True)
    print("Maps reconstructed      : albedo RGB + legacy normal alpha/green XY", flush=True)
    print("Packed material policy  : original runtime map is preserved; no invented PGS ground truth", flush=True)
    print("Family split            : complete albedo/normal family, never random crop leakage", flush=True)
    print("============================================================", flush=True)

    records: list[CropRecord] = []
    family_payloads: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="nsamdr_authored_decode_") as temp_name:
        temp = Path(temp_name)
        for family_index, family in enumerate(families, 1):
            decoded: dict[str, Path] = {}
            for source in (family.albedo, family.normal, family.material):
                if source is None:
                    continue
                decoded[source.role] = _decode_source(
                    source, temp / f"{family.family_id}_{source.role}.png", repo_root, converter)
            albedo = _load_rgb(decoded["albedo"])
            normal, normal_encoding = load_normal_training_rgb(decoded["normal"])
            material_valid = family.material is not None and "material" in decoded
            if material_valid:
                material = _load_rgb(decoded["material"])
            else:
                # Neutral placeholder exists only to preserve the current tensor/API
                # shape. The model ignores it and the losses mask it completely.
                material = np.full((albedo.shape[0], albedo.shape[1], 3), 128, dtype=np.uint8)
            rng = random.Random(config.seed ^ int(family.family_id[:8], 16))
            boxes = _select_crop_boxes(
                albedo, normal, material, config.source_crop_size, config.crops_per_family, rng)
            family_crop_count = 0
            for crop_index, (box, detail_score) in enumerate(boxes):
                crop_id = f"{family.family_id}_{crop_index:03d}"
                destination = crop_root / family.split / f"{crop_id}.npz"
                destination.parent.mkdir(parents=True, exist_ok=True)
                albedo_crop = _crop_aligned(albedo, box, config.source_crop_size, nearest=False)
                normal_crop = _crop_aligned(normal, box, config.source_crop_size, nearest=False)
                material_crop = _crop_aligned(material, box, config.source_crop_size, nearest=True)
                metadata = {
                    "schema": CROP_SCHEMA,
                    "cropId": crop_id,
                    "familyId": family.family_id,
                    "familyStem": family.stem,
                    "split": family.split,
                    "normalizedBox": list(box),
                    "detailScore": detail_score,
                    "sources": {
                        "albedo": asdict(family.albedo),
                        "normal": asdict(family.normal),
                        "material": asdict(family.material) if family.material is not None else None,
                    },
                    "normalEncoding": normal_encoding,
                    "materialEncoding": "legacy-pgs-rgb" if material_valid else "runtime-passthrough-unsupervised",
                    "materialSupervision": bool(material_valid),
                }
                np.savez_compressed(
                    destination,
                    albedo=albedo_crop,
                    normal=normal_crop,
                    material=material_crop,
                    material_valid=np.asarray([1.0 if material_valid else 0.0], dtype=np.float32),
                    metadata=np.asarray(json.dumps(metadata), dtype=np.str_),
                )
                records.append(CropRecord(
                    crop_id=crop_id,
                    family_id=family.family_id,
                    split=family.split,
                    path=str(destination.resolve()),
                    source_box=tuple(int(round(value * 1_000_000)) for value in box),
                    detail_score=detail_score,
                    albedo_logical=family.albedo.logical,
                    normal_logical=family.normal.logical,
                    material_logical=family.material.logical if family.material is not None else "",
                ))
                family_crop_count += 1
            family_payloads.append({
                "familyId": family.family_id,
                "stem": family.stem,
                "split": family.split,
                "albedo": asdict(family.albedo),
                "normal": asdict(family.normal),
                "material": asdict(family.material) if family.material is not None else None,
                "normalEncoding": normal_encoding,
                "materialEncoding": "legacy-pgs-rgb" if material_valid else "runtime-passthrough-unsupervised",
                "materialSupervision": bool(material_valid),
                "cropCount": family_crop_count,
            })
            print(
                f"  [{family_index:03d}/{len(families):03d}] {family.split:10s} "
                f"{family.albedo.width}x{family.albedo.height} mips={family.albedo.mip_count:2d} "
                f"crops={family_crop_count:2d} {family.stem}",
                flush=True,
            )

    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]
    if not validation_records and len(families) > 1:
        # Deterministically move one complete family to validation.
        validation_family = families[-1].family_id
        for index, record in enumerate(records):
            if record.family_id == validation_family:
                records[index] = CropRecord(**{**asdict(record), "split": "validation"})
        for family in family_payloads:
            if family["familyId"] == validation_family:
                family["split"] = "validation"
        train_records = [record for record in records if record.split == "train"]
        validation_records = [record for record in records if record.split == "validation"]

    payload: dict[str, object] = {
        "schema": DATASET_SCHEMA,
        "fingerprint": fingerprint,
        "source": source_metadata,
        "config": {
            "maxFamilies": config.max_families,
            "cropsPerFamily": config.crops_per_family,
            "sourceCropSize": config.source_crop_size,
            "minSourceDimension": config.min_source_dimension,
            "minAuxiliaryDimension": config.min_auxiliary_dimension,
            "validationFraction": config.validation_fraction,
        },
        "trainingTarget": "highest-authored-albedo-plus-highest-aligned-normal-material-passthrough",
        "trainingInput": "albedo-normal-soft-client-degradation-at-half-resolution",
        "materialPolicy": "runtime-source-passthrough-no-material-reconstruction",
        "families": family_payloads,
        "crops": [asdict(record) for record in records],
        "counts": {
            "families": len(families),
            "trainFamilies": len({record.family_id for record in train_records}),
            "validationFamilies": len({record.family_id for record in validation_records}),
            "trainCrops": len(train_records),
            "validationCrops": len(validation_records),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("============================================================", flush=True)
    print("NSAMDR AUTHORED TEXTURE DATASET READY", flush=True)
    print(f"Manifest                : {manifest_path}", flush=True)
    print(f"Train families/crops    : {payload['counts']['trainFamilies']} / {payload['counts']['trainCrops']}", flush=True)  # type: ignore[index]
    print(f"Validation families/crops: {payload['counts']['validationFamilies']} / {payload['counts']['validationCrops']}", flush=True)  # type: ignore[index]
    print("============================================================", flush=True)
    return payload
