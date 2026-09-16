#!/usr/bin/env python3
"""Build a broad genuine-4x authored EVE prior corpus from the census.

The split unit is the complete albedo+normal authority, never an individual crop.
This prevents authored-pixel leakage between train and validation.  The builder
uses native authored pixels as HR truth and stores 512x512 crops; V16 creates the
corresponding 128x128 LR input at training time.

Direct PGS/material textures are used when the census exposes them.  Missing full
SOF material semantics are not invented; those authorities keep neutral material
placeholders and are used primarily for albedo+normal structural pretraining.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
NSAMDR_ROOT = HERE.parent
if str(NSAMDR_ROOT) not in sys.path:
    sys.path.insert(0, str(NSAMDR_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import authored_texture_dataset as authored
import eve_asset_test as eve  # type: ignore


SCHEMA = "NSAMDR_V16_AUTHORED_PRIOR_CORPUS_V1"
DEFAULT_OUTPUT = "artifacts/nsamdr/training_v16_authored_prior"
DEFAULT_MANIFEST = f"{DEFAULT_OUTPUT}/dataset_manifest.json"


@dataclass
class PriorCorpusConfig:
    dataset_manifest: str = DEFAULT_MANIFEST
    dataset_root: str = DEFAULT_OUTPUT
    max_families: int = 10_000
    crops_per_family: int = 2
    source_crop_size: int = 512
    min_source_dimension: int = 1024
    min_auxiliary_dimension: int = 512
    validation_fraction: float = 0.12
    seed: int = 16201

    def validate(self) -> None:
        if self.max_families < 1:
            raise ValueError("max_families must be positive")
        if self.crops_per_family < 1:
            raise ValueError("crops_per_family must be positive")
        if self.source_crop_size < 64 or self.source_crop_size % 4:
            raise ValueError("source_crop_size must be >=64 and divisible by four")
        if self.min_source_dimension < self.source_crop_size:
            raise ValueError("min_source_dimension must cover source_crop_size")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in (0, 0.5)")


def _latest_census(repo_root: Path) -> Path:
    root = repo_root / "artifacts/nsamdr/diagnostics/eve_census"
    latest = root / "LATEST.txt"
    if latest.is_file():
        value = Path(latest.read_text(encoding="utf-8").strip())
        if not value.is_absolute():
            value = repo_root / value
        candidate = value / "eve_authored_corpus_census.json"
        if candidate.is_file():
            return candidate.resolve()
    candidates = sorted(root.glob("census_*/eve_authored_corpus_census.json"))
    if not candidates:
        raise RuntimeError("No EVE corpus census exists. Run: scripts\\build\\nsamdr.bat eve-census")
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def _load_census(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("authorities"), list):
        raise RuntimeError(f"Census contains no authority list: {path}")
    return payload


def _texture_source(
    role: str,
    record: dict[str, Any],
    resfiles: Path,
) -> authored.TextureSource:
    hashed = str(record.get("hashed") or "")
    path = resfiles / Path(hashed)
    if not hashed or not path.is_file():
        raise RuntimeError(f"Census resource is not local: {record.get('logical')} -> {path}")
    return authored.TextureSource(
        role=role,
        logical=str(record.get("logical") or ""),
        path=str(path.resolve()),
        width=int(record.get("width") or 0),
        height=int(record.get("height") or 0),
        mip_count=int(record.get("mipCount") or 1),
        format=str(record.get("format") or "unknown"),
    )


def _direct_material(
    authority: dict[str, Any],
    resfiles: Path,
    minimum_dimension: int,
) -> authored.TextureSource | None:
    auxiliaries = dict(authority.get("directAuxiliaryTextures") or {})
    for role in ("pgs", "material"):
        values = auxiliaries.get(role)
        if not isinstance(values, list):
            continue
        candidates = sorted(
            (value for value in values if isinstance(value, dict)),
            key=lambda value: min(int(value.get("width") or 0), int(value.get("height") or 0)),
            reverse=True,
        )
        for value in candidates:
            if min(int(value.get("width") or 0), int(value.get("height") or 0)) < minimum_dimension:
                continue
            try:
                return _texture_source("material", value, resfiles)
            except RuntimeError:
                continue
    return None


class CensusAuthorityRepository(authored.AuthoredTextureDatasetRepository):
    """Use the measured ship-authority census as the only family authority."""

    def __init__(self, census_path: Path) -> None:
        super().__init__()
        self.census_path = census_path.resolve()

    def discover_shared_cache_families(
        self,
        repo_root: Path,
        config: PriorCorpusConfig,
        shared_cache: str | None,
    ) -> tuple[list[authored.PBRFamily], dict[str, object]]:
        cache_root, indexes, resfiles = eve.resolve_layout(shared_cache, allow_prompt=False)
        census = _load_census(self.census_path)
        families: list[authored.PBRFamily] = []
        rejected_not_local = 0
        rejected_dimension = 0

        for authority in census["authorities"]:
            if not isinstance(authority, dict):
                continue
            if int(authority.get("resolutionTier") or 0) < config.min_source_dimension:
                rejected_dimension += 1
                continue
            try:
                albedo = _texture_source("albedo", dict(authority["albedo"]), resfiles)
                normal = _texture_source("normal", dict(authority["normal"]), resfiles)
            except (KeyError, TypeError, RuntimeError):
                rejected_not_local += 1
                continue
            authority_id = str(authority.get("authorityId") or "").strip()
            if not authority_id:
                authority_id = hashlib.sha1(
                    f"{albedo.logical}|{normal.logical}".encode("utf-8")
                ).hexdigest()[:16]
            material = _direct_material(
                authority,
                resfiles,
                config.min_auxiliary_dimension,
            )
            families.append(
                authored.PBRFamily(
                    family_id=authority_id,
                    stem=f"eve-authority/{authority_id}",
                    split=self._family_split(authority_id, config.validation_fraction),
                    albedo=albedo,
                    normal=normal,
                    material=material,
                )
            )

        families.sort(
            key=lambda family: (
                -min(family.albedo.width, family.albedo.height),
                family.family_id,
            )
        )
        families = families[: int(config.max_families)]
        fingerprint_payload = {
            "schema": SCHEMA,
            "censusSha256": hashlib.sha256(self.census_path.read_bytes()).hexdigest(),
            "selectedAuthorityIds": [family.family_id for family in families],
            "cropSize": config.source_crop_size,
            "cropsPerAuthority": config.crops_per_family,
            "validationFraction": config.validation_fraction,
            "minimumDimension": config.min_source_dimension,
            "seed": config.seed,
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        metadata: dict[str, object] = {
            "sourceType": "eve-authored-census-authorities",
            "cacheRoot": str(cache_root),
            "indexes": [str(path) for path in indexes],
            "resfiles": str(resfiles),
            "censusPath": str(self.census_path),
            "pairedAlbedoNormalFamilies": len(families),
            "materialSupervisedFamilies": sum(1 for family in families if family.material is not None),
            "selectedFamilies": len(families),
            "fingerprint": fingerprint,
            "discoveryAudit": {
                "authoritySplitUnit": "complete-albedo-normal-authority",
                "censusAuthorities": len(census["authorities"]),
                "acceptedAuthorities": len(families),
                "rejectedBelowMinimumDimension": rejected_dimension,
                "rejectedNotLocal": rejected_not_local,
                "roleThresholds": {
                    "albedo": config.min_source_dimension,
                    "normal": config.min_source_dimension,
                    "material": config.min_auxiliary_dimension,
                },
                "pairedAlbedoNormalAfterThreshold": len(families),
                "materialSupervisedFamiliesAfterThreshold": sum(
                    1 for family in families if family.material is not None
                ),
            },
        }
        return families, metadata


def build(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    census_path = (
        args.census.resolve()
        if args.census is not None
        else _latest_census(repo_root)
    )
    max_families = int(args.max_authorities) if int(args.max_authorities) > 0 else 10_000
    config = PriorCorpusConfig(
        max_families=max_families,
        crops_per_family=max(1, int(args.crops_per_authority)),
        source_crop_size=int(args.hr_crop_size),
        min_source_dimension=max(int(args.minimum_native_dimension), int(args.hr_crop_size)),
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
    )
    config.validate()
    repository = CensusAuthorityRepository(census_path)
    payload = repository.prepare_dataset(
        repo_root,
        config,
        shared_cache=args.shared_cache,
        rebuild=bool(args.rebuild),
        audit_only=bool(args.audit_only),
    )
    manifest_path = (repo_root / config.dataset_manifest).resolve()
    if args.audit_only:
        return 0, manifest_path

    payload["schema"] = SCHEMA
    payload["censusPath"] = str(census_path)
    payload["authoritySplit"] = "complete-authority-hash-split-no-crop-leakage"
    payload["scale"] = 4
    payload["trainingTarget"] = "native-authored-HR-albedo-normal-with-direct-material-when-available"
    payload["trainingInput"] = "generated-4x-LR-from-authored-HR-at-training-time"
    payload["materialPolicy"] = (
        "direct-census-auxiliary-only-when-present; otherwise-neutral-unsupervised-placeholder"
    )
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    counts = dict(payload.get("counts") or {})
    print("=" * 76, flush=True)
    print("NSAMDR V16 BROAD AUTHORED PRIOR CORPUS READY", flush=True)
    print(f"Census                    : {census_path}", flush=True)
    print(f"Authorities               : {counts.get('families', '?')}", flush=True)
    print(f"Train authorities/crops   : {counts.get('trainFamilies', '?')} / {counts.get('trainCrops', '?')}", flush=True)
    print(f"Held-out authorities/crops: {counts.get('validationFamilies', '?')} / {counts.get('validationCrops', '?')}", flush=True)
    print("Split unit                 : complete authored authority", flush=True)
    print(f"Manifest                   : {manifest_path}", flush=True)
    print("=" * 76, flush=True)
    return 0, manifest_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Build broad authority-disjoint genuine-4x NSAMDR prior corpus"
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    value.add_argument("--census", type=Path)
    value.add_argument("--max-authorities", type=int, default=0, help="0 uses all >= minimum")
    value.add_argument("--crops-per-authority", type=int, default=2)
    value.add_argument("--hr-crop-size", type=int, default=512)
    value.add_argument("--minimum-native-dimension", type=int, default=1024)
    value.add_argument("--validation-fraction", type=float, default=0.12)
    value.add_argument("--seed", type=int, default=16201)
    value.add_argument("--rebuild", action="store_true")
    value.add_argument("--audit-only", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    code, _ = build(parser().parse_args(argv))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
