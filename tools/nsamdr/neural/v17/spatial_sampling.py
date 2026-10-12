"""V17.3 native authored spatial sampling; no held-out/test source decoding."""
from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np
import torch

try:
    from ..authored_texture_dataset import AuthoredTextureDatasetRepository, TextureSource
    from ..v14.dataset import _degrade, _normalise_xy
    from ..v16.broad_prior import _detail_map
except ImportError:
    from authored_texture_dataset import AuthoredTextureDatasetRepository, TextureSource
    from v14.dataset import _degrade, _normalise_xy
    from v16.broad_prior import _detail_map


def _id(record: dict[str, Any]) -> str:
    return str(record.get("family_id") or record.get("familyId") or "")


def _crop_id(record: dict[str, Any]) -> str:
    return str(record.get("crop_id") or record.get("cropId") or "")


def _family_ids(crops: list[str]) -> set[str]:
    return {str(crop).rsplit("_", 1)[0] for crop in crops}


def _partitions(manifest: dict[str, Any], v2: dict[str, Any],
                *, train_count: int = 12, dev_count: int = 4,
                test_count: int = 8, min_size: int = 1024) -> dict[str, list[dict[str, Any]]]:
    """Select disjoint *whole* authorities without reading any sealed pixels.

    V17.3 must not revisit ANY V2 crop authority: full native-source sampling
    could otherwise reveal the V2 sealed sibling even without opening its NPZ.
    V17.3 sealed final-test authorities are selected from untouched validation
    authorities, never decoded during training/checkpoint selection.
    """
    if min(train_count, dev_count, test_count) < 1:
        raise ValueError("all partition sizes must be positive")
    if str(v2.get("schema")) != "NSAMDR_V17_GENERALIZATION_V2_PROOF_V1":
        raise RuntimeError("V17.3 requires the completed authoritative V2 proof")
    protected = _family_ids([
        *(v2.get("trainingCropIds") or []),
        *(v2.get("validationCropIds") or []),
        *(v2.get("finalTestCropIds") or []),
    ])
    if not protected:
        raise RuntimeError("V2 proof does not define protected authorities")
    families = {str(f.get("familyId") or f.get("family_id") or ""): f
                for f in manifest.get("families", [])}
    by_split: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "validation": {}}
    for item in manifest.get("crops", []):
        split = str(item.get("split") or "")
        family = _id(item)
        if split not in by_split or family in protected or not family:
            continue
        if not _crop_id(item).endswith("_000"):
            continue
        f = families.get(family, {})
        a, n = f.get("albedo") or {}, f.get("normal") or {}
        if min(int(a.get("width") or 0), int(a.get("height") or 0),
               int(n.get("width") or 0), int(n.get("height") or 0)) < min_size:
            continue
        if not a.get("path") or not n.get("path") or not item.get("path"):
            continue
        by_split[split][family] = dict(item)
    training = [by_split["train"][k] for k in sorted(by_split["train"])[:train_count]]
    holdouts = [by_split["validation"][k] for k in sorted(by_split["validation"])]
    validation = holdouts[:dev_count]
    final_test = holdouts[dev_count:dev_count + test_count]
    if (len(training), len(validation), len(final_test)) != (train_count, dev_count, test_count):
        raise RuntimeError(
            "insufficient independent native authorities: "
            f"train={len(training)}/{train_count}, "
            f"validation={len(validation)}/{dev_count}, "
            f"sealedFinalTest={len(final_test)}/{test_count}"
        )
    subsets = {"training": training, "validation": validation, "final_test": final_test}
    identity = [_id(item) for rows in subsets.values() for item in rows]
    if len(identity) != len(set(identity)) or protected.intersection(identity):
        raise RuntimeError("V17.3 authority leakage across V2, training, development or final-test")
    return subsets


@dataclass
class NativeFamily:
    authority_id: str
    albedo: np.ndarray        # genuine native 8-bit RGB pixels
    normal: np.ndarray        # decoded encoded-normal XY, native map dimensions
    material: np.ndarray | None
    fixed_origin: tuple[int, int]
    detail_integral: np.ndarray
    normal_encoding: str
    material_supervised: bool


def _detail_integral(albedo: np.ndarray, normal: np.ndarray) -> np.ndarray:
    # One-time low-resolution importance map avoids rescoring a full atlas
    # at every training update.
    factor = max(1, min(albedo.shape[:2]) // 256)
    a = albedo[::factor, ::factor].astype(np.float32) / 255.0
    n = normal[::max(1, normal.shape[0] // a.shape[0]),
               ::max(1, normal.shape[1] // a.shape[1]), :2]
    n = cv2.resize(n, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    n = _normalise_xy(n.astype(np.float32) / 127.5 - 1.0)
    score = _detail_map(a, n)
    return cv2.integral(score.astype(np.float32), sdepth=cv2.CV_64F)


def _score_box(integral: np.ndarray, x: int, y: int, size: int,
               source_width: int, source_height: int) -> float:
    h, w = integral.shape[0] - 1, integral.shape[1] - 1
    x0 = min(w - 1, max(0, int(x * w / source_width)))
    y0 = min(h - 1, max(0, int(y * h / source_height)))
    x1 = min(w, max(x0 + 1, int(np.ceil((x + size) * w / source_width))))
    y1 = min(h, max(y0 + 1, int(np.ceil((y + size) * h / source_height))))
    total = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    return float(total / ((x1 - x0) * (y1 - y0)))


def _fixed_origin(record: dict[str, Any], shape: tuple[int, int], size: int) -> tuple[int, int]:
    vals = record.get("source_box") or record.get("sourceBox")
    if not isinstance(vals, (list, tuple)) or len(vals) != 4:
        raise RuntimeError(f"crop {_crop_id(record)} has no normalized source provenance")
    y, x = shape
    center_x = ((float(vals[0]) + float(vals[2])) / 2e6) * x
    center_y = ((float(vals[1]) + float(vals[3])) / 2e6) * y
    return (max(0, min(x - size, round(center_x - size / 2))),
            max(0, min(y - size, round(center_y - size / 2))))


def _aligned_crop(array: np.ndarray, x: int, y: int, size: int,
                  source_shape: tuple[int, int], *, nearest: bool) -> np.ndarray:
    source_h, source_w = source_shape
    h, w = array.shape[:2]
    x0 = max(0, min(w - 1, round(x * w / source_w)))
    y0 = max(0, min(h - 1, round(y * h / source_h)))
    x1 = max(x0 + 1, min(w, round((x + size) * w / source_w)))
    y1 = max(y0 + 1, min(h, round((y + size) * h / source_h)))
    tile = array[y0:y1, x0:x1]
    if tile.shape[:2] == (size, size):
        return np.ascontiguousarray(tile)
    return cv2.resize(tile, (size, size),
                      interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


class NativeAuthoritySampler:
    """Decode only the 12 training authorities, once; sample moving HR regions."""

    def __init__(self, repo: Path, manifest: dict[str, Any],
                 records: list[dict[str, Any]], *, hr_size: int = 512,
                 seed: int = 17301) -> None:
        self.hr_size = int(hr_size)
        self.seed = int(seed)
        if self.hr_size != 512:
            raise ValueError("V17.3 matched experiment uses authored 512 HR crop size")
        loader = AuthoredTextureDatasetRepository()
        indexed = {str(f.get("familyId") or ""): f for f in manifest.get("families", [])}
        self.families: dict[str, NativeFamily] = {}
        self.audit: dict[str, Any] = {}
        self._scratch = tempfile.TemporaryDirectory(prefix="nsamdr_v173_native_")
        tmp = Path(self._scratch.name)
        try:
            for item in records:
                authority = _id(item)
                family = indexed.get(authority)
                if family is None:
                    raise RuntimeError(f"missing native source metadata: {authority}")
                sources: dict[str, np.ndarray] = {}
                metadata: dict[str, Any] = {}
                for role in ("albedo", "normal", "material"):
                    meta = family.get(role)
                    if meta is None:
                        continue
                    if not Path(str(meta.get("path") or "")).is_file():
                        if role == "material":
                            continue
                        raise RuntimeError(f"missing {role} authored source for {authority}")
                    source = TextureSource(**meta)
                    converted = loader._decode_source(
                        source, tmp / f"{authority}_{role}.png", repo, None,
                    )
                    if role == "normal":
                        value, encoding = loader.load_normal_training_rgb(converted)
                        sources[role] = np.ascontiguousarray(value[..., :2])
                        metadata["normalEncoding"] = encoding
                    else:
                        sources[role] = loader._load_rgb(converted)
                    converted.unlink(missing_ok=True)
                a, n = sources["albedo"], sources["normal"]
                if min(*a.shape[:2], *n.shape[:2]) < 1024:
                    raise RuntimeError(f"V17.3 native source below 1024: {authority}")
                fixed = _fixed_origin(item, a.shape[:2], self.hr_size)
                self.families[authority] = NativeFamily(
                    authority, a, n, sources.get("material"), fixed,
                    _detail_integral(a, n),
                    metadata.get("normalEncoding", "unknown"),
                    "material" in sources,
                )
                self.audit[authority] = {
                    "albedoSize": [int(a.shape[1]), int(a.shape[0])],
                    "normalSize": [int(n.shape[1]), int(n.shape[0])],
                    "normalEncoding": metadata.get("normalEncoding", "unknown"),
                    "materialSupervised": "material" in sources,
                    "fixedOrigin": list(fixed),
                }
                print(f"[v17-spatial] native-ready authority={authority} "
                      f"albedo={a.shape[1]}x{a.shape[0]} "
                      f"normal={n.shape[1]}x{n.shape[0]} "
                      f"normalEncoding={metadata.get('normalEncoding','unknown')} "
                      f"materialSupervised={'material' in sources}", flush=True)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._scratch.cleanup()

    def _origin(self, f: NativeFamily, exposure: int,
                mode: str) -> tuple[int, int, bool]:
        if mode == "fixed":
            return *f.fixed_origin, False
        if mode != "spatial":
            raise ValueError(f"unknown V17.3 sampling arm: {mode}")
        rng = random.Random(self.seed + exposure * 1_000_003
                            + int(f.authority_id[:8], 16))
        a = f.albedo
        max_x, max_y = a.shape[1] - self.hr_size, a.shape[0] - self.hr_size
        # Spatial balance: alternate uniform vs structure-biased by authority exposure.
        detail = bool(exposure % 2)
        if not detail:
            return rng.randint(0, max_x), rng.randint(0, max_y), False
        candidates = [(rng.randint(0, max_x), rng.randint(0, max_y))
                      for _ in range(24)]
        x, y = max(candidates, key=lambda pos: _score_box(
            f.detail_integral, *pos, self.hr_size, a.shape[1], a.shape[0],
        ))
        return x, y, True

    def sample(self, authority: str, exposure: int, *,
               mode: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        f = self.families[authority]
        x, y, detail = self._origin(f, exposure, mode)
        size = self.hr_size
        a = f.albedo[y:y+size, x:x+size].astype(np.float32) / 255.0
        n = _aligned_crop(f.normal, x, y, size, f.albedo.shape[:2],
                          nearest=False).astype(np.float32) / 127.5 - 1.0
        n = _normalise_xy(n)
        if f.material is None:
            m = np.full((size, size, 3), 128.0/255.0, dtype=np.float32)
        else:
            m = _aligned_crop(f.material, x, y, size, f.albedo.shape[:2],
                              nearest=True).astype(np.float32) / 255.0
        lr_a, lr_n, lr_m = _degrade(
            a, n, m, size // 4, "clean", random.Random(self.seed),
        )
        def tensor(rgb: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
        return {
            "lr_albedo": tensor(lr_a),
            "lr_normal": tensor(lr_n),
            "lr_material": tensor(lr_m),
            "target_albedo": tensor(a),
            "target_normal": tensor(n),
            "target_material": tensor(m),
        }, {"authorityId": authority, "origin": [x, y],
            "samplingPolicy": "detail" if detail else "uniform" if mode == "spatial" else "fixed"}

