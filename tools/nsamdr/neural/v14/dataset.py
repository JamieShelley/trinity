from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import V14Config


def _normalise_xy(normal: np.ndarray) -> np.ndarray:
    length = np.sqrt(np.maximum((normal * normal).sum(axis=-1, keepdims=True), 1.0e-8))
    return (normal / np.maximum(1.0, length / 0.999)).astype(np.float32)


def _augment(
    albedo: np.ndarray,
    normal: np.ndarray,
    material: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    turns = rng.randrange(4)
    if turns:
        albedo = np.rot90(albedo, turns).copy()
        normal = np.rot90(normal, turns).copy()
        material = np.rot90(material, turns).copy()
        for _ in range(turns):
            x, y = normal[..., 0].copy(), normal[..., 1].copy()
            normal[..., 0], normal[..., 1] = -y, x
    if rng.random() < 0.5:
        albedo, normal, material = albedo[:, ::-1].copy(), normal[:, ::-1].copy(), material[:, ::-1].copy()
        normal[..., 0] *= -1.0
    if rng.random() < 0.5:
        albedo, normal, material = albedo[::-1].copy(), normal[::-1].copy(), material[::-1].copy()
        normal[..., 1] *= -1.0
    return albedo, _normalise_xy(normal), material


def _degrade(
    albedo_hr: np.ndarray,
    normal_hr: np.ndarray,
    material_hr: np.ndarray,
    lr_size: int,
    mode: str,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    albedo = cv2.resize(albedo_hr, (lr_size, lr_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    normal = cv2.resize(normal_hr, (lr_size, lr_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    material = cv2.resize(material_hr, (lr_size, lr_size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    normal = _normalise_xy(normal)

    if mode == "robust":
        if rng.random() < 0.65:
            sigma = rng.uniform(0.25, 0.70)
            albedo = cv2.GaussianBlur(albedo, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if rng.random() < 0.50:
            levels = rng.choice((127.0, 191.0, 255.0))
            albedo = np.round(np.clip(albedo, 0.0, 1.0) * levels) / levels
        if rng.random() < 0.35:
            luma = albedo[..., 0:1] * 0.2126 + albedo[..., 1:2] * 0.7152 + albedo[..., 2:3] * 0.0722
            albedo = luma + (albedo - luma) * rng.uniform(0.80, 0.95)
        if rng.random() < 0.50:
            normal = cv2.GaussianBlur(normal, (0, 0), sigmaX=rng.uniform(0.15, 0.45))
            normal = _normalise_xy(normal)

    return (
        np.clip(albedo, 0.0, 1.0).astype(np.float32),
        normal.astype(np.float32),
        np.clip(material, 0.0, 1.0).astype(np.float32),
    )


def load_manifest(repo_root: Path, config: V14Config) -> dict[str, Any]:
    path = repo_root / config.dataset_manifest
    if not path.is_file():
        raise RuntimeError(f"V14 Raven dataset manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("crops"), list) or not payload["crops"]:
        raise RuntimeError(f"V14 dataset contains no crop records: {path}")
    return payload


class RavenSRDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        manifest: dict[str, Any],
        config: V14Config,
        split: str,
        length: int,
        *,
        seed: int,
        degradation: str,
    ) -> None:
        self.config = config
        self.split = split
        self.length = max(1, int(length))
        self.seed = int(seed)
        self.degradation = degradation
        self.records = [record for record in manifest["crops"] if record.get("split") == split]
        if not self.records:
            raise RuntimeError(f"V14 dataset has no {split} crops")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = random.Random(self.seed + int(index) * 1_000_003)
        record = self.records[int(index) % len(self.records)] if self.split != "train" else self.records[rng.randrange(len(self.records))]
        with np.load(record["path"], allow_pickle=False) as bundle:
            albedo = np.asarray(bundle["albedo"], dtype=np.uint8).astype(np.float32) / 255.0
            normal = np.asarray(bundle["normal"], dtype=np.uint8)[..., :2].astype(np.float32) / 127.5 - 1.0
            material = np.asarray(bundle["material"], dtype=np.uint8)[..., :3].astype(np.float32) / 255.0
        normal = _normalise_xy(normal)

        hr_size = self.config.train_hr_size if self.split == "train" else self.config.validation_hr_size
        lr_size = hr_size // self.config.scale
        if min(albedo.shape[:2]) < hr_size:
            raise RuntimeError(
                f"V14 refuses to invent HR supervision: crop {record.get('crop_id')} is {albedo.shape[1]}x{albedo.shape[0]} < {hr_size}"
            )
        max_x = albedo.shape[1] - hr_size
        max_y = albedo.shape[0] - hr_size
        if self.split == "train":
            x = rng.randrange(max_x + 1) if max_x else 0
            y = rng.randrange(max_y + 1) if max_y else 0
        else:
            x = max_x // 2
            y = max_y // 2
        albedo_hr = np.ascontiguousarray(albedo[y:y + hr_size, x:x + hr_size])
        normal_hr = np.ascontiguousarray(normal[y:y + hr_size, x:x + hr_size])
        material_hr = np.ascontiguousarray(material[y:y + hr_size, x:x + hr_size])
        if self.split == "train":
            albedo_hr, normal_hr, material_hr = _augment(albedo_hr, normal_hr, material_hr, rng)

        mode = self.degradation if self.split == "train" else "clean"
        lr_albedo, lr_normal, lr_material = _degrade(albedo_hr, normal_hr, material_hr, lr_size, mode, rng)

        def chw(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(value.transpose(2, 0, 1).copy())

        return {
            "lr_albedo": chw(lr_albedo),
            "lr_normal": chw(lr_normal),
            "lr_material": chw(lr_material),
            "target_albedo": chw(albedo_hr),
            "target_normal": chw(normal_hr),
            "target_material": chw(material_hr),
            "record_index": torch.tensor(int(index), dtype=torch.int64),
        }


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read V14 authored source: {path}")
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] >= 3:
        image = image[..., :3][:, :, ::-1]
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return image.astype(np.uint8)


def _read_rgba(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read V14 authored source: {path}")
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        rgb = np.repeat(image, 3, axis=-1)
        alpha = np.full_like(image, 255)
        return np.concatenate((rgb, alpha), axis=-1).astype(np.uint8)
    if image.shape[-1] == 3:
        rgb = image[:, :, ::-1]
        alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate((rgb, alpha), axis=-1)
    return image[..., [2, 1, 0, 3]].astype(np.uint8)


def _semantic_plane(path: Path | None, channel: int, width: int, height: int, default: int) -> np.ndarray:
    if path is None or not path.is_file():
        return np.full((height, width), default, dtype=np.uint8)
    rgba = _read_rgba(path)
    plane = rgba[..., max(0, min(3, int(channel)))]
    if plane.shape != (height, width):
        plane = cv2.resize(plane, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.asarray(plane, dtype=np.uint8)


def _canonical_native_family(family: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    albedo_path = Path(str(family.get("albedo") or ""))
    normal_path = Path(str(family.get("normal") or ""))
    if not albedo_path.is_file() or not normal_path.is_file():
        return None

    albedo = _read_rgb(albedo_path)
    try:
        import authored_texture_dataset as authored_dataset
        normal_rgb, _normal_encoding = authored_dataset.load_normal_training_rgb(normal_path)
    except (ImportError, AttributeError):
        normal_rgb = _read_rgb(normal_path)
    h, w = albedo.shape[:2]
    if normal_rgb.shape[:2] != (h, w):
        normal_rgb = cv2.resize(normal_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    channels = dict(family.get("channels") or {})
    material_path = Path(str(family.get("material") or "")) if family.get("material") else None
    roughness_path = Path(str(family.get("roughnessMap") or "")) if family.get("roughnessMap") else None
    glow_path = Path(str(family.get("glow") or "")) if family.get("glow") else None
    material_source = material_path if material_path and material_path.is_file() else None
    roughness_source = roughness_path if roughness_path and roughness_path.is_file() else material_source
    glow_source = glow_path if glow_path and glow_path.is_file() else material_source
    material = np.stack((
        _semantic_plane(material_source, int(channels.get("material", 0)), w, h, 0),
        _semantic_plane(glow_source, int(channels.get("glow", 1)), w, h, 0),
        _semantic_plane(roughness_source, int(channels.get("roughness", 2)), w, h, 128),
    ), axis=-1)

    normal = normal_rgb[..., :2].astype(np.float32) / 127.5 - 1.0
    normal = _normalise_xy(normal)
    return (
        albedo.astype(np.float32) / 255.0,
        normal,
        material.astype(np.float32) / 255.0,
    )


def native_family_samples(manifest: dict[str, Any], config: V14Config) -> list[dict[str, torch.Tensor | str]]:
    result: list[dict[str, torch.Tensor | str]] = []
    for family in list(manifest.get("families") or [])[: config.native_validation_max_families]:
        canonical = _canonical_native_family(family)
        if canonical is None:
            continue
        albedo_full, normal_full, material_full = canonical
        h, w = albedo_full.shape[:2]
        side = min(h, w)
        side -= side % config.scale
        if side < config.scale * 64:
            continue
        y = (h - side) // 2
        x = (w - side) // 2
        albedo = np.ascontiguousarray(albedo_full[y:y+side, x:x+side])
        normal = np.ascontiguousarray(normal_full[y:y+side, x:x+side])
        material = np.ascontiguousarray(material_full[y:y+side, x:x+side])
        lr_size = side // config.scale
        lr_albedo, lr_normal, lr_material = _degrade(albedo, normal, material, lr_size, "clean", random.Random(config.seed))

        def batch(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(value.transpose(2, 0, 1).copy()).unsqueeze(0)

        result.append({
            "name": str(family.get("familyId") or "native-family"),
            "lr_albedo": batch(lr_albedo),
            "lr_normal": batch(lr_normal),
            "lr_material": batch(lr_material),
            "target_albedo": batch(albedo),
            "target_normal": batch(normal),
            "target_material": batch(material),
        })
    return result
