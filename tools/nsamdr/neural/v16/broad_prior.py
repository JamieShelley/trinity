"""Authority-balanced sampling for NSAMDR broad authored-prior proofs.

The broad corpus contains multiple crops per authored texture authority. Random
crop sampling can leave many authorities unseen during short diagnostics, so this
module cycles through complete authorities before repeating one. The crop chosen
for an authority rotates on later cycles.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any
import random

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    # Package import path used by unittest and normal module imports.
    from ..v14.config import V14Config
    from ..v14.dataset import _augment, _degrade, _normalise_xy
except ImportError:
    # Script-mode compatibility when tools/nsamdr/neural is placed on sys.path.
    from v14.config import V14Config
    from v14.dataset import _augment, _degrade, _normalise_xy


def _family_id(record: dict[str, Any]) -> str:
    value = str(record.get("family_id") or record.get("familyId") or "").strip()
    if not value:
        raise ValueError("broad-prior crop record has no family/authority id")
    return value


D4_AUGMENTATION_VARIANTS = 8
AUGMENTATION_POLICIES = ("legacy-random", "d4-cyclic")


def _augment_d4(
    albedo: np.ndarray,
    normal: np.ndarray,
    material: np.ndarray,
    variant: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one deterministic D4 transform with normal-vector correction."""

    value = int(variant)
    if value < 0 or value >= D4_AUGMENTATION_VARIANTS:
        raise ValueError(
            f"D4 augmentation variant must be 0..{D4_AUGMENTATION_VARIANTS - 1}"
        )
    turns = value % 4
    mirror_x = value >= 4

    if turns:
        albedo = np.rot90(albedo, turns).copy()
        normal = np.rot90(normal, turns).copy()
        material = np.rot90(material, turns).copy()
        for _ in range(turns):
            x = normal[..., 0].copy()
            y = normal[..., 1].copy()
            normal[..., 0] = -y
            normal[..., 1] = x
    else:
        albedo = np.ascontiguousarray(albedo)
        normal = np.ascontiguousarray(normal)
        material = np.ascontiguousarray(material)

    if mirror_x:
        albedo = albedo[:, ::-1].copy()
        normal = normal[:, ::-1].copy()
        material = material[:, ::-1].copy()
        normal[..., 0] *= -1.0

    return albedo, _normalise_xy(normal), material


def authority_balanced_record_indices(
    records: list[dict[str, Any]],
    length: int,
    *,
    seed: int,
) -> list[int]:
    """Return a deterministic schedule that covers every authority per cycle."""

    if not records:
        raise ValueError("records must not be empty")
    requested = max(1, int(length))
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_family[_family_id(record)].append(index)
    for indices in by_family.values():
        indices.sort(
            key=lambda index: (
                str(records[index].get("crop_id") or records[index].get("cropId") or ""),
                str(records[index].get("path") or ""),
                index,
            )
        )

    families = sorted(by_family)
    schedule: list[int] = []
    cycle = 0
    while len(schedule) < requested:
        order = list(families)
        random.Random(int(seed) + cycle * 1_000_003).shuffle(order)
        for family in order:
            choices = by_family[family]
            schedule.append(choices[cycle % len(choices)])
            if len(schedule) >= requested:
                break
        cycle += 1
    return schedule


class AuthorityBalancedSRDataset(Dataset[dict[str, torch.Tensor]]):
    """V16 SR dataset whose sampling unit is the complete authored authority."""

    def __init__(
        self,
        manifest: dict[str, Any],
        config: V14Config,
        split: str,
        length: int,
        *,
        seed: int,
        degradation: str,
        augmentation_policy: str = "legacy-random",
    ) -> None:
        self.config = config
        self.split = str(split)
        self.seed = int(seed)
        self.degradation = str(degradation)
        self.augmentation_policy = str(augmentation_policy)
        if self.augmentation_policy not in AUGMENTATION_POLICIES:
            raise ValueError(
                f"unknown broad-prior augmentation policy: {self.augmentation_policy}"
            )
        self.records = [
            record for record in manifest.get("crops", []) if record.get("split") == self.split
        ]
        if not self.records:
            raise RuntimeError(f"V16 broad-prior dataset has no {self.split} crops")

        self.family_crop_counts: dict[str, int] = defaultdict(int)
        for record in self.records:
            self.family_crop_counts[_family_id(record)] += 1

        requested = max(1, int(length))
        if self.split != "train":
            requested = min(requested, len(self.records))
        self.record_indices = authority_balanced_record_indices(
            self.records,
            requested,
            seed=self.seed,
        )
        self.authority_count = len({_family_id(record) for record in self.records})

    def __len__(self) -> int:
        return len(self.record_indices)

    def selected_authority_count(self) -> int:
        return len({_family_id(self.records[index]) for index in self.record_indices})

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        global_index = int(index)
        record = self.records[self.record_indices[global_index]]
        rng = random.Random(self.seed + global_index * 1_000_003)

        with np.load(record["path"], allow_pickle=False) as bundle:
            albedo = np.asarray(bundle["albedo"], dtype=np.uint8).astype(np.float32) / 255.0
            normal = (
                np.asarray(bundle["normal"], dtype=np.uint8)[..., :2].astype(np.float32)
                / 127.5
                - 1.0
            )
            material = (
                np.asarray(bundle["material"], dtype=np.uint8)[..., :3].astype(np.float32)
                / 255.0
            )
        normal = _normalise_xy(normal)

        hr_size = (
            self.config.train_hr_size
            if self.split == "train"
            else self.config.validation_hr_size
        )
        lr_size = hr_size // self.config.scale
        if min(albedo.shape[:2]) < hr_size:
            raise RuntimeError(
                f"V16 refuses to invent HR supervision: crop {record.get('crop_id')} "
                f"is {albedo.shape[1]}x{albedo.shape[0]} < {hr_size}"
            )

        max_x = albedo.shape[1] - hr_size
        max_y = albedo.shape[0] - hr_size
        if self.split == "train":
            x = rng.randrange(max_x + 1) if max_x else 0
            y = rng.randrange(max_y + 1) if max_y else 0
        else:
            x = max_x // 2
            y = max_y // 2

        albedo_hr = np.ascontiguousarray(albedo[y : y + hr_size, x : x + hr_size])
        normal_hr = np.ascontiguousarray(normal[y : y + hr_size, x : x + hr_size])
        material_hr = np.ascontiguousarray(material[y : y + hr_size, x : x + hr_size])
        if self.split == "train":
            if self.augmentation_policy == "d4-cyclic":
                family = _family_id(record)
                family_crop_count = max(1, int(self.family_crop_counts[family]))
                authority_cycle = global_index // max(1, int(self.authority_count))
                variant = (
                    authority_cycle // family_crop_count
                ) % D4_AUGMENTATION_VARIANTS
                albedo_hr, normal_hr, material_hr = _augment_d4(
                    albedo_hr,
                    normal_hr,
                    material_hr,
                    variant,
                )
            else:
                albedo_hr, normal_hr, material_hr = _augment(
                    albedo_hr,
                    normal_hr,
                    material_hr,
                    rng,
                )

        mode = self.degradation if self.split == "train" else "clean"
        lr_albedo, lr_normal, lr_material = _degrade(
            albedo_hr,
            normal_hr,
            material_hr,
            lr_size,
            mode,
            rng,
        )

        def chw(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(value.transpose(2, 0, 1).copy())

        return {
            "lr_albedo": chw(lr_albedo),
            "lr_normal": chw(lr_normal),
            "lr_material": chw(lr_material),
            "target_albedo": chw(albedo_hr),
            "target_normal": chw(normal_hr),
            "target_material": chw(material_hr),
            "record_index": torch.tensor(self.record_indices[global_index], dtype=torch.int64),
        }
