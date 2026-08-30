"""B1b procedural proof-domain support without proof-case leakage.

The failed long Quick run showed classic domain overfit:
    training classifier CE ~= 0.058
    held-out 29-case accuracy = 72.4%

The old B1b bank repeatedly generated the same 448 random physical-map
degradations, while the permanent proof ladder is built by analytic HR drawing
followed by direct 4x area/linear downsampling. More epochs only memorised that
fixed bank.

This contract keeps the held-out 29 named cases untouched and generates a
separate class-balanced PROCEDURAL bank with random parameters. Its LR physical
maps are formed by the same *kind* of 4x measurement process used by the proof
ladder: antialiased analytic HR geometry -> PBR companions -> area/linear
downsample, with randomized contrast and generic blur/halo stress.

No held-out case name, permanent proof parameter tuple, or held-out case object is
used for training.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable

import cv2
import numpy as np
import torch

from . import dataset as _dataset
from . import parametric_primitives as _primitives
from .geometry_proof_ladder import pbr_companions

_ORIGINAL_RANDOM_TARGET: Callable[..., Any] | None = None
_ORIGINAL_PARAMETRIC_GETITEM: Callable[..., Any] | None = None


class ClassifierGeneralisationContract:
    # Purpose: Implement supported target for ClassifierGeneralisationContract.
    # Called by: _proof_domain_parametric_getitem
    # Calls: No same-class helper methods.
    def _supported_target(
        self,
        size: int,
        rng: random.Random,
        *,
        forced_class: int | None = None,
    ):
        assert _ORIGINAL_RANDOM_TARGET is not None
        target = _ORIGINAL_RANDOM_TARGET(size, rng, forced_class=forced_class)
        if forced_class is None:
            return target

        cls = int(target.class_index)
        params = np.asarray(target.params, dtype=np.float32).copy()
        mask = np.asarray(target.mask, dtype=np.float32).copy()

        if rng.random() < 0.55:
            params[0] = np.float32(rng.uniform(0.44, 0.56))
            params[1] = np.float32(rng.uniform(0.44, 0.56))

        if cls in (0, 1, 2, 3, 4, 6) and rng.random() < 0.55:
            angle_deg = rng.choice(tuple(range(0, 180, 15))) + rng.uniform(-2.5, 2.5)
            angle = math.radians(float(angle_deg))
            params[2] = np.float32(math.cos(angle))
            params[3] = np.float32(math.sin(angle))

        if cls == 2:
            params[6] = np.float32(
                rng.uniform(0.10, 0.86)
            )
        elif cls == 5:
            inner = rng.uniform(0.10, 0.58)
            gap = (
                rng.uniform(0.035, 0.13)
                if rng.random() < 0.65
                else rng.uniform(0.13, 0.30)
            )
            outer = min(0.92, inner + gap)
            if outer - inner < 0.035:
                inner = max(0.08, outer - 0.035)
            params[4] = np.float32(inner)
            params[5] = np.float32(outer)

        return _primitives.PrimitiveTarget(cls, params, mask)

    # Purpose: Implement proof domain parametric getitem for ClassifierGeneralisationContract.
    # Called by: External callers and the owning workflow.
    # Calls: _supported_target
    def _proof_domain_parametric_getitem(
        self,
        instance: Any,
        index: int,
    ) -> dict[str, torch.Tensor]:
        rng = random.Random(int(instance.seed) + int(index) * 104729)
        config = instance.config
        target_size = int(config.tile_size) * int(config.target_scale)
        lr_size = int(config.tile_size)
        target = self._supported_target(
            target_size, rng, forced_class=int(index) % int(_primitives.PRIMITIVE_COUNT)
        )
        signed_distance = _primitives.render_parametric_sdf_numpy(
            target, target_size
        )
        coverage = np.clip(
            0.5 - signed_distance, 0.0, 1.0
        )[..., None].astype(np.float32)

        # Randomized proof-domain photometry. Shape/family is never correlated with
        # a fixed colour tuple.
        background = rng.uniform(0.10, 0.28)
        if rng.random() < 0.18:
            contrast = rng.uniform(0.07, 0.18)
        else:
            contrast = rng.uniform(0.40, 0.78)
        foreground = min(0.96, background + contrast)
        tint = np.asarray(
            [rng.uniform(0.94, 1.06) for _ in range(3)], dtype=np.float32
        )
        rgb_scalar = background * (1.0 - coverage) + foreground * coverage
        target_rgb = np.clip(rgb_scalar * tint.reshape(1, 1, 3), 0.0, 1.0)
        target_rgb = np.ascontiguousarray(target_rgb.astype(np.float32))

        target_normal, target_material = pbr_companions(target_rgb)

        low_size = (lr_size, lr_size)
        low_rgb = cv2.resize(
            target_rgb, low_size, interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        low_normal = cv2.resize(
            target_normal, low_size, interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        low_material = cv2.resize(
            target_material, low_size, interpolation=cv2.INTER_AREA
        ).astype(np.float32)

        stress = rng.random()
        if stress < 0.10:
            low_rgb = cv2.GaussianBlur(low_rgb, (5, 5), 1.0)
            low_material = cv2.GaussianBlur(low_material, (5, 5), 0.75)
        elif stress < 0.20:
            blur = cv2.GaussianBlur(low_rgb, (0, 0), 0.95)
            low_rgb = np.clip(
                low_rgb + (low_rgb - blur) * rng.uniform(0.25, 0.45),
                0.0,
                1.0,
            )
            low_rgb = cv2.GaussianBlur(low_rgb, (3, 3), 0.45)

        model_input = _dataset.build_model_input(
            np.ascontiguousarray(low_rgb),
            np.ascontiguousarray(low_normal),
            np.ascontiguousarray(low_material),
            degradation_level=1.0,
            contour_sdf_max_distance_pixels=float(
                config.contour_sdf_max_distance_pixels
            ),
            target_scale=int(config.target_scale),
        )

        sdf, _orientation, _edge = _dataset.analytic_contour_targets(
            signed_distance
        )
        return {
            "input": torch.from_numpy(
                np.ascontiguousarray(model_input.astype(np.float32))
            ),
            "target_sdf": torch.from_numpy(
                np.ascontiguousarray(sdf.transpose(2, 0, 1))
            ),
            "primitive_valid": torch.tensor([1.0], dtype=torch.float32),
            "primitive_class": torch.tensor(
                int(target.class_index), dtype=torch.int64
            ),
            "primitive_params": torch.from_numpy(
                np.asarray(target.params, dtype=np.float32).copy()
            ),
            "primitive_param_mask": torch.from_numpy(
                np.asarray(target.mask, dtype=np.float32).copy()
            ),
        }

    # Purpose: Implement install classifier generalisation contract for ClassifierGeneralisationContract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def install_classifier_generalisation_contract(self) -> None:
        global _ORIGINAL_RANDOM_TARGET, _ORIGINAL_PARAMETRIC_GETITEM

        current_target = _dataset.random_primitive_target
        if not bool(getattr(current_target, "_nsamdr_proof_domain_support_v4", False)):
            _ORIGINAL_RANDOM_TARGET = current_target
            _supported_target._nsamdr_proof_domain_support_v4 = True  # type: ignore[attr-defined]
            _dataset.random_primitive_target = _supported_target
            _primitives.random_primitive_target = _supported_target

        current_getitem = _dataset.ParametricPrimitiveTrainingDataset.__getitem__
        if not bool(getattr(
            current_getitem, "_nsamdr_proof_domain_b1b_bank_v4", False
        )):
            _ORIGINAL_PARAMETRIC_GETITEM = current_getitem
            _proof_domain_parametric_getitem._nsamdr_proof_domain_b1b_bank_v4 = True  # type: ignore[attr-defined]
            _dataset.ParametricPrimitiveTrainingDataset.__getitem__ = (
                _proof_domain_parametric_getitem
            )

_classifier_generalisation_contract = ClassifierGeneralisationContract()
_supported_target = _classifier_generalisation_contract._supported_target
_proof_domain_parametric_getitem = _classifier_generalisation_contract._proof_domain_parametric_getitem
install_classifier_generalisation_contract = _classifier_generalisation_contract.install_classifier_generalisation_contract
