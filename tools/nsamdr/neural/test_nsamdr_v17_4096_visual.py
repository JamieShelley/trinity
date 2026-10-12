"""Inference-only native 4096 diagnostic contracts and tiled-stitching regression."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from tools.nsamdr.nsamdr_cli import _app
from tools.nsamdr.neural.probe_nsamdr_v17_4096_visual import (
    HR_SIZE, LR_SIZE, _choose_reviewed_source, _metrics, _region_box,
    _reviewed_source, _tile_ranges, _tiled_4x,
)
from tools.nsamdr.neural.probe_nsamdr_v17_generalization_v2 import (
    PREVIOUS_VALIDATION_AUTHORITIES,
)


class Toy4x(torch.nn.Module):
    """Small spatial-convolution model to detect LR tile-edge disagreements."""
    def forward(self, lr_albedo: torch.Tensor, lr_normal: torch.Tensor,
                lr_material: torch.Tensor) -> dict[str, torch.Tensor]:
        baseline = F.interpolate(
            lr_albedo, scale_factor=4, mode="bicubic", align_corners=False,
        ).clamp(0, 1)
        kernel = torch.ones((3, 1, 3, 3), device=lr_albedo.device)/9.0
        context = F.conv2d(F.pad(lr_albedo, (1, 1, 1, 1), mode="replicate"),
                           kernel, groups=3)
        candidate = F.interpolate(
            context, scale_factor=4, mode="bicubic", align_corners=False,
        ).clamp(0, 1)
        return {"baseline_albedo": baseline, "candidate_albedo": candidate}


class V17Visual4096Tests(unittest.TestCase):
    def test_tile_ranges_complete_without_overlap_or_holes(self) -> None:
        self.assertEqual(_tile_ranges(1024, 128)[0], (0, 128))
        self.assertEqual(_tile_ranges(1024, 128)[-1], (896, 1024))
        self.assertEqual(_tile_ranges(13, 8), [(0, 8), (8, 13)])
        with self.assertRaises(ValueError):
            _tile_ranges(0, 8)

    def test_halo_tile_output_matches_monolithic_spatial_model(self) -> None:
        rng = np.random.default_rng(2701)
        albedo = rng.random((19, 23, 3), dtype=np.float32)
        normal = np.zeros((19, 23, 2), dtype=np.float32)
        material = np.full_like(albedo, 0.5)
        model = Toy4x().eval()
        stitched_b, stitched_c = _tiled_4x(
            model, albedo, normal, material, device=torch.device("cpu"),
            precision="auto", tile_lr=8, halo_lr=4,
        )
        rgb = torch.from_numpy(albedo.transpose(2, 0, 1).copy())[None]
        with torch.inference_mode():
            full = model(rgb, torch.zeros(1, 2, 19, 23), torch.ones_like(rgb)*0.5)
        for key, result in (("baseline_albedo", stitched_b), ("candidate_albedo", stitched_c)):
            expected = (
                full[key][0].mul(255).round().byte().permute(1, 2, 0).numpy()
            )
            np.testing.assert_array_equal(result, expected)

    def test_reviewed_development_only_and_never_final_test(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.png").touch()
            (root / "n.png").touch()
            anchor = PREVIOUS_VALIDATION_AUTHORITIES[0]
            original = {
                "familyId": anchor,
                "albedo": {"path": str(root / "a.png"), "width": 4096, "height": 4096},
                "normal": {"path": str(root / "n.png"), "width": 2048, "height": 2048},
            }
            manifest = {
                "families": [original],
                "crops": [{"crop_id": f"{anchor}_001",
                           "source_box": [0, 0, 125000, 125000]}],
            }
            report = {
                "validationCropIds": [f"{x}_001" for x in PREVIOUS_VALIDATION_AUTHORITIES],
                "finalTestCropIds": ["sealed_001"],
            }
            chosen, family, crop = _choose_reviewed_source(manifest, report, "auto")
            self.assertEqual(chosen, anchor)
            self.assertEqual(family, original)
            self.assertEqual(crop["source_box"], [0, 0, 125000, 125000])
            with self.assertRaisesRegex(RuntimeError, "not an already-reviewed"):
                _reviewed_source(manifest, report, "sealed")
            # Even if accidentally present in both sets, explicit sealed exclusion wins.
            report["finalTestCropIds"].append(f"{anchor}_001")
            with self.assertRaisesRegex(RuntimeError, "sealed"):
                _reviewed_source(manifest, report, anchor)

    def test_native_1024_input_fallback_without_false_4096_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.png").touch()
            (root / "n.png").touch()
            authority = PREVIOUS_VALIDATION_AUTHORITIES[0]
            data = {
                "families": [{
                    "familyId": authority,
                    "albedo": {"path": str(root / "a.png"),
                               "width": 2048, "height": 2048},
                    "normal": {"path": str(root / "n.png")},
                }],
                "crops": [{"crop_id": f"{authority}_001",
                           "source_box": [0, 0, 250000, 250000]}],
            }
            report = {"validationCropIds": [f"{authority}_001"],
                      "finalTestCropIds": []}
            family, crop = _reviewed_source(data, report, authority)
            self.assertEqual(family["albedo"]["width"], 2048)
            selected, _, _ = _choose_reviewed_source(data, report, "auto")
            self.assertEqual(selected, authority)
            self.assertEqual(
                _region_box(crop, (2048, 2048), size=1024),
                (0, 0, 1024, 1024),
            )
            # A real 1024 native input is valid; it is NOT native 4096 truth.
            data["families"][0]["albedo"]["width"] = 512
            data["families"][0]["albedo"]["height"] = 512
            with self.assertRaisesRegex(RuntimeError, "smaller than 1024"):
                _reviewed_source(data, report, authority)
            with self.assertRaisesRegex(RuntimeError, "No reviewed authority"):
                _choose_reviewed_source(data, report, "auto")

    def test_native_4096_source_is_preferred_to_native_1024_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("a.png", "n.png"):
                (root / name).touch()
            first, second = PREVIOUS_VALIDATION_AUTHORITIES[:2]
            manifest = {
                "families": [
                    {
                        "familyId": first,
                        "albedo": {"path": str(root / "a.png"), "width": 2048, "height": 2048},
                        "normal": {"path": str(root / "n.png")},
                    },
                    {
                        "familyId": second,
                        "albedo": {"path": str(root / "a.png"), "width": 4096, "height": 4096},
                        "normal": {"path": str(root / "n.png")},
                    },
                ],
                "crops": [
                    {"crop_id": f"{first}_001", "source_box": [0, 0, 500000, 500000]},
                    {"crop_id": f"{second}_001", "source_box": [0, 0, 250000, 250000]},
                ],
            }
            report = {
                "validationCropIds": [f"{first}_001", f"{second}_001"],
                "finalTestCropIds": [],
            }
            selected, _, _ = _choose_reviewed_source(manifest, report, "auto")
            self.assertEqual(selected, second)

    def test_native_region_is_not_upscaled(self) -> None:
        self.assertEqual(
            _region_box({"source_box": [500000, 500000, 562500, 562500]},
                        (8192, 8192)),
            (2304, 2304, 6400, 6400),
        )
        with self.assertRaisesRegex(RuntimeError, "upscale"):
            _region_box({"source_box": [0, 0, 500000, 500000]}, (2048, 2048))

    def test_metrics_identical_candidate_has_same_error(self) -> None:
        ref = np.zeros((8, 8, 3), dtype=np.uint8)
        baseline = ref.copy()
        candidate = ref.copy()
        result = _metrics(ref, baseline, candidate)
        self.assertEqual(result["candidateMAE"], result["deterministicMAE"])
        self.assertEqual(result["candidateChangedFraction1LSB"], 0.0)

    def test_cli_diagnostic_defaults_are_1024_to_4096(self) -> None:
        self.assertEqual(LR_SIZE, 1024)
        self.assertEqual(HR_SIZE, 4096)
        parsed = _app.build_parser().parse_args(["v17-4096-visual-proof"])
        self.assertEqual(parsed.tile_lr, 128)
        self.assertEqual(parsed.halo_lr, 64)
        self.assertEqual(parsed.authority_id, "auto")


if __name__ == "__main__":
    unittest.main()
