"""V17.7 conservative physical semantic baseline and frozen C0/C1 Raven controls."""
from __future__ import annotations
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from tools.nsamdr.neural.v177.semantic_baseline import SemanticBaseline4x
from tools.nsamdr.neural.v176.raven_renderer import _tiled_maps


class V177SemanticBaselineTests(unittest.TestCase):
    def setUp(self):
        self.model = SemanticBaseline4x()

    def test_categorical_v5_preserves_source_slot_identities(self):
        a = torch.full((1,3,6,8), .5)
        n = torch.zeros((1,2,6,8))
        m = torch.zeros((1,3,6,8))
        m[:,:, :,4:] = 1.0
        m[:,0,:,:4] = 1.0 / 3.0
        m[:,0,:,4:] = 1.0
        _, _, result = self.model(a,n,m,material_policy="v5_selector")
        values = torch.unique(result[:,0]).tolist()
        self.assertEqual(len(values), 2)
        self.assertAlmostEqual(min(values), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(max(values), 1.0, places=6)
        self.assertEqual(tuple(result.shape), (1,3,24,32))

    def test_legacy_masks_blend_only_near_edges(self):
        a = torch.ones((1,3,8,8))
        n = torch.zeros((1,2,8,8))
        m = torch.zeros((1,3,8,8))
        m[:,0,:,4:] = 1.0
        _, _, out = self.model(a,n,m,material_policy="legacy_mask")
        self.assertTrue(torch.all(out[:,0,:,0:8] == 0))
        self.assertTrue(torch.all(out[:,0,:,-8:] == 1))
        self.assertTrue(((out[:,0,:,15:17] > 0) & (out[:,0,:,15:17] < 1)).any())
        self.assertTrue(torch.all(out >= 0) and torch.all(out <= 1))

    def test_unknown_selector_does_not_invent_classes(self):
        a = torch.zeros((1,3,8,8))
        n = torch.zeros((1,2,8,8))
        m = torch.zeros((1,3,8,8))
        m[:,0,:,4:] = .75
        _, _, out = self.model(a,n,m,material_policy="unknown")
        self.assertEqual(set(torch.unique(out[:,0]).tolist()), {0.0,.75})
        with self.assertRaisesRegex(ValueError,"policy"):
            self.model(a,n,m,material_policy="ambiguous")

    def test_normal_components_interpolate_jointly(self):
        a = torch.zeros((1,3,8,8))
        n = torch.ones((1,2,8,8))
        m = torch.zeros((1,3,8,8))
        _, out, _ = self.model(a,n,m,material_policy="legacy_mask")
        self.assertTrue(torch.isfinite(out).all())
        self.assertLessEqual(float(out.square().sum(1).max()), .999**2 + 1e-5)
        self.assertEqual(tuple(out.shape), (1,2,32,32))

    def test_native_packed_channels_preserve_exact_values(self):
        import cv2
        from tools.nsamdr.neural.v176.raven_renderer import (
            _ensure_canvas, _resized_channel,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "packed.png"
            source = np.array([
                [[0, 0, 0, 0], [255, 255, 255, 255]],
                [[255, 255, 255, 255], [0, 0, 0, 0]],
            ], dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), source))
            canvas = _ensure_canvas({}, path, 8, 8)
            self.assertEqual(set(np.unique(canvas[:, :, 0]).tolist()), {0, 255})
            categorical = _resized_channel(path, 0, 8, 8, 0.0, categorical=True)
            self.assertEqual(set(np.unique(categorical).tolist()), {0.0, 1.0})
            continuous = _resized_channel(path, 0, 8, 8, 0.0, categorical=False)
            self.assertTrue(((continuous > 0.0) & (continuous < 1.0)).any())

    def test_cross_semantic_shader_family_conflict_rejected(self):
        import cv2
        from tools.nsamdr.neural.v176.raven_renderer import _collect_contexts
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "albedo.png"
            self.assertTrue(cv2.imwrite(str(a), np.zeros((4,4,4),dtype=np.uint8)))
            rows = [
                {"albedo":"albedo.png", "shader_family":"legacy_pgs"},
                {"albedo":"albedo.png", "shader_family":"v5_packed"},
            ]
            with self.assertRaisesRegex(RuntimeError, "conflicting Raven material semantics"):
                _collect_contexts(rows, root)

    def test_c0_and_c1_export_as_different_independent_planes(self):
        class Dummy(torch.nn.Module):
            def forward(self,a,n,m):
                h,w=a.shape[-2:]
                def plane(ch,val):
                    return torch.full((1,ch,4*h,4*w),val)
                return {
                    "base_candidate_albedo": plane(3,.42),
                    "candidate_albedo":plane(3,.73),
                    "candidate_normal":plane(2,0),
                    "candidate_material":plane(3,.61),
                }
        config = SimpleNamespace(
            production_tile_lr=6, production_overlap_lr=2, scale=4,
        )
        with tempfile.TemporaryDirectory() as tmp:
            maps = _tiled_maps(
                Dummy(), config,
                torch.full((1,3,11,13), .2),
                torch.zeros((1,2,11,13)),
                torch.zeros((1,3,11,13)),
                device=torch.device("cpu"),
                scratch=Path(tmp), material_policy="v5_selector",
            )
            try:
                self.assertEqual(tuple(maps["candidate_albedo"].shape),(44,52,3))
                self.assertTrue(np.allclose(maps["backbone_albedo"],.42,atol=1e-5))
                self.assertTrue(np.allclose(maps["candidate_albedo"],.73,atol=1e-5))
                self.assertTrue(np.allclose(maps["baseline_albedo"],.2,atol=1e-5))
                self.assertFalse(np.allclose(maps["candidate_albedo"],maps["backbone_albedo"]))
            finally:
                for m in maps.values():
                    m._mmap.close()


if __name__ == "__main__":
    unittest.main()
