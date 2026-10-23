"""V17.6 Raven loader contract: training source only, no target-assisted field."""
from __future__ import annotations
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from tools.nsamdr.neural import render_nsamdr_v176_raven_preview as raven


class V176RavenPreviewTests(unittest.TestCase):
    def test_research_only_schema_and_models(self):
        self.assertEqual(raven.SCHEMA, "NSAMDR_V176_RAVEN_RESEARCH_PREVIEW_V1")
        self.assertEqual(raven.STAGE, 6144)
        p = raven.parser()
        parsed = p.parse_args(["--authority-index", "1", "--mode", "hr_features58"])
        self.assertEqual((parsed.authority_index, parsed.mode), (1, "hr_features58"))

    def test_unavailable_comparison_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "No authenticated"):
                raven._resolve_comparison(Path(root), 10, "output10")

    def test_generic_renderer_requires_preverified_model(self):
        from tools.nsamdr.neural.v176 import raven_renderer
        with self.assertRaisesRegex(RuntimeError, "authenticated model"):
            with tempfile.TemporaryDirectory() as tmp:
                empty = Path(tmp) / "empty.pt"
                empty.write_bytes(b"test")
                source = raven_renderer.PreviewSource("V176", 0, "research", empty)
                raven_renderer._generate_candidate(
                    repo_root=Path(tmp), source=source,
                    device=raven.torch.device("cpu"), obj_path=Path(tmp) / "x.obj",
                    materials=Path(tmp) / "materials.tsv",
                    model_override=None,
                )

    def test_exact_native_source_geometry_never_downsamples(self):
        import numpy as np
        from tools.nsamdr.neural.v176.raven_renderer import _output_size
        self.assertEqual(_output_size(np.zeros((512, 1024, 4), np.uint8), 4),
                         (4096, 2048))
        self.assertEqual(_output_size(np.zeros((2048, 2048, 4), np.uint8), 4),
                         (8192, 8192))
        with self.assertRaisesRegex(RuntimeError, "16384"):
            _output_size(np.zeros((4097, 1, 4), np.uint8), 4)

    def test_no_preview_reduction_permitted(self):
        import numpy as np
        import torch
        from tools.nsamdr.neural.v176.raven_renderer import _model_inputs, TextureContext
        # This check is before source processing: mismatched output/input geometry
        # fails closed even with a loaded source file.
        with tempfile.TemporaryDirectory() as d:
            import cv2
            path = Path(d) / "source.png"
            original = np.zeros((8, 8, 4), dtype=np.uint8)
            original[:, :, :3] = 192
            self.assertTrue(cv2.imwrite(str(path), original))
            a, n, m = _model_inputs(path, TextureContext(),
                                    out_width=32, out_height=32,
                                    scale=4, device=torch.device("cpu"))
            self.assertEqual(tuple(a.shape), (1, 3, 8, 8))
            self.assertEqual(tuple(n.shape), (1, 2, 8, 8))
            self.assertEqual(tuple(m.shape), (1, 3, 8, 8))
            self.assertAlmostEqual(float(a.mean()), 192.0 / 255, places=6)
            with self.assertRaisesRegex(RuntimeError, "would resize original source"):
                _model_inputs(path, TextureContext(),
                              out_width=16, out_height=16,
                              scale=4, device=torch.device("cpu"))

    def test_native_tiling_disk_buffers_preserve_constants_without_gaps(self):
        import types
        import numpy as np
        import torch
        from tools.nsamdr.neural.v176.raven_renderer import _tiled_maps

        class Dummy(torch.nn.Module):
            def forward(self, a, n, m):
                h, w = a.shape[-2:]
                def tensor(channels, value):
                    return torch.full((1, channels, 4*h, 4*w), value,
                                      device=a.device, dtype=torch.float32)
                return {
                    "baseline_albedo": tensor(3, .25),
                    "baseline_normal": tensor(2, .0),
                    "baseline_material": tensor(3, .5),
                    "candidate_albedo": tensor(3, .75),
                    "candidate_normal": tensor(2, .0),
                    "candidate_material": tensor(3, .5),
                }
        config = types.SimpleNamespace(
            production_tile_lr=6, production_overlap_lr=2, scale=4)
        with tempfile.TemporaryDirectory() as root:
            maps = _tiled_maps(
                Dummy(), config,
                torch.ones((1, 3, 11, 13)),
                torch.zeros((1, 2, 11, 13)),
                torch.zeros((1, 3, 11, 13)),
                device=torch.device("cpu"), scratch=Path(root),
            )
            try:
                self.assertEqual(maps["candidate_albedo"].shape, (44, 52, 3))
                self.assertTrue(np.allclose(maps["candidate_albedo"], .75, atol=1e-5))
                self.assertTrue(np.allclose(maps["baseline_albedo"], .25, atol=1e-5))
                self.assertTrue(np.isfinite(maps["candidate_normal"]).all())
                self.assertTrue((Path(root)/"candidate_albedo.f32").is_file())
            finally:
                for plane in maps.values():
                    plane._mmap.close()


if __name__ == "__main__":
    unittest.main()
