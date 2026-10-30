"""Focused V178 composition invariants and experimental training provenance."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tools.nsamdr.neural.train_nsamdr_v178_continuity import (
    MODEL_SCHEMA, REPORT_SCHEMA, parser, source_hash,
)
from tools.nsamdr.neural.v178.composed_geometry import (
    KINDS, make_case, mixed_batch,
)


def pixel_coverage(target: torch.Tensor, x: float, y: float) -> float:
    return float(target[0, min(target.shape[-2] - 1, int(y * 4)),
                        min(target.shape[-1] - 1, int(x * 4))])


class ComposedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(2, torch.get_num_threads()))

    def test_batch_determinism_is_independent_of_global_rng(self):
        for kind in KINDS:
            first = make_case(32, 45681, kind=kind)
            torch.manual_seed(918310)
            second = make_case(32, 45681, kind=kind)
            for a, b in zip(first[:4], second[:4]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(first[4], second[4])

    def test_parallel_grooves_both_exist_and_are_separate(self):
        for seed in range(1310, 1326):
            _, _, _, target, details = make_case(32, seed, kind="parallel")
            (p, q), (r, s) = details["segments"]
            for t in (.2, .5, .8):
                x1, y1 = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
                x2, y2 = r[0] + (s[0] - r[0]) * t, r[1] + (s[1] - r[1]) * t
                self.assertGreater(pixel_coverage(target, x1, y1), .3)
                self.assertGreater(pixel_coverage(target, x2, y2), .3)
                self.assertLess(pixel_coverage(target, (x1 + x2) / 2,
                                               (y1 + y2) / 2), .1)
            self.assertGreater(details["separation"], 2.)

    def test_t_and_x_junctions_have_both_arms_and_center(self):
        for kind in ("T_junction", "X_junction"):
            for seed in range(340, 350):
                _, _, _, target, details = make_case(32, seed, kind=kind)
                cx, cy = details["junction"]
                self.assertGreater(pixel_coverage(target, cx, cy), .3)
                self.assertEqual(len(details["segments"]), 2)
                for p, q in details["segments"]:
                    for t in (.2, .5, .8):
                        x = p[0] + (q[0] - p[0]) * t
                        y = p[1] + (q[1] - p[1]) * t
                        self.assertGreater(pixel_coverage(target, x, y), .3)

    def test_paired_sampling_retains_clean_geometry_and_auxiliary(self):
        for kind in KINDS:
            area = make_case(32, 16392, kind=kind, sampling="area")
            point = make_case(32, 16392, kind=kind, sampling="point", phase=(2, 1))
            torch.testing.assert_close(area[3], point[3], rtol=0, atol=0)
            torch.testing.assert_close(area[2][1:], point[2][1:], rtol=0, atol=0)
            self.assertEqual(area[4]["segments"], point[4]["segments"])
            self.assertEqual(area[4]["junction"], point[4]["junction"])
        self.assertGreater(float((area[0] - point[0]).abs().max()), 0)

    def test_protected_appearance_never_becomes_contour_target(self):
        a, n, obs, target, _ = make_case(32, 2139, kind="protected_texture")
        self.assertEqual(int(torch.count_nonzero(target)), 0)
        self.assertGreater(float(a.std()), .03)
        self.assertTrue(bool(torch.isfinite(obs).all()))
        self.assertEqual(tuple(n.shape), (2, 32, 32))

    def test_composed_auxiliary_not_target_encoded(self):
        # Same seed, independently chosen geometry: nuisance observations must
        # remain bitwise equal although analytic targets differ strongly.
        par = make_case(32, 6131, kind="parallel")
        jun = make_case(32, 6131, kind="T_junction")
        torch.testing.assert_close(par[2][1:], jun[2][1:], rtol=0, atol=0)
        self.assertGreater(float((par[3] - jun[3]).abs().sum()), 20.)

    def test_v5_mixed_observations_and_targets_are_unmodified_by_inversion(self):
        plain = mixed_batch(4, 32, 17852, torch.device("cpu"), augment_rgb=False)
        inverted = mixed_batch(4, 32, 17852, torch.device("cpu"), augment_rgb=True)
        for i in (1, 2, 3):
            torch.testing.assert_close(plain[i], inverted[i], rtol=0, atol=0)
        self.assertEqual(tuple(plain[3].shape), (4, 1, 128, 128))

    def test_registered_schema_does_not_replace_v5_default(self):
        from tools.nsamdr.neural.train_nsamdr_v178_contour import parser as v5_parser
        from tools.nsamdr.neural.v178.contour_model import SCHEMA as v5_schema
        self.assertIn("V5", v5_schema)
        self.assertIn("V6", MODEL_SCHEMA)
        self.assertIn("V4", REPORT_SCHEMA)
        self.assertEqual(v5_parser().parse_args([]).seed, 17842)
        self.assertEqual(parser().parse_args([]).seed, 17852)

    def test_source_identity_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from tools.nsamdr.neural.train_nsamdr_v178_continuity import FILES
            stage = Path(__file__).resolve().parents[3]
            for name in FILES:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((stage / name).read_bytes())
            initial = source_hash(root)
            file = root / FILES[-1]
            file.write_bytes(file.read_bytes() + b"\n# altered\n")
            self.assertNotEqual(initial, source_hash(root))


if __name__ == "__main__":
    unittest.main()
