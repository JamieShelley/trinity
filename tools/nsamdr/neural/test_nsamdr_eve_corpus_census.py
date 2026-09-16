from __future__ import annotations

import unittest

from tools.nsamdr.neural.scan_eve_authored_corpus import _authority_id, _resolution_tier


class EveCorpusCensusTests(unittest.TestCase):
    def test_authority_identity_depends_only_on_albedo_and_normal(self) -> None:
        a = _authority_id("res:/ship/a_d.dds", "res:/ship/a_n.dds")
        b = _authority_id("RES:/SHIP/A_D.DDS", "res:/ship/a_n.dds")
        c = _authority_id("res:/ship/b_d.dds", "res:/ship/a_n.dds")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_resolution_tier_uses_minimum_aligned_dimension(self) -> None:
        albedo = {"width": 4096, "height": 2048}
        normal = {"width": 4096, "height": 2048}
        self.assertEqual(_resolution_tier(albedo, normal), 2048)

        square = {"width": 4096, "height": 4096}
        self.assertEqual(_resolution_tier(square, square), 4096)


if __name__ == "__main__":
    unittest.main()
