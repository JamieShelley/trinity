from __future__ import annotations

import unittest

from tools.nsamdr.neural.probe_nsamdr_v16_full_broad import (
    _full_config,
    _parse_stages,
)


class FullBroadProbeTests(unittest.TestCase):
    def test_stage_parser_sorts_and_deduplicates(self) -> None:
        self.assertEqual(_parse_stages("1024,256,512,512"), [256, 512, 1024])

    def test_full_config_keeps_production_v16_capacity(self) -> None:
        config = _full_config("manifest.json", 512)
        self.assertEqual(config.train_lr_size, 128)
        self.assertEqual(config.train_hr_size, 512)
        self.assertEqual(config.lr_context_channels, 32)
        self.assertEqual(config.lr_blocks, 5)
        self.assertEqual(config.hr_channels, 96)
        self.assertEqual(config.swin_groups, 6)
        self.assertEqual(config.swin_blocks_per_group, 6)
        self.assertEqual(config.swin_depth, 36)
        self.assertEqual(config.swin_num_heads, 6)
        self.assertEqual(config.map_tail_blocks, 2)
        self.assertTrue(config.use_gradient_checkpointing)


if __name__ == "__main__":
    unittest.main()
