from __future__ import annotations

import unittest

from tools.nsamdr.neural.probe_nsamdr_v16_structure_conditioning import (
    NO_BENEFIT,
    _continue_after_checkpoint,
    _parse_stages,
)


class StructureProbeLadderTests(unittest.TestCase):
    def test_default_ladder_parsing_is_cumulative_and_sorted(self) -> None:
        self.assertEqual(_parse_stages("2048,512,1024,1024", 0), [512, 1024, 2048])

    def test_explicit_steps_is_single_checkpoint_override(self) -> None:
        self.assertEqual(_parse_stages("512,1024,2048", 256), [256])

    def test_first_checkpoint_always_allows_second(self) -> None:
        keep_going, reason = _continue_after_checkpoint(0, 3, NO_BENEFIT)
        self.assertTrue(keep_going)
        self.assertEqual(reason, "minimum-two-checkpoints")

    def test_second_no_benefit_stops_before_long_stage(self) -> None:
        keep_going, reason = _continue_after_checkpoint(1, 3, NO_BENEFIT)
        self.assertFalse(keep_going)
        self.assertEqual(reason, "no-heldout-benefit-after-second-checkpoint")

    def test_marginal_signal_allows_long_stage(self) -> None:
        keep_going, _reason = _continue_after_checkpoint(
            1,
            3,
            "structure-conditioning-signal-marginal",
        )
        self.assertTrue(keep_going)


if __name__ == "__main__":
    unittest.main()
