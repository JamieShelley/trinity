"""Contract checks for V12.8 detail-anchored specialist safety."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))

CONTRACT = ROOT / "tools/nsamdr/neural/v9/parallel_specialist_safety_contract.py"
INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"


class TestParallelSpecialistSafetyContract:
    def test_install_order_is_after_v127_training_isolation(self) -> None:
        source = INIT.read_text(encoding="utf-8")
        isolation = source.index("install_parallel_specialist_training_isolation_contract")
        safety = source.index("install_parallel_specialist_safety_contract")
        assert isolation < safety

    def test_detail_anchor_is_exact_when_auxiliary_authority_is_zero(self) -> None:
        from v9.parallel_specialist_safety_contract import _detail_anchored_fusion

        baseline = torch.zeros((1, 1, 1, 2))
        detail = torch.tensor([[[[0.25, -0.20]]]])
        structure = torch.tensor([[[[0.90, 0.90]]]])
        seam = torch.tensor([[[[-0.80, -0.80]]]])
        zero = torch.zeros((1, 1, 1, 2))
        fused = _detail_anchored_fusion(
            baseline, detail, structure, seam, zero, zero
        )
        torch.testing.assert_close(fused, detail, rtol=0.0, atol=0.0)

    def test_high_detail_support_suppresses_auxiliary_dilution(self) -> None:
        from v9.parallel_specialist_safety_contract import (
            _complementary_auxiliary_weight,
        )

        raw = torch.full((1, 1, 1, 1), 0.25)
        detail_evidence = torch.full_like(raw, 0.80)
        conflict = torch.zeros_like(raw)
        effective = _complementary_auxiliary_weight(
            raw, detail_evidence, conflict
        )
        assert float(effective.item()) <= 0.0100001

    def test_conflict_removes_auxiliary_authority(self) -> None:
        from v9.parallel_specialist_safety_contract import (
            _complementary_auxiliary_weight,
        )

        raw = torch.ones((1, 1, 1, 1))
        detail_evidence = torch.zeros_like(raw)
        conflict = torch.ones_like(raw)
        effective = _complementary_auxiliary_weight(
            raw, detail_evidence, conflict
        )
        torch.testing.assert_close(effective, torch.zeros_like(raw))

    def test_combined_auxiliary_authority_is_bounded(self) -> None:
        from v9.parallel_specialist_safety_contract import (
            AUXILIARY_TOTAL_WEIGHT_MAX,
            _cap_auxiliary_pair,
        )

        structure = torch.ones((1, 1, 2, 2))
        seam = torch.ones_like(structure)
        g, s = _cap_auxiliary_pair(structure, seam)
        assert float((g + s).max().item()) <= float(AUXILIARY_TOTAL_WEIGHT_MAX) + 1e-7

    def test_protected_selector_oracle_respects_one_code_drift(self) -> None:
        from v9.parallel_specialist_safety_contract import _safe_selector_oracle
        from v9.baseline_relative_specialist_contract import PROTECTED_DRIFT_TOLERANCE

        baseline = torch.zeros((1, 3, 1, 1))
        candidate = torch.ones_like(baseline)
        # This target is still inside the V12.4 protected-B definition but its
        # least-squares optimum exceeds the one-code-value deployment drift limit.
        target = torch.full_like(baseline, 2.0 / 255.0)
        oracle, safe_max, protected, _motion = _safe_selector_oracle(
            baseline, candidate, target
        )
        assert bool(protected.item())
        assert float(oracle.item()) <= float(safe_max.item()) + 1e-8
        assert float(safe_max.item()) <= float(PROTECTED_DRIFT_TOLERANCE) + 1e-8

    def test_v124_preservation_remains_outermost(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert "preservation._ORIGINAL_PRODUCTION_HEAD_LOSS = (" in source
        assert "_loss_with_protected_selector_safety" in source
        assert "preservation._loss_with_protected_preservation" in source
        assert 'candidate = outputs["fused_candidate_albedo"]' in source
