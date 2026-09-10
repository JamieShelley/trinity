"""Contract checks for V12.6 specialist support arbitration."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))

CONTRACT = ROOT / "tools/nsamdr/neural/v9/parallel_specialist_arbitration_contract.py"
INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"


class TestParallelSpecialistArbitrationContract:
    def test_install_order_is_after_v125_fusion_and_training(self) -> None:
        source = INIT.read_text(encoding="utf-8")
        fusion = source.index("install_parallel_specialist_fusion_contract")
        training = source.index("install_parallel_specialist_training_contract")
        arbitration = source.index("install_parallel_specialist_arbitration_contract")
        assert fusion < training < arbitration

    def test_detail_support_uses_confidence_and_inverse_regret(self) -> None:
        from v9.parallel_specialist_arbitration_contract import _detail_support

        confidence = torch.tensor([[[[0.8]]]])
        regret = torch.tensor([[[[0.25]]]])
        support = _detail_support(confidence, regret)
        torch.testing.assert_close(support, torch.tensor([[[[0.6]]]]))

    def test_support_sharpening_preserves_identity_and_suppresses_weak_support(self) -> None:
        from v9.parallel_specialist_arbitration_contract import _sharpen_support

        support = torch.tensor([[[[1.0, 0.44, 0.11]]]])
        sharpened = _sharpen_support(support)
        assert float(sharpened[0, 0, 0, 0]) == 1.0
        assert float(sharpened[0, 0, 0, 1]) < 0.04
        assert float(sharpened[0, 0, 0, 2]) < 0.001

    def test_detail_only_fusion_remains_exact_detail(self) -> None:
        from v9.parallel_specialist_arbitration_contract import _detail_support, _sharpen_support
        from v9.parallel_specialist_fusion_contract import _weighted_delta_fusion

        baseline = torch.zeros((1, 1, 1, 1))
        detail = torch.full_like(baseline, 0.25)
        identity = baseline.clone()
        zero = torch.zeros((1, 1, 1, 1))
        detail_weight = _sharpen_support(
            _detail_support(torch.zeros_like(zero), torch.ones_like(zero))
        )
        fused = _weighted_delta_fusion(
            baseline,
            (identity, identity, detail),
            (zero, zero, detail_weight),
        )
        torch.testing.assert_close(fused, detail, rtol=0.0, atol=0.0)

    def test_weak_competitors_do_not_materially_dilute_dominant_detail(self) -> None:
        from v9.parallel_specialist_arbitration_contract import _sharpen_support
        from v9.parallel_specialist_fusion_contract import _weighted_delta_fusion

        baseline = torch.zeros((1, 1, 1, 1))
        structure = torch.zeros_like(baseline)
        seam = torch.zeros_like(baseline)
        detail = torch.ones_like(baseline)
        supports = tuple(
            _sharpen_support(torch.full_like(baseline, value))
            for value in (0.11, 0.44, 1.0)
        )
        fused = _weighted_delta_fusion(
            baseline,
            (structure, seam, detail),
            supports,
        )
        assert float(fused.item()) > 0.96

    def test_detail_support_training_is_baseline_relative_and_preservation_stays_outer(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert 'baseline = outputs["baseline_albedo"].detach().float()' in source
        assert 'candidate = outputs["detail_candidate_albedo"].detach().float()' in source
        assert "improvement = (baseline_error - candidate_error).detach()" in source
        assert 'outputs["detail_confidence_logits"]' in source
        assert 'outputs["detail_regret_logits"]' in source
        assert 'losses["total"] = losses["total"].float() + support_loss' in source
        assert "preservation._ORIGINAL_PRODUCTION_HEAD_LOSS = _loss_with_detail_support_calibration" in source
        assert "preservation._loss_with_protected_preservation" in source
