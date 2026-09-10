"""Contract checks for V12.5 independent parallel specialist fusion."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))

CONTRACT = ROOT / "tools/nsamdr/neural/v9/parallel_specialist_fusion_contract.py"
INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"


class TestParallelSpecialistFusionContract:
    # Purpose: Require package-wide installation after detail isolation/preservation.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_install_order_preserves_v12_3_and_v12_4_contracts(self) -> None:
        source = INIT.read_text(encoding="utf-8")
        detail = source.index("install_parallel_detail_contract")
        preservation = source.index("install_baseline_relative_specialist_contract")
        fusion = source.index("install_parallel_specialist_fusion_contract")
        assert detail < preservation < fusion

    # Purpose: Ensure seam production is anchored to B and source-observable evidence.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_seam_candidate_has_no_learned_geometry_prerequisite(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert 'PARALLEL_SPECIALIST_FUSION_REVISION = "V12.5"' in source
        assert "_deterministic_baseline_from_sources" in source
        assert "neutral_sdf = torch.full_like(observed_edge, 1.0e4)" in source
        assert "geometry_normal=None" in source
        assert "source_albedo=source_albedo" in source

    # Purpose: Keep structure/seam/detail candidates independently observable.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_forward_exposes_independent_candidates_and_fused_candidate(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        for key in (
            '"structure_candidate_albedo"',
            '"seam_candidate_albedo"',
            '"detail_candidate_albedo"',
            '"fused_candidate_albedo"',
            '"parallel_fusion_conflict"',
        ):
            assert key in source
        assert "final_gate = selector_probability" in source
        assert "confidence_support * regret_suppression" not in source

    # Purpose: Verify a sole supported candidate survives bounded fusion exactly.
    # Called by: pytest.
    # Calls: V12.5 pure fusion helper.
    def test_single_supported_candidate_is_retained_exactly(self) -> None:
        from v9.parallel_specialist_fusion_contract import (
            _weighted_delta_fusion,
        )

        baseline = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        identity = baseline.clone()
        detail = torch.full_like(baseline, 0.125)
        zero = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        one = torch.ones((1, 1, 2, 2), dtype=torch.float32)

        fused = _weighted_delta_fusion(
            baseline,
            (identity, identity, detail),
            (zero, zero, one),
        )
        torch.testing.assert_close(fused, detail, rtol=0.0, atol=0.0)

    # Purpose: Verify overlapping supported candidates cannot exceed their delta envelope.
    # Called by: pytest.
    # Calls: V12.5 pure fusion helper.
    def test_overlap_is_a_support_weighted_convex_delta(self) -> None:
        from v9.parallel_specialist_fusion_contract import (
            _weighted_delta_fusion,
        )

        baseline = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
        structure = torch.full_like(baseline, -0.20)
        seam = torch.full_like(baseline, 0.10)
        detail = torch.full_like(baseline, 0.05)
        one = torch.ones((1, 1, 1, 1), dtype=torch.float32)

        fused = _weighted_delta_fusion(
            baseline,
            (structure, seam, detail),
            (one, one, one),
        )
        value = float(fused.item())
        assert -0.20 <= value <= 0.10
        assert abs(value - ((-0.20 + 0.10 + 0.05) / 3.0)) < 1.0e-7
