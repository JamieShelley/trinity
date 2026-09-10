"""Contract checks for V12.7 staged specialist training isolation."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "tools/nsamdr/neural/v9/parallel_specialist_training_isolation_contract.py"
INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"


class TestParallelSpecialistTrainingIsolationContract:
    def test_install_order_is_after_v126_arbitration(self) -> None:
        source = INIT.read_text(encoding="utf-8")
        arbitration = source.index("install_parallel_specialist_arbitration_contract")
        isolation = source.index("install_parallel_specialist_training_isolation_contract")
        assert arbitration < isolation

    def test_support_heads_cannot_rewrite_detail_backbone(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert 'PARALLEL_SPECIALIST_TRAINING_ISOLATION_REVISION = "V12.7"' in source
        assert "support_x = x.detach()" in source
        assert '"albedo_raw": torch.tanh(self.albedo_head(x))' in source
        assert '"confidence_logits": self.confidence_head(support_x)' in source
        assert '"regret_logits": self.regret_head(support_x)' in source

    def test_detail_support_targets_are_relative_to_deterministic_b(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert 'baseline = outputs["baseline_albedo"].detach().float()' in source
        assert 'candidate = outputs["detail_candidate_albedo"].detach().float()' in source
        assert "improvement = (baseline_error - candidate_error).detach()" in source
        assert "candidate_error > baseline_error + 0.001" in source
        assert 'losses["total"] = losses["total"].float() - previous_support.float()' in source

    def test_b1b_scores_the_deployed_structure_candidate(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert 'required = ("structure_candidate_albedo", "baseline_albedo")' in source
        assert 'candidate = outputs["structure_candidate_albedo"].float()' in source
        assert '"b1b_deployed_structure_objective"' in source
        assert "b1._live_production_objective = _live_deployed_structure_objective" in source

    def test_preservation_remains_outermost(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert "preservation._ORIGINAL_PRODUCTION_HEAD_LOSS = (" in source
        assert "_loss_with_b_relative_detail_support" in source
        assert "preservation._loss_with_protected_preservation" in source
