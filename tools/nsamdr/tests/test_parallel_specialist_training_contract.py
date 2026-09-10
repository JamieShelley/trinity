"""Static contract checks for V12.5 parallel-specialist training alignment."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "tools/nsamdr/neural/v9/parallel_specialist_training_contract.py"
INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"


class TestParallelSpecialistTrainingContract:
    # Purpose: Ensure training alignment installs only after V12.5 production fusion.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_install_order_is_fusion_then_training(self) -> None:
        source = INIT.read_text(encoding="utf-8")
        fusion = source.index("install_parallel_specialist_fusion_contract")
        training = source.index("install_parallel_specialist_training_contract")
        assert fusion < training

    # Purpose: Ensure seam authority is judged from deterministic B, not serial G output.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_seam_authority_targets_b_relative_candidate(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert 'PARALLEL_SPECIALIST_TRAINING_REVISION = "V12.5"' in source
        assert 'baseline = outputs["baseline_albedo"].detach().float()' in source
        assert 'deployed = outputs["seam_candidate_albedo"].float()' in source
        assert 'proposal = (baseline + phase_delta).clamp(0.0, 1.0)' in source
        assert 'before = outputs["boundary_pre_seam_albedo"]' not in source

    # Purpose: Ensure final authority trains against complete fused U rather than D alone.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_selector_targets_fused_candidate(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert 'candidate = outputs["fused_candidate_albedo"].detach().float()' in source
        assert '_production_base_total(outputs, batch, config, phase) * 0.25' in source
        assert 'candidate = outputs["detail_candidate_albedo"].detach().float()' not in source

    # Purpose: Preserve V12.4 as the outer loss owner after V12.5 alignment.
    # Called by: pytest.
    # Calls: source-file reads only.
    def test_protected_preservation_remains_outermost(self) -> None:
        source = CONTRACT.read_text(encoding="utf-8")
        assert "preservation._ORIGINAL_PRODUCTION_HEAD_LOSS = (" in source
        assert "_compute_losses_with_parallel_specialist_supervision" in source
        assert "backend._compute_losses_with_production_head_supervision = (" in source
        assert "preservation._loss_with_protected_preservation" in source
