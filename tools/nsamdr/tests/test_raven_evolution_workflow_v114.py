from __future__ import annotations

from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[3]
TRAIN = ROOT / "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py"
EVOLUTION = ROOT / "tools/nsamdr/neural/v9/evolutionary_recovery.py"
CONTRACT = ROOT / "tools/nsamdr/neural/v9/local_boundary_production_contract.py"
README = ROOT / "tools/nsamdr/README.md"


class TestRavenEvolutionWorkflowV114:
    # Purpose: Implement test sources parse for TestRavenEvolutionWorkflowV114.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_sources_parse(self) -> None:
        for path in (TRAIN, EVOLUTION, CONTRACT):
            ast.parse(path.read_text(encoding="utf-8"))

    # Purpose: Implement test quick search is bounded for TestRavenEvolutionWorkflowV114.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_quick_search_is_bounded(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        assert 'population=4 if args.training_mode == "quick" else 6' in text
        assert 'micro_steps=3 if args.training_mode == "quick" else 5' in text
        assert "max_recoveries=2" in text
        assert "no candidate passed after two bounded generations" in text

    # Purpose: Implement test structural failure can recover but software cannot for TestRavenEvolutionWorkflowV114.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_structural_failure_can_recover_but_software_cannot(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        assert "FailureKind.REPRESENTATION" in text
        assert "recover_after_structural_failure" in text
        assert "_archive_and_reset_structural_attempt" in text
        # Exceptions still flow to the outer fail-closed handler.
        assert "except BaseException:" in text
        assert '"status": "interrupted-or-failed"' in text

    # Purpose: Implement test readme references evolution diagram for TestRavenEvolutionWorkflowV114.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_readme_references_evolution_diagram(self) -> None:
        text = README.read_text(encoding="utf-8")
        assert "NSAMDR_EVOLUTIONARY_RECOVERY_ARCHITECTURE.png" in text
        assert "EvolutionController" in text
        assert "training-only" in text.lower()
