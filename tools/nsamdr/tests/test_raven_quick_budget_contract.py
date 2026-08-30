"""Quick stays bounded while respecting V9Config validator minima after OOP refactor."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


class TestRavenQuickBudgetContract:
    def test_quick_is_a_short_local_capacity_run(self) -> None:
        """Verify the unchanged Quick work budget moved into application configuration.

        Purpose:
            Preserve the bounded Quick runtime contract across source reorganisation.
        Called by:
            pytest.
        Calls:
            Imports QUICK_WORK_BUDGET.
        """
        from v9.application.configuration import QUICK_WORK_BUDGET as budget

        assert budget["identity_epochs"] == 3
        assert budget["residual_epochs"] == 1
        assert budget["tiles_per_epoch"] == 64
        assert budget["raven_downstream_tiles_per_epoch"] == 16
        assert budget["seam_proof_epochs"] == 1
        assert budget["seam_authority_epochs"] == 1
        assert budget["detail_epochs"] == 1

    def test_retired_primitive_bank_uses_only_validator_minimum(self) -> None:
        """Verify the retired primitive-loader allocation remains at its legal minimum.

        Purpose:
            Prevent source refactoring from expanding obsolete B1b work.
        Called by:
            pytest.
        Calls:
            Imports QUICK_WORK_BUDGET.
        """
        from v9.application.configuration import QUICK_WORK_BUDGET as budget

        assert budget["parametric_primitive_train_tiles_per_epoch"] == 14
