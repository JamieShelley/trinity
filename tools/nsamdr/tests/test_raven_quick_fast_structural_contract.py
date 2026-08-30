"""Quick local structure must fail fast before expensive downstream work after OOP refactor."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


class TestRavenQuickFastStructuralContract:
    def test_quick_local_gate_precedes_seams(self) -> None:
        """Verify local geometry remains the first pass-driven production stage.

        Purpose:
            Preserve fail-fast structural ordering.
        Called by:
            pytest.
        Calls:
            QualificationGates(), StagePlan().
        """
        from v9.application.gates import QualificationGates, StagePlan

        phases = [
            definition.phase
            for definition in StagePlan(QualificationGates()).definitions
        ]
        assert phases.index("sdf-bootstrap") < phases.index("seam-proof")

    def test_no_primitive_accuracy_or_mae_gate_in_local_promotion(self) -> None:
        """Verify local geometry gate contains no retired whole-tile classifier thresholds.

        Purpose:
            Prevent the source refactor from reintroducing obsolete B1b authority.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.application.gates import QualificationGates

        source = inspect.getsource(QualificationGates.local_geometry_gate)
        assert "parametric_primitive_class_accuracy_required" not in source
        assert "parametric_primitive_param_mae_required" not in source
