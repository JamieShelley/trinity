"""Contract checks for the V12.9.1 Raven hard-qualification corrections."""
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))

V6 = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic_v6.py"
LAUNCHER = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic.py"


def test_global_sdf_gauge_selects_equivalent_negative_polarity() -> None:
    from run_nsamdr_v9_raven_micro_diagnostic_v6 import _align_global_sdf_polarity

    target = np.asarray([[-2.0, -0.5, 0.5, 2.0]], dtype=np.float32)
    field = -target
    aligned, polarity = _align_global_sdf_polarity(field, target)
    assert polarity == -1.0
    np.testing.assert_allclose(aligned, target, rtol=0.0, atol=0.0)


def test_g0_uses_production_topology_non_regression_not_absolute_hr_match() -> None:
    source = V6.read_text(encoding="utf-8")
    assert 'float(geometry["topologyRegression"]) == 0.0' in source
    assert 'float(geometry["predictedMissingContourFraction"])' in source
    assert 'geometry["predictedTopologyMismatch"] == 0.0' not in source


def test_g1_adds_rendered_topology_and_catastrophic_chamfer_safety() -> None:
    source = V6.read_text(encoding="utf-8")
    assert '"renderedTopologyRegression"' in source
    assert '"sdf_catastrophic_chamfer_pixels"' in source
    assert '"sdf_improvement_margin_pixels"' in source
    assert '"sdf_sign_gauge_invariant"' in source


def test_compatibility_launcher_routes_to_v1291() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "from run_nsamdr_v9_raven_micro_diagnostic_v6 import main" in source
