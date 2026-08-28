from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def test_genome_is_bounded_and_state_dict_compatible() -> None:
    torch = pytest.importorskip("torch")
    from v9.config import V9Config
    from v9.evolutionary_recovery import GENOME_NAMES, Genome
    from v9.local_boundary_production_contract import set_active_evolution_genome
    from v9.model import FidelityResidualNetV9

    genome = Genome(
        feature_gain=99.0,
        evidence_gain=-99.0,
        distance_scale=1.2,
        curvature_scale=1.1,
        ribbon_scale=0.9,
        extra_branch_gain=1.3,
        csg_logit_scale=0.8,
        correction_scale=1.05,
    ).bounded()
    set_active_evolution_genome(genome.to_dict())
    model = FidelityResidualNetV9(V9Config())
    key = "geometry_net.production_structure.evolution_genome"
    assert key in model.state_dict()
    tensor = model.state_dict()[key]
    assert tuple(tensor.shape) == (len(GENOME_NAMES),)
    assert torch.isfinite(tensor).all()


def test_evolution_controller_is_not_a_production_component() -> None:
    from v9.config import V9Config
    from v9.model import FidelityResidualNetV9

    model = FidelityResidualNetV9(V9Config())
    contract = model.architecture_contract()
    components = contract["productionComponents"]
    assert "evolution" not in " ".join(str(k).lower() for k in components)
    assert contract["evolutionaryGenomeCheckpointed"] is True
    assert contract["evolutionaryControllerInferenceAuthority"] is False
    assert components["structural representation"] == "geometry_net.production_structure"


def test_full_production_forward_observes_evolved_structure() -> None:
    torch = pytest.importorskip("torch")
    from v9.config import V9Config
    from v9.evolutionary_recovery import Genome
    from v9.local_boundary_production_contract import set_active_evolution_genome
    from v9.model import FidelityResidualNetV9

    set_active_evolution_genome(Genome(correction_scale=0.85).to_dict())
    model = FidelityResidualNetV9(V9Config()).eval()
    calls = {"count": 0}
    handle = model.geometry_net.production_structure.register_forward_hook(
        lambda *_args: calls.__setitem__("count", calls["count"] + 1)
    )
    sample = torch.rand((1, 17, 16, 16), dtype=torch.float32)
    sample[:, 3:5] = sample[:, 3:5] * 2.0 - 1.0
    try:
        with torch.inference_mode():
            output = model(sample)
    finally:
        handle.remove()
    assert calls["count"] > 0
    for key in ("albedo", "normal_xy", "material", "roughness", "emissive"):
        assert key in output
        assert torch.isfinite(output[key]).all()
        assert tuple(output[key].shape[-2:]) == (64, 64)


def test_failure_classifier_never_evolves_software_exception() -> None:
    from v9.evolutionary_recovery import FailureKind, classify_failure

    assert classify_failure(error=AttributeError("missing tensor")) == FailureKind.SOFTWARE
    assert classify_failure(error=RuntimeError("CUDA out of memory")) == FailureKind.NUMERICAL
    assert classify_failure(metrics={
        "sdf_stageb_topology_regression_fraction": 1.0,
        "sdf_predicted_missing_contour_fraction": 0.2,
        "sdf_source_missing_contour_fraction": 0.1,
        "sdf_zero_contour_relative_gain_mean": -0.2,
        "sdf_zero_contour_chamfer_pixels": 5.0,
    }) == FailureKind.REPRESENTATION
