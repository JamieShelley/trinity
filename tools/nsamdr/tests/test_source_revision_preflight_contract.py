from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def test_preflight_validates_exact_trainer_contract_and_explicit_refiner():
    from raven_architecture_contract import RavenArchitectureContract, _COMPONENT_PATHS
    from v9 import FidelityResidualNetV9, V9Config

    audit = RavenArchitectureContract()
    config = V9Config()
    model = FidelityResidualNetV9(config)
    audit._validate_trainer_contract(ROOT, model.architecture_contract())
    assert _COMPONENT_PATHS["ExplicitRefiner"] == (
        "geometry_net.production_structure.geometry_refiner",
    )


def test_preflight_fingerprints_all_structural_v12_owners():
    from raven_architecture_contract import RavenArchitectureContract

    hashes = RavenArchitectureContract()._source_fingerprints(ROOT)
    for relative in (
        "tools/nsamdr/neural/v9/local_boundary_production_contract.py",
        "tools/nsamdr/neural/v9/explicit_spline_refiner.py",
        "tools/nsamdr/neural/v9/edge_constrained_spline_graph.py",
        "tools/nsamdr/neural/v9/spline_graph.py",
        "tools/nsamdr/neural/v9/application/backend.py",
    ):
        assert relative in hashes
        assert len(hashes[relative]) == 64


def test_git_provenance_records_exact_head_and_clean_tracked_state():
    from raven_architecture_contract import RavenArchitectureContract

    provenance = RavenArchitectureContract()._git_provenance(ROOT)
    assert provenance["available"] is True
    assert isinstance(provenance["head"], str) and len(provenance["head"]) == 40
    assert isinstance(provenance["trackedDirty"], bool)
