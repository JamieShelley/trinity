from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def test_preflight_validates_exact_trainer_contract_and_retired_refiner_inventory():
    from raven_architecture_contract import (
        RavenArchitectureContract,
        _ACTIVE_COMPONENTS,
        _RETIRED_COMPONENTS,
    )
    from v9 import FidelityResidualNetV9, V9Config

    audit = RavenArchitectureContract()
    config = V9Config()
    model = FidelityResidualNetV9(config)
    audit._validate_trainer_contract(ROOT, model.architecture_contract())
    assert _RETIRED_COMPONENTS["ExplicitRefiner"] == (
        "geometry_net.production_structure.geometry_refiner",
    )
    assert _ACTIVE_COMPONENTS["DetailNet"] == ("detail_net",)
    assert _ACTIVE_COMPONENTS["BenefitSelector"] == ("benefit_selector",)


def test_preflight_fingerprints_v133_active_runtime_owners():
    from raven_architecture_contract import RavenArchitectureContract

    hashes = RavenArchitectureContract()._source_fingerprints(ROOT)
    for relative in (
        "tools/nsamdr/neural/v9/__init__.py",
        "tools/nsamdr/neural/v9/model.py",
        "tools/nsamdr/neural/v9/sr_first_contract.py",
        "tools/nsamdr/neural/v9/sr_first_runtime_contract.py",
        "tools/nsamdr/neural/v9/sr_first_generalization_contract.py",
        "tools/nsamdr/neural/v9/application/backend.py",
        "tools/nsamdr/neural/raven_architecture_contract.py",
    ):
        assert relative in hashes
        assert len(hashes[relative]) == 64


def test_git_provenance_records_exact_head_and_clean_tracked_state():
    from raven_architecture_contract import RavenArchitectureContract

    provenance = RavenArchitectureContract()._git_provenance(ROOT)
    assert provenance["available"] is True
    assert isinstance(provenance["head"], str) and len(provenance["head"]) == 40
    assert isinstance(provenance["trackedDirty"], bool)
