from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from v9 import sr_first_contract as sr
from v9 import sr_first_runtime_contract as runtime
from v9 import sr_first_runtime_integrity_contract as integrity
from v9 import sr_first_trainer_contract as trainer


ROOT = Path(__file__).resolve().parents[3]
V9 = ROOT / "tools/nsamdr/neural/v9"
RUNTIME = V9 / "sr_first_runtime_contract.py"
INTEGRITY = V9 / "sr_first_runtime_integrity_contract.py"
TRAINER = V9 / "sr_first_trainer_contract.py"
ARCHITECTURE = ROOT / "tools/nsamdr/neural/raven_architecture_contract.py"
CONFIGURATION = V9 / "application/configuration.py"


def test_v133_detail_lr_is_the_proven_effective_rate() -> None:
    config = SimpleNamespace(learning_rate=9.0, detail_learning_rate=9.0)
    assert runtime.SR_RUNTIME_REVISION == "V13.3"
    assert runtime.SR_DETAIL_BODY_LR == 1.0e-3
    assert runtime.SR_DETAIL_ALBEDO_HEAD_LR == 3.0e-3
    assert runtime._phase_lr_sr_first("detail-reconstruction", config) == 1.0e-3

    source = CONFIGURATION.read_text(encoding="utf-8")
    assert '"detail_learning_rate": 1.0e-3,' in source
    assert '"detail_learning_rate": 1.0e-3 / 3.0' not in source


def test_v133_active_forward_does_not_call_retired_graph() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    ast.parse(source)
    start = source.index("def _active_sr_forward(")
    end = source.index("\ndef _weighted_mean(", start)
    body = source[start:end]

    assert "self.detail_net" in body
    assert "self.benefit_selector" in body
    assert "direct_detail._deterministic_baseline" in body
    assert "self.geometry_net(" not in body
    assert "self.boundary_renderer(" not in body
    assert "self.boundary_specialist(" not in body
    assert "self.seam_restorer(" not in body
    assert '_PREVIOUS_FORWARD_IMPL(' not in body

    audit = ARCHITECTURE.read_text(encoding="utf-8")
    assert '"GeometryNet": ("geometry_net",)' in audit
    assert '"DetailNet": ("detail_net",)' in audit
    assert "retired module executed" in audit


def test_v133_owns_one_detail_protection_policy() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    assert "SR_PROTECTED_DETAIL_WEIGHT = 24.0" in source
    assert "+ protected * float(SR_PROTECTED_DETAIL_WEIGHT)" in source
    assert "protected * 96.0" not in source
    assert "DETAIL_PROTECTED_EXTRA_WEIGHT" not in source
    assert "preservation._loss_with_protected_preservation" not in source
    assert "backend._compute_losses_with_production_head_supervision = _active_sr_loss" in source


def test_v133_final_runtime_integrity_requires_only_active_sr_graph() -> None:
    source = INTEGRITY.read_text(encoding="utf-8")
    ast.parse(source)
    assert integrity.SR_RUNTIME_INTEGRITY_REVISION == "V13.3"
    assert "boundary_refined_coverage" not in integrity._REQUIRED_OUTPUTS
    assert "seam_authority" not in integrity._REQUIRED_OUTPUTS
    assert "detail_candidate_albedo" in integrity._REQUIRED_OUTPUTS
    assert "benefit_selector_probability" in integrity._REQUIRED_OUTPUTS
    assert "executedRetiredComponents" in source
    assert "retiredComponentsBypassed" in source
    assert "service._run_final_qualification = _run_v133_runtime_integrity" in source
    assert "<locals>" not in source


def test_v133_trainer_has_no_legacy_training_authority() -> None:
    source = TRAINER.read_text(encoding="utf-8")
    ast.parse(source)
    assert trainer.SR_TRAINER_REVISION == "V13.3"
    assert set(trainer._ACTIVE_COMPONENT_PATHS) == {
        "conditioned detail",
        "albedo physical head",
        "normal physical head",
        "material physical head",
        "confidence",
        "regret",
        "BenefitSelector",
    }
    assert trainer._RETIRED_PHASES == {
        "sdf-bootstrap",
        "sdf-proof",
        "seam-proof",
        "seam-authority",
        "gate-proof",
    }
    assert "service._production_component_modules = _active_component_modules" in source
    assert "service._explicit_primitive_structure_microproof = _retired_structure_microproof" in source
    assert "service._profile_specialist_microproof = _retired_profile_microproof" in source
    assert "service._phase_seam_sr_microproof = _retired_seam_microproof" in source


def test_grid_excess_does_not_punish_a_real_target_edge() -> None:
    target = torch.zeros(1, 3, 16, 16)
    target[..., 4:] = 1.0
    candidate = target.clone()
    assert torch.isclose(sr._grid_excess(candidate, target, 4), torch.tensor(0.0))


def test_grid_excess_detects_candidate_only_four_pixel_blocks() -> None:
    target = torch.zeros(1, 3, 16, 16)
    candidate = target.clone()
    candidate[..., 4:8] = 0.25
    candidate[..., 12:16] = 0.25
    assert float(sr._grid_excess(candidate, target, 4)) > 0.0


def test_identity_candidate_has_zero_grid_excess() -> None:
    target = torch.rand(1, 3, 16, 16)
    assert torch.isclose(sr._grid_excess(target, target, 4), torch.tensor(0.0))


def test_runtime_integrity_terminology_replaces_false_quality_name() -> None:
    value = {
        "finalQualification": {"passed": True, "kind": "runtime"},
        "detailQualified": False,
    }
    renamed = runtime._rename_runtime_integrity(value)
    assert "finalQualification" not in renamed
    assert renamed["productionRuntimeIntegrity"]["passed"] is True
    assert renamed["detailQualified"] is False
