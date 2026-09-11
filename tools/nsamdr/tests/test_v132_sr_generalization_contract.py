from __future__ import annotations

from pathlib import Path

from v9.config import V9Config
from v9 import sr_first_generalization_contract as v132


def _good_row() -> dict[str, float]:
    return {
        "candidateEdgeRecovery": 0.68,
        "candidateGlobalRecovery": 0.52,
        "candidateGradientRecovery": 0.42,
        "candidateNormalRecovery": 0.18,
        "candidateMaterialRecovery": 0.24,
        "finalEdgeRecovery": 0.64,
        "finalGlobalRecovery": 0.49,
        "finalGradientRecovery": 0.39,
        "selectorMean": 0.73,
    }


def test_v132_quick_schedule_survives_legacy_validation() -> None:
    config = V9Config()
    for key, value in v132.QUICK_SR_WORK_BUDGET.items():
        setattr(config, key, value)
    config.validate()

    assert v132.is_sr_first_quick_config(config)
    assert config.identity_epochs == 0
    assert config.residual_epochs == 0
    assert config.seam_proof_epochs == 0
    assert config.seam_authority_epochs == 0
    assert config.boundary_epochs == 0
    assert config.detail_epochs == 8
    assert config.physical_finetune_epochs == 3
    assert config.raven_downstream_tiles_per_epoch == 384
    assert config.validation_tiles >= 32
    assert config.detail_albedo_max_delta == 0.40


def test_v132_representative_summary_accepts_quality_bank() -> None:
    report = v132.summarize_representative_validation(
        [_good_row() for _ in range(32)],
        protected_safe=995,
        protected_count=1000,
    )

    assert report["pass"] is True
    assert report["patchCount"] == 32
    assert report["selectorEdgeRetention"] >= 0.90
    assert report["protectedPreservation"] >= 0.99


def test_v132_representative_summary_rejects_one_catastrophic_patch() -> None:
    rows = [_good_row() for _ in range(32)]
    rows[7] = {
        **_good_row(),
        "candidateEdgeRecovery": -0.15,
        "candidateGlobalRecovery": -0.12,
        "finalEdgeRecovery": -0.16,
        "finalGlobalRecovery": -0.13,
    }
    report = v132.summarize_representative_validation(
        rows,
        protected_safe=995,
        protected_count=1000,
    )

    assert report["pass"] is False
    assert report["worstCandidateRecovery"] < -0.10
    assert any("catastrophic" in reason for reason in report["reasons"])


def test_v132_representative_summary_requires_protected_preservation() -> None:
    report = v132.summarize_representative_validation(
        [_good_row() for _ in range(32)],
        protected_safe=980,
        protected_count=1000,
    )

    assert report["pass"] is False
    assert any("protected-B preservation" in reason for reason in report["reasons"])


def test_v132_architecture_contract_is_installed() -> None:
    config = V9Config()
    model = __import__("v9.model", fromlist=["FidelityResidualNetV9"]).FidelityResidualNetV9(config)
    contract = model.architecture_contract()

    assert contract["srGeneralizationRevision"] == "V13.2"
    assert contract["representativeRavenQualification"]["patches"] == 32
    assert contract["finalCandidate"] == "sr_candidate"
