from __future__ import annotations

from types import SimpleNamespace

from v9 import sr_first_quality_contract as quality


def test_v131_quality_targets_are_stricter_than_v13_capacity_gate() -> None:
    assert quality.SR_QUALITY_REVISION == "V13.1"
    assert quality.SR_ALBEDO_RESIDUAL_CAP == 0.40
    assert quality.SR_REQUIRED_EDGE_RECOVERY == 0.70
    assert quality.SR_REQUIRED_GLOBAL_RECOVERY == 0.50
    assert quality.SR_REQUIRED_GRADIENT_RECOVERY == 0.40
    assert quality.SR_REQUIRED_SELECTOR_RETENTION == 0.90
    assert quality.SR_PLATEAU_MIN_STEPS < 3072
    assert quality.SR_PLATEAU_STALE_EVALS > 0


def test_selector_ranking_treats_99_percent_as_constraint_not_reward() -> None:
    lower_preservation_higher_fidelity = quality.selector_checkpoint_key(
        preservation=0.991,
        edge_recovery=0.66,
        global_recovery=0.52,
        edge_retention=0.94,
        global_retention=0.93,
        preservation_required=0.99,
    )
    higher_preservation_lower_fidelity = quality.selector_checkpoint_key(
        preservation=0.999,
        edge_recovery=0.60,
        global_recovery=0.47,
        edge_retention=0.90,
        global_retention=0.90,
        preservation_required=0.99,
    )
    assert lower_preservation_higher_fidelity > higher_preservation_lower_fidelity


def test_any_safe_selector_checkpoint_beats_unsafe_checkpoint() -> None:
    safe = quality.selector_checkpoint_key(
        preservation=0.990,
        edge_recovery=0.40,
        global_recovery=0.30,
        edge_retention=0.90,
        global_retention=0.90,
        preservation_required=0.99,
    )
    unsafe = quality.selector_checkpoint_key(
        preservation=0.989,
        edge_recovery=0.90,
        global_recovery=0.80,
        edge_retention=1.00,
        global_retention=1.00,
        preservation_required=0.99,
    )
    assert safe > unsafe
