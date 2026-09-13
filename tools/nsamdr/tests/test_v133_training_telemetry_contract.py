from __future__ import annotations

import torch

from v9 import sr_first_training_telemetry_contract as telemetry


def test_detail_sr_metrics_supply_only_required_legacy_progress_keys() -> None:
    total = torch.tensor(3.0)
    losses = {
        "total": total,
        "sr_albedo_global": torch.tensor(0.25),
        "sr_regret": torch.tensor(0.125),
    }
    result = telemetry._add_v133_trainer_metrics(losses, "detail-reconstruction")

    assert torch.equal(result["albedo"], losses["sr_albedo_global"])
    assert torch.equal(result["regret"], losses["sr_regret"])
    assert float(result["sdf"]) == 0.0
    assert float(result["tangent_coherence"]) == 0.0
    assert float(result["curvature_coherence"]) == 0.0


def test_selector_metrics_map_to_final_reconstruction_progress() -> None:
    losses = {
        "total": torch.tensor(2.0),
        "selector_edge_reconstruction": torch.tensor(0.20),
        "selector_edge_regret": torch.tensor(0.05),
    }
    result = telemetry._add_v133_trainer_metrics(losses, "physical-finetune")

    assert torch.equal(result["albedo"], losses["selector_edge_reconstruction"])
    assert torch.equal(result["regret"], losses["selector_edge_regret"])
    assert float(result["sdf"]) == 0.0


def test_detail_outcome_metrics_drive_legacy_detail_gate_truthfully() -> None:
    target = torch.zeros(2, 3, 8, 8)
    baseline = torch.full_like(target, 0.5)
    candidate = baseline.clone()
    candidate[0].fill_(0.25)
    candidate[1].fill_(0.75)
    outputs = {
        "baseline_albedo": baseline,
        "sr_candidate_albedo": candidate,
        "albedo": candidate,
    }
    batch = {"target_albedo": target}

    metrics = telemetry._reconstruction_outcome_metrics(
        outputs, batch, "detail-reconstruction"
    )
    assert torch.isclose(metrics["recovery"], torch.tensor(0.0))
    assert torch.isclose(metrics["win_fraction"], torch.tensor(0.5))
    assert torch.isclose(metrics["regression_fraction"], torch.tensor(0.5))


def test_selector_outcome_metrics_measure_final_not_raw_candidate() -> None:
    target = torch.zeros(1, 3, 8, 8)
    baseline = torch.full_like(target, 0.5)
    raw_candidate = torch.full_like(target, 0.9)
    final = torch.full_like(target, 0.25)
    outputs = {
        "baseline_albedo": baseline,
        "sr_candidate_albedo": raw_candidate,
        "albedo": final,
    }
    batch = {"target_albedo": target}

    metrics = telemetry._reconstruction_outcome_metrics(
        outputs, batch, "physical-finetune"
    )
    assert float(metrics["recovery"]) > 0.0
    assert float(metrics["win_fraction"]) == 1.0
    assert float(metrics["regression_fraction"]) == 0.0


def test_adapter_does_not_modify_total_or_sr_authority_metrics() -> None:
    losses = {
        "total": torch.tensor(7.0, requires_grad=True),
        "sr_grid_excess": torch.tensor(0.33),
    }
    result = telemetry._add_v133_trainer_metrics(losses, "detail-reconstruction")

    assert result["total"] is losses["total"]
    assert result["sr_grid_excess"] is losses["sr_grid_excess"]
