from __future__ import annotations

from types import SimpleNamespace

import torch

from v9 import sr_first_contract as sr
from v9 import sr_first_runtime_contract as runtime


def test_observable_sr_condition_has_six_lr_only_channels() -> None:
    inputs = torch.zeros(1, 17, 8, 8)
    inputs[:, 16] = 0.25
    inputs[:, 9] = 0.10
    inputs[:, 10] = -0.20
    inputs[:, 11] = -0.30
    inputs[:, 12] = 0.40
    inputs[:, 13] = -0.50

    condition = sr._observable_sr_condition(inputs, (32, 32))

    assert condition.shape == (1, 6, 32, 32)
    assert torch.allclose(condition[:, 0], torch.full_like(condition[:, 0], 0.25))
    assert torch.allclose(condition[:, 1], torch.full_like(condition[:, 1], 0.10))
    assert torch.allclose(condition[:, 2], torch.full_like(condition[:, 2], -0.20))
    assert torch.allclose(condition[:, 3], torch.full_like(condition[:, 3], 0.30))
    assert torch.allclose(condition[:, 4], torch.full_like(condition[:, 4], 0.40))
    assert torch.allclose(condition[:, 5], torch.full_like(condition[:, 5], -0.50))


def test_observable_sr_condition_ignores_unrelated_physical_values() -> None:
    left = torch.zeros(1, 17, 8, 8)
    right = left.clone()
    right[:, 0:8] = torch.rand_like(right[:, 0:8])
    right[:, 8] = 0.77
    right[:, 14:16] = torch.rand_like(right[:, 14:16])

    assert torch.equal(
        sr._observable_sr_condition(left, (32, 32)),
        sr._observable_sr_condition(right, (32, 32)),
    )


def test_target_material_matches_production_three_channel_encoding() -> None:
    batch = {
        "target_material_class": torch.tensor([[[0, 1], [2, 3]]]),
        "target_emissive": torch.full((1, 1, 2, 2), 0.25),
        "target_roughness": torch.full((1, 1, 2, 2), 0.75),
    }
    config = SimpleNamespace(material_classes=4)

    target = sr._target_material(batch, config)

    assert target.shape == (1, 3, 2, 2)
    assert torch.allclose(
        target[:, 0],
        torch.tensor([[[0.125, 0.375], [0.625, 0.875]]]),
    )
    assert torch.allclose(target[:, 1], torch.full((1, 2, 2), 0.25))
    assert torch.allclose(target[:, 2], torch.full((1, 2, 2), 0.75))


def test_v13_contract_declares_sr_as_final_candidate() -> None:
    assert sr.SR_FIRST_REVISION == "V13.2"
    assert runtime.SR_RUNTIME_REVISION == "V13.3"
    assert sr.SR_LAPLACIAN_WEIGHT > 0.0
    assert sr.SR_PYRAMID_WEIGHT > 0.0
    assert sr.SR_GRID_EXCESS_WEIGHT > 0.0
