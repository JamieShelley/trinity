from __future__ import annotations

"""V12.10 direct same-edge geometry authority for B1/G training.

The V12.9.1 Raven hard-qualification run exposed a concrete training mismatch in
B1b/G.  The production spline loss computes exact same-owning-edge target nodes and
tangents, and the semantic config explicitly declares those terms to be the primary
continuous-geometry authority.  Later deployment-alignment contracts nevertheless
replaced the B1b total with the rendered G objective, leaving the direct spline
teacher as telemetry only.  On the failing Raven patch, 3,072 B1b updates therefore
left same-edge point wins near 27% and deployed G below deterministic B.

V12.10 keeps the deployed B-relative G objective intact and adds the already-existing
same-edge point/tangent teacher back to SGD.  The teacher cannot invent unrelated
HR-only topology: targets exist only for crossings on the source graph's current
owning edges.  No inference values, trainable parameters, checkpoint keys or input
contracts change.
"""

from typing import Any

import torch

from .model import FidelityResidualNetV9


PARALLEL_SPECIALIST_GEOMETRY_TRAINING_REVISION = "V12.10"
SAME_EDGE_REGRET_WEIGHT_FRACTION = 0.50

_INSTALLED = False
_PREVIOUS_B1B_LOSS: Any = None
_ORIGINAL_ARCHITECTURE_CONTRACT: Any = None


def _require_scalar_loss(
    losses: dict[str, Any],
    key: str,
) -> torch.Tensor:
    value = losses.get(key)
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise RuntimeError(
            f"V12.10 B1 same-edge supervision requires scalar loss {key!r}"
        )
    return value.float()


def _loss_with_same_edge_geometry_authority(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
) -> dict[str, torch.Tensor]:
    """Add exact shared-node/tangent supervision to the deployed-G B1 objective."""
    if _PREVIOUS_B1B_LOSS is None:
        raise RuntimeError("V12.10 geometry training installed without previous B1 loss")

    losses = _PREVIOUS_B1B_LOSS(outputs, batch, config, phase)
    if phase != "sdf-proof":
        return losses

    point = _require_scalar_loss(losses, "spline_graph_point")
    tangent = _require_scalar_loss(losses, "spline_graph_tangent")
    point_regret = _require_scalar_loss(losses, "spline_graph_point_regret")

    # These weights already belong to the production semantic configuration.  The
    # point/tangent comments in config.py explicitly describe them as the primary
    # B1b geometry authority, so do not introduce a second diagnostic-only scale.
    point_weight = float(getattr(config, "spline_graph_point_weight", 96.0))
    tangent_weight = float(getattr(config, "spline_graph_tangent_weight", 48.0))
    regret_weight = point_weight * float(SAME_EDGE_REGRET_WEIGHT_FRACTION)

    direct_geometry = (
        point * point_weight
        + tangent * tangent_weight
        + point_regret * regret_weight
    ).float()
    if not direct_geometry.requires_grad or direct_geometry.grad_fn is None:
        raise RuntimeError(
            "V12.10 same-edge B1 objective is detached from the production spline initializer"
        )

    deployed_before = _require_scalar_loss(losses, "total")
    losses["b1b_deployed_objective_before_same_edge"] = deployed_before.detach()
    losses["b1b_same_edge_point_supervision"] = (point * point_weight).detach()
    losses["b1b_same_edge_tangent_supervision"] = (tangent * tangent_weight).detach()
    losses["b1b_same_edge_regret_supervision"] = (
        point_regret * regret_weight
    ).detach()
    losses["b1b_same_edge_direct_total"] = direct_geometry.detach()
    losses["b1b_same_edge_direct_requires_grad"] = direct_geometry.new_tensor(1.0).detach()
    losses["total"] = deployed_before + direct_geometry
    return losses


def _architecture_contract_with_geometry_training(
    self: FidelityResidualNetV9,
) -> dict[str, object]:
    if _ORIGINAL_ARCHITECTURE_CONTRACT is None:
        raise RuntimeError("V12.10 geometry training missing architecture contract")
    contract = dict(_ORIGINAL_ARCHITECTURE_CONTRACT(self))
    contract["parallelSpecialistGeometryTrainingRevision"] = (
        PARALLEL_SPECIALIST_GEOMETRY_TRAINING_REVISION
    )
    contract["structureTrainingAuthority"] = (
        "deployed B-relative structure candidate G plus exact same-owning-edge "
        "connected-spline node/tangent supervision; topology remains B1a-locked"
    )
    contract["sameEdgeGeometryTeacherAuthority"] = True
    contract["sameEdgeGeometryTeacherCanChangeTopology"] = False
    return contract


def install_parallel_specialist_geometry_training_contract() -> None:
    """Install after V12.8 while preserving every existing inference adapter."""
    global _INSTALLED, _PREVIOUS_B1B_LOSS, _ORIGINAL_ARCHITECTURE_CONTRACT
    if _INSTALLED:
        return

    from .application import backend

    current_b1b = backend._compute_losses_with_b1b_renderer_supervision
    if current_b1b is not _loss_with_same_edge_geometry_authority:
        _PREVIOUS_B1B_LOSS = current_b1b
        backend._compute_losses_with_b1b_renderer_supervision = (
            _loss_with_same_edge_geometry_authority
        )

    _ORIGINAL_ARCHITECTURE_CONTRACT = FidelityResidualNetV9.architecture_contract
    FidelityResidualNetV9.architecture_contract = (
        _architecture_contract_with_geometry_training
    )
    _INSTALLED = True
