from __future__ import annotations

"""V12.3.1 optimisation parity for the independent direct-detail specialist.

The isolated Raven proof passed with AdamW at 1e-3 on the detail body and 3e-3
on the zero-initialised albedo output head, with no weight decay. V12.3 moved the
same candidate into production composition, but the canonical optimiser still
placed the entire detail network in one generic 1x group. The first production
integration run therefore reached +45.35% edge recovery instead of the +50.75%
capacity result and never exercised BenefitSelector.

This contract preserves every existing production optimiser group while splitting
``detail_net.albedo_head`` into a 3x group and disabling weight decay only for the
detail body/head, reproducing the regime that actually demonstrated capacity.
All installed callbacks are module-level for Windows spawn safety.
"""

from typing import Any

import torch


PARALLEL_DETAIL_OPTIMIZATION_REVISION = "V12.3.1"
DETAIL_ALBEDO_HEAD_LR_MULTIPLIER = 3.0
_INSTALLED = False
_ORIGINAL_BACKEND_SYNC: Any = None


def _build_optimizer_with_parallel_detail_head_boost(
    model: torch.nn.Module,
    config: Any,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, str]:
    """Build the canonical optimizer plus the proven 3x direct-albedo head group."""
    common = {
        "lr": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "betas": (float(config.optimizer_beta1), float(config.optimizer_beta2)),
    }
    parametric_parameters: list[torch.nn.Parameter] = []
    spline_parameters: list[torch.nn.Parameter] = []
    seam_parameters: list[torch.nn.Parameter] = []
    detail_body_parameters: list[torch.nn.Parameter] = []
    detail_albedo_head_parameters: list[torch.nn.Parameter] = []
    other_parameters: list[torch.nn.Parameter] = []

    for name, parameter in model.named_parameters():
        if "geometry_net.production_structure.spline_graph" in name:
            spline_parameters.append(parameter)
        elif "geometry_net.parametric_primitive_field" in name:
            parametric_parameters.append(parameter)
        elif "seam_restorer" in name:
            seam_parameters.append(parameter)
        elif name.startswith("detail_net.albedo_head."):
            detail_albedo_head_parameters.append(parameter)
        elif name.startswith("detail_net."):
            detail_body_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    if not detail_albedo_head_parameters:
        raise RuntimeError("parallel detail optimizer found no detail_net.albedo_head parameters")
    if not detail_body_parameters:
        raise RuntimeError("parallel detail optimizer found no detail_net body parameters")

    parameter_groups = [
        {"params": other_parameters, "lr_scale": 1.0},
        # Direct Residual Capacity used weight_decay=0 for the entire detail
        # specialist. Preserve that proven local regime without changing the rest
        # of the production supernet.
        {"params": detail_body_parameters, "lr_scale": 1.0, "weight_decay": 0.0},
        {
            "params": detail_albedo_head_parameters,
            "lr_scale": DETAIL_ALBEDO_HEAD_LR_MULTIPLIER,
            "weight_decay": 0.0,
        },
        {
            "params": parametric_parameters,
            "lr_scale": float(getattr(config, "parametric_primitive_lr_multiplier", 8.0)),
        },
        {
            "params": spline_parameters,
            "lr_scale": float(getattr(config, "spline_graph_lr_multiplier", 4.0)),
        },
        {
            "params": seam_parameters,
            "lr_scale": float(getattr(config, "seam_lr_multiplier", 2.0)),
        },
    ]

    optimizer_class = torch.optim.AdamW if config.optimizer_name == "adamw" else torch.optim.Adam
    display = "AdamW" if config.optimizer_name == "adamw" else "Adam"
    if device.type == "cuda" and bool(config.fused_optimizer):
        try:
            return optimizer_class(parameter_groups, fused=True, **common), f"fused {display}"
        except (TypeError, RuntimeError):
            pass
    if device.type == "cuda":
        try:
            return optimizer_class(parameter_groups, foreach=True, **common), f"foreach {display}"
        except TypeError:
            pass
    return optimizer_class(parameter_groups, **common), f"standard {display}"


def _synchronize_with_parallel_detail_optimizer(self: Any, training: Any) -> None:
    """Run the existing backend synchronization, then install optimizer parity."""
    if _ORIGINAL_BACKEND_SYNC is None:
        raise RuntimeError("parallel detail optimizer installed without backend synchronizer")
    _ORIGINAL_BACKEND_SYNC(self, training)
    service = getattr(training, "_training_service", None)
    if service is None:
        raise RuntimeError("NSAMDR training module has no TrainingService singleton")
    service._build_optimizer = _build_optimizer_with_parallel_detail_head_boost
    training._nsamdr_parallel_detail_optimizer_revision = PARALLEL_DETAIL_OPTIMIZATION_REVISION


def install_parallel_detail_optimization_contract() -> None:
    """Install production optimizer parity without changing checkpoint topology."""
    global _INSTALLED, _ORIGINAL_BACKEND_SYNC
    if _INSTALLED:
        return

    from .application.backend import TrainingBackend

    current = TrainingBackend._synchronize_training_service_contract
    if current is not _synchronize_with_parallel_detail_optimizer:
        _ORIGINAL_BACKEND_SYNC = current
        TrainingBackend._synchronize_training_service_contract = (
            _synchronize_with_parallel_detail_optimizer
        )
    _INSTALLED = True
