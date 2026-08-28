"""Fast fail-closed B1b production training for Raven Quick and Full.

EXP_0001 with the 54-epoch override exposed two separate problems:

* Quick had become a ~47 minute structural-training job.
* At epoch 44 the B1b training CE was only ~0.058 while held-out primitive
  accuracy was 72.4%. Repeating the same expensive full-render training/audit
  path was therefore memorisation, not useful qualification work.

This contract keeps the SAME production classifier/regressor, SAME state_dict,
SAME hard thresholds and SAME final production forward. It only removes
computation that has zero optimiser authority in an isolated B1b substage.

Classifier substage:
    exact production class branch only -> cross entropy
    no parameter encoder
    no 512px analytic SDF render

Parameter substage:
    exact production parameter branch only
    teacher-routed parameter loss + exact analytic render loss
    no classifier encoder

Integration:
    unchanged canonical full B1b loss.

Held-out structural validation:
    classifier/parameter substages first run a cheap exact branch audit.
    The expensive 29-case full production forward is run immediately when the
    cheap audit reaches the existing hard threshold, to confirm qualification.
    Integration always uses the full production audit.

No qualification gate is replaced by the cheap audit.
"""
from __future__ import annotations

import copy
from types import ModuleType
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F


def select_b1b_substage(
    classifier_qualified: bool,
    parameter_qualified: bool,
) -> str:
    if not bool(classifier_qualified):
        return "classifier"
    if not bool(parameter_qualified):
        return "parameters"
    return "integration"


def _classifier_logits(field: Any, input_lr: torch.Tensor) -> torch.Tensor:
    source_sdf_prior_lr = input_lr[:, -1:].float()
    geometry_evidence = field._geometry_evidence(input_lr, source_sdf_prior_lr)
    moments = field._geometry_moments(geometry_evidence)
    evidence = field._with_coordinates(geometry_evidence)
    feature = field.class_pool(field.class_encoder(evidence)).flatten(1)
    feature = field.class_trunk(torch.cat((feature, moments), dim=1))
    return field.class_head(feature).float()


def _parameter_bank(field: Any, input_lr: torch.Tensor) -> torch.Tensor:
    source_sdf_prior_lr = input_lr[:, -1:].float()
    geometry_evidence = field._geometry_evidence(input_lr, source_sdf_prior_lr)
    moments = field._geometry_moments(geometry_evidence)
    evidence = field._with_coordinates(geometry_evidence)
    feature = field.param_pool(field.param_encoder(evidence)).flatten(1)
    feature = field.param_trunk(torch.cat((feature, moments), dim=1))
    raw = field.param_head(feature).float().view(
        -1, int(field.class_head.out_features), 12
    )
    seed = field._seed_params_from_moments(moments)
    return field._apply_residual(raw, seed)


def _classifier_losses(
    model: Any,
    batch: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    field = model.geometry_net.parametric_primitive_field
    logits = _classifier_logits(field, batch["input"])
    target = batch["primitive_class"].long().reshape(-1)
    loss = F.cross_entropy(logits, target)
    zero = loss.detach() * 0.0
    return {
        "total": (
            loss
            * float(getattr(config, "parametric_primitive_class_weight", 8.0))
        ).float(),
        "sdf": zero,
        "primitive_class": loss,
        "primitive_param": zero,
        "primitive_param_mae": zero,
        "primitive_class_accuracy": (
            logits.argmax(dim=1) == target
        ).float().mean().detach(),
        "primitive_render": zero,
    }


def _parameter_losses(
    training_module: ModuleType,
    model: Any,
    batch: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    field = model.geometry_net.parametric_primitive_field
    input_lr = batch["input"]
    target_class = batch["primitive_class"].long().reshape(-1)
    params_by_class = _parameter_bank(field, input_lr)
    bi = torch.arange(params_by_class.shape[0], device=params_by_class.device)
    predicted_params = params_by_class[bi, target_class]
    target_params = batch["primitive_params"].float()
    param_mask = batch["primitive_param_mask"].float()

    param_error = training_module.parametric_param_abs_error_torch(
        predicted_params, target_params, target_class
    ) * param_mask
    param_loss = param_error.sum() / param_mask.sum().clamp_min(1.0)

    target_sdf_pixels = (
        batch["target_sdf"].float()
        * float(config.contour_sdf_max_distance_pixels)
    )
    predicted_sdf_pixels = training_module.render_parametric_sdf_torch(
        predicted_params,
        target_class,
        int(target_sdf_pixels.shape[-2]),
        int(target_sdf_pixels.shape[-1]),
    ).clamp(
        -float(config.contour_sdf_max_distance_pixels),
        float(config.contour_sdf_max_distance_pixels),
    )
    render_band = (target_sdf_pixels.abs() <= 8.0).float().detach()
    render_loss = (
        F.smooth_l1_loss(
            predicted_sdf_pixels,
            target_sdf_pixels.detach(),
            beta=0.12,
            reduction="none",
        )
        * (0.10 + 1.90 * render_band)
    ).mean()

    total = (
        param_loss
        * float(getattr(config, "parametric_primitive_param_weight", 48.0))
        + render_loss
        * float(getattr(config, "parametric_primitive_render_weight", 4.0))
    )
    zero = total.detach() * 0.0
    return {
        "total": total.float(),
        "sdf": zero,
        "primitive_class": zero,
        "primitive_param": param_loss,
        "primitive_param_mae": param_loss.detach(),
        "primitive_class_accuracy": zero,
        "primitive_render": render_loss,
    }


@torch.no_grad()
def _fast_classifier_audit(
    model: Any,
    loader: Any,
    device: torch.device,
) -> tuple[float, list[list[int]], list[float]]:
    was_training = bool(model.training)
    model.eval()
    field = model.geometry_net.parametric_primitive_field
    count = int(field.class_head.out_features)
    confusion = torch.zeros((count, count), dtype=torch.int64)
    for batch in loader:
        input_lr = batch["input"].to(device, non_blocking=device.type == "cuda")
        target = batch["primitive_class"].long().reshape(-1)
        target_device = target.to(device)
        logits = _classifier_logits(field, input_lr)
        predicted = logits.argmax(dim=1)
        for t, p in zip(target_device.detach().cpu().tolist(),
                        predicted.detach().cpu().tolist()):
            if 0 <= int(t) < count and 0 <= int(p) < count:
                confusion[int(t), int(p)] += 1
    total = int(confusion.sum().item())
    accuracy = (
        float(confusion.diag().sum().item()) / float(total)
        if total else 0.0
    )
    recall = []
    for cls in range(count):
        denom = int(confusion[cls].sum().item())
        recall.append(
            float(confusion[cls, cls].item()) / float(denom)
            if denom else 0.0
        )
    if was_training:
        model.train()
    return accuracy, confusion.tolist(), recall


@torch.no_grad()
def _fast_parameter_audit(
    training_module: ModuleType,
    model: Any,
    loader: Any,
    device: torch.device,
) -> float:
    was_training = bool(model.training)
    model.eval()
    field = model.geometry_net.parametric_primitive_field
    numer = 0.0
    denom = 0.0
    for batch in loader:
        input_lr = batch["input"].to(device, non_blocking=device.type == "cuda")
        target_class = batch["primitive_class"].long().reshape(-1).to(device)
        target_params = batch["primitive_params"].float().to(device)
        mask = batch["primitive_param_mask"].float().to(device)
        bank = _parameter_bank(field, input_lr)
        bi = torch.arange(bank.shape[0], device=device)
        predicted = bank[bi, target_class]
        error = training_module.parametric_param_abs_error_torch(
            predicted, target_params, target_class
        ) * mask
        numer += float(error.sum().item())
        denom += float(mask.sum().item())
    result = numer / max(denom, 1.0)
    if was_training:
        model.train()
    return result


def install_b1b_staged_contract(training_module: ModuleType) -> None:
    if bool(getattr(
        training_module, "_nsamdr_b1b_fast_staged_contract_installed", False
    )):
        return

    original_losses: Callable[..., dict[str, Any]] = (
        training_module._parametric_b1b_train_losses
    )
    original_validate: Callable[..., Any] = training_module._validate

    # Last full synthetic audit. During an isolated cheap audit, all unrelated
    # geometry telemetry remains the most recent authoritative full result.
    full_audit_cache: dict[str, Any] = {}

    def staged_losses(
        model: Any,
        batch: dict[str, Any],
        config: Any,
        *,
        substage: str = "integration",
    ) -> dict[str, Any]:
        model._nsamdr_b1b_substage = str(substage)
        if substage == "classifier":
            return _classifier_losses(model, batch, config)
        if substage == "parameters":
            return _parameter_losses(
                training_module, model, batch, config
            )
        if substage == "integration":
            return original_losses(
                model, batch, config, substage=substage
            )
        raise ValueError(f"unsupported B1b substage: {substage}")

    def staged_validate(
        model: Any,
        loader: Any,
        config: Any,
        device: torch.device,
        phase: str,
        **kwargs: Any,
    ):
        exact = bool(kwargs.get("exact_geometry_metrics", False))
        stage = str(getattr(model, "_nsamdr_b1b_substage", ""))

        if phase == "sdf-proof" and exact and stage == "classifier":
            accuracy, confusion, recall = _fast_classifier_audit(
                model, loader, device
            )
            required = float(
                getattr(config, "parametric_primitive_class_accuracy_required", 0.95)
            )
            training_module._status(
                f"  [FAST-B1b] classifier held-out audit "
                f"accuracy={accuracy*100.0:.1f}% required={required*100.0:.1f}%"
            )
            if accuracy >= required or "result" not in full_audit_cache:
                # Gate-ready (or no baseline cache): confirm with the full exact
                # production forward before qualification can change state.
                result = original_validate(
                    model, loader, config, device, phase, **kwargs
                )
                full_audit_cache["result"] = result
                return result

            metrics = copy.deepcopy(full_audit_cache["result"][0])
            metrics["primitive_class_accuracy"] = accuracy
            metrics["primitive_confusion_matrix"] = confusion
            metrics["primitive_per_class_accuracy"] = recall
            return (
                metrics,
                full_audit_cache["result"][1],
                full_audit_cache["result"][2],
            )

        if phase == "sdf-proof" and exact and stage == "parameters":
            mae = _fast_parameter_audit(
                training_module, model, loader, device
            )
            required = float(
                getattr(config, "parametric_primitive_param_mae_required", 0.04)
            )
            training_module._status(
                f"  [FAST-B1b] parameter held-out audit "
                f"teacherMAE={mae:.5f} required={required:.5f}"
            )
            if mae <= required or "result" not in full_audit_cache:
                result = original_validate(
                    model, loader, config, device, phase, **kwargs
                )
                full_audit_cache["result"] = result
                return result

            metrics = copy.deepcopy(full_audit_cache["result"][0])
            metrics["primitive_teacher_param_mae"] = mae
            metrics["primitive_param_mae"] = mae
            return (
                metrics,
                full_audit_cache["result"][1],
                full_audit_cache["result"][2],
            )

        # Integration and all non-B1b validations remain completely canonical.
        result = original_validate(
            model, loader, config, device, phase, **kwargs
        )
        if exact and phase in {"sdf-bootstrap", "sdf-proof"}:
            full_audit_cache["result"] = result
        return result

    training_module._parametric_b1b_substage = select_b1b_substage
    training_module._parametric_b1b_train_losses = staged_losses
    training_module._validate = staged_validate
    training_module._nsamdr_b1b_fast_staged_contract_installed = True
