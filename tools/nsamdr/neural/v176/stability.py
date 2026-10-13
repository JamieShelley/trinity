"""Finite-magnitude guards for the V17.6 research specialist.

These limits stop a failed experiment. They do not clip model outputs,
change physical caps, or qualify a model for production.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

STABILITY_POLICY = {
    "revision": "V176_NUMERICAL_STABILITY_V1",
    "filterPrecision": "fp32",
    "waveletPrecision": "fp32",
    "cudnnTF32": False,
    "denominatorFloor": 0.001,
    "rawMaxAbs": 32.0,
    "highpassMaxAbs": 4.0,
    "lossMaxAbs": 100.0,
    "gradientMaxL2": 100.0,
    "parameterMaxAbs": 100.0,
    "optimizerMomentMaxAbs": 100.0,
    "gradientClipL2": 1.0,
    "physicalCapsChanged": False,
    "correctionClipped": False,
}


class NumericalStabilityError(RuntimeError):
    def __init__(self, name: str, value: torch.Tensor, limit: float | None):
        detached = value.detach()
        maximum = float(detached.abs().amax())
        self.diagnostic = {
            "tensor": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(torch.isfinite(detached).all()),
            "maxAbs": maximum if math.isfinite(maximum) else None,
            "limit": limit,
        }
        self.tensors = {name: detached.cpu().contiguous()}
        self.context = {}
        super().__init__(f"numerical stability guard: {name}; "
                         f"maxAbs={maximum}, limit={limit}")


def require_tensor(name: str, value: torch.Tensor, limit: float | None = None) -> None:
    maximum = value.detach().abs().amax()
    if not bool(torch.isfinite(maximum)) or (limit is not None and float(maximum) > limit):
        raise NumericalStabilityError(name, value, limit)


def _check_named(values, limit: float) -> float:
    values = list(values)
    if not values:
        return 0.0
    maxima = torch.stack([value.detach().abs().amax().float() for _, value in values])
    maximum = maxima.amax()
    if not bool(torch.isfinite(maximum)) or float(maximum) > limit:
        for name, value in values:
            require_tensor(name, value, limit)
    return float(maximum)


def tensor_statistics(value: torch.Tensor) -> dict[str, float]:
    # Squaring a large finite fp32 value must not overflow the diagnostic.
    v = value.detach().double()
    return {
        "meanAbs": float(v.abs().mean()),
        "rms": float(v.square().mean().sqrt()),
        "maxAbs": float(v.abs().amax()),
        "aboveOneLsbFraction": float((v.abs() >= (1 / 255)).double().mean()),
    }


def gradient_statistics(model) -> dict:
    norms = {}
    squared = 0.0
    count = 0
    for name, parameter in model.specialist.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().double().norm())
        squared += norm * norm
        count += 1
        if name in ("head.weight", "head.bias", "stem.weight", "stem.bias"):
            norms[name] = norm
    if any(parameter.grad is not None for parameter in model.backbone.parameters()):
        raise RuntimeError("frozen Stage 9 backbone received gradients")
    return {"totalL2": math.sqrt(squared), "tensorsWithGrad": count,
            "selectedParameterGradL2": norms}


def check_optimizer(model, optimizer) -> dict:
    parameter_max = _check_named(
        ((f"parameter/{name}", parameter)
         for name, parameter in model.specialist.named_parameters()),
        STABILITY_POLICY["parameterMaxAbs"],
    )
    moments = []
    for name, parameter in model.specialist.named_parameters():
        for key, value in optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor) and key != "step":
                moments.append((f"optimizer/{name}/{key}", value))
    moment_max = _check_named(moments, STABILITY_POLICY["optimizerMomentMaxAbs"])
    return {"maxAbsParameter": parameter_max, "maxAbsMoment": moment_max,
            "learningRates": [float(group["lr"]) for group in optimizer.param_groups]}


def guarded_train_step(model, optimizer, batch, autocast, *, collect: bool = False) -> dict:
    from .wavelet_supervision import detail_supervision_loss

    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = {}
    try:
        check_optimizer(model, optimizer)
        with autocast:
            outputs = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
            losses = detail_supervision_loss(outputs, batch)
        for name, value in losses.items():
            require_tensor(f"loss/{name}", value, STABILITY_POLICY["lossMaxAbs"])
        losses["total"].backward()
        gradients = gradient_statistics(model)
        require_tensor("gradient/totalL2", torch.tensor(gradients["totalL2"], dtype=torch.float64),
                       STABILITY_POLICY["gradientMaxL2"])
        before = ([p.detach().clone() for p in model.trainable_parameters()] if collect else [])
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(),
                                     STABILITY_POLICY["gradientClipL2"], error_if_nonfinite=True)
        optimizer.step()
        optimizer_stats = check_optimizer(model, optimizer)
        result = {"lossTerms": {key: float(value.detach()) for key, value in losses.items()}}
        if collect:
            result.update({
                "gradient": gradients,
                "rawDetail": tensor_statistics(outputs["fine_detail_raw"]),
                "highpassDetail": tensor_statistics(outputs["fine_detail_highpass"]),
                "appliedAlbedoChange": tensor_statistics(
                    outputs["candidate_albedo"] - outputs["base_candidate_albedo"]),
                "optimizer": optimizer_stats,
            })
            result["optimizer"]["parameterStepL2"] = math.sqrt(sum(
                float((parameter.detach().double() - old.double()).square().sum())
                for parameter, old in zip(model.trainable_parameters(), before)
            ))
        return result
    except NumericalStabilityError as exc:
        for name, value in batch.items():
            if isinstance(value, torch.Tensor):
                exc.tensors[f"batch/{name}"] = value.detach().cpu().contiguous()
        for name, value in outputs.items():
            if name in ("fine_detail_raw", "fine_detail_highpass", "candidate_albedo",
                        "base_candidate_albedo", "baseline_albedo"):
                exc.tensors[name] = value.detach().cpu().contiguous()
        for name, parameter in model.specialist.named_parameters():
            if parameter.grad is not None:
                exc.tensors[f"gradient/{name}"] = parameter.grad.detach().cpu().contiguous()
        optimizer.zero_grad(set_to_none=True)
        raise


def save_failure(path: Path, exc: NumericalStabilityError, model, optimizer, context: dict) -> None:
    """Save failed tensors and state as diagnostics, never as a checkpoint."""
    exc.context = dict(context)
    for name, parameter in model.specialist.named_parameters():
        if parameter.grad is not None:
            exc.tensors[f"gradient/{name}"] = parameter.grad.detach().cpu().contiguous()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "schema": "NSAMDR_V176_NUMERICAL_FAILURE_V1",
        "context": exc.context,
        "diagnostic": exc.diagnostic,
        "numericalStability": dict(STABILITY_POLICY),
        "tensors": exc.tensors,
        "detailState": model.specialist.state_dict(),
        "optimizerState": optimizer.state_dict(),
    }, temporary)
    temporary.replace(path)
