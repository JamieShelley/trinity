#!/usr/bin/env python3
"""Full-capacity V16.2 broad-authority generalisation proof.

This diagnostic trains the production-size V16 Swin candidate with the proven
structure-conditioning branch on the broad authored EVE corpus. It is deliberately
candidate-only: the reduced control-vs-conditioned experiment already established a
held-out conditioning benefit, so this stage tests whether full local capacity and
broad-authority generalisation combine without doubling GPU cost.

Material recovery remains telemetry only until complete SOF/material semantics are
resolved. Checkpoints include model and optimizer state so expensive full-capacity
runs can resume at later cumulative stages.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v14.config import V16Config
from v14.qualification import sample_metrics
from v16.broad_prior import AUGMENTATION_POLICIES, AuthorityBalancedSRDataset
from v16.conditioning import StructureConditionedV16Candidate


SCHEMA = "NSAMDR_V16_FULL_BROAD_PROBE_V1"
CHECKPOINT_SCHEMA = "NSAMDR_V16_FULL_BROAD_CHECKPOINT_V1"
DEFAULT_MANIFEST = "artifacts/nsamdr/training_v16_authored_prior/dataset_manifest.json"
METRIC_KEYS = (
    "global_recovery",
    "edge_recovery",
    "gradient_recovery",
    "normal_recovery",
    "material_recovery",
    "lattice_cell_excess",
)
DISTRIBUTION_METRIC_KEYS = METRIC_KEYS + (
    "detail_recovery_1px",
    "detail_recovery_2px",
    "detail_recovery_4px",
    "lattice_candidate_fraction",
    "lattice_target_fraction",
    "protected_preservation",
)
LOSS_KEYS = (
    "reconstruction",
    "gradient",
    "normal",
    "residual_albedo",
    "residual_normal",
    "total",
)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _parse_stages(raw: str) -> list[int]:
    values: list[int] = []
    for token in str(raw).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError("all stages must be positive")
        values.append(value)
    result = sorted(set(values))
    if not result:
        raise ValueError("at least one stage is required")
    return result


def _gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gray = value.float().mean(dim=1, keepdim=True)
    dx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    return dx, dy


def _proof_loss_terms(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: V16Config,
) -> dict[str, torch.Tensor]:
    """Return the unchanged broad-proof objective as named components."""
    ca = outputs["candidate_albedo"].float()
    cn = outputs["candidate_normal"].float()
    ba = outputs["baseline_albedo"].detach().float()
    bn = outputs["baseline_normal"].detach().float()
    ta = batch["target_albedo"].float()
    tn = batch["target_normal"].float()

    reconstruction = F.l1_loss(ca, ta)
    cgx, cgy = _gradient(ca)
    tgx, tgy = _gradient(ta)
    gradient = (cgx - tgx).abs().mean() + (cgy - tgy).abs().mean()
    normal = F.l1_loss(cn, tn)

    target_a = (ta - ba).clamp(
        -config.albedo_residual_cap,
        config.albedo_residual_cap,
    )
    target_n = (tn - bn).clamp(
        -config.normal_residual_cap,
        config.normal_residual_cap,
    )
    residual_albedo = F.l1_loss(
        outputs["predicted_residual_albedo"].float(),
        target_a,
    )
    residual_normal = F.l1_loss(
        outputs["predicted_residual_normal"].float(),
        target_n,
    )
    total = (
        reconstruction
        + 0.50 * gradient
        + 0.35 * normal
        + residual_albedo
        + 0.25 * residual_normal
    )
    return {
        "reconstruction": reconstruction,
        "gradient": gradient,
        "normal": normal,
        "residual_albedo": residual_albedo,
        "residual_normal": residual_normal,
        "total": total,
    }


def _proof_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: V16Config,
) -> torch.Tensor:
    return _proof_loss_terms(outputs, batch, config)["total"]


def _unit_slope_bounded_residual(
    raw: torch.Tensor,
    cap: float,
) -> torch.Tensor:
    """Soft-bound residuals while keeping unit slope around zero."""
    limit = max(float(cap), 1.0e-8)
    return torch.tanh(raw.float() / limit) * limit


def _unit_slope_albedo_outputs(
    outputs: dict[str, torch.Tensor],
    config: V16Config,
) -> dict[str, torch.Tensor]:
    """Inference-only albedo ablation; production model behaviour is unchanged."""
    result = dict(outputs)
    residual = _unit_slope_bounded_residual(
        outputs["candidate_raw_residual_albedo"],
        config.albedo_residual_cap,
    )
    result["predicted_residual_albedo"] = residual
    result["candidate_albedo"] = (
        outputs["baseline_albedo"].float() + residual
    ).clamp(0.0, 1.0)
    return result


def _oracle_scalar_albedo_outputs(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    max_gain: float = 4.0,
) -> tuple[dict[str, torch.Tensor], float]:
    """Qualification-only least-squares scalar on the existing albedo residual."""
    baseline = outputs["baseline_albedo"].detach().float()
    candidate_residual = outputs["candidate_albedo"].detach().float() - baseline
    target_residual = batch["target_albedo"].detach().float() - baseline
    denominator = candidate_residual.square().sum()
    if float(denominator.cpu().item()) <= 1.0e-12:
        gain = 0.0
    else:
        gain = float(
            (candidate_residual * target_residual).sum().div(denominator).cpu().item()
        )
    gain = min(max(gain, 0.0), float(max_gain))
    result = dict(outputs)
    scaled = candidate_residual * gain
    result["predicted_residual_albedo"] = scaled
    result["candidate_albedo"] = (baseline + scaled).clamp(0.0, 1.0)
    return result, gain


def _phase_abs_energy(
    value: torch.Tensor,
    scale: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for oy in range(int(scale)):
        for ox in range(int(scale)):
            phase = value[..., oy::scale, ox::scale]
            result[f"phase_y{oy}_x{ox}"] = (
                float(phase.detach().float().abs().mean().cpu().item())
                if phase.numel()
                else 0.0
            )
    return result


def _phase_spread(values: dict[str, float]) -> float:
    if not values:
        return 0.0
    items = [float(value) for value in values.values()]
    mean = sum(items) / max(1, len(items))
    return (max(items) - min(items)) / max(mean, 1.0e-8)


def _residual_diagnostics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: V16Config,
) -> dict[str, float]:
    baseline = outputs["baseline_albedo"].detach().float()
    candidate = outputs["candidate_albedo"].detach().float()
    target = batch["target_albedo"].detach().float()
    predicted = outputs["predicted_residual_albedo"].detach().float()
    raw = outputs["candidate_raw_residual_albedo"].detach().float()

    candidate_residual = candidate - baseline
    target_residual = target - baseline
    target_magnitude = float(target_residual.abs().mean().cpu().item())
    candidate_magnitude = float(candidate_residual.abs().mean().cpu().item())
    cap = max(float(config.albedo_residual_cap), 1.0e-8)

    candidate_phase = _phase_abs_energy(candidate_residual, config.scale)
    target_phase = _phase_abs_energy(target_residual, config.scale)

    candidate_flat = candidate_residual.reshape(-1)
    target_flat = target_residual.reshape(-1)
    dot = (candidate_flat * target_flat).sum()
    candidate_norm = torch.linalg.vector_norm(candidate_flat)
    target_norm = torch.linalg.vector_norm(target_flat)
    if float(candidate_norm.cpu().item()) <= 1.0e-12 or float(target_norm.cpu().item()) <= 1.0e-12:
        cosine = 0.0
    else:
        cosine = float((dot / (candidate_norm * target_norm)).cpu().item())

    target_weight = target_residual.abs()
    target_weight_sum = target_weight.sum()
    if float(target_weight_sum.cpu().item()) <= 1.0e-12:
        weighted_sign_agreement = 0.0
    else:
        sign_match = (
            torch.sign(candidate_residual) == torch.sign(target_residual)
        ).float()
        weighted_sign_agreement = float(
            (sign_match * target_weight).sum().div(target_weight_sum).cpu().item()
        )

    candidate_energy = candidate_residual.square().sum()
    if float(candidate_energy.cpu().item()) <= 1.0e-12:
        least_squares_gain = 0.0
    else:
        least_squares_gain = float((dot / candidate_energy).cpu().item())

    result = {
        "raw_predicted_residual_magnitude": float(
            raw.abs().mean().cpu().item()
        ),
        "bounded_predicted_residual_magnitude": float(
            predicted.abs().mean().cpu().item()
        ),
        "applied_candidate_residual_magnitude": candidate_magnitude,
        "target_residual_magnitude": target_magnitude,
        "candidate_to_target_residual_ratio": (
            candidate_magnitude / max(target_magnitude, 1.0e-8)
        ),
        "residual_cap_saturation": float(
            (predicted.abs() >= (0.98 * cap)).float().mean().cpu().item()
        ),
        "target_exceeds_residual_cap": float(
            (target_residual.abs() >= cap).float().mean().cpu().item()
        ),
        "residual_cosine_similarity": cosine,
        "target_weighted_sign_agreement": weighted_sign_agreement,
        "least_squares_residual_gain": least_squares_gain,
        "candidate_phase_energy_spread": _phase_spread(candidate_phase),
        "target_phase_energy_spread": _phase_spread(target_phase),
    }
    for key, value in candidate_phase.items():
        result[f"candidate_{key}"] = float(value)
    for key, value in target_phase.items():
        result[f"target_{key}"] = float(value)
    return result


def _loss_values(terms: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: float(terms[key].detach().float().cpu().item())
        for key in LOSS_KEYS
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray(
        [float(value) for value in values if np.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p25": float(np.percentile(finite, 25)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "p75": float(np.percentile(finite, 75)),
        "max": float(np.max(finite)),
        "positiveFraction": float(np.mean(finite > 0.0)),
    }


def _dictionary_distributions(
    rows: list[dict[str, float]],
    keys: tuple[str, ...] | list[str],
) -> dict[str, dict[str, float | int]]:
    return {
        key: _distribution(
            [float(row[key]) for row in rows if key in row]
        )
        for key in keys
    }


def _per_authority(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["authorityId"]), []).append(row)

    result: list[dict[str, Any]] = []
    for authority in sorted(grouped):
        samples = grouped[authority]
        metric_rows = [dict(sample["metrics"]) for sample in samples]
        loss_rows = [dict(sample["lossTerms"]) for sample in samples]
        diagnostic_rows = [dict(sample["residualDiagnostics"]) for sample in samples]
        result.append(
            {
                "authorityId": authority,
                "sampleCount": len(samples),
                "cropIds": [str(sample["cropId"]) for sample in samples],
                "metrics": {
                    key: float(statistics.median(
                        float(row[key]) for row in metric_rows if key in row
                    ))
                    for key in DISTRIBUTION_METRIC_KEYS
                    if any(key in row for row in metric_rows)
                },
                "lossTerms": {
                    key: float(statistics.median(
                        float(row[key]) for row in loss_rows if key in row
                    ))
                    for key in LOSS_KEYS
                    if any(key in row for row in loss_rows)
                },
                "residualDiagnostics": {
                    key: float(statistics.median(
                        float(row[key]) for row in diagnostic_rows if key in row
                    ))
                    for key in sorted(
                        {
                            key
                            for row in diagnostic_rows
                            for key in row
                        }
                    )
                },
            }
        )
    return result



def _to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def _full_config(manifest: str, hr_size: int) -> V16Config:
    if hr_size % 32:
        raise ValueError("hr-size must be divisible by 32")
    config = V16Config(
        dataset_manifest=manifest,
        train_hr_size=hr_size,
        train_lr_size=hr_size // 4,
        validation_hr_size=hr_size,
        validation_lr_size=hr_size // 4,
        minimum_heldout_samples=4,
        tiles_per_epoch=1,
        validation_tiles=1,
        clean_epochs=1,
        robust_epochs=1,
        selector_epochs=1,
    )
    config.validate()
    return config


def _optimizer(
    model: StructureConditionedV16Candidate,
    config: V16Config,
) -> torch.optim.Optimizer:
    model.set_candidate_training()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("full broad proof has no trainable candidate parameters")
    return torch.optim.Adam(
        parameters,
        lr=config.sr_learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        foreach=False,
    )


def _visited_authorities(
    dataset: AuthorityBalancedSRDataset,
    start_step: int,
    end_step: int,
) -> int:
    selected = dataset.record_indices[start_step:end_step]
    values = {
        str(
            dataset.records[index].get("family_id")
            or dataset.records[index].get("familyId")
            or ""
        )
        for index in selected
    }
    values.discard("")
    return len(values)


def _vram(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"peakAllocatedGiB": 0.0, "peakReservedGiB": 0.0}
    gib = float(1024**3)
    return {
        "peakAllocatedGiB": float(torch.cuda.max_memory_allocated(device) / gib),
        "peakReservedGiB": float(torch.cuda.max_memory_reserved(device) / gib),
    }


def _train_segment(
    model: StructureConditionedV16Candidate,
    optimizer: torch.optim.Optimizer,
    manifest: dict[str, Any],
    config: V16Config,
    *,
    start_step: int,
    end_step: int,
    device: torch.device,
    seed: int,
    precision: str,
    augmentation_policy: str,
) -> dict[str, Any]:
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step")

    dataset = AuthorityBalancedSRDataset(
        manifest,
        config,
        "train",
        end_step,
        seed=seed,
        degradation="clean",
        augmentation_policy=augmentation_policy,
    )
    loader = DataLoader(
        Subset(dataset, range(start_step, end_step)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    loss_history: dict[str, list[float]] = {key: [] for key in LOSS_KEYS}
    telemetry: list[dict[str, Any]] = []
    started = time.monotonic()
    segment_steps = end_step - start_step

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for offset, batch in enumerate(loader, start=1):
        global_step = start_step + offset
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, precision):
            outputs = model(
                batch["lr_albedo"],
                batch["lr_normal"],
                batch["lr_material"],
            )
            terms = _proof_loss_terms(outputs, batch, config)
            loss = terms["total"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()

        values = _loss_values(terms)
        for key, value in values.items():
            loss_history[key].append(value)

        should_log = offset == 1 or global_step % 32 == 0 or global_step == end_step
        if should_log:
            diagnostics = _residual_diagnostics(outputs, batch, config)
            telemetry.append(
                {
                    "step": int(global_step),
                    "lossTerms": values,
                    "residualDiagnostics": diagnostics,
                }
            )
            elapsed = max(time.monotonic() - started, 1.0e-6)
            rate = offset / elapsed
            eta = (segment_steps - offset) / max(rate, 1.0e-6)
            print(
                f"[full-broad] step {global_step:4d}/{end_step} "
                f"loss={values['total']:.6f} "
                f"rec={values['reconstruction']:.6f} "
                f"grad={values['gradient']:.6f} "
                f"normal={values['normal']:.6f} "
                f"resA={values['residual_albedo']:.6f} "
                f"ratio={diagnostics['candidate_to_target_residual_ratio']:.3f} "
                f"cap={diagnostics['residual_cap_saturation']*100:.1f}% "
                f"elapsed={elapsed/60.0:.1f}m "
                f"segment-eta={eta/60.0:.1f}m",
                flush=True,
            )

    tail = min(32, len(loss_history["total"]))
    result: dict[str, Any] = {
        "fromStep": int(start_step),
        "toStep": int(end_step),
        "segmentSteps": int(segment_steps),
        "firstLoss": loss_history["total"][0],
        "finalLoss": loss_history["total"][-1],
        "medianTailLoss": float(
            statistics.median(loss_history["total"][-tail:])
        ),
        "medianTailLossTerms": {
            key: float(statistics.median(values[-tail:]))
            for key, values in loss_history.items()
        },
        "telemetry": telemetry,
        "elapsedSeconds": time.monotonic() - started,
        "availableTrainAuthorityCount": int(dataset.authority_count),
        "visitedAuthorityCount": _visited_authorities(dataset, start_step, end_step),
        "augmentationPolicy": str(augmentation_policy),
    }
    result.update(_vram(device))
    return result




def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value)
    )
    cleaned = cleaned.strip("_")
    return cleaned[:80] or "unknown"


def _rgb_u8(value: torch.Tensor) -> np.ndarray:
    image = (
        value.detach()
        .float()
        .clamp(0.0, 1.0)[0, :3]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return np.round(image * 255.0).astype(np.uint8)


def _normal_rgb_u8(value: torch.Tensor) -> np.ndarray:
    normal = value.detach().float()[0]
    x = normal[0].clamp(-1.0, 1.0)
    y = normal[1].clamp(-1.0, 1.0)
    z = torch.sqrt((1.0 - x.square() - y.square()).clamp_min(0.0))
    image = torch.stack((x, y, z), dim=-1)
    image = ((image + 1.0) * 0.5).clamp(0.0, 1.0).cpu().numpy()
    return np.round(image * 255.0).astype(np.uint8)


def _edge_u8(value: torch.Tensor) -> np.ndarray:
    gray = value.detach().float().mean(dim=1, keepdim=True)
    dx = F.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    edge = dx + dy
    edge = edge / edge.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)
    plane = edge[0, 0].cpu().numpy()
    return np.round(np.clip(plane, 0.0, 1.0) * 255.0).astype(np.uint8)


def _label_panel(panel: np.ndarray, label: str) -> np.ndarray:
    if panel.ndim == 2:
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2RGB)
    result = np.ascontiguousarray(panel.copy())
    cv2.rectangle(result, (0, 0), (result.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(
        result,
        label,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def _write_panel_row(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
) -> str:
    rendered = [_label_panel(panel, label) for label, panel in panels]
    canvas = np.concatenate(rendered, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write preview image: {path}")
    return str(path.resolve())


def _write_preview_sample(
    preview_root: Path,
    *,
    sample_index: int,
    record: dict[str, Any],
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    config: V16Config,
) -> dict[str, Any]:
    authority = str(record.get("family_id") or record.get("familyId") or "unknown")
    crop = str(record.get("crop_id") or record.get("cropId") or sample_index)
    sample_dir = preview_root / (
        f"sample_{sample_index:02d}_{_safe_name(authority)}_{_safe_name(crop)}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)

    target = batch["target_albedo"].float()
    baseline = outputs["baseline_albedo"].float()
    candidate = outputs["candidate_albedo"].float()
    unit_slope_outputs = _unit_slope_albedo_outputs(outputs, config)
    unit_slope_candidate = unit_slope_outputs["candidate_albedo"].float()
    lr = batch["lr_albedo"].float()

    target_rgb = _rgb_u8(target)
    baseline_rgb = _rgb_u8(baseline)
    candidate_rgb = _rgb_u8(candidate)
    unit_slope_rgb = _rgb_u8(unit_slope_candidate)
    lr_rgb = _rgb_u8(lr)
    lr_rgb = cv2.resize(
        lr_rgb,
        (target_rgb.shape[1], target_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    residual = (candidate - baseline).mean(dim=1, keepdim=True)
    residual_scale = max(float(config.albedo_residual_cap), 1.0e-8)
    residual_view = (
        0.5 + 0.5 * (residual / residual_scale)
    ).clamp(0.0, 1.0)
    residual_u8 = np.round(
        residual_view[0, 0].detach().cpu().numpy() * 255.0
    ).astype(np.uint8)

    baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    candidate_error = (candidate - target).abs().mean(dim=1, keepdim=True)
    unit_slope_error = (
        unit_slope_candidate - target
    ).abs().mean(dim=1, keepdim=True)
    error_max = torch.maximum(
        torch.maximum(
            baseline_error.amax(),
            candidate_error.amax(),
        ),
        unit_slope_error.amax(),
    ).clamp_min(1.0e-8)
    baseline_error_u8 = np.round(
        (baseline_error / error_max)[0, 0].detach().cpu().numpy() * 255.0
    ).astype(np.uint8)
    candidate_error_u8 = np.round(
        (candidate_error / error_max)[0, 0].detach().cpu().numpy() * 255.0
    ).astype(np.uint8)
    unit_slope_error_u8 = np.round(
        (unit_slope_error / error_max)[0, 0].detach().cpu().numpy() * 255.0
    ).astype(np.uint8)

    unit_residual = (
        unit_slope_candidate - baseline
    ).mean(dim=1, keepdim=True)
    unit_residual_view = (
        0.5 + 0.5 * (unit_residual / residual_scale)
    ).clamp(0.0, 1.0)
    unit_residual_u8 = np.round(
        unit_residual_view[0, 0].detach().cpu().numpy() * 255.0
    ).astype(np.uint8)

    files = {
        "albedoComparison": _write_panel_row(
            sample_dir / "albedo_comparison.png",
            [
                ("LR INPUT x4 NEAREST", lr_rgb),
                ("B BASELINE", baseline_rgb),
                ("C CANDIDATE", candidate_rgb),
                ("A AUTHORED HR", target_rgb),
            ],
        ),
        "albedoDiagnostics": _write_panel_row(
            sample_dir / "albedo_diagnostics.png",
            [
                ("C-B SIGNED", residual_u8),
                ("|A-B|", baseline_error_u8),
                ("|A-C|", candidate_error_u8),
            ],
        ),
        "edgeComparison": _write_panel_row(
            sample_dir / "edge_comparison.png",
            [
                ("A EDGE", _edge_u8(target)),
                ("B EDGE", _edge_u8(baseline)),
                ("C EDGE", _edge_u8(candidate)),
            ],
        ),
        "normalComparison": _write_panel_row(
            sample_dir / "normal_comparison.png",
            [
                ("B NORMAL", _normal_rgb_u8(outputs["baseline_normal"])),
                ("C NORMAL", _normal_rgb_u8(outputs["candidate_normal"])),
                ("A NORMAL", _normal_rgb_u8(batch["target_normal"])),
            ],
        ),
        "unitSlopeAlbedoComparison": _write_panel_row(
            sample_dir / "unit_slope_albedo_comparison.png",
            [
                ("B BASELINE", baseline_rgb),
                ("C CURRENT", candidate_rgb),
                ("C UNIT-SLOPE", unit_slope_rgb),
                ("A AUTHORED HR", target_rgb),
            ],
        ),
        "unitSlopeAlbedoDiagnostics": _write_panel_row(
            sample_dir / "unit_slope_albedo_diagnostics.png",
            [
                ("CURRENT C-B", residual_u8),
                ("UNIT C-B", unit_residual_u8),
                ("|A-C CURRENT|", candidate_error_u8),
                ("|A-C UNIT|", unit_slope_error_u8),
            ],
        ),
        "unitSlopeEdgeComparison": _write_panel_row(
            sample_dir / "unit_slope_edge_comparison.png",
            [
                ("A EDGE", _edge_u8(target)),
                ("B EDGE", _edge_u8(baseline)),
                ("C CURRENT EDGE", _edge_u8(candidate)),
                ("C UNIT EDGE", _edge_u8(unit_slope_candidate)),
            ],
        ),
    }
    metadata = {
        "authorityId": authority,
        "cropId": crop,
        "recordPath": str(record.get("path") or ""),
        "sampleIndex": int(sample_index),
        "metrics": sample_metrics(outputs, batch, final=False),
        "residualDiagnostics": _residual_diagnostics(
            outputs,
            batch,
            config,
        ),
        "unitSlopeResidualAblation": {
            "metrics": sample_metrics(
                unit_slope_outputs,
                batch,
                final=False,
            ),
            "residualDiagnostics": _residual_diagnostics(
                unit_slope_outputs,
                batch,
                config,
            ),
        },
        "files": files,
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def _evaluate(
    model: StructureConditionedV16Candidate,
    manifest: dict[str, Any],
    config: V16Config,
    *,
    split: str,
    samples: int,
    device: torch.device,
    seed: int,
    precision: str,
    preview_root: Path | None = None,
    preview_samples: int = 0,
    augmentation_policy: str = "legacy-random",
) -> dict[str, Any]:
    dataset = AuthorityBalancedSRDataset(
        manifest,
        config,
        split,
        samples,
        seed=seed,
        degradation="clean",
        augmentation_policy=augmentation_policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    rows: list[dict[str, Any]] = []
    unit_slope_rows: list[dict[str, Any]] = []
    oracle_scalar_rows: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for sample_index, batch in enumerate(loader):
            record_index = int(batch["record_index"][0].item())
            record = dataset.records[record_index]
            authority = str(
                record.get("family_id")
                or record.get("familyId")
                or "unknown"
            )
            crop = str(
                record.get("crop_id")
                or record.get("cropId")
                or sample_index
            )
            batch = _to_device(batch, device)
            with _autocast(device, precision):
                outputs = model(
                    batch["lr_albedo"],
                    batch["lr_normal"],
                    batch["lr_material"],
                )
                loss_terms = _proof_loss_terms(outputs, batch, config)
            metrics = sample_metrics(outputs, batch, final=False)
            diagnostics = _residual_diagnostics(outputs, batch, config)
            rows.append(
                {
                    "authorityId": authority,
                    "cropId": crop,
                    "metrics": metrics,
                    "lossTerms": _loss_values(loss_terms),
                    "residualDiagnostics": diagnostics,
                }
            )

            unit_slope_outputs = _unit_slope_albedo_outputs(outputs, config)
            unit_slope_rows.append(
                {
                    "authorityId": authority,
                    "cropId": crop,
                    "metrics": sample_metrics(
                        unit_slope_outputs,
                        batch,
                        final=False,
                    ),
                    "residualDiagnostics": _residual_diagnostics(
                        unit_slope_outputs,
                        batch,
                        config,
                    ),
                }
            )

            oracle_outputs, oracle_gain = _oracle_scalar_albedo_outputs(
                outputs,
                batch,
            )
            oracle_scalar_rows.append(
                {
                    "authorityId": authority,
                    "cropId": crop,
                    "gain": oracle_gain,
                    "metrics": sample_metrics(
                        oracle_outputs,
                        batch,
                        final=False,
                    ),
                    "residualDiagnostics": _residual_diagnostics(
                        oracle_outputs,
                        batch,
                        config,
                    ),
                }
            )
            if preview_root is not None and sample_index < max(0, int(preview_samples)):
                previews.append(
                    _write_preview_sample(
                        preview_root,
                        sample_index=sample_index,
                        record=record,
                        batch=batch,
                        outputs=outputs,
                        config=config,
                    )
                )

    metric_rows = [dict(row["metrics"]) for row in rows]
    loss_rows = [dict(row["lossTerms"]) for row in rows]
    diagnostic_rows = [dict(row["residualDiagnostics"]) for row in rows]
    diagnostic_keys = sorted(
        {
            key
            for row in diagnostic_rows
            for key in row
        }
    )

    summary: dict[str, Any] = {
        "split": split,
        "sampleCount": len(rows),
        "authorityCount": dataset.selected_authority_count(),
        "availableAuthorityCount": dataset.authority_count,
        "materialMetricQualified": False,
        "previewArtifacts": previews,
        "perAuthority": _per_authority(rows),
        "metricDistributions": _dictionary_distributions(
            metric_rows,
            DISTRIBUTION_METRIC_KEYS,
        ),
        "lossDistributions": _dictionary_distributions(
            loss_rows,
            LOSS_KEYS,
        ),
        "residualDiagnosticDistributions": _dictionary_distributions(
            diagnostic_rows,
            diagnostic_keys,
        ),
    }
    if split == "validation":
        summary["heldOutAuthorityCount"] = dataset.selected_authority_count()
        summary["availableHeldOutAuthorityCount"] = dataset.authority_count

    for key in METRIC_KEYS:
        summary[f"median_{key}"] = float(
            statistics.median(float(item[key]) for item in metric_rows)
        )

    unit_metric_rows = [
        dict(row["metrics"]) for row in unit_slope_rows
    ]
    unit_diagnostic_rows = [
        dict(row["residualDiagnostics"]) for row in unit_slope_rows
    ]
    unit_diagnostic_keys = sorted(
        {
            key
            for row in unit_diagnostic_rows
            for key in row
        }
    )
    unit_summary: dict[str, Any] = {
        "kind": "inference-only-unit-slope-albedo-residual-bound",
        "formula": "cap*tanh(raw/cap)",
        "productionFormula": "cap*tanh(raw)",
        "sampleCount": len(unit_slope_rows),
        "metricDistributions": _dictionary_distributions(
            unit_metric_rows,
            DISTRIBUTION_METRIC_KEYS,
        ),
        "residualDiagnosticDistributions": _dictionary_distributions(
            unit_diagnostic_rows,
            unit_diagnostic_keys,
        ),
        "perAuthority": unit_slope_rows,
    }
    for key in METRIC_KEYS:
        unit_summary[f"median_{key}"] = float(
            statistics.median(
                float(item[key]) for item in unit_metric_rows
            )
        )
    unit_summary["deltaVsCurrentMedian"] = {
        key: float(
            unit_summary[f"median_{key}"]
            - summary[f"median_{key}"]
        )
        for key in METRIC_KEYS
    }
    summary["unitSlopeResidualAblation"] = unit_summary

    oracle_metric_rows = [
        dict(row["metrics"]) for row in oracle_scalar_rows
    ]
    oracle_summary: dict[str, Any] = {
        "kind": "qualification-only-per-sample-oracle-scalar-gain",
        "gainClamp": [0.0, 4.0],
        "sampleCount": len(oracle_scalar_rows),
        "gainDistribution": _distribution(
            [float(row["gain"]) for row in oracle_scalar_rows]
        ),
        "metricDistributions": _dictionary_distributions(
            oracle_metric_rows,
            DISTRIBUTION_METRIC_KEYS,
        ),
        "perAuthority": oracle_scalar_rows,
    }
    for key in METRIC_KEYS:
        oracle_summary[f"median_{key}"] = float(
            statistics.median(
                float(item[key]) for item in oracle_metric_rows
            )
        )
    oracle_summary["deltaVsCurrentMedian"] = {
        key: float(
            oracle_summary[f"median_{key}"]
            - summary[f"median_{key}"]
        )
        for key in METRIC_KEYS
    }
    summary["oracleScalarGainAblation"] = oracle_summary
    return summary



def _write_unit_slope_summary(
    preview_root: Path,
    *,
    validation: dict[str, Any],
    train_validation: dict[str, Any],
) -> dict[str, str]:
    held = dict(validation["unitSlopeResidualAblation"])
    seen = dict(train_validation["unitSlopeResidualAblation"])
    payload = {
        "schema": "NSAMDR_V16_UNIT_SLOPE_RESIDUAL_ABLATION_V1",
        "formula": held["formula"],
        "productionFormula": held["productionFormula"],
        "seen": {
            "current": {
                key: train_validation[f"median_{key}"]
                for key in METRIC_KEYS
            },
            "ablation": {
                key: seen[f"median_{key}"]
                for key in METRIC_KEYS
            },
            "delta": seen["deltaVsCurrentMedian"],
        },
        "heldOut": {
            "current": {
                key: validation[f"median_{key}"]
                for key in METRIC_KEYS
            },
            "ablation": {
                key: held[f"median_{key}"]
                for key in METRIC_KEYS
            },
            "delta": held["deltaVsCurrentMedian"],
        },
    }
    preview_root.mkdir(parents=True, exist_ok=True)
    json_path = preview_root / "unit_slope_ablation_summary.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    text_path = preview_root / "unit_slope_ablation_summary.txt"
    text_path.write_text(
        "\n".join(
            [
                "NSAMDR V16 UNIT-SLOPE RESIDUAL-BOUND ABLATION",
                f"formula: {held['formula']}",
                f"production: {held['productionFormula']}",
                "",
                "SEEN",
                (
                    "current  "
                    f"global={train_validation['median_global_recovery']*100:+.2f}% "
                    f"edge={train_validation['median_edge_recovery']*100:+.2f}% "
                    f"grad={train_validation['median_gradient_recovery']*100:+.2f}% "
                    f"lattice={train_validation['median_lattice_cell_excess']*100:+.2f}%"
                ),
                (
                    "ablation "
                    f"global={seen['median_global_recovery']*100:+.2f}% "
                    f"edge={seen['median_edge_recovery']*100:+.2f}% "
                    f"grad={seen['median_gradient_recovery']*100:+.2f}% "
                    f"lattice={seen['median_lattice_cell_excess']*100:+.2f}%"
                ),
                "",
                "HELD-OUT",
                (
                    "current  "
                    f"global={validation['median_global_recovery']*100:+.2f}% "
                    f"edge={validation['median_edge_recovery']*100:+.2f}% "
                    f"grad={validation['median_gradient_recovery']*100:+.2f}% "
                    f"lattice={validation['median_lattice_cell_excess']*100:+.2f}%"
                ),
                (
                    "ablation "
                    f"global={held['median_global_recovery']*100:+.2f}% "
                    f"edge={held['median_edge_recovery']*100:+.2f}% "
                    f"grad={held['median_gradient_recovery']*100:+.2f}% "
                    f"lattice={held['median_lattice_cell_excess']*100:+.2f}%"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "json": str(json_path.resolve()),
        "text": str(text_path.resolve()),
    }


def _write_residual_alignment_summary(
    preview_root: Path,
    *,
    validation: dict[str, Any],
    train_validation: dict[str, Any],
) -> dict[str, str]:
    def split_payload(summary: dict[str, Any]) -> dict[str, Any]:
        diagnostics = dict(summary["residualDiagnosticDistributions"])
        oracle = dict(summary["oracleScalarGainAblation"])
        return {
            "alignment": {
                "residualCosineMedian": diagnostics[
                    "residual_cosine_similarity"
                ]["median"],
                "targetWeightedSignAgreementMedian": diagnostics[
                    "target_weighted_sign_agreement"
                ]["median"],
                "leastSquaresResidualGainMedian": diagnostics[
                    "least_squares_residual_gain"
                ]["median"],
            },
            "current": {
                key: summary[f"median_{key}"]
                for key in METRIC_KEYS
            },
            "oracleScalar": {
                key: oracle[f"median_{key}"]
                for key in METRIC_KEYS
            },
            "oracleDelta": oracle["deltaVsCurrentMedian"],
            "oracleGainDistribution": oracle["gainDistribution"],
        }

    payload = {
        "schema": "NSAMDR_V16_RESIDUAL_ALIGNMENT_V1",
        "description": (
            "Qualification-only residual direction/support telemetry and "
            "per-sample least-squares scalar oracle; no model weights changed."
        ),
        "seen": split_payload(train_validation),
        "heldOut": split_payload(validation),
    }
    preview_root.mkdir(parents=True, exist_ok=True)
    json_path = preview_root / "residual_alignment_summary.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    seen = payload["seen"]
    held = payload["heldOut"]
    text_path = preview_root / "residual_alignment_summary.txt"
    text_path.write_text(
        "\n".join(
            [
                "NSAMDR V16 RESIDUAL ALIGNMENT / ORACLE SCALAR DIAGNOSTIC",
                "qualification-only; no model weights changed",
                "",
                "SEEN",
                (
                    "alignment "
                    f"cos={seen['alignment']['residualCosineMedian']:+.4f} "
                    f"sign={seen['alignment']['targetWeightedSignAgreementMedian']*100:.2f}% "
                    f"ls-gain={seen['alignment']['leastSquaresResidualGainMedian']:.3f}"
                ),
                (
                    "current   "
                    f"global={seen['current']['global_recovery']*100:+.2f}% "
                    f"edge={seen['current']['edge_recovery']*100:+.2f}% "
                    f"grad={seen['current']['gradient_recovery']*100:+.2f}%"
                ),
                (
                    "oracle    "
                    f"global={seen['oracleScalar']['global_recovery']*100:+.2f}% "
                    f"edge={seen['oracleScalar']['edge_recovery']*100:+.2f}% "
                    f"grad={seen['oracleScalar']['gradient_recovery']*100:+.2f}%"
                ),
                "",
                "HELD-OUT",
                (
                    "alignment "
                    f"cos={held['alignment']['residualCosineMedian']:+.4f} "
                    f"sign={held['alignment']['targetWeightedSignAgreementMedian']*100:.2f}% "
                    f"ls-gain={held['alignment']['leastSquaresResidualGainMedian']:.3f}"
                ),
                (
                    "current   "
                    f"global={held['current']['global_recovery']*100:+.2f}% "
                    f"edge={held['current']['edge_recovery']*100:+.2f}% "
                    f"grad={held['current']['gradient_recovery']*100:+.2f}%"
                ),
                (
                    "oracle    "
                    f"global={held['oracleScalar']['global_recovery']*100:+.2f}% "
                    f"edge={held['oracleScalar']['edge_recovery']*100:+.2f}% "
                    f"grad={held['oracleScalar']['gradient_recovery']*100:+.2f}%"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "json": str(json_path.resolve()),
        "text": str(text_path.resolve()),
    }


def _gate_status(metrics: dict[str, Any], config: V16Config) -> dict[str, Any]:
    checks = {
        "global": float(metrics["median_global_recovery"])
        >= config.candidate_global_recovery_required,
        "edge": float(metrics["median_edge_recovery"])
        >= config.candidate_edge_recovery_required,
        "gradient": float(metrics["median_gradient_recovery"])
        >= config.candidate_gradient_recovery_required,
        "lattice": float(metrics["median_lattice_cell_excess"])
        <= config.candidate_lattice_cell_excess_max,
    }
    return {
        "checks": checks,
        "albedoNormalNumericalGatesPass": bool(all(checks.values())),
        "fullyPromotable": False,
        "reason": "material-semantics-unresolved-and-selector-not-qualified",
    }


def _save_checkpoint(
    path: Path,
    *,
    model: StructureConditionedV16Candidate,
    optimizer: torch.optim.Optimizer,
    config: V16Config,
    step: int,
    seed: int,
    manifest_path: Path,
    curve: list[dict[str, Any]],
    augmentation_policy: str,
    initialization: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "step": int(step),
            "seed": int(seed),
            "manifest": str(manifest_path),
            "config": config.to_dict(),
            "augmentationPolicy": str(augmentation_policy),
            "initialization": dict(initialization),
            "modelState": model.state_dict(),
            "optimizerState": optimizer.state_dict(),
            "curve": curve,
        },
        path,
    )


def _load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    manifest_path: Path,
) -> tuple[
    StructureConditionedV16Candidate,
    torch.optim.Optimizer,
    V16Config,
    int,
    int,
    list[dict[str, Any]],
]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"full broad checkpoint schema mismatch: {path}")
    if Path(str(payload.get("manifest", ""))).resolve() != manifest_path.resolve():
        raise RuntimeError("resume checkpoint belongs to a different corpus manifest")

    raw_config = dict(payload.get("config") or {})
    fields = V16Config.__dataclass_fields__
    config = V16Config(**{key: value for key, value in raw_config.items() if key in fields})
    config.validate()

    model = StructureConditionedV16Candidate(
        config,
        structure_channels=48,
        structure_blocks=4,
    ).to(device)
    model.load_state_dict(payload["modelState"], strict=True)
    optimizer = _optimizer(model, config)
    optimizer.load_state_dict(payload["optimizerState"])
    return (
        model,
        optimizer,
        config,
        int(payload["step"]),
        int(payload["seed"]),
        list(payload.get("curve") or []),
    )


def _checkpoint_training_metadata(path: Path) -> tuple[str, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"full broad checkpoint schema mismatch: {path}")
    return (
        str(payload.get("augmentationPolicy") or "legacy-random"),
        dict(payload.get("initialization") or {"kind": "legacy-checkpoint"}),
    )


def _run_directory(repo_root: Path) -> Path:
    root = repo_root / "artifacts/nsamdr/diagnostics/v16_full_broad"
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / f"probe_{stamp}"
    suffix = 1
    while path.exists():
        path = root / f"probe_{stamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def _write_report(
    run_dir: Path,
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    config: V16Config,
    device: torch.device,
    precision: str,
    requested_stages: list[int],
    curve: list[dict[str, Any]],
    checkpoint_path: Path,
    augmentation_policy: str,
    initialization: dict[str, Any],
) -> Path:
    final = curve[-1]
    report = {
        "schema": SCHEMA,
        "promotable": False,
        "manifest": str(manifest_path),
        "authoritySplit": manifest.get("authoritySplit"),
        "samplingPolicy": "authority-balanced-complete-cycle-before-repeat",
        "augmentationPolicy": str(augmentation_policy),
        "initialization": dict(initialization),
        "architecture": {
            "kind": "production-size-v16-plus-structure-conditioning",
            "hrSize": config.train_hr_size,
            "lrSize": config.train_lr_size,
            "lrContextChannels": config.lr_context_channels,
            "lrBlocks": config.lr_blocks,
            "hrChannels": config.hr_channels,
            "swinGroups": config.swin_groups,
            "swinBlocksPerGroup": config.swin_blocks_per_group,
            "swinDepth": config.swin_depth,
            "swinHeads": config.swin_num_heads,
            "swinWindow": config.swin_window_size,
            "mapTailBlocks": config.map_tail_blocks,
            "gradientCheckpointing": config.use_gradient_checkpointing,
            "structureChannels": 48,
            "structureBlocks": 4,
        },
        "device": str(device),
        "ampPrecision": precision,
        "requestedStages": requested_stages,
        "completedStages": [int(item["step"]) for item in curve],
        "materialQualification": "excluded-until-SOF-semantic-authority-is-resolved",
        "curve": curve,
        "final": final,
        "candidateGateStatus": _gate_status(dict(final["validation"]), config),
        "resumeCheckpoint": str(checkpoint_path),
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = [
        "NSAMDR V16 FULL-CAPACITY BROAD-AUTHORITY PROOF",
        "=" * 86,
        f"Architecture        : 96ch, 6x6 Swin, depth {config.swin_depth}, HR {config.train_hr_size}",
        f"Completed stages    : {report['completedStages']}",
        f"Augmentation        : {augmentation_policy}",
        f"Initialization      : {initialization.get('kind', 'unknown')}",
        f"Held-out authorities: {final['validation']['heldOutAuthorityCount']}",
        f"Seen diag authorities: {final.get('trainValidation', {}).get('authorityCount', 0)}",
        f"Seen global recovery: {final.get('trainValidation', {}).get('median_global_recovery', float('nan'))*100:+.2f}%",
        f"Seen edge recovery  : {final.get('trainValidation', {}).get('median_edge_recovery', float('nan'))*100:+.2f}%",
        f"Global recovery     : {final['validation']['median_global_recovery']*100:+.2f}%",
        f"Edge recovery       : {final['validation']['median_edge_recovery']*100:+.2f}%",
        f"Gradient recovery   : {final['validation']['median_gradient_recovery']*100:+.2f}%",
        f"Normal recovery     : {final['validation']['median_normal_recovery']*100:+.2f}%",
        f"Lattice excess      : {final['validation']['median_lattice_cell_excess']*100:+.2f}%",
        "Material            : telemetry only; SOF semantics unresolved",
        f"Resume checkpoint   : {checkpoint_path}",
        f"JSON                : {report_path}",
    ]
    (run_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary), flush=True)
    return report_path


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Broad prior manifest is missing: {manifest_path}\n"
            "Run: scripts\\build\\nsamdr.bat authored-prior-corpus"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = dict(manifest.get("counts") or {})
    validation_families = int(counts.get("validationFamilies") or 0)
    if validation_families < 4:
        raise RuntimeError("full broad proof requires at least four held-out authorities")

    device = _device(args.device)
    stages = _parse_stages(args.stages)
    validation_samples = (
        int(args.validation_samples)
        if int(args.validation_samples) > 0
        else validation_families
    )
    train_families = int(counts.get("trainFamilies") or 0)
    if train_families < 1:
        raise RuntimeError("full broad proof requires train authorities")
    train_validation_samples = (
        int(args.train_validation_samples)
        if int(args.train_validation_samples) > 0
        else min(validation_samples, train_families)
    )

    if args.resume and args.initialize_from:
        raise RuntimeError("--resume and --initialize-from are mutually exclusive")

    if args.resume:
        checkpoint_source = Path(args.resume)
        if not checkpoint_source.is_absolute():
            checkpoint_source = (repo_root / checkpoint_source).resolve()
        if not checkpoint_source.is_file():
            raise RuntimeError(f"resume checkpoint is missing: {checkpoint_source}")
        (
            model,
            optimizer,
            config,
            start_step,
            seed,
            curve,
        ) = _load_checkpoint(
            checkpoint_source,
            device=device,
            manifest_path=manifest_path,
        )
        checkpoint_augmentation_policy, initialization = _checkpoint_training_metadata(
            checkpoint_source
        )
        run_dir = checkpoint_source.parent
        if int(args.hr_size) != config.train_hr_size:
            raise RuntimeError(
                f"resume HR size is {config.train_hr_size}, not requested {args.hr_size}"
            )
        if str(args.augmentation_policy) != checkpoint_augmentation_policy:
            raise RuntimeError(
                "resume augmentation policy mismatch: "
                f"checkpoint={checkpoint_augmentation_policy} "
                f"requested={args.augmentation_policy}"
            )
        augmentation_policy = checkpoint_augmentation_policy
    elif args.initialize_from:
        initialization_source = Path(args.initialize_from)
        if not initialization_source.is_absolute():
            initialization_source = (repo_root / initialization_source).resolve()
        if not initialization_source.is_file():
            raise RuntimeError(
                f"initialization checkpoint is missing: {initialization_source}"
            )
        (
            model,
            _source_optimizer,
            config,
            source_step,
            seed,
            _source_curve,
        ) = _load_checkpoint(
            initialization_source,
            device=device,
            manifest_path=manifest_path,
        )
        source_augmentation_policy, _source_initialization = _checkpoint_training_metadata(
            initialization_source
        )
        if int(args.hr_size) != config.train_hr_size:
            raise RuntimeError(
                f"initialization HR size is {config.train_hr_size}, not requested {args.hr_size}"
            )
        optimizer = _optimizer(model, config)
        start_step = 0
        curve = []
        augmentation_policy = str(args.augmentation_policy)
        initialization = {
            "kind": "weights-only-from-full-broad-checkpoint",
            "sourceCheckpoint": str(initialization_source.resolve()),
            "sourceStep": int(source_step),
            "sourceAugmentationPolicy": source_augmentation_policy,
            "optimizerReset": True,
        }
        run_dir = _run_directory(repo_root)
    else:
        seed = int(args.seed)
        config = _full_config(
            str(manifest_path.relative_to(repo_root)),
            int(args.hr_size),
        )
        torch.manual_seed(seed)
        model = StructureConditionedV16Candidate(
            config,
            structure_channels=48,
            structure_blocks=4,
        ).to(device)
        optimizer = _optimizer(model, config)
        start_step = 0
        curve = []
        augmentation_policy = str(args.augmentation_policy)
        initialization = {"kind": "fresh-random"}
        run_dir = _run_directory(repo_root)

    remaining = [stage for stage in stages if stage > start_step]
    if not remaining and not args.preview_only:
        raise RuntimeError(
            f"no requested stage is greater than checkpoint step {start_step}"
        )

    print("NSAMDR V16 FULL-CAPACITY BROAD-AUTHORITY PROOF", flush=True)
    print(f"Manifest          : {manifest_path}", flush=True)
    print(f"Device / AMP      : {device} / {args.amp_precision}", flush=True)
    print(f"HR/LR             : {config.train_hr_size}/{config.train_lr_size}", flush=True)
    print(
        "Backbone          : "
        f"{config.hr_channels}ch, {config.swin_groups}x"
        f"{config.swin_blocks_per_group} Swin, depth {config.swin_depth}",
        flush=True,
    )
    print(f"Requested stages  : {stages}", flush=True)
    print(f"Starting step     : {start_step}", flush=True)
    print(f"Train authorities : {counts.get('trainFamilies', '?')}", flush=True)
    print(f"Held-out auth.    : {validation_families}", flush=True)
    print(f"Train diag samples: {train_validation_samples}", flush=True)
    print(f"Preview samples   : {args.preview_samples}", flush=True)
    print(f"Augmentation      : {augmentation_policy}", flush=True)
    print(f"Initialization    : {initialization.get('kind', 'unknown')}", flush=True)

    checkpoint_path = run_dir / "resume_checkpoint.pt"

    if args.preview_only:
        if not args.resume:
            raise RuntimeError("--preview-only requires --resume")
        preview_root = run_dir / "previews" / f"step_{start_step:06d}"
        validation = _evaluate(
            model,
            manifest,
            config,
            split="validation",
            samples=validation_samples,
            device=device,
            seed=seed + 7001,
            precision=args.amp_precision,
            preview_root=preview_root,
            preview_samples=int(args.preview_samples),
            augmentation_policy=augmentation_policy,
        )
        train_validation = _evaluate(
            model,
            manifest,
            config,
            split="train",
            samples=train_validation_samples,
            device=device,
            seed=seed + 8001,
            precision=args.amp_precision,
            augmentation_policy=augmentation_policy,
        )
        ablation_summary = _write_unit_slope_summary(
            preview_root,
            validation=validation,
            train_validation=train_validation,
        )
        alignment_summary = _write_residual_alignment_summary(
            preview_root,
            validation=validation,
            train_validation=train_validation,
        )
        preview_report = {
            "schema": "NSAMDR_V16_FULL_BROAD_PREVIEW_V1",
            "step": int(start_step),
            "validation": validation,
            "trainValidation": train_validation,
            "previewRoot": str(preview_root.resolve()),
            "unitSlopeAblationSummary": ablation_summary,
            "residualAlignmentSummary": alignment_summary,
        }
        preview_report_path = run_dir / f"preview_step_{start_step:06d}.json"
        preview_report_path.write_text(
            json.dumps(preview_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        held_ablation = validation["unitSlopeResidualAblation"]
        seen_ablation = train_validation["unitSlopeResidualAblation"]
        print(
            "Unit-slope ablation: "
            f"held-global={held_ablation['median_global_recovery']*100:+.2f}% "
            f"held-edge={held_ablation['median_edge_recovery']*100:+.2f}% "
            f"held-grad={held_ablation['median_gradient_recovery']*100:+.2f}% "
            f"held-lattice={held_ablation['median_lattice_cell_excess']*100:+.2f}% "
            f"seen-global={seen_ablation['median_global_recovery']*100:+.2f}% "
            f"seen-edge={seen_ablation['median_edge_recovery']*100:+.2f}%",
            flush=True,
        )
        held_oracle = validation["oracleScalarGainAblation"]
        seen_oracle = train_validation["oracleScalarGainAblation"]
        held_diag = validation["residualDiagnosticDistributions"]
        seen_diag = train_validation["residualDiagnosticDistributions"]
        print(
            "Residual alignment: "
            f"held-cos={held_diag['residual_cosine_similarity']['median']:+.3f} "
            f"held-sign={held_diag['target_weighted_sign_agreement']['median']*100:.1f}% "
            f"held-ls-gain={held_diag['least_squares_residual_gain']['median']:.2f} "
            f"held-oracle-global={held_oracle['median_global_recovery']*100:+.2f}% "
            f"seen-cos={seen_diag['residual_cosine_similarity']['median']:+.3f} "
            f"seen-sign={seen_diag['target_weighted_sign_agreement']['median']*100:.1f}% "
            f"seen-oracle-global={seen_oracle['median_global_recovery']*100:+.2f}%",
            flush=True,
        )
        print(f"Preview root      : {preview_root}", flush=True)
        print(f"Preview report    : {preview_report_path}", flush=True)
        return 0, preview_report_path
    for end_step in remaining:
        training = _train_segment(
            model,
            optimizer,
            manifest,
            config,
            start_step=start_step,
            end_step=end_step,
            device=device,
            seed=seed,
            precision=args.amp_precision,
            augmentation_policy=augmentation_policy,
        )
        preview_root = run_dir / "previews" / f"step_{end_step:06d}"
        validation = _evaluate(
            model,
            manifest,
            config,
            split="validation",
            samples=validation_samples,
            device=device,
            seed=seed + 7001,
            precision=args.amp_precision,
            preview_root=preview_root,
            preview_samples=int(args.preview_samples),
            augmentation_policy=augmentation_policy,
        )
        train_validation = _evaluate(
            model,
            manifest,
            config,
            split="train",
            samples=train_validation_samples,
            device=device,
            seed=seed + 8001,
            precision=args.amp_precision,
            augmentation_policy=augmentation_policy,
        )
        item = {
            "step": int(end_step),
            "training": training,
            "trainValidation": train_validation,
            "validation": validation,
            "gateStatus": _gate_status(validation, config),
        }
        curve.append(item)
        _save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            config=config,
            step=end_step,
            seed=seed,
            manifest_path=manifest_path,
            curve=curve,
            augmentation_policy=augmentation_policy,
            initialization=initialization,
        )
        start_step = end_step

        print(
            f"Checkpoint {end_step}: "
            f"held-global={validation['median_global_recovery']*100:+.2f}% "
            f"held-edge={validation['median_edge_recovery']*100:+.2f}% "
            f"held-grad={validation['median_gradient_recovery']*100:+.2f}% "
            f"held-lattice={validation['median_lattice_cell_excess']*100:+.2f}% "
            f"seen-global={train_validation['median_global_recovery']*100:+.2f}% "
            f"seen-edge={train_validation['median_edge_recovery']*100:+.2f}%",
            flush=True,
        )

    report_path = _write_report(
        run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        config=config,
        device=device,
        precision=args.amp_precision,
        requested_stages=stages,
        curve=curve,
        checkpoint_path=checkpoint_path,
        augmentation_policy=augmentation_policy,
        initialization=initialization,
    )
    return 0, report_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Train the production-size V16+structure candidate on the broad "
            "authority-disjoint authored corpus"
        )
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--manifest", default=DEFAULT_MANIFEST)
    value.add_argument(
        "--stages",
        default="256",
        help=(
            "cumulative checkpoints; default 256 for a bounded first proof. "
            "Do not resume the current 596 checkpoint until the exact "
            "memorization diagnostic is resolved."
        ),
    )
    value.add_argument("--hr-size", type=int, default=512)
    value.add_argument(
        "--validation-samples",
        type=int,
        default=0,
        help="0 evaluates one sample from every held-out authority",
    )
    value.add_argument(
        "--train-validation-samples",
        type=int,
        default=0,
        help=(
            "fixed seen-authority diagnostic sample count; "
            "0 matches the held-out sample count"
        ),
    )
    value.add_argument("--seed", type=int, default=16201)
    value.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    value.add_argument(
        "--amp-precision",
        choices=("auto", "bf16", "fp16"),
        default="auto",
    )
    value.add_argument(
        "--augmentation-policy",
        choices=AUGMENTATION_POLICIES,
        default="legacy-random",
        help="training augmentation; d4-cyclic is the promoted broad-authority recipe",
    )
    value.add_argument(
        "--resume",
        default="",
        help="resume_checkpoint.pt from an earlier full broad proof",
    )
    value.add_argument(
        "--initialize-from",
        default="",
        help=(
            "load model weights/config from a full-broad checkpoint, reset Adam "
            "and start the requested schedule at step 0"
        ),
    )
    value.add_argument(
        "--preview-samples",
        type=int,
        default=4,
        help="fixed held-out samples saved as visual comparisons at each checkpoint",
    )
    value.add_argument(
        "--preview-only",
        action="store_true",
        help="load --resume and write previews for the current checkpoint without training",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    code, _ = run(parser().parse_args(argv))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
