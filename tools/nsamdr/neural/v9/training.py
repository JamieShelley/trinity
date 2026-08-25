"""Resumable high-throughput mixed-precision training for NSAMDR V10."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import ctypes
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from datetime import datetime
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from .config import V9Config
from .dataset import (
    PhysicalTileDatasetV9, ParametricPrimitiveTrainingDataset, SyntheticGeometryValidationDataset,
    dataset_fingerprint, load_dataset_manifest,
)
from .inference import resolve_device
from .geometry_metrics import (
    synthetic_region_chamfer_improvement,
    topology_mismatch,
    sdf_topology_mismatch,
    profile_width_ratio,
    zero_contour_distance,
    line_perpendicular_jitter_pixels,
    circle_radial_roughness_pixels,
    line_staircase_recovery,
)
from .losses import compute_losses
from .model import MODEL_SCHEMA, FidelityResidualNetV9, architecture_summary, model_hash, parameter_count
from .direct_coverage_specialist import BoundaryProfileSpecialist
from .seam_restoration import PhaseAwareSeamSR, multi_map_ridge_response
from .geometry_proof_ladder import build_proof_case
from .parametric_boundary import LocalParametricBoundaryDecoder, make_query_grid
from .oracle_patch_distillation import OraclePatchSDFPredictor, extract_target_patches
from .parametric_primitives import (
    PRIMITIVE_COUNT, PRIMITIVE_NAMES, PARAM_DIM, parametric_param_abs_error_torch,
    proof_case_primitive_target, render_parametric_sdf_torch,
)


class NonFiniteTrainingError(RuntimeError):
    pass


def _status(message: str = "") -> None:
    """Emit one training status line immediately, including under GUI pipes."""
    print(message, flush=True)


def _render_profile_metrics_numpy(
    image_rgb: np.ndarray,
    target_rgb: np.ndarray,
    target_sdf_pixels: np.ndarray,
) -> dict[str, float]:
    """Rendered transition metrics used for the hard Panel-3 -> Panel-2 gate."""
    image = np.asarray(image_rgb, dtype=np.float32)
    target = np.asarray(target_rgb, dtype=np.float32)
    sdf = np.asarray(target_sdf_pixels, dtype=np.float32)
    if sdf.ndim == 3:
        sdf = sdf[..., 0]
    distance = np.abs(sdf)
    gray = image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722
    gy, gx = np.gradient(gray)
    grad = np.sqrt(gx * gx + gy * gy + 1.0e-10)
    band = distance <= 6.0
    weight = grad * band.astype(np.float32)
    total = float(np.sum(weight))
    width = (
        float(np.sqrt(np.sum(weight * distance * distance) / total))
        if total > 1.0e-8
        else float("inf")
    )
    target_lo = np.min(target, axis=(0, 1), keepdims=True).astype(np.float32)
    target_hi = np.max(target, axis=(0, 1), keepdims=True).astype(np.float32)
    below = np.maximum(target_lo - image, 0.0)
    above = np.maximum(image - target_hi, 0.0)
    halo = float(max(float(np.max(below)), float(np.max(above))) * 255.0)
    return {"widthRmsPixels": width, "haloOvershoot8bit": halo}


def _oracle_render_match_numpy(
    candidate_rgb: np.ndarray,
    oracle_rgb: np.ndarray,
    target_rgb: np.ndarray,
    target_sdf_pixels: np.ndarray,
) -> dict[str, float]:
    """Direct rendered equivalence metrics for learned Panel 3 versus oracle Panel 2."""
    candidate = np.asarray(candidate_rgb, dtype=np.float32)
    oracle = np.asarray(oracle_rgb, dtype=np.float32)
    sdf = np.asarray(target_sdf_pixels, dtype=np.float32)
    if sdf.ndim == 3:
        sdf = sdf[..., 0]
    band = np.abs(sdf) <= 6.0
    if not np.any(band):
        return {
            "bandMae": float("inf"),
            "gradientMae": float("inf"),
            "profileCorrelation": -1.0,
            "widthRelativeError": float("inf"),
            "haloDelta8bit": float("inf"),
        }
    band_mae = float(np.mean(np.abs(candidate - oracle)[band]))
    candidate_gray = candidate[..., 0] * 0.2126 + candidate[..., 1] * 0.7152 + candidate[..., 2] * 0.0722
    oracle_gray = oracle[..., 0] * 0.2126 + oracle[..., 1] * 0.7152 + oracle[..., 2] * 0.0722
    cgy, cgx = np.gradient(candidate_gray)
    ogy, ogx = np.gradient(oracle_gray)
    cgrad = np.sqrt(cgx * cgx + cgy * cgy + 1.0e-10)
    ograd = np.sqrt(ogx * ogx + ogy * ogy + 1.0e-10)
    gradient_mae = float(np.mean(np.abs(cgrad[band] - ograd[band])))
    cv = cgrad[band].astype(np.float64)
    ov = ograd[band].astype(np.float64)
    cv -= cv.mean()
    ov -= ov.mean()
    denom = float(np.sqrt(np.sum(cv * cv) * np.sum(ov * ov)))
    correlation = (
        float(np.sum(cv * ov) / denom)
        if denom > 1.0e-12
        else (-1.0 if np.any(np.abs(cv - ov) > 1.0e-8) else 1.0)
    )
    candidate_profile = _render_profile_metrics_numpy(candidate, target_rgb, sdf)
    oracle_profile = _render_profile_metrics_numpy(oracle, target_rgb, sdf)
    width_relative_error = abs(
        float(candidate_profile["widthRmsPixels"]) - float(oracle_profile["widthRmsPixels"])
    ) / max(float(oracle_profile["widthRmsPixels"]), 1.0e-6)
    halo_delta = abs(
        float(candidate_profile["haloOvershoot8bit"]) - float(oracle_profile["haloOvershoot8bit"])
    )
    return {
        "bandMae": band_mae,
        "gradientMae": gradient_mae,
        "profileCorrelation": correlation,
        "widthRelativeError": width_relative_error,
        "haloDelta8bit": halo_delta,
    }


# These fields only affect throughput. They are intentionally omitted from the
# resume hash so a checkpoint created before the performance patch can continue.
_RUNTIME_ONLY_CONFIG_FIELDS = {
    "performance_profile",
    "data_loader_workers",
    "data_loader_prefetch_factor",
    "data_loader_persistent_workers",
    "cuda_prefetch",
    "reactive_vram_enabled",
    "reactive_vram_target_free_fraction",
    "reactive_vram_pause_free_fraction",
    "reactive_vram_resume_free_fraction",
    "reactive_vram_expand_hysteresis_fraction",
    "reactive_vram_expand_stable_steps",
    "reactive_vram_poll_seconds",
    "reactive_vram_oom_retries",
    "reactive_vram_release_cache",
    "reactive_vram_burst_reserve_fraction",
    "reactive_vram_stability_samples",
    "reactive_vram_stability_interval_seconds",
    "reactive_vram_dynamic_allocator_ceiling",
    "reactive_vram_start_in_offload",
    "reactive_host_pause_free_fraction",
    "reactive_host_resume_free_fraction",
    "channels_last",
    "amp_dtype",
    "fused_optimizer",
    "cudnn_benchmark",
    "allow_tf32",
    "loss_precision",
    "torch_compile_mode",
}


def _config_hash(config: V9Config) -> str:
    payload = config.to_dict()
    for field_name in _RUNTIME_ONLY_CONFIG_FIELDS:
        payload.pop(field_name, None)
    # Preserve resume compatibility with V9.2.x states created before the
    # optimizer/scheduler selectors existed. Default values are semantically
    # identical to that legacy behavior and are omitted from the hash.
    legacy_optimizer_defaults = {
        "optimizer_name": "adamw",
        "optimizer_beta1": 0.90,
        "optimizer_beta2": 0.99,
        "scheduler_name": "phase",
        "scheduler_min_lr_ratio": 0.25,
    }
    if all(payload.get(key) == value for key, value in legacy_optimizer_defaults.items()):
        for key in legacy_optimizer_defaults:
            payload.pop(key, None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _phase_for_epoch(epoch: int, config: V9Config) -> str:
    boundary = config.identity_epochs
    if epoch <= boundary:
        return "sdf-bootstrap"
    boundary += config.residual_epochs
    if epoch <= boundary:
        return "sdf-proof"
    boundary += config.seam_proof_epochs
    if epoch <= boundary:
        return "seam-proof"
    boundary += config.seam_authority_epochs
    if epoch <= boundary:
        return "seam-authority"
    boundary += config.boundary_epochs
    if epoch <= boundary:
        return "gate-proof"
    boundary += config.detail_epochs
    if epoch <= boundary:
        return "detail-reconstruction"
    return "physical-finetune"


def _parametric_b1b_substage(
    classifier_qualified: bool, parameter_qualified: bool
) -> str:
    """Train both checkpointed primitive heads in the canonical integration stage."""
    _ = classifier_qualified, parameter_qualified
    return "integration"


def _phase_lr(phase: str, config: V9Config, epoch: int | None = None) -> float:
    base = {
        "sdf-bootstrap": config.identity_learning_rate,
        "sdf-proof": config.learning_rate,
        "seam-proof": config.learning_rate * float(getattr(config, "seam_lr_multiplier", 2.0)),
        "seam-authority": config.learning_rate,
        "gate-proof": config.boundary_learning_rate,
        "detail-reconstruction": config.detail_learning_rate,
        "boundary-hardening": config.finetune_learning_rate,
        "physical-finetune": config.finetune_learning_rate,
    }[phase]
    if config.scheduler_name != "cosine-phase" or epoch is None:
        return base

    lengths = {
        "sdf-bootstrap": config.identity_epochs,
        "sdf-proof": config.residual_epochs,
        "seam-proof": config.seam_proof_epochs,
        "seam-authority": config.seam_authority_epochs,
        "gate-proof": config.boundary_epochs,
        "detail-reconstruction": config.detail_epochs,
        "boundary-hardening": config.physical_finetune_epochs,
        "physical-finetune": config.physical_finetune_epochs,
    }
    starts = {
        "sdf-bootstrap": 1,
        "sdf-proof": 1 + config.identity_epochs,
        "seam-proof": 1 + config.identity_epochs + config.residual_epochs,
        "seam-authority": 1 + config.identity_epochs + config.residual_epochs + config.seam_proof_epochs,
        "gate-proof": 1 + config.identity_epochs + config.residual_epochs + config.seam_proof_epochs + config.seam_authority_epochs,
        "detail-reconstruction": 1 + config.identity_epochs + config.residual_epochs + config.seam_proof_epochs + config.seam_authority_epochs + config.boundary_epochs,
        "boundary-hardening": 1 + config.identity_epochs + config.residual_epochs + config.seam_proof_epochs + config.seam_authority_epochs + config.boundary_epochs + config.detail_epochs,
        "physical-finetune": 1 + config.identity_epochs + config.residual_epochs + config.seam_proof_epochs + config.seam_authority_epochs + config.boundary_epochs + config.detail_epochs,
    }
    length = max(1, lengths[phase])
    if length <= 1:
        return base
    position = min(max(epoch - starts[phase], 0), length - 1) / float(length - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * position))
    ratio = config.scheduler_min_lr_ratio + (1.0 - config.scheduler_min_lr_ratio) * cosine
    return base * ratio


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", mode="w", encoding="utf-8", delete=False
    ) as temporary:
        json.dump(payload, temporary, indent=2)
        temporary.write("\n")
        temp_path = Path(temporary.name)
    os.replace(temp_path, path)


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    channels_last: bool,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        tensor = value.to(device, non_blocking=device.type == "cuda")
        if channels_last and tensor.ndim == 4 and tensor.is_floating_point():
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        moved[key] = tensor
    return moved


def _finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def _bad_gradient_names(model: torch.nn.Module, limit: int = 24) -> list[str]:
    # Detailed per-parameter scans are diagnostic-only. The hot path uses one
    # aggregate gradient norm instead of synchronizing once per parameter.
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not _finite_tensor(parameter.grad):
            names.append(name)
            if len(names) >= limit:
                break
    return names


def _bad_parameter_names(model: torch.nn.Module, limit: int = 24) -> list[str]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if not _finite_tensor(parameter):
            names.append(name)
            if len(names) >= limit:
                break
    return names


def _parameters_are_finite(model: torch.nn.Module) -> bool:
    tensors = [parameter.detach() for parameter in model.parameters()]
    if not tensors:
        return True
    try:
        norms = torch._foreach_norm(tensors)
        finite = torch.isfinite(torch.stack([value.float() for value in norms])).all()
    except (AttributeError, RuntimeError):
        finite = torch.stack([
            torch.isfinite(value).all() for value in tensors
        ]).all()
    return bool(finite.item())


def _abort_nonfinite(
    output_dir: Path,
    *,
    epoch: int,
    phase: str,
    batch_index: int,
    stage: str,
    details: dict[str, Any],
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor] | None = None,
) -> None:
    diagnostics = output_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    stem = f"nonfinite_epoch_{epoch:03d}_{phase}_batch_{batch_index:05d}_{stage}"
    report_path = diagnostics / f"{stem}.json"
    tensor_path = diagnostics / f"{stem}.pt"
    report = {
        "schema": "NSAMDR_V9_NONFINITE_DIAGNOSTIC_V1",
        "epoch": epoch,
        "phase": phase,
        "batch": batch_index,
        "stage": stage,
        "details": details,
        "badGradients": _bad_gradient_names(model),
        "badParameters": _bad_parameter_names(model),
    }
    _atomic_json(report, report_path)
    payload: dict[str, Any] = {"report": report, "state_dict": model.state_dict()}
    if batch is not None:
        payload["batch"] = {key: value.detach().cpu() for key, value in batch.items()}
    _atomic_torch_save(payload, tensor_path)
    raise NonFiniteTrainingError(
        f"Non-finite V9 training state at epoch {epoch}, phase {phase}, batch {batch_index}, "
        f"stage {stage}. Diagnostic report: {report_path}"
    )


def _capture_rng() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _rng_byte_tensor(value: Any) -> torch.Tensor:
    """Return a contiguous CPU ByteTensor accepted by PyTorch RNG APIs.

    Resume states are loaded with map_location=device so an RNG state saved on
    CPU can be remapped to CUDA with the rest of the checkpoint. PyTorch RNG
    restoration APIs require contiguous CPU uint8 tensors.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.array(value, dtype=np.uint8, copy=True)).contiguous()
    return torch.tensor(value, dtype=torch.uint8, device="cpu").contiguous()


def _restore_rng(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])

    numpy_state = payload["numpy"]
    if isinstance(numpy_state, list):
        numpy_state = tuple(numpy_state)
    if (
        isinstance(numpy_state, tuple)
        and len(numpy_state) >= 2
        and not isinstance(numpy_state[1], np.ndarray)
    ):
        numpy_state = (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            *numpy_state[2:],
        )
    np.random.set_state(numpy_state)

    torch.set_rng_state(_rng_byte_tensor(payload["torch"]))

    if torch.cuda.is_available() and "cuda" in payload:
        raw_cuda_states = payload["cuda"]
        if isinstance(raw_cuda_states, torch.Tensor):
            raw_cuda_states = [raw_cuda_states]
        cuda_states = [_rng_byte_tensor(state) for state in raw_cuda_states]
        device_count = torch.cuda.device_count()
        if len(cuda_states) == device_count:
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            for device_index, state in enumerate(cuda_states[:device_count]):
                torch.cuda.set_rng_state(state, device=device_index)


def _save_contact_sheet(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    path: Path,
    phase: str = "",
) -> None:
    from PIL import Image, ImageDraw

    def rgb(tensor: torch.Tensor) -> np.ndarray:
        value = tensor[0].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        return np.uint8(np.round(value * 255.0))

    target = rgb(batch["target_albedo"])
    baseline = rgb(outputs["baseline_albedo"])
    geometry_only = rgb(outputs["boundary_reconstructed_albedo"])
    reconstructed = rgb(outputs["albedo"])
    edge = outputs["edge_logits"][0, 0].detach().float().cpu().sigmoid().numpy()
    seam_diag_tensor = outputs.get("seam_authority")
    if seam_diag_tensor is not None:
        second_diagnostic = seam_diag_tensor[0, 0].detach().float().cpu().numpy()
        second_diagnostic_label = "seam authority"
    else:
        second_diagnostic = outputs.get("detail_confidence", outputs["confidence"])[0, 0].detach().float().cpu().numpy()
        second_diagnostic_label = "detail confidence"
    sdf = outputs["sdf"][0, 0].detach().float().cpu().numpy()
    residual = outputs.get("albedo_delta_medium")
    if residual is None:
        residual_rgb = np.zeros_like(baseline)
    else:
        r = residual[0].detach().float().cpu().permute(1, 2, 0).numpy()
        residual_rgb = np.uint8(np.clip(r * 5.0 * 0.5 + 0.5, 0.0, 1.0) * 255.0)
    diagnostics = [
        np.repeat(np.uint8(np.clip(edge, 0.0, 1.0)[..., None] * 255.0), 3, axis=2),
        np.repeat(np.uint8(np.clip(second_diagnostic, 0.0, 1.0)[..., None] * 255.0), 3, axis=2),
        residual_rgb,
        np.stack((
            np.uint8(np.clip(sdf * 0.5 + 0.5, 0.0, 1.0) * 255.0),
            np.uint8(np.clip(-sdf * 0.5 + 0.5, 0.0, 1.0) * 255.0),
            np.zeros_like(np.uint8(sdf)),
        ), axis=2),
    ]
    if phase in {"sdf-bootstrap", "sdf-proof"}:
        teacher = rgb(outputs.get("sdf_teacher_boundary_albedo", outputs["boundary_reconstructed_albedo"]))
        primitive_valid = batch.get("primitive_valid")
        raven_blocked = bool(
            phase == "sdf-proof"
            and primitive_valid is not None
            and float(primitive_valid.detach().float().max().cpu().item()) < 0.5
        )
        candidate = baseline if raven_blocked else rgb(outputs["boundary_reconstructed_albedo"])
        error = np.uint8(np.clip(np.abs(candidate.astype(np.float32) - teacher.astype(np.float32)) * 8.0, 0.0, 255.0))
        images = [baseline, teacher, candidate, error, *diagnostics]
        if phase == "sdf-bootstrap":
            labels = [
                "1 baseline / damaged LR", "2 GT geometry + analytic redraw",
                "3 B1a topology geometry + SAME renderer", "4 |P3-P2| x8",
                "edge", second_diagnostic_label, "detail residual x5", "contour SDF",
            ]
        elif raven_blocked:
            labels = [
                "1 Raven baseline", "2 diagnostic GT-SDF redraw",
                "3 RAVEN AUTHORITY BLOCKED - baseline passthrough", "4 diagnostic |P3-P2| x8",
                "edge", second_diagnostic_label, "detail residual x5", "contour SDF",
            ]
        else:
            labels = [
                "1 baseline / damaged LR", "2 GT primitive + analytic redraw",
                "3 predicted primitive + SAME analytic redraw", "4 |P3-P2| x8",
                "edge", second_diagnostic_label, "detail residual x5", "contour SDF",
            ]
    elif phase == "seam-proof":
        err = np.uint8(np.clip(np.abs(reconstructed.astype(np.float32)-target.astype(np.float32))*4.0,0,255))
        images = [baseline, target, reconstructed, err, *diagnostics]
        labels = [
            "1 degraded/bicubic seam", "2 authored HR seam",
            "3 B3 forced-GT-authority seam reconstruction", "4 B3 error x4",
            "edge", second_diagnostic_label, "detail residual x5", "contour SDF",
        ]
    elif phase == "seam-authority":
        gt = batch["target_edge"][0,0].detach().float().cpu().numpy()
        gt_rgb = np.repeat(np.uint8(np.clip(gt,0,1)[...,None]*255.0),3,axis=2)
        auth_rgb = diagnostics[1]
        images = [baseline, gt_rgb, auth_rgb, reconstructed, *diagnostics]
        labels = [
            "1 degraded input", "2 B4 GT seam/ridge target",
            "3 B4 predicted authority", "4 B4 learned seam reconstruction",
            "edge", second_diagnostic_label, "detail residual x5", "contour SDF",
        ]
    else:
        images = [baseline, target, geometry_only, reconstructed, *diagnostics]
        labels = [
            "1 true bicubic baseline", "2 authored HR target",
            "3 geometry + vector seam", "4 full geometry+detail",
            "edge", second_diagnostic_label, "detail residual x5", "contour SDF",
        ]
    height, width = images[0].shape[:2]
    canvas = Image.new("RGB", (width * 4, height * 2 + 28 * 2))
    draw = ImageDraw.Draw(canvas)
    for index, (array, label) in enumerate(zip(images, labels)):
        row, column = divmod(index, 4)
        x, y = column * width, row * (height + 28)
        canvas.paste(Image.fromarray(array, mode="RGB"), (x, y + 28))
        draw.text((x + 6, y + 6), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_live_same_renderer_case_sheet(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    live_root: Path,
    *,
    epoch: int,
    phase: str,
    config: V9Config,
) -> None:
    """Overwrite Stage-A blank evidence with the current learned candidate.

    V10.7.9 keeps the original oracle-preflight evidence paths deliberately live:
    Stage A initially writes Panels 3/4 as NOT EVALUATED, then every B1a/B1b
    synthetic audit overwrites those exact PNGs. Panel 2 remains the deterministic
    GT-geometry reference; Panel 3 is current predicted geometry rendered through
    the identical BoundaryRenderer; Panel 4 is |P3-P2| x8. This lets the user keep
    one folder open and watch the learned result converge toward Panel 2.
    """
    from PIL import Image, ImageDraw

    case_index = int(batch["synthetic_case_index"].detach().flatten()[0].cpu().item())
    case = build_proof_case(
        case_index,
        size=int(batch["target_albedo"].shape[-1]),
        max_distance=float(config.contour_sdf_max_distance_pixels),
    )

    def _rgb(tensor: torch.Tensor) -> np.ndarray:
        arr = tensor[0].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        return np.uint8(np.round(arr * 255.0))

    baseline = _rgb(outputs["baseline_albedo"])
    oracle = _rgb(outputs.get("sdf_teacher_boundary_albedo", outputs["boundary_reconstructed_albedo"]))
    candidate = _rgb(outputs["boundary_reconstructed_albedo"])
    error = np.uint8(
        np.clip(
            np.abs(candidate.astype(np.float32) - oracle.astype(np.float32)) * 8.0,
            0.0,
            255.0,
        )
    )
    p3p2_mae = float(np.mean(np.abs(candidate.astype(np.float32) - oracle.astype(np.float32))) / 255.0)

    stage_name = "B1a topology" if phase == "sdf-bootstrap" else "B1b parametric primitive"
    labels = (
        "1 baseline / damaged LR",
        "2 GT geometry + analytic redraw",
        f"3 LIVE e{epoch:03d} {stage_name} + SAME renderer",
        f"4 LIVE |P3-P2| x8  MAE={p3p2_mae:.4f}",
    )
    images = (baseline, oracle, candidate, error)
    height, width = baseline.shape[:2]
    label_height = 34
    canvas = Image.new("RGB", (width * 4, height + label_height), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    for column, (array, label) in enumerate(zip(images, labels)):
        x = column * width
        canvas.paste(Image.fromarray(array, mode="RGB"), (x, label_height))
        draw.text((x + 7, 9), label, fill=(235, 235, 235))

    targets = (
        live_root / "staged_evidence" / f"{case_index:02d}_{case.name}_stages.png",
        live_root / "staged_evidence_detailed" / f"{case_index:02d}_{case.name}_profile_stages.png",
    )
    # Encode once; the simple/detailed live paths intentionally show the same
    # four-panel truth contract while training is in B1a/B1b.
    from io import BytesIO
    encoded = BytesIO()
    canvas.save(encoded, format="PNG")
    payload = encoded.getvalue()
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.stem + ".live.tmp.png")
        temp.write_bytes(payload)
        os.replace(temp, target)


class _MetricAccumulator:
    """Accumulate scalar losses on-device and synchronize only when reported."""

    def __init__(self) -> None:
        self.names: tuple[str, ...] = ()
        self.values: torch.Tensor | None = None
        self.count = 0

    def add(self, losses: dict[str, torch.Tensor]) -> None:
        if not self.names:
            self.names = tuple(losses.keys())
            self.values = torch.zeros(
                len(self.names), device=losses[self.names[0]].device, dtype=torch.float32
            )
        assert self.values is not None
        self.values.add_(torch.stack([losses[name].detach().float() for name in self.names]))
        self.count += 1

    def average_value(self, name: str) -> float:
        if self.values is None or self.count <= 0:
            return 0.0
        index = self.names.index(name)
        return float((self.values[index] / self.count).item())

    def averages(self) -> dict[str, float]:
        if self.values is None or self.count <= 0:
            return {}
        values = (self.values / self.count).detach().cpu().tolist()
        return dict(zip(self.names, (float(value) for value in values)))


def _data_worker_init(worker_id: int) -> None:
    # OpenCV otherwise creates its own thread team inside every DataLoader
    # process, oversubscribing the CPU and starving the CUDA submission thread.
    del worker_id
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)


def _sampler_epoch_for_phase(phase: str, epoch: int, config: V9Config, dataset_length: int) -> int:
    """Map training phase/epoch to a deterministic dataset index block.

    B1b cycles a bounded analytic geometry bank so each case is revisited and
    the independent geometry mapper can converge. Other phases retain the
    historical fresh-block schedule.
    """
    epoch = max(1, int(epoch))
    dataset_length = max(1, int(dataset_length))
    if phase == "sdf-proof":
        bank_tiles = int(getattr(config, "spline_geometry_fixed_bank_tiles", 128))
        bank_blocks = max(1, (bank_tiles + dataset_length - 1) // dataset_length)
        b1b_epoch = max(0, epoch - int(config.identity_epochs) - 1)
        return b1b_epoch % bank_blocks
    return epoch - 1


class _EpochOffsetSampler(Sampler[int]):
    """Emit a fresh deterministic index block every training epoch.

    V9.8.5 re-used the same 96 deterministic train indices on every epoch.
    With ~82% synthetic geometry that meant the 7.9M-parameter GeometryNet saw
    only ~80 unique analytic shapes repeatedly, while Stage-B judged unseen G0-G5
    geometry.  The sampler keeps runs exactly reproducible but offsets indices by
    one dataset length per epoch, so persistent Windows workers receive new
    deterministic synthetic shapes/crops without rebuilding the DataLoader.
    """

    def __init__(self, length: int) -> None:
        self.length = max(1, int(length))
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))

    def __iter__(self):
        start = self.epoch * self.length
        return iter(range(start, start + self.length))

    def __len__(self) -> int:
        return self.length


def _build_loader(
    dataset: PhysicalTileDatasetV9,
    *,
    batch_size: int,
    device: torch.device,
    workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    rolling_epoch_indices: bool = False,
) -> DataLoader:
    sampler = _EpochOffsetSampler(len(dataset)) if rolling_epoch_indices else None
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if workers > 0:
        kwargs.update({
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "worker_init_fn": _data_worker_init,
            # Never let a dead Windows worker make the GUI look frozen forever.
            # Normal V9 batches arrive far faster than this; timeout converts a
            # genuine worker startup/decode deadlock into an actionable error.
            "timeout": 120.0,
        })
    return DataLoader(**kwargs)


class _CudaPrefetcher(Iterator[dict[str, torch.Tensor]]):
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        *,
        channels_last: bool,
    ) -> None:
        self.iterator = iter(loader)
        self.device = device
        self.channels_last = channels_last
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch: dict[str, torch.Tensor] | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            raw_batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = _move_batch(
                raw_batch, self.device, channels_last=self.channels_last
            )

    def __iter__(self) -> "_CudaPrefetcher":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self.next_batch is None:
            raise StopIteration
        current = torch.cuda.current_stream(self.device)
        current.wait_stream(self.stream)
        batch = self.next_batch
        for tensor in batch.values():
            tensor.record_stream(current)
        self._preload()
        return batch


def _iter_batches(
    loader: DataLoader,
    device: torch.device,
    *,
    channels_last: bool,
    cuda_prefetch: bool,
) -> Iterator[dict[str, torch.Tensor]]:
    if device.type == "cuda" and cuda_prefetch:
        return _CudaPrefetcher(loader, device, channels_last=channels_last)
    return iter(
        _move_batch(raw_batch, device, channels_last=channels_last)
        for raw_batch in loader
    )


def _production_component_modules(
    model: FidelityResidualNetV9,
) -> dict[str, tuple[str, torch.nn.Module]]:
    paths = model.architecture_contract().get("productionComponents")
    if not isinstance(paths, dict):
        raise RuntimeError("productionComponents architecture mapping is missing")
    named = dict(model.named_modules())
    components: dict[str, tuple[str, torch.nn.Module]] = {}
    for label, path_value in paths.items():
        path = str(path_value)
        module = named.get(path)
        if module is None:
            raise RuntimeError(
                f"production component {label!r} names missing module {path!r}"
            )
        components[str(label)] = (path, module)
    return components


def _register_component_forward_hooks(
    components: Mapping[str, tuple[str, torch.nn.Module]],
) -> tuple[dict[str, int], list[Any]]:
    counts = {label: 0 for label in components}
    handles: list[Any] = []
    for label, (_path, module) in components.items():
        def record(_module, _inputs, _output, *, component_label: str = label) -> None:
            counts[component_label] += 1
        handles.append(module.register_forward_hook(record))
    return counts, handles


def _snapshot_trainable_parameters(
    model: FidelityResidualNetV9,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _parameter_belongs_to(path: str, parameter_name: str) -> bool:
    return parameter_name == path or parameter_name.startswith(path + ".")


def _component_gradient_norms(
    components: Mapping[str, tuple[str, torch.nn.Module]],
) -> dict[str, tuple[float, int]]:
    result: dict[str, tuple[float, int]] = {}
    for label, (_path, module) in components.items():
        square_sum = 0.0
        with_gradient = 0
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            square_sum += float(gradient.square().sum().cpu().item())
            with_gradient += int(parameter.numel())
        result[label] = (math.sqrt(max(square_sum, 0.0)), with_gradient)
    return result


_COMPONENT_LOSS_TERMS: dict[str, tuple[str, ...]] = {
    "geometry": (
        "sdf_surface", "sdf_sign", "sdf_topology_sign", "sdf_eikonal",
        "edge", "orientation", "hardness",
    ),
    "structural representation": (
        "primitive_class", "primitive_param", "primitive_render",
    ),
    "boundary renderer": ("sdf_teacher_render", "boundary_fuzz", "boundary_halo"),
    "boundary/profile": (
        "boundary_specialist_coverage", "boundary_specialist_coverage_gradient",
        "boundary_specialist_profile_moment", "boundary_specialist_recovery",
    ),
    "PhaseAwareSeamSR": (
        "seam_reconstruction", "seam_phase_residual", "seam_recovery",
    ),
    "seam authority": ("seam_authority_teacher", "seam_authority_regularization"),
    "conditioned detail": (
        "detail_candidate_albedo", "detail_candidate_normal",
        "detail_candidate_roughness", "detail_candidate_emissive",
        "detail_candidate_material",
        "detail_laplacian", "detail_cross_map", "detail_recovery",
    ),
    "albedo physical head": ("detail_candidate_albedo", "detail_candidate_albedo_gradient"),
    "normal physical head": ("detail_candidate_normal", "detail_candidate_normal_gradient"),
    "material physical head": (
        "detail_candidate_roughness", "detail_candidate_emissive", "detail_candidate_material",
    ),
    "confidence": ("detail_confidence",),
    "regret": ("detail_regret_classifier", "appearance_regret", "normal_regret"),
    "BenefitSelector": ("boundary_gate", "boundary_pixel_regret", "final_recovery"),
}


def _component_epoch_evidence(
    model: FidelityResidualNetV9,
    components: Mapping[str, tuple[str, torch.nn.Module]],
    forward_counts: Mapping[str, int],
    gradient_evidence: Mapping[str, Mapping[str, float | int]],
    stage_start: Mapping[str, torch.Tensor],
    train_metrics: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    named_parameters = dict(model.named_parameters())
    evidence: dict[str, dict[str, Any]] = {}
    for label, (path, module) in components.items():
        parameter_count_value = sum(parameter.numel() for parameter in module.parameters())
        trainable_count = sum(
            parameter.numel() for parameter in module.parameters()
            if parameter.requires_grad
        )
        delta_square_sum = 0.0
        changed_parameters = 0
        for name, before in stage_start.items():
            if not _parameter_belongs_to(path, name):
                continue
            after = named_parameters[name].detach().cpu().float()
            difference = after - before.float()
            delta_square_sum += float(difference.square().sum().item())
            changed_parameters += int(torch.count_nonzero(difference).item())
        gradients = gradient_evidence.get(label, {})
        max_gradient = float(gradients.get("max", 0.0))
        last_gradient = float(gradients.get("last", 0.0))
        loss_terms = {
            key: float(train_metrics[key])
            for key in _COMPONENT_LOSS_TERMS.get(label, ())
            if key in train_metrics and math.isfinite(float(train_metrics[key]))
        }
        delta_norm = math.sqrt(max(delta_square_sum, 0.0))
        forward_calls = int(forward_counts.get(label, 0))
        trained = bool(max_gradient > 0.0 and delta_norm > 0.0)
        if parameter_count_value == 0:
            status = "stateless-active" if forward_calls > 0 else "stateless-missed"
        elif trainable_count == 0:
            status = "frozen-active" if forward_calls > 0 else "frozen-missed"
        elif trained:
            status = "trained"
        else:
            status = "trainable-no-update"
        evidence[label] = {
            "modulePath": path,
            "active": bool(forward_calls > 0),
            "forwardCalls": forward_calls,
            "status": status,
            "trained": trained,
            "frozen": bool(trainable_count == 0 and parameter_count_value > 0),
            "parameterCount": int(parameter_count_value),
            "trainableParameterCount": int(trainable_count),
            "parametersWithGradient": int(gradients.get("parametersWithGradient", 0)),
            "maxGradientNorm": max_gradient,
            "lastGradientNorm": last_gradient,
            "stageStartWeightDeltaL2": float(delta_norm),
            "changedParameterElements": int(changed_parameters),
            "lossContribution": loss_terms,
        }
    return evidence


def _aggregate_component_participation(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    components: dict[str, dict[str, Any]] = {}
    for record in records:
        phase = str(record.get("phase", "unknown"))
        epoch = int(record.get("epoch", 0))
        for label, evidence in dict(record.get("components", {})).items():
            aggregate = components.setdefault(str(label), {
                "modulePath": str(evidence.get("modulePath", "")),
                "active": False,
                "forwardCalls": 0,
                "trained": False,
                "parameterCount": int(evidence.get("parameterCount", 0)),
                "maxGradientNorm": 0.0,
                "lastGradientNorm": 0.0,
                "stageStartWeightDeltaL2": 0.0,
                "trainedPhases": [],
                "frozenPhases": [],
                "lossContribution": {},
                "lastEpoch": 0,
            })
            aggregate["active"] = bool(aggregate["active"] or evidence.get("active", False))
            aggregate["forwardCalls"] += int(evidence.get("forwardCalls", 0))
            aggregate["trained"] = bool(aggregate["trained"] or evidence.get("trained", False))
            aggregate["maxGradientNorm"] = max(
                float(aggregate["maxGradientNorm"]),
                float(evidence.get("maxGradientNorm", 0.0)),
            )
            aggregate["lastGradientNorm"] = float(evidence.get("lastGradientNorm", 0.0))
            aggregate["stageStartWeightDeltaL2"] = max(
                float(aggregate["stageStartWeightDeltaL2"]),
                float(evidence.get("stageStartWeightDeltaL2", 0.0)),
            )
            aggregate["lastEpoch"] = epoch
            if evidence.get("trained", False) and phase not in aggregate["trainedPhases"]:
                aggregate["trainedPhases"].append(phase)
            if evidence.get("frozen", False) and phase not in aggregate["frozenPhases"]:
                aggregate["frozenPhases"].append(phase)
            for key, value in dict(evidence.get("lossContribution", {})).items():
                aggregate["lossContribution"][str(key)] = float(value)
    return {
        "productionForward": "FidelityResidualNetV9.forward(inputs)",
        "components": components,
        "epochs": records,
    }


def _run_final_qualification(
    model: FidelityResidualNetV9,
    loader: DataLoader,
    config: V9Config,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    strict_missing_keys: list[str],
    strict_unexpected_keys: list[str],
) -> dict[str, Any]:
    """Run one strict-reloaded, uncached, no-override production forward."""
    model.eval()
    raw_batch = next(iter(loader))
    batch = _move_batch(
        raw_batch, device,
        channels_last=bool(config.channels_last and device.type == "cuda"),
    )
    components = _production_component_modules(model)
    forward_counts, handles = _register_component_forward_hooks(components)
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=use_amp,
    ):
        outputs = model(batch["input"])
    for handle in handles:
        handle.remove()

    required_outputs = (
        "albedo", "normal_xy", "material", "sdf", "boundary_refined_coverage",
        "seam_authority", "detail_candidate_albedo", "detail_confidence",
        "detail_regret", "benefit_selector_probability", "final_selector_gate",
    )
    nonfinite = [
        key for key in required_outputs
        if key not in outputs or not bool(torch.isfinite(outputs[key]).all().item())
    ]
    missed = [label for label, calls in forward_counts.items() if int(calls) <= 0]
    if nonfinite or missed or strict_missing_keys or strict_unexpected_keys:
        raise RuntimeError(
            "final production qualification failed: "
            f"nonfiniteOrMissingOutputs={nonfinite}, missedComponents={missed}, "
            f"missingState={strict_missing_keys}, unexpectedState={strict_unexpected_keys}"
        )
    output_digest = hashlib.sha256()
    output_metrics: dict[str, dict[str, float]] = {}
    for key in required_outputs:
        value = outputs[key].detach().float().cpu().contiguous()
        output_digest.update(key.encode("utf-8"))
        output_digest.update(value.numpy().tobytes())
        output_metrics[key] = {
            "mean": float(value.mean().item()),
            "min": float(value.amin().item()),
            "max": float(value.amax().item()),
        }
    return {
        "passed": True,
        "strictReload": True,
        "strictMissingKeys": strict_missing_keys,
        "strictUnexpectedKeys": strict_unexpected_keys,
        "modelClass": type(model).__name__,
        "schema": MODEL_SCHEMA,
        "productionForward": "FidelityResidualNetV9.forward(inputs)",
        "modelEval": True,
        "cacheUsed": False,
        "overridesUsed": False,
        "inputShape": list(batch["input"].shape),
        "componentForwardCalls": {key: int(value) for key, value in forward_counts.items()},
        "requiredOutputs": list(required_outputs),
        "outputMetrics": output_metrics,
        "outputSha256": output_digest.hexdigest(),
        "candidateModelSha256": model_hash(model),
    }



def _host_memory_info() -> tuple[int, int] | None:
    """Return (available, total) physical RAM without third-party packages."""
    try:
        if os.name == "nt":
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys), int(status.ullTotalPhys)
            return None

        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * available_pages), int(page_size * total_pages)
    except (AttributeError, OSError, ValueError):
        return None


class _ReactiveCudaMemoryGovernor:
    """Reactively share CUDA memory with unrelated foreground applications.

    The learned model and loss remain unchanged. The governor only changes
    where autograd's saved backward tensors live:
      gpu      -> normal GPU-resident saved activations (fast)
      offload  -> saved activations live in system RAM until backward needs them
      yield    -> no batch starts until global free VRAM recovers

    This deliberately operates between complete optimizer steps, so one batch
    is always trained with one coherent memory policy.
    """

    def __init__(self, device: torch.device, config: V9Config) -> None:
        self.device = device
        self.config = config
        self.enabled = bool(device.type == "cuda" and config.reactive_vram_enabled)
        quick_gpu_first = bool(getattr(config, "quick_preview_gpu_first", False))
        self.mode = (
            "offload"
            if self.enabled and config.reactive_vram_start_in_offload and not quick_gpu_first
            else "gpu"
        )
        self.stable_recovery_steps = 0
        self._last_reported_mode = ""
        self.total_bytes = 0
        self.target_free_bytes = 0
        self.pause_free_bytes = 0
        self.resume_free_bytes = 0
        self.expand_hysteresis_bytes = 0
        self.host_total_bytes = 0
        self.host_pause_free_bytes = 0
        self.host_resume_free_bytes = 0
        self.host_critical_free_bytes = 0
        self.burst_reserve_bytes = 0
        # Once activation offload has completed successfully, the process may
        # retain/reuse those committed host pages even when Windows reports a
        # low global "available" value. Do not deadlock on a percentage guard
        # after that working set is established; retain only a small absolute
        # critical floor.
        self.offload_has_succeeded = False
        # OOM retries are pinned to activation-offload mode until one full
        # optimizer step completes successfully. This prevents the safety gate
        # from immediately promoting the failed retry back to GPU mode.
        self.force_offload_retry = False

        # Conservative initial estimate of how much additional transient VRAM a
        # normal GPU-resident step may need beyond its idle allocations. The
        # estimate is replaced by measurements from successful steps.
        self.estimated_gpu_step_extra = 0
        self.estimated_offload_step_extra = 0

        if self.enabled:
            self.total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
            self.target_free_bytes = int(
                self.total_bytes * float(config.reactive_vram_target_free_fraction)
            )
            self.pause_free_bytes = int(
                self.total_bytes * float(config.reactive_vram_pause_free_fraction)
            )
            self.resume_free_bytes = int(
                self.total_bytes * float(config.reactive_vram_resume_free_fraction)
            )
            self.expand_hysteresis_bytes = int(
                self.total_bytes * float(config.reactive_vram_expand_hysteresis_fraction)
            )
            self.burst_reserve_bytes = int(
                self.total_bytes * float(config.reactive_vram_burst_reserve_fraction)
            )
            host_memory = _host_memory_info()
            if host_memory is not None:
                _host_free, self.host_total_bytes = host_memory
                self.host_pause_free_bytes = int(
                    self.host_total_bytes * float(config.reactive_host_pause_free_fraction)
                )
                self.host_resume_free_bytes = int(
                    self.host_total_bytes * float(config.reactive_host_resume_free_fraction)
                )
                self.host_critical_free_bytes = max(
                    2 * 1024 ** 3,
                    int(self.host_total_bytes * 0.03),
                )
            self.estimated_gpu_step_extra = int(self.total_bytes * 0.34)
            self.estimated_offload_step_extra = int(self.total_bytes * 0.18)

    @staticmethod
    def _gib(value: int | float) -> float:
        return float(value) / (1024.0 ** 3)

    def _free_bytes(self) -> int:
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        return int(free_bytes)

    def _our_reserved_bytes(self) -> int:
        return int(torch.cuda.memory_reserved(self.device))

    def _our_allocated_bytes(self) -> int:
        return int(torch.cuda.memory_allocated(self.device))

    def _required_free_bytes(self, mode: str) -> int:
        transient = (
            self.estimated_gpu_step_extra
            if mode == "gpu"
            else self.estimated_offload_step_extra
        )
        # Reserve is for an unrelated foreground application that allocates
        # during our batch after the pre-step check.
        return int(transient + self.burst_reserve_bytes)

    def _physical_live_budget(self, free_bytes: int) -> int:
        """Return the V9 live-allocation budget under the coexistence reserve.

        This is deliberately advisory/gating only. PyTorch's per-process
        allocator fraction is *not* changed here: a hard allocator ceiling
        counts reserved cache against the limit and can therefore raise OOM
        while substantial physical VRAM is still free. The governor instead
        protects the foreground reserve using device-wide free-memory samples
        before every batch.
        """
        our_reserved = self._our_reserved_bytes()
        other_usage = max(0, self.total_bytes - free_bytes - our_reserved)
        return max(
            0,
            self.total_bytes - other_usage - self.burst_reserve_bytes,
        )

    def _gpu_mode_feasible(self, required_free_bytes: int) -> bool:
        """Return whether GPU-resident mode can ever satisfy the current envelope.

        The model/optimizer have persistent live allocations that cannot be
        released between batches. If the learned transient estimate plus the
        foreground reserve exceeds the maximum free VRAM possible with those
        allocations resident, waiting for GPU mode is mathematically futile.
        """
        persistent_live = self._our_allocated_bytes()
        maximum_possible_free = max(0, self.total_bytes - persistent_live)
        return int(required_free_bytes) <= int(maximum_possible_free)

    def _stable_resource_snapshot(self) -> int:
        """Require several safe device-memory samples before launching a batch."""
        samples = int(self.config.reactive_vram_stability_samples)
        interval = float(self.config.reactive_vram_stability_interval_seconds)
        minimum = None
        for index in range(samples):
            current = self._free_bytes()
            minimum = current if minimum is None else min(minimum, current)
            if index + 1 < samples:
                time.sleep(interval)
        return int(minimum if minimum is not None else self._free_bytes())

    def _release_cache(self, *, reason: str, report: bool = False) -> int:
        before = self._free_bytes()
        if self.config.reactive_vram_release_cache:
            torch.cuda.empty_cache()
        after = self._free_bytes()
        if report and after > before:
            _status(
                f"  [VRAM] released CUDA cache ({reason}): "
                f"{self._gib(before):.2f} -> {self._gib(after):.2f} GiB free"
            )
        return after

    def _set_mode(self, mode: str, *, free_bytes: int, reason: str) -> None:
        if mode == self.mode and self._last_reported_mode == mode:
            return
        previous = self.mode
        self.mode = mode
        self._last_reported_mode = mode
        _status(
            f"  [VRAM] mode {previous} -> {mode}; "
            f"free={self._gib(free_bytes):.2f} GiB; {reason}"
        )

    def _host_free_bytes(self) -> int | None:
        info = _host_memory_info()
        if info is None:
            return None
        return int(info[0])

    def _host_offload_safe(self, *, resume: bool = False) -> bool:
        if self.host_total_bytes <= 0:
            return True
        free_bytes = self._host_free_bytes()
        if free_bytes is None:
            return True

        # Windows may keep pages from a completed save_on_cpu step in this
        # process working set instead of immediately returning them to the
        # system-wide "available" pool. Requiring 20-25% global free RAM before
        # every subsequent step therefore created an infinite yield loop even
        # though the same offload working set could be reused safely.
        if self.offload_has_succeeded:
            return free_bytes >= self.host_critical_free_bytes

        floor = self.host_resume_free_bytes if resume else self.host_pause_free_bytes
        return free_bytes >= max(floor, self.host_critical_free_bytes)

    def _resource_text(self, gpu_free_bytes: int) -> str:
        host_free = self._host_free_bytes()
        if host_free is None or self.host_total_bytes <= 0:
            return f"GPU free={self._gib(gpu_free_bytes):.2f} GiB"
        return (
            f"GPU free={self._gib(gpu_free_bytes):.2f} GiB, "
            f"host RAM free={self._gib(host_free):.1f} GiB"
        )

    def _wait_for_recovery(
        self,
        free_bytes: int,
        *,
        offload_required: bool = False,
        full_required: int | None = None,
    ) -> int:
        gpu_critical = free_bytes < self.pause_free_bytes
        host_critical = offload_required and not self._host_offload_safe()
        if not gpu_critical and not host_critical:
            return free_bytes

        reason_parts = []
        if gpu_critical:
            reason_parts.append(
                f"GPU free below {self._gib(self.pause_free_bytes):.2f} GiB pause floor"
            )
        if host_critical:
            reason_parts.append(
                "host RAM too pressured for safe activation offload"
            )
        self._set_mode(
            "yield",
            free_bytes=free_bytes,
            reason="; ".join(reason_parts),
        )
        _status(
            "  [VRAM] yielding between batches; "
            "waiting for either enough GPU VRAM for normal mode or enough "
            "GPU + host RAM for activation-offload mode"
        )

        while True:
            if self.config.reactive_vram_release_cache:
                torch.cuda.empty_cache()
            time.sleep(float(self.config.reactive_vram_poll_seconds))
            free_bytes = self._free_bytes()

            enough_for_gpu = (
                full_required is not None and free_bytes >= full_required
            )
            enough_for_offload = (
                free_bytes >= self.resume_free_bytes
                and self._host_offload_safe(resume=True)
            )
            if enough_for_gpu or enough_for_offload:
                break

        self.stable_recovery_steps = 0
        self._last_reported_mode = ""
        if full_required is not None and free_bytes >= full_required:
            self.mode = "gpu"
            self._set_mode(
                "gpu",
                free_bytes=free_bytes,
                reason="resources recovered enough for GPU-resident activations",
            )
        else:
            self.mode = "offload"
            self._set_mode(
                "offload",
                free_bytes=free_bytes,
                reason=(
                    "resources recovered enough for low-VRAM activation-offload mode; "
                    + self._resource_text(free_bytes)
                ),
            )
        return free_bytes

    def before_step(self) -> tuple[str, int, int]:
        if not self.enabled:
            allocated = (
                int(torch.cuda.memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            )
            return "gpu", allocated, 0

        # Return allocator cache first, then require multiple device-wide
        # snapshots. This is deliberately conservative for a workstation where
        # another GPU application may be changing its working set.
        free_bytes = self._release_cache(
            reason="pre-step cooperative safety gate",
            report=False,
        )
        free_bytes = min(free_bytes, self._stable_resource_snapshot())

        while True:
            gpu_required = self._required_free_bytes("gpu")
            offload_required = self._required_free_bytes("offload")

            gpu_feasible = self._gpu_mode_feasible(gpu_required)
            gpu_safe = gpu_feasible and free_bytes >= gpu_required
            offload_safe = (
                free_bytes >= offload_required
                and self._host_offload_safe(resume=False)
            )

            # If current live V9 allocations alone have consumed the foreground
            # reserve, do not attempt another batch. This uses physical/device-
            # wide accounting only; there is no PyTorch hard allocator cap.
            live = self._our_allocated_bytes()
            live_budget = self._physical_live_budget(free_bytes)
            if live > live_budget:
                gpu_safe = False
                offload_safe = False

            if self.force_offload_retry:
                # A failed GPU attempt must be retried in the lower-memory mode.
                # Do not allow the normal recovery logic to promote this retry
                # directly back to GPU just because global VRAM became free.
                self.stable_recovery_steps = 0
                if offload_safe:
                    self._set_mode(
                        "offload",
                        free_bytes=free_bytes,
                        reason="OOM retry pinned to activation-offload mode",
                    )
                else:
                    self.mode = "yield"
            elif self.mode == "gpu":
                if not gpu_safe:
                    self.stable_recovery_steps = 0
                    if offload_safe:
                        reason = (
                            "GPU-resident mode is infeasible with persistent model/optimizer "
                            "allocations; staying in activation-offload mode"
                            if not gpu_feasible
                            else (
                                "strict safety envelope: GPU-resident predicted "
                                f"step + foreground reserve requires "
                                f"{self._gib(gpu_required):.2f} GiB free"
                            )
                        )
                        self._set_mode(
                            "offload",
                            free_bytes=free_bytes,
                            reason=reason,
                        )
                    else:
                        self.mode = "yield"
            elif self.mode == "offload":
                if not offload_safe:
                    self.mode = "yield"
                elif gpu_safe:
                    self.stable_recovery_steps += 1
                    if self.stable_recovery_steps >= int(
                        self.config.reactive_vram_expand_stable_steps
                    ):
                        self._set_mode(
                            "gpu",
                            free_bytes=free_bytes,
                            reason=(
                                "strict safety envelope remained satisfied for "
                                f"{self.config.reactive_vram_expand_stable_steps} "
                                "successful offload steps; expanding to GPU-resident activations"
                            ),
                        )
                        self.stable_recovery_steps = 0
                else:
                    self.stable_recovery_steps = 0
            else:
                # yield
                if offload_safe:
                    self._set_mode(
                        "offload",
                        free_bytes=free_bytes,
                        reason="strict safety envelope recovered for offload mode",
                    )
                elif gpu_safe:
                    self._set_mode(
                        "gpu",
                        free_bytes=free_bytes,
                        reason="strict safety envelope recovered for GPU mode",
                    )

            if self.mode != "yield":
                break

            self._last_reported_mode = ""
            self._set_mode(
                "yield",
                free_bytes=free_bytes,
                reason=(
                    "strict safety envelope not satisfied; refusing to launch "
                    "another batch"
                ),
            )
            host_free = self._host_free_bytes()
            host_floor = (
                self.host_critical_free_bytes
                if self.offload_has_succeeded
                else max(self.host_pause_free_bytes, self.host_critical_free_bytes)
            )
            host_text = (
                "unknown"
                if host_free is None
                else f"{self._gib(host_free):.2f}/{self._gib(host_floor):.2f} GiB"
            )
            _status(
                f"  [VRAM] safety wait: GPU mode needs "
                f"{self._gib(gpu_required):.2f} GiB free "
                f"(feasible={'yes' if gpu_feasible else 'no'}); offload mode needs "
                f"{self._gib(offload_required):.2f} GiB GPU free; host free/floor="
                f"{host_text}. Training is yielding, not paging."
            )

            if self.config.reactive_vram_release_cache:
                torch.cuda.empty_cache()
            time.sleep(float(self.config.reactive_vram_poll_seconds))
            free_bytes = self._stable_resource_snapshot()

        allocated_before = int(torch.cuda.memory_allocated(self.device))
        torch.cuda.reset_peak_memory_stats(self.device)
        return self.mode, allocated_before, free_bytes

    def autograd_context(self, mode: str):
        if (
            self.enabled
            and mode == "offload"
            and hasattr(torch.autograd, "graph")
            and hasattr(torch.autograd.graph, "save_on_cpu")
        ):
            # Do not pin: this is intentionally conservative toward overall
            # system memory while unrelated programs are active.
            return torch.autograd.graph.save_on_cpu(pin_memory=False)
        return nullcontext()

    def after_step(self, mode: str, allocated_before: int) -> tuple[int, int]:
        if not self.enabled:
            peak = (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            )
            return peak, 0

        peak = int(torch.cuda.max_memory_allocated(self.device))
        extra = max(0, peak - int(allocated_before))

        if mode == "gpu":
            # Safety high-watermark: once a larger transient requirement has
            # been observed, never predict less during this process lifetime.
            self.estimated_gpu_step_extra = max(
                self.estimated_gpu_step_extra,
                int(extra * 1.18),
            )
        elif mode == "offload":
            self.estimated_offload_step_extra = max(
                self.estimated_offload_step_extra,
                int(extra * 1.18),
            )

        if mode == "offload":
            self.offload_has_succeeded = True

        if self.force_offload_retry and mode == "offload":
            # The exact failed batch has now completed in the low-memory mode.
            # Remain in offload mode; normal hysteresis may restore GPU mode only
            # after the configured number of subsequent successful steps.
            self.force_offload_retry = False
            self.stable_recovery_steps = 0
            _status(
                "  [VRAM] OOM retry completed in activation-offload mode; "
                "GPU promotion hysteresis restarted"
            )

        free_bytes = self._free_bytes()

        # If the just-completed step left less than the sharing target, release
        # allocator cache immediately and make the next step lower-memory.
        if free_bytes < self.target_free_bytes:
            free_bytes = self._release_cache(
                reason="post-step sharing target",
                report=False,
            )
            # Re-check after returning PyTorch's allocator cache. Previously the
            # governor switched to activation offload even when empty_cache() had
            # already restored ample device-wide free VRAM. That made Quick Raven
            # spend entire epochs paging saved tensors through system RAM.
            if mode == "gpu" and free_bytes < self.target_free_bytes:
                self.stable_recovery_steps = 0
                self._set_mode(
                    "offload",
                    free_bytes=free_bytes,
                    reason="foreground GPU usage remains above sharing target after cache release",
                )
            elif mode == "gpu":
                self.mode = "gpu"
        elif mode == "offload" and self.config.reactive_vram_release_cache:
            # Offload mode is explicitly the cooperative mode: do not hoard
            # cached CUDA blocks between batches.
            torch.cuda.empty_cache()
            free_bytes = self._free_bytes()

        return peak, free_bytes

    def handle_oom(self, *, batch_index: int, attempt: int) -> None:
        if not self.enabled:
            return
        # The failed autograd graph has been dereferenced by the caller. Force a
        # collection before releasing the CUDA cache so stale Python references
        # cannot keep large graph allocations alive across the retry.
        gc.collect()
        torch.cuda.empty_cache()
        free_bytes = self._free_bytes()
        self.force_offload_retry = True
        self.mode = "offload"
        self.stable_recovery_steps = 0
        self._last_reported_mode = ""
        _status(
            f"  [VRAM] CUDA OOM intercepted at batch {batch_index}; "
            f"retry={attempt}; free after cache release={self._gib(free_bytes):.2f} GiB"
        )
        if (
            free_bytes < self._required_free_bytes("offload")
            or not self._host_offload_safe(resume=False)
        ):
            self._set_mode(
                "yield",
                free_bytes=free_bytes,
                reason=(
                    "OOM recovery waiting for the strict offload safety envelope; "
                    "the retry remains pinned to activation-offload mode"
                ),
            )
        else:
            self._set_mode(
                "offload",
                free_bytes=free_bytes,
                reason="OOM recovery pins the exact batch retry to activation-offload mode",
            )

    def status_text(self) -> str:
        if not self.enabled:
            return "fixed"
        free_bytes = self._free_bytes()
        retry = " retry=offload" if self.force_offload_retry else ""
        return (
            f"{self.mode} free={self._gib(free_bytes):.2f}GiB"
            f" reserve={self._gib(self.burst_reserve_bytes):.2f}GiB{retry}"
        )


def _is_cuda_oom(error: BaseException) -> bool:
    if isinstance(error, getattr(torch, "OutOfMemoryError", RuntimeError)):
        return "out of memory" in str(error).lower() or error.__class__.__name__ == "OutOfMemoryError"
    return "out of memory" in str(error).lower() and "cuda" in str(error).lower()


def _resolve_amp_dtype(config: V9Config, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if config.amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("ampDtype=bf16 requested but this CUDA device does not support BF16")
        return torch.bfloat16
    if config.amp_dtype == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _configure_cuda(config: V9Config, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
    except AttributeError:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)
    except AttributeError:
        pass


def _convert_model_channels_last(model: torch.nn.Module) -> torch.nn.Module:
    converter = getattr(torch.nn.utils, "convert_conv2d_weight_memory_format", None)
    if converter is not None:
        return converter(model, torch.channels_last)
    return model.to(memory_format=torch.channels_last)


def _build_optimizer(
    model: torch.nn.Module,
    config: V9Config,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, str]:
    common = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": (config.optimizer_beta1, config.optimizer_beta2),
    }
    parametric_parameters = []
    seam_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if "geometry_net.parametric_primitive_field" in name:
            parametric_parameters.append(parameter)
        elif "seam_restorer" in name:
            seam_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    parameter_groups = [
        {"params": other_parameters, "lr_scale": 1.0},
        {"params": parametric_parameters, "lr_scale": float(getattr(config, "parametric_primitive_lr_multiplier", 8.0))},
        {"params": seam_parameters, "lr_scale": float(getattr(config, "seam_lr_multiplier", 2.0))},
    ]
    optimizer_class = torch.optim.AdamW if config.optimizer_name == "adamw" else torch.optim.Adam
    display = "AdamW" if config.optimizer_name == "adamw" else "Adam"
    if device.type == "cuda" and config.fused_optimizer:
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


def _restore_optimizer_implementation(
    optimizer: torch.optim.Optimizer,
    optimizer_mode: str,
) -> None:
    """Keep the selected kernel after loading an older optimizer state.

    Optimizer.load_state_dict restores old param-group flags as well as moment
    tensors. A state produced by standard AdamW contains fused=None, which
    would silently disable the new fused kernel unless it is reapplied.
    """
    if optimizer_mode.startswith("fused "):
        optimizer.defaults["fused"] = True
        optimizer.defaults["foreach"] = None
        for group in optimizer.param_groups:
            group["fused"] = True
            group["foreach"] = None
    elif optimizer_mode.startswith("foreach "):
        optimizer.defaults["fused"] = None
        optimizer.defaults["foreach"] = True
        for group in optimizer.param_groups:
            group["fused"] = None
            group["foreach"] = True


def _parameter_memory_format(parameter: torch.Tensor) -> torch.memory_format:
    if parameter.ndim == 4 and parameter.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    if parameter.ndim == 5 and parameter.is_contiguous(memory_format=torch.channels_last_3d):
        return torch.channels_last_3d
    return torch.contiguous_format


def _align_optimizer_state_to_parameters(
    optimizer: torch.optim.Optimizer,
) -> int:
    """Match Adam state tensors to the selected optimizer implementation.

    Parameter-shaped moment buffers must share each parameter's device, dtype
    and memory layout. In addition, fused/capturable AdamW requires its scalar
    ``step`` tensors on the same device as the corresponding parameters.

    This matters when a checkpoint is loaded on CPU for benchmarking: the
    legacy optimizer state is restored before the fused flag is reapplied, so
    PyTorch leaves ``step`` on CPU unless it is migrated explicitly.
    """
    migrated = 0
    moment_names = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")

    for group in optimizer.param_groups:
        step_on_parameter_device = bool(group.get("fused")) or bool(
            group.get("capturable")
        )
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state:
                continue

            if step_on_parameter_device and "step" in state:
                step = state["step"]
                if isinstance(step, torch.Tensor):
                    aligned_step = step.detach().to(
                        device=parameter.device,
                        dtype=torch.float32,
                        non_blocking=False,
                    ).contiguous()
                else:
                    aligned_step = torch.tensor(
                        float(step),
                        device=parameter.device,
                        dtype=torch.float32,
                    )
                if (
                    not isinstance(step, torch.Tensor)
                    or step.device != aligned_step.device
                    or step.dtype != aligned_step.dtype
                    or not step.is_contiguous()
                ):
                    state["step"] = aligned_step
                    migrated += 1
                elif state["step"] is not aligned_step:
                    state["step"] = aligned_step

            memory_format = _parameter_memory_format(parameter)
            for name in moment_names:
                value = state.get(name)
                if not isinstance(value, torch.Tensor):
                    continue

                aligned = value.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                    non_blocking=False,
                )
                if aligned.shape == parameter.shape:
                    aligned = aligned.contiguous(memory_format=memory_format)
                else:
                    aligned = aligned.contiguous()

                layout_matches = (
                    aligned.stride() == parameter.stride()
                    if aligned.shape == parameter.shape
                    else True
                )
                if (
                    value.device != aligned.device
                    or value.dtype != aligned.dtype
                    or value.stride() != aligned.stride()
                    or not layout_matches
                ):
                    state[name] = aligned
                    migrated += 1
                elif state[name] is not aligned:
                    state[name] = aligned

    return migrated


def _is_fused_layout_error(error: RuntimeError) -> bool:
    text = str(error).lower()
    return (
        "params, grads, exp_avgs, and exp_avg_sqs" in text
        and "same dtype, device, and layout" in text
    )


def _compile_training_model(
    model: FidelityResidualNetV9,
    mode: str,
) -> tuple[torch.nn.Module, str]:
    mode = str(mode).strip().lower()
    if mode == "off":
        return model, "off"
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        print("WARNING: torch.compile is unavailable; using eager execution.")
        return model, "unavailable"
    try:
        compile_mode = None if mode == "default" else mode
        if compile_mode is None:
            return compiler(model, fullgraph=False, dynamic=False), "default"
        return compiler(
            model,
            mode=compile_mode,
            fullgraph=False,
            dynamic=False,
        ), mode
    except Exception as error:
        print(f"WARNING: torch.compile setup failed ({error!r}); using eager execution.")
        return model, "failed->off"


def _teacher_seam_authority(batch: dict[str, torch.Tensor], config: V9Config) -> torch.Tensor:
    """Build B3 forced authority from the same missing-detail evidence B4 learns.

    V10.8.4 forced only the authored edge mask while the B3 recovery metric also
    counted ridge/groove detail. That made part of the judged target literally
    unreachable because authority was zero there. V10.8.8 derives a broad
    teacher from target edges, target-vs-LR pixel need and multi-map ridge gain.
    """
    target_albedo = batch["target_albedo"].float()
    target_normal = batch["target_normal"].float()
    target_material_class = batch["target_material_class"].float().unsqueeze(1)
    target_emissive = batch["target_emissive"].float()
    target_roughness = batch["target_roughness"].float()
    target_material = torch.cat((
        ((target_material_class + 0.5) / max(float(config.material_classes), 1.0)).clamp(0.0, 1.0),
        target_emissive.clamp(0.0, 1.0),
        target_roughness.clamp(0.0, 1.0),
    ), dim=1)
    source = batch["input"].float()
    hr_size = target_albedo.shape[-2:]
    source_albedo = F.interpolate(source[:, 0:3], size=hr_size, mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)
    source_normal = F.interpolate(source[:, 3:5], size=hr_size, mode="bilinear", align_corners=False).clamp(-1.0, 1.0)
    source_material = F.interpolate(source[:, 5:8], size=hr_size, mode="nearest").clamp(0.0, 1.0)
    target_ridge = multi_map_ridge_response(target_albedo, target_normal, target_material).detach()
    source_ridge = multi_map_ridge_response(source_albedo, source_normal, source_material).detach()
    pixel_need = (target_albedo - source_albedo).abs().mean(dim=1, keepdim=True)
    pixel_need = (pixel_need * float(getattr(config, "seam_missing_detail_scale", 8.0))).clamp(0.0, 1.0)
    ridge_need = (target_ridge - source_ridge).clamp(0.0, 1.0)
    edge = batch["target_edge"].float().clamp(0.0, 1.0)
    teacher = torch.maximum(
        edge * (0.25 + 0.75 * pixel_need),
        ridge_need * (0.35 + 0.65 * pixel_need),
    ).clamp(0.0, 1.0)
    radius = int(getattr(config, "seam_teacher_dilation_pixels", 2))
    if radius > 0:
        teacher = F.max_pool2d(teacher, kernel_size=radius * 2 + 1, stride=1, padding=radius)
    return teacher.detach().clamp(0.0, 1.0)


def _teacher_boundary_gate(
    batch: dict[str, torch.Tensor],
    config: V9Config,
) -> torch.Tensor:
    """Hard replacement authority across the known contaminated contour band.

    Stage A/B ask whether the renderer and predicted SDF can reconstruct a
    boundary when placement authority is known. A soft edge-derived gate kept a
    fraction of the degraded transition in EXP_0003, so the teacher is now 1.0
    across the reconstruction band with only a short outer falloff.
    """
    target_sdf_pixels = (
        batch["target_sdf"].float()
        * float(config.contour_sdf_max_distance_pixels)
    ).abs()
    radius = max(
        7.0,
        min(float(config.boundary_renderer_band_pixels) + 5.0, 9.0),
    )
    falloff = 0.25
    outside = torch.relu(target_sdf_pixels - radius)
    return torch.where(
        target_sdf_pixels <= radius,
        torch.ones_like(target_sdf_pixels),
        torch.exp(-outside / falloff),
    ).clamp(0.0, 1.0)


def _aligned_teacher_sdf(
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Align HR target SDF polarity to the observable LR prior per sample."""
    target = batch["target_sdf"].float()
    source = batch.get("source_sdf")
    if source is None:
        return target
    source = source.float()
    if source.shape[-2:] != target.shape[-2:]:
        source = F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)
    weight = torch.exp(-torch.minimum(source.abs(), target.abs()) * 8.0)
    denom = weight.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-6)
    same = ((source - target).abs() * weight).sum(dim=(1, 2, 3), keepdim=True) / denom
    flipped = ((source + target).abs() * weight).sum(dim=(1, 2, 3), keepdim=True) / denom
    polarity = torch.where(same <= flipped, torch.ones_like(same), -torch.ones_like(same))
    return target * polarity


def _attach_same_renderer_candidate_outputs(
    outputs: dict[str, torch.Tensor],
    model: FidelityResidualNetV9,
    batch: dict[str, torch.Tensor],
    config: V9Config,
) -> dict[str, torch.Tensor]:
    """Render predicted geometry through the exact deterministic Panel-2 path.

    V10.7.9 structural validation permits only one pixel-generation difference
    between Panels 2 and 3: GT geometry versus predicted geometry.  Both pass
    through render_sdf_teacher/BoundaryRenderer with identical gate/hardness.
    """
    gate = _teacher_boundary_gate(batch, config)
    hardness = torch.ones_like(gate)
    rendered = model._training_render_sdf_teacher(batch["input"], outputs["sdf"], gate, hardness)
    outputs["boundary_reconstructed_albedo"] = rendered["boundary_reconstructed_albedo"]
    outputs["boundary_reconstructed_normal"] = rendered["boundary_reconstructed_normal"]
    outputs["boundary_reconstructed_material"] = rendered["boundary_reconstructed_material"]
    outputs["coverage_negative"] = rendered["coverage_negative"]
    outputs["sdf_pixels_metric"] = rendered["sdf_pixels_metric"]
    return outputs


def _attach_sdf_teacher_outputs(
    outputs: dict[str, torch.Tensor],
    teacher_model: FidelityResidualNetV9,
    batch: dict[str, torch.Tensor],
    config: V9Config,
) -> dict[str, torch.Tensor]:
    """Attach detached Panel-2 GT-SDF render targets to a Stage-B output."""
    gate = _teacher_boundary_gate(batch, config)
    hardness = torch.ones_like(gate)
    with torch.no_grad():
        teacher = teacher_model._training_render_sdf_teacher(
            batch["input"], _aligned_teacher_sdf(batch), gate, hardness
        )
    outputs["sdf_teacher_boundary_albedo"] = teacher["boundary_reconstructed_albedo"].detach()
    outputs["sdf_teacher_boundary_normal"] = teacher["boundary_reconstructed_normal"].detach()
    outputs["sdf_teacher_boundary_material"] = teacher["boundary_reconstructed_material"].detach()
    outputs["sdf_teacher_pixels_metric"] = teacher["sdf_pixels_metric"].detach()
    outputs["sdf_teacher_coverage_negative"] = teacher["coverage_negative"].detach()
    return outputs


def _parametric_b1b_train_losses(
    model: FidelityResidualNetV9,
    batch: dict[str, torch.Tensor],
    config: V9Config,
    *,
    substage: str = "integration",
) -> dict[str, torch.Tensor]:
    """Supervise the exact checkpointed primitive field used by production."""
    if substage not in {"classifier", "parameters", "integration"}:
        raise ValueError(f"unsupported B1b substage: {substage}")

    field = model.geometry_net.parametric_primitive_field
    input_lr = batch["input"]
    direct = field(input_lr, input_lr[:, -1:])
    target_class = batch["primitive_class"].long().reshape(-1)
    valid = batch["primitive_valid"].float().reshape(-1) > 0.5
    if not bool(valid.all().item()):
        raise RuntimeError("V10.7.9 B1b received a tile without a complete primitive teacher")

    logits = direct["class_logits"].float()
    all_params = direct["params_by_class"].float()
    batch_index = torch.arange(all_params.shape[0], device=all_params.device)
    # Parameter learning is intentionally teacher-routed. A wrong classifier
    # cannot starve the correct primitive head during B1b-2.
    predicted_params = all_params[batch_index, target_class]
    target_params = batch["primitive_params"].float()
    param_mask = batch["primitive_param_mask"].float()

    class_loss = F.cross_entropy(logits, target_class)
    param_error = parametric_param_abs_error_torch(predicted_params, target_params, target_class) * param_mask
    param_loss = param_error.sum() / param_mask.sum().clamp_min(1.0)

    target_sdf_pixels = (
        batch["target_sdf"].float()
        * float(config.contour_sdf_max_distance_pixels)
    )
    predicted_sdf_pixels = render_parametric_sdf_torch(
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
            predicted_sdf_pixels, target_sdf_pixels.detach(), beta=0.12,
            reduction="none",
        ) * (0.10 + 1.90 * render_band)
    ).mean()
    total = (
        class_loss * float(getattr(config, "parametric_primitive_class_weight", 8.0))
        + param_loss * float(getattr(config, "parametric_primitive_param_weight", 48.0))
        + render_loss * float(getattr(config, "parametric_primitive_render_weight", 6.0))
    )

    zero = total.detach() * 0.0
    return {
        "total": total,
        "sdf": zero,
        "primitive_class": class_loss,
        "primitive_param": param_loss,
        "primitive_param_mae": param_loss.detach(),
        "primitive_class_accuracy": (logits.argmax(dim=1) == target_class).float().mean().detach(),
        "primitive_render": render_loss,
    }


def _forward_for_phase(
    model: FidelityResidualNetV9,
    batch: dict[str, torch.Tensor],
    phase: str,
    config: V9Config,
) -> dict[str, torch.Tensor]:
    gate = _teacher_boundary_gate(batch, config)
    hardness = torch.ones_like(gate)
    if phase in {"sdf-bootstrap", "sdf-proof"}:
        outputs = model(batch["input"])
        outputs = _attach_same_renderer_candidate_outputs(outputs, model, batch, config)
        return _attach_sdf_teacher_outputs(outputs, model, batch, config)

    if phase == "seam-proof":
        return model._forward_training(
            batch["input"],
            teacher_sdf=_aligned_teacher_sdf(batch),
            teacher_gate=gate,
            teacher_hardness=hardness,
            teacher_seam_authority=_teacher_seam_authority(batch, config),
            teacher_seam_tangent=batch["target_orientation"].float(),
            phase_only_seam_teacher=True,
        )

    if phase == "seam-authority":
        return model._forward_training(
            batch["input"],
            teacher_seam_tangent=batch["target_orientation"].float(),
        )
    if phase == "gate-proof":
        outputs = model(batch["input"])
        # The profile specialist is trained against the exact Panel-2 renderer
        # target while geometry remains the exact production candidate.
        outputs = _attach_sdf_teacher_outputs(outputs, model, batch, config)
        return outputs
    return model(batch["input"])


def _validate(
    model: FidelityResidualNetV9,
    loader: DataLoader,
    config: V9Config,
    device: torch.device,
    phase: str,
    *,
    amp_dtype: torch.dtype,
    exact_geometry_metrics: bool = False,
    live_evidence_root: Path | None = None,
    live_evidence_epoch: int = 0,
    progress_label: str | None = None,
) -> tuple[dict[str, float], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model.eval()
    totals = _MetricAccumulator()
    last_outputs: dict[str, torch.Tensor] | None = None
    last_batch: dict[str, torch.Tensor] | None = None
    use_amp = device.type == "cuda"
    geometry_gains: list[float] = []
    topology_mismatches: list[float] = []
    rendered_topology_mismatches: list[float] = []
    rendered_topology_regressions_relative: list[float] = []
    zero_contour_chamfers: list[float] = []
    zero_contour_rms: list[float] = []
    source_zero_contour_chamfers: list[float] = []
    contour_relative_gains: list[float] = []
    source_missing_contours: list[float] = []
    predicted_missing_contours: list[float] = []
    topology_regressions_relative: list[float] = []
    profile_width_ratios: list[float] = []
    source_profile_width_ratios: list[float] = []
    profile_error_gains: list[float] = []
    profile_teacher_recoveries: list[float] = []
    oracle_render_band_maes: list[float] = []
    oracle_global_maes: list[float] = []
    oracle_gradient_maes: list[float] = []
    oracle_profile_width_relative_errors: list[float] = []
    oracle_profile_correlations: list[float] = []
    oracle_core_halo_deltas_8bit: list[float] = []
    line_jitters: list[float] = []
    line_staircase_recoveries: list[float] = []
    circle_roughnesses: list[float] = []
    primitive_confusion = np.zeros((PRIMITIVE_COUNT, PRIMITIVE_COUNT), dtype=np.int64)
    primitive_teacher_param_errors: list[float] = []
    primitive_integrated_param_errors: list[float] = []
    validation_started = time.perf_counter()
    validation_total_items = len(loader)
    if progress_label:
        _status(
            f"  [validation] label={progress_label} item=0/{validation_total_items} "
            "elapsed=0.0s eta=0.0s"
        )
    validation_batches = _iter_batches(
        loader,
        device,
        channels_last=config.channels_last and use_amp,
        cuda_prefetch=config.cuda_prefetch and use_amp,
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(validation_batches, 1):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = _forward_for_phase(model, batch, phase, config)
            with torch.autocast(device_type=device.type, enabled=False):
                losses = compute_losses(outputs, batch, config, phase)
            if not _finite_tensor(losses["total"]):
                _abort_nonfinite(
                    Path(config.output_dir), epoch=0, phase=phase, batch_index=batch_index,
                    stage="validation-loss", details={"total": float(losses["total"].detach().cpu())},
                    model=model, batch=batch)
            totals.add(losses)
            if exact_geometry_metrics:
                primitive_valid = batch.get("primitive_valid")
                if primitive_valid is not None and bool((primitive_valid.detach().float().reshape(-1) > 0.5).any().item()):
                    target_cls = int(batch["primitive_class"].detach().long().reshape(-1)[0].item())
                    predicted_cls = int(outputs["primitive_class_index"].detach().long().reshape(-1)[0].item())
                    if 0 <= target_cls < PRIMITIVE_COUNT and 0 <= predicted_cls < PRIMITIVE_COUNT:
                        primitive_confusion[target_cls, predicted_cls] += 1
                        params_by_class = outputs["primitive_params_by_class"].detach().float()
                        target_params = batch["primitive_params"].detach().float()
                        target_mask = batch["primitive_param_mask"].detach().float()
                        target_class_tensor = batch["primitive_class"].detach().long().reshape(-1)
                        bi = torch.arange(params_by_class.shape[0], device=params_by_class.device)
                        teacher_params = params_by_class[bi, target_class_tensor]
                        integrated_params = outputs["primitive_params"].detach().float()
                        teacher_error = parametric_param_abs_error_torch(
                            teacher_params, target_params, target_class_tensor
                        ) * target_mask
                        integrated_error = parametric_param_abs_error_torch(
                            integrated_params, target_params, target_class_tensor
                        ) * target_mask
                        denom = float(target_mask.sum().clamp_min(1.0).item())
                        primitive_teacher_param_errors.append(float(teacher_error.sum().item()) / denom)
                        primitive_integrated_param_errors.append(float(integrated_error.sum().item()) / denom)
                target_rgb = (
                    batch["target_albedo"][0].detach().float().permute(1, 2, 0).cpu().numpy()
                )
                baseline_rgb = (
                    outputs["baseline_albedo"][0].detach().float().permute(1, 2, 0).cpu().numpy()
                )
                candidate_rgb = (
                    outputs["boundary_reconstructed_albedo"][0].detach().float().permute(1, 2, 0).cpu().numpy()
                )
                _before, _after, gain = synthetic_region_chamfer_improvement(
                    baseline_rgb, candidate_rgb, target_rgb
                )
                geometry_gains.append(float(gain))
                source_rendered_topology = float(topology_mismatch(baseline_rgb, target_rgb))
                predicted_rendered_topology = float(topology_mismatch(candidate_rgb, target_rgb))
                rendered_topology_mismatches.append(predicted_rendered_topology)
                rendered_topology_regressions_relative.append(
                    float(predicted_rendered_topology > source_rendered_topology)
                )

                predicted_raw = (
                    outputs.get("predicted_sdf_raw_pixels", outputs["predicted_sdf_pixels"])[0, 0]
                    .detach().float().cpu().numpy()
                )
                source_raw = (
                    outputs.get("source_sdf_prior_pixels")[0, 0]
                    .detach().float().cpu().numpy()
                )
                target_sdf_pixels = (
                    batch["target_sdf"][0, 0].detach().float().cpu().numpy()
                    * float(config.contour_sdf_max_distance_pixels)
                )
                source_topology = float(sdf_topology_mismatch(source_raw, target_sdf_pixels))
                predicted_topology = float(sdf_topology_mismatch(predicted_raw, target_sdf_pixels))
                topology_mismatches.append(predicted_topology)
                topology_regressions_relative.append(
                    float(predicted_topology > source_topology)
                )
                contour = zero_contour_distance(predicted_raw, target_sdf_pixels)
                source_contour = zero_contour_distance(source_raw, target_sdf_pixels)
                pred_chamfer = float(contour["chamferPixels"])
                source_chamfer = float(source_contour["chamferPixels"])
                zero_contour_chamfers.append(pred_chamfer)
                zero_contour_rms.append(float(contour["rmsPixels"]))
                source_zero_contour_chamfers.append(source_chamfer)
                source_missing = not math.isfinite(source_chamfer)
                predicted_missing = not math.isfinite(pred_chamfer)
                source_missing_contours.append(float(source_missing))
                predicted_missing_contours.append(float(predicted_missing))
                if source_missing and not predicted_missing:
                    contour_relative_gains.append(1.0)
                elif not source_missing and predicted_missing:
                    contour_relative_gains.append(-1.0)
                elif source_missing and predicted_missing:
                    contour_relative_gains.append(0.0)
                else:
                    contour_relative_gains.append(
                        (source_chamfer - pred_chamfer) / max(source_chamfer, 0.25)
                    )
                predicted_profile = profile_width_ratio(candidate_rgb, target_rgb, target_sdf_pixels)
                source_profile = profile_width_ratio(baseline_rgb, target_rgb, target_sdf_pixels)
                teacher_rgb = (
                    outputs.get("sdf_teacher_boundary_albedo", outputs["boundary_reconstructed_albedo"])[0]
                    .detach().float().permute(1, 2, 0).cpu().numpy()
                )
                teacher_profile = profile_width_ratio(teacher_rgb, target_rgb, target_sdf_pixels)
                oracle_match = _oracle_render_match_numpy(
                    candidate_rgb, teacher_rgb, target_rgb, target_sdf_pixels
                )
                oracle_render_band_maes.append(float(oracle_match["bandMae"]))
                # Direct P3/P2 pixel equivalence is the primary V10.7.9.1
                # diagnostic-redraw contract. It measures exactly what the four
                # panel evidence shows and is not destabilized by a single
                # cross-section width/halo estimate.
                oracle_global_maes.append(float(np.mean(np.abs(candidate_rgb - teacher_rgb))))
                oracle_gradient_maes.append(float(oracle_match["gradientMae"]))
                case_kind = int(batch.get("synthetic_case_kind", torch.tensor(0, device=device)).flatten()[0].item())
                # Width/profile/halo are 1-D cross-section metrics. They are
                # meaningful for isolated lines and circles, but become
                # numerically discontinuous at corners, junctions, boxes and
                # multi-stroke intersections. Those structures remain gated by
                # contour, topology, band-MAE and gradient equivalence instead.
                if case_kind in (1, 2):
                    width_error = float(oracle_match["widthRelativeError"])
                    profile_corr = float(oracle_match["profileCorrelation"])
                    halo_delta = float(oracle_match["haloDelta8bit"])
                    # Very thin sub-pixel strokes can make the RMS profile
                    # estimator degenerate even for bit-identical P2/P3. Do not
                    # turn an undefined diagnostic into a structural rejection.
                    if math.isfinite(width_error) and profile_corr > -0.999:
                        oracle_profile_width_relative_errors.append(width_error)
                        oracle_profile_correlations.append(profile_corr)
                        # G5 deliberately contains injected halo; exclude only
                        # that stress control from the clean-profile halo contract.
                        is_stress_case = bool(
                            batch.get("synthetic_case_stress", torch.tensor(False, device=device))
                            .flatten()[0].item()
                        )
                        if not is_stress_case and math.isfinite(halo_delta):
                            oracle_core_halo_deltas_8bit.append(halo_delta)
                profile_width_ratios.append(predicted_profile)
                source_profile_width_ratios.append(source_profile)
                source_profile_error = abs(source_profile - 1.0)
                pred_profile_error = abs(predicted_profile - 1.0)
                teacher_profile_error = abs(teacher_profile - 1.0)
                profile_error_gains.append(
                    (source_profile_error - pred_profile_error) / max(source_profile_error, 0.05)
                )
                recoverable_profile = max(source_profile_error - teacher_profile_error, 0.05)
                profile_teacher_recoveries.append(
                    (source_profile_error - pred_profile_error) / recoverable_profile
                )
                if case_kind == 1:
                    line_jitters.append(
                        line_perpendicular_jitter_pixels(predicted_raw, target_sdf_pixels)
                    )
                    line_staircase_recoveries.append(
                        line_staircase_recovery(source_raw, predicted_raw, target_sdf_pixels)
                    )
                elif case_kind == 2:
                    circle_roughnesses.append(
                        circle_radial_roughness_pixels(predicted_raw, target_sdf_pixels)
                    )
                if live_evidence_root is not None and phase in {"sdf-bootstrap", "sdf-proof"}:
                    _save_live_same_renderer_case_sheet(
                        outputs, batch, live_evidence_root,
                        epoch=live_evidence_epoch, phase=phase, config=config,
                    )
            if progress_label:
                elapsed = time.perf_counter() - validation_started
                average = elapsed / max(1, batch_index)
                eta = average * max(0, validation_total_items - batch_index)
                _status(
                    f"  [validation] label={progress_label} item={batch_index}/{validation_total_items} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
                )
            last_outputs, last_batch = outputs, batch
    assert last_outputs is not None and last_batch is not None
    averages = totals.averages()
    if exact_geometry_metrics:
        finite_gains = [v for v in geometry_gains if math.isfinite(v)]
        finite_chamfer = [v for v in zero_contour_chamfers if math.isfinite(v)]
        finite_rms = [v for v in zero_contour_rms if math.isfinite(v)]
        finite_width = [v for v in profile_width_ratios if math.isfinite(v)]
        finite_source_chamfer = [v for v in source_zero_contour_chamfers if math.isfinite(v)]
        finite_relative_gain = [v for v in contour_relative_gains if math.isfinite(v)]
        finite_source_width = [v for v in source_profile_width_ratios if math.isfinite(v)]
        finite_profile_gain = [v for v in profile_error_gains if math.isfinite(v)]
        finite_profile_teacher_recovery = [v for v in profile_teacher_recoveries if math.isfinite(v)]
        finite_oracle_render_mae = [v for v in oracle_render_band_maes if math.isfinite(v)]
        finite_oracle_global_mae = [v for v in oracle_global_maes if math.isfinite(v)]
        finite_oracle_gradient_mae = [v for v in oracle_gradient_maes if math.isfinite(v)]
        finite_oracle_width_error = [v for v in oracle_profile_width_relative_errors if math.isfinite(v)]
        finite_oracle_profile_corr = [v for v in oracle_profile_correlations if math.isfinite(v)]
        finite_oracle_core_halo_delta = [v for v in oracle_core_halo_deltas_8bit if math.isfinite(v)]
        finite_line_jitter = [v for v in line_jitters if math.isfinite(v)]
        finite_line_staircase = [v for v in line_staircase_recoveries if math.isfinite(v) and v >= 0.0]
        finite_circle_roughness = [v for v in circle_roughnesses if math.isfinite(v)]
        averages.update({
            "sdf_stageb_geometry_gain_mean": float(np.mean(finite_gains)) if finite_gains else -1.0,
            "sdf_stageb_geometry_win_fraction": float(np.mean(np.asarray(finite_gains) > 0.0)) if finite_gains else 0.0,
            "sdf_stageb_geometry_regression_fraction": float(np.mean(np.asarray(finite_gains) < -0.02)) if finite_gains else 1.0,
            "sdf_stageb_topology_mismatch_fraction": float(np.mean(topology_mismatches)) if topology_mismatches else 1.0,
            "sdf_stageb_topology_regression_fraction": float(np.mean(topology_regressions_relative)) if topology_regressions_relative else 1.0,
            "sdf_stageb_rendered_topology_mismatch_fraction": float(np.mean(rendered_topology_mismatches)) if rendered_topology_mismatches else 1.0,
            "sdf_stageb_rendered_topology_regression_fraction": float(np.mean(rendered_topology_regressions_relative)) if rendered_topology_regressions_relative else 1.0,
            "sdf_source_zero_contour_chamfer_pixels": float(np.mean(finite_source_chamfer)) if finite_source_chamfer else float("inf"),
            "sdf_zero_contour_chamfer_pixels": float(np.mean(finite_chamfer)) if finite_chamfer else float("inf"),
            "sdf_zero_contour_rms_pixels": float(np.mean(finite_rms)) if finite_rms else float("inf"),
            "sdf_zero_contour_relative_gain_mean": float(np.mean(finite_relative_gain)) if finite_relative_gain else -1.0,
            "sdf_zero_contour_relative_win_fraction": float(np.mean(np.asarray(finite_relative_gain) > 0.0)) if finite_relative_gain else 0.0,
            "sdf_zero_contour_relative_regression_fraction": float(np.mean(np.asarray(finite_relative_gain) < -0.05)) if finite_relative_gain else 1.0,
            "sdf_source_missing_contour_fraction": float(np.mean(source_missing_contours)) if source_missing_contours else 1.0,
            "sdf_predicted_missing_contour_fraction": float(np.mean(predicted_missing_contours)) if predicted_missing_contours else 1.0,
            "sdf_stageb_profile_width_ratio_mean": float(np.mean(finite_width)) if finite_width else float("inf"),
            "sdf_source_profile_width_ratio_mean": float(np.mean(finite_source_width)) if finite_source_width else float("inf"),
            "sdf_profile_error_relative_gain_mean": float(np.mean(finite_profile_gain)) if finite_profile_gain else -1.0,
            "sdf_profile_teacher_recovery_mean": float(np.mean(finite_profile_teacher_recovery)) if finite_profile_teacher_recovery else -1.0,
            "sdf_oracle_render_band_mae_mean": float(np.mean(finite_oracle_render_mae)) if finite_oracle_render_mae else float("inf"),
            "sdf_oracle_global_mae_mean": float(np.mean(finite_oracle_global_mae)) if finite_oracle_global_mae else float("inf"),
            "sdf_oracle_global_mae_case_max": float(max(finite_oracle_global_mae)) if finite_oracle_global_mae else float("inf"),
            "sdf_oracle_gradient_mae_mean": float(np.mean(finite_oracle_gradient_mae)) if finite_oracle_gradient_mae else float("inf"),
            "sdf_oracle_profile_width_relative_error_mean": float(np.mean(finite_oracle_width_error)) if finite_oracle_width_error else float("inf"),
            "sdf_oracle_profile_correlation_mean": float(np.mean(finite_oracle_profile_corr)) if finite_oracle_profile_corr else -1.0,
            "sdf_oracle_core_halo_delta_8bit_max": float(max(finite_oracle_core_halo_delta)) if finite_oracle_core_halo_delta else float("inf"),
            "sdf_line_perpendicular_jitter_pixels_mean": float(np.mean(finite_line_jitter)) if finite_line_jitter else 0.0,
            "sdf_line_staircase_recovery_mean": float(np.mean(finite_line_staircase)) if finite_line_staircase else 1.0,
            "sdf_circle_radial_roughness_pixels_mean": float(np.mean(finite_circle_roughness)) if finite_circle_roughness else 0.0,
            "primitive_teacher_param_mae": float(np.mean(primitive_teacher_param_errors)) if primitive_teacher_param_errors else float("inf"),
            "primitive_integrated_param_mae": float(np.mean(primitive_integrated_param_errors)) if primitive_integrated_param_errors else float("inf"),
            "primitive_confusion_matrix": primitive_confusion.tolist(),
            "primitive_per_class_accuracy": [
                (float(primitive_confusion[i, i]) / float(primitive_confusion[i].sum()))
                if int(primitive_confusion[i].sum()) > 0 else 0.0
                for i in range(PRIMITIVE_COUNT)
            ],
        })
    return averages, last_outputs, last_batch


def _validate_v992_architecture_contract(contract: Mapping[str, object]) -> None:
    """Reject stale/incompatible models before optimizer allocation."""
    expected_outputs = (
        "source_sdf_prior", "parametric_primitive_geometry", "edge", "orientation", "hardness",
    )
    expected_stages = (
        "geometry-conditioning", "B1b-parametric-primitive",
        "B2-same-deterministic-redraw", "phase-aware-seam",
        "boundary-profile", "physical-detail", "benefit-selector",
    )
    production_components = contract.get("productionComponents")
    if (
        contract.get("schema") != MODEL_SCHEMA
        or contract.get("geometryModel") != "GeometryNet"
        or contract.get("renderer") != "BoundaryRenderer"
        or bool(contract.get("geometryCanPaintRgb"))
        or bool(contract.get("profileSpecialistCanPaintRgb"))
        or tuple(contract.get("geometryOutputs", ())) != expected_outputs
        or "primitive" not in str(contract.get("geometryPrediction", "")).lower()
        or "analytic" not in str(contract.get("reconstructionPrimitive", "")).lower()
        or int(contract.get("parametricPrimitiveClassCount", 0)) != PRIMITIVE_COUNT
        or int(contract.get("parametricPrimitiveParamDim", 0)) != PARAM_DIM
        or not bool(contract.get("topologyGeometryFeatureSplit"))
        or not bool(contract.get("sharedAcrossPhysicalMaps"))
        or bool(contract.get("moduloCoordinatePhase"))
        or bool(contract.get("pointwiseFourierSdfAuthority"))
        or bool(contract.get("rendererZeroContourRedistance"))
        or int(contract.get("subpixelSamples", 0)) != 9
        or not bool(contract.get("topologyFrozenDuringProof"))
        or contract.get("productionForward") != "FidelityResidualNetV9.forward(inputs) with no override authority"
        or not isinstance(production_components, dict)
        or len(production_components) < 12
        or not bool(contract.get("directionalSeamEnabled"))
        or not bool(contract.get("detailReconstructionEnabled"))
        or contract.get("profileSpecialist") != "BoundaryProfileSpecialist"
        or contract.get("benefitSelector") != "BenefitSelector"
        or contract.get("detailReconstructor") != "GeometryConditionedDetailNet"
        or bool(contract.get("detailMovesContour"))
        or tuple(contract.get("stagedProofs", ())) != expected_stages
        or "exact same deterministic boundaryrenderer" not in str(contract.get("panel3StructuralTarget", "")).lower()
        or "deterministic renderer only" not in str(contract.get("structuralPixelAuthority", "")).lower()
        or "training-only" not in str(contract.get("teacherRendererTarget", "")).lower()
    ):
        raise RuntimeError(f"V10.7.9 architecture contract failed: {contract!r}")


# Backward-compatible internal name for older unit-test imports.
_validate_v991_architecture_contract = _validate_v992_architecture_contract
_validate_v990_architecture_contract = _validate_v992_architecture_contract


def _explicit_primitive_structure_microproof(
    device: torch.device,
    config: V9Config,
) -> tuple[float, float, float, float]:
    """Representation proof for V10.7.9 compact manufactured primitives.

    This compares exact line/circle parameters against the permanent proof SDFs
    and checks that small parameter perturbations receive gradients. Local kinks
    are impossible because no local contour degrees of freedom exist.
    """
    max_distance = float(config.contour_sdf_max_distance_pixels)
    line_case = build_proof_case(5, size=512, max_distance=max_distance)  # 33deg
    circle_case = build_proof_case(13, size=512, max_distance=max_distance)
    targets = []
    preds = []
    for case in (line_case, circle_case):
        target = proof_case_primitive_target(case.name, 512)
        params = torch.from_numpy(target.params).to(device).unsqueeze(0)
        cls = torch.tensor([target.class_index], device=device, dtype=torch.int64)
        pred = render_parametric_sdf_torch(params, cls, 512, 512)[0, 0]
        preds.append(pred)
        targets.append(torch.from_numpy(case.target_sdf[..., 0]).to(device) * max_distance)
    line_pred, circle_pred = preds
    line_target, circle_target = targets
    line_band = line_target.abs() <= 6.0
    circle_band = circle_target.abs() <= 6.0
    after = 0.5 * (
        float((line_pred[line_band] - line_target[line_band]).abs().mean().item())
        + float((circle_pred[circle_band] - circle_target[circle_band]).abs().mean().item())
    )
    # Baseline is the observable LR SDF expanded to HR, matching the actual task.
    def _source_mae(case, target_tensor):
        src = torch.from_numpy(case.low_rgb[..., 0]).to(device).unsqueeze(0).unsqueeze(0)
        src = F.interpolate(src, size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
        # Only used as a monotonic reference; pixel intensity is deliberately not
        # treated as a metric SDF.
        return float(src.std().item()) + 1.0
    before = 0.5 * (_source_mae(line_case, line_target) + _source_mae(circle_case, circle_target))
    line_jitter = float(line_perpendicular_jitter_pixels(
        line_pred.detach().cpu().numpy(), line_target.detach().cpu().numpy()
    ))
    curve_rough = float(circle_radial_roughness_pixels(
        circle_pred.detach().cpu().numpy(), circle_target.detach().cpu().numpy()
    ))

    true = proof_case_primitive_target(line_case.name, 512)
    probe = torch.from_numpy(true.params).to(device).unsqueeze(0).clone().requires_grad_(True)
    with torch.no_grad():
        exact = render_parametric_sdf_torch(
            torch.from_numpy(true.params).to(device).unsqueeze(0),
            torch.tensor([true.class_index], device=device), 512, 512
        )
    perturbed = probe.clone()
    # Use a differentiable loss from a slightly wrong centre/width without
    # mutating a leaf in-place.
    delta = torch.zeros_like(perturbed)
    delta[:, 0] = 0.01
    delta[:, 8] = 0.02
    rendered = render_parametric_sdf_torch(
        (perturbed + delta).clamp(-1.0, 1.0),
        torch.tensor([true.class_index], device=device), 512, 512
    )
    band = exact.abs() <= 6.0
    probe_loss = (rendered[band] - exact[band]).square().mean()
    probe_loss.backward()
    grad = float(probe.grad.abs().sum().item()) if probe.grad is not None else 0.0
    if (
        not math.isfinite(after) or after > 0.60
        or not math.isfinite(line_jitter) or line_jitter > 0.15
        or not math.isfinite(curve_rough) or curve_rough > 0.10
        or grad <= 1.0e-8
    ):
        raise RuntimeError(
            "V10.7.9 explicit-parametric representation proof failed: "
            f"MAE={after:.6f}px lineJitter={line_jitter:.6f}px "
            f"curveRough={curve_rough:.6f}px grad={grad:.3e}"
        )
    return before, after, line_jitter, curve_rough


# Historical internal name retained for callers/tests.
def _oracle_patch_structure_microproof(
    device: torch.device, config: V9Config | None = None
) -> tuple[float, float, float, float]:
    cfg = config or V9Config()
    cfg.validate()
    return _explicit_primitive_structure_microproof(device, cfg)

def _parametric_structure_microproof(device: torch.device) -> tuple[float, float, float, float]:
    """Fast representation proof for the analytic line/arc geometry path.

    This is intentionally deterministic: startup must prove that the active
    renderer-facing representation can express a smooth shallow line and a
    smooth circle even when its observable LR prior is quantised. Optimizer
    learnability is covered separately by gradient/unit tests; it must not add
    minutes to every experiment startup.
    """
    lr = 24
    hr = lr * 4
    max_distance = 24.0
    yy = torch.arange(hr, device=device, dtype=torch.float32) + 0.5
    xx = torch.arange(hr, device=device, dtype=torch.float32) + 0.5
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    angle = math.radians(3.0)
    lnx, lny = -math.sin(angle), math.cos(angle)
    line = lnx * (gx - hr * 0.5) + lny * (gy - hr * 0.5)
    cx, cy, radius = hr * 0.52, hr * 0.48, 27.0
    circle_radius = torch.sqrt((gx - cx).square() + (gy - cy).square() + 1.0e-8)
    circle = circle_radius - radius
    target = torch.stack((line, circle), dim=0).unsqueeze(1)

    p = (torch.arange(lr, device=device, dtype=torch.float32) + 0.5) * 4.0
    py, px = torch.meshgrid(p, p, indexing="ij")
    line_d = lnx * (px - hr * 0.5) + lny * (py - hr * 0.5)
    cr = torch.sqrt((px - cx).square() + (py - cy).square() + 1.0e-8)
    circle_d = cr - radius
    source_pixels = torch.stack((line_d, circle_d), dim=0).unsqueeze(1)
    quantized_source_pixels = torch.round(source_pixels / 4.0) * 4.0
    source = (quantized_source_pixels / max_distance).clamp(-1.0, 1.0)

    d = source_pixels
    nx = torch.stack((
        torch.full_like(line_d, lnx),
        (px - cx) / cr,
    ), dim=0).unsqueeze(1)
    ny = torch.stack((
        torch.full_like(line_d, lny),
        (py - cy) / cr,
    ), dim=0).unsqueeze(1)
    curvature = torch.stack((
        torch.zeros_like(line_d),
        1.0 / cr.clamp_min(1.0),
    ), dim=0).unsqueeze(1)
    zeros = torch.zeros_like(d)
    ones = torch.ones_like(d)
    branch_d = d.repeat(1, 3, 1, 1)
    branch_nx = nx.repeat(1, 3, 1, 1)
    branch_ny = ny.repeat(1, 3, 1, 1)
    branch_k = curvature.repeat(1, 3, 1, 1)
    branch_half = torch.full_like(branch_d, 2.0)
    branch_mode = torch.zeros_like(branch_d)  # ordinary signed boundary
    branch_activation = torch.cat((ones, zeros, zeros), dim=1)
    csg_logits = torch.cat((
        torch.full_like(d, 12.0), torch.full_like(d, -12.0), torch.full_like(d, -12.0)
    ), dim=1)
    context = {
        "source_sdf_prior_lr": source,
        "branch_anchor_distance_pixels": branch_d,
        "branch_normal_x": branch_nx,
        "branch_normal_y": branch_ny,
        "branch_curvature_per_pixel": branch_k,
        "branch_half_width_pixels": branch_half,
        "branch_ribbon_mode": branch_mode,
        "branch_activation": branch_activation,
        "csg_logits": csg_logits,
        "confidence": ones,
        "anchor_distance_pixels": d,
        "normal_x": nx,
        "normal_y": ny,
        "curvature_per_pixel": curvature,
        "ribbon_half_width_pixels": torch.full_like(d, 2.0),
        "ribbon_mode": zeros,
        "distance_delta_pixels": d - quantized_source_pixels,
        "junction_hint": zeros,
    }
    decoder = LocalParametricBoundaryDecoder(
        8, 48, max_distance_pixels=max_distance,
        max_offset_pixels=6.0, max_normal_correction=1.5,
        max_curvature_per_pixel=0.35, max_ribbon_half_width_pixels=6.0,
        control_scale=1, output_scale=4,
    ).to(device)
    grid = make_query_grid(2, hr, hr, device=device)
    with torch.no_grad():
        pred = decoder.query(context, grid)["phi_pixels"]
        source_hr = F.interpolate(source, size=(hr, hr), mode="bilinear", align_corners=False) * max_distance
        band = (target.abs() <= 6.0).float()
        before_t = ((source_hr - target).abs() * band).sum() / band.sum().clamp_min(1.0)
        after_t = ((pred - target).abs() * band).sum() / band.sum().clamp_min(1.0)

    before = float(before_t.cpu())
    after = float(after_t.cpu())
    line_jitter = line_perpendicular_jitter_pixels(
        pred[0, 0].cpu().numpy(), target[0, 0].cpu().numpy()
    )
    circle_rough = circle_radial_roughness_pixels(
        pred[1, 0].cpu().numpy(), target[1, 0].cpu().numpy()
    )
    # The contour extractor itself contributes about 0.04 px radial sampling
    # variance on a perfect 96 px circle, so the proof threshold is above that
    # numerical floor while still far below visible staircase levels.
    if (
        not math.isfinite(after) or after > 0.02 or after >= before * 0.08
        or not math.isfinite(line_jitter) or line_jitter > 0.05
        or not math.isfinite(circle_rough) or circle_rough > 0.08
    ):
        raise RuntimeError(
            "V9.9.3 parametric structure representation proof failed: "
            f"MAE {before:.6f}->{after:.6f}px, lineJitter={line_jitter:.6f}px, "
            f"curveRough={circle_rough:.6f}px"
        )
    return before, after, float(line_jitter), float(circle_rough)


# Backward-compatible internal name for regression tests. It intentionally
# reuses the fast representation proof rather than running a second optimizer.
def _direct_implicit_structure_microproof(device: torch.device) -> tuple[float, float]:
    before, after, _line, _curve = _parametric_structure_microproof(device)
    return before, after


def _parametric_junction_microproof(device: torch.device) -> tuple[float, float]:
    """Fast CSG/ribbon proof for a multi-branch line junction.

    Three analytic ribbons are unioned at one crossing. This verifies the
    representation needed at the local core of Y/T junctions without doing a
    training loop during startup.
    """
    lr = 24
    hr = lr * 4
    max_distance = 24.0
    half_width = 2.0
    cx = cy = hr * 0.5
    yy = torch.arange(hr, device=device, dtype=torch.float32) + 0.5
    xx = torch.arange(hr, device=device, dtype=torch.float32) + 0.5
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    p = (torch.arange(lr, device=device, dtype=torch.float32) + 0.5) * 4.0
    py, px = torch.meshgrid(p, p, indexing="ij")

    directions = (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0)
    hr_surfaces = []
    control_surfaces = []
    normals_x = []
    normals_y = []
    for theta in directions:
        nx, ny = -math.sin(theta), math.cos(theta)
        hr_surfaces.append((nx * (gx - cx) + ny * (gy - cy)).abs() - half_width)
        control_surfaces.append(nx * (px - cx) + ny * (py - cy))
        normals_x.append(torch.full_like(px, nx))
        normals_y.append(torch.full_like(py, ny))
    target = torch.minimum(torch.minimum(hr_surfaces[0], hr_surfaces[1]), hr_surfaces[2]).unsqueeze(0).unsqueeze(0)
    source_control = torch.minimum(
        torch.minimum(control_surfaces[0].abs() - half_width, control_surfaces[1].abs() - half_width),
        control_surfaces[2].abs() - half_width,
    ).unsqueeze(0).unsqueeze(0)
    quantized_source = torch.round(source_control / 4.0) * 4.0
    source = (quantized_source / max_distance).clamp(-1.0, 1.0)

    branch_d = torch.stack(control_surfaces, dim=0).unsqueeze(0)
    branch_nx = torch.stack(normals_x, dim=0).unsqueeze(0)
    branch_ny = torch.stack(normals_y, dim=0).unsqueeze(0)
    branch_zero = torch.zeros_like(branch_d)
    branch_half = torch.full_like(branch_d, half_width)
    branch_mode = torch.ones_like(branch_d)
    branch_activation = torch.ones_like(branch_d)
    single = torch.full_like(source_control, -12.0)
    union = torch.full_like(source_control, 12.0)
    inter = torch.full_like(source_control, -12.0)
    context = {
        "source_sdf_prior_lr": source,
        "branch_anchor_distance_pixels": branch_d,
        "branch_normal_x": branch_nx,
        "branch_normal_y": branch_ny,
        "branch_curvature_per_pixel": branch_zero,
        "branch_half_width_pixels": branch_half,
        "branch_ribbon_mode": branch_mode,
        "branch_activation": branch_activation,
        "csg_logits": torch.cat((single, union, inter), dim=1),
        "confidence": torch.ones_like(source_control),
        "anchor_distance_pixels": branch_d[:, 0:1],
        "normal_x": branch_nx[:, 0:1],
        "normal_y": branch_ny[:, 0:1],
        "curvature_per_pixel": branch_zero[:, 0:1],
        "ribbon_half_width_pixels": branch_half[:, 0:1],
        "ribbon_mode": branch_mode[:, 0:1],
        "distance_delta_pixels": branch_d[:, 0:1] - quantized_source,
        "junction_hint": torch.ones_like(source_control),
    }
    decoder = LocalParametricBoundaryDecoder(
        8, 48, max_distance_pixels=max_distance,
        max_ribbon_half_width_pixels=6.0, control_scale=1, output_scale=4,
    ).to(device)
    grid = make_query_grid(1, hr, hr, device=device)
    with torch.no_grad():
        pred = decoder.query(context, grid)["phi_pixels"][0, 0].cpu().numpy()
    target_np = target[0, 0].cpu().numpy()
    topology = float(sdf_topology_mismatch(pred, target_np))
    chamfer = float(zero_contour_distance(pred, target_np)["chamferPixels"])
    if topology != 0.0 or not math.isfinite(chamfer) or chamfer > 0.16:
        raise RuntimeError(
            f"V9.9.3 parametric junction representation proof failed: topology={topology:.1f}, chamfer={chamfer:.6f}px"
        )
    return topology, chamfer


def _phase_seam_sr_microproof(device: torch.device, config: V9Config) -> tuple[float, float, float]:
    """Fail-fast capacity proof for the exact B3 4x phase residual path.

    A shallow diagonal manufactured seam is analytically authored at 4x,
    integrated to LR, then reconstructed only by PhaseAwareSeamSR. The test is
    deliberately tiny: if this direct path cannot recover most missing seam
    energy in well under a second on CPU, a real Raven B3 run is not allowed to
    consume minutes of training time.
    """
    lr = 16
    hr = lr * 4
    yy, xx = torch.meshgrid(
        torch.arange(hr, device=device, dtype=torch.float32),
        torch.arange(hr, device=device, dtype=torch.float32),
        indexing="ij",
    )
    slope = 0.55
    phi = (yy - (slope * xx + 13.4)) / math.sqrt(1.0 + slope * slope)
    base = 0.45 + 0.06 * (xx / float(hr)) + 0.03 * (yy / float(hr))
    seam = -0.22 * torch.exp(-(phi / 0.85).square())
    seam = seam + 0.07 * torch.exp(-((phi.abs() - 1.8) / 0.75).square())
    target_albedo = (base + seam).clamp(0.0, 1.0).view(1, 1, hr, hr).repeat(1, 3, 1, 1)
    target_normal = torch.zeros((1, 2, hr, hr), device=device, dtype=torch.float32)
    target_normal[:, 0:1] = (
        -0.12 * torch.sign(phi) * torch.exp(-(phi.abs() / 1.7).square())
    ).view(1, 1, hr, hr)
    target_material = torch.stack((base, torch.full_like(base, 0.35), torch.full_like(base, 0.60)), dim=0).unsqueeze(0).clamp(0.0, 1.0)
    source_albedo = F.interpolate(target_albedo, size=(lr, lr), mode="area")
    source_normal = F.interpolate(target_normal, size=(lr, lr), mode="area")
    source_material = F.interpolate(target_material, size=(lr, lr), mode="area")
    baseline_albedo = F.interpolate(source_albedo, size=(hr, hr), mode="bicubic", align_corners=False, antialias=True)
    normal_vec = torch.tensor([-slope, 1.0], device=device, dtype=torch.float32)
    normal_vec = normal_vec / normal_vec.norm().clamp_min(1.0e-6)
    geometry_normal = normal_vec.view(1, 2, 1, 1).expand(1, 2, hr, hr)
    sdf_pixels = phi.view(1, 1, hr, hr)
    edge = torch.exp(-(phi.abs() / 2.0).square()).view(1, 1, hr, hr)
    weight = 0.05 + 1.95 * edge

    devices = [] if device.type != "cuda" else [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(1086)
        module = PhaseAwareSeamSR(
            hidden=int(getattr(config, "seam_phase_sr_channels", 32)),
            max_delta=float(getattr(config, "seam_phase_sr_max_delta", 0.40)),
        ).to(device)
        optimizer = torch.optim.AdamW(module.parameters(), lr=3.0e-3, weight_decay=0.0)
        with torch.no_grad():
            before = float(((baseline_albedo - target_albedo).abs() * weight).mean().item())
        steps = int(getattr(config, "seam_microproof_steps", 80))
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            delta = module(
                source_albedo, source_normal, source_material, geometry_normal, sdf_pixels, edge, hr_size=(hr, hr)
            )
            prediction = (baseline_albedo + delta[:, 0:3]).clamp(0.0, 1.0)
            target_delta = target_albedo - baseline_albedo
            loss = ((prediction - target_albedo).abs() * weight).mean()
            loss = loss + 0.20 * ((delta[:, 0:3] - target_delta).abs() * weight).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            delta = module(
                source_albedo, source_normal, source_material, geometry_normal, sdf_pixels, edge, hr_size=(hr, hr)
            )
            prediction = (baseline_albedo + delta[:, 0:3]).clamp(0.0, 1.0)
            after = float(((prediction - target_albedo).abs() * weight).mean().item())
            recovery = float((before - after) / max(before, 1.0e-8))
    required = float(getattr(config, "seam_microproof_recovery_required", 0.80))
    if not math.isfinite(recovery) or recovery < required:
        raise RuntimeError(
            f"V10.8.8 phase-SR seam microproof failed: MAE {before:.6f} -> {after:.6f}, "
            f"recovery={recovery:+.1%} required={required:.1%}"
        )
    return before, after, recovery


def _profile_specialist_microproof(device: torch.device) -> tuple[float, float]:
    """Verify that the local specialist can deliberately overfit sharp profiles.

    This runs before the expensive experiment optimizer is created. It exercises
    the direct coverage-logit authority on one shallow line and one curved edge
    and fails fast if a future refactor makes Panel-2-like coverage unreachable.
    """
    h = w = 32
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    line_phi = (yy - (13.25 + 0.10 * xx)).unsqueeze(0).unsqueeze(0)
    circle_phi = (torch.sqrt((xx - 16.3).square() + (yy - 15.7).square() + 1.0e-6) - 8.4).unsqueeze(0).unsqueeze(0)
    phi = torch.cat((line_phi, circle_phi), dim=0)

    def coverage_from_phi(value: torch.Tensor, width: float) -> torch.Tensor:
        t = (0.5 - value / float(width)).clamp(0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    initial = coverage_from_phi(phi, 1.80)
    target = coverage_from_phi(phi, 0.70)
    gy, gx = torch.gradient(phi[:, 0], dim=(-2, -1))
    norm = torch.sqrt(gx.square() + gy.square() + 1.0e-6)
    nx = (gx / norm).unsqueeze(1)
    ny = (gy / norm).unsqueeze(1)
    gray = initial
    pg_y, pg_x = torch.gradient(gray[:, 0], dim=(-2, -1))
    gx_img = pg_x.unsqueeze(1)
    gy_img = pg_y.unsqueeze(1)
    rgb = gray.repeat(1, 3, 1, 1)
    ones = torch.ones_like(initial)
    curvature = torch.zeros_like(initial)
    features = torch.cat((
        rgb, rgb, (phi / 8.0).clamp(-1.0, 1.0), nx, ny, curvature,
        initial, ones, ones, ones,
        (gx_img * 4.0).clamp(-1.0, 1.0),
        (gy_img * 4.0).clamp(-1.0, 1.0),
        (gx_img * 4.0).clamp(-1.0, 1.0),
        (gy_img * 4.0).clamp(-1.0, 1.0),
    ), dim=1)
    band = torch.exp(-phi.abs() / 5.0)

    devices = [] if device.type != "cuda" else [device.index if device.index is not None else 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(9_911)
        specialist = BoundaryProfileSpecialist(
            in_channels=18, channels=16, max_logit_delta=16.0
        ).to(device)
        optimizer = torch.optim.AdamW(specialist.parameters(), lr=8.0e-3, weight_decay=0.0)
        with torch.no_grad():
            before = float((initial - target).abs().mean().item())
        for _ in range(36):
            optimizer.zero_grad(set_to_none=True)
            pred = specialist(features, initial, band)["coverage"]
            loss = (pred - target).abs().mean()
            pgx, pgy = torch.gradient(pred[:, 0], dim=(-2, -1))
            tgx, tgy = torch.gradient(target[:, 0], dim=(-2, -1))
            loss = loss + 0.35 * ((pgx - tgx).abs().mean() + (pgy - tgy).abs().mean())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            after = float((specialist(features, initial, band)["coverage"] - target).abs().mean().item())
    if not math.isfinite(after) or after >= before * 0.35:
        raise RuntimeError(
            f"V10 profile-specialist microproof failed: coverage MAE {before:.6f} -> {after:.6f}"
        )
    return before, after


def _structure_selection_rank(
    *,
    qualified: bool,
    topology_regression: float,
    predicted_missing: float,
    source_missing: float,
    line_jitter: float,
    curve_roughness: float,
    line_limit: float,
    curve_limit: float,
    contour_gain: float,
    contour_chamfer: float,
    profile_teacher_recovery: float,
) -> tuple[float, ...]:
    """Lexicographic V10.7.9 structural checkpoint order.

    Topology/connectivity safety dominates smoothness; smoothness dominates mean
    Chamfer gain. This prevents a high-gain but folded/jagged structure from
    defeating a safer candidate through one weighted scalar score.
    """
    missing_excess = max(0.0, float(predicted_missing) - float(source_missing))
    return (
        0.0 if qualified else 1.0,
        0.0 if float(topology_regression) == 0.0 else 1.0,
        float(topology_regression),
        missing_excess,
        max(0.0, float(line_jitter) - float(line_limit)),
        max(0.0, float(curve_roughness) - float(curve_limit)),
        -float(contour_gain),
        float(contour_chamfer),
        -float(profile_teacher_recovery),
    )


def train_v9(
    config: V9Config,
    repo_root: Path,
    requested_device: str | None = None,
    *,
    resume: bool = False,
    restart: bool = False,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
    stop_after_phase: str | None = None,
) -> dict[str, Any]:
    config.validate()
    _status("[startup] Trainer entered; configuration validated.")
    output_dir = (repo_root / config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / config.checkpoint_name
    metadata_path = output_dir / config.metadata_name
    state_path = output_dir / config.training_state_name
    best_path = output_dir / "nsamdr_v9_best_final_phase.pt"
    best_structure_path = output_dir / "nsamdr_v9_best_structure.pt"
    best_b1a_path = output_dir / "best_b1a_topology.pt"
    best_b1b_path = output_dir / "best_b1b_geometry.pt"
    best_b1b_classifier_path = output_dir / "best_b1b_classifier.pt"
    best_b1b_parameters_path = output_dir / "best_b1b_parameters.pt"
    best_b2_path = output_dir / "best_b2_redraw.pt"
    best_b3_path = output_dir / "best_b3_seam.pt"
    best_b4_path = output_dir / "best_b4_authority.pt"
    if restart:
        _status(f"[startup] Restart requested; preparing prior-state backup in {output_dir}...")
        cleanup_started = time.perf_counter()

        restart_sources = [
            path for path in (checkpoint_path, metadata_path, state_path, best_path, best_structure_path, best_b1a_path, best_b1b_path, best_b1b_classifier_path, best_b1b_parameters_path, best_b2_path, best_b3_path, best_b4_path)
            if path.is_file()
        ]
        if restart_sources:
            backup_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = output_dir / "restart_backups" / backup_stamp
            backup_dir.mkdir(parents=True, exist_ok=False)
            for path in restart_sources:
                shutil.copy2(path, backup_dir / path.name)
            _status(
                f"[startup] Restart backup created: {backup_dir} "
                f"({len(restart_sources)} file(s))"
            )

        _status(f"[startup] Restart confirmed; clearing active prior state in {output_dir}...")
        for path in (checkpoint_path, metadata_path, state_path, best_path, best_structure_path, best_b1a_path, best_b1b_path, best_b1b_classifier_path, best_b1b_parameters_path, best_b2_path, best_b3_path, best_b4_path):
            path.unlink(missing_ok=True)
        # A semantic/model restart must not leave validation sheets or forensic
        # dumps from an incompatible earlier schema mixed into the new run.
        shutil.rmtree(output_dir / "samples", ignore_errors=True)
        shutil.rmtree(output_dir / config.diagnostics_dir_name, ignore_errors=True)
        _status(
            f"[startup] Restart cleanup complete in "
            f"{time.perf_counter() - cleanup_started:.1f}s."
        )

    _status("[startup] Resolving CUDA device and runtime settings...")
    device = resolve_device(config, requested_device)
    _configure_cuda(config, device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    _status("[startup] Loading V9 dataset manifest...")
    manifest = load_dataset_manifest(repo_root, config)
    fingerprint = dataset_fingerprint(manifest, config)
    _status("[startup] Dataset manifest loaded; constructing training/validation datasets...")
    train_dataset = PhysicalTileDatasetV9(
        manifest, config, "train", config.tiles_per_epoch, seed=config.seed)
    validation_dataset = PhysicalTileDatasetV9(
        manifest, config, "validation", config.validation_tiles, seed=config.seed + 77)
    # V10 checkpoint selection always sees the full permanent 29-class
    # geometry ladder. EXP_0002/V9.8.12 briefly passed a 12-tile selector at
    # epoch 6 but failed the final 29-case proof; that selector/audit mismatch is
    # no longer allowed to choose a checkpoint.
    synthetic_validation_count = max(29, int(config.sdf_synthetic_validation_tiles))
    synthetic_validation_dataset = SyntheticGeometryValidationDataset(
        config, synthetic_validation_count, seed=config.seed + 9_911
    )

    workers = config.data_loader_workers if device.type == "cuda" else 0
    validation_workers = min(workers, 1)
    _status(
        f"[startup] Constructing DataLoaders: train workers={workers}, "
        f"validation workers={validation_workers}, prefetch={config.data_loader_prefetch_factor}..."
    )
    train_loader = _build_loader(
        train_dataset,
        batch_size=config.batch_size,
        device=device,
        workers=workers,
        prefetch_factor=config.data_loader_prefetch_factor,
        persistent_workers=config.data_loader_persistent_workers,
        rolling_epoch_indices=True,
    )
    # V10.8.0 downstream appearance learning is deliberately real-Raven only.
    # Structural synthetic primitives remain useful for B1, but seam/detail/
    # selector stages must see the authored Raven crops they are expected to
    # improve in the instrumented preview.
    downstream_config = copy.deepcopy(config)
    downstream_config.synthetic_geometry_probability = 0.0
    downstream_train_dataset = PhysicalTileDatasetV9(
        manifest, downstream_config, "train",
        int(getattr(config, "raven_downstream_tiles_per_epoch", 24)),
        seed=config.seed + 80_813,
    )
    downstream_train_loader = _build_loader(
        downstream_train_dataset,
        batch_size=config.batch_size, device=device, workers=workers,
        prefetch_factor=config.data_loader_prefetch_factor,
        persistent_workers=config.data_loader_persistent_workers,
        rolling_epoch_indices=True,
    )
    # V10.8.4 B3 recovery: the former Quick curriculum exposed only four Raven
    # crops to the seam representation before demanding 70% forced-authority
    # recovery.  That gave the zero-initialised phase-SR head only four optimizer
    # steps, which is not a meaningful capacity proof.  B3 gets a dedicated
    # 12-sample real-Raven loader; later smoke stages keep their smaller budget.
    seam_proof_train_dataset = PhysicalTileDatasetV9(
        manifest, downstream_config, "train",
        max(12, int(getattr(config, "raven_downstream_tiles_per_epoch", 24))),
        seed=config.seed + 91_173,
    )
    seam_proof_train_loader = _build_loader(
        seam_proof_train_dataset,
        batch_size=config.batch_size, device=device, workers=workers,
        prefetch_factor=config.data_loader_prefetch_factor,
        persistent_workers=config.data_loader_persistent_workers,
        rolling_epoch_indices=True,
    )

    # V10.7.9 B1b has its own fixed complete-teacher bank. The same cases are
    # revisited across epochs and trained in micro-batches because only the
    # compact primitive field is active; this is intentionally independent of
    # the batch-size=1 Raven/full-model memory setting.
    parametric_train_dataset = ParametricPrimitiveTrainingDataset(
        config, int(getattr(config, "parametric_primitive_train_tiles_per_epoch", 512)),
        seed=config.seed + 71_337,
    )
    parametric_train_loader = _build_loader(
        parametric_train_dataset,
        batch_size=int(getattr(config, "parametric_primitive_batch_size", 8)),
        device=device,
        workers=workers,
        prefetch_factor=config.data_loader_prefetch_factor,
        persistent_workers=config.data_loader_persistent_workers,
        rolling_epoch_indices=False,
    )
    validation_loader = _build_loader(
        validation_dataset,
        batch_size=1,
        device=device,
        workers=validation_workers,
        prefetch_factor=config.data_loader_prefetch_factor,
        persistent_workers=config.data_loader_persistent_workers,
    )
    synthetic_validation_loader = _build_loader(
        synthetic_validation_dataset,
        batch_size=1,
        device=device,
        workers=0,
        prefetch_factor=config.data_loader_prefetch_factor,
        persistent_workers=False,
    )

    _status(
        f"[startup] DataLoaders constructed; fixed synthetic SDF validation="
        f"{synthetic_validation_count} tiles (full proof ladder minimum)."
    )

    _status("[startup] Allocating V10.7.9 explicit-parametric deterministic-redraw model on training device...")
    model_started = time.perf_counter()
    model = FidelityResidualNetV9(config).to(device)
    contract = model.architecture_contract()
    _validate_v992_architecture_contract(contract)
    structure_before, structure_after, line_jitter, curve_rough = _explicit_primitive_structure_microproof(device, config)
    _status(
        f"[startup] Explicit primitive representation proof: {structure_before:.6f} -> {structure_after:.6f} px MAE; "
        f"lineJitter={line_jitter:.4f}px curveRough={curve_rough:.4f}px."
    )
    micro_before, micro_after = _profile_specialist_microproof(device)
    _status(
        f"[startup] Direct coverage specialist microproof: {micro_before:.6f} -> {micro_after:.6f} MAE."
    )
    if bool(getattr(config, "seam_microproof_enabled", True)) and bool(getattr(config, "raven_full_pipeline_preview_enabled", False)):
        seam_before, seam_after, seam_recovery = _phase_seam_sr_microproof(device, config)
        _status(
            f"[startup] V10.8.8 B3 phase-SR microproof: {seam_before:.6f} -> {seam_after:.6f} MAE; "
            f"recovery={seam_recovery:+.1%} PASS."
        )
    _status(
        f"[startup] Model allocated in {time.perf_counter() - model_started:.1f}s; "
        f"{parameter_count(model):,} parameters."
    )
    if device.type == "cuda" and config.channels_last:
        model = _convert_model_channels_last(model)
    _status("[startup] Building optimizer and AMP state...")
    optimizer_started = time.perf_counter()
    optimizer, optimizer_mode = _build_optimizer(model, config, device)
    amp_dtype = _resolve_amp_dtype(config, device)
    use_amp = device.type == "cuda"
    use_scaler = use_amp and amp_dtype == torch.float16
    memory_governor = _ReactiveCudaMemoryGovernor(device, config)
    # V10.9.1.1: keep the existing reactive VRAM governor active globally.
    # The cached Raven path removes frozen GeometryNet repetition, but fresh or
    # resumed runs can still pass through full structural stages before B3.
    # Disabling the governor here made those 17M-parameter stages lose their
    # established activation-offload/OOM safety path. The governor may still
    # choose GPU-resident mode naturally when the cached appearance steps fit.
    effective_cuda_prefetch = bool(
        config.cuda_prefetch and use_amp and not memory_governor.enabled
    )
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_scaler, init_scale=config.amp_initial_scale
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_scaler, init_scale=config.amp_initial_scale
        )

    _status(
        f"[startup] Optimizer/AMP state ready in "
        f"{time.perf_counter() - optimizer_started:.1f}s."
    )
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_sdf_score = float("inf")
    best_structure_score = float("inf")
    best_structure_rank: tuple[float, ...] = (float("inf"),) * 9
    best_render_mae = float("inf")
    best_seam_recovery = -float("inf")
    best_seam_iou = -float("inf")
    best_detail_rank: tuple[float, ...] = (float("inf"),) * 6
    best_selector_rank: tuple[float, ...] = (float("inf"),) * 4
    topology_bootstrapped = False  # B1a topology
    b1b_classifier_qualified = False
    b1b_parameters_qualified = False
    b1b_stage_epoch = 0
    best_classifier_accuracy = -float("inf")
    best_teacher_param_mae = float("inf")
    structure_qualified = False    # B1b integrated explicit parametric primitive
    render_qualified = False       # B2 same deterministic redraw
    seam_reconstruction_qualified = False  # B3 forced GT authority
    seam_authority_qualified = False       # B4 learned detector
    detail_qualified = False
    architecture_participation: list[dict[str, Any]] = []
    cache_equivalence: dict[str, Any] = {
        "passed": True,
        "cacheUsed": False,
        "compared": False,
        "reason": "canonical training is uncached",
    }
    optimizer_state_migrations = 0
    if resume:
        if not state_path.is_file():
            raise RuntimeError(f"V9 resume requested but training state is missing: {state_path}")
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("schema") != MODEL_SCHEMA:
            raise RuntimeError("V9 resume state schema mismatch")
        if state.get("config_hash") != _config_hash(config) or state.get("dataset_fingerprint") != fingerprint:
            raise RuntimeError("V9 resume state does not match the current semantic config or dataset")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        _restore_optimizer_implementation(optimizer, optimizer_mode)
        optimizer_state_migrations = _align_optimizer_state_to_parameters(optimizer)
        scaler.load_state_dict(state.get("scaler", {}))
        _restore_rng(state["rng"])
        history = list(state.get("history", []))
        best_validation = float(state.get("best_validation", float("inf")))
        best_sdf_score = float(state.get("best_sdf_score", float("inf")))
        best_structure_score = float(state.get("best_structure_score", float("inf")))
        best_render_mae = float(state.get("best_render_mae", float("inf")))
        best_seam_recovery = float(state.get("best_seam_recovery", -float("inf")))
        best_seam_iou = float(state.get("best_seam_iou", -float("inf")))
        stored_rank = state.get("best_structure_rank")
        if isinstance(stored_rank, (list, tuple)) and len(stored_rank) == 9:
            best_structure_rank = tuple(float(v) for v in stored_rank)
        topology_bootstrapped = bool(state.get("topology_bootstrapped", False))
        b1b_classifier_qualified = bool(state.get("b1b_classifier_qualified", False))
        b1b_parameters_qualified = bool(state.get("b1b_parameters_qualified", False))
        b1b_stage_epoch = int(state.get("b1b_stage_epoch", 0))
        best_classifier_accuracy = float(state.get("best_classifier_accuracy", -float("inf")))
        best_teacher_param_mae = float(state.get("best_teacher_param_mae", float("inf")))
        structure_qualified = bool(state.get("structure_qualified", False))
        render_qualified = bool(state.get("render_qualified", False))
        seam_reconstruction_qualified = bool(state.get("seam_reconstruction_qualified", False))
        seam_authority_qualified = bool(state.get("seam_authority_qualified", False))
        detail_qualified = bool(state.get("detail_qualified", False))
        stored_participation = state.get("architecture_participation", [])
        if isinstance(stored_participation, list):
            architecture_participation = list(stored_participation)
        stored_detail_rank = state.get("best_detail_rank")
        if isinstance(stored_detail_rank, (list, tuple)) and len(stored_detail_rank) == 6:
            best_detail_rank = tuple(float(v) for v in stored_detail_rank)
        stored_selector_rank = state.get("best_selector_rank")
        if isinstance(stored_selector_rank, (list, tuple)) and len(stored_selector_rank) == 4:
            best_selector_rank = tuple(float(v) for v in stored_selector_rank)
        start_epoch = int(state["completed_epoch"]) + 1

    participation_stage_phase: str | None = None
    participation_stage_start: dict[str, torch.Tensor] = {}

    _status("[startup] Preparing execution model...")
    execution_model, compile_status = _compile_training_model(
        model, config.torch_compile_mode if device.type == "cuda" else "off"
    )
    _status("[startup] Initialization complete; entering epoch loop.")

    precision_name = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }[amp_dtype]
    _status("=" * 68)
    _status("NSAMDR V10.7.9 STAGED PARAMETRIC PRIMITIVE REDRAW")
    _status("=" * 68)
    _status(f"Device                   : {device}")
    _status(f"Architecture schema      : {MODEL_SCHEMA}")
    _status("Model                    : learned parametric geometry -> analytic SDF -> boundary/profile -> phase seam -> detail -> selector")
    _status(f"Tile geometry            : {config.tile_size} LR -> {config.tile_size * 4} HR")
    _status(f"Widths                   : {list(config.widths)}")
    _status(f"Parameters               : {parameter_count(model):,}")
    _status("Geometry outputs         : shared movable contour nodes + tangent lines -> cubic Hermite metric SDF + edge/orientation/hardness")
    _status("Geometry RGB authority   : NONE")
    _status(f"Detail reconstruction    : {bool(getattr(config, 'detail_reconstruction_enabled', True))} (explicit 2x/4x decoder)")
    _status("Boundary primitive       : staged oracle/predicted SDF + two-sided sub-pixel coverage")
    _status("Boundary sampling        : best-of-%d at %.0f%% probability" % (
        config.boundary_candidate_count, config.boundary_sampling_probability * 100.0))
    _status(f"Exact geometry training   : {config.synthetic_geometry_probability * 100.0:.0f}% of training tiles")
    _status("Contour objective         : B1a topology -> B1b LR-only inverse geometry fit -> B2 same-renderer redraw")
    if config.loss_precision == "mixed" and use_amp:
        _status("Loss numerics            : reduced-precision kernels, FP32 scalar reductions")
    else:
        _status("Loss numerics            : complete objective in FP32")
    _status(f"Training state           : {state_path}")
    _status(f"Performance profile      : {config.performance_profile}")
    _status(f"AMP compute precision    : {precision_name}")
    _status(f"Convolution layout       : {'channels-last' if config.channels_last and use_amp else 'contiguous'}")
    _status(f"Optimizer implementation : {optimizer_mode}")
    _status(f"Learning-rate scheduler  : {config.scheduler_name}")
    _status(f"torch.compile            : {compile_status}")
    if resume:
        _status(f"Optimizer state migration: {optimizer_state_migrations} moment tensor(s) aligned")
    _status(f"Data pipeline            : workers={workers} prefetch={config.data_loader_prefetch_factor} persistent={workers > 0 and config.data_loader_persistent_workers}")
    _status(
        "Training sample schedule : B1b cycles fixed analytic bank; other phases use deterministic epoch blocks"
    )
    _status(f"CUDA batch prefetch      : {effective_cuda_prefetch}")
    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024.0 ** 3)
        if memory_governor.enabled:
            _status("CUDA memory policy      : elastic/reactive sharing")
            _status(
                f"VRAM target / pause      : "
                f"{config.reactive_vram_target_free_fraction * 100.0:.0f}% / "
                f"{config.reactive_vram_pause_free_fraction * 100.0:.0f}% free"
            )
            _status(
                "VRAM pressure response    : GPU activations -> CPU-saved "
                "activations -> yield between batches"
            )
            _status(
                f"Foreground burst reserve : "
                f"{config.reactive_vram_burst_reserve_fraction * 100.0:.0f}% "
                f"({memory_governor.burst_reserve_bytes / (1024.0 ** 3):.2f} GiB)"
            )
            _status(
                f"Pre-batch stability gate : "
                f"{config.reactive_vram_stability_samples} samples x "
                f"{config.reactive_vram_stability_interval_seconds:.2f}s"
            )
            _status(
                "Allocator ceiling         : disabled (physical-VRAM safety gate only)"
            )
            _status(
                f"VRAM expansion policy    : restore GPU activations after "
                f"{config.reactive_vram_expand_stable_steps} stable recovery steps"
            )
            host_memory = _host_memory_info()
            if host_memory is not None:
                host_free, host_total = host_memory
                _status(
                    f"Host RAM offload guard   : "
                    f"pause below {config.reactive_host_pause_free_fraction * 100.0:.0f}% free; "
                    f"resume above {config.reactive_host_resume_free_fraction * 100.0:.0f}% "
                    f"(now {host_free / (1024.0 ** 3):.1f}/{host_total / (1024.0 ** 3):.1f} GiB free)"
                )
        else:
            _status(f"CUDA memory policy      : non-reactive ({total_vram:.1f} GiB device)")
    _status(f"cuDNN benchmark / TF32   : {config.cudnn_benchmark} / {config.allow_tf32}")
    if resume:
        _status(f"Resume epoch             : {start_epoch:03d}/{config.total_epochs:03d}")
    _status("=" * 68)

    started = time.perf_counter()
    global_step = sum(int(item.get("batches", 0)) for item in history)
    previous_finetune_validation = [
        float(item.get("validation", {}).get("total"))
        for item in history
        if item.get("phase") == "physical-finetune"
        and item.get("validation", {}).get("total") is not None
    ]
    early_stop_best = min(previous_finetune_validation, default=math.inf)
    early_stop_stale = 0
    early_stopped_epoch: int | None = None
    valid_stop_phases = {
        None,
        "sdf-bootstrap",
        "sdf-proof",
        "seam-proof",
        "seam-authority",
        "gate-proof",
        "detail-reconstruction",
        "boundary-hardening",
        "physical-finetune",
    }
    if stop_after_phase not in valid_stop_phases:
        raise ValueError(f"unsupported stop_after_phase={stop_after_phase!r}")
    staged_stop_epoch = None
    if stop_after_phase is not None:
        phase_ends = {
            "sdf-bootstrap": config.identity_epochs,
            "sdf-proof": config.identity_epochs + config.residual_epochs,
            "seam-proof": config.identity_epochs + config.residual_epochs + config.seam_proof_epochs,
            "seam-authority": config.identity_epochs + config.residual_epochs + config.seam_proof_epochs + config.seam_authority_epochs,
            "gate-proof": config.identity_epochs + config.residual_epochs + config.seam_proof_epochs + config.seam_authority_epochs + config.boundary_epochs,
            "detail-reconstruction": config.identity_epochs + config.residual_epochs + config.seam_proof_epochs + config.seam_authority_epochs + config.boundary_epochs + config.detail_epochs,
            "boundary-hardening": config.total_epochs,
            "physical-finetune": config.total_epochs,
        }
        staged_stop_epoch = int(phase_ends[stop_after_phase])
        _status(
            f"Staged training stop     : after {stop_after_phase} "
            f"(epoch {staged_stop_epoch:03d})"
        )
    if early_stop_patience is not None and early_stop_patience > 0:
        _status(
            f"Tuning early stop        : physical-finetune only; patience={early_stop_patience} "
            f"minDelta={early_stop_min_delta:.6f}"
        )
    for epoch in range(start_epoch, config.total_epochs + 1):
        planned_phase = _phase_for_epoch(epoch, config)
        phase = planned_phase
        # V10.7.9 structural proof: B1a topology is frozen before B1b. B2 is
        # evaluation-only and uses the exact same deterministic renderer as Panel 2.
        if planned_phase == "sdf-proof" and not topology_bootstrapped:
            _status("  REJECTED_B1A_TOPOLOGY — B1b/B2 blocked because topology bootstrap was unsafe.")
            break
        allow_preview_downstream = bool(getattr(config, "preview_allow_unqualified_downstream", False))
        if planned_phase == "seam-proof" and not render_qualified and not allow_preview_downstream:
            _status("  REJECTED_B2_REDRAW — later stages blocked because same-renderer redraw did not qualify.")
            break
        if planned_phase == "seam-proof" and not render_qualified and allow_preview_downstream:
            _status("  B2 not promotion-qualified; continuing Raven downstream training for diagnostic preview only.")
        if planned_phase == "seam-authority" and not seam_reconstruction_qualified and not allow_preview_downstream:
            _status(
                f"  REJECTED_B3_SEAM_RECON — B4 blocked; best forced-authority phase-SR recovery="
                f"{best_seam_recovery:+.1%}, required={float(getattr(config, 'seam_forced_recovery_required', 0.70)):.1%}."
            )
            break
        if planned_phase == "seam-authority" and not seam_reconstruction_qualified and allow_preview_downstream:
            _status("  B3 not promotion-qualified; continuing authority training for diagnostic preview.")
        if planned_phase == "gate-proof" and not seam_authority_qualified and not allow_preview_downstream:
            _status("  REJECTED_B4_AUTHORITY — later stages blocked because seam detection failed.")
            break
        if planned_phase == "gate-proof" and not seam_authority_qualified and allow_preview_downstream:
            _status("  B4 not promotion-qualified; continuing profile/detail training for diagnostic preview.")
        if planned_phase == "physical-finetune" and not detail_qualified and not bool(getattr(config, "preview_allow_unqualified_downstream", False)):
            phase = "detail-reconstruction"
        previous_phase = str(history[-1].get("phase")) if history else None
        # V10.7.9 stage transitions always start from the best checkpoint of the
        # immediately preceding isolated proof. This prevents a weak final epoch
        # from contaminating the next subsystem and makes every B1-B4 result
        # independently reproducible.
        stage_restore = None
        if phase == "sdf-proof" and previous_phase != "sdf-proof":
            stage_restore = (best_b1a_path, "B1a topology", "B1b parametric primitive")
        elif phase == "seam-proof" and previous_phase != "seam-proof":
            stage_restore = (best_b2_path, "B2 deterministic redraw", "downstream reconstruction")
        elif phase == "seam-authority" and previous_phase != "seam-authority":
            stage_restore = (best_b3_path, "B3 seam reconstruction", "B4 seam authority")
        elif phase == "gate-proof" and previous_phase != "gate-proof":
            stage_restore = (best_b4_path, "B4 seam authority", "gate/profile proof")
        if stage_restore is not None:
            restore_path, restore_name, next_name = stage_restore
            if restore_path.is_file():
                stage_best = torch.load(restore_path, map_location="cpu", weights_only=False)
                model.load_state_dict(stage_best["state_dict"], strict=True)
                model.to(device)
                _status(
                    f"  Restored best {restore_name} checkpoint from epoch "
                    f"{int(stage_best['epoch']):03d} before {next_name}."
                )
            else:
                _status(
                    f"  WARNING: no explicit {restore_name} checkpoint found before {next_name}; "
                    "using current model state."
                )
        if planned_phase == "physical-finetune" and phase == "detail-reconstruction" and previous_phase != "detail-reconstruction":
            _status("  Detail proof not yet qualified; reallocating selector epochs to full-resolution detail reconstruction.")
        if phase == "physical-finetune" and previous_phase != "physical-finetune":
            # Selector loss is a different objective from detail/geometry phases;
            # reset its best-value comparator so the first actually-trained
            # selector checkpoint cannot be discarded merely because an earlier
            # phase used a numerically smaller unrelated loss scale.
            best_validation = float("inf")
            # Selector training begins from the best held-out detail candidate,
            # never merely the last detail epoch.
            if best_path.is_file():
                candidate_best = torch.load(best_path, map_location="cpu", weights_only=False)
                candidate_kind = str(candidate_best.get("selection_kind", ""))
                may_seed_selector = candidate_kind == "heldout-detail-v105-qualified"
                if may_seed_selector:
                    model.load_state_dict(candidate_best["state_dict"], strict=True)
                    model.to(device)
                    _status(
                        f"  Restored best Raven detail candidate epoch {int(candidate_best['epoch']):03d} "
                        f"({candidate_kind}) before benefit-selector training."
                    )
            else:
                _status("  WARNING: no explicit best detail checkpoint found before selector training.")
        model.set_phase(phase)
        b1b_substage = (
            _parametric_b1b_substage(b1b_classifier_qualified, b1b_parameters_qualified)
            if phase == "sdf-proof" else None
        )
        if b1b_substage is not None:
            b1b_stage_epoch += 1
            model.set_parametric_substage(b1b_substage)
        if participation_stage_phase != phase:
            participation_stage_phase = phase
            participation_stage_start = _snapshot_trainable_parameters(model)
        component_modules = _production_component_modules(model)
        component_forward_counts, component_hook_handles = (
            _register_component_forward_hooks(component_modules)
        )
        component_gradient_evidence: dict[str, dict[str, float | int]] = {
            label: {"max": 0.0, "last": 0.0, "parametersWithGradient": 0}
            for label in component_modules
        }
        model.train()
        learning_rate = _phase_lr(phase, config, epoch)
        seam_proof_passes = 1
        if phase == "seam-proof":
            learning_rate = 3.0e-3
            seam_proof_passes = 8
            _status(
                "  B3 direct phase-SR training: 12 real samples x "
                f"{seam_proof_passes} passes, effective lr={learning_rate:.7f}"
            )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))
        totals = _MetricAccumulator()
        if phase == "sdf-proof":
            epoch_loader = parametric_train_loader
        elif phase == "seam-proof":
            epoch_loader = seam_proof_train_loader
        elif phase in {"seam-authority", "gate-proof", "detail-reconstruction", "boundary-hardening", "physical-finetune"}:
            epoch_loader = downstream_train_loader
        else:
            epoch_loader = train_loader
        epoch_batch_size = max(1, int(epoch_loader.batch_size or 1))
        epoch_batch_count = len(epoch_loader) * seam_proof_passes
        epoch_tile_count = len(epoch_loader.dataset) * seam_proof_passes
        epoch_workers = workers
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _status(f"Epoch {epoch:03d}/{config.total_epochs:03d} phase={phase} lr={learning_rate:.7f}")
        if phase == "sdf-proof":
            _status(
                "  B1b learned production primitive proof: "
                f"substage={b1b_substage} stageEpoch={b1b_stage_epoch}; "
                "classifier, regressor and analytic render loss have authority"
            )
        _status(
            f"  Starting DataLoader: {workers} worker(s), "
            f"{epoch_batch_count} training batch(es). Waiting for first batch..."
        )
        data_start = time.perf_counter()

        def _phase_train_batches():
            for pass_index in range(seam_proof_passes):
                if isinstance(epoch_loader.sampler, _EpochOffsetSampler):
                    epoch_loader.sampler.set_epoch(
                        _sampler_epoch_for_phase(phase, epoch, config, len(epoch_loader.dataset))
                        + pass_index
                    )
                for batch in _iter_batches(
                    epoch_loader,
                    device,
                    channels_last=config.channels_last and use_amp,
                    cuda_prefetch=effective_cuda_prefetch,
                ):
                    yield batch

        train_batches = _phase_train_batches()
        _status(
            f"  First batch prefetched in {time.perf_counter() - data_start:.1f}s; "
            "CUDA training loop active."
        )
        epoch_peak_bytes = 0
        for batch_index, batch in enumerate(train_batches, 1):
            step_completed = False
            losses = None
            step_peak_bytes = 0
            step_free_bytes = 0
            memory_attempt = 0

            while not step_completed:
                memory_mode, allocated_before, free_before = memory_governor.before_step()

                try:
                    for retry in range(config.amp_overflow_retries + 1):
                        optimizer.zero_grad(set_to_none=True)
                        losses = None

                        # Saved tensors from the forward graph are either kept on
                        # GPU or moved to system RAM according to current global
                        # VRAM pressure. The numerical model/loss is unchanged.
                        with memory_governor.autograd_context(memory_mode):
                            with torch.autocast(
                                device_type=device.type,
                                dtype=amp_dtype,
                                enabled=use_amp,
                            ):
                                if phase == "sdf-proof":
                                    # B1b is an isolated compact supervised problem.
                                    # The expensive full-model/BoundaryRenderer graph is
                                    # reserved for held-out B2 validation after the epoch.
                                    outputs = None
                                    losses = _parametric_b1b_train_losses(model, batch, config, substage=str(b1b_substage))
                                else:
                                    # Training and validation consume the same learned
                                    # production geometry used by inference.
                                    outputs = _forward_for_phase(model, batch, phase, config)
                            if losses is None:
                                with torch.autocast(
                                    device_type=device.type,
                                    enabled=False,
                                ):
                                    losses = compute_losses(outputs, batch, config, phase)

                            if not _finite_tensor(losses["total"]):
                                _abort_nonfinite(
                                    output_dir,
                                    epoch=epoch,
                                    phase=phase,
                                    batch_index=batch_index,
                                    stage="loss",
                                    details={
                                        "total": float(
                                            losses["total"].detach().cpu()
                                        )
                                    },
                                    model=model,
                                    batch=batch,
                                )

                            scale_before = (
                                float(scaler.get_scale()) if use_scaler else 1.0
                            )
                            scaler.scale(losses["total"]).backward()

                        scaler.unscale_(optimizer)
                        batch_component_gradients = _component_gradient_norms(
                            component_modules
                        )
                        for component_label, (component_norm, parameter_total) in batch_component_gradients.items():
                            component_gradient_evidence[component_label]["last"] = float(component_norm)
                            component_gradient_evidence[component_label]["max"] = max(
                                float(component_gradient_evidence[component_label]["max"]),
                                float(component_norm),
                            )
                            component_gradient_evidence[component_label]["parametersWithGradient"] = max(
                                int(component_gradient_evidence[component_label]["parametersWithGradient"]),
                                int(parameter_total),
                            )
                        try:
                            gradient_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                config.gradient_clip_norm,
                                foreach=True,
                            )
                        except TypeError:
                            gradient_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                config.gradient_clip_norm,
                            )

                        gradient_finite = bool(
                            torch.isfinite(gradient_norm.detach()).item()
                        )
                        if not gradient_finite:
                            terminal_overflow = (
                                not use_scaler
                                or retry >= config.amp_overflow_retries
                                or scale_before <= config.amp_minimum_scale
                            )
                            bad_gradient_names = (
                                _bad_gradient_names(model)
                                if terminal_overflow
                                else []
                            )
                            optimizer.zero_grad(set_to_none=True)
                            if terminal_overflow:
                                _abort_nonfinite(
                                    output_dir,
                                    epoch=epoch,
                                    phase=phase,
                                    batch_index=batch_index,
                                    stage="gradient",
                                    details={
                                        "badGradients": bad_gradient_names,
                                        "gradientNorm": float(
                                            gradient_norm.detach().cpu()
                                        ),
                                        "scale": scale_before,
                                        "retry": retry,
                                    },
                                    model=model,
                                    batch=batch,
                                )
                            next_scale = max(
                                config.amp_minimum_scale,
                                scale_before * 0.5,
                            )
                            scaler.update(new_scale=next_scale)
                            _status(
                                f"  AMP overflow retry batch={batch_index} "
                                f"retry={retry + 1} "
                                f"scale={scale_before:.1f}->{next_scale:.1f}"
                            )
                            continue

                        try:
                            scaler.step(optimizer)
                        except RuntimeError as error:
                            if (
                                not optimizer_mode.startswith("fused ")
                                or not _is_fused_layout_error(error)
                            ):
                                raise
                            migrated = _align_optimizer_state_to_parameters(
                                optimizer
                            )
                            if migrated <= 0:
                                raise
                            _status(
                                "  Fused AdamW state-layout recovery "
                                f"batch={batch_index} aligned={migrated}"
                            )
                            scaler.step(optimizer)

                        scaler.update()
                        step_completed = True
                        break

                    if not step_completed:
                        raise RuntimeError(
                            "V9 optimizer step exhausted AMP retries"
                        )

                except RuntimeError as error:
                    if (
                        device.type != "cuda"
                        or not memory_governor.enabled
                        or not _is_cuda_oom(error)
                        or memory_attempt >= config.reactive_vram_oom_retries
                    ):
                        raise

                    optimizer.zero_grad(set_to_none=True)
                    # Break references to the failed graph before releasing the
                    # allocator cache. The current DataLoader batch remains valid
                    # and is retried unchanged.
                    try:
                        del outputs
                    except UnboundLocalError:
                        pass
                    losses = None
                    memory_attempt += 1
                    memory_governor.handle_oom(
                        batch_index=batch_index,
                        attempt=memory_attempt,
                    )
                    continue

            if losses is None:
                raise RuntimeError(
                    "V9 training step completed without a loss payload"
                )

            step_peak_bytes, step_free_bytes = memory_governor.after_step(
                memory_mode,
                allocated_before,
            )
            epoch_peak_bytes = max(epoch_peak_bytes, step_peak_bytes)

            global_step += 1

            if device.type == "cuda" and batch_index == 1:
                total_bytes = torch.cuda.get_device_properties(device).total_memory
                _status(
                    f"  First-step CUDA peak: "
                    f"{step_peak_bytes / (1024.0 ** 3):.2f} GiB "
                    f"({step_peak_bytes / max(float(total_bytes), 1.0) * 100.0:.1f}% "
                    f"of physical VRAM); mode={memory_mode}; "
                    f"free-after={step_free_bytes / (1024.0 ** 3):.2f} GiB"
                )

            if global_step % config.parameter_finite_check_interval == 0:
                if not _parameters_are_finite(model):
                    _abort_nonfinite(
                        output_dir, epoch=epoch, phase=phase, batch_index=batch_index,
                        stage="parameters", details={"badParameters": _bad_parameter_names(model)},
                        model=model, batch=batch)
            totals.add(losses)
            if (
                batch_index == 1
                or batch_index % max(1, epoch_batch_count // 8) == 0
                or batch_index == epoch_batch_count
            ):
                elapsed = max(1.0e-6, time.perf_counter() - epoch_started)
                tiles_done = min(epoch_tile_count, batch_index * epoch_batch_size)
                tiles_per_second = tiles_done / elapsed
                step_ms = elapsed * 1000.0 / batch_index
                memory_text = ""
                if device.type == "cuda":
                    peak_gib = epoch_peak_bytes / (1024.0 ** 3)
                    memory_text = (
                        f" peak={peak_gib:.2f}GiB "
                        f"vram={memory_governor.status_text()}"
                    )
                _status(
                    f"  {batch_index:5d}/{epoch_batch_count:5d} "
                    f"total={totals.average_value('total'):.5f} "
                    f"step={step_ms:.1f}ms rate={tiles_per_second:.2f}tile/s{memory_text}"
                )

        train_metrics = totals.averages()
        if memory_governor.enabled and config.reactive_vram_release_cache:
            torch.cuda.empty_cache()
        validation_metrics, validation_outputs, validation_batch = _validate(
            model, validation_loader, config, device, phase, amp_dtype=amp_dtype,
            progress_label=f"{phase}-heldout",
        )
        synthetic_sdf_metrics: dict[str, float] | None = None
        defer_exhaustive_b1b = bool(
            phase == "sdf-proof"
            and getattr(config, "quick_preview_defer_exhaustive_b1b_audit", False)
        )
        if defer_exhaustive_b1b:
            _status(
                "  [validation] label=sdf-proof-structural-audit DEFERRED for Quick Raven preview; "
                "the 29-case x 128-step promotion audit is not part of interactive appearance training. "
                "Promotion remains locked until Full structural qualification is run."
            )
        if (
            phase in {"sdf-bootstrap", "sdf-proof", "seam-proof", "seam-authority", "gate-proof"}
            and not defer_exhaustive_b1b
        ):
            # Validate structure and the profile specialist through the real
            # forced-candidate consumer. The selector is never allowed to mask
            # candidate quality during checkpoint selection.
            # exact gate + BoundaryRenderer. During bootstrap the residual head
            # is still frozen by model.set_phase; validation still exposes whether
            # the current level-set is physically useful to the production renderer.
            synthetic_sdf_metrics, _synthetic_outputs, _synthetic_batch = _validate(
                model, synthetic_validation_loader, config, device,
                phase,
                amp_dtype=amp_dtype, exact_geometry_metrics=True,
                live_evidence_root=output_dir / "previews" / "oracle_renderer_preflight",
                live_evidence_epoch=epoch,
                progress_label=f"{phase}-structural-audit",
            )
            if phase in {"sdf-bootstrap", "sdf-proof"}:
                live_progress = {
                    "schema": MODEL_SCHEMA,
                    "epoch": int(epoch),
                    "phase": phase,
                    "b1bSubstage": b1b_substage,
                    "panel2": "GT geometry + deterministic BoundaryRenderer",
                    "panel3": "current predicted geometry + identical deterministic BoundaryRenderer",
                    "panel4": "absolute Panel-3/Panel-2 error x8",
                    "renderBandMaeMean": float(synthetic_sdf_metrics.get("sdf_oracle_render_band_mae_mean", float("inf"))),
                    "globalP3P2MaeMean": float(synthetic_sdf_metrics.get("sdf_oracle_global_mae_mean", float("inf"))),
                    "globalP3P2MaeCaseMax": float(synthetic_sdf_metrics.get("sdf_oracle_global_mae_case_max", float("inf"))),
                    "lineJitterPixelsMean": float(synthetic_sdf_metrics.get("sdf_line_perpendicular_jitter_pixels_mean", float("inf"))),
                    "curveRoughnessPixelsMean": float(synthetic_sdf_metrics.get("sdf_circle_radial_roughness_pixels_mean", float("inf"))),
                    "primitiveClassLoss": float(train_metrics.get("primitive_class", 0.0)),
                    "primitiveClassAccuracy": float(synthetic_sdf_metrics.get("primitive_class_accuracy", 0.0)),
                    "primitiveTeacherParamMae": float(synthetic_sdf_metrics.get("primitive_teacher_param_mae", 999.0)),
                    "primitiveIntegratedParamMae": float(synthetic_sdf_metrics.get("primitive_integrated_param_mae", 999.0)),
                    "primitiveParamMae": float(synthetic_sdf_metrics.get("primitive_param_mae", 0.0)),
                    "primitiveRenderLoss": float(synthetic_sdf_metrics.get("primitive_render", 0.0)),
                    "primitiveConfusionMatrix": synthetic_sdf_metrics.get("primitive_confusion_matrix", []),
                    "primitivePerClassRecall": synthetic_sdf_metrics.get("primitive_per_class_accuracy", []),
                    "note": "The Stage-A PNGs in staged_evidence and staged_evidence_detailed are live and overwritten after every structural epoch.",
                }
                progress_path = output_dir / "previews" / "oracle_renderer_preflight" / "live_progress.json"
                progress_path.write_text(json.dumps(live_progress, indent=2, sort_keys=True), encoding="utf-8")
        if memory_governor.enabled and config.reactive_vram_release_cache:
            torch.cuda.empty_cache()
        seconds = time.perf_counter() - epoch_started
        tiles_per_second = epoch_tile_count / max(seconds, 1.0e-6)
        print(
            f"  train total={train_metrics['total']:.6f} "
            f"sdf={train_metrics['sdf']:.6f} "
            f"profile={train_metrics.get('boundary_profile', 0.0):.6f} "
            f"sdfSurf={train_metrics.get('sdf_surface', 0.0):.4f} "
            f"rawSurf={train_metrics.get('sdf_raw_surface', 0.0):.4f} "
            f"offset={train_metrics.get('contour_normal_offset', 0.0):.4f} "
            f"flow={train_metrics.get('contour_transport', 0.0):.4f} "
            f"dilate={train_metrics.get('contour_dilation', 0.0):.4f} "
            f"flowSm={train_metrics.get('contour_transport_smoothness', 0.0):.4f} "
            f"fold={train_metrics.get('contour_transport_fold', 0.0):.4f} "
            f"softCov={train_metrics.get('contour_soft_coverage', 0.0):.4f} "
            f"qSurf={train_metrics.get('implicit_subpixel_surface', 0.0):.4f} "
            f"qGrad={train_metrics.get('implicit_subpixel_gradient', 0.0):.4f} "
            f"qEik={train_metrics.get('implicit_subpixel_eikonal', 0.0):.4f} "
            f"spTopo={train_metrics.get('spline_graph_topology_control', 0.0):.4f} "
            f"spSign={train_metrics.get('spline_graph_topology_sign', 0.0):.4f} "
            f"spPt={train_metrics.get('spline_graph_point', 0.0):.4f} "
            f"spTan={train_metrics.get('spline_graph_tangent', 0.0):.4f} "
            f"pCls={train_metrics.get('primitive_class', 0.0):.4f} "
            f"pAcc={train_metrics.get('primitive_class_accuracy', 0.0)*100.0:.1f}% "
            f"pParam={train_metrics.get('primitive_param_mae', 0.0):.4f} "
            f"pRend={train_metrics.get('primitive_render', 0.0):.4f} "
            f"stCtr={train_metrics.get('stroke_center', 0.0):.4f} "
            f"stWid={train_metrics.get('stroke_width', 0.0):.4f} "
            f"stTan={train_metrics.get('stroke_tangent', 0.0):.4f} "
            f"stRend={train_metrics.get('stroke_render', 0.0):.4f} "
            f"stTeach={train_metrics.get('stroke_teacher_valid_fraction', 0.0):.4f} "
            f"stWmean={train_metrics.get('stroke_width_mean_pixels', 0.0):.3f}px "
            f"spMove={train_metrics.get('spline_graph_displacement', 0.0):.4f} "
            f"spSm={train_metrics.get('spline_graph_span_smoothness', 0.0):.4f} "
            f"spTC={train_metrics.get('spline_graph_span_tangent', 0.0):.4f} "
            f"spSep={train_metrics.get('spline_graph_span_separation', 0.0):.4f} "
            f"spSdf={train_metrics.get('spline_graph_sdf', 0.0):.4f} "
            f"spGrad={train_metrics.get('spline_graph_gradient', 0.0):.4f} "
            f"metOff={train_metrics.get('spline_metric_offset', 0.0):.4f} "
            f"metEik={train_metrics.get('spline_metric_eikonal_near', 0.0):.4f} "
            f"specCov={train_metrics.get('boundary_specialist_coverage', 0.0):.4f} "
            f"specGrad={train_metrics.get('boundary_specialist_coverage_gradient', 0.0):.4f} "
            f"specMom={train_metrics.get('boundary_specialist_profile_moment', 0.0):.4f} "
            f"specRec={train_metrics.get('boundary_specialist_recovery', 0.0)*100.0:.1f}% "
            f"seamRec={train_metrics.get('seam_recovery', 0.0)*100.0:+.1f}% "
            f"phaseL1={train_metrics.get('seam_phase_residual', 0.0):.4f} "
            f"teach={train_metrics.get('sdf_teacher_render', 0.0):.4f} "
            f"teachProf={train_metrics.get('sdf_teacher_profile', 0.0):.4f} "
            f"teachW={train_metrics.get('sdf_teacher_width', 0.0):.4f} "
            f"teachCorr={train_metrics.get('sdf_teacher_profile_correlation', 0.0):.4f} "
            f"teachRec={train_metrics.get('sdf_teacher_recovery', 0.0)*100.0:.1f}% "
            f"regLR={train_metrics.get('sdf_improvement_regret', 0.0):.4f} "
            f"need={train_metrics.get('geometry_need_mean', 0.0):.3f} "
            f"coarse={train_metrics.get('coarse_sdf_surface', 0.0):.4f} "
            f"eik={train_metrics.get('sdf_eikonal', 0.0):.4f} "
            f"metricGrad={train_metrics.get('sdf_metric_gradient', 0.0):.4f} "
            f"fuzz={train_metrics.get('boundary_fuzz', 0.0):.4f} halo={train_metrics.get('boundary_halo', 0.0):.4f} "
            f"zero={train_metrics.get('boundary_sdf_zero', 0.0):.4f} "
            f"rawZero={train_metrics.get('boundary_sdf_raw_zero', 0.0):.4f} "
            f"topoSign={train_metrics.get('sdf_topology_sign', 0.0):.4f} "
            f"gateRaw={train_metrics.get('boundary_gate_probability_edge_mean', 0.0):.3f} "
            f"gateEdge={train_metrics.get('boundary_gate_edge_mean', 0.0):.3f} "
            f"gateApplied={train_metrics.get('boundary_gate_applied_edge_mean', 0.0):.3f} "
            f"candWin={train_metrics.get('boundary_candidate_win_fraction', 0.0)*100.0:.1f}% "
            f"pixRegret={train_metrics.get('boundary_pixel_regret', 0.0):.5f} "
            f"width={train_metrics.get('boundary_transition_width_mean', 0.0):.3f}px"
        )
        _status(
            f"  valid total={validation_metrics['total']:.6f} "
            f"albedo={validation_metrics['albedo']:.6f} "
            f"regret={validation_metrics['regret']:.6f} "
            f"tangent={validation_metrics['tangent_coherence']:.6f} "
            f"curvature={validation_metrics['curvature_coherence']:.6f} "
            f"profile={validation_metrics.get('boundary_profile', 0.0):.6f} "
            f"sdfSurf={validation_metrics.get('sdf_surface', 0.0):.4f} "
            f"rawSurf={validation_metrics.get('sdf_raw_surface', 0.0):.4f} "
            f"offset={validation_metrics.get('contour_normal_offset', 0.0):.4f} "
            f"flow={validation_metrics.get('contour_transport', 0.0):.4f} "
            f"dilate={validation_metrics.get('contour_dilation', 0.0):.4f} "
            f"flowSm={validation_metrics.get('contour_transport_smoothness', 0.0):.4f} "
            f"fold={validation_metrics.get('contour_transport_fold', 0.0):.4f} "
            f"softCov={validation_metrics.get('contour_soft_coverage', 0.0):.4f} "
            f"qSurf={validation_metrics.get('implicit_subpixel_surface', 0.0):.4f} "
            f"qGrad={validation_metrics.get('implicit_subpixel_gradient', 0.0):.4f} "
            f"qEik={validation_metrics.get('implicit_subpixel_eikonal', 0.0):.4f} "
            f"spTopo={validation_metrics.get('spline_graph_topology_control', 0.0):.4f} "
            f"spSign={validation_metrics.get('spline_graph_topology_sign', 0.0):.4f} "
            f"spPt={validation_metrics.get('spline_graph_point', 0.0):.4f} "
            f"spTan={validation_metrics.get('spline_graph_tangent', 0.0):.4f} "
            f"pCls={validation_metrics.get('primitive_class', 0.0):.4f} "
            f"pAcc={validation_metrics.get('primitive_class_accuracy', 0.0)*100.0:.1f}% "
            f"pParam={validation_metrics.get('primitive_param_mae', 0.0):.4f} "
            f"pRend={validation_metrics.get('primitive_render', 0.0):.4f} "
            f"stCtr={validation_metrics.get('stroke_center', 0.0):.4f} "
            f"stWid={validation_metrics.get('stroke_width', 0.0):.4f} "
            f"stTan={validation_metrics.get('stroke_tangent', 0.0):.4f} "
            f"stRend={validation_metrics.get('stroke_render', 0.0):.4f} "
            f"stTeach={validation_metrics.get('stroke_teacher_valid_fraction', 0.0):.4f} "
            f"stWmean={validation_metrics.get('stroke_width_mean_pixels', 0.0):.3f}px "
            f"spMove={validation_metrics.get('spline_graph_displacement', 0.0):.4f} "
            f"spSm={validation_metrics.get('spline_graph_span_smoothness', 0.0):.4f} "
            f"spTC={validation_metrics.get('spline_graph_span_tangent', 0.0):.4f} "
            f"spSep={validation_metrics.get('spline_graph_span_separation', 0.0):.4f} "
            f"spSdf={validation_metrics.get('spline_graph_sdf', 0.0):.4f} "
            f"metOff={validation_metrics.get('spline_metric_offset', 0.0):.4f} "
            f"metEik={validation_metrics.get('spline_metric_eikonal_near', 0.0):.4f} "
            f"tfSaddle={validation_metrics.get('topology_saddle_projection_fraction', 0.0):.4f} "
            f"specCov={validation_metrics.get('boundary_specialist_coverage', 0.0):.4f} "
            f"specGrad={validation_metrics.get('boundary_specialist_coverage_gradient', 0.0):.4f} "
            f"specMom={validation_metrics.get('boundary_specialist_profile_moment', 0.0):.4f} "
            f"specRec={validation_metrics.get('boundary_specialist_recovery', 0.0)*100.0:.1f}% "
            f"seamRec={validation_metrics.get('seam_recovery', 0.0)*100.0:+.1f}% "
            f"phaseL1={validation_metrics.get('seam_phase_residual', 0.0):.4f} "
            f"teach={validation_metrics.get('sdf_teacher_render', 0.0):.4f} "
            f"teachProf={validation_metrics.get('sdf_teacher_profile', 0.0):.4f} "
            f"teachW={validation_metrics.get('sdf_teacher_width', 0.0):.4f} "
            f"teachCorr={validation_metrics.get('sdf_teacher_profile_correlation', 0.0):.4f} "
            f"teachRec={validation_metrics.get('sdf_teacher_recovery', 0.0)*100.0:.1f}% "
            f"detailRec={validation_metrics.get('detail_recovery', 0.0)*100.0:+.1f}% "
            f"detailGrad={validation_metrics.get('detail_gradient_recovery', 0.0)*100.0:+.1f}% "
            f"detailWin={validation_metrics.get('detail_win_fraction', 0.0)*100.0:.1f}% "
            f"detailReg={validation_metrics.get('detail_regression_fraction', 0.0)*100.0:.1f}% "
            f"regLR={validation_metrics.get('sdf_improvement_regret', 0.0):.4f} "
            f"need={validation_metrics.get('geometry_need_mean', 0.0):.3f} "
            f"coarse={validation_metrics.get('coarse_sdf_surface', 0.0):.4f} "
            f"eik={validation_metrics.get('sdf_eikonal', 0.0):.4f} "
            f"metricGrad={validation_metrics.get('sdf_metric_gradient', 0.0):.4f} "
            f"fuzz={validation_metrics.get('boundary_fuzz', 0.0):.4f} halo={validation_metrics.get('boundary_halo', 0.0):.4f} "
            f"zero={validation_metrics.get('boundary_sdf_zero', 0.0):.4f} "
            f"rawZero={validation_metrics.get('boundary_sdf_raw_zero', 0.0):.4f} "
            f"topoSign={validation_metrics.get('sdf_topology_sign', 0.0):.4f} "
            f"gateRaw={validation_metrics.get('boundary_gate_probability_edge_mean', 0.0):.3f} "
            f"gateEdge={validation_metrics.get('boundary_gate_edge_mean', 0.0):.3f} "
            f"gateFlat={validation_metrics.get('boundary_gate_flat_mean', 0.0):.3f} "
            f"gateApplied={validation_metrics.get('boundary_gate_applied_edge_mean', 0.0):.3f} "
            f"candWin={validation_metrics.get('boundary_candidate_win_fraction', 0.0)*100.0:.1f}% "
            f"pixRegret={validation_metrics.get('boundary_pixel_regret', 0.0):.5f} "
            f"width={validation_metrics.get('boundary_transition_width_mean', 0.0):.3f}px "
            f"delta={validation_metrics.get('boundary_delta_rms', 0.0):.5f} "
            f"proxy={validation_metrics.get('geometry_proxy_improvement', 0.0):+.6f} "
            f"regression={validation_metrics['regression_fraction']*100.0:.2f}% "
            f"improved={validation_metrics['improvement_fraction']*100.0:.2f}%"
        )
        if synthetic_sdf_metrics is not None:
            stageb_geometry_gain = float(
                synthetic_sdf_metrics.get("sdf_stageb_geometry_gain_mean", -1.0)
            )
            stageb_wins = float(
                synthetic_sdf_metrics.get("sdf_stageb_geometry_win_fraction", 0.0)
            )
            contour_chamfer = float(
                synthetic_sdf_metrics.get("sdf_zero_contour_chamfer_pixels", 999.0)
            )
            contour_rms = float(
                synthetic_sdf_metrics.get("sdf_zero_contour_rms_pixels", 999.0)
            )
            topology_regression = float(
                synthetic_sdf_metrics.get("sdf_stageb_topology_regression_fraction", 1.0)
            )
            profile_ratio = float(
                synthetic_sdf_metrics.get("sdf_stageb_profile_width_ratio_mean", 999.0)
            )
            redistanced_near_eik = float(
                synthetic_sdf_metrics.get("sdf_redistanced_near_eikonal", synthetic_sdf_metrics.get("sdf_metricized_near_eikonal", 999.0))
            )
            raw_eik = float(synthetic_sdf_metrics.get("sdf_eikonal", 999.0))
            source_chamfer = float(synthetic_sdf_metrics.get("sdf_source_zero_contour_chamfer_pixels", 999.0))
            contour_gain = float(synthetic_sdf_metrics.get("sdf_zero_contour_relative_gain_mean", -1.0))
            contour_wins = float(synthetic_sdf_metrics.get("sdf_zero_contour_relative_win_fraction", 0.0))
            contour_regress = float(synthetic_sdf_metrics.get("sdf_zero_contour_relative_regression_fraction", 1.0))
            source_missing = float(synthetic_sdf_metrics.get("sdf_source_missing_contour_fraction", 1.0))
            predicted_missing = float(synthetic_sdf_metrics.get("sdf_predicted_missing_contour_fraction", 1.0))
            profile_gain = float(synthetic_sdf_metrics.get("sdf_profile_error_relative_gain_mean", -1.0))
            profile_teacher_recovery = float(synthetic_sdf_metrics.get("sdf_profile_teacher_recovery_mean", -1.0))
            line_jitter = float(synthetic_sdf_metrics.get("sdf_line_perpendicular_jitter_pixels_mean", 999.0))
            curve_roughness = float(synthetic_sdf_metrics.get("sdf_circle_radial_roughness_pixels_mean", 999.0))
            staircase_recovery = float(synthetic_sdf_metrics.get("sdf_line_staircase_recovery_mean", -1.0))
            oracle_render_mae = float(synthetic_sdf_metrics.get("sdf_oracle_render_band_mae_mean", 999.0))
            oracle_global_mae = float(synthetic_sdf_metrics.get("sdf_oracle_global_mae_mean", 999.0))
            oracle_global_mae_max = float(synthetic_sdf_metrics.get("sdf_oracle_global_mae_case_max", 999.0))
            oracle_gradient_mae = float(synthetic_sdf_metrics.get("sdf_oracle_gradient_mae_mean", 999.0))
            oracle_width_error = float(synthetic_sdf_metrics.get("sdf_oracle_profile_width_relative_error_mean", 999.0))
            oracle_profile_corr = float(synthetic_sdf_metrics.get("sdf_oracle_profile_correlation_mean", -1.0))
            oracle_core_halo_delta = float(synthetic_sdf_metrics.get("sdf_oracle_core_halo_delta_8bit_max", 999.0))
            primitive_class_accuracy = float(synthetic_sdf_metrics.get("primitive_class_accuracy", 0.0))
            primitive_teacher_param_mae = float(synthetic_sdf_metrics.get("primitive_teacher_param_mae", synthetic_sdf_metrics.get("primitive_param_mae", 999.0)))
            primitive_integrated_param_mae = float(synthetic_sdf_metrics.get("primitive_integrated_param_mae", 999.0))
            primitive_param_mae = primitive_teacher_param_mae
            primitive_gate = (
                phase != "sdf-proof"
                or b1b_parameters_qualified
            )
            hard_structure_gate = (
                contour_gain >= float(config.sdf_relative_gain_required)
                and contour_wins >= float(config.sdf_relative_win_fraction)
                and contour_regress <= float(config.sdf_relative_regression_fraction)
                and predicted_missing <= source_missing + float(config.sdf_missing_contour_tolerance)
                and contour_chamfer <= float(config.sdf_catastrophic_chamfer_pixels)
                and topology_regression == 0.0
                and primitive_gate
                and line_jitter <= float(config.structural_line_jitter_required_pixels)
                and curve_roughness <= float(config.structural_curve_roughness_required_pixels)
                and staircase_recovery >= float(config.structural_line_staircase_recovery_required)
            )
            legacy_profile_render_gate = (
                oracle_render_mae <= float(config.sdf_oracle_render_band_mae_required)
                and oracle_gradient_mae <= float(config.sdf_oracle_gradient_mae_required)
                and oracle_width_error <= float(config.sdf_oracle_profile_width_error_required)
                and oracle_profile_corr >= float(config.sdf_oracle_profile_correlation_required)
                and oracle_core_halo_delta <= float(config.sdf_oracle_core_halo_delta_required_8bit)
            )
            direct_pixel_render_gate = (
                oracle_global_mae <= float(config.sdf_oracle_global_mae_required)
                and oracle_global_mae_max <= float(config.sdf_oracle_global_mae_case_max_required)
                and oracle_render_mae <= float(config.sdf_oracle_render_band_mae_preview_required)
                and oracle_gradient_mae <= float(config.sdf_oracle_gradient_mae_required)
            )
            # Either proof is sufficient. The profile bundle remains the strict
            # production-style local proof; direct pixel equivalence prevents an
            # unstable local width/halo statistic from blocking a visually and
            # geometrically near-identical same-renderer result.
            hard_render_gate = hard_structure_gate and (
                legacy_profile_render_gate or direct_pixel_render_gate
            )
            hard_candidate_gate = hard_render_gate and profile_teacher_recovery >= float(config.sdf_teacher_recovery_required)
            structure_score = (
                - contour_gain
                + 0.25 * contour_regress
                + 0.10 * max(0.0, predicted_missing - source_missing)
                + 0.02 * redistanced_near_eik
                + 0.20 * max(0.0, line_jitter - float(config.structural_line_jitter_required_pixels))
                + 0.20 * max(0.0, curve_roughness - float(config.structural_curve_roughness_required_pixels))
                + 5.0 * topology_regression
                + (0.0 if hard_structure_gate else 2.0)
            )
            structure_rank = _structure_selection_rank(
                qualified=hard_structure_gate,
                topology_regression=topology_regression,
                predicted_missing=predicted_missing,
                source_missing=source_missing,
                line_jitter=line_jitter,
                curve_roughness=curve_roughness,
                line_limit=float(config.structural_line_jitter_required_pixels),
                curve_limit=float(config.structural_curve_roughness_required_pixels),
                contour_gain=contour_gain,
                contour_chamfer=contour_chamfer,
                profile_teacher_recovery=profile_teacher_recovery,
            )
            sdf_score = (
                structure_score
                - 0.35 * profile_teacher_recovery
                + (0.0 if hard_candidate_gate else 2.0)
            )
            _status(
                f"  synthetic-stageB renderGain={stageb_geometry_gain:+.2%} "
                f"wins={stageb_wins*100.0:.1f}% "
                f"sourceChamfer={source_chamfer:.3f}px predChamfer={contour_chamfer:.3f}px "
                f"contourGain={contour_gain:+.2%} contourWins={contour_wins*100.0:.1f}% "
                f"missing={source_missing*100.0:.1f}%->{predicted_missing*100.0:.1f}% "
                f"profileGain={profile_gain:+.2%} teacherRec={profile_teacher_recovery:+.2%} "
                f"P3P2mae={oracle_render_mae:.4f} gradMae={oracle_gradient_mae:.4f} "
                f"widthErr={oracle_width_error:.2%} corr={oracle_profile_corr:.4f} "
                f"lineJit={line_jitter:.3f}px curveRough={curve_roughness:.3f}px "
                f"pAcc={primitive_class_accuracy*100.0:.1f}% "
                f"pTeach={primitive_teacher_param_mae:.4f} pInt={primitive_integrated_param_mae:.4f} "
                f"stairRec={staircase_recovery:+.1%} "
                f"topologyReg={topology_regression*100.0:.1f}% "
                f"gate={('CANDIDATE_PASS' if hard_candidate_gate else 'HOLD') if phase == 'gate-proof' else ('STRUCTURE_PASS' if hard_structure_gate else 'HOLD')} "
                f"score={(sdf_score if phase == 'gate-proof' else structure_score):.3f}"
            )
            per_class = synthetic_sdf_metrics.get("primitive_per_class_accuracy")
            if isinstance(per_class, list) and len(per_class) == PRIMITIVE_COUNT:
                _status(
                    "  primitive class recall: "
                    + " ".join(
                        f"{PRIMITIVE_NAMES[i]}={float(per_class[i])*100.0:.0f}%"
                        for i in range(PRIMITIVE_COUNT)
                    )
                )
        print(f"  seconds={seconds:.1f} throughput={tiles_per_second:.2f}tile/s")
        sample_path = output_dir / "samples" / f"epoch_{epoch:03d}_{phase}" / "validation_contact_sheet.png"
        _save_contact_sheet(validation_outputs, validation_batch, sample_path, phase)

        for hook_handle in component_hook_handles:
            hook_handle.remove()
        epoch_component_evidence = _component_epoch_evidence(
            model,
            component_modules,
            component_forward_counts,
            component_gradient_evidence,
            participation_stage_start,
            train_metrics,
        )
        participation_record = {
            "epoch": int(epoch),
            "phase": phase,
            "productionForward": "FidelityResidualNetV9.forward(inputs)",
            "components": epoch_component_evidence,
        }
        architecture_participation.append(participation_record)

        record = {
            "epoch": epoch,
            "phase": phase,
            "b1bSubstage": b1b_substage,
            "seconds": seconds,
            "batches": len(epoch_loader),
            "tiles": config.tiles_per_epoch,
            "tiles_per_second": tiles_per_second,
            "train": train_metrics,
            "validation": validation_metrics,
            "syntheticSdfValidation": synthetic_sdf_metrics,
            "architectureParticipation": participation_record,
            "performance": {
                "profile": config.performance_profile,
                "precision": precision_name,
                "optimizer": optimizer_mode,
                "workers": workers,
                "prefetchFactor": config.data_loader_prefetch_factor,
                "channelsLast": config.channels_last and use_amp,
                "cudaPrefetch": config.cuda_prefetch and use_amp,
            },
        }
        history.append(record)
        if synthetic_sdf_metrics is not None:
            stageb_geometry_gain = float(
                synthetic_sdf_metrics.get("sdf_stageb_geometry_gain_mean", -1.0)
            )
            contour_chamfer = float(
                synthetic_sdf_metrics.get("sdf_zero_contour_chamfer_pixels", 999.0)
            )
            topology_regression = float(
                synthetic_sdf_metrics.get("sdf_stageb_topology_regression_fraction", 1.0)
            )
            profile_ratio = float(
                synthetic_sdf_metrics.get("sdf_stageb_profile_width_ratio_mean", 999.0)
            )
            redistanced_near_eik = float(
                synthetic_sdf_metrics.get("sdf_redistanced_near_eikonal", synthetic_sdf_metrics.get("sdf_metricized_near_eikonal", 999.0))
            )
            source_chamfer = float(synthetic_sdf_metrics.get("sdf_source_zero_contour_chamfer_pixels", 999.0))
            contour_gain = float(synthetic_sdf_metrics.get("sdf_zero_contour_relative_gain_mean", -1.0))
            contour_wins = float(synthetic_sdf_metrics.get("sdf_zero_contour_relative_win_fraction", 0.0))
            contour_regress = float(synthetic_sdf_metrics.get("sdf_zero_contour_relative_regression_fraction", 1.0))
            source_missing = float(synthetic_sdf_metrics.get("sdf_source_missing_contour_fraction", 1.0))
            predicted_missing = float(synthetic_sdf_metrics.get("sdf_predicted_missing_contour_fraction", 1.0))
            profile_gain = float(synthetic_sdf_metrics.get("sdf_profile_error_relative_gain_mean", -1.0))
            profile_teacher_recovery = float(synthetic_sdf_metrics.get("sdf_profile_teacher_recovery_mean", -1.0))
            line_jitter = float(synthetic_sdf_metrics.get("sdf_line_perpendicular_jitter_pixels_mean", 999.0))
            curve_roughness = float(synthetic_sdf_metrics.get("sdf_circle_radial_roughness_pixels_mean", 999.0))
            staircase_recovery = float(synthetic_sdf_metrics.get("sdf_line_staircase_recovery_mean", -1.0))
            oracle_render_mae = float(synthetic_sdf_metrics.get("sdf_oracle_render_band_mae_mean", 999.0))
            oracle_global_mae = float(synthetic_sdf_metrics.get("sdf_oracle_global_mae_mean", 999.0))
            oracle_global_mae_max = float(synthetic_sdf_metrics.get("sdf_oracle_global_mae_case_max", 999.0))
            oracle_gradient_mae = float(synthetic_sdf_metrics.get("sdf_oracle_gradient_mae_mean", 999.0))
            oracle_width_error = float(synthetic_sdf_metrics.get("sdf_oracle_profile_width_relative_error_mean", 999.0))
            oracle_profile_corr = float(synthetic_sdf_metrics.get("sdf_oracle_profile_correlation_mean", -1.0))
            oracle_core_halo_delta = float(synthetic_sdf_metrics.get("sdf_oracle_core_halo_delta_8bit_max", 999.0))
            primitive_class_accuracy = float(synthetic_sdf_metrics.get("primitive_class_accuracy", 0.0))
            primitive_teacher_param_mae = float(synthetic_sdf_metrics.get("primitive_teacher_param_mae", synthetic_sdf_metrics.get("primitive_param_mae", 999.0)))
            primitive_integrated_param_mae = float(synthetic_sdf_metrics.get("primitive_integrated_param_mae", 999.0))
            primitive_param_mae = primitive_teacher_param_mae
            primitive_gate = (
                phase != "sdf-proof"
                or b1b_parameters_qualified
            )
            hard_structure_gate = (
                contour_gain >= float(config.sdf_relative_gain_required)
                and contour_wins >= float(config.sdf_relative_win_fraction)
                and contour_regress <= float(config.sdf_relative_regression_fraction)
                and predicted_missing <= source_missing + float(config.sdf_missing_contour_tolerance)
                and contour_chamfer <= float(config.sdf_catastrophic_chamfer_pixels)
                and topology_regression == 0.0
                and primitive_gate
                and line_jitter <= float(config.structural_line_jitter_required_pixels)
                and curve_roughness <= float(config.structural_curve_roughness_required_pixels)
                and staircase_recovery >= float(config.structural_line_staircase_recovery_required)
            )
            legacy_profile_render_gate = (
                oracle_render_mae <= float(config.sdf_oracle_render_band_mae_required)
                and oracle_gradient_mae <= float(config.sdf_oracle_gradient_mae_required)
                and oracle_width_error <= float(config.sdf_oracle_profile_width_error_required)
                and oracle_profile_corr >= float(config.sdf_oracle_profile_correlation_required)
                and oracle_core_halo_delta <= float(config.sdf_oracle_core_halo_delta_required_8bit)
            )
            direct_pixel_render_gate = (
                oracle_global_mae <= float(config.sdf_oracle_global_mae_required)
                and oracle_global_mae_max <= float(config.sdf_oracle_global_mae_case_max_required)
                and oracle_render_mae <= float(config.sdf_oracle_render_band_mae_preview_required)
                and oracle_gradient_mae <= float(config.sdf_oracle_gradient_mae_required)
            )
            # Either proof is sufficient. The profile bundle remains the strict
            # production-style local proof; direct pixel equivalence prevents an
            # unstable local width/halo statistic from blocking a visually and
            # geometrically near-identical same-renderer result.
            hard_render_gate = hard_structure_gate and (
                legacy_profile_render_gate or direct_pixel_render_gate
            )
            hard_candidate_gate = hard_render_gate and profile_teacher_recovery >= float(config.sdf_teacher_recovery_required)
            structure_score = (
                - contour_gain
                + 0.25 * contour_regress
                + 0.10 * max(0.0, predicted_missing - source_missing)
                + 0.02 * redistanced_near_eik
                + 0.20 * max(0.0, line_jitter - float(config.structural_line_jitter_required_pixels))
                + 0.20 * max(0.0, curve_roughness - float(config.structural_curve_roughness_required_pixels))
                + 5.0 * topology_regression
                + (0.0 if hard_structure_gate else 2.0)
            )
            structure_rank = _structure_selection_rank(
                qualified=hard_structure_gate,
                topology_regression=topology_regression,
                predicted_missing=predicted_missing,
                source_missing=source_missing,
                line_jitter=line_jitter,
                curve_roughness=curve_roughness,
                line_limit=float(config.structural_line_jitter_required_pixels),
                curve_limit=float(config.structural_curve_roughness_required_pixels),
                contour_gain=contour_gain,
                contour_chamfer=contour_chamfer,
                profile_teacher_recovery=profile_teacher_recovery,
            )
            sdf_score = (
                structure_score
                - 0.35 * profile_teacher_recovery
                + (0.0 if hard_candidate_gate else 2.0)
            )
            if phase == "sdf-bootstrap":
                # B1a qualifies topology only; smoothness belongs exclusively to B1b.
                topology_ok = (
                    predicted_missing <= source_missing + float(config.sdf_missing_contour_tolerance)
                    and topology_regression == 0.0
                )
                _atomic_torch_save({
                    "schema": MODEL_SCHEMA, "config": config.to_dict(), "phase": "sdf-bootstrap",
                    "epoch": epoch, "metrics": synthetic_sdf_metrics,
                    "qualified": bool(topology_ok),
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                }, best_b1a_path)
                if topology_ok and not topology_bootstrapped:
                    topology_bootstrapped = True
                    _status(f"  B1a TOPOLOGY BOOTSTRAP PASSED at epoch {epoch:03d}; topology freezes now.")
            elif phase == "sdf-proof":
                # V10.7.9 is pass-driven. Each subproblem owns its checkpoint
                # and cannot be skipped by exhausting an arbitrary epoch count.
                class_required = float(getattr(config, "parametric_primitive_class_accuracy_required", 0.95))
                param_required = float(getattr(config, "parametric_primitive_param_mae_required", 0.040))

                if b1b_substage == "classifier":
                    if primitive_class_accuracy > best_classifier_accuracy:
                        best_classifier_accuracy = primitive_class_accuracy
                        _atomic_torch_save({
                            "schema": MODEL_SCHEMA, "config": config.to_dict(), "phase": "sdf-proof",
                            "epoch": epoch, "selection_kind": "v1079-b1b-classifier",
                            "b1b_substage": b1b_substage, "metrics": synthetic_sdf_metrics,
                            "qualified": bool(primitive_class_accuracy >= class_required),
                            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        }, best_b1b_classifier_path)
                        _status(
                            f"  B1b-1 classifier checkpoint: epoch={epoch:03d} "
                            f"accuracy={primitive_class_accuracy*100.0:.1f}% "
                            f"required={class_required*100.0:.1f}%"
                        )
                    if primitive_class_accuracy >= class_required:
                        b1b_classifier_qualified = True
                        b1b_stage_epoch = 0
                        _status(
                            f"  B1b-1 CLASSIFIER PASSED at epoch {epoch:03d}: "
                            f"accuracy={primitive_class_accuracy*100.0:.1f}%. "
                            "Classifier freezes; GT-class parameter regression starts next epoch."
                        )

                elif b1b_substage == "parameters":
                    if primitive_teacher_param_mae < best_teacher_param_mae:
                        best_teacher_param_mae = primitive_teacher_param_mae
                        _atomic_torch_save({
                            "schema": MODEL_SCHEMA, "config": config.to_dict(), "phase": "sdf-proof",
                            "epoch": epoch, "selection_kind": "v1079-b1b-parameters",
                            "b1b_substage": b1b_substage, "metrics": synthetic_sdf_metrics,
                            "qualified": bool(primitive_teacher_param_mae <= param_required),
                            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        }, best_b1b_parameters_path)
                        _status(
                            f"  B1b-2 parameter checkpoint: epoch={epoch:03d} "
                            f"teacherMAE={primitive_teacher_param_mae:.4f} "
                            f"integratedMAE={primitive_integrated_param_mae:.4f} required={param_required:.4f}"
                        )
                    if primitive_teacher_param_mae <= param_required:
                        b1b_parameters_qualified = True
                        b1b_stage_epoch = 0
                        _status(
                            f"  B1b-2 GT-ROUTED PARAMETERS PASSED at epoch {epoch:03d}: "
                            f"teacherMAE={primitive_teacher_param_mae:.4f}. "
                            "Integrated predicted-class redraw starts next epoch."
                        )

                elif b1b_substage == "integration":
                    b1b_classifier_qualified = primitive_class_accuracy >= class_required
                    b1b_parameters_qualified = primitive_teacher_param_mae <= param_required
                    _status(
                        "  B1b canonical integration prerequisites: "
                        f"classifier={'PASS' if b1b_classifier_qualified else 'HOLD'} "
                        f"parameters={'PASS' if b1b_parameters_qualified else 'HOLD'}"
                    )

                # Integrated geometry is evaluated every epoch, but it may only
                # qualify after both independent prerequisites have passed.
                integration_ready = bool(b1b_parameters_qualified)
                select_structure = integration_ready and structure_rank < best_structure_rank
                if select_structure:
                    best_structure_rank = structure_rank
                    best_structure_score = structure_score
                    payload = {
                        "schema": MODEL_SCHEMA, "config": config.to_dict(), "phase": "sdf-proof",
                        "epoch": epoch, "validation_total": structure_score,
                        "selection_rank": list(structure_rank),
                        "selection_kind": "v1079-b1b-integrated-parametric-primitive",
                        "b1b_substage": b1b_substage,
                        "hard_structure_gate": bool(hard_structure_gate),
                        "synthetic_sdf_validation": synthetic_sdf_metrics,
                        "metrics": synthetic_sdf_metrics,
                        "qualified": bool(hard_structure_gate),
                        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    }
                    _atomic_torch_save(payload, best_structure_path)
                    _atomic_torch_save(payload, best_b1b_path)
                    _status(
                        f"  B1b-3 integration checkpoint: epoch={epoch:03d} "
                        f"class={primitive_class_accuracy*100.0:.1f}% "
                        f"teacherMAE={primitive_teacher_param_mae:.4f} "
                        f"P3/P2={oracle_render_mae:.4f} qualified={'YES' if hard_structure_gate else 'NO'}"
                    )
                if integration_ready and hard_structure_gate and not structure_qualified:
                    structure_qualified = True
                    _status(f"  B1b PARAMETRIC PRIMITIVE PROOF PASSED at epoch {epoch:03d}.")
                if integration_ready and hard_render_gate:
                    if not render_qualified:
                        _status(
                            f"  B2 SAME-RENDERER REDRAW PASSED at epoch {epoch:03d}; "
                            "Panel 3 differs from Panel 2 only by predicted geometry."
                        )
                    render_qualified = True
                    if oracle_render_mae <= best_render_mae:
                        best_render_mae = oracle_render_mae
                        _atomic_torch_save({
                            "schema": MODEL_SCHEMA, "config": config.to_dict(), "phase": "sdf-proof",
                            "epoch": epoch, "selection_kind": "v1079-b2-same-renderer",
                            "metrics": synthetic_sdf_metrics, "qualified": True,
                            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        }, best_b2_path)
            if phase == "seam-proof":
                seam_recovery = float(validation_metrics.get("seam_recovery", -1.0))
                if seam_recovery >= float(getattr(config, "seam_forced_recovery_required", 0.70)) and not seam_reconstruction_qualified:
                    seam_reconstruction_qualified = True
                    best_seam_recovery = seam_recovery
                    _atomic_torch_save({
                        "schema": MODEL_SCHEMA, "config": config.to_dict(), "phase": "seam-proof",
                        "epoch": epoch, "metrics": validation_metrics, "qualified": True,
                        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    }, best_b3_path)
                    _status(f"  B3 FORCED-AUTHORITY SEAM PROOF PASSED: recovery={seam_recovery:+.1%}.")
            if phase == "seam-authority":
                seam_iou = float(validation_metrics.get("seam_authority_iou", 0.0))
                if seam_iou >= float(getattr(config, "seam_authority_iou_required", 0.55)) and not seam_authority_qualified:
                    seam_authority_qualified = True
                    best_seam_iou = seam_iou
                    _atomic_torch_save({
                        "schema": MODEL_SCHEMA, "config": config.to_dict(), "phase": "seam-authority",
                        "epoch": epoch, "metrics": validation_metrics, "qualified": True,
                        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    }, best_b4_path)
                    _status(f"  B4 SEAM AUTHORITY PROOF PASSED: IoU={seam_iou:.3f}.")
            if phase == "seam-proof":
                seam_recovery_now = float(validation_metrics.get("seam_recovery", -1.0))
                if seam_recovery_now > best_seam_recovery:
                    best_seam_recovery = seam_recovery_now
                    _atomic_torch_save({
                        "schema": MODEL_SCHEMA, "config": config.to_dict(), "phase": "seam-proof",
                        "epoch": epoch, "metrics": validation_metrics,
                        "qualified": bool(seam_reconstruction_qualified),
                        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    }, best_b3_path)
            if phase == "seam-authority":
                seam_iou_now = float(validation_metrics.get("seam_authority_iou", 0.0))
                if seam_iou_now > best_seam_iou:
                    best_seam_iou = seam_iou_now
                    _atomic_torch_save({
                        "schema": MODEL_SCHEMA, "config": config.to_dict(), "phase": "seam-authority",
                        "epoch": epoch, "metrics": validation_metrics,
                        "qualified": bool(seam_authority_qualified),
                        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                    }, best_b4_path)

            if phase == "gate-proof" and sdf_score < best_sdf_score:
                best_sdf_score = sdf_score
                _atomic_torch_save({
                    "epoch": epoch,
                    "validation_total": sdf_score,
                    "selection_kind": "synthetic-stageb-branch-spline-specialist-v105",
                    "synthetic_sdf_validation": synthetic_sdf_metrics,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                }, best_path)
                _status(
                    f"  full candidate checkpoint selected: epoch={epoch:03d} "
                    f"contourGain={contour_gain:+.2%} teacherRec={profile_teacher_recovery:+.2%} "
                    f"lineJit={line_jitter:.3f}px curveRough={curve_roughness:.3f}px "
                    f"score={sdf_score:.3f}"
                )
        if phase == "detail-reconstruction":
            detail_recovery = float(validation_metrics.get("detail_recovery", -1.0))
            detail_gradient_recovery = float(validation_metrics.get("detail_gradient_recovery", -1.0))
            detail_win_fraction = float(validation_metrics.get("detail_win_fraction", 0.0))
            detail_regression_fraction = float(validation_metrics.get("detail_regression_fraction", 1.0))
            hard_detail_gate = (
                detail_recovery >= float(getattr(config, "detail_recovery_required", 0.12))
                and detail_gradient_recovery >= float(getattr(config, "detail_gradient_recovery_required", 0.08))
                and detail_win_fraction >= float(getattr(config, "detail_win_fraction_required", 0.55))
                and detail_regression_fraction <= float(getattr(config, "detail_regression_fraction_max", 0.35))
            )
            detail_rank = (
                0.0 if hard_detail_gate else 1.0,
                -detail_recovery,
                -detail_gradient_recovery,
                -detail_win_fraction,
                detail_regression_fraction,
                float(validation_metrics["total"]),
            )
            if detail_rank < best_detail_rank:
                best_detail_rank = detail_rank
                best_validation = float(validation_metrics["total"])
                _atomic_torch_save({
                    "epoch": epoch,
                    "validation_total": best_validation,
                    "selection_kind": "heldout-detail-v105-qualified" if hard_detail_gate else "heldout-detail-v105-unqualified",
                    "detail_validation": {
                        "detail_recovery": detail_recovery,
                        "detail_gradient_recovery": detail_gradient_recovery,
                        "detail_win_fraction": detail_win_fraction,
                        "detail_regression_fraction": detail_regression_fraction,
                    },
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                }, best_path)
                _status(
                    f"  V10.7.9 detail checkpoint selected: epoch={epoch:03d} "
                    f"recovery={detail_recovery:+.2%} grad={detail_gradient_recovery:+.2%} "
                    f"wins={detail_win_fraction:.1%} regress={detail_regression_fraction:.1%} "
                    f"qualified={'YES' if hard_detail_gate else 'NO'}"
                )
            if hard_detail_gate and not detail_qualified:
                detail_qualified = True
                _status(
                    f"  DETAIL PROOF PASSED at epoch {epoch:03d}; benefit-selector budget is now enabled."
                )
        if phase == "physical-finetune":
            final_recovery = float(validation_metrics.get("final_recovery", -1.0))
            final_win_fraction = float(validation_metrics.get("final_win_fraction", 0.0))
            final_regression_fraction = float(validation_metrics.get("final_regression_fraction", 1.0))
            selector_gate = (
                detail_qualified
                and final_recovery >= float(getattr(config, "selector_recovery_required", 0.05))
                and final_win_fraction >= float(getattr(config, "selector_win_fraction_required", 0.58))
                and final_regression_fraction <= float(getattr(config, "selector_regression_fraction_max", 0.20))
            )
            selector_rank = (
                0.0 if selector_gate else 1.0,
                -final_recovery,
                -final_win_fraction,
                final_regression_fraction,
            )
            _status(
                f"  selector-validation recovery={final_recovery:+.2%} "
                f"wins={final_win_fraction:.1%} regress={final_regression_fraction:.1%} "
                f"qualified={'YES' if selector_gate else 'NO'}"
            )
            if selector_gate and selector_rank < best_selector_rank:
                best_selector_rank = selector_rank
                best_validation = float(validation_metrics["total"])
                selection_kind = "production-final-selector-qualified"
                _atomic_torch_save({
                    "epoch": epoch,
                    "validation_total": best_validation,
                    "selection_kind": selection_kind,
                    "selector_validation": {
                        "final_recovery": final_recovery,
                        "final_win_fraction": final_win_fraction,
                        "final_regression_fraction": final_regression_fraction,
                        "qualified": True,
                    },
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                }, best_path)
                _status(
                    f"  SELECTOR PROOF PASSED at epoch {epoch:03d}; "
                    f"{selection_kind} selected."
                )

        _atomic_torch_save({
            "schema": MODEL_SCHEMA,
            "config_hash": _config_hash(config),
            "dataset_fingerprint": fingerprint,
            "completed_epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": _capture_rng(),
            "history": history,
            "best_validation": best_validation,
            "best_sdf_score": best_sdf_score,
            "best_structure_score": best_structure_score,
            "best_structure_rank": list(best_structure_rank),
            "best_render_mae": best_render_mae,
            "best_seam_recovery": best_seam_recovery,
            "best_seam_iou": best_seam_iou,
            "best_detail_rank": list(best_detail_rank),
            "best_selector_rank": list(best_selector_rank),
            "topology_bootstrapped": topology_bootstrapped,
            "b1b_classifier_qualified": b1b_classifier_qualified,
            "b1b_parameters_qualified": b1b_parameters_qualified,
            "b1b_stage_epoch": b1b_stage_epoch,
            "best_classifier_accuracy": best_classifier_accuracy,
            "best_teacher_param_mae": best_teacher_param_mae,
            "structure_qualified": structure_qualified,
            "render_qualified": render_qualified,
            "seam_reconstruction_qualified": seam_reconstruction_qualified,
            "seam_authority_qualified": seam_authority_qualified,
            "detail_qualified": detail_qualified,
            "architecture_participation": architecture_participation,
            "cache_equivalence": cache_equivalence,
        }, state_path)

        if phase == "physical-finetune" and early_stop_patience is not None and early_stop_patience > 0:
            current_validation = float(validation_metrics["total"])
            if current_validation < early_stop_best - max(0.0, float(early_stop_min_delta)):
                early_stop_best = current_validation
                early_stop_stale = 0
            else:
                early_stop_stale += 1
            _status(
                f"  convergence gate: best={early_stop_best:.6f} "
                f"stale={early_stop_stale}/{early_stop_patience}"
            )
            if early_stop_stale >= early_stop_patience:
                early_stopped_epoch = epoch
                _status(
                    f"  Early convergence stop at epoch {epoch:03d}; all V9 phases were entered "
                    "and held-out validation stopped improving materially."
                )
                break

        if staged_stop_epoch is not None and epoch >= staged_stop_epoch:
            _status(
                f"  Staged checkpoint stop reached after {phase}; "
                "preserving optimizer/RNG state for gated resume."
            )
            break

    isolated_stop_phases = {"sdf-bootstrap", "sdf-proof", "seam-proof", "seam-authority"}
    if stop_after_phase in isolated_stop_phases or not best_path.is_file():
        # V10.7.9 exports the highest deterministic structural proof reached.
        staged_sources = (
            (best_b2_path, "v1079-b2-same-renderer-redraw"),
            (best_b1b_path, "v1079-b1b-integrated-parametric-primitive"),
            (best_b1a_path, "v1079-b1a-topology"),
            (best_structure_path, "v1079-structure-fallback"),
        )
        selected_stage = None
        selected_kind = None
        for candidate_path, candidate_kind in staged_sources:
            if candidate_path.is_file():
                selected_stage = torch.load(candidate_path, map_location="cpu", weights_only=False)
                selected_kind = candidate_kind
                break
        if selected_stage is not None:
            selected_epoch = int(selected_stage.get("epoch", history[-1]["epoch"] if history else 0))
            selected_metrics = selected_stage.get("metrics") or selected_stage.get("synthetic_sdf_validation") or {}
            selected_validation_total = selected_stage.get("validation_total")
            if selected_validation_total is None:
                selected_validation_total = next(
                    (
                        float(row["validation"]["total"])
                        for row in history
                        if int(row.get("epoch", -1)) == selected_epoch
                    ),
                    float(selected_metrics.get("total", history[-1]["validation"]["total"] if history else float("inf"))),
                )
            wrapper = {
                "schema": MODEL_SCHEMA,
                "config": config.to_dict(),
                "phase": str(selected_stage.get("phase", "unknown")),
                "epoch": selected_epoch,
                "validation_total": float(selected_validation_total),
                "selection_kind": str(selected_kind),
                "qualified": bool(selected_stage.get("qualified", False)),
                "state_dict": selected_stage["state_dict"],
            }
            if str(selected_stage.get("phase", "")) in {"sdf-bootstrap", "sdf-proof"}:
                wrapper["synthetic_sdf_validation"] = selected_metrics
            else:
                wrapper["stage_validation"] = selected_metrics
            _atomic_torch_save(wrapper, best_path)
            _status(
                f"  Selected {selected_kind} epoch {selected_epoch:03d} for staged audit/export."
            )
        else:
            if not history:
                raise RuntimeError("V10.7.9 training ended before any auditable checkpoint was produced")
            best_validation = float(history[-1]["validation"]["total"])
            _atomic_torch_save({
                "schema": MODEL_SCHEMA,
                "config": config.to_dict(),
                "phase": str(history[-1].get("phase", "unknown")),
                "epoch": int(history[-1]["epoch"]),
                "validation_total": best_validation,
                "selection_kind": "real-validation-total",
                "qualified": False,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }, best_path)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    strict_load = model.load_state_dict(best["state_dict"], strict=True)
    model.to(device).eval()
    final_qualification = _run_final_qualification(
        model,
        validation_loader,
        config,
        device,
        amp_dtype,
        strict_missing_keys=list(strict_load.missing_keys),
        strict_unexpected_keys=list(strict_load.unexpected_keys),
    )
    architecture_participation_summary = _aggregate_component_participation(
        architecture_participation
    )
    best_synthetic = best.get("synthetic_sdf_validation") or {}
    selected_kind = str(best.get("selection_kind", ""))
    selected_is_geometry = selected_kind.startswith("synthetic-stageb-")
    selected_is_isolated = selected_kind.startswith("v1079-")
    if stop_after_phase in isolated_stop_phases and selected_is_isolated and state_path.is_file():
        # A subsequent gated resume must start from the same selected B1-B4
        # checkpoint that the external audit sees, not the final epoch state.
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        resume_state["state_dict"] = {key: value.detach().cpu() for key, value in best["state_dict"].items()}
        resume_state["selected_candidate_epoch"] = int(best["epoch"])
        resume_state["selected_candidate_phase"] = str(best.get("phase", "unknown"))
        resume_state["selected_candidate_kind"] = selected_kind
        _atomic_torch_save(resume_state, state_path)
        _status(
            f"  Synced staged resume state to {selected_kind} epoch {int(best['epoch']):03d}."
        )
    if stop_after_phase == "gate-proof" and selected_is_geometry and state_path.is_file():
        # The external Stage-B audit and the later selector resume must consume
        # the exact same frozen candidate. Preserve epoch/RNG/optimizer state,
        # but replace the model state with the selected structure+specialist
        # checkpoint. Selector parameters have not trained yet, so this is safe.
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        resume_state["state_dict"] = {key: value.detach().cpu() for key, value in best["state_dict"].items()}
        resume_state["selected_candidate_epoch"] = int(best["epoch"])
        resume_state["best_sdf_score"] = float(best["validation_total"])
        _atomic_torch_save(resume_state, state_path)
        _status(
            f"  Synced staged resume state to audited candidate epoch "
            f"{int(best['epoch']):03d}; selector will train from this exact Panel 3."
        )
    if stop_after_phase == "detail-reconstruction" and str(best.get("selection_kind", "")).startswith("heldout-detail-v105") and state_path.is_file():
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        resume_state["state_dict"] = {key: value.detach().cpu() for key, value in best["state_dict"].items()}
        resume_state["selected_detail_epoch"] = int(best["epoch"])
        resume_state["detail_qualified"] = str(best.get("selection_kind", "")) == "heldout-detail-v105-qualified"
        _atomic_torch_save(resume_state, state_path)
        _status(
            f"  Synced staged resume state to held-out detail epoch {int(best['epoch']):03d}; "
            "selector will consume this exact full candidate only after Stage C passes."
        )
    selected_is_detail = selected_kind.startswith("heldout-detail-v105")
    if selected_is_isolated:
        # For an isolated proof export, safety is the conjunction of every
        # prerequisite up to the exported stage. Reconstruction acceptance is
        # still decided by the external analytic/real-asset audit.
        stage_phase = str(best.get("phase", ""))
        if selected_kind.startswith("v1079-b2"):
            required_ok = bool(topology_bootstrapped and structure_qualified and render_qualified)
        elif selected_kind.startswith("v1079-b1b"):
            required_ok = bool(topology_bootstrapped and structure_qualified)
        elif selected_kind.startswith("v1079-b1a"):
            required_ok = bool(topology_bootstrapped)
        else:
            required_ok = bool(topology_bootstrapped and structure_qualified)
        acceptance_regression_fraction = float(
            history[int(best["epoch"]) - 1]["validation"].get("regression_fraction", 0.0)
        ) if history and 0 < int(best["epoch"]) <= len(history) else 0.0
        training_safety_pass = required_ok
    elif selected_is_geometry:
        selected_relative_regression = float(
            best_synthetic.get("sdf_zero_contour_relative_regression_fraction", 1.0)
        )
        selected_source_missing = float(
            best_synthetic.get("sdf_source_missing_contour_fraction", 1.0)
        )
        selected_predicted_missing = float(
            best_synthetic.get("sdf_predicted_missing_contour_fraction", 1.0)
        )
        training_safety_pass = (
            selected_relative_regression <= float(config.sdf_relative_regression_fraction)
            and selected_predicted_missing <= selected_source_missing + float(config.sdf_missing_contour_tolerance)
            and float(best_synthetic.get("sdf_stageb_topology_regression_fraction", 1.0)) == 0.0
        )
        acceptance_regression_fraction = selected_relative_regression
    elif selected_is_detail:
        detail_validation = best.get("detail_validation") or {}
        acceptance_regression_fraction = float(detail_validation.get("detail_regression_fraction", 1.0))
        training_safety_pass = (
            str(best.get("selection_kind", "")) == "heldout-detail-v105-qualified"
        )
    else:
        acceptance_regression_fraction = float(
            history[int(best["epoch"]) - 1]["validation"].get("regression_fraction", 1.0)
        )
        training_safety_pass = (
            acceptance_regression_fraction <= config.maximum_validation_regression_fraction
        )

    required_trained_components = (
        "geometry", "structural representation", "boundary/profile",
        "PhaseAwareSeamSR", "seam authority", "conditioned detail",
        "albedo physical head", "normal physical head", "material physical head",
        "confidence", "regret", "BenefitSelector",
    )
    component_training_complete = all(
        bool(architecture_participation_summary["components"].get(label, {}).get("trained", False))
        for label in required_trained_components
    )
    cache_qualification_passed = bool(
        cache_equivalence is None or cache_equivalence.get("passed", False)
    )
    if stop_after_phase is None:
        training_safety_pass = bool(
            training_safety_pass
            and final_qualification["passed"]
            and cache_qualification_passed
            and component_training_complete
            and selected_kind == "production-final-selector-qualified"
        )
    exported_selection_kind = selected_kind
    if stop_after_phase is None:
        exported_selection_kind = (
            "production-final" if training_safety_pass
            else "production-final-unqualified"
        )

    checkpoint = {
        "schema": MODEL_SCHEMA,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": config.to_dict(),
        "parameter_count": parameter_count(model),
        "model_sha256": model_hash(model),
        "best_epoch": int(best["epoch"]),
        "best_validation_total": float(best["validation_total"]),
        "selection_kind": exported_selection_kind,
        "source_selection_kind": selected_kind,
        "best_synthetic_sdf_validation": best.get("synthetic_sdf_validation"),
        "best_detail_validation": best.get("detail_validation"),
        "best_selector_validation": best.get("selector_validation"),
        "topology_bootstrapped": topology_bootstrapped,
        "b1b_classifier_qualified": b1b_classifier_qualified,
        "b1b_parameters_qualified": b1b_parameters_qualified,
        "b1b_stage_epoch": b1b_stage_epoch,
        "structure_qualified": structure_qualified,
        "render_qualified": render_qualified,
        "seam_reconstruction_qualified": seam_reconstruction_qualified,
        "seam_authority_qualified": seam_authority_qualified,
        "detail_qualified": detail_qualified,
        "architecture_participation": architecture_participation_summary,
        "final_qualification": final_qualification,
        "cache_equivalence": cache_equivalence,
        "component_training_complete": bool(component_training_complete),
        "acceptance_regression_fraction": acceptance_regression_fraction,
        "training_safety_pass": training_safety_pass,
        "acceptance_pass": training_safety_pass,
        "dataset_fingerprint": fingerprint,
        "history": history,
        "planned_epochs": config.total_epochs,
        "epochs_completed": len(history),
        "early_stopped": early_stopped_epoch is not None,
        "early_stop_epoch": early_stopped_epoch,
        "staged_stop_phase": stop_after_phase,
        "staged_stop_reached": bool(
            stop_after_phase is not None and len(history) < config.total_epochs
        ),
    }
    _atomic_torch_save(checkpoint, checkpoint_path)
    elapsed_minutes = (time.perf_counter() - started) / 60.0
    metadata = {
        "schema": MODEL_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpointSha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "modelSha256": checkpoint["model_sha256"],
        "bestEpoch": checkpoint["best_epoch"],
        "bestValidationTotal": checkpoint["best_validation_total"],
        "selectionKind": checkpoint["selection_kind"],
        "sourceSelectionKind": checkpoint["source_selection_kind"],
        "bestSyntheticSdfValidation": checkpoint.get("best_synthetic_sdf_validation"),
        "bestDetailValidation": checkpoint.get("best_detail_validation"),
        "bestSelectorValidation": checkpoint.get("best_selector_validation"),
        "ravenTrainOnly": bool(getattr(config, "raven_train_only_enabled", False)),
        "topologyBootstrapped": checkpoint.get("topology_bootstrapped", False),
        "geometryQualified": checkpoint.get("structure_qualified", False),
        "renderQualified": checkpoint.get("render_qualified", False),
        "redrawQualified": checkpoint.get("render_qualified", False),
        "seamReconstructionQualified": checkpoint.get("seam_reconstruction_qualified", False),
        "seamAuthorityQualified": checkpoint.get("seam_authority_qualified", False),
        "detailQualified": checkpoint.get("detail_qualified", False),
        "architectureParticipation": checkpoint["architecture_participation"],
        "finalQualification": checkpoint["final_qualification"],
        "cacheEquivalence": checkpoint["cache_equivalence"],
        "componentTrainingComplete": checkpoint["component_training_complete"],
        "acceptanceRegressionFraction": checkpoint["acceptance_regression_fraction"],
        "trainingSafetyPass": checkpoint["training_safety_pass"],
        "acceptancePass": checkpoint["acceptance_pass"],
        "trainingMinutes": elapsed_minutes,
        "plannedEpochs": config.total_epochs,
        "epochsCompleted": len(history),
        "earlyStopped": early_stopped_epoch is not None,
        "earlyStopEpoch": early_stopped_epoch,
        "stagedStopPhase": stop_after_phase,
        "stagedStopReached": bool(
            stop_after_phase is not None and len(history) < config.total_epochs
        ),
        "datasetFingerprint": fingerprint,
        "architecture": architecture_summary(model),
        "performance": history[-1].get("performance", {}),
        "qualityContract": {
            "physicalReconstruction": True,
            "illustratedPresentation": False,
            "fullMapProposals": True,
            "contourSdf": True,
            "upscaleFactor": 4,
        },
    }
    _atomic_json(metadata, metadata_path)
    print("=" * 68)
    print("NSAMDR RAVEN PRODUCTION PARAMETRIC CHECKPOINT READY")
    print(f"Checkpoint               : {checkpoint_path}")
    print(f"Metadata                 : {metadata_path}")
    print(f"Best validation total    : {checkpoint['best_validation_total']:.6f} (epoch {checkpoint['best_epoch']:03d})")
    print(f"Selected regressions     : {checkpoint['acceptance_regression_fraction']*100.0:.2f}%")
    print(f"Training safety gate     : {'PASS' if checkpoint['training_safety_pass'] else 'REJECT'}")
    print("Reconstruction acceptance: PENDING same-renderer deterministic redraw audit")
    print(f"Training time            : {elapsed_minutes:.1f} minutes")
    print("=" * 68)
    return metadata
