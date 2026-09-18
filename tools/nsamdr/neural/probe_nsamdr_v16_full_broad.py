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
from v16.broad_prior import AuthorityBalancedSRDataset
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


def _proof_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: V16Config,
) -> torch.Tensor:
    """Broad proof loss for qualified authored albedo+normal supervision."""
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
    residual = F.l1_loss(outputs["predicted_residual_albedo"].float(), target_a)
    residual = residual + 0.25 * F.l1_loss(
        outputs["predicted_residual_normal"].float(),
        target_n,
    )
    return reconstruction + 0.50 * gradient + 0.35 * normal + residual


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
    losses: list[float] = []
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
            loss = _proof_loss(outputs, batch, config)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach().float().cpu().item()))

        if offset == 1 or global_step % 32 == 0 or global_step == end_step:
            elapsed = max(time.monotonic() - started, 1.0e-6)
            rate = offset / elapsed
            eta = (segment_steps - offset) / max(rate, 1.0e-6)
            print(
                f"[full-broad] step {global_step:4d}/{end_step} "
                f"loss={losses[-1]:.6f} elapsed={elapsed/60.0:.1f}m "
                f"segment-eta={eta/60.0:.1f}m",
                flush=True,
            )

    result: dict[str, Any] = {
        "fromStep": int(start_step),
        "toStep": int(end_step),
        "segmentSteps": int(segment_steps),
        "firstLoss": losses[0],
        "finalLoss": losses[-1],
        "medianTailLoss": float(statistics.median(losses[-min(32, len(losses)) :])),
        "elapsedSeconds": time.monotonic() - started,
        "availableTrainAuthorityCount": int(dataset.authority_count),
        "visitedAuthorityCount": _visited_authorities(dataset, start_step, end_step),
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
    lr = batch["lr_albedo"].float()

    target_rgb = _rgb_u8(target)
    baseline_rgb = _rgb_u8(baseline)
    candidate_rgb = _rgb_u8(candidate)
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
    error_max = torch.maximum(
        baseline_error.amax(),
        candidate_error.amax(),
    ).clamp_min(1.0e-8)
    baseline_error_u8 = np.round(
        (baseline_error / error_max)[0, 0].detach().cpu().numpy() * 255.0
    ).astype(np.uint8)
    candidate_error_u8 = np.round(
        (candidate_error / error_max)[0, 0].detach().cpu().numpy() * 255.0
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
    }
    metadata = {
        "authorityId": authority,
        "cropId": crop,
        "recordPath": str(record.get("path") or ""),
        "sampleIndex": int(sample_index),
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
    samples: int,
    device: torch.device,
    seed: int,
    precision: str,
    preview_root: Path | None = None,
    preview_samples: int = 0,
) -> dict[str, Any]:
    dataset = AuthorityBalancedSRDataset(
        manifest,
        config,
        "validation",
        samples,
        seed=seed,
        degradation="clean",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    metrics: list[dict[str, float]] = []
    previews: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for sample_index, batch in enumerate(loader):
            record_index = int(batch["record_index"][0].item())
            record = dataset.records[record_index]
            batch = _to_device(batch, device)
            with _autocast(device, precision):
                outputs = model(
                    batch["lr_albedo"],
                    batch["lr_normal"],
                    batch["lr_material"],
                )
            metrics.append(sample_metrics(outputs, batch, final=False))
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

    summary: dict[str, Any] = {
        "sampleCount": len(metrics),
        "heldOutAuthorityCount": dataset.selected_authority_count(),
        "availableHeldOutAuthorityCount": dataset.authority_count,
        "materialMetricQualified": False,
        "previewArtifacts": previews,
    }
    for key in METRIC_KEYS:
        summary[f"median_{key}"] = float(
            statistics.median(float(item[key]) for item in metrics)
        )
    return summary


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
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "step": int(step),
            "seed": int(seed),
            "manifest": str(manifest_path),
            "config": config.to_dict(),
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
) -> Path:
    final = curve[-1]
    report = {
        "schema": SCHEMA,
        "promotable": False,
        "manifest": str(manifest_path),
        "authoritySplit": manifest.get("authoritySplit"),
        "samplingPolicy": "authority-balanced-complete-cycle-before-repeat",
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
        f"Held-out authorities: {final['validation']['heldOutAuthorityCount']}",
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
        run_dir = checkpoint_source.parent
        if int(args.hr_size) != config.train_hr_size:
            raise RuntimeError(
                f"resume HR size is {config.train_hr_size}, not requested {args.hr_size}"
            )
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
        curve: list[dict[str, Any]] = []
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
    print(f"Preview samples   : {args.preview_samples}", flush=True)

    checkpoint_path = run_dir / "resume_checkpoint.pt"

    if args.preview_only:
        if not args.resume:
            raise RuntimeError("--preview-only requires --resume")
        preview_root = run_dir / "previews" / f"step_{start_step:06d}"
        validation = _evaluate(
            model,
            manifest,
            config,
            samples=validation_samples,
            device=device,
            seed=seed + 7001,
            precision=args.amp_precision,
            preview_root=preview_root,
            preview_samples=int(args.preview_samples),
        )
        preview_report = {
            "schema": "NSAMDR_V16_FULL_BROAD_PREVIEW_V1",
            "step": int(start_step),
            "validation": validation,
            "previewRoot": str(preview_root.resolve()),
        }
        preview_report_path = run_dir / f"preview_step_{start_step:06d}.json"
        preview_report_path.write_text(
            json.dumps(preview_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
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
        )
        preview_root = run_dir / "previews" / f"step_{end_step:06d}"
        validation = _evaluate(
            model,
            manifest,
            config,
            samples=validation_samples,
            device=device,
            seed=seed + 7001,
            precision=args.amp_precision,
            preview_root=preview_root,
            preview_samples=int(args.preview_samples),
        )
        item = {
            "step": int(end_step),
            "training": training,
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
        )
        start_step = end_step

        print(
            f"Checkpoint {end_step}: "
            f"global={validation['median_global_recovery']*100:+.2f}% "
            f"edge={validation['median_edge_recovery']*100:+.2f}% "
            f"grad={validation['median_gradient_recovery']*100:+.2f}% "
            f"lattice={validation['median_lattice_cell_excess']*100:+.2f}%",
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
            "Resume later with 512,1024,2048."
        ),
    )
    value.add_argument("--hr-size", type=int, default=512)
    value.add_argument(
        "--validation-samples",
        type=int,
        default=0,
        help="0 evaluates one sample from every held-out authority",
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
        "--resume",
        default="",
        help="resume_checkpoint.pt from an earlier full broad proof",
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
