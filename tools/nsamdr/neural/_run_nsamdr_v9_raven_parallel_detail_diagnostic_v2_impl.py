#!/usr/bin/env python3
"""Fast integration proof for the V12.3 parallel direct-detail production path.

The isolated Direct Residual Capacity proof established that the existing detail
decoder can recover the hard Raven patch. This proof trains the now-independent
detail specialist through that exact baseline-relative path, then verifies that
the trained candidate is numerically preserved by the full production forward.
It does the same for BenefitSelector.

Expensive geometry/seam modules are therefore executed only for parity/final
verification, not 1,000+ times while training an independent specialist.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_nsamdr_v9_raven_direct_residual_diagnostic as direct
import run_nsamdr_v9_raven_micro_overfit as legacy
from v9.application.backend import TrainingBackend
from v9.inference import resolve_device
from v9.model import FidelityResidualNetV9, MODEL_SCHEMA, UPSCALE_FACTOR
import v9.parallel_detail_contract as parallel_contract


REPORT_SCHEMA = "NSAMDR_RAVEN_PARALLEL_DETAIL_INTEGRATION_V2"
PARITY_TOLERANCE = 1.0e-4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast proof that direct-detail and selector survive production composition."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--detail-steps", type=int, default=1536)
    parser.add_argument("--selector-steps", type=int, default=512)
    parser.add_argument("--detail-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--head-lr-multiplier", type=float, default=3.0)
    parser.add_argument("--selector-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--required-edge-recovery", type=float, default=0.50)
    parser.add_argument("--required-global-recovery", type=float, default=0.25)
    parser.add_argument("--required-retention", type=float, default=0.85)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--open-result", action="store_true")
    return parser


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=value.device, dtype=value.dtype, non_blocking=True)
    if weight.shape[1] == 1 and value.shape[1] != 1:
        weight = weight.expand(-1, value.shape[1], -1, -1)
    return (value.float() * weight.float()).sum() / weight.float().sum().clamp_min(1.0)


def _gradient_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    from v9.contours import sobel_tensor

    pgx, pgy = sobel_tensor(prediction.float().mean(dim=1, keepdim=True))
    tgx, tgy = sobel_tensor(target.float().mean(dim=1, keepdim=True))
    return _weighted_mean((pgx - tgx).abs() + (pgy - tgy).abs(), weight)


def _raw_direct_candidate(
    model: FidelityResidualNetV9,
    inputs: torch.Tensor,
    baseline_albedo: torch.Tensor,
    baseline_normal: torch.Tensor,
    baseline_material: torch.Tensor,
    residual_cap: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Run the original production detail decoder exactly as the passing V2 proof did."""
    original = parallel_contract._ORIGINAL_DETAIL_FORWARD
    if original is None:
        raise RuntimeError("parallel detail model contract did not capture original detail forward")
    geometry = torch.zeros(
        (
            inputs.shape[0],
            model.detail_net.GEOMETRY_CHANNELS,
            baseline_albedo.shape[-2],
            baseline_albedo.shape[-1],
        ),
        device=inputs.device,
        dtype=baseline_albedo.dtype,
    )
    detail = original(
        model.detail_net,
        inputs,
        baseline_albedo,
        baseline_normal,
        baseline_material,
        geometry,
    )
    raw_delta = detail["albedo_raw"].float() * float(residual_cap)
    candidate = (baseline_albedo.float() + raw_delta).clamp(0.0, 1.0)
    return candidate, detail, raw_delta


def _selector_evidence(
    model: FidelityResidualNetV9,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    detail: dict[str, torch.Tensor],
) -> torch.Tensor:
    source_support = model._source_edge_support(
        inputs, int(model.config.geometry_edge_support_radius)
    )
    observed_support = F.interpolate(
        source_support,
        scale_factor=UPSCALE_FACTOR,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    one = torch.zeros_like(observed_support)
    two = torch.zeros(
        (
            inputs.shape[0],
            2,
            observed_support.shape[-2],
            observed_support.shape[-1],
        ),
        device=inputs.device,
        dtype=observed_support.dtype,
    )
    confidence = torch.sigmoid(detail["confidence_logits"].float())
    regret = torch.sigmoid(detail["regret_logits"].float())
    return model._selector_features(
        baseline,
        candidate,
        one,
        two,
        one,
        one,
        observed_support,
        one,
        confidence,
        regret,
    )


def _optimal_gate(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta = candidate.float() - baseline.float()
    desired = target.float() - baseline.float()
    numerator = (delta * desired).sum(dim=1, keepdim=True)
    denominator = delta.square().sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    gate = (numerator / denominator).clamp(0.0, 1.0).detach()
    motion = delta.abs().mean(dim=1, keepdim=True).detach()
    return gate, motion


def _retention(candidate: float, final: float) -> float:
    if candidate <= 1.0e-8:
        return 0.0
    return final / candidate


def _full_production(
    model: FidelityResidualNetV9,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Use FP32 for the parity assertion so AMP quantisation cannot hide a mismatch."""
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=batch["input"].device.type,
        enabled=False,
    ):
        return model(batch["input"])


def _write_probe(
    path: Path,
    target: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    final: torch.Tensor,
    candidate_metrics: dict[str, float],
    final_metrics: dict[str, float],
    title: str,
) -> None:
    from PIL import Image, ImageDraw

    panels = (
        ("A TARGET", direct._rgb(target)),
        ("B BASELINE", direct._rgb(baseline)),
        ("D DIRECT DETAIL", direct._rgb(candidate)),
        ("F FINAL SELECTED", direct._rgb(final)),
    )
    h, w = panels[0][1].shape[:2]
    header = 66
    canvas = Image.new("RGB", (w * 4, h + header), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = index * w
        canvas.paste(Image.fromarray(image, mode="RGB"), (x, header))
        draw.text((x + 5, 6), label, fill=(245, 245, 245))
        if index == 2:
            draw.text(
                (x + 5, 27),
                f"edge {candidate_metrics['edgeRecovery']:+.1%}",
                fill=(225, 225, 225),
            )
            draw.text(
                (x + 5, 45),
                f"global {candidate_metrics['globalRecovery']:+.1%}",
                fill=(225, 225, 225),
            )
        elif index == 3:
            draw.text(
                (x + 5, 27),
                f"edge {final_metrics['edgeRecovery']:+.1%}",
                fill=(225, 225, 225),
            )
            draw.text(
                (x + 5, 45),
                f"global {final_metrics['globalRecovery']:+.1%}",
                fill=(225, 225, 225),
            )
    draw.text((5, h + header - 16), title, fill=(210, 210, 210))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _detail_optimizer(
    model: FidelityResidualNetV9,
    learning_rate: float,
    head_multiplier: float,
) -> torch.optim.Optimizer:
    head_ids = {id(parameter) for parameter in model.detail_net.albedo_head.parameters()}
    head = [
        parameter
        for parameter in model.detail_net.parameters()
        if id(parameter) in head_ids
    ]
    body = [
        parameter
        for parameter in model.detail_net.parameters()
        if id(parameter) not in head_ids
    ]
    return torch.optim.AdamW(
        [
            {"params": body, "lr": float(learning_rate)},
            {"params": head, "lr": float(learning_rate) * float(head_multiplier)},
        ],
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )


def _train_detail(
    *,
    model: FidelityResidualNetV9,
    inputs: torch.Tensor,
    target: torch.Tensor,
    target_edge: torch.Tensor,
    baseline_albedo: torch.Tensor,
    baseline_normal: torch.Tensor,
    baseline_material: torch.Tensor,
    config: Any,
    steps: int,
    learning_rate: float,
    head_multiplier: float,
    required_edge: float,
    required_global: float,
    amp_dtype: torch.dtype,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.detail_net.parameters():
        parameter.requires_grad_(True)

    optimizer = _detail_optimizer(model, learning_rate, head_multiplier)
    use_amp = inputs.device.type == "cuda"
    use_scaler = use_amp and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_scaler, init_scale=config.amp_initial_scale
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_scaler, init_scale=config.amp_initial_scale
        )

    edge_weight = (0.20 + target_edge * 3.80).detach()
    baseline_error = (baseline_albedo.detach().float() - target).abs().mean(
        dim=1, keepdim=True
    )
    cap = float(config.detail_albedo_max_delta)
    desired_residual = (target.detach() - baseline_albedo.detach()).clamp(-cap, cap)

    best_score = -float("inf")
    best_state = copy.deepcopy(model.detail_net.state_dict())
    best_metrics = direct._metrics(
        baseline_albedo, baseline_albedo, target, target_edge
    )
    best_step = 0
    interval = max(16, min(64, int(steps) // 24))
    model.detail_net.train()

    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=inputs.device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            candidate, _detail, raw_delta = _raw_direct_candidate(
                model,
                inputs,
                baseline_albedo,
                baseline_normal,
                baseline_material,
                cap,
            )
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            error = (candidate.float() - target).abs().mean(dim=1, keepdim=True)
            global_reconstruction = error.mean()
            edge_reconstruction = _weighted_mean(error, edge_weight)
            regret = _weighted_mean(F.relu(error - baseline_error), edge_weight)
            gradient = _gradient_error(candidate.float(), target, edge_weight)
            residual_supervision = (
                raw_delta.float() - desired_residual
            ).abs().mean()
            total = (
                global_reconstruction * 4.0
                + edge_reconstruction * 12.0
                + gradient * 5.0
                + regret * 16.0
                + residual_supervision * 8.0
            ).float()
        if not bool(torch.isfinite(total).item()):
            raise RuntimeError(f"non-finite direct detail loss step={step}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.detail_net.parameters(), float(config.gradient_clip_norm)
        )
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError(f"non-finite direct detail gradient step={step}")
        scaler.step(optimizer)
        scaler.update()

        if step != 1 and step != int(steps) and step % interval != 0:
            continue

        model.detail_net.eval()
        with torch.no_grad(), torch.autocast(
            device_type=inputs.device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            evaluated, _detail, _delta = _raw_direct_candidate(
                model,
                inputs,
                baseline_albedo,
                baseline_normal,
                baseline_material,
                cap,
            )
        current = direct._metrics(evaluated, baseline_albedo, target, target_edge)
        score = float(current["edgeRecovery"]) + 0.5 * float(
            current["globalRecovery"]
        )
        rows.append(
            {
                "phase": "detail-fast",
                "step": step,
                "loss": float(total.detach().cpu().item()),
                "edgeRecovery": float(current["edgeRecovery"]),
                "globalRecovery": float(current["globalRecovery"]),
                "selectorMean": "",
            }
        )
        print(
            f"[parallel-v2/detail] {step:4d}/{int(steps):4d} "
            f"edge={current['edgeRecovery']:+.1%} "
            f"global={current['globalRecovery']:+.1%}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_step = step
            best_metrics = dict(current)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.detail_net.state_dict().items()
            }
        model.detail_net.train()
        if (
            float(current["edgeRecovery"]) >= float(required_edge)
            and float(current["globalRecovery"]) >= float(required_global)
        ):
            break

    model.detail_net.load_state_dict(best_state, strict=True)
    return best_metrics, best_step


def _train_selector(
    *,
    model: FidelityResidualNetV9,
    inputs: torch.Tensor,
    target: torch.Tensor,
    target_edge: torch.Tensor,
    baseline_albedo: torch.Tensor,
    candidate: torch.Tensor,
    detail: dict[str, torch.Tensor],
    config: Any,
    steps: int,
    learning_rate: float,
    required_retention: float,
    candidate_metrics: dict[str, float],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], int, float]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.benefit_selector.parameters():
        parameter.requires_grad_(True)

    fixed_candidate = candidate.detach().float()
    fixed_detail = {key: value.detach() for key, value in detail.items()}
    features = _selector_evidence(
        model, inputs, baseline_albedo, fixed_candidate, fixed_detail
    ).detach()
    oracle, motion = _optimal_gate(baseline_albedo, fixed_candidate, target)
    edge_weight = (0.20 + target_edge * 3.80).detach()
    selector_weight = (
        0.10
        + target_edge * 2.90
        + (motion / 0.02).clamp(0.0, 1.0)
    ).detach()
    baseline_error = (baseline_albedo.float() - target).abs().mean(
        dim=1, keepdim=True
    )
    optimizer = torch.optim.AdamW(
        model.benefit_selector.parameters(),
        lr=float(learning_rate),
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )

    best_score = -float("inf")
    best_state = copy.deepcopy(model.benefit_selector.state_dict())
    best_metrics = direct._metrics(
        baseline_albedo, baseline_albedo, target, target_edge
    )
    best_gate = float(torch.sigmoid(model.benefit_selector(features)).mean().item())
    best_step = 0
    interval = max(8, min(32, int(steps) // 20))
    model.benefit_selector.train()

    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model.benefit_selector(features)
        probability = torch.sigmoid(logits.float()).clamp(1.0e-5, 1.0 - 1.0e-5)
        final = (
            baseline_albedo.float() * (1.0 - probability)
            + fixed_candidate * probability
        ).clamp(0.0, 1.0)
        final_error = (final - target).abs().mean(dim=1, keepdim=True)
        selector_bce = _weighted_mean(
            F.binary_cross_entropy(probability, oracle, reduction="none"),
            selector_weight,
        )
        reconstruction = _weighted_mean(final_error, edge_weight)
        regret = _weighted_mean(
            F.relu(final_error - baseline_error), edge_weight
        )
        gradient = _gradient_error(final, target, edge_weight)
        total = (
            selector_bce * float(config.benefit_selector_weight)
            + reconstruction * float(config.albedo_weight) * 2.0
            + gradient * float(config.albedo_gradient_weight)
            + regret * float(config.regret_weight) * 3.0
        ).float()
        if not bool(torch.isfinite(total).item()):
            raise RuntimeError(f"non-finite selector loss step={step}")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.benefit_selector.parameters(), float(config.gradient_clip_norm)
        )
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError(f"non-finite selector gradient step={step}")
        optimizer.step()

        if step != 1 and step != int(steps) and step % interval != 0:
            continue

        model.benefit_selector.eval()
        with torch.no_grad():
            probability_eval = torch.sigmoid(
                model.benefit_selector(features).float()
            )
            final_eval = (
                baseline_albedo.float() * (1.0 - probability_eval)
                + fixed_candidate * probability_eval
            ).clamp(0.0, 1.0)
        current = direct._metrics(
            final_eval, baseline_albedo, target, target_edge
        )
        gate_mean = float(probability_eval.mean().item())
        score = float(current["edgeRecovery"]) + 0.5 * float(
            current["globalRecovery"]
        )
        rows.append(
            {
                "phase": "selector-fast",
                "step": step,
                "loss": float(total.detach().cpu().item()),
                "edgeRecovery": float(current["edgeRecovery"]),
                "globalRecovery": float(current["globalRecovery"]),
                "selectorMean": gate_mean,
            }
        )
        print(
            f"[parallel-v2/selector] {step:4d}/{int(steps):4d} "
            f"edge={current['edgeRecovery']:+.1%} "
            f"global={current['globalRecovery']:+.1%} gate={gate_mean:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_step = step
            best_metrics = dict(current)
            best_gate = gate_mean
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.benefit_selector.state_dict().items()
            }
        edge_retention = _retention(
            float(candidate_metrics["edgeRecovery"]),
            float(current["edgeRecovery"]),
        )
        global_retention = _retention(
            float(candidate_metrics["globalRecovery"]),
            float(current["globalRecovery"]),
        )
        model.benefit_selector.train()
        if (
            edge_retention >= float(required_retention)
            and global_retention >= float(required_retention)
        ):
            break

    model.benefit_selector.load_state_dict(best_state, strict=True)
    return best_metrics, best_step, best_gate


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    if int(args.tile_size) < 32 or int(args.tile_size) % 16 != 0:
        raise SystemExit("--tile-size must be >=32 and divisible by 16")
    if int(args.detail_steps) < 64 or int(args.selector_steps) < 32:
        raise SystemExit("detail/selector step budgets are too small")
    if float(args.detail_learning_rate) <= 0.0 or float(args.selector_learning_rate) <= 0.0:
        raise SystemExit("learning rates must be >0")
    if float(args.head_lr_multiplier) < 1.0:
        raise SystemExit("--head-lr-multiplier must be >=1")

    legacy._ensure_raven_dataset(
        root,
        root / legacy.RAVEN_CONFIG,
        args.shared_cache,
        bool(args.rebuild_dataset),
    )
    args.steps_per_epoch = 32
    config, _raven = legacy._micro_config(root, args)
    config.training_activation_checkpointing = False
    config.validate()
    manifest = legacy.load_dataset_manifest(root, config)
    sample, patch_metadata = legacy._select_hard_patch(manifest, config)
    batch_cpu = legacy._batch(sample)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        root
        / "artifacts/nsamdr/parallel_detail_diagnostics"
        / f"PARALLEL_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "selected_patch.json").write_text(
        json.dumps(patch_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    backend = TrainingBackend()
    _ = backend
    import v9.training as training

    service = training._training_service
    device = resolve_device(config, args.device)
    service._configure_cuda(config, device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model = FidelityResidualNetV9(config).to(device)
    service._validate_v992_architecture_contract(model.architecture_contract())
    moved = service._move_batch(batch_cpu, device, channels_last=False)
    target = moved["target_albedo"].float()
    target_edge = moved["target_edge"].float().clamp(0.0, 1.0)
    baseline_albedo, baseline_normal, baseline_material = direct._baseline(
        model, moved["input"]
    )
    amp_dtype = service._resolve_amp_dtype(config, device)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    print("=" * 78, flush=True)
    print("NSAMDR RAVEN PARALLEL DETAIL INTEGRATION PROOF V2", flush=True)
    print("Authority                : diagnostic only / NON-PROMOTABLE", flush=True)
    print(
        "Training                 : independent detail + selector specialists only",
        flush=True,
    )
    print(
        "Production verification  : full model(input) candidate/final parity",
        flush=True,
    )
    print(
        f"Detail LR body/head      : {float(args.detail_learning_rate):.3g} / "
        f"{float(args.detail_learning_rate) * float(args.head_lr_multiplier):.3g}",
        flush=True,
    )
    print(
        f"Detail/selector steps    : {int(args.detail_steps)} / {int(args.selector_steps)}",
        flush=True,
    )
    print(f"Artifacts                : {output_dir}", flush=True)
    print("=" * 78, flush=True)

    detail_metrics, detail_best_step = _train_detail(
        model=model,
        inputs=moved["input"],
        target=target,
        target_edge=target_edge,
        baseline_albedo=baseline_albedo,
        baseline_normal=baseline_normal,
        baseline_material=baseline_material,
        config=config,
        steps=int(args.detail_steps),
        learning_rate=float(args.detail_learning_rate),
        head_multiplier=float(args.head_lr_multiplier),
        required_edge=float(args.required_edge_recovery),
        required_global=float(args.required_global_recovery),
        amp_dtype=amp_dtype,
        rows=rows,
    )

    model.detail_net.eval()
    with torch.no_grad():
        direct_candidate, direct_detail, _raw_delta = _raw_direct_candidate(
            model,
            moved["input"],
            baseline_albedo,
            baseline_normal,
            baseline_material,
            float(config.detail_albedo_max_delta),
        )
    production_after_detail = _full_production(model, moved)
    production_candidate = production_after_detail["detail_candidate_albedo"].float()
    candidate_parity_max_abs = float(
        (production_candidate - direct_candidate.float()).abs().max().item()
    )
    production_candidate_metrics = direct._metrics(
        production_candidate, baseline_albedo, target, target_edge
    )
    candidate_pass = (
        float(production_candidate_metrics["edgeRecovery"])
        >= float(args.required_edge_recovery)
        and float(production_candidate_metrics["globalRecovery"])
        >= float(args.required_global_recovery)
        and candidate_parity_max_abs <= PARITY_TOLERANCE
    )
    print(
        f"[parallel-v2/parity] candidate maxAbs={candidate_parity_max_abs:.3g} "
        f"edge={production_candidate_metrics['edgeRecovery']:+.1%} "
        f"global={production_candidate_metrics['globalRecovery']:+.1%}",
        flush=True,
    )

    selector_fast_metrics, selector_best_step, selector_fast_mean = _train_selector(
        model=model,
        inputs=moved["input"],
        target=target,
        target_edge=target_edge,
        baseline_albedo=baseline_albedo,
        candidate=direct_candidate,
        detail=direct_detail,
        config=config,
        steps=int(args.selector_steps),
        learning_rate=float(args.selector_learning_rate),
        required_retention=float(args.required_retention),
        candidate_metrics=production_candidate_metrics,
        rows=rows,
    )

    production_final = _full_production(model, moved)
    final_candidate = production_final["detail_candidate_albedo"].float()
    final = production_final["albedo"].float()
    final_candidate_metrics = direct._metrics(
        final_candidate, baseline_albedo, target, target_edge
    )
    final_metrics = direct._metrics(final, baseline_albedo, target, target_edge)

    with torch.no_grad():
        fast_candidate, fast_detail, _ = _raw_direct_candidate(
            model,
            moved["input"],
            baseline_albedo,
            baseline_normal,
            baseline_material,
            float(config.detail_albedo_max_delta),
        )
        fast_features = _selector_evidence(
            model,
            moved["input"],
            baseline_albedo,
            fast_candidate,
            fast_detail,
        )
        fast_gate = torch.sigmoid(model.benefit_selector(fast_features).float())
        fast_final = (
            baseline_albedo.float() * (1.0 - fast_gate)
            + fast_candidate.float() * fast_gate
        ).clamp(0.0, 1.0)
    final_parity_max_abs = float((final - fast_final).abs().max().item())
    edge_retention = _retention(
        float(final_candidate_metrics["edgeRecovery"]),
        float(final_metrics["edgeRecovery"]),
    )
    global_retention = _retention(
        float(final_candidate_metrics["globalRecovery"]),
        float(final_metrics["globalRecovery"]),
    )
    selector_pass = (
        edge_retention >= float(args.required_retention)
        and global_retention >= float(args.required_retention)
        and final_parity_max_abs <= PARITY_TOLERANCE
        and float(final_metrics["edgeRecovery"]) > 0.0
        and float(final_metrics["globalRecovery"]) > 0.0
    )
    passed = candidate_pass and selector_pass
    status = (
        "passed"
        if passed
        else (
            "failed-production-candidate-parity"
            if candidate_parity_max_abs > PARITY_TOLERANCE
            else (
                "failed-parallel-detail-candidate"
                if not candidate_pass
                else (
                    "failed-production-final-parity"
                    if final_parity_max_abs > PARITY_TOLERANCE
                    else "failed-selector-retention"
                )
            )
        )
    )

    final_probe = output_dir / "parallel_probe_final.png"
    _write_probe(
        final_probe,
        target,
        baseline_albedo,
        final_candidate,
        final,
        final_candidate_metrics,
        final_metrics,
        "V12.3 FAST SPECIALIST TRAINING -> FULL PRODUCTION FORWARD",
    )
    with (output_dir / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "phase",
                "step",
                "loss",
                "edgeRecovery",
                "globalRecovery",
                "selectorMean",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "selectionKind": "diagnostic-non-promotable",
        "promotable": False,
        "qualifiedProductionCheckpoint": False,
        "sourceRevision": legacy._git_state(root),
        "modelSchema": MODEL_SCHEMA,
        "architecture": model.architecture_contract(),
        "detailStepsRequested": int(args.detail_steps),
        "detailBestStep": int(detail_best_step),
        "selectorStepsRequested": int(args.selector_steps),
        "selectorBestStep": int(selector_best_step),
        "requiredEdgeRecovery": float(args.required_edge_recovery),
        "requiredGlobalRecovery": float(args.required_global_recovery),
        "requiredRetention": float(args.required_retention),
        "detailFastPath": detail_metrics,
        "productionCandidate": production_candidate_metrics,
        "finalCandidate": final_candidate_metrics,
        "final": final_metrics,
        "candidatePass": bool(candidate_pass),
        "selectorPass": bool(selector_pass),
        "candidateParityMaxAbs": candidate_parity_max_abs,
        "finalParityMaxAbs": final_parity_max_abs,
        "parityTolerance": PARITY_TOLERANCE,
        "edgeRetention": float(edge_retention),
        "globalRetention": float(global_retention),
        "selectorFastMean": float(selector_fast_mean),
        "selectorProductionMean": float(
            production_final["benefit_selector_probability"]
            .detach()
            .float()
            .mean()
            .item()
        ),
        "selectorFastBest": selector_fast_metrics,
        "elapsedSeconds": time.perf_counter() - started,
        "finalProbeSheet": str(final_probe.resolve()),
    }
    (output_dir / "parallel_detail_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    zip_path = legacy._zip_diagnostics(output_dir)

    print("=" * 78, flush=True)
    print(
        f"PARALLEL DETAIL INTEGRATION V2 : {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    print(
        f"Production D edge/global       : "
        f"{final_candidate_metrics['edgeRecovery']:+.2%} / "
        f"{final_candidate_metrics['globalRecovery']:+.2%}",
        flush=True,
    )
    print(
        f"Production F edge/global       : "
        f"{final_metrics['edgeRecovery']:+.2%} / "
        f"{final_metrics['globalRecovery']:+.2%}",
        flush=True,
    )
    print(
        f"Selector retention edge/global : {edge_retention:.1%} / {global_retention:.1%}",
        flush=True,
    )
    print(
        f"Candidate/final parity maxAbs  : "
        f"{candidate_parity_max_abs:.3g} / {final_parity_max_abs:.3g}",
        flush=True,
    )
    print(f"Diagnostics                     : {zip_path}", flush=True)
    print("=" * 78, flush=True)

    if args.open_result:
        direct._open_result(final_probe)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())