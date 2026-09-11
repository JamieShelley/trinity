#!/usr/bin/env python3
"""V13.1 Raven SR visual-fidelity proof with quality-first stopping/selection."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import _run_nsamdr_v9_raven_micro_diagnostic_impl as micro_impl
import run_nsamdr_v13_raven_sr_diagnostic as v13
import run_nsamdr_v9_raven_direct_residual_diagnostic as direct
import run_nsamdr_v9_raven_micro_overfit as legacy
from v9.application.backend import TrainingBackend
from v9.baseline_relative_specialist_contract import (
    PROTECTED_PRESERVATION_REQUIRED,
    protected_preservation_terms,
)
from v9.diagnostics_layout import run_consolidated_diagnostic
from v9.inference import resolve_device
from v9.model import FidelityResidualNetV9, MODEL_SCHEMA
from v9 import parallel_specialist_safety_contract as safety
from v9 import sr_first_contract as sr
from v9 import sr_first_quality_contract as quality


REPORT_SCHEMA = "NSAMDR_RAVEN_SR_FIRST_V13_1"
PARITY_TOLERANCE = v13.PARITY_TOLERANCE


def _parser():
    parser = v13._parser()
    parser.set_defaults(
        required_edge_recovery=quality.SR_REQUIRED_EDGE_RECOVERY,
        required_global_recovery=quality.SR_REQUIRED_GLOBAL_RECOVERY,
        required_retention=quality.SR_REQUIRED_SELECTOR_RETENTION,
    )
    parser.add_argument(
        "--required-gradient-recovery",
        type=float,
        default=quality.SR_REQUIRED_GRADIENT_RECOVERY,
    )
    return parser


def _train_sr(
    *,
    model: FidelityResidualNetV9,
    inputs: torch.Tensor,
    baseline_albedo: torch.Tensor,
    baseline_normal: torch.Tensor,
    baseline_material: torch.Tensor,
    target_albedo: torch.Tensor,
    target_normal: torch.Tensor,
    target_material: torch.Tensor,
    target_edge: torch.Tensor,
    config: Any,
    steps: int,
    learning_rate: float,
    head_multiplier: float,
    required_edge: float,
    required_global: float,
    required_gradient: float,
    amp_dtype: torch.dtype,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], int, int, bool]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.detail_net.parameters():
        parameter.requires_grad_(True)

    optimizer = v13._detail_optimizer(model, learning_rate, head_multiplier)
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
    baseline_error = (baseline_albedo.detach().float() - target_albedo).abs().mean(
        dim=1, keepdim=True
    )
    cap = float(config.detail_albedo_max_delta)
    desired_residual = (target_albedo.detach() - baseline_albedo.detach()).clamp(
        -cap, cap
    )

    best_score = -float("inf")
    best_state = copy.deepcopy(model.detail_net.state_dict())
    best_metrics = v13._candidate_metrics(
        baseline_albedo,
        baseline_normal,
        baseline_material,
        baseline_albedo,
        baseline_normal,
        baseline_material,
        target_albedo,
        target_normal,
        target_material,
        target_edge,
    )
    best_step = 0
    interval = max(16, min(64, int(steps) // 24))
    plateau_best = -float("inf")
    plateau_stale = 0
    plateau_stopped = False
    steps_completed = 0
    model.detail_net.train()

    for step in range(1, int(steps) + 1):
        steps_completed = step
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=inputs.device.type,
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            candidate_albedo, candidate_normal, candidate_material, detail = v13._sr_candidate(
                model,
                inputs,
                baseline_albedo,
                baseline_normal,
                baseline_material,
            )

        with torch.autocast(device_type=inputs.device.type, enabled=False):
            albedo_error = (
                candidate_albedo.float() - target_albedo
            ).abs().mean(dim=1, keepdim=True)
            albedo_global = albedo_error.mean()
            albedo_edge = v13._weighted_mean(albedo_error, edge_weight)
            regret = v13._weighted_mean(
                F.relu(albedo_error - baseline_error), edge_weight
            )

            from v9.contours import sobel_tensor

            cgx, cgy = sobel_tensor(
                candidate_albedo.float().mean(dim=1, keepdim=True)
            )
            tgx, tgy = sobel_tensor(target_albedo.mean(dim=1, keepdim=True))
            gradient = v13._weighted_mean(
                (cgx - tgx).abs() + (cgy - tgy).abs(), edge_weight
            )
            laplacian = v13._weighted_mean(
                (
                    v13._laplacian(
                        candidate_albedo.float().mean(dim=1, keepdim=True)
                    )
                    - v13._laplacian(target_albedo.mean(dim=1, keepdim=True))
                ).abs(),
                edge_weight,
            )
            residual = candidate_albedo.float() - baseline_albedo.float()
            residual_supervision = (residual - desired_residual).abs().mean()

            normal_error = (
                candidate_normal.float() - target_normal
            ).abs().mean(dim=1, keepdim=True)
            material_error = (
                candidate_material.float() - target_material
            ).abs().mean(dim=1, keepdim=True)
            normal_global = normal_error.mean()
            normal_edge = v13._weighted_mean(normal_error, edge_weight)
            material_global = material_error.mean()
            material_edge = v13._weighted_mean(material_error, edge_weight)

            improvement = (baseline_error - albedo_error).detach()
            scale = max(float(getattr(config, "gate_error_scale", 0.08)), 1.0e-4)
            confidence_target = (0.5 + improvement / scale).clamp(0.0, 1.0)
            regret_target = (
                albedo_error > baseline_error + 0.001
            ).float().detach()
            support = (
                F.binary_cross_entropy_with_logits(
                    detail["confidence_logits"].float(), confidence_target
                )
                * float(config.detail_confidence_weight)
                + F.binary_cross_entropy_with_logits(
                    detail["regret_logits"].float(), regret_target
                )
                * float(config.detail_regret_classifier_weight)
            )
            protected = protected_preservation_terms(
                candidate_albedo.float(), baseline_albedo.float(), target_albedo
            )["protected_excess_drift_loss"].float()

            total = (
                albedo_global * 4.0
                + albedo_edge * 12.0
                + gradient * 5.0
                + laplacian * float(sr.SR_LAPLACIAN_WEIGHT)
                + regret * 16.0
                + residual_supervision * 8.0
                + normal_global * float(sr.SR_NORMAL_GLOBAL_WEIGHT)
                + normal_edge * float(sr.SR_NORMAL_EDGE_WEIGHT)
                + material_global * float(sr.SR_MATERIAL_GLOBAL_WEIGHT)
                + material_edge * float(sr.SR_MATERIAL_EDGE_WEIGHT)
                + support
                + protected * 96.0
            ).float()

        if not bool(torch.isfinite(total).item()):
            raise RuntimeError(f"non-finite V13.1 SR loss step={step}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.detail_net.parameters(), float(config.gradient_clip_norm)
        )
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError(f"non-finite V13.1 SR gradient step={step}")
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
            evaluated_albedo, evaluated_normal, evaluated_material, _ = v13._sr_candidate(
                model,
                inputs,
                baseline_albedo,
                baseline_normal,
                baseline_material,
            )
        current = v13._candidate_metrics(
            evaluated_albedo,
            evaluated_normal,
            evaluated_material,
            baseline_albedo,
            baseline_normal,
            baseline_material,
            target_albedo,
            target_normal,
            target_material,
            target_edge,
        )
        score = (
            float(current["edgeRecovery"])
            + 0.50 * float(current["globalRecovery"])
            + 0.30 * float(current["gradientRecovery"])
            + 0.10 * float(current["normalRecovery"])
            + 0.10 * float(current["materialRecovery"])
        )
        rows.append(
            {
                "phase": "sr-v13.1",
                "step": step,
                "loss": float(total.detach().cpu().item()),
                "edgeRecovery": current["edgeRecovery"],
                "globalRecovery": current["globalRecovery"],
                "gradientRecovery": current["gradientRecovery"],
                "normalRecovery": current["normalRecovery"],
                "materialRecovery": current["materialRecovery"],
                "protectedPreservation": current["protectedPreservation"],
                "selectorMean": "",
            }
        )
        print(
            f"[v13.1/sr] {step:4d}/{int(steps):4d} "
            f"edge={current['edgeRecovery']:+.1%} "
            f"global={current['globalRecovery']:+.1%} "
            f"grad={current['gradientRecovery']:+.1%} "
            f"normal={current['normalRecovery']:+.1%} "
            f"material={current['materialRecovery']:+.1%} "
            f"cap={cap:.2f}",
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

        if score > plateau_best + float(quality.SR_PLATEAU_MIN_SCORE_DELTA):
            plateau_best = score
            plateau_stale = 0
        elif step >= int(quality.SR_PLATEAU_MIN_STEPS):
            plateau_stale += 1

        qualified = (
            float(current["edgeRecovery"]) >= float(required_edge)
            and float(current["globalRecovery"]) >= float(required_global)
            and float(current["gradientRecovery"]) >= float(required_gradient)
            and float(current["normalRecovery"]) >= 0.0
            and float(current["materialRecovery"]) >= 0.0
        )
        model.detail_net.train()
        if qualified:
            print(f"[v13.1/sr] quality target reached at step {step}", flush=True)
            break
        if (
            step >= int(quality.SR_PLATEAU_MIN_STEPS)
            and plateau_stale >= int(quality.SR_PLATEAU_STALE_EVALS)
        ):
            plateau_stopped = True
            print(
                f"[v13.1/sr] plateau stop at step {step}; "
                f"stale evaluations={plateau_stale}",
                flush=True,
            )
            break

    model.detail_net.load_state_dict(best_state, strict=True)
    return best_metrics, best_step, steps_completed, plateau_stopped


def _train_selector(
    *,
    model: FidelityResidualNetV9,
    inputs: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    detail: dict[str, torch.Tensor],
    target: torch.Tensor,
    target_edge: torch.Tensor,
    config: Any,
    steps: int,
    learning_rate: float,
    required_retention: float,
    candidate_metrics: dict[str, float],
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float], int, int, bool]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.benefit_selector.parameters():
        parameter.requires_grad_(True)

    fixed = candidate.detach().float()
    features = v13._selector_features(
        model,
        inputs,
        baseline,
        fixed,
        {key: value.detach() for key, value in detail.items()},
    ).detach()
    oracle, safe_max, protected, motion = safety._safe_selector_oracle(
        baseline, fixed, target
    )
    selector_weight = (
        0.10
        + target_edge * 2.90
        + (motion / 0.02).clamp(0.0, 1.0)
        + protected.float() * float(safety.PROTECTED_SELECTOR_WEIGHT_BOOST)
    ).detach()
    edge_weight = (0.20 + target_edge * 3.80).detach()
    baseline_error = (baseline.float() - target).abs().mean(dim=1, keepdim=True)
    optimizer = torch.optim.AdamW(
        model.benefit_selector.parameters(),
        lr=float(learning_rate),
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )

    best_key = (-1, -float("inf"), -float("inf"))
    best_state = copy.deepcopy(model.benefit_selector.state_dict())
    best_step = 0
    best_metrics = direct._metrics(baseline, baseline, target, target_edge)
    interval = max(8, min(32, int(steps) // 20))
    safe_quality_best = -float("inf")
    safe_stale = 0
    plateau_stopped = False
    steps_completed = 0
    model.benefit_selector.train()

    for step in range(1, int(steps) + 1):
        steps_completed = step
        optimizer.zero_grad(set_to_none=True)
        logits = model.benefit_selector(features)
        probability = torch.sigmoid(logits.float()).clamp(1.0e-5, 1.0 - 1.0e-5)
        final = (
            baseline.float() * (1.0 - probability) + fixed * probability
        ).clamp(0.0, 1.0)
        final_error = (final - target).abs().mean(dim=1, keepdim=True)
        bce = v13._weighted_mean(
            F.binary_cross_entropy(probability, oracle, reduction="none"),
            selector_weight,
        )
        reconstruction = v13._weighted_mean(final_error, edge_weight)
        regret = v13._weighted_mean(
            F.relu(final_error - baseline_error), edge_weight
        )
        protected_weight = protected.float()
        protected_count = protected_weight.sum().clamp_min(1.0)
        gate_excess = (
            F.relu(probability - safe_max) * protected_weight
        ).sum() / protected_count
        total = (
            bce * float(config.benefit_selector_weight)
            + reconstruction * float(config.albedo_weight) * 2.0
            + regret * float(config.regret_weight) * 3.0
            + gate_excess * float(safety.PROTECTED_GATE_EXCESS_WEIGHT)
        ).float()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            model.benefit_selector.parameters(), float(config.gradient_clip_norm)
        )
        optimizer.step()

        if step != 1 and step != int(steps) and step % interval != 0:
            continue

        model.benefit_selector.eval()
        with torch.no_grad():
            probability_eval = torch.sigmoid(
                model.benefit_selector(features).float()
            )
            final_eval = (
                baseline.float() * (1.0 - probability_eval)
                + fixed * probability_eval
            ).clamp(0.0, 1.0)
        current = direct._metrics(final_eval, baseline, target, target_edge)
        preservation = float(
            protected_preservation_terms(final_eval, baseline, target)[
                "protected_preservation_rate"
            ]
            .detach()
            .cpu()
            .item()
        )
        edge_retention = float(current["edgeRecovery"]) / max(
            float(candidate_metrics["edgeRecovery"]), 1.0e-8
        )
        global_retention = float(current["globalRecovery"]) / max(
            float(candidate_metrics["globalRecovery"]), 1.0e-8
        )
        key = quality.selector_checkpoint_key(
            preservation=preservation,
            edge_recovery=float(current["edgeRecovery"]),
            global_recovery=float(current["globalRecovery"]),
            edge_retention=edge_retention,
            global_retention=global_retention,
            preservation_required=float(PROTECTED_PRESERVATION_REQUIRED),
        )
        rows.append(
            {
                "phase": "selector-v13.1",
                "step": step,
                "loss": float(total.detach().cpu().item()),
                "edgeRecovery": current["edgeRecovery"],
                "globalRecovery": current["globalRecovery"],
                "gradientRecovery": "",
                "normalRecovery": "",
                "materialRecovery": "",
                "protectedPreservation": preservation,
                "selectorMean": float(probability_eval.mean().item()),
            }
        )
        print(
            f"[v13.1/selector] {step:4d}/{int(steps):4d} "
            f"edge={current['edgeRecovery']:+.1%} "
            f"global={current['globalRecovery']:+.1%} "
            f"retain={edge_retention:.1%}/{global_retention:.1%} "
            f"protect={preservation:.2%}",
            flush=True,
        )

        if key > best_key:
            best_key = key
            best_step = step
            best_metrics = {
                **current,
                "protectedPreservation": preservation,
                "edgeRetention": edge_retention,
                "globalRetention": global_retention,
            }
            best_state = {
                key_name: value.detach().cpu().clone()
                for key_name, value in model.benefit_selector.state_dict().items()
            }

        safe_and_retained = (
            preservation >= float(PROTECTED_PRESERVATION_REQUIRED)
            and edge_retention >= float(required_retention)
            and global_retention >= float(required_retention)
        )
        if safe_and_retained:
            safe_quality = (
                float(current["edgeRecovery"])
                + 0.50 * float(current["globalRecovery"])
            )
            if safe_quality > safe_quality_best + float(
                quality.SELECTOR_PLATEAU_MIN_SCORE_DELTA
            ):
                safe_quality_best = safe_quality
                safe_stale = 0
            elif step >= int(quality.SELECTOR_PLATEAU_MIN_STEPS):
                safe_stale += 1

        model.benefit_selector.train()
        if (
            step >= int(quality.SELECTOR_PLATEAU_MIN_STEPS)
            and safe_stale >= int(quality.SELECTOR_PLATEAU_STALE_EVALS)
        ):
            plateau_stopped = True
            print(
                f"[v13.1/selector] fidelity plateau after safety qualification "
                f"at step {step}",
                flush=True,
            )
            break

    model.benefit_selector.load_state_dict(best_state, strict=True)
    return best_metrics, best_step, steps_completed, plateau_stopped


def _main_impl(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    if int(args.tile_size) < 32 or int(args.tile_size) % 16 != 0:
        raise SystemExit("--tile-size must be >=32 and divisible by 16")
    if int(args.steps) < 64 or int(args.selector_steps) < 32:
        raise SystemExit("SR/selector step budgets are too small")

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
    output_dir = root / "artifacts/nsamdr/micro_diagnostics" / f"SR_{stamp}"
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
    target_normal = moved["target_normal"].float()
    target_material = sr._target_material(moved, config)
    baseline, baseline_normal, baseline_material = direct._baseline(
        model, moved["input"]
    )
    amp_dtype = service._resolve_amp_dtype(config, device)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    print("=" * 78, flush=True)
    print("NSAMDR V13.1 SR VISUAL-QUALITY PROOF", flush=True)
    print(
        f"Authority : B + observable-evidence SR residual (albedo cap "
        f"{float(config.detail_albedo_max_delta):.2f}) -> safe BenefitSelector",
        flush=True,
    )
    print(
        f"Target    : edge {float(args.required_edge_recovery):.0%} / "
        f"global {float(args.required_global_recovery):.0%} / "
        f"gradient {float(args.required_gradient_recovery):.0%}",
        flush=True,
    )
    print(f"Artifacts : {output_dir}", flush=True)
    print("=" * 78, flush=True)

    candidate_metrics, detail_best_step, detail_steps_completed, sr_plateau = _train_sr(
        model=model,
        inputs=moved["input"],
        baseline_albedo=baseline,
        baseline_normal=baseline_normal,
        baseline_material=baseline_material,
        target_albedo=target,
        target_normal=target_normal,
        target_material=target_material,
        target_edge=target_edge,
        config=config,
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        head_multiplier=float(args.head_lr_multiplier),
        required_edge=float(args.required_edge_recovery),
        required_global=float(args.required_global_recovery),
        required_gradient=float(args.required_gradient_recovery),
        amp_dtype=amp_dtype,
        rows=rows,
    )

    model.detail_net.eval()
    with torch.no_grad():
        candidate, candidate_normal, candidate_material, detail = v13._sr_candidate(
            model,
            moved["input"],
            baseline,
            baseline_normal,
            baseline_material,
        )

    final_metrics, selector_best_step, selector_steps_completed, selector_plateau = _train_selector(
        model=model,
        inputs=moved["input"],
        baseline=baseline,
        candidate=candidate,
        detail=detail,
        target=target,
        target_edge=target_edge,
        config=config,
        steps=int(args.selector_steps),
        learning_rate=float(args.selector_learning_rate),
        required_retention=float(args.required_retention),
        candidate_metrics=candidate_metrics,
        rows=rows,
    )

    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        production = model(moved["input"])
    production_candidate = production["sr_candidate_albedo"].float()
    production_final = production["albedo"].float()
    candidate_final_metrics = direct._metrics(
        production_candidate, baseline, target, target_edge
    )
    final_final_metrics = direct._metrics(
        production_final, baseline, target, target_edge
    )
    gradient_recovery = v13._gradient_recovery(
        baseline, production_candidate, target
    )
    normal_recovery = v13._map_recovery(
        baseline_normal,
        production["sr_candidate_normal"].float(),
        target_normal,
    )
    material_recovery = v13._map_recovery(
        baseline_material,
        production["sr_candidate_material"].float(),
        target_material,
    )
    preservation = float(
        protected_preservation_terms(production_final, baseline, target)[
            "protected_preservation_rate"
        ]
        .detach()
        .cpu()
        .item()
    )

    candidate_parity = float(
        (production_candidate - candidate.float()).abs().amax().item()
    )
    normal_parity = float(
        (
            production["sr_candidate_normal"].float()
            - candidate_normal.float()
        )
        .abs()
        .amax()
        .item()
    )
    material_parity = float(
        (
            production["sr_candidate_material"].float()
            - candidate_material.float()
        )
        .abs()
        .amax()
        .item()
    )
    expected = (
        baseline.float()
        * (1.0 - production["benefit_selector_probability"].float())
        + production_candidate
        * production["benefit_selector_probability"].float()
    ).clamp(0.0, 1.0)
    final_parity = float((production_final - expected).abs().amax().item())

    edge_retention = float(final_final_metrics["edgeRecovery"]) / max(
        float(candidate_final_metrics["edgeRecovery"]), 1.0e-8
    )
    global_retention = float(final_final_metrics["globalRecovery"]) / max(
        float(candidate_final_metrics["globalRecovery"]), 1.0e-8
    )
    candidate_pass = (
        float(candidate_final_metrics["edgeRecovery"])
        >= float(args.required_edge_recovery)
        and float(candidate_final_metrics["globalRecovery"])
        >= float(args.required_global_recovery)
        and float(gradient_recovery) >= float(args.required_gradient_recovery)
        and float(normal_recovery) >= 0.0
        and float(material_recovery) >= 0.0
    )
    selector_pass = (
        edge_retention >= float(args.required_retention)
        and global_retention >= float(args.required_retention)
        and preservation >= float(PROTECTED_PRESERVATION_REQUIRED)
    )
    parity_pass = (
        max(candidate_parity, normal_parity, material_parity, final_parity)
        <= PARITY_TOLERANCE
    )
    passed = bool(candidate_pass and selector_pass and parity_pass)

    probe = output_dir / "sr_probe_final.png"
    v13._write_probe(
        probe,
        target,
        baseline,
        production_candidate,
        production_final,
        candidate_final_metrics,
        final_final_metrics,
    )
    v13._write_csv(output_dir / "metrics.csv", rows)
    torch.save(
        {
            "schema": MODEL_SCHEMA,
            "diagnosticSchema": REPORT_SCHEMA,
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
        },
        output_dir / "best_state.pt",
    )

    report = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "selectionKind": "diagnostic-non-promotable",
        "promotable": False,
        "sourceRevision": legacy._git_state(root),
        "patch": patch_metadata,
        "device": str(device),
        "srFirstRevision": sr.SR_FIRST_REVISION,
        "srQualityRevision": quality.SR_QUALITY_REVISION,
        "albedoResidualCap": float(config.detail_albedo_max_delta),
        "detailBestStep": int(detail_best_step),
        "detailStepsCompleted": int(detail_steps_completed),
        "selectorBestStep": int(selector_best_step),
        "selectorStepsCompleted": int(selector_steps_completed),
        "stepsRequested": int(args.steps),
        "selectorStepsRequested": int(args.selector_steps),
        "srPlateauStopped": bool(sr_plateau),
        "selectorPlateauStopped": bool(selector_plateau),
        "candidate": {
            **candidate_final_metrics,
            "gradientRecovery": gradient_recovery,
            "normalRecovery": normal_recovery,
            "materialRecovery": material_recovery,
        },
        "final": {
            **final_final_metrics,
            "edgeRetention": edge_retention,
            "globalRetention": global_retention,
            "protectedPreservation": preservation,
            "selectorMean": float(
                production["benefit_selector_probability"].float().mean().item()
            ),
        },
        "requirements": {
            "edgeRecovery": float(args.required_edge_recovery),
            "globalRecovery": float(args.required_global_recovery),
            "gradientRecovery": float(args.required_gradient_recovery),
            "selectorRetention": float(args.required_retention),
            "protectedPreservation": float(PROTECTED_PRESERVATION_REQUIRED),
            "parityTolerance": PARITY_TOLERANCE,
        },
        "candidatePass": bool(candidate_pass),
        "selectorPass": bool(selector_pass),
        "parityPass": bool(parity_pass),
        "parity": {
            "candidateAlbedoMaxAbs": candidate_parity,
            "candidateNormalMaxAbs": normal_parity,
            "candidateMaterialMaxAbs": material_parity,
            "finalMaxAbs": final_parity,
        },
        "elapsedSeconds": time.perf_counter() - started,
        "finalProbeSheet": str(probe.resolve()),
    }
    (output_dir / "sr_first_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    zip_path = legacy._zip_diagnostics(output_dir)

    print("=" * 78, flush=True)
    print("V13.1 SR QUALITY: " + ("PASS" if passed else "FAIL"), flush=True)
    print(
        f"Candidate edge/global/grad : "
        f"{candidate_final_metrics['edgeRecovery']:+.1%} / "
        f"{candidate_final_metrics['globalRecovery']:+.1%} / "
        f"{gradient_recovery:+.1%}",
        flush=True,
    )
    print(
        f"Physical recovery           : normal {normal_recovery:+.1%} / "
        f"material {material_recovery:+.1%}",
        flush=True,
    )
    print(
        f"Final retention             : {edge_retention:.1%} / "
        f"{global_retention:.1%}",
        flush=True,
    )
    print(f"Protected preservation      : {preservation:.2%}", flush=True)
    print(f"Diagnostics                 : {zip_path}", flush=True)
    print("=" * 78, flush=True)
    if args.open_result:
        micro_impl._open_result(probe)
    return 0 if passed else 2


def main(argv: list[str] | None = None) -> int:
    return run_consolidated_diagnostic(
        _main_impl,
        argv,
        legacy_folder="micro_diagnostics",
        category="micro",
    )


if __name__ == "__main__":
    raise SystemExit(main())
