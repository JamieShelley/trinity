#!/usr/bin/env python3
"""Direct residual capacity proof for the production NSAMDR detail decoder.

V2 separates three questions that the staged production graph previously mixed:
1. Is the production residual amplitude mathematically capable of reaching the target?
2. Can the existing production detail decoder overfit one fixed Raven patch when
   geometry/seam/selector dependencies are removed?
3. If the production cap is too small, what bounded residual amplitude is actually
   required before production integration is reconsidered?

This run is diagnostic-only and cannot create or promote a production checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_nsamdr_v9_raven_micro_overfit as legacy
from v9.application.backend import TrainingBackend
from v9.contours import sobel_tensor
from v9.inference import resolve_device
from v9.model import FidelityResidualNetV9, MODEL_SCHEMA


REPORT_SCHEMA = "NSAMDR_RAVEN_DIRECT_RESIDUAL_CAPACITY_V2"
CAP_LADDER = (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.65, 0.80, 1.00)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overfit one fixed authored Raven patch using only the production detail "
            "decoder as a direct residual over deterministic baseline B."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--head-lr-multiplier", type=float, default=3.0)
    parser.add_argument("--required-edge-recovery", type=float, default=0.50)
    parser.add_argument("--required-global-recovery", type=float, default=0.25)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--open-result", action="store_true")
    return parser


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=value.device, dtype=value.dtype)
    if weight.shape[1] == 1 and value.shape[1] != 1:
        weight = weight.expand(-1, value.shape[1], -1, -1)
    return (value.float() * weight.float()).sum() / weight.float().sum().clamp_min(1.0)


def _gradient_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    pgx, pgy = sobel_tensor(prediction.float().mean(dim=1, keepdim=True))
    tgx, tgy = sobel_tensor(target.float().mean(dim=1, keepdim=True))
    return _weighted_mean((pgx - tgx).abs() + (pgy - tgy).abs(), weight)


def _baseline(
    model: FidelityResidualNetV9,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_albedo = inputs[:, 0:3].clamp(0.0, 1.0)
    source_normal = inputs[:, 3:5].clamp(-1.0, 1.0)
    source_material = inputs[:, 5:8].clamp(0.0, 1.0)
    albedo = F.interpolate(
        source_albedo,
        scale_factor=4,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp(0.0, 1.0)
    normal = model._normalize_xy(
        F.interpolate(source_normal, scale_factor=4, mode="bilinear", align_corners=False)
    )
    material = F.interpolate(source_material, scale_factor=4, mode="nearest")
    return albedo, normal, material


def _direct_candidate(
    model: FidelityResidualNetV9,
    inputs: torch.Tensor,
    baseline_albedo: torch.Tensor,
    baseline_normal: torch.Tensor,
    baseline_material: torch.Tensor,
    residual_cap: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    # Neutral geometry removes every upstream production dependency. The detail
    # decoder still receives all 17 native LR channels, including source SDF and
    # physical-map guidance. Only its residual amplitude is varied diagnostically.
    geometry = torch.zeros(
        (inputs.shape[0], 6, baseline_albedo.shape[-2], baseline_albedo.shape[-1]),
        device=inputs.device,
        dtype=baseline_albedo.dtype,
    )
    detail = model.detail_net(
        inputs,
        baseline_albedo,
        baseline_normal,
        baseline_material,
        geometry,
    )
    delta = detail["albedo_raw"].float() * float(residual_cap)
    candidate = (baseline_albedo.float() + delta).clamp(0.0, 1.0)
    return candidate, detail, delta


def _errors(
    value: torch.Tensor,
    target: torch.Tensor,
    edge_mask: torch.Tensor,
) -> tuple[float, float]:
    error = (value.detach().float() - target.detach().float()).abs().mean(dim=1, keepdim=True)
    global_mae = float(error.mean().item())
    edge_mae = float(error[edge_mask].mean().item()) if bool(edge_mask.any().item()) else global_mae
    return global_mae, edge_mae


def _recovery(baseline: float, candidate: float) -> float:
    return (float(baseline) - float(candidate)) / max(float(baseline), 1.0e-8)


def _metrics(
    candidate: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    edge: torch.Tensor,
) -> dict[str, float]:
    edge_mask = edge.detach().float() >= 0.08
    bg, be = _errors(baseline, target, edge_mask)
    cg, ce = _errors(candidate, target, edge_mask)
    residual = (candidate.detach().float() - baseline.detach().float()).abs()
    return {
        "baselineGlobalMae": bg,
        "baselineEdgeMae": be,
        "candidateGlobalMae": cg,
        "candidateEdgeMae": ce,
        "globalRecovery": _recovery(bg, cg),
        "edgeRecovery": _recovery(be, ce),
        "meanAbsResidual": float(residual.mean().item()),
        "maxAbsResidual": float(residual.max().item()),
    }


def _oracle_for_cap(
    baseline: torch.Tensor,
    target: torch.Tensor,
    edge: torch.Tensor,
    cap: float,
) -> dict[str, float]:
    desired = target.float() - baseline.float()
    clipped = desired.clamp(-float(cap), float(cap))
    candidate = (baseline.float() + clipped).clamp(0.0, 1.0)
    metrics = _metrics(candidate, baseline, target, edge)
    abs_desired = desired.detach().abs()
    metrics["cap"] = float(cap)
    metrics["fractionValuesBeyondCap"] = float((abs_desired > float(cap)).float().mean().item())
    metrics["fractionPixelsBeyondCap"] = float(
        (abs_desired.amax(dim=1, keepdim=True) > float(cap)).float().mean().item()
    )
    return metrics


def _cap_passes(
    metrics: dict[str, float],
    required_edge: float,
    required_global: float,
) -> bool:
    return (
        float(metrics["edgeRecovery"]) >= float(required_edge)
        and float(metrics["globalRecovery"]) >= float(required_global)
    )


def _choose_capacity_cap(
    baseline: torch.Tensor,
    target: torch.Tensor,
    edge: torch.Tensor,
    production_cap: float,
    required_edge: float,
    required_global: float,
) -> tuple[float, list[dict[str, float]]]:
    caps = sorted({float(production_cap), *CAP_LADDER})
    oracle_sweep = [_oracle_for_cap(baseline, target, edge, cap) for cap in caps]
    for result in oracle_sweep:
        if float(result["cap"]) + 1.0e-9 < float(production_cap):
            continue
        if _cap_passes(result, required_edge, required_global):
            return float(result["cap"]), oracle_sweep
    return float(caps[-1]), oracle_sweep


def _rgb(value: torch.Tensor) -> np.ndarray:
    image = value[0].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.uint8(np.round(image * 255.0))


def _write_probe(
    path: Path,
    target: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    metrics: dict[str, float],
    title: str,
) -> None:
    from PIL import Image, ImageDraw

    a = _rgb(target)
    b = _rgb(baseline)
    c = _rgb(candidate)
    error = np.abs(c.astype(np.int16) - a.astype(np.int16)).astype(np.uint8)
    panels = (
        ("A TARGET", a),
        ("B BASELINE", b),
        ("R DIRECT RESIDUAL", c),
        ("ABS ERROR", error),
    )
    h, w = a.shape[:2]
    header = 62
    canvas = Image.new("RGB", (w * len(panels), h + header), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        x = index * w
        canvas.paste(Image.fromarray(panel, mode="RGB"), (x, header))
        draw.text((x + 5, 6), label, fill=(245, 245, 245))
        if index == 2:
            draw.text((x + 5, 25), f"edge {metrics['edgeRecovery']:+.1%}", fill=(225, 225, 225))
            draw.text((x + 5, 42), f"global {metrics['globalRecovery']:+.1%}", fill=(225, 225, 225))
    draw.text((5, h + header - 17), title, fill=(210, 210, 210))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _open_result(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    if int(args.tile_size) < 32 or int(args.tile_size) % 16 != 0:
        raise SystemExit("--tile-size must be >=32 and divisible by 16")
    if int(args.steps) < 64:
        raise SystemExit("--steps must be >=64")
    if float(args.learning_rate) <= 0.0:
        raise SystemExit("--learning-rate must be >0")
    if float(args.head_lr_multiplier) < 1.0:
        raise SystemExit("--head-lr-multiplier must be >=1")
    for name, value in (
        ("--required-edge-recovery", args.required_edge_recovery),
        ("--required-global-recovery", args.required_global_recovery),
    ):
        if not 0.0 < float(value) < 1.0:
            raise SystemExit(f"{name} must be between 0 and 1")

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
    output_dir = root / "artifacts/nsamdr/direct_residual_diagnostics" / f"DIRECT_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "selected_patch.json").write_text(
        json.dumps(patch_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.detail_net.parameters():
        parameter.requires_grad_(True)
    detail_parameter_count = sum(parameter.numel() for parameter in model.detail_net.parameters())

    moved = service._move_batch(batch_cpu, device, channels_last=False)
    target = moved["target_albedo"].float()
    target_edge = moved["target_edge"].float().clamp(0.0, 1.0)
    baseline_albedo, baseline_normal, baseline_material = _baseline(model, moved["input"])

    production_cap = float(getattr(config, "detail_albedo_max_delta", 0.20))
    training_cap, oracle_sweep = _choose_capacity_cap(
        baseline_albedo,
        target,
        target_edge,
        production_cap,
        float(args.required_edge_recovery),
        float(args.required_global_recovery),
    )
    production_oracle = next(
        result for result in oracle_sweep if abs(float(result["cap"]) - production_cap) < 1.0e-9
    )
    training_oracle = next(
        result for result in oracle_sweep if abs(float(result["cap"]) - training_cap) < 1.0e-9
    )
    production_cap_feasible = _cap_passes(
        production_oracle,
        float(args.required_edge_recovery),
        float(args.required_global_recovery),
    )
    training_cap_feasible = _cap_passes(
        training_oracle,
        float(args.required_edge_recovery),
        float(args.required_global_recovery),
    )

    amp_dtype = service._resolve_amp_dtype(config, device)
    use_amp = device.type == "cuda"
    use_scaler = use_amp and amp_dtype == torch.float16

    model.detail_net.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        initial_candidate, _, _ = _direct_candidate(
            model,
            moved["input"],
            baseline_albedo,
            baseline_normal,
            baseline_material,
            training_cap,
        )
    initial_metrics = _metrics(initial_candidate, baseline_albedo, target, target_edge)
    _write_probe(
        output_dir / "direct_probe_initial.png",
        target,
        baseline_albedo,
        initial_candidate,
        initial_metrics,
        "INITIAL DIRECT RESIDUAL",
    )

    head_ids = {id(parameter) for parameter in model.detail_net.albedo_head.parameters()}
    head_parameters = [
        parameter for parameter in model.detail_net.parameters() if id(parameter) in head_ids
    ]
    body_parameters = [
        parameter for parameter in model.detail_net.parameters() if id(parameter) not in head_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": body_parameters, "lr": float(args.learning_rate)},
            {
                "params": head_parameters,
                "lr": float(args.learning_rate) * float(args.head_lr_multiplier),
            },
        ],
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_scaler, init_scale=config.amp_initial_scale
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_scaler, init_scale=config.amp_initial_scale
        )

    print("=" * 78, flush=True)
    print("NSAMDR RAVEN DIRECT RESIDUAL CAPACITY PROOF V2", flush=True)
    print("Authority                : diagnostic only / NON-PROMOTABLE", flush=True)
    print("Path                     : LR evidence + B -> production detail decoder -> B + residual", flush=True)
    print("Bypassed                 : geometry, seam authority, profile gate, BenefitSelector", flush=True)
    print(f"Patch                    : {config.tile_size} LR -> {config.tile_size * 4} HR", flush=True)
    print(f"Maximum steps            : {int(args.steps)}", flush=True)
    print(f"Body/head learning rate  : {float(args.learning_rate):.3g} / {float(args.learning_rate) * float(args.head_lr_multiplier):.3g}", flush=True)
    print(f"Production residual cap  : {production_cap:.3f}", flush=True)
    print(f"Capacity residual cap    : {training_cap:.3f}", flush=True)
    print(f"Production-cap oracle    : edge={production_oracle['edgeRecovery']:+.1%} global={production_oracle['globalRecovery']:+.1%}", flush=True)
    print(f"Capacity-cap oracle      : edge={training_oracle['edgeRecovery']:+.1%} global={training_oracle['globalRecovery']:+.1%}", flush=True)
    print(f"Detail parameters        : {detail_parameter_count}", flush=True)
    print(f"Artifacts                : {output_dir}", flush=True)
    print("=" * 78, flush=True)

    edge_weight = (0.20 + target_edge * 3.80).detach()
    baseline_error = (
        baseline_albedo.detach().float() - target
    ).abs().mean(dim=1, keepdim=True)
    desired_residual = (
        target.detach().float() - baseline_albedo.detach().float()
    ).clamp(-training_cap, training_cap)

    rows: list[dict[str, float | int]] = []
    best_score = -float("inf")
    best_step = 0
    best_metrics = dict(initial_metrics)
    best_state = copy.deepcopy(model.detail_net.state_dict())
    started = time.perf_counter()
    evaluation_interval = max(16, min(64, int(args.steps) // 24))
    completed_steps = 0

    model.detail_net.train()
    for step in range(1, int(args.steps) + 1):
        completed_steps = step
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            candidate, _detail, raw_delta = _direct_candidate(
                model,
                moved["input"],
                baseline_albedo,
                baseline_normal,
                baseline_material,
                training_cap,
            )
        with torch.autocast(device_type=device.type, enabled=False):
            candidate = candidate.float()
            raw_delta = raw_delta.float()
            error = (candidate - target).abs().mean(dim=1, keepdim=True)
            global_reconstruction = error.mean()
            edge_reconstruction = _weighted_mean(error, edge_weight)
            regret = _weighted_mean(F.relu(error - baseline_error), edge_weight)
            gradient = _gradient_error(candidate, target, edge_weight)
            residual_supervision = (raw_delta - desired_residual).abs().mean()
            total = (
                global_reconstruction * 4.0
                + edge_reconstruction * 12.0
                + gradient * 5.0
                + regret * 16.0
                + residual_supervision * 8.0
            ).float()
        if not bool(torch.isfinite(total).item()):
            raise RuntimeError(f"non-finite direct residual loss step={step}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.detail_net.parameters(), float(config.gradient_clip_norm)
        )
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError(f"non-finite direct residual gradient step={step}")
        scaler.step(optimizer)
        scaler.update()

        should_evaluate = (
            step == 1 or step == int(args.steps) or step % evaluation_interval == 0
        )
        if not should_evaluate:
            continue

        model.detail_net.eval()
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            evaluated, _, _ = _direct_candidate(
                model,
                moved["input"],
                baseline_albedo,
                baseline_normal,
                baseline_material,
                training_cap,
            )
        current = _metrics(evaluated, baseline_albedo, target, target_edge)
        score = float(current["edgeRecovery"]) + 0.50 * float(current["globalRecovery"])
        rows.append(
            {
                "step": step,
                "loss": float(total.detach().cpu().item()),
                "residualSupervision": float(residual_supervision.detach().cpu().item()),
                **current,
            }
        )
        print(
            f"[direct-v2] {step:4d}/{int(args.steps):4d} "
            f"edge={current['edgeRecovery']:+.1%} global={current['globalRecovery']:+.1%} "
            f"loss={float(total.detach().cpu().item()):.5f}",
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
            _write_probe(
                output_dir / "direct_probe_best.png",
                target,
                baseline_albedo,
                evaluated,
                current,
                f"BEST STEP {step} CAP {training_cap:.2f}",
            )

        passed = _cap_passes(
            current,
            float(args.required_edge_recovery),
            float(args.required_global_recovery),
        )
        model.detail_net.train()
        if passed:
            break

    model.detail_net.load_state_dict(best_state, strict=True)
    model.detail_net.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=use_amp
    ):
        final_candidate, _, _ = _direct_candidate(
            model,
            moved["input"],
            baseline_albedo,
            baseline_normal,
            baseline_material,
            training_cap,
        )
    final_metrics = _metrics(final_candidate, baseline_albedo, target, target_edge)
    final_probe = output_dir / "direct_probe_final.png"
    _write_probe(
        final_probe,
        target,
        baseline_albedo,
        final_candidate,
        final_metrics,
        f"BEST DIRECT RESIDUAL STEP {best_step} CAP {training_cap:.2f}",
    )

    if rows:
        with (output_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    capacity_pass = _cap_passes(
        final_metrics,
        float(args.required_edge_recovery),
        float(args.required_global_recovery),
    )
    if capacity_pass and not production_cap_feasible:
        interpretation = (
            "detail decoder capacity is proven, but the production albedo residual cap is "
            "mathematically too restrictive for the requested Raven recovery; raise or make "
            "the production residual amplitude adaptive before integration"
        )
    elif capacity_pass:
        interpretation = (
            "production detail decoder independently recovers Raven detail over B; "
            "serial geometry/seam/authority composition is the integration blocker"
        )
    elif not training_cap_feasible:
        interpretation = (
            "requested recovery is not attainable even under the diagnostic residual-cap "
            "ladder; revise the candidate formulation before production integration"
        )
    else:
        interpretation = (
            "the residual amplitude is mathematically sufficient but the isolated detail "
            "decoder still fails to fit the patch; fix decoder optimisation/capacity before "
            "rebuilding production composition"
        )

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if capacity_pass else "failed-direct-residual-capacity",
        "selectionKind": "diagnostic-non-promotable",
        "promotable": False,
        "qualifiedProductionCheckpoint": False,
        "sourceRevision": legacy._git_state(root),
        "modelSchema": MODEL_SCHEMA,
        "device": str(device),
        "ampDtype": str(amp_dtype).replace("torch.", ""),
        "detailParameterCount": detail_parameter_count,
        "stepsRequested": int(args.steps),
        "stepsCompleted": int(completed_steps),
        "bestStep": int(best_step),
        "learningRate": float(args.learning_rate),
        "headLearningRateMultiplier": float(args.head_lr_multiplier),
        "requiredEdgeRecovery": float(args.required_edge_recovery),
        "requiredGlobalRecovery": float(args.required_global_recovery),
        "productionResidualCap": production_cap,
        "trainingResidualCap": training_cap,
        "productionCapFeasible": bool(production_cap_feasible),
        "trainingCapFeasible": bool(training_cap_feasible),
        "productionCapOracle": production_oracle,
        "trainingCapOracle": training_oracle,
        "oracleCapSweep": oracle_sweep,
        "initial": initial_metrics,
        "best": best_metrics,
        "final": final_metrics,
        "capacityProofPass": bool(capacity_pass),
        "interpretation": interpretation,
        "elapsedSeconds": time.perf_counter() - started,
        "finalProbeSheet": str(final_probe.resolve()),
    }
    (output_dir / "direct_residual_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    zip_path = legacy._zip_diagnostics(output_dir)
    print("=" * 78, flush=True)
    print(f"DIRECT RESIDUAL CAPACITY : {'PASS' if capacity_pass else 'FAIL'}", flush=True)
    print(f"Best edge recovery       : {float(final_metrics['edgeRecovery']):+.2%}", flush=True)
    print(f"Best global recovery     : {float(final_metrics['globalRecovery']):+.2%}", flush=True)
    print(f"Production cap feasible  : {production_cap_feasible}", flush=True)
    print(f"Training cap             : {training_cap:.2f}", flush=True)
    print(f"Best step                : {best_step}", flush=True)
    print(f"Diagnostics              : {zip_path}", flush=True)
    print("=" * 78, flush=True)
    if args.open_result:
        _open_result(final_probe)
    return 0 if capacity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
