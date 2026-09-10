#!/usr/bin/env python3
"""Fast staged capacity/authority diagnostic on one fixed authored Raven patch."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_nsamdr_v9_raven_micro_overfit as legacy
from v9.application.backend import TrainingBackend
from v9.inference import resolve_device
from v9.model import FidelityResidualNetV9, MODEL_SCHEMA, parameter_count


REPORT_SCHEMA = "NSAMDR_RAVEN_MICRO_AUTHORITY_DIAGNOSTIC_V2"
CANDIDATE_KEYS = (
    ("rawBoundary", "boundary_initial_candidate_albedo"),
    ("profiledBoundary", "boundary_candidate_albedo"),
    ("preSeam", "boundary_pre_seam_albedo"),
    ("postSeam", "boundary_reconstructed_albedo"),
    ("detailCandidate", "detail_candidate_albedo"),
    ("final", "albedo"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact production architecture/loss schedule on one tiny fixed Raven "
            "patch while exposing each internal candidate and final authority gate."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--required-recovery", type=float, default=0.50)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--open-result", action="store_true")
    return parser


def _tensor_error(
    value: torch.Tensor,
    target: torch.Tensor,
    edge_mask: torch.Tensor,
) -> tuple[float, float]:
    error = (value.detach().float() - target).abs().mean(dim=1, keepdim=True)
    global_mae = float(error.mean().item())
    edge_mae = float(error[edge_mask].mean().item()) if bool(edge_mask.any().item()) else global_mae
    return global_mae, edge_mae


def _recovery(baseline: float, candidate: float) -> float:
    return (float(baseline) - float(candidate)) / max(float(baseline), 1.0e-8)


def _scalar_mean(outputs: dict[str, torch.Tensor], key: str) -> float | None:
    value = outputs.get(key)
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    result = float(value.detach().float().mean().item())
    return result if math.isfinite(result) else None


def _stage_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    target = batch["target_albedo"].detach().float()
    edge_mask = batch["target_edge"].detach().float() >= 0.08
    baseline = outputs["baseline_albedo"].detach().float()
    baseline_global, baseline_edge = _tensor_error(baseline, target, edge_mask)

    candidates: dict[str, Any] = {}
    for label, key in CANDIDATE_KEYS:
        value = outputs.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        global_mae, edge_mae = _tensor_error(value, target, edge_mask)
        candidates[label] = {
            "tensor": key,
            "globalMae": global_mae,
            "edgeMae": edge_mae,
            "globalRecovery": _recovery(baseline_global, global_mae),
            "edgeRecovery": _recovery(baseline_edge, edge_mae),
        }

    return {
        "baseline": {"globalMae": baseline_global, "edgeMae": baseline_edge},
        "candidates": candidates,
        "authority": {
            "structuralResidualGainMean": _scalar_mean(outputs, "structural_residual_gain"),
            "detailConfidenceMean": _scalar_mean(outputs, "detail_confidence"),
            "detailRegretMean": _scalar_mean(outputs, "detail_regret"),
            "benefitSelectorProbabilityMean": _scalar_mean(outputs, "benefit_selector_probability"),
            "finalSelectorGateMean": _scalar_mean(outputs, "final_selector_gate"),
        },
    }


def _evaluate(
    service: Any,
    model: FidelityResidualNetV9,
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    model.eval()
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=use_amp
    ):
        outputs = service._forward_for_phase(model, batch, phase, config)
    return outputs, _stage_metrics(outputs, batch)


def _rgb(tensor: torch.Tensor) -> np.ndarray:
    value = tensor[0].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.uint8(np.round(value * 255.0))


def _write_probe_sheet(
    path: Path,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    metrics: dict[str, Any],
    title: str,
) -> None:
    from PIL import Image, ImageDraw

    target = _rgb(batch["target_albedo"])
    baseline = _rgb(outputs["baseline_albedo"])
    chosen = (
        ("A TARGET", target, None),
        ("B BASELINE", baseline, None),
        ("G GEOMETRY", _rgb(outputs.get("boundary_pre_seam_albedo", outputs["boundary_reconstructed_albedo"])), "preSeam"),
        ("D RAW DETAIL", _rgb(outputs.get("detail_candidate_albedo", outputs["albedo"])), "detailCandidate"),
        ("F FINAL GATED", _rgb(outputs["albedo"]), "final"),
    )
    h, w = target.shape[:2]
    header = 64
    canvas = Image.new("RGB", (w * len(chosen), h + header), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    candidate_metrics = metrics.get("candidates", {})
    for index, (label, panel, metric_key) in enumerate(chosen):
        x = index * w
        canvas.paste(Image.fromarray(panel, mode="RGB"), (x, header))
        draw.text((x + 5, 6), label, fill=(245, 245, 245))
        if metric_key and metric_key in candidate_metrics:
            item = candidate_metrics[metric_key]
            draw.text(
                (x + 5, 25),
                f"edge {item['edgeRecovery']:+.1%}",
                fill=(225, 225, 225),
            )
            draw.text(
                (x + 5, 42),
                f"global {item['globalRecovery']:+.1%}",
                fill=(225, 225, 225),
            )
    draw.text((5, h + header - 17), title, fill=(210, 210, 210))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _flatten_metrics(epoch: int, phase: str, steps: int, elapsed: float, loss: float, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "phase": phase,
        "steps": steps,
        "loss": loss,
        "elapsedSeconds": elapsed,
        "baselineGlobalMae": metrics["baseline"]["globalMae"],
        "baselineEdgeMae": metrics["baseline"]["edgeMae"],
    }
    for label, item in metrics.get("candidates", {}).items():
        row[f"{label}GlobalMae"] = item["globalMae"]
        row[f"{label}EdgeMae"] = item["edgeMae"]
        row[f"{label}GlobalRecovery"] = item["globalRecovery"]
        row[f"{label}EdgeRecovery"] = item["edgeRecovery"]
    for key, value in metrics.get("authority", {}).items():
        row[key] = value
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_recovery(metrics: dict[str, Any], key: str, field: str) -> float:
    item = metrics.get("candidates", {}).get(key, {})
    value = item.get(field)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else -float("inf")


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
    if int(args.steps_per_epoch) < 4:
        raise SystemExit("--steps-per-epoch must be >=4")
    if not 0.0 < float(args.required_recovery) < 1.0:
        raise SystemExit("--required-recovery must be between 0 and 1")

    legacy._ensure_raven_dataset(
        root,
        root / legacy.RAVEN_CONFIG,
        args.shared_cache,
        bool(args.rebuild_dataset),
    )
    config, _raven = legacy._micro_config(root, args)
    # A 32-LR fixed patch comfortably fits on modern GPUs. Recomputing every
    # production component during backward roughly doubles this diagnostic's wall
    # time without changing parameters, losses or authority semantics.
    config.training_activation_checkpointing = False
    config.validate()

    manifest = legacy.load_dataset_manifest(root, config)
    sample, patch_metadata = legacy._select_hard_patch(manifest, config)
    batch_cpu = legacy._batch(sample)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = root / "artifacts/nsamdr/micro_diagnostics" / f"MICRO_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "selected_patch.json").write_text(
        json.dumps(patch_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    resolved = config.to_dict()
    resolved.update({
        "diagnosticMode": "raven-micro-authority-v2",
        "promotable": False,
        "stepsPerEpoch": int(args.steps_per_epoch),
        "requiredEdgeRecovery": float(args.required_recovery),
        "activationCheckpointingDisabledForTinyPatch": True,
    })
    (output_dir / "resolved_micro_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=" * 78, flush=True)
    print("NSAMDR RAVEN MICRO CAPACITY + AUTHORITY DIAGNOSTIC V2", flush=True)
    print("Authority                : diagnostic only / NON-PROMOTABLE", flush=True)
    print(f"Patch                    : {config.tile_size} LR -> {config.tile_size * 4} HR", flush=True)
    print(f"Repeated steps / epoch   : {int(args.steps_per_epoch)}", flush=True)
    print("Probe                    : geometry -> seam -> raw detail -> final selector", flush=True)
    print(f"Artifacts                : {output_dir}", flush=True)
    print("=" * 78, flush=True)

    backend = TrainingBackend()
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
    optimizer, optimizer_mode = service._build_optimizer(model, config, device)
    amp_dtype = service._resolve_amp_dtype(config, device)
    use_amp = device.type == "cuda"
    use_scaler = use_amp and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler, init_scale=config.amp_initial_scale)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler, init_scale=config.amp_initial_scale)

    moved = service._move_batch(batch_cpu, device, channels_last=False)
    model.set_phase("sdf-bootstrap")
    initial_outputs, initial_metrics = _evaluate(
        service, model, moved, config, "sdf-bootstrap", device, amp_dtype
    )
    _write_probe_sheet(
        output_dir / "micro_probe_initial.png",
        initial_outputs,
        moved,
        initial_metrics,
        "INITIAL",
    )

    rows: list[dict[str, Any]] = []
    epoch_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, config.total_epochs + 1):
        phase = service._phase_for_epoch(epoch, config)
        model.set_phase(phase)
        if phase == "sdf-proof" and hasattr(model, "set_parametric_substage"):
            model.set_parametric_substage("integration")

        learning_rate = service._phase_lr(phase, config, epoch)
        steps = int(args.steps_per_epoch)
        if phase == "seam-proof":
            learning_rate = 3.0e-3
            steps = max(steps, 48)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))

        model.train()
        epoch_started = time.perf_counter()
        loss_sum = 0.0
        loss_term_sums: dict[str, float] = {}
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = service._forward_for_phase(model, moved, phase, config)
            with torch.autocast(device_type=device.type, enabled=False):
                losses = training.compute_losses(outputs, moved, config, phase)
            total = losses["total"].float()
            if not bool(torch.isfinite(total).item()):
                raise RuntimeError(f"non-finite micro loss epoch={epoch} step={step}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError(f"non-finite micro gradient epoch={epoch} step={step}")
            scaler.step(optimizer)
            scaler.update()
            if phase == "sdf-proof":
                structure = getattr(model.geometry_net, "production_structure", None)
                restore = getattr(structure, "restore_locked_topology_parameters", None)
                if callable(restore):
                    restore()

            loss_sum += float(total.detach().cpu().item())
            for key, value in losses.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    scalar = float(value.detach().float().cpu().item())
                    if math.isfinite(scalar):
                        loss_term_sums[key] = loss_term_sums.get(key, 0.0) + scalar

            if step == 1 or step == steps or step % max(1, steps // 4) == 0:
                elapsed = max(time.perf_counter() - epoch_started, 1.0e-6)
                print(
                    f"  e{epoch:03d} {phase:<21} {step:4d}/{steps:4d} "
                    f"loss={loss_sum / step:.5f} rate={step / elapsed:.2f}step/s",
                    flush=True,
                )

        TrainingBackend._prepare_production_runtime(model)
        epoch_outputs, metrics = _evaluate(service, model, moved, config, phase, device, amp_dtype)
        elapsed = time.perf_counter() - epoch_started
        avg_loss = loss_sum / max(steps, 1)
        row = _flatten_metrics(epoch, phase, steps, elapsed, avg_loss, metrics)
        rows.append(row)
        epoch_reports.append({
            "epoch": epoch,
            "phase": phase,
            "steps": steps,
            "elapsedSeconds": elapsed,
            "metrics": metrics,
            "lossTerms": {key: value / max(steps, 1) for key, value in sorted(loss_term_sums.items())},
        })

        geom = metrics.get("candidates", {}).get("preSeam", {})
        detail = metrics.get("candidates", {}).get("detailCandidate", {})
        final = metrics.get("candidates", {}).get("final", {})
        gate = metrics.get("authority", {}).get("finalSelectorGateMean")
        gate_text = "n/a" if gate is None else f"{gate:.3f}"
        print(
            f"[micro] e{epoch:03d} {phase}: geometry={float(geom.get('edgeRecovery', 0.0)):+.1%} "
            f"rawDetail={float(detail.get('edgeRecovery', 0.0)):+.1%} "
            f"final={float(final.get('edgeRecovery', 0.0)):+.1%} finalGate={gate_text}",
            flush=True,
        )
        _write_probe_sheet(
            output_dir / "micro_probe_current.png",
            epoch_outputs,
            moved,
            metrics,
            f"EPOCH {epoch:03d} {phase}",
        )

    TrainingBackend._prepare_production_runtime(model)
    model.set_phase("physical-finetune")
    final_outputs, final_metrics = _evaluate(
        service, model, moved, config, "physical-finetune", device, amp_dtype
    )
    final_sheet = output_dir / "micro_probe_final.png"
    _write_probe_sheet(final_sheet, final_outputs, moved, final_metrics, "FINAL PRODUCTION AUTHORITY")
    _write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "epoch_diagnostics.json").write_text(
        json.dumps(epoch_reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    required = float(args.required_recovery)
    detail_edge = _candidate_recovery(final_metrics, "detailCandidate", "edgeRecovery")
    detail_global = _candidate_recovery(final_metrics, "detailCandidate", "globalRecovery")
    final_edge = _candidate_recovery(final_metrics, "final", "edgeRecovery")
    final_global = _candidate_recovery(final_metrics, "final", "globalRecovery")
    raw_capacity_pass = bool(detail_edge >= required and detail_global > 0.0)
    production_authority_pass = bool(final_edge >= required and final_global > 0.0)

    last_b1b = next(
        (item for item in reversed(epoch_reports) if item["phase"] == "sdf-proof"),
        None,
    )
    geometry_edge = (
        _candidate_recovery(last_b1b["metrics"], "preSeam", "edgeRecovery")
        if last_b1b is not None else -float("inf")
    )
    geometry_capacity_pass = bool(geometry_edge > 0.0)

    if raw_capacity_pass and production_authority_pass:
        interpretation = "local reconstruction and final production authority both demonstrated"
    elif raw_capacity_pass:
        interpretation = "raw detail capacity exists but final selector/confidence/regret authority suppresses it"
    elif geometry_capacity_pass:
        interpretation = "geometry improves locally but raw detail capacity target is not reached"
    else:
        interpretation = "local capacity not demonstrated; representation/objective must be fixed before whole-Raven training"

    report = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if production_authority_pass else "failed-capacity-or-authority",
        "selectionKind": "diagnostic-non-promotable",
        "promotable": False,
        "qualifiedProductionCheckpoint": False,
        "sourceRevision": legacy._git_state(root),
        "modelSchema": MODEL_SCHEMA,
        "parameterCount": parameter_count(model),
        "device": str(device),
        "optimizer": optimizer_mode,
        "ampDtype": str(amp_dtype).replace("torch.", ""),
        "elapsedSeconds": time.perf_counter() - started,
        "patch": patch_metadata,
        "requiredEdgeRecovery": required,
        "geometryCapacityPass": geometry_capacity_pass,
        "rawDetailCapacityPass": raw_capacity_pass,
        "productionAuthorityPass": production_authority_pass,
        "final": final_metrics,
        "interpretation": interpretation,
        "finalProbeSheet": str(final_sheet.resolve()),
    }
    (output_dir / "micro_capacity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    zip_path = legacy._zip_diagnostics(output_dir)
    print("=" * 78, flush=True)
    print(f"MICRO GEOMETRY CAPACITY : {'PASS' if geometry_capacity_pass else 'FAIL'}", flush=True)
    print(f"MICRO RAW DETAIL        : {'PASS' if raw_capacity_pass else 'FAIL'}", flush=True)
    print(f"MICRO FINAL AUTHORITY   : {'PASS' if production_authority_pass else 'FAIL'}", flush=True)
    print(f"Interpretation          : {interpretation}", flush=True)
    print(f"Diagnostics             : {zip_path}", flush=True)
    print("=" * 78, flush=True)
    if args.open_result:
        _open_result(final_sheet)
    return 0 if production_authority_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
