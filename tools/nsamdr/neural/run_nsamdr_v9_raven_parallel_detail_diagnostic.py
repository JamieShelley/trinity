#!/usr/bin/env python3
"""Fast production-composition proof for the V12.3 parallel detail path.

This diagnostic starts from the same deterministic hard Raven patch used by Direct
Residual Capacity, but unlike that isolated proof it runs the actual production
FidelityResidualNetV9 forward and production loss adapters. It trains only the two
components that should now own the useful path:

1. detail-reconstruction: independently learn B -> B + bounded detail residual;
2. physical-finetune: freeze that candidate and train BenefitSelector to retain it.

Geometry/seam still execute so production graph participation is realistic, but they
cannot alter the direct-detail candidate or selector evidence in V12.3. The run is
non-promotable and exists only to prove that the successful isolated capacity result
survives production composition.
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

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_nsamdr_v9_raven_direct_residual_diagnostic as direct
import run_nsamdr_v9_raven_micro_overfit as legacy
from v9.application.backend import TrainingBackend
from v9.inference import resolve_device
from v9.model import FidelityResidualNetV9, MODEL_SCHEMA


REPORT_SCHEMA = "NSAMDR_RAVEN_PARALLEL_DETAIL_INTEGRATION_V1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prove V12.3 direct-detail candidate and selector inside production forward."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--detail-steps", type=int, default=1280)
    parser.add_argument("--selector-steps", type=int, default=384)
    parser.add_argument("--required-edge-recovery", type=float, default=0.50)
    parser.add_argument("--required-global-recovery", type=float, default=0.25)
    parser.add_argument("--required-retention", type=float, default=0.85)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--open-result", action="store_true")
    return parser


def _candidate_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, float]]:
    target = batch["target_albedo"].float()
    edge = batch["target_edge"].float().clamp(0.0, 1.0)
    baseline = outputs["baseline_albedo"].float()
    candidate = outputs["detail_candidate_albedo"].float()
    final = outputs["albedo"].float()
    return (
        direct._metrics(candidate, baseline, target, edge),
        direct._metrics(final, baseline, target, edge),
    )


def _evaluate(
    service: Any,
    model: FidelityResidualNetV9,
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, float]]:
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=device.type == "cuda",
    ):
        outputs = service._forward_for_phase(model, batch, phase, config)
    candidate, final = _candidate_metrics(outputs, batch)
    return outputs, candidate, final


def _write_probe(
    path: Path,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    candidate_metrics: dict[str, float],
    final_metrics: dict[str, float],
    title: str,
) -> None:
    from PIL import Image, ImageDraw

    target = direct._rgb(batch["target_albedo"])
    baseline = direct._rgb(outputs["baseline_albedo"])
    candidate = direct._rgb(outputs["detail_candidate_albedo"])
    final = direct._rgb(outputs["albedo"])
    panels = (
        ("A TARGET", target),
        ("B BASELINE", baseline),
        ("D DIRECT DETAIL", candidate),
        ("F FINAL SELECTED", final),
    )
    h, w = target.shape[:2]
    header = 66
    canvas = Image.new("RGB", (w * 4, h + header), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = index * w
        canvas.paste(Image.fromarray(image, mode="RGB"), (x, header))
        draw.text((x + 5, 6), label, fill=(245, 245, 245))
        if index == 2:
            draw.text((x + 5, 27), f"edge {candidate_metrics['edgeRecovery']:+.1%}", fill=(225, 225, 225))
            draw.text((x + 5, 45), f"global {candidate_metrics['globalRecovery']:+.1%}", fill=(225, 225, 225))
        elif index == 3:
            draw.text((x + 5, 27), f"edge {final_metrics['edgeRecovery']:+.1%}", fill=(225, 225, 225))
            draw.text((x + 5, 45), f"global {final_metrics['globalRecovery']:+.1%}", fill=(225, 225, 225))
    draw.text((5, h + header - 16), title, fill=(210, 210, 210))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _configure_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate) * float(group.get("lr_scale", 1.0))


def _train_phase(
    *,
    service: Any,
    training: Any,
    model: FidelityResidualNetV9,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
    steps: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, float], dict[str, float]]:
    model.set_phase(phase)
    learning_rate = float(service._phase_lr(phase, config, None))
    _configure_optimizer_lr(optimizer, learning_rate)
    interval = max(16, min(64, int(steps) // 20))
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_outputs: dict[str, torch.Tensor] | None = None
    best_candidate: dict[str, float] | None = None
    best_final: dict[str, float] | None = None

    model.train()
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            outputs = service._forward_for_phase(model, batch, phase, config)
        with torch.autocast(device_type=device.type, enabled=False):
            losses = training.compute_losses(outputs, batch, config, phase)
            total = losses["total"].float()
        if not bool(torch.isfinite(total).item()):
            raise RuntimeError(f"non-finite {phase} loss at step {step}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        active_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(active_parameters, float(config.gradient_clip_norm))
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError(f"non-finite {phase} gradient at step {step}")
        scaler.step(optimizer)
        scaler.update()

        if step != 1 and step != int(steps) and step % interval != 0:
            continue
        evaluated, candidate, final = _evaluate(
            service, model, batch, config, phase, device, amp_dtype
        )
        selector_mean = float(
            evaluated["benefit_selector_probability"].detach().float().mean().item()
        )
        score_metrics = candidate if phase == "detail-reconstruction" else final
        score = float(score_metrics["edgeRecovery"]) + 0.5 * float(score_metrics["globalRecovery"])
        rows.append(
            {
                "phase": phase,
                "step": step,
                "learningRate": learning_rate,
                "loss": float(total.detach().cpu().item()),
                "candidateEdgeRecovery": float(candidate["edgeRecovery"]),
                "candidateGlobalRecovery": float(candidate["globalRecovery"]),
                "finalEdgeRecovery": float(final["edgeRecovery"]),
                "finalGlobalRecovery": float(final["globalRecovery"]),
                "selectorMean": selector_mean,
            }
        )
        print(
            f"[parallel] {phase:21s} {step:4d}/{int(steps):4d} "
            f"D edge={candidate['edgeRecovery']:+.1%} global={candidate['globalRecovery']:+.1%} "
            f"F edge={final['edgeRecovery']:+.1%} global={final['globalRecovery']:+.1%} "
            f"gate={selector_mean:.3f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            if phase == "detail-reconstruction":
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.detail_net.state_dict().items()
                }
            else:
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.benefit_selector.state_dict().items()
                }
            best_outputs = evaluated
            best_candidate = candidate
            best_final = final
        model.train()

    if best_state is None or best_outputs is None or best_candidate is None or best_final is None:
        raise RuntimeError(f"{phase} produced no evaluation state")
    if phase == "detail-reconstruction":
        model.detail_net.load_state_dict(best_state, strict=True)
    else:
        model.benefit_selector.load_state_dict(best_state, strict=True)
    return _evaluate(service, model, batch, config, phase, device, amp_dtype)


def _retention(candidate: float, final: float) -> float:
    if candidate <= 1.0e-8:
        return 0.0
    return final / candidate


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    if int(args.tile_size) < 32 or int(args.tile_size) % 16 != 0:
        raise SystemExit("--tile-size must be >=32 and divisible by 16")
    if int(args.detail_steps) < 64 or int(args.selector_steps) < 32:
        raise SystemExit("detail/selector step budgets are too small for a capacity proof")

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
    output_dir = root / "artifacts/nsamdr/parallel_detail_diagnostics" / f"PARALLEL_{stamp}"
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
    service._validate_v992_architecture_contract(model.architecture_contract())
    optimizer, _optimizer_mode = service._build_optimizer(model, config, device)
    amp_dtype = service._resolve_amp_dtype(config, device)
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler, init_scale=config.amp_initial_scale)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler, init_scale=config.amp_initial_scale)
    moved = service._move_batch(batch_cpu, device, channels_last=False)

    print("=" * 78, flush=True)
    print("NSAMDR RAVEN PARALLEL DETAIL INTEGRATION PROOF V1", flush=True)
    print("Authority                : diagnostic only / NON-PROMOTABLE", flush=True)
    print("Composition              : B -> direct detail candidate -> BenefitSelector -> F", flush=True)
    print("Geometry/seam            : execute, but cannot alter D candidate/selector evidence", flush=True)
    print(f"Detail/selector steps    : {args.detail_steps} / {args.selector_steps}", flush=True)
    print(f"Artifacts                : {output_dir}", flush=True)
    print("=" * 78, flush=True)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    model.set_phase("detail-reconstruction")
    initial_outputs, initial_candidate, initial_final = _evaluate(
        service, model, moved, config, "detail-reconstruction", device, amp_dtype
    )
    _write_probe(
        output_dir / "parallel_probe_initial.png",
        initial_outputs,
        moved,
        initial_candidate,
        initial_final,
        "INITIAL PARALLEL PRODUCTION FORWARD",
    )

    detail_outputs, detail_candidate, detail_final = _train_phase(
        service=service,
        training=training,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        batch=moved,
        config=config,
        phase="detail-reconstruction",
        steps=int(args.detail_steps),
        device=device,
        amp_dtype=amp_dtype,
        rows=rows,
    )
    candidate_pass = direct._cap_passes(
        detail_candidate,
        float(args.required_edge_recovery),
        float(args.required_global_recovery),
    )
    _write_probe(
        output_dir / "parallel_probe_detail.png",
        detail_outputs,
        moved,
        detail_candidate,
        detail_final,
        "DIRECT DETAIL TRAINED IN PRODUCTION GRAPH",
    )

    if candidate_pass:
        final_outputs, final_candidate, final_metrics = _train_phase(
            service=service,
            training=training,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            batch=moved,
            config=config,
            phase="physical-finetune",
            steps=int(args.selector_steps),
            device=device,
            amp_dtype=amp_dtype,
            rows=rows,
        )
    else:
        final_outputs, final_candidate, final_metrics = detail_outputs, detail_candidate, detail_final

    edge_retention = _retention(
        float(final_candidate["edgeRecovery"]), float(final_metrics["edgeRecovery"])
    )
    global_retention = _retention(
        float(final_candidate["globalRecovery"]), float(final_metrics["globalRecovery"])
    )
    selector_pass = (
        candidate_pass
        and edge_retention >= float(args.required_retention)
        and global_retention >= float(args.required_retention)
        and float(final_metrics["edgeRecovery"]) > 0.0
        and float(final_metrics["globalRecovery"]) > 0.0
    )
    status = "passed" if selector_pass else (
        "failed-parallel-detail-candidate" if not candidate_pass else "failed-selector-retention"
    )

    final_probe = output_dir / "parallel_probe_final.png"
    _write_probe(
        final_probe,
        final_outputs,
        moved,
        final_candidate,
        final_metrics,
        "FINAL PARALLEL PRODUCTION COMPOSITION",
    )
    if rows:
        with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
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
        "detailSteps": int(args.detail_steps),
        "selectorSteps": int(args.selector_steps),
        "requiredEdgeRecovery": float(args.required_edge_recovery),
        "requiredGlobalRecovery": float(args.required_global_recovery),
        "requiredRetention": float(args.required_retention),
        "initialCandidate": initial_candidate,
        "detailCandidate": detail_candidate,
        "finalCandidate": final_candidate,
        "final": final_metrics,
        "candidatePass": bool(candidate_pass),
        "selectorPass": bool(selector_pass),
        "edgeRetention": float(edge_retention),
        "globalRetention": float(global_retention),
        "selectorMean": float(
            final_outputs["benefit_selector_probability"].detach().float().mean().item()
        ),
        "elapsedSeconds": time.perf_counter() - started,
        "finalProbeSheet": str(final_probe.resolve()),
    }
    (output_dir / "parallel_detail_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    zip_path = legacy._zip_diagnostics(output_dir)

    print("=" * 78, flush=True)
    print(f"PARALLEL DETAIL INTEGRATION : {'PASS' if selector_pass else 'FAIL'}", flush=True)
    print(f"Detail candidate edge/global: {detail_candidate['edgeRecovery']:+.2%} / {detail_candidate['globalRecovery']:+.2%}", flush=True)
    print(f"Final edge/global           : {final_metrics['edgeRecovery']:+.2%} / {final_metrics['globalRecovery']:+.2%}", flush=True)
    print(f"Selector retention edge/glob: {edge_retention:.1%} / {global_retention:.1%}", flush=True)
    print(f"Diagnostics                 : {zip_path}", flush=True)
    print("=" * 78, flush=True)
    if args.open_result:
        direct._open_result(final_probe)
    return 0 if selector_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())