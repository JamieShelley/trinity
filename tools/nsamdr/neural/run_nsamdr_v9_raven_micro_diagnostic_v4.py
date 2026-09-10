#!/usr/bin/env python3
"""V12.6 staged Micro qualification for independent G/S/D -> U -> F composition.

Fast early stages keep the small repeated-patch budget. Detail receives one
production-epoch-equivalent update budget because the isolated proof needed 1,856
updates. BenefitSelector receives its own bounded capacity budget during the actual
physical-finetune epochs. Qualification checks D capacity, U/D retention, F/U
retention, regressive-specialist suppression, and >=99% protected-B preservation.
"""
from __future__ import annotations

import json
import math
import random
import time
from typing import Any

import numpy as np
import torch

import _run_nsamdr_v9_raven_micro_diagnostic_impl as _implementation
import run_nsamdr_v9_raven_micro_overfit as legacy
import run_nsamdr_v9_raven_micro_diagnostic_v3 as v3
from v9.application.backend import TrainingBackend
from v9.baseline_relative_specialist_contract import (
    PROTECTED_PRESERVATION_REQUIRED,
    protected_preservation_terms,
)
from v9.config import V9Config
from v9.diagnostics_layout import run_consolidated_diagnostic
from v9.inference import resolve_device
from v9.model import FidelityResidualNetV9, MODEL_SCHEMA, parameter_count


REPORT_SCHEMA = "NSAMDR_RAVEN_MICRO_PARALLEL_SPECIALIST_DIAGNOSTIC_V4"
DEFAULT_SELECTOR_CAPACITY_STEPS = 512
DEFAULT_FUSION_RETENTION = 0.95
DEFAULT_FINAL_RETENTION = 0.85
REGRESSIVE_WEIGHT_MAX = 0.01


def _stage_metrics_v126(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    result = v3._stage_metrics_v125(outputs, batch)
    authority = result.setdefault("authority", {})
    for label, key in (
        ("parallelStructureFusionWeightMean", "parallel_structure_fusion_weight"),
        ("parallelSeamFusionWeightMean", "parallel_seam_fusion_weight"),
        ("parallelDetailFusionWeightMean", "parallel_detail_fusion_weight"),
    ):
        value = outputs.get(key)
        authority[label] = (
            float(value.detach().float().mean().item())
            if isinstance(value, torch.Tensor) and value.numel() > 0
            else None
        )
    return result


def _parser() -> Any:
    parser = _implementation._parser()
    parser.description = (
        "Run V12.6 staged production composition on one fixed Raven patch with "
        "explicit detail/selector capacity budgets and U/F retention checks."
    )
    parser.add_argument(
        "--detail-capacity-steps",
        type=int,
        default=0,
        help="Total detail updates; 0 uses one production tilesPerEpoch budget.",
    )
    parser.add_argument(
        "--selector-capacity-steps",
        type=int,
        default=DEFAULT_SELECTOR_CAPACITY_STEPS,
        help="Total BenefitSelector updates spread across physical-finetune epochs.",
    )
    parser.add_argument(
        "--required-fusion-retention", type=float, default=DEFAULT_FUSION_RETENTION
    )
    parser.add_argument(
        "--required-final-retention", type=float, default=DEFAULT_FINAL_RETENTION
    )
    return parser


def _recovery(metrics: dict[str, Any], key: str, field: str) -> float:
    return _implementation._candidate_recovery(metrics, key, field)


def _retention(source: float, retained: float) -> float:
    if not math.isfinite(source) or not math.isfinite(retained) or source <= 1.0e-8:
        return -float("inf")
    return retained / source


def _safe_specialist(recovery: float, weight: float | None) -> bool:
    if recovery >= 0.0:
        return True
    return weight is not None and math.isfinite(weight) and weight <= REGRESSIVE_WEIGHT_MAX


def _main_impl(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    if int(args.tile_size) < 32 or int(args.tile_size) % 16 != 0:
        raise SystemExit("--tile-size must be >=32 and divisible by 16")
    if int(args.steps_per_epoch) < 4:
        raise SystemExit("--steps-per-epoch must be >=4")
    if int(args.detail_capacity_steps) < 0:
        raise SystemExit("--detail-capacity-steps must be >=0")
    if int(args.selector_capacity_steps) < 4:
        raise SystemExit("--selector-capacity-steps must be >=4")
    for name, value in (
        ("--required-recovery", args.required_recovery),
        ("--required-fusion-retention", args.required_fusion_retention),
        ("--required-final-retention", args.required_final_retention),
    ):
        if not 0.0 < float(value) <= 1.0:
            raise SystemExit(f"{name} must be in (0, 1]")

    legacy._ensure_raven_dataset(
        root,
        root / legacy.RAVEN_CONFIG,
        args.shared_cache,
        bool(args.rebuild_dataset),
    )
    config, _raven = legacy._micro_config(root, args)
    config.training_activation_checkpointing = False
    config.validate()

    production_reference = V9Config.load(root / legacy.PRODUCTION_CONFIG)
    detail_capacity_steps = (
        int(args.detail_capacity_steps)
        if int(args.detail_capacity_steps) > 0
        else int(production_reference.tiles_per_epoch)
    )
    detail_epochs = max(int(config.detail_epochs), 1)
    selector_epochs = max(int(config.physical_finetune_epochs), 1)
    detail_steps_per_epoch = max(
        int(args.steps_per_epoch),
        int(math.ceil(detail_capacity_steps / detail_epochs)),
    )
    selector_steps_per_epoch = max(
        int(args.steps_per_epoch),
        int(math.ceil(int(args.selector_capacity_steps) / selector_epochs)),
    )

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
    resolved.update(
        {
            "diagnosticMode": "raven-micro-parallel-specialists-v4",
            "promotable": False,
            "baseStepsPerEpoch": int(args.steps_per_epoch),
            "detailCapacitySteps": detail_capacity_steps,
            "detailStepsPerEpoch": detail_steps_per_epoch,
            "selectorCapacitySteps": int(args.selector_capacity_steps),
            "selectorStepsPerEpoch": selector_steps_per_epoch,
            "requiredDetailEdgeRecovery": float(args.required_recovery),
            "requiredFusionRetention": float(args.required_fusion_retention),
            "requiredFinalRetention": float(args.required_final_retention),
            "requiredProtectedPreservation": PROTECTED_PRESERVATION_REQUIRED,
            "activationCheckpointingDisabledForTinyPatch": True,
        }
    )
    (output_dir / "resolved_micro_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=" * 78, flush=True)
    print("NSAMDR RAVEN STAGED MICRO V4 — V12.6 COMPOSITION", flush=True)
    print("Authority                : diagnostic only / NON-PROMOTABLE", flush=True)
    print(f"Patch                    : {config.tile_size} LR -> {config.tile_size * 4} HR", flush=True)
    print(f"Base steps / epoch       : {int(args.steps_per_epoch)}", flush=True)
    print(
        f"Detail capacity budget   : {detail_capacity_steps} total "
        f"({detail_steps_per_epoch}/detail epoch)",
        flush=True,
    )
    print(
        f"Selector capacity budget : {int(args.selector_capacity_steps)} total "
        f"({selector_steps_per_epoch}/physical-finetune epoch)",
        flush=True,
    )
    print("Probe                    : B -> independent G/S/D -> U -> F", flush=True)
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
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_scaler, init_scale=config.amp_initial_scale
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_scaler, init_scale=config.amp_initial_scale
        )

    moved = service._move_batch(batch_cpu, device, channels_last=False)
    model.set_phase("sdf-bootstrap")
    initial_outputs, initial_metrics = _implementation._evaluate(
        service, model, moved, config, "sdf-bootstrap", device, amp_dtype
    )
    _implementation._write_probe_sheet(
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
        elif phase == "detail-reconstruction":
            steps = detail_steps_per_epoch
        elif phase == "physical-finetune":
            steps = selector_steps_per_epoch
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))

        model.train()
        epoch_started = time.perf_counter()
        loss_sum = 0.0
        loss_term_sums: dict[str, float] = {}
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                outputs = service._forward_for_phase(model, moved, phase, config)
            with torch.autocast(device_type=device.type, enabled=False):
                losses = training.compute_losses(outputs, moved, config, phase)
            total = losses["total"].float()
            if not bool(torch.isfinite(total).item()):
                raise RuntimeError(f"non-finite micro loss epoch={epoch} step={step}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
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
        epoch_outputs, metrics = _implementation._evaluate(
            service, model, moved, config, phase, device, amp_dtype
        )
        elapsed = time.perf_counter() - epoch_started
        avg_loss = loss_sum / max(steps, 1)
        rows.append(
            _implementation._flatten_metrics(
                epoch, phase, steps, elapsed, avg_loss, metrics
            )
        )
        epoch_reports.append(
            {
                "epoch": epoch,
                "phase": phase,
                "steps": steps,
                "elapsedSeconds": elapsed,
                "metrics": metrics,
                "lossTerms": {
                    key: value / max(steps, 1)
                    for key, value in sorted(loss_term_sums.items())
                },
            }
        )

        candidates = metrics.get("candidates", {})
        g = float(candidates.get("preSeam", {}).get("edgeRecovery", 0.0))
        s = float(candidates.get("postSeam", {}).get("edgeRecovery", 0.0))
        d = float(candidates.get("detailCandidate", {}).get("edgeRecovery", 0.0))
        u = float(candidates.get("fusedCandidate", {}).get("edgeRecovery", 0.0))
        f = float(candidates.get("final", {}).get("edgeRecovery", 0.0))
        gate = metrics.get("authority", {}).get("finalSelectorGateMean")
        gate_text = "n/a" if gate is None else f"{gate:.3f}"
        print(
            f"[micro-v4] e{epoch:03d} {phase}: G={g:+.1%} S={s:+.1%} "
            f"D={d:+.1%} U={u:+.1%} F={f:+.1%} gate={gate_text}",
            flush=True,
        )
        _implementation._write_probe_sheet(
            output_dir / "micro_probe_current.png",
            epoch_outputs,
            moved,
            metrics,
            f"EPOCH {epoch:03d} {phase}",
        )

    TrainingBackend._prepare_production_runtime(model)
    model.set_phase("physical-finetune")
    final_outputs, final_metrics = _implementation._evaluate(
        service, model, moved, config, "physical-finetune", device, amp_dtype
    )
    final_sheet = output_dir / "micro_probe_final.png"
    _implementation._write_probe_sheet(
        final_sheet,
        final_outputs,
        moved,
        final_metrics,
        "FINAL V12.6 PRODUCTION COMPOSITION",
    )
    _implementation._write_csv(output_dir / "metrics.csv", rows)
    (output_dir / "epoch_diagnostics.json").write_text(
        json.dumps(epoch_reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    d_edge = _recovery(final_metrics, "detailCandidate", "edgeRecovery")
    d_global = _recovery(final_metrics, "detailCandidate", "globalRecovery")
    u_edge = _recovery(final_metrics, "fusedCandidate", "edgeRecovery")
    u_global = _recovery(final_metrics, "fusedCandidate", "globalRecovery")
    f_edge = _recovery(final_metrics, "final", "edgeRecovery")
    f_global = _recovery(final_metrics, "final", "globalRecovery")
    g_edge = _recovery(final_metrics, "preSeam", "edgeRecovery")
    s_edge = _recovery(final_metrics, "postSeam", "edgeRecovery")

    detail_capacity_pass = bool(
        d_edge >= float(args.required_recovery) and d_global > 0.0
    )
    fusion_edge_retention = _retention(d_edge, u_edge)
    fusion_global_retention = _retention(d_global, u_global)
    fusion_retention_pass = bool(
        detail_capacity_pass
        and fusion_edge_retention >= float(args.required_fusion_retention)
        and fusion_global_retention >= float(args.required_fusion_retention)
    )
    final_edge_retention = _retention(u_edge, f_edge)
    final_global_retention = _retention(u_global, f_global)
    final_retention_pass = bool(
        fusion_retention_pass
        and final_edge_retention >= float(args.required_final_retention)
        and final_global_retention >= float(args.required_final_retention)
    )

    authority = final_metrics.get("authority", {})
    structure_weight = authority.get("parallelStructureFusionWeightMean")
    seam_weight = authority.get("parallelSeamFusionWeightMean")
    structure_safety_pass = _safe_specialist(g_edge, structure_weight)
    seam_safety_pass = _safe_specialist(s_edge, seam_weight)
    specialist_safety_pass = bool(structure_safety_pass and seam_safety_pass)

    preservation = protected_preservation_terms(
        final_outputs["albedo"],
        final_outputs["baseline_albedo"],
        moved["target_albedo"],
    )
    preservation_rate = float(
        preservation["protected_preservation_rate"].detach().cpu().item()
    )
    protected_preservation_pass = bool(
        preservation_rate >= float(PROTECTED_PRESERVATION_REQUIRED)
    )

    production_authority_pass = bool(
        detail_capacity_pass
        and fusion_retention_pass
        and final_retention_pass
        and specialist_safety_pass
        and protected_preservation_pass
    )
    if production_authority_pass:
        interpretation = (
            "detail capacity, specialist fusion, final selector retention and protected "
            "baseline preservation all demonstrated"
        )
    elif not detail_capacity_pass:
        interpretation = "staged production detail path did not reproduce isolated D capacity"
    elif not fusion_retention_pass:
        interpretation = "full specialist fusion diluted the proven D improvement"
    elif not final_retention_pass:
        interpretation = "BenefitSelector suppressed too much of fused U improvement"
    elif not specialist_safety_pass:
        interpretation = "a regressive specialist retained material fusion authority"
    else:
        interpretation = "final selector violated protected baseline preservation"

    report = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if production_authority_pass else "failed-composition",
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
        "baseStepsPerEpoch": int(args.steps_per_epoch),
        "detailCapacitySteps": detail_capacity_steps,
        "detailStepsPerEpoch": detail_steps_per_epoch,
        "selectorCapacitySteps": int(args.selector_capacity_steps),
        "selectorStepsPerEpoch": selector_steps_per_epoch,
        "requiredDetailEdgeRecovery": float(args.required_recovery),
        "requiredFusionRetention": float(args.required_fusion_retention),
        "requiredFinalRetention": float(args.required_final_retention),
        "requiredProtectedPreservation": float(PROTECTED_PRESERVATION_REQUIRED),
        "detailCapacityPass": detail_capacity_pass,
        "fusionRetentionPass": fusion_retention_pass,
        "finalRetentionPass": final_retention_pass,
        "structureSafetyPass": structure_safety_pass,
        "seamSafetyPass": seam_safety_pass,
        "specialistSafetyPass": specialist_safety_pass,
        "protectedPreservationPass": protected_preservation_pass,
        "productionAuthorityPass": production_authority_pass,
        "fusionEdgeRetention": fusion_edge_retention,
        "fusionGlobalRetention": fusion_global_retention,
        "finalEdgeRetention": final_edge_retention,
        "finalGlobalRetention": final_global_retention,
        "protectedPreservationRate": preservation_rate,
        "final": final_metrics,
        "interpretation": interpretation,
        "finalProbeSheet": str(final_sheet.resolve()),
    }
    (output_dir / "micro_capacity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    zip_path = legacy._zip_diagnostics(output_dir)

    print("=" * 78, flush=True)
    print(f"MICRO DETAIL CAPACITY      : {'PASS' if detail_capacity_pass else 'FAIL'}", flush=True)
    print(f"MICRO FUSION RETENTION     : {'PASS' if fusion_retention_pass else 'FAIL'}", flush=True)
    print(f"MICRO FINAL RETENTION      : {'PASS' if final_retention_pass else 'FAIL'}", flush=True)
    print(f"MICRO SPECIALIST SAFETY    : {'PASS' if specialist_safety_pass else 'FAIL'}", flush=True)
    print(f"MICRO PROTECTED PRESERVE   : {'PASS' if protected_preservation_pass else 'FAIL'} ({preservation_rate:.2%})", flush=True)
    print(f"MICRO COMPOSITION          : {'PASS' if production_authority_pass else 'FAIL'}", flush=True)
    print(f"Interpretation             : {interpretation}", flush=True)
    print(f"Diagnostics                : {zip_path}", flush=True)
    print("=" * 78, flush=True)
    if args.open_result:
        _implementation._open_result(final_sheet)
    return 0 if production_authority_pass else 2


def main(argv: list[str] | None = None) -> int:
    _implementation.REPORT_SCHEMA = REPORT_SCHEMA
    _implementation.CANDIDATE_KEYS = v3.CANDIDATE_KEYS
    _implementation._stage_metrics = _stage_metrics_v126
    _implementation._write_probe_sheet = v3._write_probe_sheet_v125
    return run_consolidated_diagnostic(
        _main_impl,
        argv,
        legacy_folder="micro_diagnostics",
        category="micro",
    )


if __name__ == "__main__":
    raise SystemExit(main())
