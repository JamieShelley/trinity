#!/usr/bin/env python3
"""V12.9 hard-gated staged Raven qualification ladder.

This diagnostic is intentionally non-promotable.  It gives each production
specialist a realistic bounded capacity budget, evaluates it periodically, saves
its best state/evidence, and stops at the first failed prerequisite.  A later
specialist can therefore never hide an unqualified upstream specialist.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

import _run_nsamdr_v9_raven_micro_diagnostic_impl as _implementation
import run_nsamdr_v9_raven_micro_overfit as legacy
import run_nsamdr_v9_raven_micro_diagnostic_v3 as v3
import run_nsamdr_v9_raven_micro_diagnostic_v4 as v4
from v9.application.backend import TrainingBackend
from v9.baseline_relative_specialist_contract import (
    PROTECTED_PRESERVATION_REQUIRED,
    protected_preservation_terms,
)
from v9.diagnostics_layout import run_consolidated_diagnostic
from v9.geometry_metrics import sdf_topology_mismatch, zero_contour_distance
from v9.inference import resolve_device
from v9.model import FidelityResidualNetV9, MODEL_SCHEMA, parameter_count


REPORT_SCHEMA = "NSAMDR_RAVEN_MICRO_HARD_QUALIFICATION_V5"
QUALIFICATION_REVISION = "V12.9"

DEFAULT_STAGE_BUDGETS = {
    "geometryTopology": (512, 64),
    "geometryMetric": (3072, 128),
    "profile": (1536, 64),
    "seamCapacity": (1536, 64),
    "seamAuthority": (1024, 64),
    "detail": (3072, 128),
    "selector": (1024, 64),
}
REGRESSIVE_WEIGHT_MAX = 0.01
FUSION_RETENTION_REQUIRED = 0.95
FINAL_RETENTION_REQUIRED = 0.85


def _parser() -> Any:
    parser = _implementation._parser()
    parser.description = (
        "Run the V12.9 hard-gated Raven qualification ladder. Each specialist "
        "must pass before the next stage is allowed to train."
    )
    parser.add_argument("--geometry-topology-steps", type=int, default=512)
    parser.add_argument("--geometry-steps", type=int, default=3072)
    parser.add_argument("--profile-steps", type=int, default=1536)
    parser.add_argument("--seam-capacity-steps", type=int, default=1536)
    parser.add_argument("--seam-authority-steps", type=int, default=1024)
    parser.add_argument("--detail-capacity-steps", type=int, default=3072)
    parser.add_argument("--selector-capacity-steps", type=int, default=1024)
    # Retain the V4 CLI/GUI flags so existing launchers continue to work while
    # V12.9 owns the stricter ladder. The values remain user-configurable.
    parser.add_argument(
        "--required-fusion-retention",
        type=float,
        default=FUSION_RETENTION_REQUIRED,
    )
    parser.add_argument(
        "--required-final-retention",
        type=float,
        default=FINAL_RETENTION_REQUIRED,
    )
    parser.add_argument("--geometry-eval-interval", type=int, default=128)
    parser.add_argument("--stage-eval-interval", type=int, default=64)
    return parser


def _cpu_state(model: FidelityResidualNetV9) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _save_state(
    path: Path,
    model: FidelityResidualNetV9,
    *,
    stage: str,
    step: int,
    score: float,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": MODEL_SCHEMA,
            "diagnosticSchema": REPORT_SCHEMA,
            "stage": stage,
            "step": int(step),
            "score": float(score),
            "metrics": metrics,
            "state_dict": _cpu_state(model),
        },
        path,
    )


def _restore_state(model: FidelityResidualNetV9, path: Path) -> None:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    out = value.detach().float().cpu().numpy()
    if out.ndim == 4:
        out = out[0, 0]
    elif out.ndim == 3:
        out = out[0]
    return np.asarray(out, dtype=np.float32)


def _geometry_metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, float]:
    max_distance = float(config.contour_sdf_max_distance_pixels)
    target = _tensor_to_numpy(batch["target_sdf"]) * max_distance

    source_value = outputs.get("source_sdf_prior_pixels")
    if not isinstance(source_value, torch.Tensor):
        source_value = outputs["source_sdf_prior"].float() * max_distance
    source = _tensor_to_numpy(source_value)

    predicted_value = outputs.get("sdf_pixels")
    if not isinstance(predicted_value, torch.Tensor):
        predicted_value = outputs.get("coarse_sdf_pixels")
    if not isinstance(predicted_value, torch.Tensor):
        predicted_value = outputs["sdf_raw"].float() * max_distance
    predicted = _tensor_to_numpy(predicted_value)

    source_contour = zero_contour_distance(source, target)
    predicted_contour = zero_contour_distance(predicted, target)
    source_chamfer = float(source_contour["chamferPixels"])
    predicted_chamfer = float(predicted_contour["chamferPixels"])
    if math.isfinite(source_chamfer) and math.isfinite(predicted_chamfer):
        relative_gain = (
            source_chamfer - predicted_chamfer
        ) / max(abs(source_chamfer), 1.0e-5)
    elif math.isfinite(predicted_chamfer):
        relative_gain = 1.0
    else:
        relative_gain = -1.0

    band = np.abs(target) <= float(config.sdf_metric_band_pixels)
    if not np.any(band):
        band = np.ones_like(target, dtype=bool)
    source_error = np.abs(source - target)
    predicted_error = np.abs(predicted - target)
    wins = float(np.mean(predicted_error[band] < source_error[band]))
    regressions = float(
        np.mean(
            predicted_error[band]
            > source_error[band] + float(getattr(config, "geometry_regret_margin", 0.002))
        )
    )

    source_topology = float(sdf_topology_mismatch(source, target))
    predicted_topology = float(sdf_topology_mismatch(predicted, target))
    source_missing = float(float(source_contour["predictedCrossings"]) <= 0.0)
    predicted_missing = float(float(predicted_contour["predictedCrossings"]) <= 0.0)

    return {
        "sourceChamferPixels": source_chamfer,
        "predictedChamferPixels": predicted_chamfer,
        "relativeContourGain": float(relative_gain),
        "winFraction": wins,
        "regressionFraction": regressions,
        "sourceTopologyMismatch": source_topology,
        "predictedTopologyMismatch": predicted_topology,
        "topologyRegression": float(predicted_topology > source_topology),
        "sourceMissingContourFraction": source_missing,
        "predictedMissingContourFraction": predicted_missing,
        "predictedCrossings": float(predicted_contour["predictedCrossings"]),
        "targetCrossings": float(predicted_contour["targetCrossings"]),
    }


def _candidate_recovery(
    metrics: dict[str, Any],
    candidate: str,
    field: str,
) -> float:
    item = metrics.get("candidates", {}).get(candidate, {})
    value = item.get(field)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return -float("inf")


def _retention(source: float, retained: float) -> float:
    if not math.isfinite(source) or not math.isfinite(retained) or source <= 1.0e-8:
        return -float("inf")
    return float(retained / source)


def _safe_specialist(recovery: float, weight: float | None) -> bool:
    if recovery >= 0.0:
        return True
    return (
        weight is not None
        and math.isfinite(float(weight))
        and float(weight) <= REGRESSIVE_WEIGHT_MAX
    )


def _profile_teacher_metrics(
    model: FidelityResidualNetV9,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    target_sdf = batch["target_sdf"].float()
    ones = torch.ones_like(target_sdf)
    zeros = torch.zeros_like(target_sdf)
    with torch.no_grad():
        outputs = model._forward_impl(
            batch["input"],
            sdf_override=target_sdf,
            gate_override=ones,
            hardness_override=zeros,
        )
    metrics = v4._stage_metrics_v126(outputs, batch)
    return {
        "teacherGeometryEdgeRecovery": _candidate_recovery(
            metrics, "profiledBoundary", "edgeRecovery"
        ),
        "teacherGeometryGlobalRecovery": _candidate_recovery(
            metrics, "profiledBoundary", "globalRecovery"
        ),
    }


def _forced_seam_metrics(
    model: FidelityResidualNetV9,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    height = int(batch["input"].shape[-2]) * 4
    width = int(batch["input"].shape[-1]) * 4
    ones = torch.ones(
        (batch["input"].shape[0], 1, height, width),
        device=batch["input"].device,
        dtype=batch["input"].dtype,
    )
    with torch.no_grad():
        outputs = model._forward_impl(
            batch["input"],
            seam_authority_override=ones,
        )
    metrics = v4._stage_metrics_v126(outputs, batch)
    return (
        {
            "forcedEdgeRecovery": _candidate_recovery(
                metrics, "postSeam", "edgeRecovery"
            ),
            "forcedGlobalRecovery": _candidate_recovery(
                metrics, "postSeam", "globalRecovery"
            ),
        },
        outputs,
    )


def _seam_authority_iou(
    outputs: dict[str, torch.Tensor],
    forced_outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> float:
    authority = outputs.get("seam_authority")
    if not isinstance(authority, torch.Tensor):
        return 0.0
    baseline = forced_outputs["baseline_albedo"].detach().float()
    candidate = forced_outputs["seam_candidate_albedo"].detach().float()
    target = batch["target_albedo"].detach().float()
    edge = batch["target_edge"].detach().float() >= 0.08

    baseline_error = (baseline - target).abs().mean(dim=1, keepdim=True)
    candidate_error = (candidate - target).abs().mean(dim=1, keepdim=True)
    teacher = (candidate_error + 0.001 < baseline_error) & edge
    prediction = (authority.detach().float() >= 0.5) & edge

    intersection = float((teacher & prediction).sum().item())
    union = float((teacher | prediction).sum().item())
    if union <= 0.0:
        return 1.0 if float(prediction.sum().item()) <= 0.0 else 0.0
    return intersection / union


def _preservation_rate(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> float:
    terms = protected_preservation_terms(
        outputs["albedo"],
        outputs["baseline_albedo"],
        batch["target_albedo"],
    )
    return float(terms["protected_preservation_rate"].detach().cpu().item())


def _stage_probe(
    path: Path,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    metrics: dict[str, Any],
    title: str,
) -> None:
    v3._write_probe_sheet_v125(path, outputs, batch, metrics, title)


def _train_updates(
    *,
    model: FidelityResidualNetV9,
    service: Any,
    training: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
    device: torch.device,
    amp_dtype: torch.dtype,
    start_step: int,
    stop_step: int,
    learning_rate: float,
) -> dict[str, float]:
    model.set_phase(phase)
    if phase == "sdf-proof" and hasattr(model, "set_parametric_substage"):
        model.set_parametric_substage("integration")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))

    use_amp = device.type == "cuda"
    loss_sums: dict[str, float] = {}
    count = 0
    model.train()
    for _step in range(start_step, stop_step + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            outputs = service._forward_for_phase(model, batch, phase, config)
        with torch.autocast(device_type=device.type, enabled=False):
            losses = training.compute_losses(outputs, batch, config, phase)
        total = losses["total"].float()
        if not bool(torch.isfinite(total).item()):
            raise RuntimeError(f"non-finite loss phase={phase} step={_step}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )
        if not bool(torch.isfinite(grad_norm).item()):
            raise RuntimeError(f"non-finite gradient phase={phase} step={_step}")
        scaler.step(optimizer)
        scaler.update()
        if phase == "sdf-proof":
            structure = getattr(model.geometry_net, "production_structure", None)
            restore = getattr(structure, "restore_locked_topology_parameters", None)
            if callable(restore):
                restore()
        count += 1
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                scalar = float(value.detach().float().cpu().item())
                if math.isfinite(scalar):
                    loss_sums[key] = loss_sums.get(key, 0.0) + scalar

    if count:
        return {key: value / count for key, value in loss_sums.items()}
    return {}


def _evaluate(
    service: Any,
    model: FidelityResidualNetV9,
    batch: dict[str, torch.Tensor],
    config: Any,
    phase: str,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    TrainingBackend._prepare_production_runtime(model)
    return _implementation._evaluate(
        service, model, batch, config, phase, device, amp_dtype
    )


def _run_stage(
    *,
    name: str,
    phase: str,
    max_steps: int,
    eval_interval: int,
    learning_rate: float,
    model: FidelityResidualNetV9,
    service: Any,
    training: Any,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    batch: dict[str, torch.Tensor],
    config: Any,
    device: torch.device,
    amp_dtype: torch.dtype,
    output_dir: Path,
    enrich: Callable[
        [dict[str, torch.Tensor], dict[str, Any]],
        dict[str, Any],
    ],
    score: Callable[[dict[str, Any]], float],
    qualified: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    stage_dir = output_dir / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    best_path = stage_dir / "best_state.pt"
    evaluations: list[dict[str, Any]] = []
    step = 0

    while step < max_steps:
        stop = min(max_steps, step + eval_interval)
        loss_terms = _train_updates(
            model=model,
            service=service,
            training=training,
            optimizer=optimizer,
            scaler=scaler,
            batch=batch,
            config=config,
            phase=phase,
            device=device,
            amp_dtype=amp_dtype,
            start_step=step + 1,
            stop_step=stop,
            learning_rate=learning_rate,
        )
        step = stop
        outputs, base_metrics = _evaluate(
            service, model, batch, config, phase, device, amp_dtype
        )
        extra = enrich(outputs, base_metrics)
        merged = {
            "step": int(step),
            "phase": phase,
            "lossTerms": loss_terms,
            "metrics": base_metrics,
            **extra,
        }
        current_score = float(score(merged))
        merged["score"] = current_score
        merged["qualified"] = bool(qualified(merged))
        evaluations.append(merged)

        if current_score > best_score:
            best_score = current_score
            _save_state(
                best_path,
                model,
                stage=name,
                step=step,
                score=current_score,
                metrics=merged,
            )
            (stage_dir / "best_metrics.json").write_text(
                json.dumps(merged, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _stage_probe(
                stage_dir / "best_probe.png",
                outputs,
                batch,
                base_metrics,
                f"{name} BEST step {step}",
            )

        print(
            f"[micro-v5] {name:<18} step={step:4d}/{max_steps:4d} "
            f"score={current_score:+.4f} "
            f"{'PASS' if merged['qualified'] else '...'}",
            flush=True,
        )
        if merged["qualified"]:
            _save_state(
                stage_dir / "qualified_state.pt",
                model,
                stage=name,
                step=step,
                score=current_score,
                metrics=merged,
            )
            (stage_dir / "evaluations.json").write_text(
                json.dumps(evaluations, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return {
                "name": name,
                "phase": phase,
                "status": "passed",
                "stepsUsed": int(step),
                "maxSteps": int(max_steps),
                "bestScore": float(max(best_score, current_score)),
                "final": merged,
            }

    if best_path.is_file():
        _restore_state(model, best_path)
    (stage_dir / "evaluations.json").write_text(
        json.dumps(evaluations, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "name": name,
        "phase": phase,
        "status": "failed",
        "stepsUsed": int(step),
        "maxSteps": int(max_steps),
        "bestScore": float(best_score),
        "final": evaluations[-1] if evaluations else {},
    }


def _main_impl(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    if int(args.tile_size) < 32 or int(args.tile_size) % 16 != 0:
        raise SystemExit("--tile-size must be >=32 and divisible by 16")

    for name in (
        "geometry_topology_steps",
        "geometry_steps",
        "profile_steps",
        "seam_capacity_steps",
        "seam_authority_steps",
        "selector_capacity_steps",
        "geometry_eval_interval",
        "stage_eval_interval",
    ):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >=1")
    if int(args.detail_capacity_steps) < 0:
        raise SystemExit("--detail-capacity-steps must be >=0")
    if not 0.0 < float(args.required_fusion_retention) <= 1.0:
        raise SystemExit("--required-fusion-retention must be in (0, 1]")
    if not 0.0 < float(args.required_final_retention) <= 1.0:
        raise SystemExit("--required-final-retention must be in (0, 1]")

    # V4 exposed 0 as "automatic". Preserve that launch contract and map it to
    # the V12.9 fair detail capacity budget rather than rejecting the GUI command.
    detail_capacity_steps = (
        int(args.detail_capacity_steps)
        if int(args.detail_capacity_steps) > 0
        else int(DEFAULT_STAGE_BUDGETS["detail"][0])
    )
    selector_capacity_steps = max(
        int(args.selector_capacity_steps),
        int(DEFAULT_STAGE_BUDGETS["selector"][0]),
    )

    legacy._ensure_raven_dataset(
        root,
        root / legacy.RAVEN_CONFIG,
        args.shared_cache,
        bool(args.rebuild_dataset),
    )
    config, _raven = legacy._micro_config(root, args)
    config.training_activation_checkpointing = False
    config.validate()

    manifest = legacy.load_dataset_manifest(root, config)
    sample, patch_metadata = legacy._select_hard_patch(manifest, config)
    batch_cpu = legacy._batch(sample)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        root / "artifacts/nsamdr/micro_diagnostics" / f"MICRO_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "selected_patch.json").write_text(
        json.dumps(patch_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    resolved = config.to_dict()
    resolved.update(
        {
            "diagnosticMode": "raven-micro-hard-qualification-v5",
            "qualificationRevision": QUALIFICATION_REVISION,
            "promotable": False,
            "stageBudgets": {
                "geometryTopology": int(args.geometry_topology_steps),
                "geometryMetric": int(args.geometry_steps),
                "profile": int(args.profile_steps),
                "seamCapacity": int(args.seam_capacity_steps),
                "seamAuthority": int(args.seam_authority_steps),
                "detail": detail_capacity_steps,
                "selector": selector_capacity_steps,
            },
            "requiredDetailEdgeRecovery": float(args.required_recovery),
            "requiredFusionRetention": float(args.required_fusion_retention),
            "requiredFinalRetention": float(args.required_final_retention),
            "requiredProtectedPreservation": PROTECTED_PRESERVATION_REQUIRED,
        }
    )
    (output_dir / "resolved_micro_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

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
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_scaler, init_scale=config.amp_initial_scale
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_scaler, init_scale=config.amp_initial_scale
        )

    moved = service._move_batch(batch_cpu, device, channels_last=False)

    print("=" * 78, flush=True)
    print("NSAMDR RAVEN STAGED MICRO V5 — V12.9 HARD QUALIFICATION", flush=True)
    print("Authority : diagnostic only / NON-PROMOTABLE", flush=True)
    print("Ladder    : G-topology -> G-metric/render -> profile -> S-capacity -> "
          "S-authority -> D -> U -> F", flush=True)
    print(f"Artifacts : {output_dir}", flush=True)
    print("=" * 78, flush=True)

    stage_results: list[dict[str, Any]] = []
    pass_flags = {
        "geometryTopologyPass": False,
        "geometryMetricPass": False,
        "structureRenderPass": False,
        "profilePass": False,
        "seamCapacityPass": False,
        "seamAuthorityPass": False,
        "detailPass": False,
        "fusionPass": False,
        "selectorPass": False,
        "protectedPreservationPass": False,
    }

    def stop_report(failed_stage: str) -> int:
        TrainingBackend._prepare_production_runtime(model)
        model.set_phase("physical-finetune")
        failed_outputs, failed_metrics = _implementation._evaluate(
            service, model, moved, config, "physical-finetune", device, amp_dtype
        )
        failed_sheet = output_dir / "micro_probe_final.png"
        _stage_probe(
            failed_sheet,
            failed_outputs,
            moved,
            failed_metrics,
            f"FAILED AT {failed_stage}",
        )
        report = {
            "schema": REPORT_SCHEMA,
            "qualificationRevision": QUALIFICATION_REVISION,
            "status": "failed-stage",
            "failedStage": failed_stage,
            "selectionKind": "diagnostic-non-promotable",
            "promotable": False,
            "qualifiedProductionCheckpoint": False,
            "sourceRevision": legacy._git_state(root),
            "modelSchema": MODEL_SCHEMA,
            "parameterCount": parameter_count(model),
            "device": str(device),
            "optimizer": optimizer_mode,
            "patch": patch_metadata,
            **pass_flags,
            "productionAuthorityPass": False,
            "stages": stage_results,
            "final": failed_metrics,
            "finalProbeSheet": str(failed_sheet.resolve()),
        }
        (output_dir / "stage_summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output_dir / "micro_capacity_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        zip_path = legacy._zip_diagnostics(output_dir)
        print(f"MICRO V5 FAILED AT         : {failed_stage}", flush=True)
        print(f"Diagnostics                : {zip_path}", flush=True)
        if args.open_result:
            _implementation._open_result(failed_sheet)
        return 2

    # G0 — topology only.
    topology = _run_stage(
        name="G0_geometry_topology",
        phase="sdf-bootstrap",
        max_steps=int(args.geometry_topology_steps),
        eval_interval=min(int(args.stage_eval_interval), int(args.geometry_topology_steps)),
        learning_rate=service._phase_lr("sdf-bootstrap", config),
        model=model,
        service=service,
        training=training,
        optimizer=optimizer,
        scaler=scaler,
        batch=moved,
        config=config,
        device=device,
        amp_dtype=amp_dtype,
        output_dir=output_dir,
        enrich=lambda outputs, _metrics: {
            "geometry": _geometry_metrics(outputs, moved, config)
        },
        score=lambda item: (
            -100.0 * item["geometry"]["predictedTopologyMismatch"]
            -10.0 * item["geometry"]["predictedMissingContourFraction"]
            -min(item["geometry"]["predictedChamferPixels"], 1000.0)
        ),
        qualified=lambda item: (
            item["geometry"]["predictedTopologyMismatch"] == 0.0
            and item["geometry"]["predictedMissingContourFraction"]
            <= item["geometry"]["sourceMissingContourFraction"]
            + float(config.sdf_missing_contour_tolerance)
        ),
    )
    stage_results.append(topology)
    pass_flags["geometryTopologyPass"] = topology["status"] == "passed"
    if not pass_flags["geometryTopologyPass"]:
        return stop_report("geometryTopology")

    # G1/G2 — continuous contour accuracy plus deployed rendered G non-regression.
    geometry = _run_stage(
        name="G1_geometry_metric_render",
        phase="sdf-proof",
        max_steps=int(args.geometry_steps),
        eval_interval=min(int(args.geometry_eval_interval), int(args.geometry_steps)),
        learning_rate=service._phase_lr("sdf-proof", config),
        model=model,
        service=service,
        training=training,
        optimizer=optimizer,
        scaler=scaler,
        batch=moved,
        config=config,
        device=device,
        amp_dtype=amp_dtype,
        output_dir=output_dir,
        enrich=lambda outputs, metrics: {
            "geometry": _geometry_metrics(outputs, moved, config),
            "structureEdgeRecovery": _candidate_recovery(
                metrics, "preSeam", "edgeRecovery"
            ),
            "structureGlobalRecovery": _candidate_recovery(
                metrics, "preSeam", "globalRecovery"
            ),
        },
        score=lambda item: (
            4.0 * item["geometry"]["relativeContourGain"]
            + item["geometry"]["winFraction"]
            - item["geometry"]["regressionFraction"]
            + item["structureEdgeRecovery"]
            + 0.25 * item["structureGlobalRecovery"]
            - 4.0 * item["geometry"]["topologyRegression"]
        ),
        qualified=lambda item: (
            item["geometry"]["relativeContourGain"]
            >= float(config.sdf_relative_gain_required)
            and item["geometry"]["winFraction"]
            >= float(config.sdf_relative_win_fraction)
            and item["geometry"]["regressionFraction"]
            <= float(config.sdf_relative_regression_fraction)
            and item["geometry"]["predictedMissingContourFraction"]
            <= item["geometry"]["sourceMissingContourFraction"]
            + float(config.sdf_missing_contour_tolerance)
            and item["geometry"]["topologyRegression"] == 0.0
            and item["structureEdgeRecovery"] >= 0.0
            and item["structureGlobalRecovery"] >= 0.0
        ),
    )
    stage_results.append(geometry)
    pass_flags["geometryMetricPass"] = geometry["status"] == "passed"
    pass_flags["structureRenderPass"] = bool(
        geometry.get("final", {}).get("structureEdgeRecovery", -1.0) >= 0.0
        and geometry.get("final", {}).get("structureGlobalRecovery", -1.0) >= 0.0
    )
    if not (
        pass_flags["geometryMetricPass"] and pass_flags["structureRenderPass"]
    ):
        return stop_report("geometryMetricOrRender")

    # P — boundary/profile. Record both teacher-geometry and learned-G behaviour.
    profile_required = float(
        getattr(config, "boundary_specialist_recovery_required", 0.70)
    )
    profile = _run_stage(
        name="P_boundary_profile",
        phase="gate-proof",
        max_steps=int(args.profile_steps),
        eval_interval=min(int(args.stage_eval_interval), int(args.profile_steps)),
        learning_rate=service._phase_lr("gate-proof", config),
        model=model,
        service=service,
        training=training,
        optimizer=optimizer,
        scaler=scaler,
        batch=moved,
        config=config,
        device=device,
        amp_dtype=amp_dtype,
        output_dir=output_dir,
        enrich=lambda _outputs, metrics: {
            "learnedGeometryEdgeRecovery": _candidate_recovery(
                metrics, "profiledBoundary", "edgeRecovery"
            ),
            "learnedGeometryGlobalRecovery": _candidate_recovery(
                metrics, "profiledBoundary", "globalRecovery"
            ),
            **_profile_teacher_metrics(model, moved),
        },
        score=lambda item: (
            item["learnedGeometryEdgeRecovery"]
            + item["teacherGeometryEdgeRecovery"]
            + 0.25 * item["learnedGeometryGlobalRecovery"]
        ),
        qualified=lambda item: (
            item["teacherGeometryEdgeRecovery"] >= profile_required
            and item["learnedGeometryEdgeRecovery"] >= profile_required
        ),
    )
    stage_results.append(profile)
    pass_flags["profilePass"] = profile["status"] == "passed"
    if not pass_flags["profilePass"]:
        return stop_report("profile")

    # S0 — force authority and prove reconstruction capacity.
    seam_required = float(getattr(config, "seam_forced_recovery_required", 0.70))
    seam_capacity = _run_stage(
        name="S0_forced_seam_capacity",
        phase="seam-proof",
        max_steps=int(args.seam_capacity_steps),
        eval_interval=min(int(args.stage_eval_interval), int(args.seam_capacity_steps)),
        learning_rate=3.0e-3,
        model=model,
        service=service,
        training=training,
        optimizer=optimizer,
        scaler=scaler,
        batch=moved,
        config=config,
        device=device,
        amp_dtype=amp_dtype,
        output_dir=output_dir,
        enrich=lambda _outputs, _metrics: _forced_seam_metrics(model, moved)[0],
        score=lambda item: (
            item["forcedEdgeRecovery"] + 0.25 * item["forcedGlobalRecovery"]
        ),
        qualified=lambda item: item["forcedEdgeRecovery"] >= seam_required,
    )
    stage_results.append(seam_capacity)
    pass_flags["seamCapacityPass"] = seam_capacity["status"] == "passed"
    if not pass_flags["seamCapacityPass"]:
        return stop_report("seamCapacity")

    # S1 — freeze reconstruction via model phase and train authority only.
    seam_iou_required = float(getattr(config, "seam_authority_iou_required", 0.55))

    def authority_enrich(
        outputs: dict[str, torch.Tensor],
        _metrics: dict[str, Any],
    ) -> dict[str, Any]:
        forced_metrics, forced_outputs = _forced_seam_metrics(model, moved)
        return {
            **forced_metrics,
            "seamAuthorityIoU": _seam_authority_iou(
                outputs, forced_outputs, moved
            ),
        }

    seam_authority = _run_stage(
        name="S1_seam_authority",
        phase="seam-authority",
        max_steps=int(args.seam_authority_steps),
        eval_interval=min(int(args.stage_eval_interval), int(args.seam_authority_steps)),
        learning_rate=service._phase_lr("seam-authority", config),
        model=model,
        service=service,
        training=training,
        optimizer=optimizer,
        scaler=scaler,
        batch=moved,
        config=config,
        device=device,
        amp_dtype=amp_dtype,
        output_dir=output_dir,
        enrich=authority_enrich,
        score=lambda item: item["seamAuthorityIoU"],
        qualified=lambda item: item["seamAuthorityIoU"] >= seam_iou_required,
    )
    stage_results.append(seam_authority)
    pass_flags["seamAuthorityPass"] = seam_authority["status"] == "passed"
    if not pass_flags["seamAuthorityPass"]:
        return stop_report("seamAuthority")

    # D — independent B-relative detail capacity.
    detail_required = float(args.required_recovery)
    detail = _run_stage(
        name="D_detail",
        phase="detail-reconstruction",
        max_steps=detail_capacity_steps,
        eval_interval=min(int(args.geometry_eval_interval), detail_capacity_steps),
        learning_rate=service._phase_lr("detail-reconstruction", config),
        model=model,
        service=service,
        training=training,
        optimizer=optimizer,
        scaler=scaler,
        batch=moved,
        config=config,
        device=device,
        amp_dtype=amp_dtype,
        output_dir=output_dir,
        enrich=lambda _outputs, metrics: {
            "detailEdgeRecovery": _candidate_recovery(
                metrics, "detailCandidate", "edgeRecovery"
            ),
            "detailGlobalRecovery": _candidate_recovery(
                metrics, "detailCandidate", "globalRecovery"
            ),
        },
        score=lambda item: (
            item["detailEdgeRecovery"] + 0.25 * item["detailGlobalRecovery"]
        ),
        qualified=lambda item: (
            item["detailEdgeRecovery"] >= detail_required
            and item["detailGlobalRecovery"] > 0.0
        ),
    )
    stage_results.append(detail)
    pass_flags["detailPass"] = detail["status"] == "passed"
    if not pass_flags["detailPass"]:
        return stop_report("detail")

    # U — no training authority in V12.8; this is a hard composition checkpoint.
    model.set_phase("physical-finetune")
    u_outputs, u_metrics = _evaluate(
        service, model, moved, config, "physical-finetune", device, amp_dtype
    )
    d_edge = _candidate_recovery(u_metrics, "detailCandidate", "edgeRecovery")
    d_global = _candidate_recovery(u_metrics, "detailCandidate", "globalRecovery")
    u_edge = _candidate_recovery(u_metrics, "fusedCandidate", "edgeRecovery")
    u_global = _candidate_recovery(u_metrics, "fusedCandidate", "globalRecovery")
    fusion_edge_retention = _retention(d_edge, u_edge)
    fusion_global_retention = _retention(d_global, u_global)
    authority = u_metrics.get("authority", {})
    g_edge = _candidate_recovery(u_metrics, "preSeam", "edgeRecovery")
    s_edge = _candidate_recovery(u_metrics, "postSeam", "edgeRecovery")
    structure_weight = authority.get("parallelStructureFusionWeightMean")
    seam_weight = authority.get("parallelSeamFusionWeightMean")
    specialist_safe = (
        _safe_specialist(g_edge, structure_weight)
        and _safe_specialist(s_edge, seam_weight)
    )
    pass_flags["fusionPass"] = bool(
        fusion_edge_retention >= float(args.required_fusion_retention)
        and fusion_global_retention >= float(args.required_fusion_retention)
        and specialist_safe
    )
    stage_results.append(
        {
            "name": "U_fusion",
            "phase": "composition",
            "status": "passed" if pass_flags["fusionPass"] else "failed",
            "detailEdgeRecovery": d_edge,
            "detailGlobalRecovery": d_global,
            "fusedEdgeRecovery": u_edge,
            "fusedGlobalRecovery": u_global,
            "edgeRetention": fusion_edge_retention,
            "globalRetention": fusion_global_retention,
            "structureRecovery": g_edge,
            "seamRecovery": s_edge,
            "structureWeight": structure_weight,
            "seamWeight": seam_weight,
            "specialistSafetyPass": specialist_safe,
        }
    )
    (output_dir / "U_fusion").mkdir(parents=True, exist_ok=True)
    _stage_probe(
        output_dir / "U_fusion" / "fusion_probe.png",
        u_outputs,
        moved,
        u_metrics,
        "U FUSION QUALIFICATION",
    )
    if not pass_flags["fusionPass"]:
        return stop_report("fusion")

    # F — selector only. Qualification requires both retention and >=99% protected B.
    def selector_enrich(
        outputs: dict[str, torch.Tensor],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        local_u_edge = _candidate_recovery(
            metrics, "fusedCandidate", "edgeRecovery"
        )
        local_u_global = _candidate_recovery(
            metrics, "fusedCandidate", "globalRecovery"
        )
        f_edge = _candidate_recovery(metrics, "final", "edgeRecovery")
        f_global = _candidate_recovery(metrics, "final", "globalRecovery")
        return {
            "fusedEdgeRecovery": local_u_edge,
            "fusedGlobalRecovery": local_u_global,
            "finalEdgeRecovery": f_edge,
            "finalGlobalRecovery": f_global,
            "edgeRetention": _retention(local_u_edge, f_edge),
            "globalRetention": _retention(local_u_global, f_global),
            "protectedPreservationRate": _preservation_rate(outputs, moved),
        }

    selector = _run_stage(
        name="F_selector_preservation",
        phase="physical-finetune",
        max_steps=selector_capacity_steps,
        eval_interval=min(int(args.stage_eval_interval), selector_capacity_steps),
        learning_rate=service._phase_lr("physical-finetune", config),
        model=model,
        service=service,
        training=training,
        optimizer=optimizer,
        scaler=scaler,
        batch=moved,
        config=config,
        device=device,
        amp_dtype=amp_dtype,
        output_dir=output_dir,
        enrich=selector_enrich,
        score=lambda item: (
            4.0 * item["protectedPreservationRate"]
            + item["edgeRetention"]
            + item["globalRetention"]
            + 0.25 * item["finalEdgeRecovery"]
        ),
        qualified=lambda item: (
            item["edgeRetention"] >= float(args.required_final_retention)
            and item["globalRetention"] >= float(args.required_final_retention)
            and item["protectedPreservationRate"]
            >= float(PROTECTED_PRESERVATION_REQUIRED)
        ),
    )
    stage_results.append(selector)
    pass_flags["selectorPass"] = selector["status"] == "passed"
    pass_flags["protectedPreservationPass"] = bool(
        selector.get("final", {}).get("protectedPreservationRate", 0.0)
        >= float(PROTECTED_PRESERVATION_REQUIRED)
    )
    if not (
        pass_flags["selectorPass"]
        and pass_flags["protectedPreservationPass"]
    ):
        return stop_report("selectorOrPreservation")

    final_outputs, final_metrics = _evaluate(
        service, model, moved, config, "physical-finetune", device, amp_dtype
    )
    final_sheet = output_dir / "micro_probe_final.png"
    _stage_probe(
        final_sheet,
        final_outputs,
        moved,
        final_metrics,
        "FINAL V12.9 HARD-QUALIFIED COMPOSITION",
    )

    production_pass = all(pass_flags.values())
    report = {
        "schema": REPORT_SCHEMA,
        "qualificationRevision": QUALIFICATION_REVISION,
        "status": "passed" if production_pass else "failed-stage",
        "failedStage": None if production_pass else "unknown",
        "selectionKind": "diagnostic-non-promotable",
        "promotable": False,
        "qualifiedProductionCheckpoint": False,
        "sourceRevision": legacy._git_state(root),
        "modelSchema": MODEL_SCHEMA,
        "parameterCount": parameter_count(model),
        "device": str(device),
        "optimizer": optimizer_mode,
        "patch": patch_metadata,
        **pass_flags,
        "productionAuthorityPass": production_pass,
        "stages": stage_results,
        "final": final_metrics,
        "finalProbeSheet": str(final_sheet.resolve()),
    }
    (output_dir / "stage_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "micro_capacity_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    zip_path = legacy._zip_diagnostics(output_dir)

    print("=" * 78, flush=True)
    print("MICRO V5 HARD QUALIFICATION: PASS", flush=True)
    print(f"Diagnostics                : {zip_path}", flush=True)
    print("=" * 78, flush=True)
    if args.open_result:
        _implementation._open_result(final_sheet)
    return 0


def main(argv: list[str] | None = None) -> int:
    _implementation.REPORT_SCHEMA = REPORT_SCHEMA
    _implementation.CANDIDATE_KEYS = v3.CANDIDATE_KEYS
    _implementation._stage_metrics = v4._stage_metrics_v126
    _implementation._write_probe_sheet = v3._write_probe_sheet_v125
    return run_consolidated_diagnostic(
        _main_impl,
        argv,
        legacy_folder="micro_diagnostics",
        category="micro",
    )


if __name__ == "__main__":
    raise SystemExit(main())
