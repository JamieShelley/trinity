#!/usr/bin/env python3
"""V12.11 Raven Micro dependency-correct structural qualification.

The V12.10 run proved the direct same-edge spline teacher is now working: point
wins reached ~98% and point gain stayed strongly positive.  The run nevertheless
spent all 3,072 G1 steps because V12.10 still required the *rendered* structure
candidate to be non-regressive before the boundary/profile specialist had ever
been trained.

That orders the dependency backwards.  V12.11 therefore makes G1 a pure geometry
qualification, then gives P ownership of profile learning and only after P is
trained requires the deployed pre-seam structure candidate to be non-regressive.
The profile gate also uses the canonical ``boundary_specialist_recovery`` metric
(coverage-error recovery) rather than incorrectly comparing the configured 70%
threshold to full-image RGB edge recovery.

This file changes diagnostic semantics only.  V12.10 production geometry training
remains installed unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

import run_nsamdr_v9_raven_micro_diagnostic_v5 as v5
import run_nsamdr_v9_raven_micro_diagnostic_v7 as v7


REPORT_SCHEMA = "NSAMDR_RAVEN_MICRO_HARD_QUALIFICATION_V8"
QUALIFICATION_REVISION = "V12.11"
PROFILE_PLATEAU_MIN_STEPS = 384
PROFILE_PLATEAU_PATIENCE_EVALS = 5
PROFILE_PLATEAU_MIN_SCORE_DELTA = 0.005

_INSTALLED = False
_PREVIOUS_RUN_STAGE: Any = None


def _finite_metric(item: dict[str, Any], key: str, default: float) -> float:
    value = item.get(key, default)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float(default)


def _g1_geometry_only_score(item: dict[str, Any]) -> float:
    """Rank the actual connected-spline geometry, not the untrained profile render."""
    geometry = item["geometry"]
    losses = item.get("lossTerms", {})
    point_gain = _finite_metric(losses, "spline_graph_point_gain", -1.0)
    point_win = _finite_metric(losses, "spline_graph_point_win_fraction", 0.0)
    point_regret = _finite_metric(losses, "spline_graph_point_regret", 1.0)
    contour_gain = _finite_metric(geometry, "relativeContourGain", -1.0)
    topology_penalty = _finite_metric(geometry, "topologyRegression", 1.0)
    return float(
        2.0 * point_gain
        + point_win
        - point_regret
        + 0.50 * contour_gain
        - 4.0 * topology_penalty
    )


def _g1_geometry_only_qualified(item: dict[str, Any], config: Any) -> bool:
    """Qualify continuous geometry independently of the not-yet-trained profile stage."""
    geometry = item["geometry"]
    losses = item.get("lossTerms", {})
    point = _finite_metric(losses, "spline_graph_point", float("inf"))
    point_gain = _finite_metric(losses, "spline_graph_point_gain", -float("inf"))
    point_win = _finite_metric(losses, "spline_graph_point_win_fraction", 0.0)
    point_regret = _finite_metric(losses, "spline_graph_point_regret", float("inf"))
    teacher_coverage = _finite_metric(
        losses, "spline_graph_same_edge_teacher_coverage", 0.0
    )
    win_required = float(getattr(config, "sdf_relative_win_fraction", 0.65))
    regression_max = float(
        getattr(config, "sdf_relative_regression_fraction", 0.20)
    )
    missing_tolerance = float(
        getattr(config, "sdf_missing_contour_tolerance", 0.0)
    )
    catastrophic = float(
        getattr(config, "sdf_catastrophic_chamfer_pixels", 48.0)
    )
    return bool(
        teacher_coverage > 0.10
        and point_gain > 0.0
        and point_win >= win_required
        and point_regret <= point
        and _finite_metric(geometry, "relativeContourGain", -1.0) > 0.0
        and _finite_metric(geometry, "regressionFraction", 1.0) <= regression_max
        and _finite_metric(geometry, "predictedMissingContourFraction", 1.0)
        <= _finite_metric(geometry, "sourceMissingContourFraction", 0.0)
        + missing_tolerance
        and _finite_metric(geometry, "topologyRegression", 1.0) == 0.0
        and _finite_metric(geometry, "predictedChamferPixels", float("inf"))
        <= catastrophic
    )


def _profile_eval_recovery(
    training: Any,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
) -> float:
    """Read the canonical coverage-error recovery from the current evaluated state."""
    with torch.no_grad(), torch.autocast(device_type=outputs["albedo"].device.type, enabled=False):
        losses = training.compute_losses(outputs, batch, config, "gate-proof")
    value = losses.get("boundary_specialist_recovery")
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        scalar = float(value.detach().float().cpu().item())
        if math.isfinite(scalar):
            return scalar
    return -float("inf")


def _run_profile_failfast(**kwargs: Any) -> dict[str, Any]:
    """Train P, qualify its canonical recovery, then require deployed G to be safe."""
    name = str(kwargs["name"])
    phase = str(kwargs["phase"])
    max_steps = int(kwargs["max_steps"])
    eval_interval = max(1, int(kwargs["eval_interval"]))
    learning_rate = float(kwargs["learning_rate"])
    model = kwargs["model"]
    service = kwargs["service"]
    training = kwargs["training"]
    optimizer = kwargs["optimizer"]
    scaler = kwargs["scaler"]
    batch = kwargs["batch"]
    config = kwargs["config"]
    device = kwargs["device"]
    amp_dtype = kwargs["amp_dtype"]
    output_dir = Path(kwargs["output_dir"])
    enrich = kwargs["enrich"]

    required = float(getattr(config, "boundary_specialist_recovery_required", 0.70))
    stage_dir = output_dir / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    meaningful_best = -float("inf")
    stale_evaluations = 0
    best_path = stage_dir / "best_state.pt"
    evaluations: list[dict[str, Any]] = []
    step = 0

    while step < max_steps:
        stop = min(max_steps, step + eval_interval)
        loss_terms = v5._train_updates(
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
        outputs, base_metrics = v5._evaluate(
            service, model, batch, config, phase, device, amp_dtype
        )
        extra = enrich(outputs, base_metrics)
        profile_recovery = _profile_eval_recovery(
            training, outputs, batch, config
        )
        structure_edge = v5._candidate_recovery(
            base_metrics, "preSeam", "edgeRecovery"
        )
        structure_global = v5._candidate_recovery(
            base_metrics, "preSeam", "globalRecovery"
        )
        merged = {
            "step": int(step),
            "phase": phase,
            "lossTerms": loss_terms,
            "metrics": base_metrics,
            **extra,
            "profileRecovery": float(profile_recovery),
            "profileRecoveryRequired": float(required),
            "structureEdgeRecoveryAfterProfile": float(structure_edge),
            "structureGlobalRecoveryAfterProfile": float(structure_global),
        }
        current_score = float(
            3.0 * profile_recovery
            + structure_edge
            + 0.25 * structure_global
        )
        qualified = bool(
            profile_recovery >= required
            and structure_edge >= 0.0
            and structure_global >= 0.0
        )
        merged["score"] = current_score
        merged["qualified"] = qualified
        evaluations.append(merged)

        if current_score > best_score:
            best_score = current_score
            v5._save_state(
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
            v5._stage_probe(
                stage_dir / "best_probe.png",
                outputs,
                batch,
                base_metrics,
                f"{name} BEST step {step}",
            )

        if current_score >= meaningful_best + float(PROFILE_PLATEAU_MIN_SCORE_DELTA):
            meaningful_best = current_score
            stale_evaluations = 0
        elif step >= int(PROFILE_PLATEAU_MIN_STEPS):
            stale_evaluations += 1

        print(
            f"[micro-v8] {name:<25} step={step:4d}/{max_steps:4d} "
            f"profileRec={profile_recovery:+.1%}/{required:.1%} "
            f"Gedge={structure_edge:+.1%} Gglobal={structure_global:+.1%} "
            f"stale={stale_evaluations}/{PROFILE_PLATEAU_PATIENCE_EVALS} "
            f"{'PASS' if qualified else '...'}",
            flush=True,
        )

        if qualified:
            v5._save_state(
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
                "stoppedEarly": False,
                "final": merged,
            }

        if (
            step >= int(PROFILE_PLATEAU_MIN_STEPS)
            and stale_evaluations >= int(PROFILE_PLATEAU_PATIENCE_EVALS)
        ):
            if best_path.is_file():
                v5._restore_state(model, best_path)
            (stage_dir / "evaluations.json").write_text(
                json.dumps(evaluations, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"[micro-v8] {name} FAIL-FAST plateau at step {step}; "
                f"bestScore={best_score:+.4f}",
                flush=True,
            )
            return {
                "name": name,
                "phase": phase,
                "status": "failed-plateau",
                "stepsUsed": int(step),
                "maxSteps": int(max_steps),
                "bestScore": float(best_score),
                "stoppedEarly": True,
                "stopReason": "profile-or-structure-plateau",
                "plateauPatienceEvaluations": int(PROFILE_PLATEAU_PATIENCE_EVALS),
                "final": evaluations[-1],
            }

    if best_path.is_file():
        v5._restore_state(model, best_path)
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
        "stoppedEarly": False,
        "final": evaluations[-1] if evaluations else {},
    }


def _run_stage_v1211(**kwargs: Any) -> dict[str, Any]:
    name = str(kwargs.get("name", ""))
    if _PREVIOUS_RUN_STAGE is None:
        raise RuntimeError("V12.11 diagnostic installed without previous stage runner")

    if name == "G1_geometry_metric_render":
        result = _PREVIOUS_RUN_STAGE(**kwargs)
        if result.get("status") == "passed":
            # V12.9's outer main still has a legacy immediate rendered-G check.
            # Preserve the real pre-profile values explicitly and defer that check
            # to P, which now owns profile training and final pre-seam render safety.
            final = result.get("final")
            if isinstance(final, dict):
                final["preProfileStructureEdgeRecovery"] = final.get(
                    "structureEdgeRecovery"
                )
                final["preProfileStructureGlobalRecovery"] = final.get(
                    "structureGlobalRecovery"
                )
                final["structureRenderDeferredToProfile"] = True
                final["structureEdgeRecovery"] = 0.0
                final["structureGlobalRecovery"] = 0.0
        return result

    if name == "P_boundary_profile":
        return _run_profile_failfast(**kwargs)

    return _PREVIOUS_RUN_STAGE(**kwargs)


def install_v1211_diagnostic_semantics() -> None:
    global _INSTALLED, _PREVIOUS_RUN_STAGE
    if _INSTALLED:
        return
    v7.install_v1210_diagnostic_semantics()
    # V7's runner resolves these functions dynamically, so replacing them keeps
    # its best-checkpoint/fail-fast machinery while separating geometry from P.
    v7._g1_direct_geometry_score = _g1_geometry_only_score
    v7._g1_direct_geometry_qualified = _g1_geometry_only_qualified
    _PREVIOUS_RUN_STAGE = v5._run_stage
    v5.REPORT_SCHEMA = REPORT_SCHEMA
    v5.QUALIFICATION_REVISION = QUALIFICATION_REVISION
    v5._run_stage = _run_stage_v1211
    _INSTALLED = True


def main(argv: list[str] | None = None) -> int:
    install_v1211_diagnostic_semantics()
    print(
        "[micro-v8] V12.11: geometry-only G1 -> canonical P recovery -> deployed G render safety",
        flush=True,
    )
    return v5.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
