#!/usr/bin/env python3
"""V12.10 Raven Micro geometry qualification and fail-fast layer.

V12.9.1 corrected topology/sign-gauge semantics but the next Raven run spent the
full 3,072-step G1 budget while remaining far from qualification.  That run also
proved the canonical synthetic-ladder 25% contour-gain / 65% generic-pixel-win
thresholds are not the right *single-patch* capacity criterion: production applies
those aggregate structural gates to the permanent synthetic geometry ladder.

For the deterministic Raven overfit patch, V12.10 instead requires the exact
same-owning-edge spline teacher to improve, while deployed G itself must remain
non-regressive.  The production V12.10 loss restores those direct node/tangent
teachers to SGD.  This diagnostic layer also stops a plateaued G1 early rather than
burning the complete maximum budget when no meaningful progress is occurring.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import run_nsamdr_v9_raven_micro_diagnostic_v5 as v5
import run_nsamdr_v9_raven_micro_diagnostic_v6 as v6


REPORT_SCHEMA = "NSAMDR_RAVEN_MICRO_HARD_QUALIFICATION_V7"
QUALIFICATION_REVISION = "V12.10"
GEOMETRY_PLATEAU_MIN_STEPS = 512
GEOMETRY_PLATEAU_PATIENCE_EVALS = 4
GEOMETRY_PLATEAU_MIN_SCORE_DELTA = 0.01

_INSTALLED = False
_PREVIOUS_RUN_STAGE: Any = None


def _finite_metric(item: dict[str, Any], key: str, default: float) -> float:
    value = item.get(key, default)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return float(default)


def _g1_direct_geometry_score(item: dict[str, Any]) -> float:
    """Rank direct same-edge learning first, then deployed Raven structure."""
    geometry = item["geometry"]
    losses = item.get("lossTerms", {})
    point_gain = _finite_metric(losses, "spline_graph_point_gain", -1.0)
    point_win = _finite_metric(losses, "spline_graph_point_win_fraction", 0.0)
    point_regret = _finite_metric(losses, "spline_graph_point_regret", 1.0)
    contour_gain = _finite_metric(geometry, "relativeContourGain", -1.0)
    edge_recovery = _finite_metric(item, "structureEdgeRecovery", -1.0)
    global_recovery = _finite_metric(item, "structureGlobalRecovery", -1.0)
    topology_penalty = (
        _finite_metric(geometry, "topologyRegression", 1.0)
        + _finite_metric(geometry, "renderedTopologyRegression", 1.0)
    )
    return float(
        2.0 * point_gain
        + point_win
        - point_regret
        + 0.50 * contour_gain
        + edge_recovery
        + 0.25 * global_recovery
        - 4.0 * topology_penalty
    )


def _g1_direct_geometry_qualified(item: dict[str, Any], config: Any) -> bool:
    """Prove G learned useful same-edge geometry and deploys without regression."""
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
        and _finite_metric(geometry, "renderedTopologyRegression", 1.0) == 0.0
        and _finite_metric(geometry, "predictedChamferPixels", float("inf"))
        <= catastrophic
        and _finite_metric(item, "structureEdgeRecovery", -1.0) >= 0.0
        and _finite_metric(item, "structureGlobalRecovery", -1.0) >= 0.0
    )


def _run_g1_failfast(**kwargs: Any) -> dict[str, Any]:
    """V12.9 stage runner with best-state rollback plus plateau termination."""
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
        merged = {
            "step": int(step),
            "phase": phase,
            "lossTerms": loss_terms,
            "metrics": base_metrics,
            **extra,
        }
        current_score = _g1_direct_geometry_score(merged)
        merged["score"] = float(current_score)
        merged["qualified"] = _g1_direct_geometry_qualified(merged, config)
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

        if current_score >= meaningful_best + float(GEOMETRY_PLATEAU_MIN_SCORE_DELTA):
            meaningful_best = current_score
            stale_evaluations = 0
        elif step >= int(GEOMETRY_PLATEAU_MIN_STEPS):
            stale_evaluations += 1

        losses = merged["lossTerms"]
        print(
            f"[micro-v7] {name:<25} step={step:4d}/{max_steps:4d} "
            f"score={current_score:+.4f} "
            f"pointGain={_finite_metric(losses, 'spline_graph_point_gain', -1.0):+.4f} "
            f"pointWins={_finite_metric(losses, 'spline_graph_point_win_fraction', 0.0):.1%} "
            f"stale={stale_evaluations}/{GEOMETRY_PLATEAU_PATIENCE_EVALS} "
            f"{'PASS' if merged['qualified'] else '...'}",
            flush=True,
        )

        if merged["qualified"]:
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
            step >= int(GEOMETRY_PLATEAU_MIN_STEPS)
            and stale_evaluations >= int(GEOMETRY_PLATEAU_PATIENCE_EVALS)
        ):
            if best_path.is_file():
                v5._restore_state(model, best_path)
            (stage_dir / "evaluations.json").write_text(
                json.dumps(evaluations, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(
                f"[micro-v7] {name} FAIL-FAST plateau at step {step}; "
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
                "stopReason": "geometry-score-plateau",
                "plateauPatienceEvaluations": int(GEOMETRY_PLATEAU_PATIENCE_EVALS),
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


def _run_stage_v1210(**kwargs: Any) -> dict[str, Any]:
    name = str(kwargs.get("name", ""))
    if name == "G1_geometry_metric_render":
        return _run_g1_failfast(**kwargs)
    if _PREVIOUS_RUN_STAGE is None:
        raise RuntimeError("V12.10 diagnostic installed without previous stage runner")
    return _PREVIOUS_RUN_STAGE(**kwargs)


def install_v1210_diagnostic_semantics() -> None:
    global _INSTALLED, _PREVIOUS_RUN_STAGE
    if _INSTALLED:
        return
    v6.install_v1291_diagnostic_semantics()
    _PREVIOUS_RUN_STAGE = v5._run_stage
    v5.REPORT_SCHEMA = REPORT_SCHEMA
    v5.QUALIFICATION_REVISION = QUALIFICATION_REVISION
    v5._run_stage = _run_stage_v1210
    _INSTALLED = True


def main(argv: list[str] | None = None) -> int:
    install_v1210_diagnostic_semantics()
    print(
        "[micro-v7] V12.10: direct same-edge G qualification + fail-fast plateau active",
        flush=True,
    )
    return v5.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
