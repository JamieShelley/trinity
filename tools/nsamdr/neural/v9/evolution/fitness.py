"""Structural training objective and validation fitness for evolutionary recovery."""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from .tensor_math import align_polarity, central_difference, weighted_mean


class StructuralObjective:
    """V12 proposal-space capacity objective for evolutionary recovery."""

    def __init__(self, config: Any | None = None) -> None:
        """Bind the production configuration used by the proposal-space objective.

        Purpose:
            Keep the evolutionary capacity proof aligned with the active B1b spline
            proposal and SDF-polarity contracts.
        Called by:
            CandidateEvaluator.__init__() and tests constructing the objective directly.
        Calls:
            No project functions.
        """
        self.config = config

    def evaluate(
        self,
        geometry: Mapping[str, torch.Tensor],
        sample: Mapping[str, torch.Tensor],
        max_distance: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Train the neural spline proposal, never the detached explicit-refiner result.

        Purpose:
            Measure short-horizon real-Raven capacity in the gradient-bearing neural
            proposal while leaving final refined geometry as qualification authority.
        Called by:
            CandidateEvaluator._train_candidate() and focused capacity-contract tests.
        Calls:
            _same_edge_targets(), _baseline_relative_point_objective(), and the canonical
            global SDF-polarity helper when gauge invariance is enabled.

        V12 deliberately makes final refined geometry parameter-free and detached from
        outer SGD.  The evolutionary capacity proof therefore uses the same authored
        same-edge point/tangent teacher as B1b.  Held-out fitness below still measures
        the final refined SDF, so proposal learning cannot masquerade as qualification.
        """
        from ..edge_constrained_spline_graph import (
            _baseline_relative_point_objective,
            _same_edge_targets,
        )

        required = (
            "spline_graph_control_phi_pixels",
            "source_sdf_prior_pixels",
            "spline_graph_mask_h",
            "spline_graph_mask_v",
            "spline_source_control_point_h_lr",
            "spline_source_control_point_v_lr",
            "spline_proposal_control_point_h_lr",
            "spline_proposal_control_point_v_lr",
            "spline_proposal_control_tangent_h",
            "spline_proposal_control_tangent_v",
        )
        missing = [name for name in required if name not in geometry]
        if missing:
            raise RuntimeError(
                "V12 evolutionary capacity proof requires neural proposal outputs: "
                + ", ".join(missing)
            )

        raw_target = sample["target_sdf"].float() * float(max_distance)
        source_prior = geometry["source_sdf_prior_pixels"].detach().float()
        if bool(getattr(self.config, "sdf_sign_gauge_invariant", True)):
            from .. import losses as canonical_losses
            polarity = canonical_losses._losses_service._sdf_global_polarity(
                source_prior,
                raw_target,
                float(getattr(self.config, "sdf_metric_band_pixels", 6.0)),
            )
            target = raw_target * polarity
        else:
            target = raw_target
        control = geometry["spline_graph_control_phi_pixels"]
        control_scale = float(getattr(self.config, "spline_graph_control_scale", 2))
        control_spacing_hr = 4.0 / max(control_scale, 1.0)
        target_h, target_v, target_tan_h, target_tan_v, valid_h, valid_v = (
            _same_edge_targets(
                target,
                tuple(control.shape[-2:]),
                control_spacing_hr=control_spacing_hr,
                control_origin=2.0,
            )
        )

        proposal_h = geometry["spline_proposal_control_point_h_lr"].float()
        proposal_v = geometry["spline_proposal_control_point_v_lr"].float()
        source_h = geometry["spline_source_control_point_h_lr"].detach().float()
        source_v = geometry["spline_source_control_point_v_lr"].detach().float()
        tangent_h = geometry["spline_proposal_control_tangent_h"].float()
        tangent_v = geometry["spline_proposal_control_tangent_v"].float()
        mask_h = geometry["spline_graph_mask_h"][:, 0].float() * valid_h.float()
        mask_v = geometry["spline_graph_mask_v"][:, 0].float() * valid_v.float()

        point, regret, gain, wins = _baseline_relative_point_objective(
            proposal_h,
            proposal_v,
            source_h,
            source_v,
            target_h,
            target_v,
            mask_h,
            mask_v,
        )
        active_teacher = mask_h.sum() + mask_v.sum()
        active_source = (
            geometry["spline_graph_mask_h"][:, 0].float().sum()
            + geometry["spline_graph_mask_v"][:, 0].float().sum()
        ).clamp_min(1.0)
        denom = active_teacher.clamp_min(1.0)
        dot_h = (tangent_h * target_tan_h).sum(dim=-1).abs().clamp(0.0, 1.0)
        dot_v = (tangent_v * target_tan_v).sum(dim=-1).abs().clamp(0.0, 1.0)
        tangent = (
            ((1.0 - dot_h) * mask_h).sum()
            + ((1.0 - dot_v) * mask_v).sum()
        ) / denom

        point_weight = max(
            float(getattr(self.config, "spline_graph_point_weight", 1.0)), 1.0e-6
        )
        tangent_ratio = float(
            getattr(self.config, "spline_graph_tangent_weight", 1.0)
        ) / point_weight
        total = point + regret + tangent * tangent_ratio
        if not total.requires_grad:
            raise RuntimeError(
                "V12 evolutionary proposal objective is detached from outer SGD"
            )
        metrics = {
            "proposalPoint": float(point.detach().item()),
            "proposalRegret": float(regret.detach().item()),
            "proposalGain": float(gain.detach().item()),
            "proposalWins": float(wins.detach().item()),
            "proposalTangent": float(tangent.detach().item()),
            "teacherCoverage": float((active_teacher / active_source).detach().item()),
        }
        return total, metrics


class StructuralFitness:
    """Non-training evaluator that converts Raven structural evidence into fitness."""

    def measure(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        *,
        train_loss_before: float,
        train_loss_after: float,
        topology_regression_fraction: float = 0.0,
    ) -> dict[str, float | bool]:
        """Measure hard checks and scalar fitness for a trained candidate.

        Purpose:
            Rank candidates while requiring real held-out improvement, measurable
            optimisation descent, and no topology/sign regression.
        Called by:
            CandidateEvaluator._measure_candidate().
        Calls:
            align_polarity(), central_difference(), weighted_mean().
        """
        if target.shape[-2:] != predicted.shape[-2:]:
            target = F.interpolate(target, size=predicted.shape[-2:], mode="bilinear", align_corners=False)
        if source.shape[-2:] != predicted.shape[-2:]:
            source = F.interpolate(source, size=predicted.shape[-2:], mode="bilinear", align_corners=False)

        band = 0.15 + 1.85 * torch.exp(-target.abs() / 4.0)
        predicted = align_polarity(predicted, target, band)
        source = align_polarity(source, target, band)
        source_mae = float(weighted_mean((source - target).abs(), band).item())
        predicted_mae = float(weighted_mean((predicted - target).abs(), band).item())
        gain = (source_mae - predicted_mae) / max(source_mae, 1.0e-6)

        target_inside = target < 0.0
        pred_inside = predicted < 0.0
        source_inside = source < 0.0
        confident = (target.abs() >= 0.75) & (target.abs() <= 6.0)
        denom = float(confident.float().sum().item())
        if denom > 0.0:
            pred_sign = float(((pred_inside != target_inside) & confident).float().sum().item() / denom)
            src_sign = float(((source_inside != target_inside) & confident).float().sum().item() / denom)
        else:
            pred_sign = src_sign = 0.0
        sign_regression = pred_sign - src_sign

        pgx, pgy = central_difference(predicted)
        tgx, tgy = central_difference(target)
        grad_mae = float(weighted_mean((pgx - tgx).abs() + (pgy - tgy).abs(), band).item())
        correction_rms = float(weighted_mean((predicted - source).square(), band).sqrt().item())
        finite = all(math.isfinite(value) for value in (
            train_loss_before,
            train_loss_after,
            source_mae,
            predicted_mae,
            gain,
            sign_regression,
            grad_mae,
            correction_rms,
            topology_regression_fraction,
        ))
        learning_gain = (
            (train_loss_before - train_loss_after) / max(abs(train_loss_before), 1.0e-6)
        )
        fitness = (
            4.5 * gain
            + 1.2 * learning_gain
            - 2.5 * max(sign_regression, 0.0)
            - 0.12 * grad_mae
            - 0.015 * correction_rms
            - 8.0 * max(topology_regression_fraction, 0.0)
        )

        # The microproof is a capacity check, not a tiny-training-speed benchmark.
        # Quick and Full deliberately use different micro-step budgets, so a fixed
        # 1% short-horizon loss-drop threshold made the gate depend on work budget.
        # Require actual descent, then use held-out Raven improvement as the hard
        # evidence that the representation is already better than the source prior.
        passed = bool(
            finite
            and learning_gain > 0.0
            and gain > 0.0
            and sign_regression <= 0.025
            and topology_regression_fraction <= 0.0
        )
        return {
            "finite": finite,
            "source_mae": source_mae,
            "predicted_mae": predicted_mae,
            "gain": gain,
            "sign_regression": sign_regression,
            "gradient_mae": grad_mae,
            "correction_rms": correction_rms,
            "topology_regression_fraction": float(topology_regression_fraction),
            "fitness": fitness,
            "passed": passed,
        }
