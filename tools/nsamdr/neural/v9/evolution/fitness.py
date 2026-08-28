"""Structural training objective and validation fitness for evolutionary recovery."""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch.nn import functional as F

from .tensor_math import align_polarity, central_difference, weighted_mean


class StructuralObjective:
    """Differentiable short-horizon objective for candidate capacity proof training."""

    def evaluate(
        self,
        geometry: Mapping[str, torch.Tensor],
        sample: Mapping[str, torch.Tensor],
        max_distance: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Calculate the bounded real-Raven microproof loss.

        Purpose:
            Train only enough structural capacity to compare bounded genomes cheaply.
        Called by:
            CandidateEvaluator._train_candidate().
        Calls:
            align_polarity(), central_difference(), weighted_mean().
        """
        predicted = geometry["primitive_phi_pixels"].float()
        target = sample["target_sdf"].float() * float(max_distance)
        source = sample["source_sdf"].float() * float(max_distance)
        if target.shape[-2:] != predicted.shape[-2:]:
            target = F.interpolate(target, size=predicted.shape[-2:], mode="bilinear", align_corners=False)
        if source.shape[-2:] != predicted.shape[-2:]:
            source = F.interpolate(source, size=predicted.shape[-2:], mode="bilinear", align_corners=False)

        band = 0.15 + 1.85 * torch.exp(-target.abs() / 4.0)
        predicted = align_polarity(predicted, target, band)
        source = align_polarity(source, target, band)

        surface = weighted_mean(
            F.smooth_l1_loss(predicted, target, beta=0.20, reduction="none"), band
        )
        pgx, pgy = central_difference(predicted)
        tgx, tgy = central_difference(target)
        gradient = weighted_mean(
            F.smooth_l1_loss(pgx, tgx, beta=0.12, reduction="none")
            + F.smooth_l1_loss(pgy, tgy, beta=0.12, reduction="none"),
            band,
        )
        inside = (target < 0.0).float()
        sign = weighted_mean(
            F.binary_cross_entropy_with_logits(-predicted / 1.5, inside, reduction="none"),
            band,
        )
        source_error = (source - target).abs().detach()
        predicted_error = (predicted - target).abs()
        regret = weighted_mean(F.relu(predicted_error - source_error - 0.05), band)
        correction = weighted_mean((predicted - source).square(), band).sqrt()

        total = surface + 0.35 * gradient + 0.20 * sign + 0.80 * regret + 0.015 * correction
        metrics = {
            "surface": float(surface.detach().item()),
            "gradient": float(gradient.detach().item()),
            "sign": float(sign.detach().item()),
            "regret": float(regret.detach().item()),
            "correctionRms": float(correction.detach().item()),
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
    ) -> dict[str, float | bool]:
        """Measure hard checks and scalar fitness for a trained candidate.

        Purpose:
            Rank candidates while rejecting topology/sign/numerical regressions.
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
        )
        passed = bool(
            finite
            and learning_gain >= 0.01
            and predicted_mae <= source_mae * 1.08
            and sign_regression <= 0.025
        )
        return {
            "finite": finite,
            "source_mae": source_mae,
            "predicted_mae": predicted_mae,
            "gain": gain,
            "sign_regression": sign_regression,
            "gradient_mae": grad_mae,
            "correction_rms": correction_rms,
            "fitness": fitness,
            "passed": passed,
        }
