"""Candidate model construction, micro-training, and structural evaluation."""
from __future__ import annotations

import math
import time
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from .domain import CandidateResult, Genome
from .fitness import StructuralFitness, StructuralObjective
from .resources import EvolutionGenomeSession, ModelResource
from .tensor_math import batch_to_device


class CandidateEvaluator:
    """Evaluate one bounded genome using a disposable production model instance."""

    def __init__(
        self,
        *,
        config: Any,
        device: str,
        seed: int,
        micro_steps: int,
        objective: StructuralObjective | None = None,
        fitness: StructuralFitness | None = None,
    ) -> None:
        """Compose candidate training/evaluation dependencies.

        Purpose:
            Keep optimiser/model/resource concerns out of the controller.
        Called by:
            EvolutionaryRecoveryController.__init__().
        Calls:
            No project functions.
        """
        self.config = config
        self.device_request = str(device)
        self.seed = int(seed)
        self.micro_steps = max(1, int(micro_steps))
        self.objective = objective or StructuralObjective()
        self.fitness = fitness or StructuralFitness()

    def _device(self) -> torch.device:
        """Resolve the candidate device without changing production inference policy.

        Purpose:
            Use CUDA when requested/available and otherwise provide deterministic CPU fallback.
        Called by:
            CandidateEvaluator.evaluate().
        Calls:
            torch.cuda.is_available().
        """
        if self.device_request.startswith("cuda") and torch.cuda.is_available():
            return torch.device(self.device_request)
        return torch.device("cpu")

    def _seed_device(self, device: torch.device) -> None:
        """Reset candidate RNG state so genome comparisons start from equal conditions.

        Purpose:
            Make candidate fitness differences attributable to genome/training, not RNG drift.
        Called by:
            CandidateEvaluator.evaluate().
        Calls:
            torch.manual_seed(), torch.cuda.manual_seed_all().
        """
        torch.manual_seed(self.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)

    def _build_model(self, device: torch.device) -> Any:
        """Construct one complete production model containing the active genome buffer.

        Purpose:
            Create the disposable candidate using the real production topology.
        Called by:
            CandidateEvaluator.evaluate().
        Calls:
            FidelityResidualNetV9().
        """
        from ..model import FidelityResidualNetV9
        return FidelityResidualNetV9(self.config).to(device)

    def _train_candidate(
        self,
        model: Any,
        train_batch: Mapping[str, torch.Tensor],
        max_distance: float,
    ) -> tuple[float, float]:
        """Run the tiny structural-only optimisation budget for one candidate.

        Purpose:
            Test whether the candidate representation can learn the real Raven pair quickly.
        Called by:
            CandidateEvaluator.evaluate().
        Calls:
            StructuralObjective.evaluate(), AdamW.step(), clip_grad_norm_().
        """
        model.train()
        parameters = [
            parameter
            for parameter in model.geometry_net.production_structure.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(parameters, lr=2.0e-3, weight_decay=1.0e-4)
        before = math.inf
        after = math.inf
        for step in range(self.micro_steps):
            optimizer.zero_grad(set_to_none=True)
            geometry = model.geometry_net(train_batch["input"])
            loss, _ = self.objective.evaluate(geometry, train_batch, max_distance)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite evolutionary microproof loss")
            if step == 0:
                before = float(loss.detach().item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 2.0)
            optimizer.step()
            after = float(loss.detach().item())
        return before, after

    def _measure_permanent_topology(
        self,
        model: Any,
        max_distance: float,
    ) -> float:
        """Measure regression on the same permanent topology proof used by B1/B2.

        Purpose:
            Reject genomes that improve one Raven crop while damaging connected-component
            or hole topology on the permanent production structural proof family.
        Called by:
            CandidateEvaluator._measure_candidate().
        Calls:
            SyntheticGeometryValidationDataset(), batch_to_device(),
            sdf_topology_mismatch(), model.geometry_net().
        """
        from ..dataset import SyntheticGeometryValidationDataset
        from ..geometry_metrics import sdf_topology_mismatch

        proof_count = max(29, int(getattr(self.config, "sdf_synthetic_validation_tiles", 29)))
        proof_seed = int(getattr(self.config, "seed", self.seed)) + 9_911
        dataset = SyntheticGeometryValidationDataset(
            self.config, proof_count, seed=proof_seed
        )
        device = next(model.parameters()).device
        regressions = 0
        model.eval()
        with torch.inference_mode():
            for case_index in range(proof_count):
                batch = batch_to_device(dict(dataset[case_index]), device)
                geometry = model.geometry_net(batch["input"])
                predicted = (
                    geometry["primitive_phi_pixels"][0, 0]
                    .detach().float().cpu().numpy()
                )
                source = (
                    geometry["source_sdf_prior_pixels"][0, 0]
                    .detach().float().cpu().numpy()
                )
                target = (
                    batch["target_sdf"][0, 0].detach().float().cpu().numpy()
                    * float(max_distance)
                )
                source_topology = float(sdf_topology_mismatch(source, target))
                predicted_topology = float(sdf_topology_mismatch(predicted, target))
                regressions += int(predicted_topology > source_topology)
        return float(regressions) / float(max(proof_count, 1))

    def _measure_candidate(
        self,
        model: Any,
        validation_batch: Mapping[str, torch.Tensor],
        max_distance: float,
        *,
        train_loss_before: float,
        train_loss_after: float,
    ) -> dict[str, float | bool]:
        """Run validation forward and calculate hard fitness evidence.

        Purpose:
            Decide candidate viability without downstream appearance modules influencing fitness.
        Called by:
            CandidateEvaluator.evaluate().
        Calls:
            CandidateEvaluator._measure_permanent_topology(), StructuralFitness.measure().
        """
        model.eval()
        topology_regression_fraction = self._measure_permanent_topology(model, max_distance)
        with torch.inference_mode():
            geometry = model.geometry_net(validation_batch["input"])
            predicted = geometry["primitive_phi_pixels"].float()
            target = validation_batch["target_sdf"].float() * max_distance
            source = validation_batch["source_sdf"].float() * max_distance
            return self.fitness.measure(
                predicted,
                target,
                source,
                train_loss_before=train_loss_before,
                train_loss_after=train_loss_after,
                topology_regression_fraction=topology_regression_fraction,
            )

    def _error_result(
        self,
        genome: Genome,
        *,
        generation: int,
        index: int,
        started: float,
        before: float,
        after: float,
        error: BaseException,
    ) -> CandidateResult:
        """Convert an evaluation exception into fail-closed candidate evidence.

        Purpose:
            Preserve population progress without converting a failed candidate into a winner.
        Called by:
            CandidateEvaluator.evaluate().
        Calls:
            No project functions.
        """
        return CandidateResult(
            generation=generation,
            index=index,
            genome=genome,
            finite=False,
            train_loss_before=float(before),
            train_loss_after=float(after),
            source_band_mae=math.inf,
            predicted_band_mae=math.inf,
            relative_gain=-math.inf,
            sign_regression=math.inf,
            gradient_mae=math.inf,
            correction_rms=math.inf,
            fitness=-math.inf,
            topology_regression_fraction=math.inf,
            elapsed_seconds=time.perf_counter() - started,
            passed_microproof=False,
            error=f"{type(error).__name__}: {error}",
        )

    def evaluate(
        self,
        genome: Genome,
        *,
        generation: int,
        index: int,
        train_sample: Mapping[str, Any],
        validation_sample: Mapping[str, Any],
    ) -> CandidateResult:
        """Evaluate one genome from resource acquisition through final evidence.

        Purpose:
            Provide the population controller with one fully isolated candidate result.
        Called by:
            EvolutionaryRecoveryController.search().
        Calls:
            CandidateEvaluator._device(), _seed_device(), _build_model(),
            _train_candidate(), _measure_candidate(), _error_result(),
            batch_to_device(), EvolutionGenomeSession, ModelResource.
        """
        started = time.perf_counter()
        device = self._device()
        self._seed_device(device)
        before = math.inf
        after = math.inf
        try:
            with EvolutionGenomeSession(genome):
                model_instance = self._build_model(device)
                with ModelResource(model_instance, device) as model:
                    train_batch = batch_to_device(dict(train_sample), device)
                    validation_batch = batch_to_device(dict(validation_sample), device)
                    max_distance = float(self.config.contour_sdf_max_distance_pixels)
                    before, after = self._train_candidate(model, train_batch, max_distance)
                    evidence = self._measure_candidate(
                        model,
                        validation_batch,
                        max_distance,
                        train_loss_before=before,
                        train_loss_after=after,
                    )
        except BaseException as exc:
            return self._error_result(
                genome,
                generation=generation,
                index=index,
                started=started,
                before=before,
                after=after,
                error=exc,
            )

        return CandidateResult(
            generation=generation,
            index=index,
            genome=genome,
            finite=bool(evidence["finite"]),
            train_loss_before=before,
            train_loss_after=after,
            source_band_mae=float(evidence["source_mae"]),
            predicted_band_mae=float(evidence["predicted_mae"]),
            relative_gain=float(evidence["gain"]),
            sign_regression=float(evidence["sign_regression"]),
            gradient_mae=float(evidence["gradient_mae"]),
            correction_rms=float(evidence["correction_rms"]),
            fitness=float(evidence["fitness"]),
            topology_regression_fraction=float(evidence["topology_regression_fraction"]),
            elapsed_seconds=time.perf_counter() - started,
            passed_microproof=bool(evidence["passed"]),
        )
