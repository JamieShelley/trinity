"""Bounded evolutionary recovery for the NSAMDR production local-boundary supernet.

This module is training-only. It never participates in inference and it never
creates an alternate Raven network. The production model contains one fixed
superset representation (three local analytic branches + compact CSG). Evolution
changes only a small, bounded genome that scales authorities inside that same
state-dict-compatible production module.

The controller has two jobs:

1. run a very short real-Raven capacity microproof before expensive training;
2. if the first structural promotion gate fails, mutate the bounded genome,
   restart only the structural stage from a clean deterministic seed, and
   continue automatically when the same production gate passes.

Software/contract failures are never treated as architecture evidence. They stop
immediately with diagnostics. Numerical failures are classified separately and
also fail closed; this controller does not hide them by mutating geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping

import torch
from torch.nn import functional as F


EVOLUTION_SCHEMA = "NSAMDR_LOCAL_BOUNDARY_EVOLUTION_V1"
GENOME_NAMES = (
    "feature_gain",
    "evidence_gain",
    "distance_scale",
    "curvature_scale",
    "ribbon_scale",
    "extra_branch_gain",
    "csg_logit_scale",
    "correction_scale",
)
GENOME_BOUNDS: dict[str, tuple[float, float]] = {
    "feature_gain": (0.65, 1.35),
    "evidence_gain": (0.65, 1.35),
    "distance_scale": (0.45, 1.70),
    "curvature_scale": (0.45, 1.70),
    "ribbon_scale": (0.45, 1.70),
    "extra_branch_gain": (0.35, 1.75),
    "csg_logit_scale": (0.50, 1.75),
    "correction_scale": (0.45, 1.45),
}
DEFAULT_GENOME: dict[str, float] = {name: 1.0 for name in GENOME_NAMES}


class FailureKind(str, Enum):
    SOFTWARE = "software"
    NUMERICAL = "numerical"
    LEARNING = "learning"
    REPRESENTATION = "representation"


@dataclass(frozen=True)
class Genome:
    feature_gain: float = 1.0
    evidence_gain: float = 1.0
    distance_scale: float = 1.0
    curvature_scale: float = 1.0
    ribbon_scale: float = 1.0
    extra_branch_gain: float = 1.0
    csg_logit_scale: float = 1.0
    correction_scale: float = 1.0

    def bounded(self) -> "Genome":
        values: dict[str, float] = {}
        for name in GENOME_NAMES:
            lo, hi = GENOME_BOUNDS[name]
            values[name] = min(hi, max(lo, float(getattr(self, name))))
        return Genome(**values)

    def vector(self) -> list[float]:
        bounded = self.bounded()
        return [float(getattr(bounded, name)) for name in GENOME_NAMES]

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, float]:
        bounded = self.bounded()
        return {name: float(getattr(bounded, name)) for name in GENOME_NAMES}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "Genome":
        if not value:
            return cls()
        return cls(**{
            name: float(value.get(name, 1.0))
            for name in GENOME_NAMES
        }).bounded()


@dataclass
class CandidateResult:
    generation: int
    index: int
    genome: Genome
    finite: bool
    train_loss_before: float
    train_loss_after: float
    source_band_mae: float
    predicted_band_mae: float
    relative_gain: float
    sign_regression: float
    gradient_mae: float
    correction_rms: float
    fitness: float
    elapsed_seconds: float
    passed_microproof: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["genome"] = self.genome.to_dict()
        payload["genomeSha256"] = self.genome.fingerprint()
        return payload


@dataclass
class EvolutionResult:
    parent: Genome
    winner: Genome
    candidates: list[CandidateResult]
    passed: bool
    generation: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EVOLUTION_SCHEMA,
            "parent": self.parent.to_dict(),
            "parentSha256": self.parent.fingerprint(),
            "winner": self.winner.to_dict(),
            "winnerSha256": self.winner.fingerprint(),
            "passed": bool(self.passed),
            "generation": int(self.generation),
            "reason": self.reason,
            "candidates": [item.to_dict() for item in self.candidates],
        }


def classify_failure(
    *,
    error: BaseException | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> FailureKind:
    """Classify a failure before deciding whether evolution is permitted."""
    if error is not None:
        text = f"{type(error).__name__}: {error}".lower()
        numerical_markers = (
            "out of memory", "cuda oom", "nan", "nonfinite", "non-finite",
            "overflow", "underflow", "inf loss", "gradient explosion",
        )
        if any(marker in text for marker in numerical_markers):
            return FailureKind.NUMERICAL
        return FailureKind.SOFTWARE

    data = metrics or {}
    try:
        topology_regression = float(data.get("sdf_stageb_topology_regression_fraction", 0.0))
        missing = float(data.get("sdf_predicted_missing_contour_fraction", 0.0))
        source_missing = float(data.get("sdf_source_missing_contour_fraction", 0.0))
        gain = float(data.get("sdf_zero_contour_relative_gain_mean", -1.0))
        chamfer = float(data.get("sdf_zero_contour_chamfer_pixels", math.inf))
    except (TypeError, ValueError):
        return FailureKind.SOFTWARE

    if (
        topology_regression > 0.0
        or missing > source_missing + 1.0e-6
        or gain < 0.0
        or not math.isfinite(chamfer)
    ):
        return FailureKind.REPRESENTATION
    return FailureKind.LEARNING


def _central_difference(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padded_x = F.pad(value.float(), (1, 1, 0, 0), mode="replicate")
    padded_y = F.pad(value.float(), (0, 0, 1, 1), mode="replicate")
    gx = 0.5 * (padded_x[..., 2:] - padded_x[..., :-2])
    gy = 0.5 * (padded_y[..., 2:, :] - padded_y[..., :-2, :])
    return gx, gy


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(1)
    if weight.shape[1] == 1 and value.shape[1] != 1:
        weight = weight.expand(-1, value.shape[1], -1, -1)
    return (value.float() * weight.float()).sum() / weight.float().sum().clamp_min(1.0)


def _align_polarity(predicted: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    positive = _weighted_mean((predicted - target).abs(), weight)
    negative = _weighted_mean((predicted + target).abs(), weight)
    return predicted if float(positive.item()) <= float(negative.item()) else -predicted


def _micro_objective(
    geometry: Mapping[str, torch.Tensor],
    sample: Mapping[str, torch.Tensor],
    max_distance: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    predicted = geometry["primitive_phi_pixels"].float()
    target = sample["target_sdf"].float() * float(max_distance)
    source = sample["source_sdf"].float() * float(max_distance)
    if target.shape[-2:] != predicted.shape[-2:]:
        target = F.interpolate(target, size=predicted.shape[-2:], mode="bilinear", align_corners=False)
    if source.shape[-2:] != predicted.shape[-2:]:
        source = F.interpolate(source, size=predicted.shape[-2:], mode="bilinear", align_corners=False)

    band = 0.15 + 1.85 * torch.exp(-target.abs() / 4.0)
    predicted = _align_polarity(predicted, target, band)
    source = _align_polarity(source, target, band)

    surface = _weighted_mean(F.smooth_l1_loss(predicted, target, beta=0.20, reduction="none"), band)
    pgx, pgy = _central_difference(predicted)
    tgx, tgy = _central_difference(target)
    gradient = _weighted_mean(
        F.smooth_l1_loss(pgx, tgx, beta=0.12, reduction="none")
        + F.smooth_l1_loss(pgy, tgy, beta=0.12, reduction="none"),
        band,
    )
    inside = (target < 0.0).float()
    sign = _weighted_mean(
        F.binary_cross_entropy_with_logits(-predicted / 1.5, inside, reduction="none"),
        band,
    )
    source_error = (source - target).abs().detach()
    predicted_error = (predicted - target).abs()
    regret = _weighted_mean(F.relu(predicted_error - source_error - 0.05), band)
    correction = _weighted_mean((predicted - source).square(), band).sqrt()

    total = surface + 0.35 * gradient + 0.20 * sign + 0.80 * regret + 0.015 * correction
    metrics = {
        "surface": float(surface.detach().item()),
        "gradient": float(gradient.detach().item()),
        "sign": float(sign.detach().item()),
        "regret": float(regret.detach().item()),
        "correctionRms": float(correction.detach().item()),
    }
    return total, metrics


def _batch_to_device(sample: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            tensor = value.unsqueeze(0) if value.ndim in {0, 1, 2, 3} else value
            result[key] = tensor.to(device)
        else:
            result[key] = value
    return result


class EvolutionaryRecoveryController:
    """Small deterministic evolution strategy around the production supernet."""

    def __init__(
        self,
        *,
        repo_root: Path,
        experiment_dir: Path,
        config: Any,
        device: str,
        seed: int | None = None,
        population: int = 4,
        micro_steps: int = 3,
        max_recoveries: int = 2,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.experiment_dir = Path(experiment_dir).resolve()
        self.config = config
        self.device_request = str(device)
        self.seed = int(seed if seed is not None else getattr(config, "seed", 1337))
        self.population = max(2, int(population))
        self.micro_steps = max(1, int(micro_steps))
        self.max_recoveries = max(0, int(max_recoveries))
        self.recovery_count = 0
        self.evolution_dir = self.experiment_dir / "evolution"
        self.evolution_dir.mkdir(parents=True, exist_ok=True)
        self.production_genome_path = (
            self.repo_root / "artifacts/nsamdr/evolution/locked_local_boundary_genome.json"
        )
        self.production_genome_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_genome = self._load_repository_genome() or Genome()

    def _load_repository_genome(self) -> Genome | None:
        path = self.production_genome_path
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("schema") != EVOLUTION_SCHEMA:
            return None
        value = payload.get("genome")
        return Genome.from_mapping(value if isinstance(value, Mapping) else None)

    def _device(self) -> torch.device:
        if self.device_request.startswith("cuda") and torch.cuda.is_available():
            return torch.device(self.device_request)
        return torch.device("cpu")

    def _samples(self) -> tuple[dict[str, Any], dict[str, Any]]:
        from .dataset import PhysicalTileDatasetV9, load_dataset_manifest

        manifest = load_dataset_manifest(self.repo_root, self.config)
        available = {str(record.get("split", "")) for record in manifest.get("crops", [])}
        train_split = "train" if "train" in available else next(iter(available), "train")
        if "validation" in available:
            validation_split = "validation"
        elif "val" in available:
            validation_split = "val"
        else:
            validation_split = train_split
        train = PhysicalTileDatasetV9(
            manifest, self.config, train_split, 1, seed=self.seed + 71001
        )[0]
        validation = PhysicalTileDatasetV9(
            manifest, self.config, validation_split, 1, seed=self.seed + 91001
        )[0]
        return train, validation

    def _mutate(self, parent: Genome, rng: random.Random, sigma: float) -> Genome:
        values: dict[str, float] = {}
        for name in GENOME_NAMES:
            lo, hi = GENOME_BOUNDS[name]
            span = hi - lo
            value = float(getattr(parent, name)) + rng.gauss(0.0, sigma * span)
            values[name] = min(hi, max(lo, value))
        return Genome(**values).bounded()

    def _candidate_genomes(self, parent: Genome, generation: int) -> list[Genome]:
        rng = random.Random(self.seed + 1009 * generation + self.recovery_count * 65537)
        sigma = min(0.24, 0.075 * (1.0 + generation * 0.45 + self.recovery_count * 0.55))
        candidates = [parent.bounded()]
        while len(candidates) < self.population:
            candidates.append(self._mutate(parent, rng, sigma))
        return candidates

    def _evaluate_candidate(
        self,
        genome: Genome,
        *,
        generation: int,
        index: int,
        train_sample: Mapping[str, Any],
        validation_sample: Mapping[str, Any],
    ) -> CandidateResult:
        from .local_boundary_production_contract import set_active_evolution_genome
        from .model import FidelityResidualNetV9

        started = time.perf_counter()
        device = self._device()
        torch.manual_seed(self.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        set_active_evolution_genome(genome.to_dict())

        model = FidelityResidualNetV9(self.config).to(device)
        model.train()
        # The capacity proof deliberately trains only the evolvable structural
        # module. The production encoder is present and supplies its real feature
        # tensor, but no downstream appearance module is involved.
        parameters = [
            p for p in model.geometry_net.production_structure.parameters()
            if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(parameters, lr=2.0e-3, weight_decay=1.0e-4)
        train_batch = _batch_to_device(train_sample, device)
        validation_batch = _batch_to_device(validation_sample, device)
        max_distance = float(self.config.contour_sdf_max_distance_pixels)
        before = math.inf
        after = math.inf

        try:
            for step in range(self.micro_steps):
                optimizer.zero_grad(set_to_none=True)
                geometry = model.geometry_net(train_batch["input"])
                loss, _ = _micro_objective(geometry, train_batch, max_distance)
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("non-finite evolutionary microproof loss")
                if step == 0:
                    before = float(loss.detach().item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 2.0)
                optimizer.step()
                after = float(loss.detach().item())

            model.eval()
            with torch.inference_mode():
                geometry = model.geometry_net(validation_batch["input"])
                predicted = geometry["primitive_phi_pixels"].float()
                target = validation_batch["target_sdf"].float() * max_distance
                source = validation_batch["source_sdf"].float() * max_distance
                if target.shape[-2:] != predicted.shape[-2:]:
                    target = F.interpolate(target, size=predicted.shape[-2:], mode="bilinear", align_corners=False)
                if source.shape[-2:] != predicted.shape[-2:]:
                    source = F.interpolate(source, size=predicted.shape[-2:], mode="bilinear", align_corners=False)
                band = 0.15 + 1.85 * torch.exp(-target.abs() / 4.0)
                predicted = _align_polarity(predicted, target, band)
                source = _align_polarity(source, target, band)
                source_mae = float(_weighted_mean((source - target).abs(), band).item())
                predicted_mae = float(_weighted_mean((predicted - target).abs(), band).item())
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
                pgx, pgy = _central_difference(predicted)
                tgx, tgy = _central_difference(target)
                grad_mae = float(_weighted_mean((pgx - tgx).abs() + (pgy - tgy).abs(), band).item())
                correction_rms = float(_weighted_mean((predicted - source).square(), band).sqrt().item())
                finite = all(math.isfinite(v) for v in (
                    before, after, source_mae, predicted_mae, gain,
                    sign_regression, grad_mae, correction_rms,
                ))
                learning_gain = (before - after) / max(abs(before), 1.0e-6)
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
        except BaseException as exc:
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
                elapsed_seconds=time.perf_counter() - started,
                passed_microproof=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return CandidateResult(
            generation=generation,
            index=index,
            genome=genome,
            finite=finite,
            train_loss_before=before,
            train_loss_after=after,
            source_band_mae=source_mae,
            predicted_band_mae=predicted_mae,
            relative_gain=gain,
            sign_regression=sign_regression,
            gradient_mae=grad_mae,
            correction_rms=correction_rms,
            fitness=fitness,
            elapsed_seconds=time.perf_counter() - started,
            passed_microproof=passed,
        )

    def search(self, *, generation: int = 0, parent: Genome | None = None) -> EvolutionResult:
        parent_genome = (parent or self.current_genome).bounded()
        train_sample, validation_sample = self._samples()
        results: list[CandidateResult] = []
        for index, genome in enumerate(self._candidate_genomes(parent_genome, generation)):
            result = self._evaluate_candidate(
                genome,
                generation=generation,
                index=index,
                train_sample=train_sample,
                validation_sample=validation_sample,
            )
            results.append(result)
            print(
                "[evolution] "
                f"g={generation} c={index} fitness={result.fitness:+.4f} "
                f"gain={result.relative_gain:+.2%} signReg={result.sign_regression:+.3f} "
                f"learn={result.train_loss_before:.4f}->{result.train_loss_after:.4f} "
                f"{'PASS' if result.passed_microproof else 'reject'}",
                flush=True,
            )

        viable = [item for item in results if item.finite]
        winner_result = max(viable, key=lambda item: item.fitness) if viable else results[0]
        passed_results = [item for item in viable if item.passed_microproof]
        if passed_results:
            winner_result = max(passed_results, key=lambda item: item.fitness)
            passed = True
            reason = "at least one production-supernet candidate passed the real-Raven capacity microproof"
        else:
            passed = False
            reason = "no bounded genome passed the real-Raven capacity microproof"

        self.current_genome = winner_result.genome.bounded()
        evolution = EvolutionResult(
            parent=parent_genome,
            winner=self.current_genome,
            candidates=results,
            passed=passed,
            generation=generation,
            reason=reason,
        )
        path = self.evolution_dir / f"generation_{generation:02d}_recovery_{self.recovery_count:02d}.json"
        path.write_text(json.dumps(evolution.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return evolution

    def discover_before_training(self) -> EvolutionResult:
        """Run a bounded real-Raven capacity search before expensive training.

        A failed first generation automatically breeds once more around its best
        survivor. This is the first self-recovery loop and is deliberately capped
        so Quick cannot turn into an unbounded NAS job.
        """
        from .local_boundary_production_contract import set_active_evolution_genome

        result = self.search(generation=0, parent=self.current_genome)
        if not result.passed:
            result = self.search(generation=1, parent=result.winner)
        set_active_evolution_genome(result.winner.to_dict())
        candidate_path = self.evolution_dir / "candidate_genome.json"
        candidate_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return result

    def can_recover(self, metrics: Mapping[str, Any]) -> bool:
        return (
            self.recovery_count < self.max_recoveries
            and classify_failure(metrics=metrics) == FailureKind.REPRESENTATION
        )

    def recover_after_structural_failure(self, metrics: Mapping[str, Any]) -> EvolutionResult:
        if not self.can_recover(metrics):
            raise RuntimeError("evolutionary recovery requested for a non-recoverable failure")
        from .local_boundary_production_contract import set_active_evolution_genome

        self.recovery_count += 1
        result = self.search(
            generation=self.recovery_count,
            parent=self.current_genome,
        )
        set_active_evolution_genome(result.winner.to_dict())
        return result

    def lock_production_genome(
        self,
        *,
        experiment_id: str,
        metrics: Mapping[str, Any],
    ) -> Path:
        """Persist the genome only after the real structural gate passes."""
        payload = {
            "schema": EVOLUTION_SCHEMA,
            "experiment": str(experiment_id),
            "genome": self.current_genome.to_dict(),
            "genomeSha256": self.current_genome.fingerprint(),
            "fitnessSource": "real Raven capacity microproof + production structural gate",
            "structuralMetrics": dict(metrics),
            "trainingOnlyController": True,
            "inferenceAuthority": False,
        }
        self.production_genome_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        local = self.evolution_dir / "locked_genome.json"
        local.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return local
