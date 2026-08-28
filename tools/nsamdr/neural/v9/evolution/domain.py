"""Value objects used by bounded NSAMDR evolutionary recovery.

The module contains data only. Runtime services live in sibling modules and are
composed by :class:`EvolutionaryRecoveryController`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping


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
    """Categorise failures so only representation failures trigger evolution."""

    SOFTWARE = "software"
    NUMERICAL = "numerical"
    LEARNING = "learning"
    REPRESENTATION = "representation"


@dataclass(frozen=True)
class Genome:
    """Bounded production-supernet authority values persisted with checkpoints."""

    feature_gain: float = 1.0
    evidence_gain: float = 1.0
    distance_scale: float = 1.0
    curvature_scale: float = 1.0
    ribbon_scale: float = 1.0
    extra_branch_gain: float = 1.0
    csg_logit_scale: float = 1.0
    correction_scale: float = 1.0

    def bounded(self) -> "Genome":
        """Clamp every genome field to the production-safe range.

        Purpose:
            Produce a valid state-dict-compatible production genome.
        Called by:
            Genome.vector(), Genome.to_dict(), Genome.from_mapping(), population services.
        Calls:
            No project functions.
        """
        values: dict[str, float] = {}
        for name in GENOME_NAMES:
            lo, hi = GENOME_BOUNDS[name]
            values[name] = min(hi, max(lo, float(getattr(self, name))))
        return Genome(**values)

    def vector(self) -> list[float]:
        """Return the bounded genome in the fixed production field order.

        Purpose:
            Convert the value object to the exact persistent-buffer order.
        Called by:
            Model/evolution diagnostics and tests.
        Calls:
            Genome.bounded().
        """
        bounded = self.bounded()
        return [float(getattr(bounded, name)) for name in GENOME_NAMES]

    def to_dict(self) -> dict[str, float]:
        """Serialise the bounded genome by stable field name.

        Purpose:
            Create JSON/model-orchestration input without exposing dataclass internals.
        Called by:
            Genome.fingerprint(), repositories, results, genome sessions.
        Calls:
            Genome.bounded().
        """
        bounded = self.bounded()
        return {name: float(getattr(bounded, name)) for name in GENOME_NAMES}

    def fingerprint(self) -> str:
        """Calculate the stable SHA-256 identity of the bounded genome.

        Purpose:
            Bind evidence and checkpoint metadata to one exact genome.
        Called by:
            Result serialisation and GenomeRepository.lock().
        Calls:
            Genome.to_dict().
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "Genome":
        """Build and clamp a genome from persisted or caller-supplied values.

        Purpose:
            Safely restore evolutionary state from JSON-like mappings.
        Called by:
            GenomeRepository.load().
        Calls:
            Genome.bounded().
        """
        if not value:
            return cls()
        genome = cls(**{name: float(value.get(name, 1.0)) for name in GENOME_NAMES})
        return genome.bounded()


@dataclass
class CandidateResult:
    """Immutable-style evidence record for one candidate microproof evaluation."""

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
        """Serialise candidate evidence for experiment diagnostics.

        Purpose:
            Persist all fitness and failure evidence in a JSON-safe form.
        Called by:
            EvolutionResult.to_dict().
        Calls:
            Genome.to_dict(), Genome.fingerprint().
        """
        payload = asdict(self)
        payload["genome"] = self.genome.to_dict()
        payload["genomeSha256"] = self.genome.fingerprint()
        return payload


@dataclass
class EvolutionResult:
    """Result of one bounded population search generation."""

    parent: Genome
    winner: Genome
    candidates: list[CandidateResult]
    passed: bool
    generation: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise generation evidence including exact parent/winner identities.

        Purpose:
            Produce the canonical generation evidence record.
        Called by:
            GenomeRepository.write_generation(), controller candidate persistence.
        Calls:
            Genome.to_dict(), Genome.fingerprint(), CandidateResult.to_dict().
        """
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
