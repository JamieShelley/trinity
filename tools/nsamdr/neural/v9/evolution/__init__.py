"""Public composition-oriented evolutionary recovery API."""
from .candidate import CandidateEvaluator
from .controller import EvolutionaryRecoveryController
from .domain import (
    DEFAULT_GENOME,
    EVOLUTION_SCHEMA,
    GENOME_BOUNDS,
    GENOME_NAMES,
    CandidateResult,
    EvolutionResult,
    FailureKind,
    Genome,
)
from .failure import FailureDetector, classify_failure
from .fitness import StructuralFitness, StructuralObjective
from .population import PopulationGenerator
from .repository import GenomeRepository
from .samples import RavenSampleProvider

__all__ = [
    "DEFAULT_GENOME",
    "EVOLUTION_SCHEMA",
    "GENOME_BOUNDS",
    "GENOME_NAMES",
    "CandidateEvaluator",
    "CandidateResult",
    "EvolutionResult",
    "EvolutionaryRecoveryController",
    "FailureDetector",
    "FailureKind",
    "Genome",
    "GenomeRepository",
    "PopulationGenerator",
    "RavenSampleProvider",
    "StructuralFitness",
    "StructuralObjective",
    "classify_failure",
]
