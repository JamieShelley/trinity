"""Bounded mutation/population service for local-boundary genomes."""
from __future__ import annotations

import random

from .domain import GENOME_BOUNDS, GENOME_NAMES, Genome


class PopulationGenerator:
    """Generate deterministic bounded mutation populations around an elite parent."""

    def __init__(self, *, seed: int, population: int) -> None:
        """Store deterministic population policy parameters.

        Purpose:
            Make search breadth and randomness explicit controller dependencies.
        Called by:
            EvolutionaryRecoveryController.__init__().
        Calls:
            No project functions.
        """
        self.seed = int(seed)
        self.population = max(2, int(population))

    def _mutate(self, parent: Genome, rng: random.Random, sigma: float) -> Genome:
        """Mutate one genome within declared production-safe bounds.

        Purpose:
            Produce one neighbouring candidate without changing model topology.
        Called by:
            PopulationGenerator.create().
        Calls:
            Genome.bounded().
        """
        values: dict[str, float] = {}
        for name in GENOME_NAMES:
            lo, hi = GENOME_BOUNDS[name]
            span = hi - lo
            value = float(getattr(parent, name)) + rng.gauss(0.0, sigma * span)
            values[name] = min(hi, max(lo, value))
        return Genome(**values).bounded()

    def create(
        self,
        parent: Genome,
        *,
        generation: int,
        recovery_count: int,
    ) -> list[Genome]:
        """Create an elitist population with the parent as candidate zero.

        Purpose:
            Generate one deterministic bounded generation for evaluation.
        Called by:
            EvolutionaryRecoveryController.search().
        Calls:
            PopulationGenerator._mutate(), Genome.bounded().
        """
        rng = random.Random(self.seed + 1009 * generation + int(recovery_count) * 65537)
        sigma = min(0.24, 0.075 * (1.0 + generation * 0.45 + int(recovery_count) * 0.55))
        candidates = [parent.bounded()]
        while len(candidates) < self.population:
            candidates.append(self._mutate(parent, rng, sigma))
        return candidates
