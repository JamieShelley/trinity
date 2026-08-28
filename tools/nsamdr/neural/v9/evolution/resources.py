"""RAII-style resource owners for evolutionary candidate evaluation."""
from __future__ import annotations

from typing import Any

import torch

from .domain import Genome


class EvolutionGenomeSession:
    """Own process-local active-genome replacement and deterministic restoration."""

    def __init__(self, genome: Genome) -> None:
        """Store the requested temporary genome without mutating global state yet.

        Purpose:
            Prepare deterministic context ownership for a candidate genome.
        Called by:
            CandidateEvaluator.evaluate() through a with-statement.
        Calls:
            No project functions.
        """
        self._genome = genome
        self._previous: dict[str, float] | None = None

    def __enter__(self) -> "EvolutionGenomeSession":
        """Install the candidate genome and remember the previous process state.

        Purpose:
            Acquire the global genome resource at lexical scope entry.
        Called by:
            Python context-manager protocol.
        Calls:
            active_evolution_genome(), set_active_evolution_genome().
        """
        from ..local_boundary_production_contract import (
            active_evolution_genome,
            set_active_evolution_genome,
        )
        self._previous = active_evolution_genome()
        set_active_evolution_genome(self._genome.to_dict())
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        """Restore the previous active genome regardless of success or exception.

        Purpose:
            Release process-global candidate state deterministically.
        Called by:
            Python context-manager protocol.
        Calls:
            set_active_evolution_genome().
        """
        from ..local_boundary_production_contract import set_active_evolution_genome
        set_active_evolution_genome(self._previous)
        return False


class ModelResource:
    """Own one candidate model and release GPU/cache resources on scope exit."""

    def __init__(self, model: Any, device: torch.device) -> None:
        """Bind a model to its resource-owning device session.

        Purpose:
            Make model/CUDA cleanup explicit rather than scattered finally blocks.
        Called by:
            CandidateEvaluator.evaluate().
        Calls:
            No project functions.
        """
        self.model = model
        self.device = device

    def __enter__(self) -> Any:
        """Expose the owned model to the candidate evaluation scope.

        Purpose:
            Acquire model lifetime for one candidate evaluation.
        Called by:
            Python context-manager protocol.
        Calls:
            No project functions.
        """
        return self.model

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        """Release model references and CUDA cache deterministically.

        Purpose:
            Prevent candidate populations from accumulating GPU resources.
        Called by:
            Python context-manager protocol.
        Calls:
            torch.cuda.empty_cache() when applicable.
        """
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return False
