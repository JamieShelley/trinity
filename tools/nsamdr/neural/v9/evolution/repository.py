"""Persistence service for evolutionary genomes and evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .domain import EVOLUTION_SCHEMA, EvolutionResult, Genome


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON by temporary file followed by atomic replacement.

    Purpose:
        Prevent interrupted evidence/checkpoint metadata writes from appearing valid.
    Called by:
        GenomeRepository.write_generation(), write_candidate(), lock().
    Calls:
        tempfile.mkstemp(), os.replace().
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(payload), stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class GenomeRepository:
    """Own evolutionary JSON locations and all persistent genome/evidence writes."""

    def __init__(self, repo_root: Path, experiment_dir: Path) -> None:
        """Resolve repository and experiment persistence locations.

        Purpose:
            Centralise path ownership that was previously embedded in the controller.
        Called by:
            EvolutionaryRecoveryController.__init__().
        Calls:
            Path.resolve(), Path.mkdir().
        """
        self.repo_root = Path(repo_root).resolve()
        self.experiment_dir = Path(experiment_dir).resolve()
        self.evolution_dir = self.experiment_dir / "evolution"
        self.evolution_dir.mkdir(parents=True, exist_ok=True)
        self.production_genome_path = (
            self.repo_root / "artifacts/nsamdr/evolution/locked_local_boundary_genome.json"
        )
        self.production_genome_path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Genome | None:
        """Load a previously locked production genome when schema-compatible.

        Purpose:
            Seed new experiments from the last qualified production genome.
        Called by:
            EvolutionaryRecoveryController.__init__().
        Calls:
            Genome.from_mapping().
        """
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

    def write_generation(self, result: EvolutionResult, recovery_count: int) -> Path:
        """Persist one complete population generation atomically.

        Purpose:
            Preserve candidate evidence even when later generations fail.
        Called by:
            EvolutionaryRecoveryController.search().
        Calls:
            EvolutionResult.to_dict(), write_json_atomic().
        """
        path = self.evolution_dir / (
            f"generation_{result.generation:02d}_recovery_{int(recovery_count):02d}.json"
        )
        write_json_atomic(path, result.to_dict())
        return path

    def write_candidate(self, result: EvolutionResult) -> Path:
        """Persist the selected pre-training candidate generation.

        Purpose:
            Record the exact genome that will enter structural training.
        Called by:
            EvolutionaryRecoveryController.discover_before_training().
        Calls:
            EvolutionResult.to_dict(), write_json_atomic().
        """
        path = self.evolution_dir / "candidate_genome.json"
        write_json_atomic(path, result.to_dict())
        return path

    def lock(
        self,
        genome: Genome,
        *,
        experiment_id: str,
        metrics: Mapping[str, Any],
    ) -> Path:
        """Atomically persist a genome after the real structural gate passes.

        Purpose:
            Make the qualified winning genome available to later production runs.
        Called by:
            EvolutionaryRecoveryController.lock_production_genome().
        Calls:
            Genome.to_dict(), Genome.fingerprint(), write_json_atomic().
        """
        payload = {
            "schema": EVOLUTION_SCHEMA,
            "experiment": str(experiment_id),
            "genome": genome.to_dict(),
            "genomeSha256": genome.fingerprint(),
            "fitnessSource": "real Raven capacity microproof + production structural gate",
            "structuralMetrics": dict(metrics),
            "trainingOnlyController": True,
            "inferenceAuthority": False,
        }
        write_json_atomic(self.production_genome_path, payload)
        local = self.evolution_dir / "locked_genome.json"
        write_json_atomic(local, payload)
        return local
