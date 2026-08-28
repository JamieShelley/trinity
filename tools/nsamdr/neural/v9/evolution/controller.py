"""High-level bounded evolutionary recovery controller composed from small services."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .candidate import CandidateEvaluator
from .domain import CandidateResult, EvolutionResult, FailureKind, Genome
from .failure import FailureDetector
from .population import PopulationGenerator
from .repository import GenomeRepository
from .samples import RavenSampleProvider


class EvolutionaryRecoveryController:
    """Coordinate bounded discovery/recovery without owning low-level responsibilities."""

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
        repository: GenomeRepository | None = None,
        samples: RavenSampleProvider | None = None,
        population_generator: PopulationGenerator | None = None,
        evaluator: CandidateEvaluator | None = None,
        failure_detector: FailureDetector | None = None,
    ) -> None:
        """Compose all controller collaborators and load the current qualified genome.

        Purpose:
            Preserve the original public constructor while enforcing composition internally.
        Called by:
            NSAMDR Raven Quick/full orchestration.
        Calls:
            GenomeRepository(), RavenSampleProvider(), PopulationGenerator(),
            CandidateEvaluator(), FailureDetector(), GenomeRepository.load().
        """
        resolved_seed = int(seed if seed is not None else getattr(config, "seed", 1337))
        self.config = config
        self.max_recoveries = max(0, int(max_recoveries))
        self.recovery_count = 0
        self.repository = repository or GenomeRepository(repo_root, experiment_dir)
        self.samples = samples or RavenSampleProvider(repo_root, config, resolved_seed)
        self.population_generator = population_generator or PopulationGenerator(
            seed=resolved_seed,
            population=population,
        )
        self.evaluator = evaluator or CandidateEvaluator(
            config=config,
            device=device,
            seed=resolved_seed,
            micro_steps=micro_steps,
        )
        self.failure_detector = failure_detector or FailureDetector()
        self.current_genome = self.repository.load() or Genome()

    def _print_result(self, result: CandidateResult) -> None:
        """Emit one compact line of candidate evidence for GUI/CLI progress streams.

        Purpose:
            Keep presentation out of the evaluation service while preserving existing logs.
        Called by:
            EvolutionaryRecoveryController.search().
        Calls:
            print().
        """
        print(
            "[evolution] "
            f"g={result.generation} c={result.index} fitness={result.fitness:+.4f} "
            f"gain={result.relative_gain:+.2%} signReg={result.sign_regression:+.3f} "
            f"learn={result.train_loss_before:.4f}->{result.train_loss_after:.4f} "
            f"{'PASS' if result.passed_microproof else 'reject'}",
            flush=True,
        )

    def _select_result(
        self,
        parent: Genome,
        generation: int,
        results: list[CandidateResult],
    ) -> EvolutionResult:
        """Select the highest-fitness viable/pass candidate with fail-closed semantics.

        Purpose:
            Apply elitist population selection independently from candidate execution.
        Called by:
            EvolutionaryRecoveryController.search().
        Calls:
            Genome.bounded().
        """
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
        return EvolutionResult(
            parent=parent,
            winner=winner_result.genome.bounded(),
            candidates=results,
            passed=passed,
            generation=generation,
            reason=reason,
        )

    def search(self, *, generation: int = 0, parent: Genome | None = None) -> EvolutionResult:
        """Evaluate one bounded population and persist the complete generation evidence.

        Purpose:
            Execute one self-contained evolutionary generation.
        Called by:
            discover_before_training(), recover_after_structural_failure().
        Calls:
            RavenSampleProvider.load_pair(), PopulationGenerator.create(),
            CandidateEvaluator.evaluate(), _print_result(), _select_result(),
            GenomeRepository.write_generation().
        """
        parent_genome = (parent or self.current_genome).bounded()
        train_sample, validation_sample = self.samples.load_pair()
        results: list[CandidateResult] = []
        for index, genome in enumerate(self.population_generator.create(
            parent_genome,
            generation=generation,
            recovery_count=self.recovery_count,
        )):
            result = self.evaluator.evaluate(
                genome,
                generation=generation,
                index=index,
                train_sample=train_sample,
                validation_sample=validation_sample,
            )
            results.append(result)
            self._print_result(result)
        evolution = self._select_result(parent_genome, generation, results)
        self.current_genome = evolution.winner
        self.repository.write_generation(evolution, self.recovery_count)
        return evolution

    def discover_before_training(self) -> EvolutionResult:
        """Run at most two bounded generations before expensive production training.

        Purpose:
            Reject or improve an unsuitable representation before a long Raven run.
        Called by:
            Raven Quick/full training orchestration before structural training.
        Calls:
            search(), GenomeRepository.write_candidate(), set_active_evolution_genome().
        """
        from ..local_boundary_production_contract import set_active_evolution_genome

        result = self.search(generation=0, parent=self.current_genome)
        if not result.passed:
            result = self.search(generation=1, parent=result.winner)
        set_active_evolution_genome(result.winner.to_dict())
        self.repository.write_candidate(result)
        return result

    def can_recover(self, metrics: Mapping[str, Any]) -> bool:
        """Report whether one structural failure is eligible for bounded evolution.

        Purpose:
            Ensure only representation failures consume the recovery budget.
        Called by:
            recover_after_structural_failure() and training orchestration.
        Calls:
            FailureDetector.classify().
        """
        return (
            self.recovery_count < self.max_recoveries
            and self.failure_detector.classify(metrics=metrics) == FailureKind.REPRESENTATION
        )

    def recover_after_structural_failure(self, metrics: Mapping[str, Any]) -> EvolutionResult:
        """Search one new generation after an eligible production structural-gate failure.

        Purpose:
            Self-adjust bounded representation authorities and return control to training.
        Called by:
            Production structural promotion failure handling.
        Calls:
            can_recover(), search(), set_active_evolution_genome().
        """
        if not self.can_recover(metrics):
            raise RuntimeError("evolutionary recovery requested for a non-recoverable failure")
        from ..local_boundary_production_contract import set_active_evolution_genome

        self.recovery_count += 1
        result = self.search(generation=self.recovery_count, parent=self.current_genome)
        set_active_evolution_genome(result.winner.to_dict())
        return result

    def lock_production_genome(
        self,
        *,
        experiment_id: str,
        metrics: Mapping[str, Any],
    ) -> Path:
        """Persist the current genome only after the real production gate passes.

        Purpose:
            Promote the winning bounded genome to reusable qualified production state.
        Called by:
            Structural promotion success handling.
        Calls:
            GenomeRepository.lock().
        """
        return self.repository.lock(
            self.current_genome,
            experiment_id=experiment_id,
            metrics=metrics,
        )
