# NSAMDR Python OOP/readability architecture

Baseline: `a8c8bbcc8b2f97d140c58b0bcdea0d12191c507d`.

The refactor changes source ownership and readability, not the deployable model
contract. PyTorch `nn.Module` inheritance remains where required by PyTorch;
application/training orchestration uses **composition over inheritance**.

## Hard source rules

1. Every implementation function/method documents:
   - `Purpose:` what it does;
   - `Called by:` where control comes from;
   - `Calls:` important callees/resources.
2. If local `A()` directly calls local `B()`, `B()` is declared above `A()`.
   The same rule applies to `self.B()`.
3. Public workflow methods therefore appear near the bottom of files/classes,
   after the primitives they compose.
4. Application behavioural classes do not inherit from one another.
5. Resource ownership uses RAII-style Python context managers where lifetime and
   cleanup matter.
6. Compatibility facades stay thin and contain no orchestration logic.

## Stage 1 — evolutionary recovery

```text
EvolutionaryRecoveryController
├── GenomeRepository
├── RavenSampleProvider
├── PopulationGenerator
├── CandidateEvaluator
│   ├── StructuralObjective
│   ├── StructuralFitness
│   ├── EvolutionGenomeSession       # RAII
│   └── ModelResource                # RAII
└── FailureDetector
```

Implementation: `tools/nsamdr/neural/v9/evolution/`.

## Stage 2 — experiment/training application

The previous `train_nsamdr_v9_preview_experiment.py` was roughly a thousand
lines and owned CLI parsing, config resolution, experiment allocation, manifest
status, structural gates, checkpoint-state promotion, evolutionary retry,
trainer invocation, and finalisation.

It is now a thin compatibility CLI. The implementation is composed as:

```text
TrainingApplication
├── ConfigResolver
├── ResultWriter
├── ExperimentService
│   └── ExperimentRunSession          # RAII lifecycle owner
├── EvolutionaryRecoveryController
└── PassDrivenPipeline
    ├── TrainingBackend
    ├── QualificationGates
    ├── StagePlan
    ├── TrainingStateService
    └── ExperimentService
```

Files:

```text
v9/application/
├── domain.py
├── clock.py
├── configuration.py
├── results.py
├── gates.py
├── training_state.py
├── backend.py
├── experiment.py
├── pipeline.py
├── cli.py
└── runner.py
```

### RAII lifecycle change

`ExperimentRunSession` owns the `running` manifest state. If discovery or
training raises, `__exit__` marks the experiment `interrupted-or-failed`.
Previously the large script had exception handling only around the later
training call; an exception during evolutionary discovery could leave the
manifest looking `running`.

This change affects lifecycle correctness only; it does not change the model,
losses, stage gates, work budget, or checkpoint schema.

## Executable readability contract

Run:

```powershell
artifacts\nsamdr\python-env\Scripts\python.exe `
  tools\nsamdr\neural\nsamdr_readability_contract.py
```

It now covers both:

```text
v9/evolution/
v9/application/
```

The contract rejects:
- missing `Purpose:` / `Called by:` / `Calls:`;
- local callees declared below their callers;
- `self`/`cls` callees declared below their callers;
- application-level inheritance outside allowed framework/value bases.

## Next slice

Stage 3 should decompose the large `v9/training.py` runtime itself, beginning
with resource/runtime ownership (device/memory/data-loader/forward-hook
sessions), then phase state/checkpoint selection, then validation/metrics.
Numerical loss/model code should move only after orchestration ownership is
clear.
