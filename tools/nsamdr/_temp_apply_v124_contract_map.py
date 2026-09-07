from pathlib import Path

contract_path = Path('tools/nsamdr/neural/v9/local_boundary_production_contract.py')
text = contract_path.read_text(encoding='utf-8')
old = '''        production = dict(contract.get("productionComponents") or {})
        production["structural representation"] = "geometry_net.production_structure"
        contract["productionComponents"] = production
'''
new = '''        production = dict(contract.get("productionComponents") or {})
        production["structural representation"] = "geometry_net.production_structure"
        production["explicit geometry refiner"] = (
            "geometry_net.production_structure.geometry_refiner"
        )
        contract["productionComponents"] = production
'''
if old not in text:
    raise SystemExit('V12.4 architecture-contract anchor not found')
text = text.replace(old, new, 1)
contract_path.write_text(text, encoding='utf-8')

fitness_path = Path('tools/nsamdr/neural/v9/evolution/fitness.py')
fitness = fitness_path.read_text(encoding='utf-8')
old_init = '''    def __init__(self, config: Any | None = None) -> None:
        self.config = config
'''
new_init = '''    def __init__(self, config: Any | None = None) -> None:
        """Bind the production configuration used by the proposal-space objective.

        Purpose:
            Keep the evolutionary capacity proof aligned with the active B1b spline
            proposal and SDF-polarity contracts.
        Called by:
            CandidateEvaluator.__init__() and tests constructing the objective directly.
        Calls:
            No project functions.
        """
        self.config = config
'''
if old_init not in fitness:
    raise SystemExit('V12.4 StructuralObjective.__init__ anchor not found')
fitness = fitness.replace(old_init, new_init, 1)
old_eval = '''        """Train the neural spline proposal, never the detached explicit-refiner result.

        V12 deliberately makes final refined geometry parameter-free and detached from
        outer SGD.  The evolutionary capacity proof therefore uses the same authored
        same-edge point/tangent teacher as B1b.  Held-out fitness below still measures
        the final refined SDF, so proposal learning cannot masquerade as qualification.
        """
'''
new_eval = '''        """Train the neural spline proposal, never the detached explicit-refiner result.

        Purpose:
            Measure short-horizon real-Raven capacity in the gradient-bearing neural
            proposal while leaving final refined geometry as qualification authority.
        Called by:
            CandidateEvaluator._train_candidate() and focused capacity-contract tests.
        Calls:
            _same_edge_targets(), _baseline_relative_point_objective(), and the canonical
            global SDF-polarity helper when gauge invariance is enabled.

        V12 deliberately makes final refined geometry parameter-free and detached from
        outer SGD.  The evolutionary capacity proof therefore uses the same authored
        same-edge point/tangent teacher as B1b.  Held-out fitness below still measures
        the final refined SDF, so proposal learning cannot masquerade as qualification.
        """
'''
if old_eval not in fitness:
    raise SystemExit('V12.4 StructuralObjective.evaluate docstring anchor not found')
fitness = fitness.replace(old_eval, new_eval, 1)
fitness_path.write_text(fitness, encoding='utf-8')

facade = Path('tools/nsamdr/neural/v9/evolutionary_recovery.py')
facade.write_text('''"""Compatibility facade for the composition-oriented evolutionary recovery package."""\nfrom .evolution import (\n    DEFAULT_GENOME,\n    EVOLUTION_SCHEMA,\n    GENOME_BOUNDS,\n    GENOME_NAMES,\n    CandidateEvaluator,\n    CandidateResult,\n    EvolutionResult,\n    EvolutionaryRecoveryController,\n    FailureDetector,\n    FailureKind,\n    Genome,\n    GenomeRepository,\n    PopulationGenerator,\n    RavenSampleProvider,\n    StructuralFitness,\n    StructuralObjective,\n    classify_failure,\n)\n\n__all__ = [\n    "DEFAULT_GENOME",\n    "EVOLUTION_SCHEMA",\n    "GENOME_BOUNDS",\n    "GENOME_NAMES",\n    "CandidateEvaluator",\n    "CandidateResult",\n    "EvolutionResult",\n    "EvolutionaryRecoveryController",\n    "FailureDetector",\n    "FailureKind",\n    "Genome",\n    "GenomeRepository",\n    "PopulationGenerator",\n    "RavenSampleProvider",\n    "StructuralFitness",\n    "StructuralObjective",\n    "classify_failure",\n]\n''', encoding='utf-8')


test = Path('tools/nsamdr/tests/test_v124_trainer_architecture_contract.py')
test.write_text('''from __future__ import annotations\n\nfrom pathlib import Path\nimport sys\n\nROOT = Path(__file__).resolve().parents[3]\nNEURAL = ROOT / "tools/nsamdr/neural"\nif str(NEURAL) not in sys.path:\n    sys.path.insert(0, str(NEURAL))\n\n\ndef test_v124_trainer_contract_declares_explicit_refiner():\n    from v9 import FidelityResidualNetV9, V9Config\n    from v9 import training as training_module\n    from v9.local_boundary_production_contract import (\n        install_local_boundary_training_contract,\n    )\n\n    # Reproduce the canonical trainer entrypoint ordering. This redirects the\n    # legacy V10.7.9 validator/component map to the installed V12 local contract.\n    install_local_boundary_training_contract(training_module)\n\n    config = V9Config()\n    config.training_activation_checkpointing = False\n    model = FidelityResidualNetV9(config)\n    contract = model.architecture_contract()\n\n    production = contract["productionComponents"]\n    assert production["structural representation"] == "geometry_net.production_structure"\n    assert production["explicit geometry refiner"] == (\n        "geometry_net.production_structure.geometry_refiner"\n    )\n\n    # This is the exact trainer validator that aborted the real Raven run.\n    training_module._validate_v992_architecture_contract(contract)\n\n    modules = training_module._production_component_modules(model)\n    path, module = modules["explicit geometry refiner"]\n    assert path == "geometry_net.production_structure.geometry_refiner"\n    assert module is model.geometry_net.production_structure.geometry_refiner\n''', encoding='utf-8')
