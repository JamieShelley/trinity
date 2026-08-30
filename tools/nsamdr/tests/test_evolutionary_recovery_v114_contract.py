from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


class TestEvolutionaryRecoveryV114Contract:
    # Purpose: Implement test genome is bounded and state dict compatible for TestEvolutionaryRecoveryV114Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_genome_is_bounded_and_state_dict_compatible(self) -> None:
        torch = pytest.importorskip("torch")
        from v9.config import V9Config
        from v9.evolutionary_recovery import GENOME_NAMES, Genome
        from v9.local_boundary_production_contract import set_active_evolution_genome
        from v9.model import FidelityResidualNetV9

        genome = Genome(
            feature_gain=99.0,
            evidence_gain=-99.0,
            distance_scale=1.2,
            curvature_scale=1.1,
            ribbon_scale=0.9,
            extra_branch_gain=1.3,
            csg_logit_scale=0.8,
            correction_scale=1.05,
        ).bounded()
        set_active_evolution_genome(genome.to_dict())
        model = FidelityResidualNetV9(V9Config())
        key = "geometry_net.production_structure.evolution_genome"
        assert key in model.state_dict()
        tensor = model.state_dict()[key]
        assert tuple(tensor.shape) == (len(GENOME_NAMES),)
        assert torch.isfinite(tensor).all()

    # Purpose: Implement test evolution controller is not a production component for TestEvolutionaryRecoveryV114Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_evolution_controller_is_not_a_production_component(self) -> None:
        from v9.config import V9Config
        from v9.model import FidelityResidualNetV9

        model = FidelityResidualNetV9(V9Config())
        contract = model.architecture_contract()
        components = contract["productionComponents"]
        assert "evolution" not in " ".join(str(k).lower() for k in components)
        assert contract["evolutionaryGenomeCheckpointed"] is True
        assert contract["evolutionaryControllerInferenceAuthority"] is False
        assert components["structural representation"] == "geometry_net.production_structure"

    # Purpose: Implement test full production forward observes evolved structure for TestEvolutionaryRecoveryV114Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_full_production_forward_observes_evolved_structure(self) -> None:
        torch = pytest.importorskip("torch")
        from v9.config import V9Config
        from v9.evolutionary_recovery import Genome
        from v9.local_boundary_production_contract import set_active_evolution_genome
        from v9.model import FidelityResidualNetV9

        set_active_evolution_genome(Genome(correction_scale=0.85).to_dict())
        model = FidelityResidualNetV9(V9Config()).eval()
        calls = {"count": 0}
        handle = model.geometry_net.production_structure.register_forward_hook(
            lambda *_args: calls.__setitem__("count", calls["count"] + 1)
        )
        sample = torch.rand((1, 17, 16, 16), dtype=torch.float32)
        sample[:, 3:5] = sample[:, 3:5] * 2.0 - 1.0
        try:
            with torch.inference_mode():
                output = model(sample)
        finally:
            handle.remove()
        assert calls["count"] > 0
        for key in ("albedo", "normal_xy", "material", "roughness", "emissive"):
            assert key in output
            assert torch.isfinite(output[key]).all()
            assert tuple(output[key].shape[-2:]) == (64, 64)

    # Purpose: Verify the capacity gate accepts Raven-scale descent only when held-out structure improves.
    # Called by: Pytest discovery and evolutionary recovery regression validation.
    # Calls: StructuralFitness.measure().
    def test_capacity_gate_uses_held_out_improvement_not_fixed_one_percent_descent(self) -> None:
        torch = pytest.importorskip("torch")
        from v9.evolution.fitness import StructuralFitness

        target = torch.tensor(
            [[[[-2.0, -1.0], [1.0, 2.0]]]],
            dtype=torch.float32,
        )
        source = target + 0.50
        improved = target + 0.25
        fitness = StructuralFitness()

        evidence = fitness.measure(
            improved,
            target,
            source,
            train_loss_before=14.0508,
            train_loss_after=14.0353,
        )
        observed_learning_gain = (14.0508 - 14.0353) / 14.0508
        assert 0.0 < observed_learning_gain < 0.01
        assert float(evidence["gain"]) > 0.0
        assert evidence["passed"] is True

        no_descent = fitness.measure(
            improved,
            target,
            source,
            train_loss_before=14.0508,
            train_loss_after=14.0508,
        )
        assert no_descent["passed"] is False

        validation_regression = fitness.measure(
            target + 0.75,
            target,
            source,
            train_loss_before=14.0508,
            train_loss_after=14.0353,
        )
        assert float(validation_regression["gain"]) < 0.0
        assert validation_regression["passed"] is False

    # Purpose: Implement test failure classifier never evolves software exception for TestEvolutionaryRecoveryV114Contract.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_failure_classifier_never_evolves_software_exception(self) -> None:
        from v9.evolutionary_recovery import FailureKind, classify_failure

        assert classify_failure(error=AttributeError("missing tensor")) == FailureKind.SOFTWARE
        assert classify_failure(error=RuntimeError("CUDA out of memory")) == FailureKind.NUMERICAL
        assert classify_failure(metrics={
            "sdf_stageb_topology_regression_fraction": 1.0,
            "sdf_predicted_missing_contour_fraction": 0.2,
            "sdf_source_missing_contour_fraction": 0.1,
            "sdf_zero_contour_relative_gain_mean": -0.2,
            "sdf_zero_contour_chamfer_pixels": 5.0,
        }) == FailureKind.REPRESENTATION
