"""Pass-driven orchestration contracts after composition-oriented Stage 2 refactor."""
from __future__ import annotations

import inspect
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


class TestPassDrivenPipelineContract:
    def test_stage_order_is_local_then_full_production_downstream(self) -> None:
        """Verify explicit StagePlan order while B1/B2 executes through sdf-proof.

        Purpose:
            Preserve the current local-boundary curriculum across OOP decomposition.
        Called by:
            pytest.
        Calls:
            QualificationGates(), StagePlan().
        """
        from v9.application.gates import QualificationGates, StagePlan

        phases = [
            definition.phase
            for definition in StagePlan(QualificationGates()).definitions
        ]
        assert phases == [
            "sdf-bootstrap",
            "seam-proof",
            "seam-authority",
            "gate-proof",
            "detail-reconstruction",
        ]
        assert "sdf-proof" not in phases

    def test_each_stage_is_still_canonical_train_v9(self) -> None:
        """Verify stage and final execution both route through TrainingBackend.

        Purpose:
            Ensure the refactor did not introduce alternate trainer implementations.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.application.pipeline import PassDrivenPipeline

        stage = inspect.getsource(PassDrivenPipeline._run_stage)
        final = inspect.getsource(PassDrivenPipeline._run_final)
        invoke = inspect.getsource(PassDrivenPipeline._invoke)
        assert "self._invoke" in stage
        assert '"sdf-proof" if definition.phase == "sdf-bootstrap"' in stage
        assert "stop_after_phase=trainer_stop_phase" in stage
        assert "self._invoke" in final
        assert "stop_after_phase=None" in final
        assert "self.backend.run" in invoke

    def test_local_geometry_promotion_is_fail_closed(self) -> None:
        """Verify structural success promotes explicitly and failure rejects downstream.

        Purpose:
            Preserve the original B1/B2 promotion/rejection semantics.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.application.pipeline import PassDrivenPipeline

        source = inspect.getsource(PassDrivenPipeline._run_stage)
        assert "_complete_structural_stage" in source
        assert "self.experiments.reject" in source
        assert "Downstream stages were not run." in source

    def test_final_stage_requires_real_production_final(self) -> None:
        """Verify final stage still requires trainingSafetyPass and production-final.

        Purpose:
            Preserve strict final selection semantics across orchestration refactoring.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.application.pipeline import PassDrivenPipeline

        source = inspect.getsource(PassDrivenPipeline._run_final)
        assert 'latest.get("trainingSafetyPass", False)' in source
        assert '== "production-final"' in source
    def test_backend_patch_keeps_windows_worker_initializer_pickleable(self) -> None:
        """Verify V11.4 callbacks do not break spawn-based DataLoader workers.

        Purpose:
            Prevent nested trainer callbacks from making the TrainingService singleton
            unpickleable when Windows serializes its bound worker initializer.
        Called by:
            pytest.
        Calls:
            TrainingBackend(), pickle.dumps().
        """
        import v9.training as training
        from v9.application.backend import TrainingBackend

        TrainingBackend()
        callback = training._training_service._explicit_primitive_structure_microproof
        assert "<locals>" not in callback.__qualname__
        pickle.dumps(training._data_worker_init)

    def test_v114_sdf_proof_uses_live_local_production_graph(self) -> None:
        """Prevent the retired B1b loss from replacing V11.4 gradients.

        Purpose:
            Keep sdf-proof on the current production_structure while retaining the
            legacy compact loss only for schemas without that structure.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.training import TrainingService

        source = inspect.getsource(TrainingService.train_v9)
        assert 'hasattr(model.geometry_net, "production_structure")' in source
        assert 'phase == "sdf-proof" and not local_structure_phase' in source
        assert 'b1b_classifier_qualified = True' in source
        assert 'b1b_parameters_qualified = True' in source

    def test_v114_sdf_proof_uses_production_batch_loader(self) -> None:
        """Keep the full V11.4 graph off the retired batch-8 primitive loader.

        Purpose:
            Ensure local-boundary sdf-proof uses the canonical production loader so
            the full 128-to-512 graph remains batch-size 1 for bounded VRAM/RAM use.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.training import TrainingService

        source = inspect.getsource(TrainingService.train_v9)
        assert 'train_loader if local_structure_phase else parametric_train_loader' in source

    def test_b1a_topology_checkpoint_locks_after_first_pass(self) -> None:
        """Prevent later bootstrap epochs from replacing qualified B1a topology.

        Purpose:
            Keep the first topology-safe B1a checkpoint authoritative for sdf-proof.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.training import TrainingService

        source = inspect.getsource(TrainingService.train_v9)
        block = source.split(
            '# B1a qualifies topology only; smoothness belongs exclusively to B1b.', 1
        )[1].split('elif phase == "sdf-proof":', 1)[0]
        assert 'if not topology_bootstrapped:' in block
        assert block.index('if not topology_bootstrapped:') < block.index('best_b1a_path')
        assert 'topology checkpoint locked for sdf-proof' in block

    def test_evolution_requires_permanent_topology_audit(self) -> None:
        """Prevent one-crop evolutionary fitness from bypassing production topology.

        Purpose:
            Require candidate acceptance to use the permanent 29-case proof family and
            the same connected-component/hole topology mismatch used by B1/B2.
        Called by:
            pytest.
        Calls:
            inspect.getsource().
        """
        from v9.evolution.candidate import CandidateEvaluator
        from v9.evolution.fitness import StructuralFitness

        topology_source = inspect.getsource(CandidateEvaluator._measure_permanent_topology)
        fitness_source = inspect.getsource(StructuralFitness.measure)
        assert 'SyntheticGeometryValidationDataset' in topology_source
        assert 'max(29' in topology_source
        assert '+ 9_911' in topology_source
        assert 'sdf_topology_mismatch' in topology_source
        assert 'predicted_topology > source_topology' in topology_source
        assert 'topology_regression_fraction <= 0.0' in fitness_source
