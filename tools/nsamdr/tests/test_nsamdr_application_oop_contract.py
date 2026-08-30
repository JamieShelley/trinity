"""Static contracts for the composition-oriented NSAMDR training application."""
from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
APPLICATION = NEURAL / "v9/application"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


class TestNsamdrApplicationOopContract:
    def _implementation_files(self) -> list[Path]:
        """Return application implementation files excluding package export boilerplate.

        Purpose:
            Provide one canonical file set for application readability tests.
        Called by:
            test_application_readability_contract().
        Calls:
            Path.rglob().
        """
        return [
            path
            for path in sorted(APPLICATION.rglob("*.py"))
            if path.name != "__init__.py"
        ]

    def test_application_readability_contract(self) -> None:
        """Enforce structured docs, declaration order, and composition on Stage 2 files.

        Purpose:
            Prevent the new application layer from regressing into monolithic orchestration.
        Called by:
            pytest.
        Calls:
            _implementation_files(), nsamdr_readability_contract.check_file().
        """
        from nsamdr_readability_contract import check_file

        failures: list[str] = []
        for path in self._implementation_files():
            failures.extend(check_file(path))
        assert failures == []

    def test_entry_script_is_only_a_compatibility_surface(self) -> None:
        """Ensure the executable script stays small and delegates to the application package.

        Purpose:
            Prevent orchestration logic from accumulating again in the CLI facade.
        Called by:
            pytest.
        Calls:
            ast.parse().
        """
        entry = NEURAL / "train_nsamdr_v9_preview_experiment.py"
        tree = ast.parse(entry.read_text(encoding="utf-8"))
        functions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        assert functions == []
        assert classes == []
        assert len(entry.read_text(encoding="utf-8").splitlines()) < 60

    def test_training_backend_synchronizes_object_owned_contract_callbacks(self) -> None:
        """Require installed V11.4 callbacks to reach the TrainingService singleton.

        Purpose:
            Prevent module-level compatibility patches from leaving train_v9 on
            stale object-owned validators, microproofs, or component maps.
        Called by:
            pytest.
        Calls:
            ast.parse().
        """
        backend = APPLICATION / "backend.py"
        tree = ast.parse(backend.read_text(encoding="utf-8"))
        assignments: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "service"
                ):
                    assignments.add(target.attr)
        assert {
            "_validate_v992_architecture_contract",
            "_explicit_primitive_structure_microproof",
            "_production_component_modules",
        } <= assignments

    def test_failed_runs_persist_traceback_evidence(self) -> None:
        """Require exception details to survive in exported experiment diagnostics.

        Purpose:
            Prevent diagnostics ZIPs from reducing runtime failures to a generic
            ``interrupted-or-failed`` status with no actionable traceback.
        Called by:
            pytest.
        Calls:
            Path.read_text().
        """
        source = (APPLICATION / "experiment.py").read_text(encoding="utf-8")
        assert "runtime_failure.json" in source
        assert "traceback.format_exception" in source
        assert '"failedExceptionType"' in source
        assert '"failedExceptionMessage"' in source
        assert '"failureEvidence"' in source

    def test_workflow_diagnostics_include_runtime_and_evolution_evidence(self) -> None:
        """Require the diagnostic archive to include failure-relevant folders.

        Purpose:
            Preserve runtime, evolution, and trainer diagnostics needed to diagnose
            a failed Quick/Full workflow without a separate GUI log.
        Called by:
            pytest.
        Calls:
            Path.read_text().
        """
        source = (NEURAL / "run_nsamdr_v9_raven_tune_preview.py").read_text(
            encoding="utf-8"
        )
        assert 'experiment_directory / "diagnostics"' in source
        assert 'experiment_directory / "evolution"' in source
        assert 'experiment_directory / "evidence"' in source
        assert 'experiment_directory / "nsamdr_v9_fidelity.json"' in source

    def test_application_layer_uses_composition_not_behavioral_inheritance(self) -> None:
        """Reject application class inheritance in the newly refactored package.

        Purpose:
            Encode the explicit composition-over-inheritance design constraint.
        Called by:
            pytest.
        Calls:
            _implementation_files(), ast.parse().
        """
        for path in self._implementation_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    assert node.bases == [], (path, node.name)

_test_nsamdr_application_oop_contract = TestNsamdrApplicationOopContract()
_implementation_files = _test_nsamdr_application_oop_contract._implementation_files
