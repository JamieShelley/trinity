"""Regression tests for the first composition-oriented NSAMDR refactor slice."""
from __future__ import annotations

import ast
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def _evolution_files() -> list[Path]:
    """Return refactored implementation files excluding package export boilerplate.

    Purpose:
        Provide one canonical target set for readability regression tests.
    Called by:
        test_every_function_has_structured_docstring(), test_call_dependency_order().
    Calls:
        Path.rglob().
    """
    return [
        path
        for path in sorted((NEURAL / "v9/evolution").rglob("*.py"))
        if path.name != "__init__.py"
    ]


def test_every_function_has_structured_docstring() -> None:
    """Verify every refactored function/method documents purpose, callers, and callees.

    Purpose:
        Prevent readability documentation from degrading after this refactor.
    Called by:
        pytest.
    Calls:
        _evolution_files(), ast.parse(), ast.walk(), ast.get_docstring().
    """
    for path in _evolution_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False) or ""
                assert "Purpose:" in doc, (path, node.name)
                assert "Called by:" in doc, (path, node.name)
                assert "Calls:" in doc, (path, node.name)


def test_call_dependency_order() -> None:
    """Run the executable declaration-before-caller contract over evolution package.

    Purpose:
        Preserve the user's explicit B-before-A rule whenever A directly calls B.
    Called by:
        pytest.
    Calls:
        _evolution_files(), nsamdr_readability_contract.check_file().
    """
    from nsamdr_readability_contract import check_file

    failures: list[str] = []
    for path in _evolution_files():
        failures.extend(check_file(path))
    assert failures == []


def test_legacy_evolutionary_recovery_api_is_preserved() -> None:
    """Verify the old import surface remains available through the compatibility facade.

    Purpose:
        Allow staged refactoring without breaking training orchestration imports.
    Called by:
        pytest.
    Calls:
        Imports v9.evolutionary_recovery.
    """
    import v9.evolutionary_recovery as legacy

    for name in (
        "Genome",
        "CandidateResult",
        "EvolutionResult",
        "FailureKind",
        "classify_failure",
        "EvolutionaryRecoveryController",
    ):
        assert hasattr(legacy, name), name
