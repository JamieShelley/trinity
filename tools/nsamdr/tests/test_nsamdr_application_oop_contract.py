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


def _implementation_files() -> list[Path]:
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


def test_application_readability_contract() -> None:
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
    for path in _implementation_files():
        failures.extend(check_file(path))
    assert failures == []


def test_entry_script_is_only_a_compatibility_surface() -> None:
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


def test_application_layer_uses_composition_not_behavioral_inheritance() -> None:
    """Reject application class inheritance in the newly refactored package.

    Purpose:
        Encode the explicit composition-over-inheritance design constraint.
    Called by:
        pytest.
    Calls:
        _implementation_files(), ast.parse().
    """
    for path in _implementation_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                assert node.bases == [], (path, node.name)
