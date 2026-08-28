"""Quick stays bounded while respecting V9Config validator minima in V11.3."""
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py"


def _literal_assignment(name: str) -> dict[str, int]:
    tree = ast.parse(ENTRY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            value = ast.literal_eval(node.value)
            assert isinstance(value, dict)
            return value
    raise AssertionError(name)


def test_quick_is_a_short_local_capacity_run() -> None:
    budget = _literal_assignment("QUICK_WORK_BUDGET")
    assert budget["identity_epochs"] == 3
    assert budget["residual_epochs"] == 1
    assert budget["tiles_per_epoch"] == 64
    assert budget["raven_downstream_tiles_per_epoch"] == 16
    assert budget["seam_proof_epochs"] == 1
    assert budget["seam_authority_epochs"] == 1
    assert budget["detail_epochs"] == 1


def test_retired_primitive_bank_uses_only_validator_minimum() -> None:
    budget = _literal_assignment("QUICK_WORK_BUDGET")
    assert budget["parametric_primitive_train_tiles_per_epoch"] == 14
