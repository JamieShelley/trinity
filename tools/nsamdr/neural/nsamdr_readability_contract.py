"""Static readability contract for refactored NSAMDR Python orchestration."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable


REQUIRED_DOC_FIELDS = ("Purpose:", "Called by:", "Calls:")
DEFAULT_TARGETS = (
    "tools/nsamdr/neural/v9/evolution",
    "tools/nsamdr/neural/v9/application",
)
ALLOWED_BASES = {"Enum", "str", "nn.Module", "torch.nn.Module", "Dataset"}


def _dotted_name(node: ast.AST) -> str:
    """Return a dotted source-like name for Name/Attribute base expressions.

    Purpose:
        Normalise inheritance expressions for composition-policy checks.
    Called by:
        _check_class_inheritance().
    Calls:
        _dotted_name() recursively for Attribute nodes.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _local_calls(function: ast.AST, names: set[str]) -> set[str]:
    """Collect direct same-module function calls from one function body.

    Purpose:
        Build dependency-order evidence for top-level functions.
    Called by:
        _check_module_function_order().
    Calls:
        ast.walk().
    """
    calls: set[str] = set()
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in names
        ):
            calls.add(node.func.id)
    return calls


def _self_calls(function: ast.AST, names: set[str]) -> set[str]:
    """Collect self/cls method calls to methods declared in the same class.

    Purpose:
        Build dependency-order evidence for object methods.
    Called by:
        _check_class_method_order().
    Calls:
        ast.walk().
    """
    calls: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        if (
            isinstance(owner, ast.Name)
            and owner.id in {"self", "cls"}
            and node.func.attr in names
        ):
            calls.add(node.func.attr)
    return calls


def _check_docstring(path: Path, node: ast.AST, qualified_name: str) -> list[str]:
    """Validate required explanatory fields on one function or method.

    Purpose:
        Enforce explicit purpose/caller/callee documentation for every callable.
    Called by:
        _check_tree().
    Calls:
        ast.get_docstring().
    """
    doc = ast.get_docstring(node, clean=False) or ""
    failures: list[str] = []
    for field in REQUIRED_DOC_FIELDS:
        if field not in doc:
            failures.append(
                f"{path}:{getattr(node, 'lineno', 0)} "
                f"{qualified_name} missing '{field}'"
            )
    return failures


def _check_module_function_order(path: Path, tree: ast.Module) -> list[str]:
    """Require each directly called local function to be declared above its caller.

    Purpose:
        Make module source read from primitives to orchestration.
    Called by:
        _check_tree().
    Calls:
        _local_calls().
    """
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    positions = {node.name: index for index, node in enumerate(functions)}
    names = set(positions)
    failures: list[str] = []
    for caller in functions:
        for callee in _local_calls(caller, names):
            if positions[callee] > positions[caller.name]:
                failures.append(
                    f"{path}:{caller.lineno} {caller.name} calls {callee}, "
                    f"but {callee} is declared below it"
                )
    return failures


def _check_class_method_order(path: Path, cls: ast.ClassDef) -> list[str]:
    """Require each self/cls callee to be declared above its caller.

    Purpose:
        Make object implementations read from low-level operations to public workflows.
    Called by:
        _check_tree().
    Calls:
        _self_calls().
    """
    methods = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    positions = {node.name: index for index, node in enumerate(methods)}
    names = set(positions)
    failures: list[str] = []
    for caller in methods:
        for callee in _self_calls(caller, names):
            if positions[callee] > positions[caller.name]:
                failures.append(
                    f"{path}:{caller.lineno} {cls.name}.{caller.name} calls {callee}, "
                    f"but {callee} is declared below it"
                )
    return failures


def _check_class_inheritance(path: Path, cls: ast.ClassDef) -> list[str]:
    """Reject application-level inheritance while allowing framework/value bases.

    Purpose:
        Enforce composition over inheritance without breaking required framework protocols.
    Called by:
        _check_tree().
    Calls:
        _dotted_name().
    """
    failures: list[str] = []
    for base in cls.bases:
        name = _dotted_name(base)
        if name and name not in ALLOWED_BASES:
            failures.append(
                f"{path}:{cls.lineno} {cls.name} inherits from {name}; "
                "use composition unless it is a framework base"
            )
    return failures


def _check_tree(path: Path, tree: ast.Module) -> list[str]:
    """Run all documentation, ordering, and composition checks on one AST.

    Purpose:
        Produce a complete per-file readability failure list.
    Called by:
        check_file().
    Calls:
        _check_docstring(), _check_module_function_order(),
        _check_class_method_order(), _check_class_inheritance().
    """
    failures = _check_module_function_order(path, tree)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            failures.extend(_check_docstring(path, node, node.name))
        elif isinstance(node, ast.ClassDef):
            failures.extend(_check_class_inheritance(path, node))
            failures.extend(_check_class_method_order(path, node))
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    failures.extend(
                        _check_docstring(path, method, f"{node.name}.{method.name}")
                    )
    return failures


def check_file(path: Path) -> list[str]:
    """Parse and validate one Python file against the NSAMDR readability contract.

    Purpose:
        Provide a reusable unit for CLI/tests and validation integration.
    Called by:
        check_targets().
    Calls:
        _check_tree().
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _check_tree(path, tree)


def check_targets(
    repo_root: Path,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> list[str]:
    """Validate every Python file under requested repository-relative targets.

    Purpose:
        Enforce the readability contract across all refactored NSAMDR packages.
    Called by:
        main() and tests.
    Calls:
        check_file().
    """
    failures: list[str] = []
    for target in targets:
        path = Path(repo_root) / target
        for python_file in sorted(path.rglob("*.py")):
            if python_file.name == "__init__.py":
                continue
            failures.extend(check_file(python_file))
    return failures


def main() -> int:
    """Run default repository readability validation.

    Purpose:
        Expose a command suitable for nsamdr.bat validation integration.
    Called by:
        Python __main__ block or build scripts.
    Calls:
        check_targets().
    """
    repo_root = Path(__file__).resolve().parents[3]
    failures = check_targets(repo_root)
    if failures:
        for failure in failures:
            print(f"[readability] FAIL: {failure}")
        return 2
    print("[readability] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
