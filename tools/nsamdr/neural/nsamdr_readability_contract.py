"""Whole-tree readability and object-ownership contract for NSAMDR Python."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable


REQUIRED_DOC_FIELDS = ("Purpose:", "Called by:", "Calls:")
DEFAULT_TARGETS = ("tools/nsamdr",)
ALLOWED_BASES = {
    "ABC",
    "Dataset",
    "Enum",
    "Exception",
    "IntEnum",
    "IterableDataset",
    "KeyError",
    "NamedTuple",
    "Protocol",
    "RuntimeError",
    "StrEnum",
    "TypeError",
    "ValueError",
    "nn.Module",
    "object",
    "str",
    "torch.nn.Module",
    "torch.utils.data.Dataset",
    "torch.utils.data.IterableDataset",
    "tuple",
    "unittest.TestCase",
}


class NSAMDRReadabilityContract:
    def _dotted_name(self, node: ast.AST) -> str:
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
            prefix = self._dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    def _local_calls(self, function: ast.AST, names: set[str]) -> set[str]:
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

    def _self_calls(self, function: ast.AST, names: set[str]) -> set[str]:
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

    def _reaches(self, graph: dict[str, set[str]], start: str, target: str) -> bool:
        """Return whether one same-class method reaches another in the call graph.

        Purpose:
            Distinguish impossible-to-order recursive cycles from ordinary ordering errors.
        Called by:
            _mutually_recursive().
        Calls:
            No local helper methods.
        """
        pending = [start]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(graph.get(current, set()) - seen)
        return False

    def _mutually_recursive(
        self,
        graph: dict[str, set[str]],
        caller: str,
        callee: str,
    ) -> bool:
        """Return whether caller and callee belong to one recursive cycle.

        Purpose:
            Exempt only dependency cycles that cannot satisfy a linear source order.
        Called by:
            _check_class_method_order().
        Calls:
            _reaches().
        """
        return self._reaches(graph, caller, callee) and self._reaches(graph, callee, caller)

    def _structured_evidence(self, path: Path, node: ast.AST) -> str:
        """Return structured callable documentation from docstrings and source comments.

        Purpose:
            Accept the requested Purpose/Called by/Calls source comments while retaining
            compatibility with existing structured docstrings.
        Called by:
            _check_documentation().
        Calls:
            ast.get_docstring().
        """
        doc = ast.get_docstring(node, clean=False) or ""
        decorators = getattr(node, "decorator_list", [])
        start = min([getattr(node, "lineno", 1)] + [item.lineno for item in decorators])
        lines = path.read_text(encoding="utf-8").splitlines()
        comments = "\n".join(lines[max(0, start - 5): max(0, start - 1)])
        return f"{doc}\n{comments}"

    def _check_documentation(self, path: Path, node: ast.AST, qualified_name: str) -> list[str]:
        """Validate required explanatory fields on one function or method.

        Purpose:
            Enforce explicit purpose/caller/callee documentation for every callable.
        Called by:
            _check_tree().
        Calls:
            _structured_evidence().
        """
        evidence = self._structured_evidence(path, node)
        failures: list[str] = []
        for field in REQUIRED_DOC_FIELDS:
            if field not in evidence:
                failures.append(
                    f"{path}:{getattr(node, 'lineno', 0)} "
                    f"{qualified_name} missing '{field}'"
                )
        return failures

    def _check_module_function_order(self, path: Path, tree: ast.Module) -> list[str]:
        """Require each directly called local function to be declared above its caller.

        Purpose:
            Report dependency ordering before the stronger object-ownership check reports
            any remaining module-level implementation functions.
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
            for callee in self._local_calls(caller, names):
                if positions[callee] > positions[caller.name]:
                    failures.append(
                        f"{path}:{caller.lineno} {caller.name} calls {callee}, "
                        f"but {callee} is declared below it"
                    )
        return failures

    def _check_class_method_order(self, path: Path, cls: ast.ClassDef) -> list[str]:
        """Require each self/cls callee to be declared above its caller.

        Purpose:
            Make object implementations read from low-level operations to public workflows.
        Called by:
            _check_tree().
        Calls:
            _self_calls(), _mutually_recursive().
        """
        methods = [
            node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        positions = {node.name: index for index, node in enumerate(methods)}
        names = set(positions)
        graph = {node.name: self._self_calls(node, names) for node in methods}
        failures: list[str] = []
        for caller in methods:
            for callee in graph[caller.name]:
                if (
                    positions[callee] > positions[caller.name]
                    and not self._mutually_recursive(graph, caller.name, callee)
                ):
                    failures.append(
                        f"{path}:{caller.lineno} {cls.name}.{caller.name} calls {callee}, "
                        f"but {callee} is declared below it"
                    )
        return failures

    def _check_class_inheritance(self, path: Path, cls: ast.ClassDef) -> list[str]:
        """Reject application-level inheritance while allowing framework/value bases.

        Purpose:
            Enforce composition over inheritance without breaking required framework,
            protocol, exception, dataset, enum, or value-type bases.
        Called by:
            _check_tree().
        Calls:
            _dotted_name().
        """
        failures: list[str] = []
        for base in cls.bases:
            name = self._dotted_name(base)
            if name and name not in ALLOWED_BASES:
                failures.append(
                    f"{path}:{cls.lineno} {cls.name} inherits from {name}; "
                    "use composition unless it is a framework/value base"
                )
        return failures

    def _check_tree(self, path: Path, tree: ast.Module) -> list[str]:
        """Run whole-file object ownership, documentation, ordering and composition checks.

        Purpose:
            Produce a complete per-file readability failure list.
        Called by:
            check_file().
        Calls:
            _check_documentation(), _check_module_function_order(),
            _check_class_method_order(), _check_class_inheritance().
        """
        failures = self._check_module_function_order(path, tree)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                failures.append(
                    f"{path}:{node.lineno} top-level implementation function "
                    f"'{node.name}' must belong to an object"
                )
                failures.extend(self._check_documentation(path, node, node.name))
            elif isinstance(node, ast.ClassDef):
                failures.extend(self._check_class_inheritance(path, node))
                failures.extend(self._check_class_method_order(path, node))
                for method in node.body:
                    if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        failures.extend(
                            self._check_documentation(path, method, f"{node.name}.{method.name}")
                        )
        return failures

    def check_file(self, path: Path) -> list[str]:
        """Parse and validate one Python file against the NSAMDR readability contract.

        Purpose:
            Provide a reusable unit for CLI/tests and validation integration.
        Called by:
            check_targets().
        Calls:
            _check_tree().
        """
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        return self._check_tree(path, tree)

    def check_targets(
        self,
        repo_root: Path,
        targets: Iterable[str] = DEFAULT_TARGETS,
    ) -> list[str]:
        """Validate every Python file under requested repository-relative targets.

        Purpose:
            Enforce the readability/OOP contract across the complete NSAMDR Python tree.
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
                failures.extend(self.check_file(python_file))
        return failures

    def main(self) -> int:
        """Run default whole-repository NSAMDR readability validation.

        Purpose:
            Expose a command suitable for nsamdr.bat validation integration.
        Called by:
            Python __main__ block or build scripts.
        Calls:
            check_targets().
        """
        repo_root = Path(__file__).resolve().parents[3]
        failures = self.check_targets(repo_root)
        if failures:
            for failure in failures:
                print(f"[readability] FAIL: {failure}")
            return 2
        print("[readability] PASS")
        return 0

_n_s_a_m_d_r_readability_contract = NSAMDRReadabilityContract()
_dotted_name = _n_s_a_m_d_r_readability_contract._dotted_name
_local_calls = _n_s_a_m_d_r_readability_contract._local_calls
_self_calls = _n_s_a_m_d_r_readability_contract._self_calls
_reaches = _n_s_a_m_d_r_readability_contract._reaches
_mutually_recursive = _n_s_a_m_d_r_readability_contract._mutually_recursive
_structured_evidence = _n_s_a_m_d_r_readability_contract._structured_evidence
_check_documentation = _n_s_a_m_d_r_readability_contract._check_documentation
_check_module_function_order = _n_s_a_m_d_r_readability_contract._check_module_function_order
_check_class_method_order = _n_s_a_m_d_r_readability_contract._check_class_method_order
_check_class_inheritance = _n_s_a_m_d_r_readability_contract._check_class_inheritance
_check_tree = _n_s_a_m_d_r_readability_contract._check_tree
check_file = _n_s_a_m_d_r_readability_contract.check_file
check_targets = _n_s_a_m_d_r_readability_contract.check_targets
main = _n_s_a_m_d_r_readability_contract.main


if __name__ == "__main__":
    raise SystemExit(main())
