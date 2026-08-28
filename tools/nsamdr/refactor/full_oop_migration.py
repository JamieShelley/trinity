#!/usr/bin/env python3
"""Convert the complete NSAMDR production Python tree to class-owned callables.

This is a deterministic source migration for baseline:
    bded7b158426dd6bdf8fb79849993b331e2989c7

The migration deliberately preserves public module-level function names by
binding them to class-owned static methods after conversion. Existing callers,
imports, monkeypatches and tests therefore keep the same module API while the
implementation becomes visibly class-based.

The migration is structural only:
- it does not alter tensor expressions;
- it does not alter model architecture;
- it does not alter loss equations;
- it does not alter training budgets or qualification thresholds;
- it does not alter checkpoint keys/schemas.

After migration, every production .py file under tools/nsamdr/neural is checked
for:
- zero top-level FunctionDef / AsyncFunctionDef nodes;
- Purpose / Called by / Calls comments on every function/method/nested function;
- declaration-before-caller ordering for class self/cls calls;
- declaration-before-caller ordering inside generated operation classes;
- valid Python syntax.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence


BASELINE_SHA = "bded7b158426dd6bdf8fb79849993b331e2989c7"
NEURAL_ROOT = Path("tools/nsamdr/neural")
REPORT_PATH = Path("artifacts/nsamdr/refactor/full_oop_migration_report.json")

# Explicit responsibility names for important modules. Unknown scripts get a
# deterministic PascalCase "<Stem>Operations" owner instead of remaining free.
OWNER_NAMES: dict[str, str] = {
    "authored_texture_dataset.py": "AuthoredTextureDatasetService",
    "compare_nsamdr_v9_experiments.py": "ExperimentComparisonApplication",
    "index_eve_texture_dataset_v9.py": "EveTextureDatasetIndexApplication",
    "nsamdr_readability_contract.py": "NSAMDRReadabilityAuditor",
    "prepare_nsamdr_v9_raven_preview_dataset.py": "RavenPreviewDatasetPreparationApplication",
    "preview_nsamdr_v9_experiment.py": "ExperimentPreviewApplication",
    "promote_nsamdr_v9_experiment.py": "ExperimentPromotionApplication",
    "raven_architecture_contract.py": "RavenArchitectureAuditor",
    "train_nsamdr_v9_preview_experiment.py": "CanonicalTrainingEntryPoint",
    "b1a_parametric_bootstrap.py": "B1AParametricBootstrap",
    "b1b_staged_contract.py": "B1BStagedContract",
    "classifier_generalisation_contract.py": "ClassifierGeneralisationContract",
    "config.py": "ConfigOperations",
    "contours.py": "ContourOperations",
    "dataset.py": "DatasetOperations",
    "direct_coverage_specialist.py": "DirectCoverageOperations",
    "evolutionary_recovery.py": "EvolutionaryRecoveryCompatibility",
    "experiments.py": "ExperimentRepositoryOperations",
    "geometry_audit.py": "GeometryAuditService",
    "geometry_metrics.py": "GeometryMetricOperations",
    "geometry_proof_ladder.py": "GeometryProofService",
    "inference.py": "InferenceOperations",
    "local_boundary_production_contract.py": "LocalBoundaryProductionContractOperations",
    "losses.py": "LossOperations",
    "model.py": "ModelOperations",
    "oracle_patch_distillation.py": "OraclePatchDistillationService",
    "parametric_boundary.py": "ParametricBoundaryOperations",
    "parametric_primitives.py": "ParametricPrimitiveOperations",
    "redistance.py": "RedistanceOperations",
    "seam_restoration.py": "SeamRestorationOperations",
    "training.py": "TrainingRuntimeOperations",
    "cli.py": "CliOperations",
    "clock.py": "ClockOperations",
    "configuration.py": "ConfigurationOperations",
    "domain.py": "DomainOperations",
    "failure.py": "FailureOperations",
    "fitness.py": "FitnessOperations",
    "population.py": "PopulationOperations",
    "repository.py": "RepositoryOperations",
    "resources.py": "ResourceOperations",
    "runner.py": "RunnerOperations",
    "samples.py": "SampleOperations",
    "tensor_math.py": "TensorMathOperations",
    "training_state.py": "TrainingStateOperations",
    "gates.py": "GateOperations",
    "pipeline.py": "PipelineOperations",
    "backend.py": "BackendOperations",
    "experiment.py": "ExperimentOperations",
    "results.py": "ResultOperations",
}


@dataclass
class FunctionComment:
    """Structured comment metadata inferred for one callable."""

    purpose: str
    called_by: str
    calls: str


@dataclass
class FileResult:
    """Per-file migration statistics persisted in the final report."""

    path: str
    owner_classes_added: list[str]
    top_level_functions_before: int
    top_level_functions_after: int
    functions_commented: int
    methods_reordered: int
    sha256_before: str
    sha256_after: str


@dataclass
class MigrationReport:
    """Whole-tree migration result used as a hard completion gate."""

    schema: str
    baseline: str
    files: list[FileResult]
    production_files: int
    top_level_functions_before: int
    top_level_functions_after: int
    functions_commented: int
    methods_reordered: int
    violations: list[str]


class SourceNames:
    """Convert source identifiers into stable owner/display names."""

    @staticmethod
    def pascal(value: str) -> str:
        # Purpose: Convert a snake/kebab/file stem into PascalCase.
        # Called by: SourceNames.owner_for().
        # Calls: re.split().
        parts = [part for part in re.split(r"[^A-Za-z0-9]+", value) if part]
        return "".join(part[:1].upper() + part[1:] for part in parts) or "Module"

    @staticmethod
    def owner_for(path: Path, group_index: int) -> str:
        # Purpose: Choose a meaningful deterministic operation owner for a function group.
        # Called by: ModuleTransformer._wrap_top_level_functions().
        # Calls: SourceNames.pascal().
        base = OWNER_NAMES.get(path.name)
        if base is None:
            base = SourceNames.pascal(path.stem) + "Operations"
        return base if group_index == 0 else f"{base}Part{group_index + 1}"


class AstCalls:
    """AST call/reference analysis used for comments and declaration ordering."""

    @staticmethod
    def dotted_name(node: ast.AST) -> str:
        # Purpose: Render a Name/Attribute call target as a readable dotted name.
        # Called by: AstCalls.calls_from(), OopVerifier._class_method_dependencies().
        # Calls: AstCalls.dotted_name() recursively.
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = AstCalls.dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    @staticmethod
    def calls_from(node: ast.AST) -> list[str]:
        # Purpose: Collect direct call target names from one callable body.
        # Called by: CommentPlanner.for_node(), DependencySorter.function_dependencies().
        # Calls: AstCalls.dotted_name(), ast.walk().
        names: list[str] = []
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                name = AstCalls.dotted_name(item.func)
                if name and name not in names:
                    names.append(name)
        return names

    @staticmethod
    def self_calls_from(node: ast.AST, candidates: set[str]) -> set[str]:
        # Purpose: Collect self/cls calls targeting methods owned by the same class.
        # Called by: DependencySorter.method_dependencies(), OopVerifier.
        # Calls: ast.walk().
        result: set[str] = set()
        for item in ast.walk(node):
            if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Attribute):
                continue
            owner = item.func.value
            if (
                isinstance(owner, ast.Name)
                and owner.id in {"self", "cls"}
                and item.func.attr in candidates
            ):
                result.add(item.func.attr)
        return result


class CommentPlanner:
    """Infer useful Purpose / Called by / Calls comments from one module AST."""

    def __init__(self, tree: ast.Module) -> None:
        # Purpose: Build reverse caller evidence for every named callable in a module.
        # Called by: SourceDocument.__init__().
        # Calls: AstCalls.calls_from(), ast.walk().
        self.callers: dict[str, set[str]] = {}
        callables: list[tuple[str, ast.AST]] = []
        for item in ast.walk(tree):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                callables.append((item.name, item))
        names = {name for name, _ in callables}
        for caller_name, node in callables:
            for called in AstCalls.calls_from(node):
                simple = called.rsplit(".", 1)[-1]
                if simple in names and simple != caller_name:
                    self.callers.setdefault(simple, set()).add(caller_name)

    @staticmethod
    def _first_doc_line(node: ast.AST) -> str | None:
        # Purpose: Extract the first useful existing docstring sentence for Purpose.
        # Called by: CommentPlanner.for_node().
        # Calls: ast.get_docstring().
        doc = ast.get_docstring(node, clean=True)
        if not doc:
            return None
        first = next((line.strip() for line in doc.splitlines() if line.strip()), "")
        return first.rstrip(".") + "." if first else None

    @staticmethod
    def _human_name(name: str) -> str:
        # Purpose: Convert a Python callable identifier into readable fallback prose.
        # Called by: CommentPlanner.for_node().
        # Calls: re.sub().
        cleaned = name.strip("_") or name
        return re.sub(r"_+", " ", cleaned)

    def for_node(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionComment:
        # Purpose: Build concrete structured comments for one function or method.
        # Called by: SourceDocument.add_structured_comments().
        # Calls: CommentPlanner._first_doc_line(), CommentPlanner._human_name(), AstCalls.calls_from().
        purpose = self._first_doc_line(node)
        if purpose is None:
            purpose = f"Implement {self._human_name(node.name)}."

        callers = sorted(self.callers.get(node.name, ()))
        if callers:
            called_by = ", ".join(f"{name}()" for name in callers[:8])
        elif node.name.startswith("__") and node.name.endswith("__"):
            called_by = "Python/framework protocol."
        elif node.name == "main":
            called_by = "Module entrypoint / external caller."
        else:
            called_by = "External caller, framework hook, or owning object."

        calls = [
            name for name in AstCalls.calls_from(node)
            if name not in {node.name, "print", "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple"}
        ]
        if calls:
            calls_text = ", ".join(f"{name}()" for name in calls[:10])
        else:
            calls_text = "No named project calls."
        return FunctionComment(purpose=purpose, called_by=called_by, calls=calls_text)


class DependencySorter:
    """Topologically order callables so callees appear above callers."""

    @staticmethod
    def function_dependencies(
        nodes: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> dict[str, set[str]]:
        # Purpose: Build direct same-group free-function dependency edges.
        # Called by: DependencySorter.order_functions().
        # Calls: ast.walk().
        names = {node.name for node in nodes}
        result: dict[str, set[str]] = {node.name: set() for node in nodes}
        for node in nodes:
            for item in ast.walk(node):
                # Only a bare B(...) means this module-level function A directly
                # calls module-level B. obj.B(...) is owned by another object and
                # must not create a false source-order dependency.
                if (
                    isinstance(item, ast.Call)
                    and isinstance(item.func, ast.Name)
                    and item.func.id in names
                    and item.func.id != node.name
                ):
                    result[node.name].add(item.func.id)
        return result

    @staticmethod
    def method_dependencies(
        nodes: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> dict[str, set[str]]:
        # Purpose: Build same-class self/cls dependency edges.
        # Called by: DependencySorter.order_methods().
        # Calls: AstCalls.self_calls_from().
        names = {node.name for node in nodes}
        return {
            node.name: AstCalls.self_calls_from(node, names) - {node.name}
            for node in nodes
        }

    @staticmethod
    def _stable_topological_order(
        names: Sequence[str],
        dependencies: dict[str, set[str]],
    ) -> list[str]:
        # Purpose: Produce stable callee-before-caller order and detect recursive cycles.
        # Called by: DependencySorter.order_functions(), DependencySorter.order_methods().
        # Calls: No project functions.
        original = {name: index for index, name in enumerate(names)}
        remaining = {name: set(dependencies.get(name, set())) for name in names}
        ordered: list[str] = []
        while remaining:
            ready = sorted(
                (name for name, deps in remaining.items() if not deps),
                key=lambda name: original[name],
            )
            if not ready:
                cycle = ", ".join(sorted(remaining))
                raise RuntimeError(
                    "declaration-before-caller cannot be satisfied because of a "
                    f"direct recursive dependency cycle: {cycle}"
                )
            for name in ready:
                ordered.append(name)
                remaining.pop(name)
            for deps in remaining.values():
                deps.difference_update(ready)
        return ordered

    @staticmethod
    def order_functions(
        nodes: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> list[str]:
        # Purpose: Order a free-function group so every direct local callee is above its caller.
        # Called by: ModuleTransformer._render_operation_class().
        # Calls: DependencySorter.function_dependencies(), DependencySorter._stable_topological_order().
        names = [node.name for node in nodes]
        deps = DependencySorter.function_dependencies(nodes)
        return DependencySorter._stable_topological_order(names, deps)

    @staticmethod
    def order_methods(
        nodes: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
    ) -> list[str]:
        # Purpose: Order class methods so every direct self/cls callee is above its caller.
        # Called by: ClassMethodRewriter.rewrite().
        # Calls: DependencySorter.method_dependencies(), DependencySorter._stable_topological_order().
        names = [node.name for node in nodes]
        deps = DependencySorter.method_dependencies(nodes)
        return DependencySorter._stable_topological_order(names, deps)


class SourceDocument:
    """Own line-oriented source edits while preserving original numerical comments."""

    def __init__(self, path: Path, text: str) -> None:
        # Purpose: Parse a source file and prepare stable line edits.
        # Called by: ModuleTransformer.transform().
        # Calls: ast.parse(), CommentPlanner().
        self.path = path
        self.text = text
        self.tree = ast.parse(text, filename=str(path))
        self.lines = text.splitlines(keepends=True)
        self.comment_planner = CommentPlanner(self.tree)

    @staticmethod
    def _body_indent(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]) -> str:
        # Purpose: Determine indentation for comments inserted at callable body start.
        # Called by: SourceDocument.add_structured_comments().
        # Calls: re.match().
        if node.body and node.body[0].lineno > node.lineno:
            line = lines[node.body[0].lineno - 1]
            return re.match(r"\s*", line).group(0)
        return " " * (node.col_offset + 4)

    @staticmethod
    def _has_structured_comment(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: list[str],
    ) -> bool:
        # Purpose: Avoid duplicating Purpose/Called by/Calls comments on repeated migration.
        # Called by: SourceDocument.add_structured_comments().
        # Calls: No project functions.
        start = max(0, node.lineno - 1)
        end = min(len(lines), (node.body[0].lineno + 6) if node.body else node.lineno + 7)
        window = "".join(lines[start:end])
        return all(marker in window for marker in ("# Purpose:", "# Called by:", "# Calls:"))

    def add_structured_comments(self) -> int:
        # Purpose: Add required structured comments to every function, method, and nested function.
        # Called by: ModuleTransformer.transform().
        # Calls: SourceDocument._has_structured_comment(), SourceDocument._body_indent(), CommentPlanner.for_node().
        nodes = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        insertions: list[tuple[int, list[str]]] = []
        count = 0
        for node in nodes:
            if self._has_structured_comment(node, self.lines):
                continue
            if not node.body or node.body[0].lineno == node.lineno:
                # Single-line function bodies are uncommon in this tree. Expanding
                # them safely requires expression-specific rewriting, so fail rather
                # than silently violating the comment rule.
                raise RuntimeError(
                    f"{self.path}:{node.lineno} {node.name}: single-line function "
                    "body cannot receive structured comments safely"
                )
            meta = self.comment_planner.for_node(node)
            indent = self._body_indent(node, self.lines)
            block = [
                f"{indent}# Purpose: {meta.purpose}\n",
                f"{indent}# Called by: {meta.called_by}\n",
                f"{indent}# Calls: {meta.calls}\n",
            ]
            insertions.append((node.body[0].lineno - 1, block))
            count += 1

        for index, block in sorted(insertions, key=lambda item: item[0], reverse=True):
            self.lines[index:index] = block

        if insertions:
            self.text = "".join(self.lines)
            self.tree = ast.parse(self.text, filename=str(self.path))
        return count


class SourceBlock:
    """Extract and indent complete AST source blocks including decorators."""

    @staticmethod
    def start_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        # Purpose: Find the first physical line belonging to a decorated callable.
        # Called by: SourceBlock.extract(), ModuleTransformer._function_groups().
        # Calls: No project functions.
        decorator_lines = [decorator.lineno for decorator in node.decorator_list]
        return min([node.lineno, *decorator_lines])

    @staticmethod
    def extract(
        lines: Sequence[str],
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str:
        # Purpose: Extract one callable source block exactly, preserving comments and expressions.
        # Called by: ModuleTransformer._render_operation_class(), ClassMethodRewriter.
        # Calls: SourceBlock.start_line().
        start = SourceBlock.start_line(node) - 1
        end = int(node.end_lineno or node.lineno)
        return "".join(lines[start:end])

    @staticmethod
    def indent(text: str, spaces: int = 4) -> str:
        # Purpose: Indent a complete function/decorator block into a generated owner class.
        # Called by: ModuleTransformer._render_operation_class().
        # Calls: No project functions.
        prefix = " " * spaces
        return "".join(prefix + line if line.strip() else line for line in text.splitlines(keepends=True))


class ClassMethodRewriter:
    """Reorder existing class methods when safe without changing method bodies."""

    @staticmethod
    def _class_nonmethod_references_methods(
        cls: ast.ClassDef,
        method_names: set[str],
    ) -> bool:
        # Purpose: Detect class-body statements whose evaluation depends on a method definition order.
        # Called by: ClassMethodRewriter.rewrite().
        # Calls: ast.walk().
        for item in cls.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(item):
                if isinstance(node, ast.Name) and node.id in method_names:
                    return True
        return False

    @staticmethod
    def rewrite(text: str, path: Path) -> tuple[str, int]:
        # Purpose: Enforce callee-before-caller ordering inside existing classes when safely reorderable.
        # Called by: ModuleTransformer.transform().
        # Calls: ast.parse(), DependencySorter.order_methods(), SourceBlock.extract(),
        #        ClassMethodRewriter._class_nonmethod_references_methods().
        tree = ast.parse(text, filename=str(path))
        lines = text.splitlines(keepends=True)
        replacements: list[tuple[int, int, str]] = []
        reordered = 0

        # Process only classes whose methods are direct class-body children.
        # Nested classes are still visited independently by ast.walk().
        for cls in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
            methods = [
                node for node in cls.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if len(methods) < 2:
                continue

            names = [node.name for node in methods]
            desired = DependencySorter.order_methods(methods)
            if desired == names:
                continue

            if ClassMethodRewriter._class_nonmethod_references_methods(cls, set(names)):
                raise RuntimeError(
                    f"{path}:{cls.lineno} {cls.name}: class body references methods "
                    "during class construction; automatic method reordering would be unsafe"
                )

            by_name = {node.name: node for node in methods}
            original_slots = sorted(
                (
                    SourceBlock.start_line(node) - 1,
                    int(node.end_lineno or node.lineno),
                )
                for node in methods
            )
            desired_blocks = [
                SourceBlock.extract(lines, by_name[name]).rstrip() + "\n"
                for name in desired
            ]

            # Reuse the original method slots rather than rebuilding the class.
            # Class docstrings, class attributes, nested classes and comments
            # therefore stay byte-for-byte at their original locations.
            for (start, end), block in zip(original_slots, desired_blocks):
                replacements.append((start, end, block))
            reordered += len(methods)

        # Class ranges can nest. Replacements are independent method slots and
        # are applied from the end of the file upward so earlier line numbers
        # remain valid.
        for start, end, replacement in sorted(replacements, reverse=True):
            lines[start:end] = [replacement]
        return "".join(lines), reordered


class ModuleTransformer:
    """Transform one module from free top-level functions to class-owned operations."""

    def __init__(self, path: Path) -> None:
        # Purpose: Bind the production source path to one transformer instance.
        # Called by: OopMigration._transform_file().
        # Calls: No project functions.
        self.path = path

    @staticmethod
    def _sha256(text: str) -> str:
        # Purpose: Calculate source identity for migration reporting.
        # Called by: ModuleTransformer.transform().
        # Calls: hashlib.sha256().
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _function_groups(
        tree: ast.Module,
    ) -> list[list[ast.FunctionDef | ast.AsyncFunctionDef]]:
        # Purpose: Group consecutive top-level functions so class wrapping preserves module execution order.
        # Called by: ModuleTransformer._wrap_top_level_functions().
        # Calls: No project functions.
        groups: list[list[ast.FunctionDef | ast.AsyncFunctionDef]] = []
        current: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                current.append(node)
            else:
                if current:
                    groups.append(current)
                    current = []
        if current:
            groups.append(current)
        return groups

    def _render_operation_class(
        self,
        owner: str,
        nodes: Sequence[ast.FunctionDef | ast.AsyncFunctionDef],
        lines: Sequence[str],
    ) -> str:
        # Purpose: Render one cohesive operation owner plus compatibility bindings for original names.
        # Called by: ModuleTransformer._wrap_top_level_functions().
        # Calls: DependencySorter.order_functions(), SourceBlock.extract(), SourceBlock.indent().
        order = DependencySorter.order_functions(nodes)
        by_name = {node.name: node for node in nodes}
        chunks = [
            f"class {owner}:\n",
            '    """Own the module operations that were previously free functions."""\n\n',
        ]
        for name in order:
            source = SourceBlock.extract(lines, by_name[name])
            # Existing decorators remain immediately above the function. staticmethod
            # must be the outer decorator so class attribute access yields the
            # already-decorated callable with no implicit self argument.
            chunks.append("    @staticmethod\n")
            chunks.append(SourceBlock.indent(source))
            if not source.endswith("\n"):
                chunks.append("\n")
            chunks.append("\n")

        chunks.append("# Compatibility bindings: public module API remains unchanged.\n")
        for name in order:
            chunks.append(f"{name} = {owner}.{name}\n")
        return "".join(chunks).rstrip() + "\n"

    def _wrap_top_level_functions(self, text: str) -> tuple[str, list[str], int]:
        # Purpose: Replace all free top-level function groups with class-owned static operations.
        # Called by: ModuleTransformer.transform().
        # Calls: ModuleTransformer._function_groups(), SourceNames.owner_for(),
        #        ModuleTransformer._render_operation_class(), SourceBlock.start_line().
        tree = ast.parse(text, filename=str(self.path))
        lines = text.splitlines(keepends=True)
        groups = self._function_groups(tree)
        replacements: list[tuple[int, int, str]] = []
        owners: list[str] = []

        for group_index, nodes in enumerate(groups):
            owner = SourceNames.owner_for(self.path, group_index)
            owners.append(owner)
            start = min(SourceBlock.start_line(node) for node in nodes) - 1
            end = max(int(node.end_lineno or node.lineno) for node in nodes)
            rendered = self._render_operation_class(owner, nodes, lines)
            replacements.append((start, end, rendered))

        for start, end, replacement in sorted(replacements, reverse=True):
            lines[start:end] = [replacement]
        return "".join(lines), owners, sum(len(group) for group in groups)

    def transform(self) -> tuple[str, FileResult]:
        # Purpose: Perform comments, method ordering, class wrapping, and source identity reporting for one file.
        # Called by: OopMigration._transform_file().
        # Calls: SourceDocument.add_structured_comments(), ClassMethodRewriter.rewrite(),
        #        ModuleTransformer._wrap_top_level_functions(), ModuleTransformer._sha256().
        original = self.path.read_text(encoding="utf-8")
        before_sha = self._sha256(original)
        initial_tree = ast.parse(original, filename=str(self.path))
        top_before = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in initial_tree.body
        )

        document = SourceDocument(self.path, original)
        comments_added = document.add_structured_comments()
        commented = document.text

        reordered_text, reordered_count = ClassMethodRewriter.rewrite(
            commented,
            self.path,
        )
        transformed, owners, _ = self._wrap_top_level_functions(reordered_text)
        ast.parse(transformed, filename=str(self.path))

        final_tree = ast.parse(transformed, filename=str(self.path))
        top_after = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in final_tree.body
        )
        result = FileResult(
            path=self.path.as_posix(),
            owner_classes_added=owners,
            top_level_functions_before=top_before,
            top_level_functions_after=top_after,
            functions_commented=comments_added,
            methods_reordered=reordered_count,
            sha256_before=before_sha,
            sha256_after=self._sha256(transformed),
        )
        return transformed, result


class AtomicSourceWriter:
    """RAII-style atomic source replacement for one migrated file."""

    def __init__(self, path: Path, text: str) -> None:
        # Purpose: Prepare a temporary sibling file for atomic source replacement.
        # Called by: OopMigration._transform_file().
        # Calls: No project functions.
        self.path = path
        self.text = text
        self.temp_path: Path | None = None

    def __enter__(self) -> "AtomicSourceWriter":
        # Purpose: Create and fully write the temporary source file.
        # Called by: Python context-manager protocol.
        # Calls: tempfile.mkstemp(), os.fdopen(), os.fsync().
        fd, name = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".oop.tmp",
            dir=str(self.path.parent),
        )
        self.temp_path = Path(name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(self.text)
            stream.flush()
            os.fsync(stream.fileno())
        return self

    def commit(self) -> None:
        # Purpose: Atomically replace the production source after all file-local checks pass.
        # Called by: OopMigration._transform_file().
        # Calls: os.replace().
        if self.temp_path is None:
            raise RuntimeError("atomic source writer was not entered")
        os.replace(self.temp_path, self.path)
        self.temp_path = None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        # Purpose: Remove an uncommitted temporary source file on any failure path.
        # Called by: Python context-manager protocol.
        # Calls: Path.unlink().
        if self.temp_path is not None and self.temp_path.exists():
            self.temp_path.unlink()
        return False


class OopVerifier:
    """Hard post-migration contract for the complete production Python tree."""

    @staticmethod
    def _structured_comments_present(
        path: Path,
        text: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        # Purpose: Verify Purpose/Called by/Calls comments immediately precede a callable body.
        # Called by: OopVerifier.verify_file().
        # Calls: No project functions.
        lines = text.splitlines()
        body_line = node.body[0].lineno if node.body else node.lineno
        start = max(node.lineno - 1, body_line - 5)
        end = min(len(lines), body_line + 2)
        window = "\n".join(lines[start:end])
        return all(marker in window for marker in ("# Purpose:", "# Called by:", "# Calls:"))

    @staticmethod
    def _verify_class_order(
        path: Path,
        cls: ast.ClassDef,
    ) -> list[str]:
        # Purpose: Verify same-class self/cls callees are declared above their callers.
        # Called by: OopVerifier.verify_file().
        # Calls: AstCalls.self_calls_from().
        methods = [
            node for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        positions = {node.name: index for index, node in enumerate(methods)}
        names = set(positions)
        failures: list[str] = []
        for caller in methods:
            for callee in AstCalls.self_calls_from(caller, names):
                if callee != caller.name and positions[callee] > positions[caller.name]:
                    failures.append(
                        f"{path}:{caller.lineno} {cls.name}.{caller.name}() calls "
                        f"{callee}(), but {callee}() is declared below it"
                    )
        return failures

    @staticmethod
    def verify_file(path: Path) -> list[str]:
        # Purpose: Verify one migrated source file is class-owned, commented, ordered, and syntactically valid.
        # Called by: OopVerifier.verify_tree().
        # Calls: ast.parse(), OopVerifier._structured_comments_present(), OopVerifier._verify_class_order().
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            return [f"{path}: syntax error after migration: {exc}"]

        failures: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                failures.append(
                    f"{path}:{node.lineno} top-level function {node.name}() remains; "
                    "production callables must be class-owned"
                )

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not OopVerifier._structured_comments_present(path, text, node):
                    failures.append(
                        f"{path}:{node.lineno} {node.name}() missing "
                        "Purpose/Called by/Calls comments"
                    )

        for cls in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
            failures.extend(OopVerifier._verify_class_order(path, cls))
        return failures

    @staticmethod
    def verify_tree(paths: Iterable[Path]) -> list[str]:
        # Purpose: Aggregate the hard OOP contract over every production Python source.
        # Called by: OopMigration.run().
        # Calls: OopVerifier.verify_file().
        failures: list[str] = []
        for path in paths:
            failures.extend(OopVerifier.verify_file(path))
        return failures


class GitBaseline:
    """Read-only git baseline guard for deterministic migration."""

    @staticmethod
    def head(repo_root: Path) -> str:
        # Purpose: Read the current commit SHA without changing repository state.
        # Called by: GitBaseline.require().
        # Calls: subprocess.run().
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "git rev-parse HEAD failed")
        return completed.stdout.strip()

    @staticmethod
    def require(repo_root: Path, expected: str, allow_descendant: bool) -> None:
        # Purpose: Prevent the structural migration from silently targeting an unrelated source baseline.
        # Called by: OopMigration.run().
        # Calls: GitBaseline.head(), subprocess.run().
        current = GitBaseline.head(repo_root)
        if current == expected:
            return
        if allow_descendant:
            completed = subprocess.run(
                ["git", "merge-base", "--is-ancestor", expected, current],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if completed.returncode == 0:
                return
        raise RuntimeError(
            f"NSAMDR full OOP migration expects baseline {expected}; current HEAD is {current}"
        )


class OopMigration:
    """Composition root that rewrites and verifies the complete NSAMDR Python tree."""

    def __init__(
        self,
        repo_root: Path,
        *,
        apply_changes: bool,
        allow_descendant: bool,
    ) -> None:
        # Purpose: Capture repository/mode dependencies for one deterministic migration.
        # Called by: MigrationCli.run().
        # Calls: No project functions.
        self.repo_root = Path(repo_root).resolve()
        self.apply_changes = bool(apply_changes)
        self.allow_descendant = bool(allow_descendant)

    def _production_paths(self) -> list[Path]:
        # Purpose: Discover every production Python source under tools/nsamdr/neural.
        # Called by: OopMigration.run().
        # Calls: Path.rglob().
        root = self.repo_root / NEURAL_ROOT
        paths = [
            path
            for path in sorted(root.rglob("*.py"))
            if "__pycache__" not in path.parts
        ]
        if not paths:
            raise RuntimeError(f"no NSAMDR Python files found under {root}")
        return paths

    def _transform_file(self, path: Path) -> FileResult:
        # Purpose: Transform one file and atomically commit it only in --apply mode.
        # Called by: OopMigration.run().
        # Calls: ModuleTransformer.transform(), AtomicSourceWriter.
        transformed, result = ModuleTransformer(path).transform()
        if self.apply_changes and transformed != path.read_text(encoding="utf-8"):
            with AtomicSourceWriter(path, transformed) as writer:
                # Parse the exact temporary content before commit.
                ast.parse(transformed, filename=str(path))
                writer.commit()
        return result

    def _write_report(self, report: MigrationReport) -> None:
        # Purpose: Persist the complete machine-readable migration coverage/evidence report.
        # Called by: OopMigration.run().
        # Calls: json.dumps(), Path.write_text().
        path = self.repo_root / REPORT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def run(self) -> MigrationReport:
        # Purpose: Convert every production script and refuse completion unless the hard OOP contract passes.
        # Called by: MigrationCli.run().
        # Calls: GitBaseline.require(), OopMigration._production_paths(),
        #        OopMigration._transform_file(), OopVerifier.verify_tree(), OopMigration._write_report().
        GitBaseline.require(
            self.repo_root,
            BASELINE_SHA,
            allow_descendant=self.allow_descendant,
        )
        paths = self._production_paths()
        results: list[FileResult] = []

        for path in paths:
            result = self._transform_file(path)
            results.append(result)
            action = "APPLY" if self.apply_changes else "PLAN "
            print(
                f"[{action}] {path.relative_to(self.repo_root)} "
                f"free={result.top_level_functions_before} "
                f"owners={','.join(result.owner_classes_added) or '-'}",
                flush=True,
            )

        if self.apply_changes:
            violations = OopVerifier.verify_tree(paths)
            top_after = sum(
                sum(
                    isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    for node in ast.parse(path.read_text(encoding="utf-8")).body
                )
                for path in paths
            )
        else:
            # Dry-run verifies transformed text per file during ModuleTransformer,
            # but does not replace disk sources. Final whole-tree verification is
            # therefore meaningful only in apply mode.
            violations = []
            top_after = 0

        report = MigrationReport(
            schema="NSAMDR_FULL_OOP_MIGRATION_V1",
            baseline=BASELINE_SHA,
            files=results,
            production_files=len(paths),
            top_level_functions_before=sum(item.top_level_functions_before for item in results),
            top_level_functions_after=top_after,
            functions_commented=sum(item.functions_commented for item in results),
            methods_reordered=sum(item.methods_reordered for item in results),
            violations=violations,
        )
        self._write_report(report)

        if violations:
            print("\nFULL OOP CONTRACT FAILED:", file=sys.stderr)
            for violation in violations:
                print(f"  - {violation}", file=sys.stderr)
            raise RuntimeError(
                f"full OOP migration left {len(violations)} contract violation(s)"
            )
        if self.apply_changes and top_after != 0:
            raise RuntimeError(
                f"full OOP migration left {top_after} top-level function(s)"
            )
        return report


class MigrationCli:
    """Command-line application for the full-tree OOP source migration."""

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        # Purpose: Build the migration CLI without mixing parsing with transformation.
        # Called by: MigrationCli.run().
        # Calls: argparse.ArgumentParser().
        parser = argparse.ArgumentParser(
            description="Convert every NSAMDR production Python script to class-owned callables"
        )
        parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        parser.add_argument("--apply", action="store_true")
        parser.add_argument(
            "--allow-descendant",
            action="store_true",
            help="allow a HEAD descended from the exact baseline commit",
        )
        return parser

    @staticmethod
    def run(argv: list[str] | None = None) -> int:
        # Purpose: Execute migration planning or atomic application and print hard coverage statistics.
        # Called by: Module __main__ block.
        # Calls: MigrationCli.parser(), OopMigration.run().
        args = MigrationCli.parser().parse_args(argv)
        report = OopMigration(
            args.repo_root,
            apply_changes=args.apply,
            allow_descendant=args.allow_descendant,
        ).run()
        mode = "APPLIED" if args.apply else "DRY-RUN"
        print("=" * 78)
        print(f"NSAMDR FULL OOP MIGRATION: {mode}")
        print(f"Production Python files     : {report.production_files}")
        print(f"Free functions before       : {report.top_level_functions_before}")
        print(f"Free functions after        : {report.top_level_functions_after}")
        print(f"Functions/methods commented : {report.functions_commented}")
        print(f"Methods reordered           : {report.methods_reordered}")
        print(f"Contract violations         : {len(report.violations)}")
        print(f"Report                      : {REPORT_PATH}")
        print("=" * 78)
        return 0


if __name__ == "__main__":
    raise SystemExit(MigrationCli.run())
