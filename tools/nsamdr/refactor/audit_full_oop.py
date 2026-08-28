#!/usr/bin/env python3
"""Independent hard audit for the fully class-owned NSAMDR production tree."""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re
import sys
from typing import Iterable


NEURAL_ROOT = Path("tools/nsamdr/neural")
REQUIRED_COMMENTS = ("# Purpose:", "# Called by:", "# Calls:")


@dataclass
class AuditResult:
    """Summary of one complete production-tree OOP audit."""

    files: int
    classes: int
    methods: int
    top_level_functions: int
    violations: list[str]


class OopSourceAudit:
    """Read-only auditor for class ownership, comments, and dependency ordering."""

    @staticmethod
    def _dotted_name(node: ast.AST) -> str:
        # Purpose: Render a call expression name for dependency analysis.
        # Called by: OopSourceAudit._self_calls().
        # Calls: OopSourceAudit._dotted_name() recursively.
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = OopSourceAudit._dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    @staticmethod
    def _self_calls(node: ast.AST, names: set[str]) -> set[str]:
        # Purpose: Collect direct self/cls calls targeting methods in the same class.
        # Called by: OopSourceAudit._class_order_violations().
        # Calls: ast.walk().
        result: set[str] = set()
        for item in ast.walk(node):
            if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Attribute):
                continue
            owner = item.func.value
            if (
                isinstance(owner, ast.Name)
                and owner.id in {"self", "cls"}
                and item.func.attr in names
            ):
                result.add(item.func.attr)
        return result

    @staticmethod
    def _comment_window(
        lines: list[str],
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> str:
        # Purpose: Read the source region where structured callable comments must appear.
        # Called by: OopSourceAudit._file_violations().
        # Calls: No project functions.
        first_body = node.body[0].lineno if node.body else node.lineno
        start = max(node.lineno - 1, first_body - 5)
        end = min(len(lines), first_body + 2)
        return "\n".join(lines[start:end])

    @staticmethod
    def _class_order_violations(path: Path, cls: ast.ClassDef) -> list[str]:
        # Purpose: Enforce callee-before-caller order for methods in one class.
        # Called by: OopSourceAudit._file_violations().
        # Calls: OopSourceAudit._self_calls().
        methods = [
            node for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        positions = {node.name: index for index, node in enumerate(methods)}
        names = set(positions)
        failures: list[str] = []
        for caller in methods:
            for callee in OopSourceAudit._self_calls(caller, names):
                if callee == caller.name:
                    continue
                if positions[callee] > positions[caller.name]:
                    failures.append(
                        f"{path}:{caller.lineno}: {cls.name}.{caller.name}() "
                        f"calls later method {callee}()"
                    )
        return failures

    @staticmethod
    def _file_violations(path: Path) -> tuple[list[str], int, int, int]:
        # Purpose: Audit one production module and return violations plus structural counts.
        # Called by: OopSourceAudit.run().
        # Calls: ast.parse(), OopSourceAudit._comment_window(), OopSourceAudit._class_order_violations().
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        tree = ast.parse(text, filename=str(path))
        failures: list[str] = []
        top = 0
        classes = 0
        methods = 0

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                top += 1
                failures.append(
                    f"{path}:{node.lineno}: top-level function {node.name}() remains"
                )

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes += 1
                failures.extend(OopSourceAudit._class_order_violations(path, node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods += 1
                window = OopSourceAudit._comment_window(lines, node)
                for marker in REQUIRED_COMMENTS:
                    if marker not in window:
                        failures.append(
                            f"{path}:{node.lineno}: {node.name}() missing {marker}"
                        )
        return failures, classes, methods, top

    @staticmethod
    def _paths(repo_root: Path) -> list[Path]:
        # Purpose: Discover all production Python scripts covered by the hard audit.
        # Called by: OopSourceAudit.run().
        # Calls: Path.rglob().
        return [
            path
            for path in sorted((repo_root / NEURAL_ROOT).rglob("*.py"))
            if "__pycache__" not in path.parts
        ]

    @staticmethod
    def run(repo_root: Path) -> AuditResult:
        # Purpose: Audit the complete NSAMDR neural Python tree and aggregate exact coverage.
        # Called by: AuditCli.run().
        # Calls: OopSourceAudit._paths(), OopSourceAudit._file_violations().
        failures: list[str] = []
        classes = methods = top = 0
        paths = OopSourceAudit._paths(repo_root)
        for path in paths:
            file_failures, file_classes, file_methods, file_top = (
                OopSourceAudit._file_violations(path)
            )
            failures.extend(file_failures)
            classes += file_classes
            methods += file_methods
            top += file_top
        return AuditResult(
            files=len(paths),
            classes=classes,
            methods=methods,
            top_level_functions=top,
            violations=failures,
        )


class AuditCli:
    """Command-line shell for the independent OOP source audit."""

    @staticmethod
    def parser() -> argparse.ArgumentParser:
        # Purpose: Build the independent OOP audit command-line parser.
        # Called by: AuditCli.run().
        # Calls: argparse.ArgumentParser().
        parser = argparse.ArgumentParser()
        parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        parser.add_argument("--json", type=Path)
        return parser

    @staticmethod
    def run(argv: list[str] | None = None) -> int:
        # Purpose: Run the independent audit, optionally persist JSON, and fail on any violation.
        # Called by: Module __main__ block.
        # Calls: AuditCli.parser(), OopSourceAudit.run(), json.dumps().
        args = AuditCli.parser().parse_args(argv)
        result = OopSourceAudit.run(args.repo_root.resolve())
        if args.json is not None:
            path = (
                args.json
                if args.json.is_absolute()
                else args.repo_root.resolve() / args.json
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        print("=" * 72)
        print("NSAMDR FULL OOP SOURCE AUDIT")
        print(f"Python files             : {result.files}")
        print(f"Classes                  : {result.classes}")
        print(f"Functions/methods        : {result.methods}")
        print(f"Top-level functions      : {result.top_level_functions}")
        print(f"Violations               : {len(result.violations)}")
        print("=" * 72)
        if result.violations:
            for violation in result.violations:
                print(f"FAIL: {violation}")
            return 2
        print("PASS: every production callable is class-owned/commented/ordered.")
        return 0


if __name__ == "__main__":
    raise SystemExit(AuditCli.run())
