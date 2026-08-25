"""Smoke and routing tests for the cleaned canonical NSAMDR command dispatcher."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[3]
CLI_PATH = REPO_ROOT / "tools" / "nsamdr" / "nsamdr_cli.py"

_SPEC = importlib.util.spec_from_file_location("nsamdr_cli_under_test", CLI_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Unable to load NSAMDR dispatcher from {CLI_PATH}")
CLI = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CLI)


PUBLIC_COMMAND_PATHS = {
    ("gui",),
    ("setup",),
    ("raven-quick",),
    ("full-train",),
    ("index",),
    ("index", "eve"),
    ("index", "raven"),
    ("preview",),
    ("validate",),
    ("test",),
    ("test", "contract"),
    ("test", "architecture"),
    ("test", "checkpoint"),
    ("cleanup",),
    ("integrate",),
    ("native",),
    ("native", "build"),
    ("native", "obj"),
    ("native", "eve"),
}

MINIMAL_LEAF_ARGV = {
    ("gui",): ["gui"],
    ("setup",): ["setup", "cuda"],
    ("raven-quick",): ["raven-quick"],
    ("full-train",): ["full-train"],
    ("index", "eve"): ["index", "eve"],
    ("index", "raven"): ["index", "raven"],
    ("preview",): ["preview", "EXP_0001"],
    ("validate",): ["validate", "--layout-only"],
    ("test", "contract"): ["test", "contract"],
    ("test", "architecture"): ["test", "architecture", "--device", "cpu"],
    ("test", "checkpoint"): ["test", "checkpoint"],
    ("cleanup",): ["cleanup", "--dry-run"],
    ("integrate",): ["integrate", "--check"],
    ("native", "build"): ["native", "build"],
    ("native", "obj"): ["native", "obj"],
    ("native", "eve"): ["native", "eve"],
}


def _subparser_choices(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:  # argparse exposes no public subparser iterator
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _registered_paths(parser: argparse.ArgumentParser) -> set[tuple[str, ...]]:
    result: set[tuple[str, ...]] = set()

    def visit(current: argparse.ArgumentParser, prefix: tuple[str, ...]) -> None:
        for name, child in _subparser_choices(current).items():
            path = (*prefix, name)
            result.add(path)
            visit(child, path)

    visit(parser, ())
    return result


def _leaf_paths(parser: argparse.ArgumentParser) -> set[tuple[str, ...]]:
    result: set[tuple[str, ...]] = set()

    def visit(current: argparse.ArgumentParser, prefix: tuple[str, ...]) -> None:
        children = _subparser_choices(current)
        if prefix and not children:
            result.add(prefix)
        for name, child in children.items():
            visit(child, (*prefix, name))

    visit(parser, ())
    return result


class ParserSmokeTests(unittest.TestCase):
    def test_public_command_inventory_matches_cleaned_parser(self) -> None:
        self.assertEqual(PUBLIC_COMMAND_PATHS, _registered_paths(CLI.build_parser()))

    def test_every_leaf_parses_without_launching_work(self) -> None:
        parser = CLI.build_parser()
        self.assertEqual(set(MINIMAL_LEAF_ARGV), _leaf_paths(parser))
        for path, argv in sorted(MINIMAL_LEAF_ARGV.items()):
            with self.subTest(command=" ".join(path)):
                namespace = parser.parse_args(argv)
                self.assertTrue(callable(namespace.handler))

    def test_every_public_command_path_prints_help(self) -> None:
        parser = CLI.build_parser()
        for path in [(), *sorted(PUBLIC_COMMAND_PATHS)]:
            with self.subTest(command=" ".join(path) or "<root>"):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    with self.assertRaises(SystemExit) as raised:
                        parser.parse_args([*path, "--help"])
                self.assertEqual(0, raised.exception.code)
                self.assertIn("usage:", output.getvalue().lower())

    def test_script_help_works_outside_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            completed = subprocess.run(
                [sys.executable, os.fspath(CLI_PATH), "--help"],
                cwd=temporary_directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertIn("usage:", completed.stdout.lower())


class DispatcherRoutingTests(unittest.TestCase):
    def test_raven_quick_is_only_quick_workflow_mode(self) -> None:
        with mock.patch.object(CLI, "_command_workflow", return_value=73) as backend:
            result = CLI.main(["raven-quick"])
        self.assertEqual(73, result)
        backend.assert_called_once()
        self.assertEqual("quick", backend.call_args.args[1])

    def test_full_train_is_only_full_workflow_mode(self) -> None:
        with mock.patch.object(CLI, "_command_workflow", return_value=74) as backend:
            result = CLI.main(["full-train"])
        self.assertEqual(74, result)
        backend.assert_called_once()
        self.assertEqual("full", backend.call_args.args[1])

    def test_preview_routes_only_to_qualified_experiment_previewer(self) -> None:
        with mock.patch.object(CLI, "_python_script", return_value=29) as backend:
            result = CLI.main(["preview", "EXP_0042", "--target-size", "2048", "--device", "cpu"])
        self.assertEqual(29, result)
        backend.assert_called_once_with(
            "tools/nsamdr/neural/preview_nsamdr_v9_experiment.py",
            [
                "--repo-root",
                os.fspath(CLI.REPO_ROOT),
                "--experiment",
                "EXP_0042",
                "--target-size",
                "2048",
                "--device",
                "cpu",
            ],
        )

    def test_contract_route_uses_canonical_contract_script(self) -> None:
        with mock.patch.object(CLI, "_python_script", return_value=19) as backend:
            result = CLI.main(["test", "contract"])
        self.assertEqual(19, result)
        backend.assert_called_once_with("tools/nsamdr/neural/test_nsamdr_v9_contract.py", [])

    def test_validate_layout_only_does_not_launch_training(self) -> None:
        with mock.patch.object(CLI, "validate_layout", return_value=0) as validate:
            with mock.patch.object(CLI, "_command_test") as tests:
                result = CLI.main(["validate", "--layout-only"])
        self.assertEqual(0, result)
        validate.assert_called_once_with()
        tests.assert_not_called()

    def test_cleanup_rejects_unknown_options_before_deleting(self) -> None:
        with mock.patch.object(CLI.shutil, "rmtree") as remove_tree:
            result = CLI.main(["cleanup", "--definitely-not-valid"])
        self.assertEqual(2, result)
        remove_tree.assert_not_called()


if __name__ == "__main__":
    unittest.main()
