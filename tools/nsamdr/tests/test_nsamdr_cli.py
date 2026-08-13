"""Smoke and routing tests for the unified NSAMDR command dispatcher."""
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
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - importlib contract guard
    raise RuntimeError(f"Unable to load NSAMDR dispatcher from {CLI_PATH}")
CLI = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CLI)


# Include both command groups and executable leaves. This inventory deliberately
# makes adding a route require adding parse/help coverage in the same change.
PUBLIC_COMMAND_PATHS = {
    ("gui",),
    ("setup",),
    ("tune",),
    ("index",),
    ("index", "eve"),
    ("index", "raven"),
    ("train",),
    ("train", "preview"),
    ("train", "full"),
    ("preview",),
    ("candidate",),
    ("compare",),
    ("promote",),
    ("validate",),
    ("test",),
    ("test", "contract"),
    ("test", "architecture"),
    ("test", "checkpoint"),
    ("cleanup",),
    ("integrate",),
    ("run",),
    ("retrain-preview",),
    ("native",),
    ("native", "build"),
    ("native", "obj"),
    ("native", "eve"),
}

MINIMAL_LEAF_ARGV = {
    ("gui",): ["gui"],
    ("setup",): ["setup", "cuda"],
    ("tune",): ["tune"],
    ("index", "eve"): ["index", "eve"],
    ("index", "raven"): ["index", "raven"],
    ("train", "preview"): ["train", "preview"],
    ("train", "full"): ["train", "full"],
    ("preview",): ["preview", "EXP_0001"],
    ("candidate",): ["candidate"],
    ("compare",): ["compare", "EXP_0001", "EXP_0002"],
    ("promote",): ["promote", "EXP_0001"],
    ("validate",): ["validate", "--layout-only"],
    ("test", "contract"): ["test", "contract"],
    ("test", "architecture"): ["test", "architecture", "--device", "cpu"],
    ("test", "checkpoint"): ["test", "checkpoint"],
    ("cleanup",): ["cleanup", "--dry-run"],
    ("integrate",): ["integrate", "--check"],
    ("run",): ["run"],
    ("retrain-preview",): ["retrain-preview"],
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
    paths: set[tuple[str, ...]] = set()

    def visit(current: argparse.ArgumentParser, prefix: tuple[str, ...]) -> None:
        for name, child in _subparser_choices(current).items():
            path = (*prefix, name)
            paths.add(path)
            visit(child, path)

    visit(parser, ())
    return paths


def _leaf_paths(parser: argparse.ArgumentParser) -> set[tuple[str, ...]]:
    paths: set[tuple[str, ...]] = set()

    def visit(current: argparse.ArgumentParser, prefix: tuple[str, ...]) -> None:
        children = _subparser_choices(current)
        if prefix and not children:
            paths.add(prefix)
        for name, child in children.items():
            visit(child, (*prefix, name))

    visit(parser, ())
    return paths


class ParserSmokeTests(unittest.TestCase):
    def test_public_command_inventory_matches_parser(self) -> None:
        self.assertEqual(PUBLIC_COMMAND_PATHS, _registered_paths(CLI.build_parser()))

    def test_every_leaf_parses_without_launching_a_workload(self) -> None:
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

    def test_script_help_works_from_outside_the_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            for path in [(), *sorted(PUBLIC_COMMAND_PATHS)]:
                with self.subTest(command=" ".join(path) or "<root>"):
                    completed = subprocess.run(
                        [sys.executable, os.fspath(CLI_PATH), *path, "--help"],
                        cwd=temporary_directory,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    self.assertEqual(
                        0,
                        completed.returncode,
                        msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
                    )
                    self.assertIn("usage:", completed.stdout.lower())


class DispatcherRoutingTests(unittest.TestCase):
    def test_index_eve_forwards_exact_arguments_and_exit_code(self) -> None:
        with mock.patch.object(CLI, "_python_script", return_value=73) as backend:
            result = CLI.main(
                [
                    "index",
                    "eve",
                    "--config",
                    "config with spaces.json",
                    "--source-root",
                    r"D:\EVE authored textures",
                    "--rebuild",
                    "--max-families",
                    "17",
                    "--crops-per-family",
                    "4",
                ]
            )

        self.assertEqual(73, result)
        backend.assert_called_once_with(
            "tools/nsamdr/neural/index_eve_texture_dataset_v9.py",
            [
                "--repo-root",
                os.fspath(CLI.REPO_ROOT),
                "--config",
                "config with spaces.json",
                "--source-root",
                r"D:\EVE authored textures",
                "--rebuild",
                "--max-families",
                "17",
                "--crops-per-family",
                "4",
            ],
        )

    def test_candidate_forwards_exact_arguments_and_exit_code(self) -> None:
        with mock.patch.object(CLI, "_python_script", return_value=41) as backend:
            result = CLI.main(
                [
                    "candidate",
                    "raven",
                    "--target-size",
                    "2048",
                    "--super-resolution-backend",
                    "classic",
                    "--inference-device",
                    "cpu",
                    "--install-dependencies",
                    "--force",
                ]
            )

        asset_dir = CLI.REPO_ROOT / "artifacts" / "nsamdr" / "eve_assets" / "raven"
        self.assertEqual(41, result)
        backend.assert_called_once_with(
            "tools/nsamdr/generate_strategy_candidates.py",
            [
                "--obj",
                os.fspath(asset_dir / "raven.obj"),
                "--materials",
                os.fspath(asset_dir / "ship.materials.tsv"),
                "--asset-manifest",
                os.fspath(asset_dir / "asset_manifest.json"),
                "--output-root",
                os.fspath(asset_dir / "strategy_candidates_2048"),
                "--target-size",
                "2048",
                "--super-resolution-backend",
                "classic",
                "--inference-device",
                "cpu",
                "--install-dependencies",
                "--force",
            ],
        )

    def test_experiment_preview_alias_forwards_exact_arguments_and_exit_code(self) -> None:
        with mock.patch.object(CLI, "_python_script", return_value=29) as backend:
            result = CLI.main(["preview", "experiment", "EXP_0042"])

        self.assertEqual(29, result)
        backend.assert_called_once_with(
            "tools/nsamdr/neural/preview_nsamdr_v9_experiment.py",
            ["--repo-root", os.fspath(CLI.REPO_ROOT), "--experiment", "EXP_0042"],
        )

    def test_contract_route_propagates_backend_exit_code(self) -> None:
        with mock.patch.object(CLI, "_python_script", return_value=19) as backend:
            result = CLI.main(["test", "contract"])

        self.assertEqual(19, result)
        backend.assert_called_once_with(
            "tools/nsamdr/neural/test_nsamdr_v9_contract.py",
            [],
        )

    def test_forwarded_options_before_positionals_are_not_consumed_as_ids(self) -> None:
        with mock.patch.object(CLI, "_python_script", return_value=0) as backend:
            result = CLI.main(
                [
                    "preview",
                    "--target-size",
                    "2048",
                    "--device",
                    "cpu",
                    "experiment",
                    "EXP_0042",
                ]
            )

        self.assertEqual(0, result)
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

    def test_cleanup_rejects_unknown_options_before_deleting(self) -> None:
        with mock.patch.object(CLI.shutil, "rmtree") as remove_tree:
            result = CLI.main(["cleanup", "--definitely-not-valid"])

        self.assertEqual(2, result)
        remove_tree.assert_not_called()


if __name__ == "__main__":
    unittest.main()
