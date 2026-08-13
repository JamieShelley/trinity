"""Prevent removed NSAMDR launchers and tooling from returning."""
from __future__ import annotations

import os
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_FILE = Path(__file__).resolve()

RETAINED_BATS = {
    "nsamdr.bat",
    "run_nsamdr_obj_preview_dx11.bat",
    "run_nsamdr_v9_gui.bat",
    "setup_nsamdr_cpu.bat",
    "setup_nsamdr_cuda.bat",
}

# This is the authoritative migration set: 34 historical BAT files removed
# while the four specialized launchers survive and nsamdr.bat is added.
PROHIBITED_BATS = {
    "apply_nsamdr_trinity_settings.bat",
    "build_nsamdr_obj_preview_dx11.bat",
    "cleanup_nsamdr_intermediate_modes.bat",
    "cleanup_nsamdr_legacy.bat",
    "cleanup_nsamdr_v9.bat",
    "compare_nsamdr_v9_experiments.bat",
    "generate_nsamdr_strategy_candidates.bat",
    "index_nsamdr_eve_cache.bat",
    "index_nsamdr_v9_eve_cache.bat",
    "index_nsamdr_v9_raven_preview.bat",
    "preview_nsamdr_v9_experiment.bat",
    "promote_nsamdr_v9_experiment.bat",
    "retrain_nsamdr_and_preview.bat",
    "retrain_nsamdr_v9_and_preview.bat",
    "run_nsamdr_eve_asset_dx11.bat",
    "run_nsamdr_v7_all.bat",
    "run_nsamdr_v9_all.bat",
    "run_nsamdr_v9_pilot.bat",
    "run_nsamdr_v9_raven_tune_preview.bat",
    "test_nsamdr.bat",
    "test_nsamdr_real_eve_asset.bat",
    "test_nsamdr_stability_checkpoint.bat",
    "test_nsamdr_v6_architecture.bat",
    "test_nsamdr_v7_architecture.bat",
    "test_nsamdr_v9_architecture.bat",
    "test_nsamdr_v9_checkpoint.bat",
    "test_nsamdr_v9_contract.bat",
    "train_nsamdr.bat",
    "train_nsamdr_quality_cuda.bat",
    "train_nsamdr_v9_preview_experiment.bat",
    "train_nsamdr_v9_quality_cuda.bat",
    "validate_nsamdr_v9_pipeline.bat",
    "verify_and_clean_nsamdr_layout.bat",
    "verify_nsamdr_v9_layout.bat",
}

PROHIBITED_NON_BAT_PATHS = {
    "scripts/build/nsamdr_v8_workflow_gui_typed_controls.py",
    "scripts/build/nsamdr/cleanup_nsamdr_legacy.ps1",
    "scripts/build/nsamdr/cleanup_nsamdr_v9.ps1",
    "scripts/build/nsamdr/NSAMDRCMakePresets.json",
    "scripts/build/nsamdr/NSAMDRRepositoryRoot.CMakeLists.txt",
    "scripts/build/nsamdr/repair_nsamdr_cmake_presets.ps1",
    "scripts/build/nsamdr/repair_nsamdr_root_cmake.ps1",
    "scripts/build/nsamdr/RepairMissingTrinityALMarker.ps1",
    "tools/nsamdr/integration/NSAMDR_TRINITY_SETTINGS.patch",
    "tools/nsamdr/neural/default_training_config.json",
    "tools/nsamdr/neural/index_eve_texture_dataset.py",
    "tools/nsamdr/neural/smoke_test_nsamdr_v6.py",
    "tools/nsamdr/neural/smoke_test_nsamdr_v7.py",
    "tools/nsamdr/neural/stability_training_config.json",
    "tools/nsamdr/neural/test_nsamdr_kernel.py",
    "tools/nsamdr/neural/test_nsamdr_numerical_safety.py",
    "tools/nsamdr/neural/train_nsamdr_kernel.py",
    "tools/nsamdr/neural/v7",
    "tools/nsamdr/neural/v7/__init__.py",
    "tools/nsamdr/neural/v7/config.py",
    "tools/nsamdr/neural/v7/dataset.py",
    "tools/nsamdr/neural/v7/inference.py",
    "tools/nsamdr/neural/v7/losses.py",
    "tools/nsamdr/neural/v7/model.py",
    "tools/nsamdr/neural/v7/training.py",
    "trinity/tools/nsamdr",
    "trinity/tools/nsamdr/NSAMDRRealShipViewer.cpp",
    "trinity/tools/nsamdr/NSAMDRRealShipViewer.h",
    "trinity/tools/nsamdr/NSAMDRRealShipViewerMain.cpp",
    "trinity/tools/nsamdr/README.md",
}

ACTIVE_EXTENSIONS = {
    ".bat",
    ".c",
    ".cc",
    ".cfg",
    ".cmake",
    ".cmd",
    ".cpp",
    ".cs",
    ".cxx",
    ".h",
    ".hlsl",
    ".hpp",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
ACTIVE_FILENAMES = {"cmakelists.txt", "makefile"}
PRUNED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "artifacts",
    "dist",
    "node_modules",
    "out",
    "vendor",
    "venv",
}


def _is_active_text(path: Path) -> bool:
    return path.name.casefold() in ACTIVE_FILENAMES or path.suffix.casefold() in ACTIVE_EXTENSIONS


def _active_files():
    for directory, directory_names, file_names in os.walk(REPO_ROOT):
        current = Path(directory)
        current_relative = current.relative_to(REPO_ROOT)
        kept_directories = []
        for name in directory_names:
            folded = name.casefold()
            if folded in PRUNED_DIRECTORY_NAMES or folded.startswith(".cmake-build"):
                continue
            # scripts/build is source; other directories literally named build
            # are generated output and are intentionally outside this scan.
            if folded == "build" and current_relative != Path("scripts"):
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in file_names:
            path = current / name
            if path.resolve() != THIS_FILE and _is_active_text(path):
                yield path


def _reference_tokens() -> set[str]:
    tokens = {name.casefold() for name in PROHIBITED_BATS}
    for relative in PROHIBITED_NON_BAT_PATHS:
        normalized = relative.casefold().replace("\\", "/")
        tokens.add(normalized)
        basename = Path(relative).name.casefold()
        if (
            "/v7/" not in normalized
            and not normalized.endswith("/v7")
            and basename not in {"nsamdr", "readme.md"}
        ):
            tokens.add(basename)
    tokens.update(
        {
            "tools.nsamdr.neural.v7",
            "tools/nsamdr/neural/v7",
        }
    )
    return tokens


class LegacyReferenceTests(unittest.TestCase):
    def test_migration_inventory_is_complete_and_disjoint(self) -> None:
        self.assertEqual(34, len(PROHIBITED_BATS))
        self.assertTrue(RETAINED_BATS.isdisjoint(PROHIBITED_BATS))

    def test_prohibited_legacy_files_are_absent(self) -> None:
        prohibited_paths = {
            *(f"scripts/build/{name}" for name in PROHIBITED_BATS),
            *PROHIBITED_NON_BAT_PATHS,
        }
        present = sorted(relative for relative in prohibited_paths if (REPO_ROOT / relative).exists())
        self.assertEqual([], present, "Prohibited legacy NSAMDR paths still exist:\n" + "\n".join(present))

    def test_active_code_and_docs_do_not_reference_legacy_names(self) -> None:
        tokens = _reference_tokens()
        violations = []
        for path in _active_files():
            relative = path.relative_to(REPO_ROOT).as_posix()
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as exc:
                self.fail(f"Unable to scan {relative}: {exc}")
            for line_number, line in enumerate(lines, 1):
                normalized_line = line.casefold().replace("\\", "/")
                matches = sorted(token for token in tokens if token in normalized_line)
                if matches:
                    violations.append(f"{relative}:{line_number}: {', '.join(matches)}")

        self.assertEqual(
            [],
            violations,
            "Active references to removed NSAMDR tooling remain:\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
