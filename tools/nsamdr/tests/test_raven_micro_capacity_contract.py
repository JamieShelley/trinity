from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MICRO = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic.py"
LEGACY_MICRO = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_overfit.py"
GUI = ROOT / "tools/nsamdr/gui/nsamdr_v9_workflow_gui_micro.py"
LAUNCHER = ROOT / "scripts/build/nsamdr.bat"


def test_micro_capacity_mode_is_non_promotable_and_uses_production_schedule():
    source = MICRO.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert tree is not None
    assert 'import run_nsamdr_v9_raven_micro_overfit as legacy' in source
    assert '"selectionKind": "diagnostic-non-promotable"' in source
    assert '"promotable": False' in source
    assert '"qualifiedProductionCheckpoint": False' in source
    assert 'model = FidelityResidualNetV9(config).to(device)' in source
    assert 'losses = training.compute_losses(outputs, moved, config, phase)' in source
    assert 'config.training_activation_checkpointing = False' in source


def test_micro_exposes_raw_and_final_authority_candidates():
    source = MICRO.read_text(encoding="utf-8")
    for token in (
        '"rawBoundary", "boundary_initial_candidate_albedo"',
        '"preSeam", "boundary_pre_seam_albedo"',
        '"detailCandidate", "detail_candidate_albedo"',
        '"final", "albedo"',
        '"finalSelectorGateMean"',
        '"rawDetailCapacityPass"',
        '"productionAuthorityPass"',
    ):
        assert token in source


def test_micro_defaults_to_valid_32_lr_patch_and_shorter_probe_budget():
    source = MICRO.read_text(encoding="utf-8")
    legacy_source = LEGACY_MICRO.read_text(encoding="utf-8")
    assert 'parser.add_argument("--tile-size", type=int, default=32)' in source
    assert 'parser.add_argument("--steps-per-epoch", type=int, default=32)' in source
    assert '--tile-size must be >=32 and divisible by 16' in source
    assert 'production.tile_size = int(args.tile_size)' in legacy_source
    assert 'production.target_scale' not in legacy_source


def test_gui_places_micro_capacity_as_first_neural_diagnostic():
    source = GUI.read_text(encoding="utf-8")
    assert '"micro",\n    "1",\n    "Raven Micro Capacity"' in source
    assert '_quick = _renumber(_existing["quick"], "2")' in source
    assert '_train = _renumber(_existing["train"], "3")' in source
    assert '"preview",\n    "4",\n    "Preview"' in source
    assert '_existing["setup"],\n    _micro,\n    _quick,\n    _train,\n    _preview,' in source
    assert 'run_nsamdr_v9_raven_micro_diagnostic.py' in source
    assert 'DIAGNOSTIC ONLY' in source


def test_standard_windows_gui_launcher_uses_micro_extension():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert 'if /I "%~1"=="gui"' in launcher
    assert 'nsamdr_v9_workflow_gui_micro.py' in launcher
