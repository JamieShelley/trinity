from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MICRO = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_overfit.py"
GUI = ROOT / "tools/nsamdr/gui/nsamdr_v9_workflow_gui_micro.py"
LAUNCHER = ROOT / "scripts/build/nsamdr.bat"


def test_micro_capacity_mode_is_non_promotable_and_uses_production_schedule():
    source = MICRO.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert tree is not None
    assert 'PRODUCTION_CONFIG = "tools/nsamdr/neural/configs/v9_fidelity_full.json"' in source
    assert 'production.synthetic_geometry_probability = 0.0' in source
    assert '"selectionKind": "diagnostic-non-promotable"' in source
    assert '"promotable": False' in source
    assert '"qualifiedProductionCheckpoint": False' in source
    assert 'model = FidelityResidualNetV9(config).to(device)' in source
    assert 'losses = training.compute_losses(outputs, moved, config, phase)' in source


def test_micro_defaults_to_known_valid_32_lr_128_hr_patch():
    source = MICRO.read_text(encoding="utf-8")
    assert 'parser.add_argument("--tile-size", type=int, default=32)' in source
    assert 'production.tile_size = int(args.tile_size)' in source
    assert 'production.target_scale' not in source
    assert '--tile-size must be >=32 and divisible by 16' in source


def test_gui_places_micro_capacity_immediately_before_preview():
    source = GUI.read_text(encoding="utf-8")
    assert '"micro",\n    "3",\n    "Raven Micro Capacity"' in source
    assert '"preview",\n    "4",\n    "Preview"' in source
    assert '_micro,\n    _preview,' in source
    assert '("raven-micro",)' in source
    assert 'DIAGNOSTIC ONLY' in source


def test_standard_windows_gui_launcher_uses_micro_extension():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert 'if /I "%~1"=="gui"' in launcher
    assert 'nsamdr_v9_workflow_gui_micro.py' in launcher
