from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DIRECT = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_direct_residual_diagnostic.py"
MICRO = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_diagnostic.py"
LEGACY_MICRO = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_micro_overfit.py"
GUI = ROOT / "tools/nsamdr/gui/nsamdr_v9_workflow_gui_micro.py"
LAUNCHER = ROOT / "scripts/build/nsamdr.bat"


def test_direct_residual_capacity_proof_is_isolated_and_non_promotable():
    source = DIRECT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert tree is not None
    assert 'REPORT_SCHEMA = "NSAMDR_RAVEN_DIRECT_RESIDUAL_CAPACITY_V2"' in source
    assert 'model.detail_net(' in source
    assert 'geometry = torch.zeros(' in source
    assert 'detail_albedo_max_delta' in source
    assert '"promotable": False' in source
    assert '"qualifiedProductionCheckpoint": False' in source
    assert '"capacityProofPass": bool(capacity_pass)' in source
    assert 'parser.add_argument("--steps", type=int, default=1536)' in source
    assert 'parser.add_argument("--learning-rate", type=float, default=1.0e-3)' in source
    assert 'parser.add_argument("--head-lr-multiplier", type=float, default=3.0)' in source
    assert 'parser.add_argument("--required-edge-recovery", type=float, default=0.50)' in source
    assert 'parser.add_argument("--required-global-recovery", type=float, default=0.25)' in source
    # The proof deliberately bypasses the serial production prerequisites.
    assert 'model.geometry_net(' not in source
    assert 'model.seam_restorer(' not in source
    assert 'model.benefit_selector(' not in source


def test_direct_residual_uses_exact_production_baseline_and_decisive_cap_oracle():
    source = DIRECT.read_text(encoding="utf-8")
    assert 'mode="bicubic"' in source
    assert 'antialias=True' in source
    assert 'baseline_albedo.float() + delta' in source
    assert 'model._normalize_xy(' in source
    assert 'mode="nearest"' in source
    assert 'def _oracle_for_cap(' in source
    assert 'CAP_LADDER = ' in source
    assert '"productionCapOracle": production_oracle' in source
    assert '"trainingCapOracle": training_oracle' in source
    assert '"oracleCapSweep": oracle_sweep' in source
    assert '"productionCapFeasible": bool(production_cap_feasible)' in source
    assert '"trainingResidualCap": training_cap' in source
    assert 'fractionPixelsBeyondCap' in source


def test_direct_residual_removes_zero_head_dead_start_and_supervises_exact_residual():
    source = DIRECT.read_text(encoding="utf-8")
    assert 'model.detail_net.albedo_head.parameters()' in source
    assert 'float(args.learning_rate) * float(args.head_lr_multiplier)' in source
    assert 'desired_residual = (' in source
    assert 'residual_supervision = (raw_delta - desired_residual).abs().mean()' in source
    assert 'residual_supervision * 8.0' in source
    assert 'global_reconstruction * 4.0' in source
    assert 'edge_reconstruction * 12.0' in source
    assert 'gradient * 5.0' in source
    assert 'regret * 16.0' in source


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


def test_gui_orders_direct_capacity_before_staged_micro_and_training():
    source = GUI.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert tree is not None
    assert '"direct",\n    "1",\n    "Direct Residual Capacity"' in source
    assert '"micro",\n    "2",\n    "Raven Staged Micro"' in source
    assert '_quick = _renumber(_existing["quick"], "3")' in source
    assert '_train = _renumber(_existing["train"], "4")' in source
    assert '"preview",\n    "5",\n    "Preview"' in source
    assert '_existing["setup"],\n    _direct,\n    _micro,\n    _quick,\n    _train,\n    _preview,' in source
    assert 'run_nsamdr_v9_raven_direct_residual_diagnostic.py' in source
    assert 'run_nsamdr_v9_raven_micro_diagnostic.py' in source
    assert 'self._value("direct_steps", "1536")' in source
    assert 'Sweeps residual amplitude first' in source
    assert 'Direct Residual Capacity should pass first' in source
    assert 'DIAGNOSTIC ONLY' in source


def test_standard_windows_gui_launcher_uses_micro_extension():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert 'if /I "%~1"=="gui"' in launcher
    assert 'nsamdr_v9_workflow_gui_micro.py' in launcher
