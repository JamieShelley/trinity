from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
V9 = ROOT / "tools/nsamdr/neural/v9"
PARALLEL = V9 / "parallel_detail_contract.py"
GUI = ROOT / "tools/nsamdr/gui/nsamdr_v9_workflow_gui_micro.py"
DIAGNOSTIC = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_parallel_detail_diagnostic.py"
DIAGNOSTIC_V2 = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_parallel_detail_diagnostic_v2.py"


def test_parallel_detail_contract_is_package_wide_and_spawn_safe():
    source = PARALLEL.read_text(encoding="utf-8")
    package = (V9 / "__init__.py").read_text(encoding="utf-8")
    ast.parse(source)
    assert 'PARALLEL_DETAIL_REVISION = "V12.3"' in source
    assert "install_parallel_detail_contract()" in package
    assert "<locals>" not in source


def test_detail_candidate_is_anchored_to_deterministic_baseline():
    source = PARALLEL.read_text(encoding="utf-8")
    assert "def _deterministic_baseline(" in source
    assert 'mode="bicubic"' in source
    assert "antialias=True" in source
    assert 'mode="bilinear"' in source
    assert 'mode="nearest"' in source
    assert "neutral_geometry = torch.zeros(" in source
    assert "baseline_albedo," in source
    assert "baseline_normal," in source
    assert "baseline_material," in source
    assert "desired_albedo" in source
    assert "base_albedo_hr.detach()" in source
    assert "desired_normal_pre" in source
    assert "base_normal_hr.detach()" in source
    assert "desired_material" in source
    assert "base_material_hr.detach()" in source


def test_selector_features_do_not_depend_on_learned_geometry_or_seam():
    source = PARALLEL.read_text(encoding="utf-8")
    assert "def _parallel_selector_features(" in source
    assert "del sdf_pixels, normal, coverage, profile_confidence, edge_probability" in source
    assert "baseline_gray" in source
    assert "candidate_gray" in source
    assert "signed_delta" in source
    assert "gradient_delta" in source
    assert "observed_support.float().clamp(0.0, 1.0)" in source
    assert "detail_confidence.float().clamp(0.0, 1.0)" in source
    assert "detail_regret.float().clamp(0.0, 1.0)" in source


def test_detail_training_matches_successful_direct_capacity_objective():
    source = PARALLEL.read_text(encoding="utf-8")
    assert 'if phase != "detail-reconstruction":' in source
    assert 'baseline = outputs["baseline_albedo"].detach().float()' in source
    assert 'candidate = outputs["detail_candidate_albedo"].float()' in source
    assert "desired_residual = (target.detach() - baseline).clamp(-cap, cap)" in source
    assert "global_reconstruction * 4.0" in source
    assert "edge_reconstruction * 12.0" in source
    assert "gradient * 5.0" in source
    assert "regret * 16.0" in source
    assert "residual_supervision * 8.0" in source


def test_architecture_contract_marks_direct_detail_as_independent_specialist():
    source = PARALLEL.read_text(encoding="utf-8")
    assert 'contract["parallelDetailRevision"] = PARALLEL_DETAIL_REVISION' in source
    assert 'contract["detailGeometryConditioningAuthority"] = False' in source
    assert 'contract["detailBase"] = "deterministic bicubic/bilinear/nearest B"' in source
    assert "geometry/seam outputs cannot alter the detail candidate" in source
    assert "no learned geometry or seam prerequisite" in source


def test_parallel_entrypoint_routes_to_fast_v2_proof():
    wrapper = DIAGNOSTIC.read_text(encoding="utf-8")
    source = DIAGNOSTIC_V2.read_text(encoding="utf-8")
    ast.parse(wrapper)
    ast.parse(source)
    assert "from run_nsamdr_v9_raven_parallel_detail_diagnostic_v2 import main" in wrapper
    assert 'REPORT_SCHEMA = "NSAMDR_RAVEN_PARALLEL_DETAIL_INTEGRATION_V2"' in source
    assert "parallel_contract._ORIGINAL_DETAIL_FORWARD" in source
    assert '"detail-fast"' in source
    assert '"selector-fast"' in source
    assert "production_candidate = production_after_detail[\"detail_candidate_albedo\"].float()" in source
    assert "candidate_parity_max_abs" in source
    assert "final_parity_max_abs" in source
    assert '"candidatePass": bool(candidate_pass)' in source
    assert '"selectorPass": bool(selector_pass)' in source
    assert '"promotable": False' in source


def test_fast_parallel_proof_does_not_repeat_geometry_seam_during_specialist_steps():
    source = DIAGNOSTIC_V2.read_text(encoding="utf-8")
    detail_start = source.index("def _train_detail(")
    selector_start = source.index("def _train_selector(")
    main_start = source.index("def main(")
    detail_body = source[detail_start:selector_start]
    selector_body = source[selector_start:main_start]
    assert "model.geometry_net(" not in detail_body
    assert "model.seam_restorer(" not in detail_body
    assert "model.geometry_net(" not in selector_body
    assert "model.seam_restorer(" not in selector_body
    assert "_raw_direct_candidate(" in detail_body
    assert "model.benefit_selector(" in selector_body


def test_fast_parallel_detail_reproduces_passing_optimizer_and_raw_residual_supervision():
    source = DIAGNOSTIC_V2.read_text(encoding="utf-8")
    assert 'parser.add_argument("--detail-learning-rate", type=float, default=1.0e-3)' in source
    assert 'parser.add_argument("--head-lr-multiplier", type=float, default=3.0)' in source
    assert 'parser.add_argument("--detail-steps", type=int, default=1536)' in source
    assert 'raw_delta = detail["albedo_raw"].float() * float(residual_cap)' in source
    assert "raw_delta.float() - desired_residual" in source
    assert "weight_decay=0.0" in source
    assert 'betas=(0.9, 0.99)' in source


def test_gui_places_parallel_integration_between_direct_and_staged_micro():
    source = GUI.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"direct",\n    "1",\n    "Direct Residual Capacity"' in source
    assert '"parallel",\n    "2",\n    "Parallel Detail Integration"' in source
    assert '"micro",\n    "3",\n    "Raven Staged Micro"' in source
    assert 'run_nsamdr_v9_raven_parallel_detail_diagnostic.py' in source
    assert 'self._value("parallel_detail_steps", "1280")' in source
    assert 'self._value("parallel_selector_steps", "384")' in source
    assert 'self._value("parallel_retention", "0.85")' in source
