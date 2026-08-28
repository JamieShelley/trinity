"""Contracts for the V11.3 local-boundary production override."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "tools/nsamdr/neural/v9/local_boundary_production_contract.py"
ENTRY = ROOT / "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py"
INIT = ROOT / "tools/nsamdr/neural/v9/__init__.py"


def test_current_v11_is_patched_not_replaced() -> None:
    source = CONTRACT.read_text(encoding="utf-8")
    assert "_ORIGINAL_GEOMETRY_INIT(self, config)" in source
    assert "requires the current V11 model.py" in source
    assert "parametric_primitive_field" in source


def test_structural_authority_is_auditable_wrapper() -> None:
    source = CONTRACT.read_text(encoding="utf-8")
    assert "class LocalBoundaryProductionStructure(nn.Module)" in source
    assert "LocalParametricBoundaryDecoder" in source
    assert 'production["structural representation"] = "geometry_net.production_structure"' in source
    assert '"wholeTilePrimitiveClassifierAuthority": False' in source


def test_production_structure_is_called_through_forward_not_query() -> None:
    source = CONTRACT.read_text(encoding="utf-8")
    forward = source[source.index("def _geometry_forward"):source.index("def _geometry_query_from_outputs")]
    assert "self.production_structure(" in forward
    assert "self.production_structure.decoder.query(" not in forward
    # Query-from-saved-outputs may call the child decoder directly; the main
    # production forward has already triggered the wrapper audit hook.
    query = source[source.index("def _geometry_query_from_outputs"):source.index("def _set_phase")]
    assert "self.production_structure.decoder.query(context, query_grid)" in query


def test_no_duplicate_module_aliases_in_contract_source() -> None:
    source = CONTRACT.read_text(encoding="utf-8")
    init = source[source.index("def _geometry_init"):source.index("def _geometry_encode")]
    assert "self.production_structure =" in init
    assert "self.local_boundary_decoder =" not in init
    assert "self.local_boundary_feature_project =" not in init


def test_complete_production_graph_is_preserved() -> None:
    source = CONTRACT.read_text(encoding="utf-8")
    for path in (
        "model.boundary_renderer",
        "model.boundary_specialist",
        "model.seam_restorer.phase_sr",
        "model.seam_restorer.authority",
        "model.detail_net",
        "model.detail_net.albedo_head",
        "model.detail_net.normal_head",
        "model.detail_net.material_head",
        "model.detail_net.confidence_head",
        "model.detail_net.regret_head",
        "model.benefit_selector",
    ):
        assert path in source


def test_local_losses_are_existing_physical_geometry_losses() -> None:
    source = CONTRACT.read_text(encoding="utf-8")
    for name in (
        'losses["sdf_surface"]',
        'losses["sdf_sign"]',
        'losses["sdf_topology_sign"]',
        'losses["sdf_eikonal"]',
        'losses["sdf_metric_gradient"]',
        'losses["sdf_improvement_regret"]',
        'losses["parametric_anchor"]',
        'losses["parametric_normal"]',
        'losses["parametric_curvature"]',
        'losses["parametric_offset_smoothness"]',
    ):
        assert name in source


def test_quick_uses_validator_legal_minimum_for_retired_slot() -> None:
    source = ENTRY.read_text(encoding="utf-8")
    quick = source[source.index("QUICK_WORK_BUDGET"):source.index("FULL_MINIMUM_WORK_BUDGET")]
    assert '"identity_epochs": 3' in quick
    assert '"residual_epochs": 1' in quick
    assert '"parametric_primitive_train_tiles_per_epoch": 14' in quick
    assert '"tiles_per_epoch": 64' in quick


def test_pass_driven_plan_never_invokes_retired_sdf_proof() -> None:
    source = ENTRY.read_text(encoding="utf-8")
    plan = source[source.index("_PASS_DRIVEN_STAGE_PLAN"):source.index("def _stage_already_qualified")]
    assert '"sdf-bootstrap"' in plan
    assert '"sdf-proof"' not in plan
    assert '"seam-proof"' in plan


def test_local_promotion_advances_resume_cursor_past_retired_slot() -> None:
    source = ENTRY.read_text(encoding="utf-8")
    promotion = source[source.index("def _promote_local_geometry_state"):source.index("_PASS_DRIVEN_STAGE_PLAN")]
    assert "retired_end = int(config.identity_epochs) + int(config.residual_epochs)" in promotion
    assert 'state["completed_epoch"] = max(learned_epoch, retired_end)' in promotion
    assert 'state["retired_b1b_epochs_skipped"] = retired_skipped' in promotion
    assert 'state["b1b_classifier_qualified"] = True' not in promotion
    assert 'state["b1b_parameters_qualified"] = True' not in promotion


def test_local_gate_does_not_consume_primitive_classifier_metrics() -> None:
    source = ENTRY.read_text(encoding="utf-8")
    gate = source[source.index("def _local_geometry_gate"):source.index("def _promote_local_geometry_state")]
    assert "primitive_class_accuracy" not in gate
    assert "primitive_teacher_param_mae" not in gate
    assert "sdf_stageb_topology_regression_fraction" in gate
    assert "sdf_zero_contour_relative_gain_mean" in gate
    assert "sdf_oracle_render_band_mae_mean" in gate


def test_package_init_installs_one_local_model_contract() -> None:
    source = INIT.read_text(encoding="utf-8")
    assert "install_local_boundary_model_contract()" in source
    assert "install_b1a_parametric_bootstrap" not in source
    assert "install_classifier_generalisation_contract" not in source


def test_runtime_forward_hook_sees_production_structure() -> None:
    """Regression test for the exact V11.1 preflight failure."""
    torch = pytest.importorskip("torch")
    neural = ROOT / "tools/nsamdr/neural"
    if str(neural) not in sys.path:
        sys.path.insert(0, str(neural))
    from v9.config import V9Config  # type: ignore
    from v9.model import FidelityResidualNetV9  # type: ignore

    config = V9Config()
    model = FidelityResidualNetV9(config)
    calls = {"count": 0}

    def observed(_module, _args, _output):
        calls["count"] += 1

    handle = model.geometry_net.production_structure.register_forward_hook(observed)
    try:
        sample = torch.rand((1, 17, 16, 16), dtype=torch.float32)
        sample[:, 3:5] = sample[:, 3:5] * 2.0 - 1.0
        model.eval()
        with torch.inference_mode():
            geometry = model.geometry_net(sample)
        assert geometry["sdf"].shape[-2:] == (64, 64)
        assert torch.isfinite(geometry["sdf"]).all()
        assert calls["count"] == 1
    finally:
        handle.remove()


def test_runtime_component_map_matches_canonical_trainer_shape() -> None:
    """Regression for the exact V11.2 startup abort.

    The canonical trainer unpacks every component entry as (path, module).
    Returning bare modules crashes before the first epoch.
    """
    torch = pytest.importorskip("torch")
    neural = ROOT / "tools/nsamdr/neural"
    if str(neural) not in sys.path:
        sys.path.insert(0, str(neural))

    import v9.training as training  # type: ignore
    from v9.config import V9Config  # type: ignore
    from v9.model import FidelityResidualNetV9  # type: ignore
    from v9.local_boundary_production_contract import (  # type: ignore
        install_local_boundary_training_contract,
    )

    install_local_boundary_training_contract(training)
    model = FidelityResidualNetV9(V9Config())
    components = training._production_component_modules(model)

    assert components
    for label, value in components.items():
        assert isinstance(value, tuple), (label, type(value))
        assert len(value) == 2, (label, value)
        path, module = value
        assert isinstance(path, str) and path
        assert isinstance(module, torch.nn.Module)

    assert components["structural representation"][0] == (
        "geometry_net.production_structure"
    )


def test_full_production_forward_smoke_after_contract_install() -> None:
    """Exercise the complete production model, not GeometryNet alone."""
    torch = pytest.importorskip("torch")
    neural = ROOT / "tools/nsamdr/neural"
    if str(neural) not in sys.path:
        sys.path.insert(0, str(neural))

    from v9.config import V9Config  # type: ignore
    from v9.model import FidelityResidualNetV9  # type: ignore

    model = FidelityResidualNetV9(V9Config()).eval()
    sample = torch.rand((1, 17, 16, 16), dtype=torch.float32)
    sample[:, 3:5] = sample[:, 3:5] * 2.0 - 1.0
    with torch.inference_mode():
        outputs = model(sample)

    for key in ("albedo", "normal_xy", "material", "roughness", "emissive"):
        assert key in outputs
        assert tuple(outputs[key].shape[-2:]) == (64, 64)
        assert torch.isfinite(outputs[key]).all()
