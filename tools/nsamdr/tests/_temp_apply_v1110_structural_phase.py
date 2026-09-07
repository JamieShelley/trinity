from pathlib import Path

ROOT = Path.cwd()


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    source = path.read_text(encoding="utf-8")
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{relative}: expected one anchor, found {count}: {old[:120]!r}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


LOSSES = "tools/nsamdr/neural/v9/losses.py"
SMOKE = "tools/nsamdr/neural/v9/application/baseline_relative_smoke.py"
LOCAL = "tools/nsamdr/neural/v9/local_boundary_production_contract.py"
EDGE = "tools/nsamdr/neural/v9/edge_constrained_spline_graph.py"
TEST = "tools/nsamdr/tests/test_v117_baseline_relative_contract.py"
DESIGN = "tools/nsamdr/NSAMDR_BASELINE_RELATIVE_DESIGN.md"

# B1a/B1b are structural stages. The public model forward intentionally keeps
# downstream seam/detail components connected, but structural supervision and
# acceptance must consume the same pre-seam C that the live "structural" preview
# displays. Otherwise frozen downstream seam authority contaminates C-vs-B.
replace_once(
    LOSSES,
    '''            losses["seam_phase_residual"] = zero
            losses["seam_recovery"] = zero.detach()
            losses["seam_authority_iou"] = zero.detach()
            losses["seam_projected_view"] = zero

        # V9.9.3 Panel-2 teacher.''',
    '''            losses["seam_phase_residual"] = zero
            losses["seam_recovery"] = zero.detach()
            losses["seam_authority_iou"] = zero.detach()
            losses["seam_projected_view"] = zero

        # V11.10 B1 phase isolation. The public forward keeps every production
        # component connected, but B1a/B1b structural supervision must consume
        # exactly the pre-seam structural stage shown as live C. Frozen downstream
        # seam authority is not structural evidence and must not change C-vs-B.
        if phase in {"sdf-bootstrap", "sdf-proof"}:
            reconstructed_albedo = seam_source_albedo
            reconstructed_normal = seam_source_normal
            reconstructed_material = seam_source_material

        # V9.9.3 Panel-2 teacher.''',
)

replace_once(
    LOSSES,
    '''        losses["sdf_stageb_renderer_improvement"] = (
            (baseline_boundary_mae - stageb_boundary_mae)
            / baseline_boundary_mae.clamp_min(1.0e-6)
        ).detach()
''',
    '''        losses["sdf_stageb_renderer_improvement"] = (
            (baseline_boundary_mae - stageb_boundary_mae)
            / baseline_boundary_mae.clamp_min(1.0e-6)
        ).detach()
        # Explicit V11.10 names prevent the application smoke gate from silently
        # drifting back to a downstream/post-seam consumer.
        losses["structural_baseline_mae"] = baseline_boundary_mae.detach()
        losses["structural_stage_mae"] = stageb_boundary_mae.detach()
        losses["structural_relative_gain"] = losses["sdf_stageb_renderer_improvement"]
''',
)

replace_once(
    LOSSES,
    '''        losses["regression_fraction"] = self._mean_fp32(
            (
                reconstructed_geometry_local
                > baseline_geometry_local + 1.0e-4
            ).float()
        )
''',
    '''        losses["regression_fraction"] = self._mean_fp32(
            (
                reconstructed_geometry_local
                > baseline_geometry_local + 1.0e-4
            ).float()
        )
        losses["structural_improvement_fraction"] = losses["improvement_fraction"].detach()
        losses["structural_regression_fraction"] = losses["regression_fraction"].detach()
''',
)

replace_once(
    SMOKE,
    '''        return {
            "baselineMae": value("sdf_stageb_baseline_mae", float("inf")),
            "candidateMae": value("sdf_stageb_renderer_mae", float("inf")),
            "relativeGain": value("sdf_stageb_renderer_improvement", float("-inf")),
            "improvementFraction": value("improvement_fraction", 0.0),
            "regressionFraction": value("regression_fraction", 1.0),
        }
''',
    '''        return {
            "baselineMae": value("structural_baseline_mae", float("inf")),
            "candidateMae": value("structural_stage_mae", float("inf")),
            "relativeGain": value("structural_relative_gain", float("-inf")),
            "improvementFraction": value("structural_improvement_fraction", 0.0),
            "regressionFraction": value("structural_regression_fraction", 1.0),
        }
''',
)

for relative in (LOCAL, EDGE):
    replace_once(
        relative,
        "NSAMDR_RAVEN_PRODUCTION_B1A_IDENTITY_B1B_RESIDUAL_SPLINE_GRAPH_4X_V11_9_0",
        "NSAMDR_RAVEN_PRODUCTION_B1A_IDENTITY_B1B_PRESEAM_RESIDUAL_SPLINE_GRAPH_4X_V11_10_0",
    )

replace_once(
    TEST,
    'assert MODEL_SCHEMA == "NSAMDR_RAVEN_PRODUCTION_B1A_IDENTITY_B1B_RESIDUAL_SPLINE_GRAPH_4X_V11_9_0"',
    'assert MODEL_SCHEMA == "NSAMDR_RAVEN_PRODUCTION_B1A_IDENTITY_B1B_PRESEAM_RESIDUAL_SPLINE_GRAPH_4X_V11_10_0"',
)

with (ROOT / TEST).open("a", encoding="utf-8") as handle:
    handle.write('''\n\ndef test_v1110_structural_smoke_is_pre_seam_and_uses_explicit_metrics():\n    losses = text("tools/nsamdr/neural/v9/losses.py")\n    smoke = text("tools/nsamdr/neural/v9/application/baseline_relative_smoke.py")\n    assert 'if phase in {"sdf-bootstrap", "sdf-proof"}:' in losses\n    assert 'reconstructed_albedo = seam_source_albedo' in losses\n    assert 'reconstructed_normal = seam_source_normal' in losses\n    assert 'reconstructed_material = seam_source_material' in losses\n    assert 'losses["structural_baseline_mae"]' in losses\n    assert 'losses["structural_stage_mae"]' in losses\n    assert 'losses["structural_relative_gain"]' in losses\n    assert 'losses["structural_improvement_fraction"]' in losses\n    assert 'losses["structural_regression_fraction"]' in losses\n    assert 'value("structural_baseline_mae"' in smoke\n    assert 'value("structural_stage_mae"' in smoke\n    assert 'value("structural_relative_gain"' in smoke\n    assert 'value("structural_improvement_fraction"' in smoke\n    assert 'value("structural_regression_fraction"' in smoke\n    assert 'value("sdf_stageb_renderer_mae"' not in smoke\n''')

with (ROOT / DESIGN).open("a", encoding="utf-8") as handle:
    handle.write('''\n\n## V11.10 structural-stage consumer contract\n\nB1a and B1b are evaluated on the pre-seam structural output, matching the live `structural` C preview. The public production forward remains fully connected, but frozen downstream seam/detail components cannot contribute to B1 structural training, regret, or baseline-relative acceptance evidence. B1a therefore preserves exact B when structural residual gain is zero; B1b must earn strict C > B using the structural stage itself.\n''')

print("Applied V11.10 structural phase-isolation correction")
