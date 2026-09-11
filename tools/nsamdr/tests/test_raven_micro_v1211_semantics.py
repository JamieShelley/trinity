from __future__ import annotations

from types import SimpleNamespace

import run_nsamdr_v9_raven_micro_diagnostic_v8 as v8


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        sdf_relative_win_fraction=0.65,
        sdf_relative_regression_fraction=0.20,
        sdf_missing_contour_tolerance=0.0,
        sdf_catastrophic_chamfer_pixels=48.0,
    )


def test_g1_geometry_can_qualify_before_profile_render() -> None:
    item = {
        "geometry": {
            "relativeContourGain": 0.03,
            "regressionFraction": 0.09,
            "predictedMissingContourFraction": 0.0,
            "sourceMissingContourFraction": 0.0,
            "topologyRegression": 0.0,
            "predictedChamferPixels": 6.1,
        },
        "lossTerms": {
            "spline_graph_point": 0.08,
            "spline_graph_point_gain": 0.13,
            "spline_graph_point_win_fraction": 0.73,
            "spline_graph_point_regret": 0.01,
            "spline_graph_same_edge_teacher_coverage": 0.51,
        },
        # Deliberately regressive: V12.11 defers this to the trained P stage.
        "structureEdgeRecovery": -0.01,
        "structureGlobalRecovery": -0.02,
    }
    assert v8._g1_geometry_only_qualified(item, _config())


def test_g1_still_rejects_bad_connected_spline_geometry() -> None:
    item = {
        "geometry": {
            "relativeContourGain": 0.03,
            "regressionFraction": 0.09,
            "predictedMissingContourFraction": 0.0,
            "sourceMissingContourFraction": 0.0,
            "topologyRegression": 0.0,
            "predictedChamferPixels": 6.1,
        },
        "lossTerms": {
            "spline_graph_point": 0.08,
            "spline_graph_point_gain": 0.13,
            "spline_graph_point_win_fraction": 0.40,
            "spline_graph_point_regret": 0.01,
            "spline_graph_same_edge_teacher_coverage": 0.51,
        },
    }
    assert not v8._g1_geometry_only_qualified(item, _config())
