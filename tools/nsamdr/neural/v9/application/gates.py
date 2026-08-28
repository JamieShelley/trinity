"""Production promotion gates and pass-driven stage plan."""
from __future__ import annotations

from typing import Any, Callable

from ..config import V9Config
from .domain import StageDefinition


StageGate = Callable[[dict[str, Any], V9Config], bool]


class QualificationGates:
    """Evaluate persisted trainer metrics without owning training or checkpoints."""

    def _metric_float(
        self,
        metrics: dict[str, Any],
        key: str,
        default: float,
    ) -> float:
        """Read one metric as float with a fail-closed default.

        Purpose:
            Normalise exported JSON/tensor-like numeric gate inputs.
        Called by:
            QualificationGates.production_gate(), QualificationGates.local_geometry_gate().
        Calls:
            float().
        """
        value = metrics.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def local_geometry_metrics(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Select the strongest available structural validation metric bundle.

        Purpose:
            Reuse persisted structural metrics across stage invocations and recovery.
        Called by:
            QualificationGates.local_geometry_gate(), PassDrivenPipeline.
        Calls:
            No project functions.
        """
        metrics = metadata.get("bestSyntheticSdfValidation")
        if isinstance(metrics, dict) and metrics:
            return metrics
        metrics = metadata.get("syntheticSdfValidation")
        return metrics if isinstance(metrics, dict) else {}

    def production_gate(self, metadata: dict[str, Any], config: V9Config) -> bool:
        """Re-evaluate the canonical gate-proof promotion rule from exported metrics.

        Purpose:
            Preserve the original boundary/profile promotion decision outside the monolithic CLI.
        Called by:
            StagePlan gate-proof definition.
        Calls:
            QualificationGates._metric_float().
        """
        metrics = metadata.get("bestSyntheticSdfValidation")
        if not isinstance(metrics, dict) or not metrics:
            return False

        contour_gain = self._metric_float(metrics, "sdf_zero_contour_relative_gain_mean", -1.0)
        contour_wins = self._metric_float(metrics, "sdf_zero_contour_relative_win_fraction", 0.0)
        contour_regress = self._metric_float(metrics, "sdf_zero_contour_relative_regression_fraction", 1.0)
        source_missing = self._metric_float(metrics, "sdf_source_missing_contour_fraction", 1.0)
        predicted_missing = self._metric_float(metrics, "sdf_predicted_missing_contour_fraction", 1.0)
        contour_chamfer = self._metric_float(metrics, "sdf_zero_contour_chamfer_pixels", 999.0)
        topology_regression = self._metric_float(metrics, "sdf_stageb_topology_regression_fraction", 1.0)
        line_jitter = self._metric_float(metrics, "sdf_line_perpendicular_jitter_pixels_mean", 999.0)
        curve_roughness = self._metric_float(metrics, "sdf_circle_radial_roughness_pixels_mean", 999.0)
        staircase_recovery = self._metric_float(metrics, "sdf_line_staircase_recovery_mean", -1.0)

        hard_structure_gate = (
            contour_gain >= float(config.sdf_relative_gain_required)
            and contour_wins >= float(config.sdf_relative_win_fraction)
            and contour_regress <= float(config.sdf_relative_regression_fraction)
            and predicted_missing
            <= source_missing + float(config.sdf_missing_contour_tolerance)
            and contour_chamfer <= float(config.sdf_catastrophic_chamfer_pixels)
            and topology_regression == 0.0
            and line_jitter <= float(config.structural_line_jitter_required_pixels)
            and curve_roughness <= float(config.structural_curve_roughness_required_pixels)
            and staircase_recovery
            >= float(config.structural_line_staircase_recovery_required)
        )

        oracle_render_mae = self._metric_float(metrics, "sdf_oracle_render_band_mae_mean", 999.0)
        oracle_global_mae = self._metric_float(metrics, "sdf_oracle_global_mae_mean", 999.0)
        oracle_global_mae_max = self._metric_float(metrics, "sdf_oracle_global_mae_case_max", 999.0)
        oracle_gradient_mae = self._metric_float(metrics, "sdf_oracle_gradient_mae_mean", 999.0)
        oracle_width_error = self._metric_float(metrics, "sdf_oracle_profile_width_relative_error_mean", 999.0)
        oracle_profile_corr = self._metric_float(metrics, "sdf_oracle_profile_correlation_mean", -1.0)
        oracle_core_halo_delta = self._metric_float(metrics, "sdf_oracle_core_halo_delta_8bit_max", 999.0)

        profile_render_gate = (
            oracle_render_mae <= float(config.sdf_oracle_render_band_mae_required)
            and oracle_gradient_mae <= float(config.sdf_oracle_gradient_mae_required)
            and oracle_width_error <= float(config.sdf_oracle_profile_width_error_required)
            and oracle_profile_corr >= float(config.sdf_oracle_profile_correlation_required)
            and oracle_core_halo_delta
            <= float(config.sdf_oracle_core_halo_delta_required_8bit)
        )
        direct_pixel_render_gate = (
            oracle_global_mae <= float(config.sdf_oracle_global_mae_required)
            and oracle_global_mae_max
            <= float(config.sdf_oracle_global_mae_case_max_required)
            and oracle_render_mae
            <= float(config.sdf_oracle_render_band_mae_preview_required)
            and oracle_gradient_mae <= float(config.sdf_oracle_gradient_mae_required)
        )

        profile_teacher_recovery = self._metric_float(
            metrics,
            "sdf_profile_teacher_recovery_mean",
            -1.0,
        )
        return bool(
            hard_structure_gate
            and (profile_render_gate or direct_pixel_render_gate)
            and profile_teacher_recovery >= float(config.sdf_teacher_recovery_required)
        )

    def local_geometry_gate(self, metadata: dict[str, Any], config: V9Config) -> bool:
        """Apply the B1/B2 structural/redraw thresholds without obsolete class gating.

        Purpose:
            Decide whether local analytic geometry may advance to downstream production stages.
        Called by:
            StagePlan sdf-bootstrap definition and pipeline recovery loop.
        Calls:
            QualificationGates.local_geometry_metrics(), QualificationGates._metric_float().
        """
        metrics = self.local_geometry_metrics(metadata)
        if not metrics:
            return False

        contour_gain = self._metric_float(metrics, "sdf_zero_contour_relative_gain_mean", -1.0)
        contour_wins = self._metric_float(metrics, "sdf_zero_contour_relative_win_fraction", 0.0)
        contour_regress = self._metric_float(metrics, "sdf_zero_contour_relative_regression_fraction", 1.0)
        source_missing = self._metric_float(metrics, "sdf_source_missing_contour_fraction", 1.0)
        predicted_missing = self._metric_float(metrics, "sdf_predicted_missing_contour_fraction", 1.0)
        contour_chamfer = self._metric_float(metrics, "sdf_zero_contour_chamfer_pixels", 999.0)
        topology_regression = self._metric_float(metrics, "sdf_stageb_topology_regression_fraction", 1.0)
        line_jitter = self._metric_float(metrics, "sdf_line_perpendicular_jitter_pixels_mean", 999.0)
        curve_roughness = self._metric_float(metrics, "sdf_circle_radial_roughness_pixels_mean", 999.0)
        staircase_recovery = self._metric_float(metrics, "sdf_line_staircase_recovery_mean", -1.0)

        structure = (
            contour_gain >= float(config.sdf_relative_gain_required)
            and contour_wins >= float(config.sdf_relative_win_fraction)
            and contour_regress <= float(config.sdf_relative_regression_fraction)
            and predicted_missing <= source_missing + float(config.sdf_missing_contour_tolerance)
            and contour_chamfer <= float(config.sdf_catastrophic_chamfer_pixels)
            and topology_regression == 0.0
            and line_jitter <= float(config.structural_line_jitter_required_pixels)
            and curve_roughness <= float(config.structural_curve_roughness_required_pixels)
            and staircase_recovery >= float(config.structural_line_staircase_recovery_required)
        )
        if not structure:
            return False

        oracle_render_mae = self._metric_float(metrics, "sdf_oracle_render_band_mae_mean", 999.0)
        oracle_global_mae = self._metric_float(metrics, "sdf_oracle_global_mae_mean", 999.0)
        oracle_global_mae_max = self._metric_float(metrics, "sdf_oracle_global_mae_case_max", 999.0)
        oracle_gradient_mae = self._metric_float(metrics, "sdf_oracle_gradient_mae_mean", 999.0)
        oracle_width_error = self._metric_float(metrics, "sdf_oracle_profile_width_relative_error_mean", 999.0)
        oracle_profile_corr = self._metric_float(metrics, "sdf_oracle_profile_correlation_mean", -1.0)
        oracle_core_halo_delta = self._metric_float(metrics, "sdf_oracle_core_halo_delta_8bit_max", 999.0)

        profile_render = (
            oracle_render_mae <= float(config.sdf_oracle_render_band_mae_required)
            and oracle_gradient_mae <= float(config.sdf_oracle_gradient_mae_required)
            and oracle_width_error <= float(config.sdf_oracle_profile_width_error_required)
            and oracle_profile_corr >= float(config.sdf_oracle_profile_correlation_required)
            and oracle_core_halo_delta <= float(config.sdf_oracle_core_halo_delta_required_8bit)
        )
        direct_render = (
            oracle_global_mae <= float(config.sdf_oracle_global_mae_required)
            and oracle_global_mae_max <= float(config.sdf_oracle_global_mae_case_max_required)
            and oracle_render_mae <= float(config.sdf_oracle_render_band_mae_preview_required)
            and oracle_gradient_mae <= float(config.sdf_oracle_gradient_mae_required)
        )
        return bool(profile_render or direct_render)

    def simple_gate(self, key: str) -> StageGate:
        """Create a persisted-boolean stage gate.

        Purpose:
            Express downstream qualification gates without ad-hoc lambdas in the stage plan.
        Called by:
            StagePlan.__init__().
        Calls:
            No project functions.
        """
        def gate(metadata: dict[str, Any], _config: V9Config) -> bool:
            """Read one persisted boolean qualification flag.

            Purpose:
                Implement the small gate returned by QualificationGates.simple_gate().
            Called by:
                StagePlan.already_qualified() and PassDrivenPipeline stage evaluation.
            Calls:
                dict.get().
            """
            return bool(metadata.get(key, False))
        return gate


class StagePlan:
    """Own the ordered pass-driven production stage definitions."""

    def __init__(self, gates: QualificationGates) -> None:
        """Build the canonical local-geometry-to-detail stage sequence.

        Purpose:
            Make stage order explicit data owned by one application object.
        Called by:
            TrainingApplication._build_pipeline().
        Calls:
            QualificationGates.simple_gate().
        """
        self.definitions = (
            StageDefinition(
                "sdf-bootstrap",
                "B1 local analytic geometry + B2 same-renderer redraw",
                gates.local_geometry_gate,
            ),
            StageDefinition(
                "seam-proof",
                "B3 forced-authority seam reconstruction",
                gates.simple_gate("seamReconstructionQualified"),
            ),
            StageDefinition(
                "seam-authority",
                "B4 learned seam authority",
                gates.simple_gate("seamAuthorityQualified"),
            ),
            StageDefinition(
                "gate-proof",
                "boundary/profile candidate",
                gates.production_gate,
            ),
            StageDefinition(
                "detail-reconstruction",
                "geometry-conditioned physical detail",
                gates.simple_gate("detailQualified"),
            ),
        )

    def already_qualified(
        self,
        phase: str,
        snapshot: dict[str, Any],
        config: V9Config,
    ) -> bool:
        """Check whether one stage gate already passed in persisted state.

        Purpose:
            Avoid replaying qualified stages after a staged resume.
        Called by:
            PassDrivenPipeline.run().
        Calls:
            StageDefinition.gate callables.
        """
        for definition in self.definitions:
            if definition.phase == phase:
                return bool(definition.gate(snapshot, config))
        return False

    def legacy_tuple(self) -> tuple[tuple[str, str, StageGate], ...]:
        """Expose the historic tuple shape for compatibility tests/importers.

        Purpose:
            Keep staged refactoring source-compatible while the application API replaces internals.
        Called by:
            train_nsamdr_v9_preview_experiment compatibility facade.
        Calls:
            No project functions.
        """
        return tuple(
            (definition.phase, definition.gate_label, definition.gate)
            for definition in self.definitions
        )
