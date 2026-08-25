"""Configuration for NSAMDR V10.7.9 deterministic geometry-redraw proofs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class V9Config:
    # Dataset: V9 owns its own manifest and artifact roots.
    dataset_manifest: str = "artifacts/nsamdr/training_v9/dataset_manifest.json"
    dataset_root: str = "artifacts/nsamdr/training_v9"
    max_families: int = 192
    crops_per_family: int = 10
    source_crop_size: int = 640
    min_source_dimension: int = 1024
    min_auxiliary_dimension: int = 128
    validation_fraction: float = 0.10
    require_complete_pbr_family: bool = False

    # V10.7.9 isolates the deterministic structural problem before any full-detail training.
    # B1 geometry -> B2 renderer/profile -> B3 forced-authority seam reconstruction
    # -> B4 learned seam authority. The legacy boundary/detail phases remain after
    # those proofs, but quick Raven testing stops at B4.
    identity_epochs: int = 1
    residual_epochs: int = 3
    seam_proof_epochs: int = 1
    seam_authority_epochs: int = 1
    boundary_epochs: int = 1
    detail_epochs: int = 4
    physical_finetune_epochs: int = 4
    tiles_per_epoch: int = 2048
    validation_tiles: int = 128
    batch_size: int = 1
    identity_learning_rate: float = 5.0e-5
    learning_rate: float = 1.0e-4
    boundary_learning_rate: float = 8.0e-5
    detail_learning_rate: float = 6.0e-5
    finetune_learning_rate: float = 3.0e-5
    weight_decay: float = 1.0e-5
    optimizer_name: str = "adamw"
    optimizer_beta1: float = 0.90
    optimizer_beta2: float = 0.99
    scheduler_name: str = "phase"
    scheduler_min_lr_ratio: float = 0.25
    tile_size: int = 128
    target_scale: int = 4
    boundary_candidate_count: int = 8
    boundary_sampling_probability: float = 0.62

    input_channels: int = 17
    material_classes: int = 4
    widths: tuple[int, int, int, int] = (80, 128, 192, 256)
    blocks_per_level: tuple[int, int, int, int] = (3, 4, 6, 6)
    decoder_blocks: tuple[int, int, int] = (4, 3, 3)
    attention_heads: int = 8
    attention_window: int = 8
    drop_path: float = 0.03
    orientation_normalization_epsilon: float = 1.0e-2

    # V9.4 appearance authority is deliberately tiny and is disabled during
    # geometry proof. These bounds become active only after geometry is frozen.
    albedo_medium_delta: float = 0.025
    albedo_fine_delta: float = 0.0
    normal_medium_delta: float = 0.030
    normal_fine_delta: float = 0.0
    material_delta: float = 0.025
    auxiliary_delta: float = 0.025
    initial_gate_bias: float = -2.25
    appearance_enabled: bool = True  # V10.6 enables full learned physical-map output authority
    appearance_edge_suppression: float = 0.80  # legacy parse-only

    # V10.6 full physical-detail reconstruction. This is a true 4x decoder,
    # conditioned on the accepted spline SDF/profile field. Residual authority is
    # bounded per physical map; the later selector may revert exactly to baseline.
    detail_reconstruction_enabled: bool = True
    detail_feature_channels: int = 48
    detail_mid_channels: int = 40
    detail_hr_channels: int = 32
    detail_albedo_max_delta: float = 0.20
    detail_normal_max_delta: float = 0.15
    detail_material_max_delta: float = 0.18
    detail_confidence_initial_bias: float = -0.50
    detail_regret_initial_bias: float = 0.00
    detail_contrast_weight: float = 0.60
    detail_cross_map_weight: float = 0.35
    detail_confidence_weight: float = 0.20
    detail_regret_classifier_weight: float = 0.40
    detail_recovery_required: float = 0.12
    detail_gradient_recovery_required: float = 0.08
    detail_normal_recovery_required: float = 0.00
    detail_win_fraction_required: float = 0.55
    detail_regression_fraction_max: float = 0.35

    # V10.8.0 Raven full-pipeline preview contract. Structural B1/B2 remains
    # promotion-gated, but the diagnostic Raven tuning action proceeds into the
    # real downstream appearance modules so the current checkpoint can be judged
    # visually rather than only through synthetic geometry proofs.
    raven_full_pipeline_preview_enabled: bool = True
    # V10.8.8 Raven-only representative-preview mode. This development mode
    # trains every learned appearance stage only on the fixed Raven dataset and
    # previews the actual final full-pipeline checkpoint even when promotion
    # gates are not yet satisfied. Gates remain telemetry/promotion authority.
    raven_representative_preview_enabled: bool = False
    raven_train_only_enabled: bool = False
    preview_allow_unqualified_downstream: bool = False
    raven_downstream_tiles_per_epoch: int = 24

    # V10.7.1 directional/vector seam restoration. Long manufactured panel seams
    # receive a RAISR/BLADE-like anisotropic cleanup after deterministic SDF
    # rendering. The branch shares one structure tensor across albedo/normal/
    # material and cannot move spline topology.
    seam_directional_enabled: bool = True
    seam_directional_channels: int = 24
    seam_directional_angle_bins: int = 12
    seam_directional_kernel_size: int = 7
    seam_directional_kernel_residual_scale: float = 0.10
    seam_phase_sr_channels: int = 32
    seam_phase_sr_max_delta: float = 0.40
    seam_phase_only_reconstruction: bool = False
    seam_phase_residual_weight: float = 36.0
    seam_microproof_enabled: bool = True
    seam_microproof_steps: int = 80
    seam_microproof_recovery_required: float = 0.80
    seam_structure_tensor_radius: int = 2
    seam_structure_strength_gain: float = 4.0
    seam_coherence_floor: float = 0.45
    seam_tangent_sample_pixels: float = 1.35
    seam_normal_sample_pixels: float = 0.90
    seam_max_normal_sharpen: float = 1.35
    seam_max_authority: float = 0.90
    seam_geometry_band_pixels: float = 4.0
    seam_directional_weight: float = 22.0
    seam_lr_multiplier: float = 2.0
    seam_tangent_smoothness_weight: float = 16.0
    seam_normal_profile_weight: float = 18.0
    seam_authority_regularization_weight: float = 0.20
    seam_projected_view_weight: float = 6.0
    seam_authority_teacher_weight: float = 18.0
    seam_reconstruction_weight: float = 28.0
    seam_forced_recovery_required: float = 0.70
    seam_authority_iou_required: float = 0.55
    seam_ridge_weight: float = 0.65
    seam_missing_detail_scale: float = 8.0
    seam_teacher_dilation_pixels: int = 2
    structural_stale_patience: int = 3

    # DDS-aware degradation. Authored crop metadata carries the original DDS
    # format; training reproduces block-aligned BC1/BC3/BC5-like endpoint/palette
    # quantisation instead of the old generic 4x block blur.
    dds_codec_degradation_enabled: bool = True
    dds_codec_probability: float = 0.90
    dds_codec_blend_min: float = 0.45
    dds_codec_blend_max: float = 1.00

    # V9.4 source-grid geometric transport. Flow is predicted in SOURCE pixels,
    # then smoothly upsampled and converted to output-pixel units.
    geometry_displacement_source_pixels: float = 1.0
    geometry_edge_support_radius: int = 2

    # Retained only so old JSON fields parse cleanly. V9.4 model code does not
    # use this output-pixel V9.3 value.
    geometry_displacement_pixels: float = 2.0

    # V9.8 geometry-convergence SDF renderer. The SDF predicts boundary position; the
    # renderer samples deterministic texture values from both sides and rebuilds
    # a controlled sub-pixel transition shared by every physical map.
    contour_sdf_max_distance_pixels: float = 24.0
    boundary_renderer_band_pixels: float = 3.5
    boundary_renderer_sample_pixels: float = 3.75
    boundary_renderer_hard_width_pixels: float = 0.70
    boundary_renderer_soft_width_pixels: float = 1.80
    boundary_renderer_edge_threshold: float = 0.08
    boundary_renderer_confidence_floor: float = 0.00
    boundary_renderer_gate_gain: float = 1.60
    boundary_renderer_far_sample_multiplier: float = 1.70
    boundary_renderer_far_sample_weight: float = 0.22
    boundary_gate_initial_bias: float = -1.40
    implicit_sdf_hidden_channels: int = 64
    implicit_sdf_coordinate_scale: float = 1.0
    implicit_sdf_residual_pixels: float = 2.0
    coarse_sdf_surface_weight: float = 6.0
    sdf_residual_l1_weight: float = 0.30
    boundary_renderer_plateau_samples: int = 5
    boundary_renderer_plateau_max_multiplier: float = 2.20
    boundary_renderer_plateau_stability_scale: float = 14.0

    # Legacy V9.4 flow weights remain parseable but are unused by V9.8.
    displacement_smoothness_weight: float = 0.50
    displacement_off_contour_weight: float = 0.80
    displacement_tangent_weight: float = 0.75
    displacement_sparsity_weight: float = 0.20
    geometry_regret_weight: float = 3.00
    geometry_regret_margin: float = 0.002
    edge_regret_multiplier: float = 3.00
    geometry_photometric_weight: float = 0.25
    boundary_photometric_weight: float = 2.00
    boundary_profile_weight: float = 1.50
    boundary_fuzz_weight: float = 2.50
    boundary_halo_weight: float = 1.75
    boundary_hardness_weight: float = 0.80
    boundary_gate_weight: float = 2.00
    boundary_off_contour_weight: float = 2.00
    boundary_regret_weight: float = 5.00
    boundary_sdf_zero_weight: float = 3.00
    boundary_edge_sdf_consistency_weight: float = 1.50
    boundary_pixel_regret_weight: float = 3.00
    boundary_gate_need_scale: float = 0.075
    boundary_gate_exact_floor: float = 0.35
    sdf_surface_weight: float = 8.00
    sdf_sign_weight: float = 2.00
    # V9.9.3 protects connectivity explicitly. A mean sign loss can hide a
    # one-pixel cut through a long thin material feature, so the worst confident
    # sign violations near the target contour receive separate authority.
    sdf_topology_weight: float = 8.00
    sdf_topology_margin_pixels: float = 0.35
    sdf_topology_core_pixels: float = 0.75
    sdf_topology_band_pixels: float = 6.00
    sdf_topology_worst_fraction: float = 0.005
    sdf_eikonal_weight: float = 8.00
    sdf_gradient_alignment_weight: float = 2.00
    sdf_metric_gradient_weight: float = 6.00
    # V9.9.3 retains SDF polarity as a gauge choice: +SDF and -SDF describe the
    # same physical contour when the two material sides are swapped together.
    sdf_sign_gauge_invariant: bool = True
    sdf_metric_band_pixels: float = 12.0
    sdf_coarse_init_std: float = 0.0005
    sdf_synthetic_validation_tiles: int = 12
    sdf_zero_band_pixels: float = 0.50
    sdf_bootstrap_residual_pixels: float = 0.0
    sdf_proof_residual_pixels: float = 1.0
    sdf_proof_renderer_weight: float = 2.50

    # V9.9.3 decomposes LR->HR contour correction into continuous arbitrary-
    # coordinate 2-D transport plus scalar dilation and a tightly bounded metric
    # residual. Transport moves both sides of thin features coherently; dilation
    # handles width restoration without raster phase heads.
    contour_transport_max_pixels: float = 8.0
    contour_dilation_max_pixels: float = 2.5
    contour_transport_min_jacobian: float = 0.35
    contour_transport_weight: float = 12.0
    contour_dilation_weight: float = 10.0
    contour_transport_smoothness_weight: float = 2.0
    contour_dilation_smoothness_weight: float = 1.5
    contour_transport_fold_weight: float = 10.0
    contour_soft_coverage_weight: float = 24.0
    contour_normal_offset_weight: float = 0.75
    contour_normal_offset_tangent_weight: float = 0.0
    contour_normal_offset_normal_weight: float = 0.0

    # Deprecated V9.8.11/V9.8.12 scalar actuator fields remain parseable only so
    # old resolved configs fail explicitly on schema rather than unknown keys.
    contour_normal_offset_max_pixels: float = 8.0
    contour_phase_refine_max_pixels: float = 0.75
    contour_projected_offset_weight: float = 8.0
    contour_phase_refine_weight: float = 5.0
    sdf_teacher_render_weight: float = 30.0
    sdf_teacher_gradient_weight: float = 18.0
    sdf_teacher_profile_weight: float = 18.0
    sdf_teacher_recovery_required: float = 0.70

    # V10.2 topology-anchored shared zero-crossing field.
    topology_field_feature_channels: int = 64
    topology_field_hidden_channels: int = 96
    topology_field_control_scale: int = 1
    topology_field_max_log_magnitude_delta: float = 8.0
    topology_field_magnitude_floor_pixels: float = 0.01
    topology_field_edit_band_pixels: float = 12.0
    topology_field_sdf_weight: float = 64.0
    topology_field_control_weight: float = 56.0
    topology_field_crossing_weight: float = 96.0
    topology_field_gradient_weight: float = 36.0
    topology_field_eikonal_weight: float = 8.0
    topology_field_curvature_weight: float = 8.0
    topology_field_render_weight: float = 32.0
    topology_field_render_gradient_weight: float = 20.0
    topology_field_render_profile_weight: float = 20.0


    # V10.4 topology-safe branch-smooth spline graph. Topology edits are confined
    # to the observable LR ambiguity band, then frozen while a separately refined
    # geometry branch fits smooth shared contour nodes and Hermite tangents.
    spline_graph_feature_channels: int = 64
    spline_graph_hidden_channels: int = 96
    spline_graph_control_scale: int = 2
    spline_graph_max_topology_delta_pixels: float = 8.0
    spline_graph_topology_edit_band_pixels: float = 4.0
    spline_graph_max_displacement_pixels: float = 4.0
    spline_graph_max_tangent_residual: float = 0.75
    spline_graph_edit_band_pixels: float = 12.0
    spline_graph_neighbour_radius: int = 2
    spline_graph_samples_per_span: int = 4
    spline_ordered_branch_enabled: bool = True
    spline_branch_smoothing_passes: int = 3
    spline_branch_smoothing_strength: float = 0.82
    spline_branch_smoothing_window: int = 51
    spline_branch_corner_cosine: float = 0.58
    spline_branch_corner_window: int = 5
    spline_graph_lr_multiplier: float = 4.0

    # V10.7.9 finite-width seam/ridge geometry. One medial centreline plus one
    # width owns both stroke sides; the legacy spline SDF remains the material-
    # boundary/topology fallback and B1a safety path.
    stroke_centerline_hidden_channels: int = 96
    stroke_centerline_max_delta_lr: float = 1.0
    stroke_centerline_max_tangent_residual: float = 0.75
    stroke_centerline_max_width_delta_pixels: float = 3.0
    stroke_centerline_ridge_min_depth_pixels: float = 0.45
    stroke_centerline_neighbourhood_radius_lr: int = 4
    stroke_centerline_segment_half_length_lr: float = 1.10
    stroke_centerline_initial_width_scale: float = 0.35
    stroke_centerline_center_weight: float = 12.0
    stroke_centerline_width_weight: float = 6.0
    stroke_centerline_tangent_weight: float = 0.75
    stroke_centerline_render_weight: float = 2.5

    # V10.7.9 staged synthetic structural authority: compact primitive classification
    # and bounded global parameter regression. Dense medial supervision is kept
    # only for old-checkpoint/source compatibility and has zero B1b authority.
    parametric_primitive_hidden_channels: int = 80
    parametric_primitive_class_weight: float = 8.0
    parametric_primitive_param_weight: float = 48.0
    parametric_primitive_render_weight: float = 4.0
    parametric_primitive_class_accuracy_required: float = 0.95
    parametric_primitive_param_mae_required: float = 0.040
    parametric_primitive_train_tiles_per_epoch: int = 448
    parametric_primitive_batch_size: int = 14
    parametric_primitive_lr_multiplier: float = 10.0
    # V10.7.9 stage budgets are maxima/guardrails only. B1b never advances
    # because an epoch number was reached; it advances only on held-out pass.
    parametric_primitive_classifier_epochs: int = 10
    parametric_primitive_parameter_epochs: int = 16
    parametric_primitive_integration_epochs: int = 6
    parametric_primitive_fit_steps: int = 128
    parametric_primitive_fit_learning_rate: float = 0.03

    # Legacy ordered-spline fixed-bank size retained for B1a/material-boundary
    # compatibility. V10.7.9 finite-width B1b uses the dedicated parametric
    # primitive bank above; the permanent 29-case audit ladder is separate.
    spline_geometry_fixed_bank_tiles: int = 128
    spline_graph_topology_control_weight: float = 96.0
    spline_graph_topology_sign_weight: float = 24.0
    spline_graph_point_weight: float = 10.0
    spline_graph_tangent_weight: float = 2.0
    spline_graph_sdf_weight: float = 48.0
    spline_graph_gradient_weight: float = 24.0
    spline_graph_eikonal_weight: float = 4.0
    spline_graph_curvature_weight: float = 10.0
    spline_graph_span_smoothness_weight: float = 28.0
    spline_graph_span_tangent_weight: float = 12.0
    spline_graph_span_separation_weight: float = 18.0
    # V10.6 makes ordered contour branches the geometry authority.  Metric
    # calibration stays disabled until the branch geometry itself satisfies the
    # anti-staircase structural proof.
    spline_graph_render_weight: float = 96.0
    spline_graph_render_gradient_weight: float = 48.0
    spline_graph_render_profile_weight: float = 64.0
    spline_metric_calibration_enabled: bool = False
    spline_metric_calibration_scale_delta: float = 0.25
    spline_metric_calibration_bias_pixels: float = 0.35
    spline_metric_calibration_band_pixels: float = 3.0
    spline_metric_offset_weight: float = 72.0
    spline_metric_eikonal_near_weight: float = 28.0
    spline_metric_scale_regularization_weight: float = 0.20
    spline_metric_bias_regularization_weight: float = 0.35
    sdf_oracle_render_band_mae_required: float = 0.025
    sdf_oracle_gradient_mae_required: float = 0.080
    # V10.7.9.1 adds a direct same-renderer pixel-equivalence gate. The legacy
    # cross-section profile bundle remains a strict alternate proof, but may not
    # reject an otherwise near-identical Panel 3 because of one unstable local
    # width/halo statistic. These limits are measured on the full P3/P2 images.
    sdf_oracle_global_mae_required: float = 0.0025
    sdf_oracle_global_mae_case_max_required: float = 0.0075
    sdf_oracle_render_band_mae_preview_required: float = 0.035
    sdf_oracle_profile_width_error_required: float = 0.10
    sdf_oracle_profile_correlation_required: float = 0.95
    sdf_oracle_core_halo_delta_required_8bit: float = 1.0

    # V10.2 shared-edge marching-squares field. The network predicts only
    # zero-crossing fractions on source-crossed LR edges; deterministic geometry
    # reconstructs the continuous SDF.
    edge_crossing_feature_channels: int = 64
    edge_crossing_hidden_channels: int = 96
    edge_crossing_max_logit_delta: float = 6.0
    edge_crossing_edit_band_pixels: float = 12.0
    edge_crossing_neighbour_radius: int = 1
    edge_crossing_lr_multiplier: float = 4.0
    edge_crossing_fraction_weight: float = 180.0
    edge_crossing_sdf_weight: float = 40.0
    edge_crossing_gradient_weight: float = 20.0
    edge_crossing_eikonal_weight: float = 4.0
    edge_crossing_curvature_weight: float = 8.0
    edge_crossing_render_weight: float = 40.0
    edge_crossing_render_gradient_weight: float = 24.0
    edge_crossing_render_profile_weight: float = 24.0

    # Legacy V10.0 oracle-patch fields retained for old resolved-config parsing only.
    oracle_patch_feature_channels: int = 64
    oracle_patch_hidden_channels: int = 96
    oracle_patch_footprint_lr: int = 3
    oracle_patch_max_delta_pixels: float = 8.0
    oracle_patch_max_coverage_logit_delta: float = 10.0
    oracle_patch_edit_band_pixels: float = 8.0
    oracle_patch_sdf_weight: float = 48.0
    oracle_patch_sign_weight: float = 16.0
    oracle_patch_coverage_weight: float = 48.0
    oracle_patch_coverage_bce_weight: float = 16.0
    oracle_patch_consistency_weight: float = 14.0
    oracle_patch_gradient_weight: float = 24.0
    oracle_patch_aggregate_coverage_weight: float = 36.0
    oracle_patch_render_weight: float = 30.0
    oracle_patch_render_gradient_weight: float = 18.0
    oracle_patch_render_profile_weight: float = 18.0

    # V9.9.3 local parametric boundary + direct coverage-profile specialist.
    implicit_boundary_feature_channels: int = 48
    implicit_boundary_residual_max_pixels: float = 0.75  # deprecated parse-only
    implicit_boundary_supersample_grid: int = 3
    # V9.9.3 local analytic line/arc primitive geometry.
    parametric_boundary_control_scale: int = 1
    parametric_boundary_max_offset_pixels: float = 6.0
    parametric_boundary_max_normal_correction: float = 1.5
    parametric_boundary_max_curvature_per_pixel: float = 0.35
    parametric_boundary_max_ribbon_half_width_pixels: float = 6.0
    parametric_boundary_anchor_weight: float = 24.0
    parametric_boundary_normal_weight: float = 14.0
    parametric_boundary_curvature_weight: float = 8.0
    parametric_boundary_offset_smoothness_weight: float = 10.0
    boundary_specialist_channels: int = 48
    boundary_specialist_band_pixels: float = 5.0
    boundary_specialist_logit_delta_max: float = 16.0
    boundary_specialist_coverage_weight: float = 48.0
    boundary_specialist_coverage_bce_weight: float = 18.0
    boundary_specialist_coverage_gradient_weight: float = 42.0
    boundary_specialist_profile_moment_weight: float = 48.0
    boundary_specialist_gradient_weight: float = 28.0
    boundary_specialist_profile_weight: float = 36.0
    boundary_specialist_recovery_required: float = 0.70
    # Deprecated V9.9.0 additive-coverage limit: parse only.
    boundary_specialist_max_coverage_delta: float = 0.35
    benefit_selector_channels: int = 24
    benefit_selector_weight: float = 8.0
    structural_line_jitter_required_pixels: float = 0.35
    structural_curve_roughness_required_pixels: float = 0.45
    structural_line_staircase_recovery_required: float = 0.90

    # Deprecated V9.8.7-V9.8.10 delta-SDF fields remain parseable so old JSON
    # produces an explicit schema mismatch rather than a config parse failure.
    sdf_delta_max_pixels: float = 12.0
    sdf_delta_surface_weight: float = 10.0
    # Deprecated V9.8.10 dense-delta regularisation controls. They remain
    # parseable for old resolved configs but do not provide V9.9.3 geometry
    # authority; active geometry authority lives in the local parametric decoder instead.
    sdf_delta_tangent_weight: float = 3.00
    sdf_delta_laplacian_weight: float = 1.50
    sdf_improvement_regret_weight: float = 6.0
    sdf_improvement_margin_pixels: float = 0.05
    geometry_need_floor: float = 0.12
    geometry_need_sdf_scale_pixels: float = 4.0

    # Stage-B is primarily relative to the LR baseline. Absolute thresholds are
    # retained only as catastrophic safety rails.
    sdf_relative_gain_required: float = 0.25
    sdf_relative_win_fraction: float = 0.65
    sdf_relative_regression_fraction: float = 0.20
    sdf_catastrophic_chamfer_pixels: float = 48.0
    sdf_missing_contour_tolerance: float = 0.00

    # V9.4.3 flow-supervision fields are retained only so old JSONs produce a
    # clear schema mismatch rather than failing config parsing.
    direct_flow_weight: float = 0.75
    bootstrap_direct_flow_weight: float = 12.00
    flow_activity_threshold_source_pixels: float = 0.030

    # Reconstruction losses.
    albedo_weight: float = 1.00
    albedo_gradient_weight: float = 0.50
    albedo_pyramid_weight: float = 0.25
    normal_weight: float = 0.85
    normal_gradient_weight: float = 0.30
    roughness_weight: float = 0.20
    emissive_weight: float = 0.16
    material_weight: float = 0.30
    sdf_weight: float = 4.00
    edge_weight: float = 2.00
    orientation_weight: float = 1.00
    cross_map_weight: float = 0.20
    seam_weight: float = 0.12

    # Fidelity-first terms.
    regret_weight: float = 2.50
    normal_regret_weight: float = 1.25
    gate_target_weight: float = 0.60
    residual_l1_weight: float = 0.18
    fine_zero_mean_weight: float = 0.20
    detail_laplacian_weight: float = 0.28
    ringing_regret_weight: float = 0.18
    unchanged_region_weight: float = 1.00

    # Geometric-coherence terms. These penalise contour wiggle rather than
    # merely rewarding local sharpness. Tangents are axial (+t == -t), and
    # curvature is allowed where the target contains a genuine arc or corner.
    geometric_alignment_weight: float = 0.48
    tangent_coherence_weight: float = 0.36
    curvature_coherence_weight: float = 0.30
    sdf_curvature_weight: float = 0.16
    tangent_variation_margin: float = 0.018
    curvature_variation_margin: float = 0.020
    synthetic_geometry_probability: float = 0.82
    synthetic_geometry_loss_boost: float = 1.50

    local_regret_patch: int = 8
    gate_error_scale: float = 0.08
    gate_edge_bonus: float = 0.15
    unchanged_error_threshold: float = 0.025
    maximum_validation_regression_fraction: float = 0.08

    # Synthetic degradation distribution used for controlled reconstruction A/B.
    lod_bias_min: float = 0.50
    lod_bias_max: float = 1.80
    anisotropic_blur_probability: float = 0.80
    bc_block_probability: float = 0.85
    chroma_loss_probability: float = 0.60
    ringing_probability: float = 0.45
    halo_probability: float = 0.45
    renderer_sampling_probability: float = 0.65
    renderer_anisotropy_max: float = 4.0
    renderer_subpixel_jitter: float = 0.45

    seed: int = 20260806
    output_dir: str = "artifacts/nsamdr/neural_v9"
    checkpoint_name: str = "nsamdr_v9_fidelity.pt"
    metadata_name: str = "nsamdr_v9_fidelity.json"
    training_state_name: str = "nsamdr_v9_training_state.pt"
    diagnostics_dir_name: str = "diagnostics"
    inference_tile_size: int = 128
    inference_overlap: int = 24
    device: str = "cuda"
    cuda_device_index: int = 0
    matmul_precision: str = "high"
    gradient_clip_norm: float = 1.25
    amp_initial_scale: float = 512.0
    amp_minimum_scale: float = 1.0
    amp_overflow_retries: int = 8
    parameter_finite_check_interval: int = 64

    performance_profile: str = "optimized"
    data_loader_workers: int = 4
    data_loader_prefetch_factor: int = 2
    data_loader_persistent_workers: bool = True
    cuda_prefetch: bool = True

    # Legacy field retained for old config/checkpoint compatibility. V9.2.2 no
    # longer applies a fixed per-process CUDA fraction; memory is governed
    # reactively between batches.
    cuda_memory_fraction: float = 0.88

    # Elastic CUDA sharing. These are runtime-only and do not change the model,
    # objective or resume compatibility.
    reactive_vram_enabled: bool = True
    reactive_vram_target_free_fraction: float = 0.30
    reactive_vram_pause_free_fraction: float = 0.12
    reactive_vram_resume_free_fraction: float = 0.20
    reactive_vram_expand_hysteresis_fraction: float = 0.05
    reactive_vram_expand_stable_steps: int = 8
    reactive_vram_poll_seconds: float = 0.50
    reactive_vram_oom_retries: int = 3
    reactive_vram_release_cache: bool = True

    # Strict coexistence envelope. Training only enters a batch when the
    # predicted V9 transient requirement plus this foreground-application
    # reserve fits in physical VRAM. The reserve is intentionally large.
    reactive_vram_burst_reserve_fraction: float = 0.35
    reactive_vram_stability_samples: int = 3
    reactive_vram_stability_interval_seconds: float = 0.20
    # Legacy compatibility switch. Hard PyTorch per-process ceilings are no
    # longer applied because allocator-reserved cache counts against the cap and
    # can trigger false OOM while device-wide VRAM is still available.
    reactive_vram_dynamic_allocator_ceiling: bool = False
    reactive_vram_start_in_offload: bool = True

    # Host-memory safety for CPU-saved autograd activations. The old 20/25%
    # free-RAM guard deadlocked on Windows because the process working set may
    # retain already-freed offload pages. Keep only a modest bootstrap guard;
    # after one successful offload step the runtime uses a small absolute
    # critical floor and reuses the established working set.
    reactive_host_pause_free_fraction: float = 0.05
    reactive_host_resume_free_fraction: float = 0.08

    channels_last: bool = False
    amp_dtype: str = "auto"
    fused_optimizer: bool = True
    cudnn_benchmark: bool = True
    allow_tf32: bool = True
    loss_precision: str = "mixed"
    torch_compile_mode: str = "off"

    @property
    def total_epochs(self) -> int:
        return (
            self.identity_epochs + self.residual_epochs + self.seam_proof_epochs
            + self.seam_authority_epochs + self.boundary_epochs + self.detail_epochs
            + self.physical_finetune_epochs
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["widths"] = list(self.widths)
        payload["blocks_per_level"] = list(self.blocks_per_level)
        payload["decoder_blocks"] = list(self.decoder_blocks)
        return payload

    @classmethod
    def load(cls, path: Path | None) -> "V9Config":
        config = cls()
        if path is not None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            aliases = {
                "datasetManifest": "dataset_manifest", "datasetRoot": "dataset_root",
                "maxFamilies": "max_families", "cropsPerFamily": "crops_per_family",
                "sourceCropSize": "source_crop_size", "minSourceDimension": "min_source_dimension",
                "minAuxiliaryDimension": "min_auxiliary_dimension", "validationFraction": "validation_fraction",
                "requireCompletePbrFamily": "require_complete_pbr_family",
                "identityEpochs": "identity_epochs", "residualEpochs": "residual_epochs",
                "seamProofEpochs": "seam_proof_epochs", "seamAuthorityEpochs": "seam_authority_epochs",
                "seamPhaseOnlyReconstruction": "seam_phase_only_reconstruction",
                "seamPhaseResidualWeight": "seam_phase_residual_weight",
                "seamMicroproofEnabled": "seam_microproof_enabled",
                "seamMicroproofSteps": "seam_microproof_steps",
                "seamMicroproofRecoveryRequired": "seam_microproof_recovery_required",
                "boundaryEpochs": "boundary_epochs", "detailEpochs": "detail_epochs",
                "physicalFinetuneEpochs": "physical_finetune_epochs", "tilesPerEpoch": "tiles_per_epoch",
                "validationTiles": "validation_tiles", "batchSize": "batch_size",
                "identityLearningRate": "identity_learning_rate", "learningRate": "learning_rate",
                "boundaryLearningRate": "boundary_learning_rate", "detailLearningRate": "detail_learning_rate",
                "finetuneLearningRate": "finetune_learning_rate", "weightDecay": "weight_decay",
                "optimizerName": "optimizer_name", "optimizerBeta1": "optimizer_beta1",
                "optimizerBeta2": "optimizer_beta2", "schedulerName": "scheduler_name",
                "schedulerMinLrRatio": "scheduler_min_lr_ratio",
                "tileSize": "tile_size", "targetScale": "target_scale",
                "boundaryCandidateCount": "boundary_candidate_count",
                "boundarySamplingProbability": "boundary_sampling_probability",
                "inputChannels": "input_channels", "materialClasses": "material_classes",
                "blocksPerLevel": "blocks_per_level", "decoderBlocks": "decoder_blocks",
                "attentionHeads": "attention_heads", "attentionWindow": "attention_window",
                "dropPath": "drop_path", "orientationNormalizationEpsilon": "orientation_normalization_epsilon",
                "albedoMediumDelta": "albedo_medium_delta", "albedoFineDelta": "albedo_fine_delta",
                "normalMediumDelta": "normal_medium_delta", "normalFineDelta": "normal_fine_delta",
                "materialDelta": "material_delta", "auxiliaryDelta": "auxiliary_delta",
                "initialGateBias": "initial_gate_bias",
                "appearanceEnabled": "appearance_enabled",
                "appearanceEdgeSuppression": "appearance_edge_suppression",
                "detailReconstructionEnabled": "detail_reconstruction_enabled",
                "detailFeatureChannels": "detail_feature_channels",
                "detailMidChannels": "detail_mid_channels",
                "detailHrChannels": "detail_hr_channels",
                "detailAlbedoMaxDelta": "detail_albedo_max_delta",
                "detailNormalMaxDelta": "detail_normal_max_delta",
                "detailMaterialMaxDelta": "detail_material_max_delta",
                "detailConfidenceInitialBias": "detail_confidence_initial_bias",
                "detailRegretInitialBias": "detail_regret_initial_bias",
                "seamDirectionalEnabled": "seam_directional_enabled",
                "seamDirectionalChannels": "seam_directional_channels",
                "seamDirectionalAngleBins": "seam_directional_angle_bins",
                "seamDirectionalKernelSize": "seam_directional_kernel_size",
                "seamDirectionalKernelResidualScale": "seam_directional_kernel_residual_scale",
                "seamPhaseSrChannels": "seam_phase_sr_channels",
                "seamPhaseSrMaxDelta": "seam_phase_sr_max_delta",
                "seamStructureTensorRadius": "seam_structure_tensor_radius",
                "seamStructureStrengthGain": "seam_structure_strength_gain",
                "seamCoherenceFloor": "seam_coherence_floor",
                "seamTangentSamplePixels": "seam_tangent_sample_pixels",
                "seamNormalSamplePixels": "seam_normal_sample_pixels",
                "seamMaxNormalSharpen": "seam_max_normal_sharpen",
                "seamMaxAuthority": "seam_max_authority",
                "seamGeometryBandPixels": "seam_geometry_band_pixels",
                "seamDirectionalWeight": "seam_directional_weight",
                "seamLrMultiplier": "seam_lr_multiplier",
                "seamTangentSmoothnessWeight": "seam_tangent_smoothness_weight",
                "seamNormalProfileWeight": "seam_normal_profile_weight",
                "seamAuthorityRegularizationWeight": "seam_authority_regularization_weight",
                "seamAuthorityTeacherWeight": "seam_authority_teacher_weight",
                "seamReconstructionWeight": "seam_reconstruction_weight",
                "seamForcedRecoveryRequired": "seam_forced_recovery_required",
                "seamAuthorityIouRequired": "seam_authority_iou_required",
                "seamRidgeWeight": "seam_ridge_weight",
                "seamMissingDetailScale": "seam_missing_detail_scale",
                "seamTeacherDilationPixels": "seam_teacher_dilation_pixels",
                "structuralStalePatience": "structural_stale_patience",
                "seamProjectedViewWeight": "seam_projected_view_weight",
                "ddsCodecDegradationEnabled": "dds_codec_degradation_enabled",
                "ddsCodecProbability": "dds_codec_probability",
                "ddsCodecBlendMin": "dds_codec_blend_min",
                "ddsCodecBlendMax": "dds_codec_blend_max",
                "detailContrastWeight": "detail_contrast_weight",
                "detailCrossMapWeight": "detail_cross_map_weight",
                "detailConfidenceWeight": "detail_confidence_weight",
                "detailRegretClassifierWeight": "detail_regret_classifier_weight",
                "detailRecoveryRequired": "detail_recovery_required",
                "detailGradientRecoveryRequired": "detail_gradient_recovery_required",
                "detailNormalRecoveryRequired": "detail_normal_recovery_required",
                "detailWinFractionRequired": "detail_win_fraction_required",
                "detailRegressionFractionMax": "detail_regression_fraction_max",
                "geometryDisplacementSourcePixels": "geometry_displacement_source_pixels",
                "geometryEdgeSupportRadius": "geometry_edge_support_radius",
                "geometryDisplacementPixels": "geometry_displacement_pixels",
                "contourSdfMaxDistancePixels": "contour_sdf_max_distance_pixels",
                "boundaryRendererBandPixels": "boundary_renderer_band_pixels",
                "boundaryRendererSamplePixels": "boundary_renderer_sample_pixels",
                "boundaryRendererHardWidthPixels": "boundary_renderer_hard_width_pixels",
                "boundaryRendererSoftWidthPixels": "boundary_renderer_soft_width_pixels",
                "boundaryRendererEdgeThreshold": "boundary_renderer_edge_threshold",
                "boundaryRendererConfidenceFloor": "boundary_renderer_confidence_floor",
                "boundaryRendererGateGain": "boundary_renderer_gate_gain",
                "boundaryRendererFarSampleMultiplier": "boundary_renderer_far_sample_multiplier",
                "boundaryRendererFarSampleWeight": "boundary_renderer_far_sample_weight",
                "boundaryGateInitialBias": "boundary_gate_initial_bias",
                "implicitSdfHiddenChannels": "implicit_sdf_hidden_channels",
                "implicitSdfCoordinateScale": "implicit_sdf_coordinate_scale",
                "implicitSdfResidualPixels": "implicit_sdf_residual_pixels",
                "coarseSdfSurfaceWeight": "coarse_sdf_surface_weight",
                "sdfResidualL1Weight": "sdf_residual_l1_weight",
                "boundaryRendererPlateauSamples": "boundary_renderer_plateau_samples",
                "boundaryRendererPlateauMaxMultiplier": "boundary_renderer_plateau_max_multiplier",
                "boundaryRendererPlateauStabilityScale": "boundary_renderer_plateau_stability_scale",
                "displacementSmoothnessWeight": "displacement_smoothness_weight",
                "displacementOffContourWeight": "displacement_off_contour_weight",
                "displacementTangentWeight": "displacement_tangent_weight",
                "displacementSparsityWeight": "displacement_sparsity_weight",
                "geometryRegretWeight": "geometry_regret_weight",
                "geometryRegretMargin": "geometry_regret_margin",
                "edgeRegretMultiplier": "edge_regret_multiplier",
                "geometryPhotometricWeight": "geometry_photometric_weight",
                "boundaryPhotometricWeight": "boundary_photometric_weight",
                "boundaryProfileWeight": "boundary_profile_weight",
                "boundaryFuzzWeight": "boundary_fuzz_weight",
                "boundaryHaloWeight": "boundary_halo_weight",
                "boundaryHardnessWeight": "boundary_hardness_weight",
                "boundaryGateWeight": "boundary_gate_weight",
                "boundaryOffContourWeight": "boundary_off_contour_weight",
                "boundaryRegretWeight": "boundary_regret_weight",
                "boundarySdfZeroWeight": "boundary_sdf_zero_weight",
                "boundaryEdgeSdfConsistencyWeight": "boundary_edge_sdf_consistency_weight",
                "boundaryPixelRegretWeight": "boundary_pixel_regret_weight",
                "boundaryGateNeedScale": "boundary_gate_need_scale",
                "boundaryGateExactFloor": "boundary_gate_exact_floor",
                "sdfSurfaceWeight": "sdf_surface_weight",
                "sdfSignWeight": "sdf_sign_weight",
                "sdfTopologyWeight": "sdf_topology_weight",
                "sdfTopologyMarginPixels": "sdf_topology_margin_pixels",
                "sdfTopologyCorePixels": "sdf_topology_core_pixels",
                "sdfTopologyBandPixels": "sdf_topology_band_pixels",
                "sdfTopologyWorstFraction": "sdf_topology_worst_fraction",
                "sdfEikonalWeight": "sdf_eikonal_weight",
                "sdfGradientAlignmentWeight": "sdf_gradient_alignment_weight",
                "sdfMetricGradientWeight": "sdf_metric_gradient_weight",
                "sdfSignGaugeInvariant": "sdf_sign_gauge_invariant",
                "sdfMetricBandPixels": "sdf_metric_band_pixels",
                "sdfCoarseInitStd": "sdf_coarse_init_std",
                "sdfSyntheticValidationTiles": "sdf_synthetic_validation_tiles",
                "sdfZeroBandPixels": "sdf_zero_band_pixels",
                "sdfBootstrapResidualPixels": "sdf_bootstrap_residual_pixels",
                "sdfProofResidualPixels": "sdf_proof_residual_pixels",
                "sdfProofRendererWeight": "sdf_proof_renderer_weight",
                "contourTransportMaxPixels": "contour_transport_max_pixels",
                "contourDilationMaxPixels": "contour_dilation_max_pixels",
                "contourTransportMinJacobian": "contour_transport_min_jacobian",
                "contourTransportWeight": "contour_transport_weight",
                "contourDilationWeight": "contour_dilation_weight",
                "contourTransportSmoothnessWeight": "contour_transport_smoothness_weight",
                "contourDilationSmoothnessWeight": "contour_dilation_smoothness_weight",
                "contourTransportFoldWeight": "contour_transport_fold_weight",
                "contourNormalOffsetMaxPixels": "contour_normal_offset_max_pixels",
                "contourPhaseRefineMaxPixels": "contour_phase_refine_max_pixels",
                "contourProjectedOffsetWeight": "contour_projected_offset_weight",
                "contourPhaseRefineWeight": "contour_phase_refine_weight",
                "contourSoftCoverageWeight": "contour_soft_coverage_weight",
                "contourNormalOffsetWeight": "contour_normal_offset_weight",
                "contourNormalOffsetTangentWeight": "contour_normal_offset_tangent_weight",
                "contourNormalOffsetNormalWeight": "contour_normal_offset_normal_weight",
                "sdfTeacherRenderWeight": "sdf_teacher_render_weight",
                "sdfTeacherGradientWeight": "sdf_teacher_gradient_weight",
                "sdfTeacherProfileWeight": "sdf_teacher_profile_weight",
                "sdfTeacherRecoveryRequired": "sdf_teacher_recovery_required",
                "implicitBoundaryFeatureChannels": "implicit_boundary_feature_channels",
                "implicitBoundaryResidualMaxPixels": "implicit_boundary_residual_max_pixels",
                "implicitBoundarySupersampleGrid": "implicit_boundary_supersample_grid",
                "topologyFieldFeatureChannels": "topology_field_feature_channels",
                "topologyFieldHiddenChannels": "topology_field_hidden_channels",
                "topologyFieldControlScale": "topology_field_control_scale",
                "topologyFieldMaxLogMagnitudeDelta": "topology_field_max_log_magnitude_delta",
                "topologyFieldMagnitudeFloorPixels": "topology_field_magnitude_floor_pixels",
                "topologyFieldEditBandPixels": "topology_field_edit_band_pixels",
                "topologyFieldSdfWeight": "topology_field_sdf_weight",
                "topologyFieldControlWeight": "topology_field_control_weight",
                "topologyFieldCrossingWeight": "topology_field_crossing_weight",
                "topologyFieldGradientWeight": "topology_field_gradient_weight",
                "topologyFieldEikonalWeight": "topology_field_eikonal_weight",
                "topologyFieldCurvatureWeight": "topology_field_curvature_weight",
                "topologyFieldRenderWeight": "topology_field_render_weight",
                "topologyFieldRenderGradientWeight": "topology_field_render_gradient_weight",
                "topologyFieldRenderProfileWeight": "topology_field_render_profile_weight",
                "splineGraphRenderWeight": "spline_graph_render_weight",
                "splineGraphRenderGradientWeight": "spline_graph_render_gradient_weight",
                "splineGraphRenderProfileWeight": "spline_graph_render_profile_weight",
                "splineOrderedBranchEnabled": "spline_ordered_branch_enabled",
                "splineBranchSmoothingPasses": "spline_branch_smoothing_passes",
                "splineBranchSmoothingStrength": "spline_branch_smoothing_strength",
                "splineBranchSmoothingWindow": "spline_branch_smoothing_window",
                "splineBranchCornerCosine": "spline_branch_corner_cosine",
                "splineBranchCornerWindow": "spline_branch_corner_window",
                "strokeCenterlineHiddenChannels": "stroke_centerline_hidden_channels",
                "strokeCenterlineMaxDeltaLr": "stroke_centerline_max_delta_lr",
                "strokeCenterlineMaxTangentResidual": "stroke_centerline_max_tangent_residual",
                "strokeCenterlineMaxWidthDeltaPixels": "stroke_centerline_max_width_delta_pixels",
                "strokeCenterlineRidgeMinDepthPixels": "stroke_centerline_ridge_min_depth_pixels",
                "strokeCenterlineNeighbourhoodRadiusLr": "stroke_centerline_neighbourhood_radius_lr",
                "strokeCenterlineSegmentHalfLengthLr": "stroke_centerline_segment_half_length_lr",
                "strokeCenterlineInitialWidthScale": "stroke_centerline_initial_width_scale",
                "strokeCenterlineCenterWeight": "stroke_centerline_center_weight",
                "strokeCenterlineWidthWeight": "stroke_centerline_width_weight",
                "strokeCenterlineTangentWeight": "stroke_centerline_tangent_weight",
                "strokeCenterlineRenderWeight": "stroke_centerline_render_weight",
                "parametricPrimitiveHiddenChannels": "parametric_primitive_hidden_channels",
                "parametricPrimitiveClassWeight": "parametric_primitive_class_weight",
                "parametricPrimitiveParamWeight": "parametric_primitive_param_weight",
                "parametricPrimitiveRenderWeight": "parametric_primitive_render_weight",
                "parametricPrimitiveClassAccuracyRequired": "parametric_primitive_class_accuracy_required",
                "parametricPrimitiveParamMaeRequired": "parametric_primitive_param_mae_required",
                "parametricPrimitiveTrainTilesPerEpoch": "parametric_primitive_train_tiles_per_epoch",
                "parametricPrimitiveBatchSize": "parametric_primitive_batch_size",
                "parametricPrimitiveLrMultiplier": "parametric_primitive_lr_multiplier",
                "parametricPrimitiveClassifierEpochs": "parametric_primitive_classifier_epochs",
                "parametricPrimitiveParameterEpochs": "parametric_primitive_parameter_epochs",
                "parametricPrimitiveIntegrationEpochs": "parametric_primitive_integration_epochs",
                "parametricPrimitiveFitSteps": "parametric_primitive_fit_steps",
                "parametricPrimitiveFitLearningRate": "parametric_primitive_fit_learning_rate",
                "splineGeometryFixedBankTiles": "spline_geometry_fixed_bank_tiles",
                "splineMetricCalibrationEnabled": "spline_metric_calibration_enabled",
                "splineMetricCalibrationScaleDelta": "spline_metric_calibration_scale_delta",
                "splineMetricCalibrationBiasPixels": "spline_metric_calibration_bias_pixels",
                "splineMetricCalibrationBandPixels": "spline_metric_calibration_band_pixels",
                "splineMetricOffsetWeight": "spline_metric_offset_weight",
                "splineMetricEikonalNearWeight": "spline_metric_eikonal_near_weight",
                "splineMetricScaleRegularizationWeight": "spline_metric_scale_regularization_weight",
                "splineMetricBiasRegularizationWeight": "spline_metric_bias_regularization_weight",
                "sdfOracleRenderBandMaeRequired": "sdf_oracle_render_band_mae_required",
                "sdfOracleGradientMaeRequired": "sdf_oracle_gradient_mae_required",
                "sdfOracleGlobalMaeRequired": "sdf_oracle_global_mae_required",
                "sdfOracleGlobalMaeCaseMaxRequired": "sdf_oracle_global_mae_case_max_required",
                "sdfOracleRenderBandMaePreviewRequired": "sdf_oracle_render_band_mae_preview_required",
                "sdfOracleProfileWidthErrorRequired": "sdf_oracle_profile_width_error_required",
                "sdfOracleProfileCorrelationRequired": "sdf_oracle_profile_correlation_required",
                "sdfOracleCoreHaloDeltaRequired8bit": "sdf_oracle_core_halo_delta_required_8bit",
                "oraclePatchFeatureChannels": "oracle_patch_feature_channels",
                "oraclePatchHiddenChannels": "oracle_patch_hidden_channels",
                "oraclePatchFootprintLR": "oracle_patch_footprint_lr",
                "oraclePatchMaxDeltaPixels": "oracle_patch_max_delta_pixels",
                "oraclePatchMaxCoverageLogitDelta": "oracle_patch_max_coverage_logit_delta",
                "oraclePatchEditBandPixels": "oracle_patch_edit_band_pixels",
                "oraclePatchSdfWeight": "oracle_patch_sdf_weight",
                "oraclePatchSignWeight": "oracle_patch_sign_weight",
                "oraclePatchCoverageWeight": "oracle_patch_coverage_weight",
                "oraclePatchCoverageBceWeight": "oracle_patch_coverage_bce_weight",
                "oraclePatchConsistencyWeight": "oracle_patch_consistency_weight",
                "oraclePatchGradientWeight": "oracle_patch_gradient_weight",
                "oraclePatchAggregateCoverageWeight": "oracle_patch_aggregate_coverage_weight",
                "oraclePatchRenderWeight": "oracle_patch_render_weight",
                "oraclePatchRenderGradientWeight": "oracle_patch_render_gradient_weight",
                "oraclePatchRenderProfileWeight": "oracle_patch_render_profile_weight",
                "parametricBoundaryControlScale": "parametric_boundary_control_scale",
                "parametricBoundaryMaxOffsetPixels": "parametric_boundary_max_offset_pixels",
                "parametricBoundaryMaxNormalCorrection": "parametric_boundary_max_normal_correction",
                "parametricBoundaryMaxCurvaturePerPixel": "parametric_boundary_max_curvature_per_pixel",
                "parametricBoundaryMaxRibbonHalfWidthPixels": "parametric_boundary_max_ribbon_half_width_pixels",
                "parametricBoundaryAnchorWeight": "parametric_boundary_anchor_weight",
                "parametricBoundaryNormalWeight": "parametric_boundary_normal_weight",
                "parametricBoundaryCurvatureWeight": "parametric_boundary_curvature_weight",
                "parametricBoundaryOffsetSmoothnessWeight": "parametric_boundary_offset_smoothness_weight",
                "boundarySpecialistChannels": "boundary_specialist_channels",
                "boundarySpecialistBandPixels": "boundary_specialist_band_pixels",
                "boundarySpecialistMaxCoverageDelta": "boundary_specialist_max_coverage_delta",
                "boundarySpecialistLogitDeltaMax": "boundary_specialist_logit_delta_max",
                "boundarySpecialistCoverageWeight": "boundary_specialist_coverage_weight",
                "boundarySpecialistCoverageBceWeight": "boundary_specialist_coverage_bce_weight",
                "boundarySpecialistCoverageGradientWeight": "boundary_specialist_coverage_gradient_weight",
                "boundarySpecialistProfileMomentWeight": "boundary_specialist_profile_moment_weight",
                "boundarySpecialistGradientWeight": "boundary_specialist_gradient_weight",
                "boundarySpecialistProfileWeight": "boundary_specialist_profile_weight",
                "boundarySpecialistRecoveryRequired": "boundary_specialist_recovery_required",
                "benefitSelectorChannels": "benefit_selector_channels",
                "benefitSelectorWeight": "benefit_selector_weight",
                "structuralLineJitterRequiredPixels": "structural_line_jitter_required_pixels",
                "structuralCurveRoughnessRequiredPixels": "structural_curve_roughness_required_pixels",
                "structuralLineStaircaseRecoveryRequired": "structural_line_staircase_recovery_required",
                "directFlowWeight": "direct_flow_weight",
                "bootstrapDirectFlowWeight": "bootstrap_direct_flow_weight",
                "flowActivityThresholdSourcePixels": "flow_activity_threshold_source_pixels",
                "albedoWeight": "albedo_weight", "albedoGradientWeight": "albedo_gradient_weight",
                "albedoPyramidWeight": "albedo_pyramid_weight", "normalWeight": "normal_weight",
                "normalGradientWeight": "normal_gradient_weight", "roughnessWeight": "roughness_weight",
                "emissiveWeight": "emissive_weight", "materialWeight": "material_weight",
                "sdfWeight": "sdf_weight", "edgeWeight": "edge_weight",
                "orientationWeight": "orientation_weight", "crossMapWeight": "cross_map_weight",
                "seamWeight": "seam_weight", "regretWeight": "regret_weight",
                "normalRegretWeight": "normal_regret_weight", "gateTargetWeight": "gate_target_weight",
                "residualL1Weight": "residual_l1_weight", "fineZeroMeanWeight": "fine_zero_mean_weight",
                "detailLaplacianWeight": "detail_laplacian_weight",
                "ringingRegretWeight": "ringing_regret_weight",
                "unchangedRegionWeight": "unchanged_region_weight",
                "geometricAlignmentWeight": "geometric_alignment_weight",
                "tangentCoherenceWeight": "tangent_coherence_weight",
                "curvatureCoherenceWeight": "curvature_coherence_weight",
                "sdfCurvatureWeight": "sdf_curvature_weight",
                "tangentVariationMargin": "tangent_variation_margin",
                "curvatureVariationMargin": "curvature_variation_margin",
                "syntheticGeometryProbability": "synthetic_geometry_probability",
                "syntheticGeometryLossBoost": "synthetic_geometry_loss_boost",
                "localRegretPatch": "local_regret_patch",
                "gateErrorScale": "gate_error_scale", "gateEdgeBonus": "gate_edge_bonus",
                "unchangedErrorThreshold": "unchanged_error_threshold",
                "maximumValidationRegressionFraction": "maximum_validation_regression_fraction",
                "rendererSamplingProbability": "renderer_sampling_probability",
                "rendererAnisotropyMax": "renderer_anisotropy_max",
                "rendererSubpixelJitter": "renderer_subpixel_jitter",
                "cudaMemoryFraction": "cuda_memory_fraction",
                "reactiveVramEnabled": "reactive_vram_enabled",
                "reactiveVramTargetFreeFraction": "reactive_vram_target_free_fraction",
                "reactiveVramPauseFreeFraction": "reactive_vram_pause_free_fraction",
                "reactiveVramResumeFreeFraction": "reactive_vram_resume_free_fraction",
                "reactiveVramExpandHysteresisFraction": "reactive_vram_expand_hysteresis_fraction",
                "reactiveVramExpandStableSteps": "reactive_vram_expand_stable_steps",
                "reactiveVramPollSeconds": "reactive_vram_poll_seconds",
                "reactiveVramOomRetries": "reactive_vram_oom_retries",
                "reactiveVramReleaseCache": "reactive_vram_release_cache",
                "reactiveVramBurstReserveFraction": "reactive_vram_burst_reserve_fraction",
                "reactiveVramStabilitySamples": "reactive_vram_stability_samples",
                "reactiveVramStabilityIntervalSeconds": "reactive_vram_stability_interval_seconds",
                "reactiveVramDynamicAllocatorCeiling": "reactive_vram_dynamic_allocator_ceiling",
                "reactiveVramStartInOffload": "reactive_vram_start_in_offload",
                "reactiveHostPauseFreeFraction": "reactive_host_pause_free_fraction",
                "reactiveHostResumeFreeFraction": "reactive_host_resume_free_fraction",
                "lodBiasMin": "lod_bias_min", "lodBiasMax": "lod_bias_max",
                "anisotropicBlurProbability": "anisotropic_blur_probability",
                "bcBlockProbability": "bc_block_probability",
                "chromaLossProbability": "chroma_loss_probability", "ringingProbability": "ringing_probability",
                "haloProbability": "halo_probability", "outputDir": "output_dir",
                "checkpointName": "checkpoint_name", "metadataName": "metadata_name",
                "trainingStateName": "training_state_name", "diagnosticsDirName": "diagnostics_dir_name",
                "inferenceTileSize": "inference_tile_size", "inferenceOverlap": "inference_overlap",
                "cudaDeviceIndex": "cuda_device_index", "matmulPrecision": "matmul_precision",
                "gradientClipNorm": "gradient_clip_norm", "ampInitialScale": "amp_initial_scale",
                "ampMinimumScale": "amp_minimum_scale", "ampOverflowRetries": "amp_overflow_retries",
                "parameterFiniteCheckInterval": "parameter_finite_check_interval",
                "performanceProfile": "performance_profile", "dataLoaderWorkers": "data_loader_workers",
                "dataLoaderPrefetchFactor": "data_loader_prefetch_factor",
                "dataLoaderPersistentWorkers": "data_loader_persistent_workers", "cudaPrefetch": "cuda_prefetch",
                "channelsLast": "channels_last", "ampDtype": "amp_dtype",
                "fusedOptimizer": "fused_optimizer", "cudnnBenchmark": "cudnn_benchmark",
                "allowTf32": "allow_tf32", "lossPrecision": "loss_precision",
                "torchCompileMode": "torch_compile_mode",
            }
            for key, value in payload.items():
                target = aliases.get(key, key)
                if hasattr(config, target):
                    if target in {"widths", "blocks_per_level", "decoder_blocks"}:
                        value = tuple(int(item) for item in value)
                    setattr(config, target, value)
        config.validate()
        return config

    def validate(self) -> None:
        self.max_families = max(8, min(int(self.max_families), 4096))
        self.crops_per_family = max(2, min(int(self.crops_per_family), 128))
        self.source_crop_size = max(512, min(int(self.source_crop_size), 2048))
        self.source_crop_size -= self.source_crop_size % 32
        self.min_source_dimension = max(512, int(self.min_source_dimension))
        self.min_auxiliary_dimension = max(32, min(int(self.min_auxiliary_dimension), self.min_source_dimension))
        self.validation_fraction = float(min(max(self.validation_fraction, 0.02), 0.40))
        for name in ("identity_epochs", "residual_epochs", "detail_epochs"):
            setattr(self, name, max(1, min(int(getattr(self, name)), 100)))
        for name in ("seam_proof_epochs", "seam_authority_epochs", "boundary_epochs"):
            setattr(self, name, max(0, min(int(getattr(self, name)), 100)))
        self.physical_finetune_epochs = max(1, min(int(self.physical_finetune_epochs), 100))
        self.tiles_per_epoch = max(4, min(int(self.tiles_per_epoch), 1_000_000))
        self.validation_tiles = max(2, min(int(self.validation_tiles), 8192))
        self.batch_size = max(1, min(int(self.batch_size), 8))
        for name in ("identity_learning_rate", "learning_rate", "boundary_learning_rate", "detail_learning_rate", "finetune_learning_rate"):
            setattr(self, name, float(min(max(getattr(self, name), 1.0e-6), 1.0e-2)))
        self.weight_decay = float(min(max(self.weight_decay, 0.0), 1.0e-2))
        self.optimizer_name = str(self.optimizer_name).strip().lower()
        if self.optimizer_name not in {"adamw", "adam"}:
            raise ValueError("optimizerName must be adamw or adam")
        self.optimizer_beta1 = float(min(max(self.optimizer_beta1, 0.0), 0.9999))
        self.optimizer_beta2 = float(min(max(self.optimizer_beta2, self.optimizer_beta1 + 1.0e-4), 0.99999))
        self.scheduler_name = str(self.scheduler_name).strip().lower()
        if self.scheduler_name not in {"phase", "cosine-phase"}:
            raise ValueError("schedulerName must be phase or cosine-phase")
        self.scheduler_min_lr_ratio = float(min(max(self.scheduler_min_lr_ratio, 0.01), 1.0))
        self.tile_size = max(32, min(int(self.tile_size), 256)); self.tile_size -= self.tile_size % 16
        self.target_scale = 4; self.input_channels = 17
        self.material_classes = max(2, min(int(self.material_classes), 16))
        if len(self.widths) != 4 or len(self.blocks_per_level) != 4 or len(self.decoder_blocks) != 3:
            raise ValueError("V9 widths/blocks must describe four encoder and three decoder levels")
        self.widths = tuple(max(32, min(int(v), 512)) for v in self.widths)
        self.blocks_per_level = tuple(max(1, min(int(v), 10)) for v in self.blocks_per_level)
        self.decoder_blocks = tuple(max(1, min(int(v), 8)) for v in self.decoder_blocks)
        self.attention_heads = max(1, min(int(self.attention_heads), 16))
        if self.widths[-1] % self.attention_heads != 0:
            raise ValueError("attentionHeads must divide the deepest V9 width")
        self.attention_window = max(4, min(int(self.attention_window), 16))
        self.drop_path = float(min(max(self.drop_path, 0.0), 0.30))
        self.orientation_normalization_epsilon = float(min(max(self.orientation_normalization_epsilon, 1e-4), 0.25))
        for name in ("albedo_medium_delta", "albedo_fine_delta", "normal_medium_delta", "normal_fine_delta", "material_delta", "auxiliary_delta"):
            setattr(self, name, float(min(max(getattr(self, name), 0.0), 0.5)))
        self.initial_gate_bias = float(min(max(self.initial_gate_bias, -12.0), -1.0))
        self.appearance_enabled = bool(self.appearance_enabled)
        self.appearance_edge_suppression = float(min(max(self.appearance_edge_suppression, 0.0), 1.0))
        self.detail_reconstruction_enabled = bool(self.detail_reconstruction_enabled)
        self.detail_feature_channels = max(24, min(int(self.detail_feature_channels), 96))
        self.detail_mid_channels = max(20, min(int(self.detail_mid_channels), 80))
        self.detail_hr_channels = max(16, min(int(self.detail_hr_channels), 64))
        self.detail_albedo_max_delta = float(min(max(self.detail_albedo_max_delta, 0.01), 0.50))
        self.detail_normal_max_delta = float(min(max(self.detail_normal_max_delta, 0.01), 0.40))
        self.detail_material_max_delta = float(min(max(self.detail_material_max_delta, 0.01), 0.50))
        self.detail_confidence_initial_bias = float(min(max(self.detail_confidence_initial_bias, -6.0), 2.0))
        self.detail_regret_initial_bias = float(min(max(self.detail_regret_initial_bias, -4.0), 4.0))
        for name in ("detail_contrast_weight", "detail_cross_map_weight", "detail_confidence_weight", "detail_regret_classifier_weight"):
            setattr(self, name, float(min(max(getattr(self, name), 0.0), 10.0)))
        self.detail_recovery_required = float(min(max(self.detail_recovery_required, 0.0), 0.95))
        self.detail_gradient_recovery_required = float(min(max(self.detail_gradient_recovery_required, 0.0), 0.95))
        self.detail_normal_recovery_required = float(min(max(self.detail_normal_recovery_required, -0.25), 0.95))
        self.detail_win_fraction_required = float(min(max(self.detail_win_fraction_required, 0.0), 1.0))
        self.detail_regression_fraction_max = float(min(max(self.detail_regression_fraction_max, 0.0), 1.0))
        self.raven_full_pipeline_preview_enabled = bool(self.raven_full_pipeline_preview_enabled)
        self.raven_representative_preview_enabled = bool(self.raven_representative_preview_enabled)
        self.raven_train_only_enabled = bool(self.raven_train_only_enabled)
        self.preview_allow_unqualified_downstream = bool(self.preview_allow_unqualified_downstream)
        self.raven_downstream_tiles_per_epoch = int(min(max(self.raven_downstream_tiles_per_epoch, 4), 512))
        self.seam_directional_enabled = bool(self.seam_directional_enabled)
        self.seam_directional_channels = max(8, min(int(self.seam_directional_channels), 64))
        self.seam_directional_angle_bins = max(4, min(int(self.seam_directional_angle_bins), 24))
        self.seam_directional_kernel_size = max(3, min(int(self.seam_directional_kernel_size) | 1, 11))
        self.seam_directional_kernel_residual_scale = float(min(max(self.seam_directional_kernel_residual_scale, 0.0), 0.30))
        self.seam_forced_recovery_required = float(min(max(self.seam_forced_recovery_required, 0.0), 1.0))
        self.seam_authority_iou_required = float(min(max(self.seam_authority_iou_required, 0.0), 1.0))
        self.seam_ridge_weight = float(min(max(self.seam_ridge_weight, 0.0), 2.0))
        self.seam_missing_detail_scale = float(min(max(self.seam_missing_detail_scale, 0.1), 64.0))
        self.seam_teacher_dilation_pixels = max(0, min(int(self.seam_teacher_dilation_pixels), 8))
        self.structural_stale_patience = max(1, min(int(self.structural_stale_patience), 20))
        self.seam_phase_sr_channels = max(16, min(int(self.seam_phase_sr_channels), 64))
        self.seam_phase_sr_max_delta = float(min(max(self.seam_phase_sr_max_delta, 0.0), 0.6))
        self.seam_phase_only_reconstruction = bool(self.seam_phase_only_reconstruction)
        self.seam_phase_residual_weight = float(min(max(self.seam_phase_residual_weight, 0.0), 256.0))
        self.seam_microproof_enabled = bool(self.seam_microproof_enabled)
        self.seam_microproof_steps = max(16, min(int(self.seam_microproof_steps), 256))
        self.seam_microproof_recovery_required = float(min(max(self.seam_microproof_recovery_required, 0.0), 0.99))
        self.seam_structure_tensor_radius = max(1, min(int(self.seam_structure_tensor_radius), 5))
        self.seam_structure_strength_gain = float(min(max(self.seam_structure_strength_gain, 0.5), 12.0))
        self.seam_coherence_floor = float(min(max(self.seam_coherence_floor, 0.0), 0.95))
        self.seam_tangent_sample_pixels = float(min(max(self.seam_tangent_sample_pixels, 0.25), 3.0))
        self.seam_normal_sample_pixels = float(min(max(self.seam_normal_sample_pixels, 0.25), 2.0))
        self.seam_max_normal_sharpen = float(min(max(self.seam_max_normal_sharpen, 0.0), 3.0))
        self.seam_max_authority = float(min(max(self.seam_max_authority, 0.0), 1.0))
        self.seam_geometry_band_pixels = float(min(max(self.seam_geometry_band_pixels, 0.5), 12.0))
        for name in ("seam_directional_weight", "seam_tangent_smoothness_weight", "seam_normal_profile_weight", "seam_authority_regularization_weight", "seam_projected_view_weight", "seam_authority_teacher_weight", "seam_reconstruction_weight"):
            setattr(self, name, float(min(max(getattr(self, name), 0.0), 128.0)))
        self.seam_lr_multiplier = float(min(max(self.seam_lr_multiplier, 0.25), 8.0))
        self.dds_codec_degradation_enabled = bool(self.dds_codec_degradation_enabled)
        self.dds_codec_probability = float(min(max(self.dds_codec_probability, 0.0), 1.0))
        self.dds_codec_blend_min = float(min(max(self.dds_codec_blend_min, 0.0), 1.0))
        self.dds_codec_blend_max = float(min(max(self.dds_codec_blend_max, self.dds_codec_blend_min), 1.0))
        self.geometry_displacement_source_pixels = float(min(max(self.geometry_displacement_source_pixels, 0.0), 2.0))
        self.geometry_edge_support_radius = max(0, min(int(self.geometry_edge_support_radius), 6))
        self.geometry_displacement_pixels = float(min(max(self.geometry_displacement_pixels, 0.0), 8.0))
        self.contour_sdf_max_distance_pixels = float(min(max(self.contour_sdf_max_distance_pixels, 4.0), 64.0))
        self.boundary_renderer_band_pixels = float(min(max(self.boundary_renderer_band_pixels, 1.0), 12.0))
        self.boundary_renderer_sample_pixels = float(min(max(self.boundary_renderer_sample_pixels, 0.75), 8.0))
        self.boundary_renderer_hard_width_pixels = float(min(max(self.boundary_renderer_hard_width_pixels, 0.50), 2.5))
        self.boundary_renderer_soft_width_pixels = float(min(max(self.boundary_renderer_soft_width_pixels, self.boundary_renderer_hard_width_pixels), 6.0))
        self.boundary_renderer_edge_threshold = float(min(max(self.boundary_renderer_edge_threshold, 0.01), 0.80))
        self.boundary_renderer_confidence_floor = float(min(max(self.boundary_renderer_confidence_floor, 0.0), 0.80))
        self.boundary_renderer_gate_gain = float(min(max(self.boundary_renderer_gate_gain, 0.25), 4.0))
        self.boundary_renderer_far_sample_multiplier = float(min(max(self.boundary_renderer_far_sample_multiplier, 1.0), 3.0))
        self.boundary_renderer_far_sample_weight = float(min(max(self.boundary_renderer_far_sample_weight, 0.0), 0.75))
        self.boundary_gate_initial_bias = float(min(max(self.boundary_gate_initial_bias, -8.0), 2.0))
        self.implicit_sdf_hidden_channels = max(64, min(int(self.implicit_sdf_hidden_channels), 128))
        self.implicit_sdf_coordinate_scale = float(min(max(self.implicit_sdf_coordinate_scale, 0.1), 4.0))
        self.implicit_sdf_residual_pixels = float(min(max(self.implicit_sdf_residual_pixels, 0.25), 6.0))
        self.sdf_sign_gauge_invariant = bool(self.sdf_sign_gauge_invariant)
        self.sdf_metric_band_pixels = float(min(max(self.sdf_metric_band_pixels, 2.0), self.contour_sdf_max_distance_pixels))
        self.sdf_coarse_init_std = float(min(max(self.sdf_coarse_init_std, 0.0), 0.01))
        self.sdf_synthetic_validation_tiles = max(4, min(int(self.sdf_synthetic_validation_tiles), 64))
        self.sdf_zero_band_pixels = float(min(max(self.sdf_zero_band_pixels, 0.10), 1.50))
        self.sdf_bootstrap_residual_pixels = float(min(max(self.sdf_bootstrap_residual_pixels, 0.0), self.implicit_sdf_residual_pixels))
        self.sdf_proof_residual_pixels = float(min(max(self.sdf_proof_residual_pixels, self.sdf_bootstrap_residual_pixels), self.implicit_sdf_residual_pixels))
        self.sdf_topology_margin_pixels = float(min(max(self.sdf_topology_margin_pixels, 0.05), 1.5))
        self.sdf_topology_core_pixels = float(min(max(self.sdf_topology_core_pixels, self.sdf_topology_margin_pixels + 0.05), 4.0))
        self.sdf_topology_band_pixels = float(min(max(self.sdf_topology_band_pixels, self.sdf_topology_core_pixels + 0.25), self.sdf_metric_band_pixels))
        self.sdf_topology_worst_fraction = float(min(max(self.sdf_topology_worst_fraction, 1.0e-4), 0.10))
        self.contour_transport_max_pixels = float(min(max(self.contour_transport_max_pixels, 0.5), self.contour_sdf_max_distance_pixels))
        self.contour_dilation_max_pixels = float(min(max(self.contour_dilation_max_pixels, 0.25), 6.0))
        self.contour_transport_min_jacobian = float(min(max(self.contour_transport_min_jacobian, 0.05), 0.95))
        self.contour_normal_offset_max_pixels = float(min(max(self.contour_normal_offset_max_pixels, 0.5), self.contour_sdf_max_distance_pixels))
        self.contour_phase_refine_max_pixels = float(min(max(self.contour_phase_refine_max_pixels, 0.0), 1.25))
        self.sdf_teacher_recovery_required = float(min(max(self.sdf_teacher_recovery_required, 0.0), 1.0))
        self.implicit_boundary_feature_channels = max(16, min(int(self.implicit_boundary_feature_channels), 128))
        self.implicit_boundary_residual_max_pixels = float(min(max(self.implicit_boundary_residual_max_pixels, 0.0), 2.0))
        self.implicit_boundary_supersample_grid = max(1, min(int(self.implicit_boundary_supersample_grid), 5))
        self.topology_field_feature_channels = max(24, min(int(self.topology_field_feature_channels), 128))
        self.topology_field_hidden_channels = max(32, min(int(self.topology_field_hidden_channels), 192))
        self.topology_field_control_scale = int(self.topology_field_control_scale)
        if self.topology_field_control_scale not in (1, 2, 4):
            raise ValueError("topology_field_control_scale must be 1, 2 or 4")
        if 4 % self.topology_field_control_scale != 0:
            raise ValueError("topology_field_control_scale must divide the 4x target scale")
        self.topology_field_max_log_magnitude_delta = float(min(max(self.topology_field_max_log_magnitude_delta, 0.5), 12.0))
        self.topology_field_magnitude_floor_pixels = float(min(max(self.topology_field_magnitude_floor_pixels, 0.005), 0.25))
        self.topology_field_edit_band_pixels = float(min(max(self.topology_field_edit_band_pixels, 4.0), 24.0))
        self.spline_graph_feature_channels = max(24, min(int(self.spline_graph_feature_channels), 128))
        self.spline_graph_hidden_channels = max(32, min(int(self.spline_graph_hidden_channels), 192))
        self.spline_graph_control_scale = int(min(max(int(self.spline_graph_control_scale), 1), 2))
        self.spline_graph_max_topology_delta_pixels = float(min(max(self.spline_graph_max_topology_delta_pixels, 1.0), 16.0))
        self.spline_graph_topology_edit_band_pixels = float(min(max(self.spline_graph_topology_edit_band_pixels, 1.0), 8.0))
        self.spline_graph_max_displacement_pixels = float(min(max(self.spline_graph_max_displacement_pixels, 0.5), 16.0))
        self.spline_graph_max_tangent_residual = float(min(max(self.spline_graph_max_tangent_residual, 0.05), 2.0))
        self.spline_graph_edit_band_pixels = float(min(max(self.spline_graph_edit_band_pixels, 4.0), 24.0))
        self.spline_graph_neighbour_radius = int(min(max(self.spline_graph_neighbour_radius, 1), 3))
        self.spline_graph_samples_per_span = int(min(max(self.spline_graph_samples_per_span, 2), 8))
        self.spline_ordered_branch_enabled = bool(self.spline_ordered_branch_enabled)
        self.spline_branch_smoothing_passes = int(min(max(self.spline_branch_smoothing_passes, 1), 8))
        self.spline_branch_smoothing_strength = float(min(max(self.spline_branch_smoothing_strength, 0.0), 1.0))
        self.spline_branch_smoothing_window = int(min(max(self.spline_branch_smoothing_window, 5), 127))
        if self.spline_branch_smoothing_window % 2 == 0:
            self.spline_branch_smoothing_window += 1
        self.spline_branch_corner_cosine = float(min(max(self.spline_branch_corner_cosine, -1.0), 1.0))
        self.spline_branch_corner_window = int(min(max(self.spline_branch_corner_window, 2), 8))
        self.spline_graph_lr_multiplier = float(min(max(self.spline_graph_lr_multiplier, 1.0), 10.0))
        self.stroke_centerline_hidden_channels = int(min(max(self.stroke_centerline_hidden_channels, 32), 192))
        self.stroke_centerline_max_delta_lr = float(min(max(self.stroke_centerline_max_delta_lr, 0.10), 2.0))
        self.stroke_centerline_max_tangent_residual = float(min(max(self.stroke_centerline_max_tangent_residual, 0.0), 2.0))
        self.stroke_centerline_max_width_delta_pixels = float(min(max(self.stroke_centerline_max_width_delta_pixels, 0.25), 8.0))
        self.stroke_centerline_ridge_min_depth_pixels = float(min(max(self.stroke_centerline_ridge_min_depth_pixels, 0.10), 4.0))
        self.stroke_centerline_neighbourhood_radius_lr = int(min(max(self.stroke_centerline_neighbourhood_radius_lr, 2), 5))
        self.stroke_centerline_segment_half_length_lr = float(min(max(self.stroke_centerline_segment_half_length_lr, 0.55), 2.0))
        self.stroke_centerline_initial_width_scale = float(min(max(self.stroke_centerline_initial_width_scale, 0.15), 0.75))
        for name in (
            "stroke_centerline_center_weight", "stroke_centerline_width_weight",
            "stroke_centerline_tangent_weight", "stroke_centerline_render_weight",
        ):
            setattr(self, name, float(min(max(getattr(self, name), 0.0), 128.0)))
        self.parametric_primitive_hidden_channels = int(min(max(self.parametric_primitive_hidden_channels, 48), 192))
        for name in ("parametric_primitive_class_weight", "parametric_primitive_param_weight", "parametric_primitive_render_weight"):
            setattr(self, name, float(min(max(getattr(self, name), 0.0), 256.0)))
        self.parametric_primitive_class_accuracy_required = float(min(max(self.parametric_primitive_class_accuracy_required, 0.50), 1.0))
        self.parametric_primitive_param_mae_required = float(min(max(self.parametric_primitive_param_mae_required, 0.001), 0.25))
        self.parametric_primitive_train_tiles_per_epoch = int(min(max(self.parametric_primitive_train_tiles_per_epoch, self.parametric_primitive_batch_size), 8192))
        self.parametric_primitive_batch_size = int(min(max(self.parametric_primitive_batch_size, 1), 32))
        self.parametric_primitive_lr_multiplier = float(min(max(self.parametric_primitive_lr_multiplier, 1.0), 64.0))
        self.parametric_primitive_classifier_epochs = int(min(max(self.parametric_primitive_classifier_epochs, 1), 64))
        self.parametric_primitive_parameter_epochs = int(min(max(self.parametric_primitive_parameter_epochs, 1), 128))
        self.parametric_primitive_integration_epochs = int(min(max(self.parametric_primitive_integration_epochs, 1), 64))
        self.parametric_primitive_fit_steps = int(min(max(self.parametric_primitive_fit_steps, 8), 256))
        self.parametric_primitive_fit_learning_rate = float(min(max(self.parametric_primitive_fit_learning_rate, 1.0e-4), 0.25))
        self.spline_geometry_fixed_bank_tiles = int(min(max(self.spline_geometry_fixed_bank_tiles, 32), 4096))
        self.spline_metric_calibration_enabled = bool(self.spline_metric_calibration_enabled)
        self.spline_metric_calibration_scale_delta = float(min(max(self.spline_metric_calibration_scale_delta, 0.0), 0.50))
        self.spline_metric_calibration_bias_pixels = float(min(max(self.spline_metric_calibration_bias_pixels, 0.0), 0.75))
        self.spline_metric_calibration_band_pixels = float(min(max(self.spline_metric_calibration_band_pixels, 1.0), 6.0))
        for name in (
            "spline_metric_offset_weight", "spline_metric_eikonal_near_weight",
            "spline_metric_scale_regularization_weight", "spline_metric_bias_regularization_weight",
        ):
            setattr(self, name, float(min(max(getattr(self, name), 0.0), 256.0)))
        self.sdf_oracle_render_band_mae_required = float(min(max(self.sdf_oracle_render_band_mae_required, 0.001), 0.25))
        self.sdf_oracle_gradient_mae_required = float(min(max(self.sdf_oracle_gradient_mae_required, 0.005), 0.50))
        self.sdf_oracle_profile_width_error_required = float(min(max(self.sdf_oracle_profile_width_error_required, 0.01), 0.50))
        self.sdf_oracle_profile_correlation_required = float(min(max(self.sdf_oracle_profile_correlation_required, 0.50), 0.9999))
        self.sdf_oracle_global_mae_required = float(min(max(self.sdf_oracle_global_mae_required, 0.0), 0.10))
        self.sdf_oracle_global_mae_case_max_required = float(min(max(self.sdf_oracle_global_mae_case_max_required, 0.0), 0.25))
        self.sdf_oracle_render_band_mae_preview_required = float(min(max(self.sdf_oracle_render_band_mae_preview_required, 0.0), 0.25))
        self.sdf_oracle_core_halo_delta_required_8bit = float(min(max(self.sdf_oracle_core_halo_delta_required_8bit, 0.0), 16.0))
        self.edge_crossing_feature_channels = max(24, min(int(self.edge_crossing_feature_channels), 128))
        self.edge_crossing_hidden_channels = max(32, min(int(self.edge_crossing_hidden_channels), 192))
        self.edge_crossing_max_logit_delta = float(min(max(self.edge_crossing_max_logit_delta, 1.0), 10.0))
        self.edge_crossing_edit_band_pixels = float(min(max(self.edge_crossing_edit_band_pixels, 4.0), 24.0))
        self.edge_crossing_neighbour_radius = int(min(max(self.edge_crossing_neighbour_radius, 0), 2))
        self.edge_crossing_lr_multiplier = float(min(max(self.edge_crossing_lr_multiplier, 1.0), 10.0))
        self.oracle_patch_feature_channels = max(24, min(int(self.oracle_patch_feature_channels), 128))
        self.oracle_patch_hidden_channels = max(32, min(int(self.oracle_patch_hidden_channels), 192))
        self.oracle_patch_footprint_lr = int(self.oracle_patch_footprint_lr)
        if self.oracle_patch_footprint_lr not in (3, 5):
            raise ValueError("oracle_patch_footprint_lr must be 3 or 5")
        self.oracle_patch_max_delta_pixels = float(min(max(self.oracle_patch_max_delta_pixels, 1.0), self.contour_sdf_max_distance_pixels))
        self.oracle_patch_max_coverage_logit_delta = float(min(max(self.oracle_patch_max_coverage_logit_delta, 2.0), 20.0))
        self.oracle_patch_edit_band_pixels = float(min(max(self.oracle_patch_edit_band_pixels, 2.0), self.contour_sdf_max_distance_pixels))
        self.parametric_boundary_control_scale = max(1, min(int(self.parametric_boundary_control_scale), 4))
        self.parametric_boundary_max_offset_pixels = float(min(max(self.parametric_boundary_max_offset_pixels, 0.5), 12.0))
        self.parametric_boundary_max_normal_correction = float(min(max(self.parametric_boundary_max_normal_correction, 0.1), 3.0))
        self.parametric_boundary_max_curvature_per_pixel = float(min(max(self.parametric_boundary_max_curvature_per_pixel, 0.02), 1.0))
        self.parametric_boundary_max_ribbon_half_width_pixels = float(min(max(self.parametric_boundary_max_ribbon_half_width_pixels, 0.5), 12.0))
        self.boundary_specialist_channels = max(16, min(int(self.boundary_specialist_channels), 96))
        self.boundary_specialist_band_pixels = float(min(max(self.boundary_specialist_band_pixels, 1.0), 8.0))
        self.boundary_specialist_max_coverage_delta = float(min(max(self.boundary_specialist_max_coverage_delta, 0.05), 0.5))
        self.boundary_specialist_logit_delta_max = float(min(max(self.boundary_specialist_logit_delta_max, 2.0), 24.0))
        self.boundary_specialist_recovery_required = float(min(max(self.boundary_specialist_recovery_required, 0.0), 1.0))
        self.benefit_selector_channels = max(12, min(int(self.benefit_selector_channels), 64))
        self.structural_line_jitter_required_pixels = float(min(max(self.structural_line_jitter_required_pixels, 0.05), 2.0))
        self.structural_curve_roughness_required_pixels = float(min(max(self.structural_curve_roughness_required_pixels, 0.05), 2.0))
        self.structural_line_staircase_recovery_required = float(min(max(self.structural_line_staircase_recovery_required, 0.0), 1.0))
        self.sdf_delta_max_pixels = float(min(max(self.sdf_delta_max_pixels, 1.0), self.contour_sdf_max_distance_pixels))
        self.sdf_improvement_margin_pixels = float(min(max(self.sdf_improvement_margin_pixels, 0.0), 2.0))
        self.geometry_need_floor = float(min(max(self.geometry_need_floor, 0.0), 1.0))
        self.geometry_need_sdf_scale_pixels = float(min(max(self.geometry_need_sdf_scale_pixels, 0.25), 24.0))
        self.sdf_relative_gain_required = float(min(max(self.sdf_relative_gain_required, 0.0), 0.95))
        self.sdf_relative_win_fraction = float(min(max(self.sdf_relative_win_fraction, 0.0), 1.0))
        self.sdf_relative_regression_fraction = float(min(max(self.sdf_relative_regression_fraction, 0.0), 1.0))
        self.sdf_catastrophic_chamfer_pixels = float(min(max(self.sdf_catastrophic_chamfer_pixels, 4.0), 256.0))
        self.sdf_missing_contour_tolerance = float(min(max(self.sdf_missing_contour_tolerance, 0.0), 1.0))
        self.boundary_renderer_plateau_samples = max(3, min(int(self.boundary_renderer_plateau_samples), 9))
        self.boundary_renderer_plateau_max_multiplier = float(min(max(self.boundary_renderer_plateau_max_multiplier, 1.25), 4.0))
        self.boundary_renderer_plateau_stability_scale = float(min(max(self.boundary_renderer_plateau_stability_scale, 1.0), 80.0))
        self.geometry_regret_margin = float(min(max(self.geometry_regret_margin, 0.0), 0.25))
        self.edge_regret_multiplier = float(min(max(self.edge_regret_multiplier, 0.0), 12.0))
        for name in (
            "albedo_weight", "albedo_gradient_weight", "albedo_pyramid_weight", "normal_weight",
            "normal_gradient_weight", "roughness_weight", "emissive_weight", "material_weight",
            "sdf_weight", "edge_weight", "orientation_weight", "cross_map_weight", "seam_weight",
            "regret_weight", "normal_regret_weight", "gate_target_weight", "residual_l1_weight",
            "fine_zero_mean_weight", "detail_laplacian_weight", "ringing_regret_weight", "unchanged_region_weight",
            "geometric_alignment_weight", "tangent_coherence_weight", "curvature_coherence_weight",
            "sdf_curvature_weight", "synthetic_geometry_loss_boost",
            "displacement_smoothness_weight", "displacement_off_contour_weight",
            "displacement_tangent_weight", "displacement_sparsity_weight",
            "geometry_regret_weight", "geometry_photometric_weight",
            "boundary_photometric_weight", "boundary_profile_weight",
            "boundary_fuzz_weight", "boundary_halo_weight",
            "boundary_hardness_weight", "boundary_gate_weight",
            "boundary_off_contour_weight", "boundary_regret_weight",
            "boundary_sdf_zero_weight", "boundary_edge_sdf_consistency_weight",
            "boundary_pixel_regret_weight",
            "sdf_surface_weight", "sdf_sign_weight", "sdf_topology_weight", "sdf_eikonal_weight",
            "sdf_gradient_alignment_weight", "sdf_metric_gradient_weight", "sdf_proof_renderer_weight",
            "coarse_sdf_surface_weight", "sdf_residual_l1_weight",
            "contour_transport_weight", "contour_dilation_weight",
            "contour_transport_smoothness_weight", "contour_dilation_smoothness_weight",
            "contour_transport_fold_weight",
            "contour_projected_offset_weight", "contour_phase_refine_weight",
            "contour_soft_coverage_weight", "contour_normal_offset_weight",
            "contour_normal_offset_tangent_weight", "contour_normal_offset_normal_weight",
            "sdf_teacher_render_weight",
            "sdf_teacher_gradient_weight", "sdf_teacher_profile_weight",
            "topology_field_sdf_weight", "topology_field_control_weight", "topology_field_crossing_weight",
            "topology_field_gradient_weight", "topology_field_eikonal_weight",
            "topology_field_curvature_weight", "topology_field_render_weight",
            "topology_field_render_gradient_weight", "topology_field_render_profile_weight",
            "spline_graph_topology_control_weight", "spline_graph_topology_sign_weight",
            "spline_graph_point_weight", "spline_graph_tangent_weight",
            "spline_graph_sdf_weight", "spline_graph_gradient_weight",
            "spline_graph_eikonal_weight", "spline_graph_curvature_weight",
            "spline_graph_span_smoothness_weight", "spline_graph_span_tangent_weight",
            "spline_graph_span_separation_weight",
            "spline_graph_render_weight", "spline_graph_render_gradient_weight",
            "spline_graph_render_profile_weight",
            "edge_crossing_fraction_weight", "edge_crossing_sdf_weight",
            "edge_crossing_gradient_weight", "edge_crossing_eikonal_weight",
            "edge_crossing_curvature_weight", "edge_crossing_render_weight",
            "edge_crossing_render_gradient_weight", "edge_crossing_render_profile_weight",
            "oracle_patch_sdf_weight", "oracle_patch_sign_weight",
            "oracle_patch_coverage_weight", "oracle_patch_coverage_bce_weight",
            "oracle_patch_consistency_weight", "oracle_patch_gradient_weight",
            "oracle_patch_aggregate_coverage_weight", "oracle_patch_render_weight",
            "oracle_patch_render_gradient_weight", "oracle_patch_render_profile_weight",
            "parametric_boundary_anchor_weight", "parametric_boundary_normal_weight", "parametric_boundary_curvature_weight",
            "parametric_boundary_offset_smoothness_weight",
            "boundary_specialist_coverage_weight", "boundary_specialist_coverage_bce_weight",
            "boundary_specialist_coverage_gradient_weight", "boundary_specialist_profile_moment_weight",
            "boundary_specialist_gradient_weight", "boundary_specialist_profile_weight", "benefit_selector_weight",
            "sdf_delta_surface_weight", "sdf_delta_tangent_weight",
            "sdf_delta_laplacian_weight", "sdf_improvement_regret_weight",
            "direct_flow_weight", "bootstrap_direct_flow_weight",
        ):
            setattr(self, name, float(min(max(getattr(self, name), 0.0), 60.0)))
        self.boundary_gate_need_scale = float(min(max(self.boundary_gate_need_scale, 1.0e-3), 1.0))
        self.boundary_gate_exact_floor = float(min(max(self.boundary_gate_exact_floor, 0.0), 1.0))
        self.tangent_variation_margin = float(min(max(self.tangent_variation_margin, 0.0), 0.5))
        self.curvature_variation_margin = float(min(max(self.curvature_variation_margin, 0.0), 0.5))
        self.flow_activity_threshold_source_pixels = float(min(max(self.flow_activity_threshold_source_pixels, 0.001), 0.50))
        self.synthetic_geometry_probability = float(min(max(self.synthetic_geometry_probability, 0.0), 1.0))
        self.local_regret_patch = max(2, min(int(self.local_regret_patch), 64))
        self.gate_error_scale = float(min(max(self.gate_error_scale, 1e-3), 1.0))
        self.gate_edge_bonus = float(min(max(self.gate_edge_bonus, 0.0), 1.0))
        self.unchanged_error_threshold = float(min(max(self.unchanged_error_threshold, 0.0), 0.5))
        self.maximum_validation_regression_fraction = float(min(max(self.maximum_validation_regression_fraction, 0.0), 1.0))
        self.lod_bias_min = float(min(max(self.lod_bias_min, 0.05), 4.0))
        self.lod_bias_max = float(min(max(self.lod_bias_max, self.lod_bias_min), 6.0))
        for name in ("anisotropic_blur_probability", "bc_block_probability", "chroma_loss_probability", "ringing_probability", "halo_probability", "renderer_sampling_probability"):
            setattr(self, name, float(min(max(getattr(self, name), 0.0), 1.0)))
        self.renderer_anisotropy_max = float(min(max(self.renderer_anisotropy_max, 1.0), 8.0))
        self.renderer_subpixel_jitter = float(min(max(self.renderer_subpixel_jitter, 0.0), 1.0))
        self.boundary_candidate_count = max(1, min(int(self.boundary_candidate_count), 32))
        self.boundary_sampling_probability = float(min(max(self.boundary_sampling_probability, 0.0), 1.0))
        self.inference_tile_size = max(32, min(int(self.inference_tile_size), 256)); self.inference_tile_size -= self.inference_tile_size % 16
        self.inference_overlap = max(8, min(int(self.inference_overlap), self.inference_tile_size // 3))
        self.gradient_clip_norm = float(min(max(self.gradient_clip_norm, 0.1), 100.0))
        self.amp_initial_scale = float(max(1.0, self.amp_initial_scale))
        self.amp_minimum_scale = float(min(max(1.0, self.amp_minimum_scale), self.amp_initial_scale))
        self.amp_overflow_retries = max(0, min(int(self.amp_overflow_retries), 32))
        self.parameter_finite_check_interval = max(1, int(self.parameter_finite_check_interval))
        self.performance_profile = str(self.performance_profile).strip().lower()
        if self.performance_profile not in {"fast", "optimized", "tuned", "compiled", "channels-last", "balanced", "compatibility"}:
            raise ValueError("invalid V9 performance profile")
        self.data_loader_workers = max(0, min(int(self.data_loader_workers), 32))
        self.data_loader_prefetch_factor = max(1, min(int(self.data_loader_prefetch_factor), 8))
        self.data_loader_persistent_workers = bool(self.data_loader_persistent_workers)
        self.cuda_prefetch = bool(self.cuda_prefetch)
        self.cuda_memory_fraction = float(min(max(self.cuda_memory_fraction, 0.50), 0.95))
        self.reactive_vram_enabled = bool(self.reactive_vram_enabled)
        self.reactive_vram_target_free_fraction = float(
            min(max(self.reactive_vram_target_free_fraction, 0.10), 0.70)
        )
        self.reactive_vram_pause_free_fraction = float(
            min(max(self.reactive_vram_pause_free_fraction, 0.03), 0.40)
        )
        self.reactive_vram_resume_free_fraction = float(
            min(max(self.reactive_vram_resume_free_fraction, self.reactive_vram_pause_free_fraction + 0.02), 0.60)
        )
        self.reactive_vram_expand_hysteresis_fraction = float(
            min(max(self.reactive_vram_expand_hysteresis_fraction, 0.01), 0.25)
        )
        self.reactive_vram_expand_stable_steps = max(
            1, min(int(self.reactive_vram_expand_stable_steps), 256)
        )
        self.reactive_vram_poll_seconds = float(
            min(max(self.reactive_vram_poll_seconds, 0.10), 10.0)
        )
        self.reactive_vram_oom_retries = max(
            0, min(int(self.reactive_vram_oom_retries), 16)
        )
        self.reactive_vram_release_cache = bool(self.reactive_vram_release_cache)
        self.reactive_vram_burst_reserve_fraction = float(
            min(max(self.reactive_vram_burst_reserve_fraction, 0.15), 0.60)
        )
        self.reactive_vram_stability_samples = max(
            2, min(int(self.reactive_vram_stability_samples), 12)
        )
        self.reactive_vram_stability_interval_seconds = float(
            min(max(self.reactive_vram_stability_interval_seconds, 0.05), 2.0)
        )
        self.reactive_vram_dynamic_allocator_ceiling = bool(
            self.reactive_vram_dynamic_allocator_ceiling
        )
        self.reactive_vram_start_in_offload = bool(
            self.reactive_vram_start_in_offload
        )
        self.reactive_host_pause_free_fraction = float(
            min(max(self.reactive_host_pause_free_fraction, 0.03), 0.50)
        )
        self.reactive_host_resume_free_fraction = float(
            min(max(self.reactive_host_resume_free_fraction, self.reactive_host_pause_free_fraction + 0.02), 0.70)
        )
        self.channels_last = bool(self.channels_last)
        self.amp_dtype = str(self.amp_dtype).strip().lower()
        if self.amp_dtype not in {"auto", "fp16", "bf16"}: raise ValueError("ampDtype must be auto, fp16 or bf16")
        self.fused_optimizer = bool(self.fused_optimizer); self.cudnn_benchmark = bool(self.cudnn_benchmark); self.allow_tf32 = bool(self.allow_tf32)
        self.loss_precision = str(self.loss_precision).strip().lower()
        if self.loss_precision not in {"mixed", "fp32"}: raise ValueError("lossPrecision must be mixed or fp32")
        self.torch_compile_mode = str(self.torch_compile_mode).strip().lower()
        if self.torch_compile_mode not in {"off", "default", "reduce-overhead", "max-autotune"}: raise ValueError("invalid torchCompileMode")

    def apply_performance_profile(self, profile: str) -> None:
        profile = str(profile).strip().lower()
        if profile in {"fast", "optimized", "tuned"}:
            self.performance_profile = profile
            self.data_loader_workers = max(self.data_loader_workers, 4)
            self.data_loader_prefetch_factor = max(self.data_loader_prefetch_factor, 2)
            self.data_loader_persistent_workers = True
            self.cuda_prefetch = True
            self.channels_last = False
            self.amp_dtype = "auto"
            self.fused_optimizer = True
            self.cudnn_benchmark = True
            self.allow_tf32 = True
            self.loss_precision = "mixed"
            self.torch_compile_mode = "off"
        elif profile == "compiled":
            self.apply_performance_profile("optimized"); self.performance_profile = profile; self.torch_compile_mode = "reduce-overhead"
        elif profile == "channels-last":
            self.apply_performance_profile("optimized"); self.performance_profile = profile; self.channels_last = True
        elif profile == "balanced":
            self.apply_performance_profile("optimized"); self.performance_profile = profile; self.data_loader_workers = 2
        elif profile == "compatibility":
            self.performance_profile = profile; self.data_loader_workers = 0; self.data_loader_prefetch_factor = 2
            self.data_loader_persistent_workers = False; self.cuda_prefetch = False; self.channels_last = False
            self.amp_dtype = "fp16"; self.fused_optimizer = False; self.cudnn_benchmark = False
            self.allow_tf32 = False; self.loss_precision = "fp32"; self.torch_compile_mode = "off"
        else:
            raise ValueError("invalid V9 performance profile")
        self.validate()
