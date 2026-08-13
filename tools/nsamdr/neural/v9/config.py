"""Configuration for NSAMDR V9.8.3 sign-gauge metric-SDF convergence."""
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

    # Five conservative phases. The high-frequency 512 branch is frozen until
    # detail reconstruction has been reached.
    identity_epochs: int = 2
    residual_epochs: int = 6
    boundary_epochs: int = 4
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

    input_channels: int = 16
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
    appearance_enabled: bool = False
    appearance_edge_suppression: float = 0.80

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
    implicit_sdf_hidden_channels: int = 48
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
    sdf_eikonal_weight: float = 8.00
    sdf_gradient_alignment_weight: float = 2.00
    sdf_metric_gradient_weight: float = 6.00
    # V9.8.3 treats SDF polarity as a gauge choice: +SDF and -SDF describe the
    # same physical contour when the two material sides are swapped together.
    sdf_sign_gauge_invariant: bool = True
    sdf_metric_band_pixels: float = 12.0
    sdf_coarse_init_std: float = 0.0005
    sdf_synthetic_validation_tiles: int = 12
    sdf_zero_band_pixels: float = 0.50
    sdf_bootstrap_residual_pixels: float = 0.0
    sdf_proof_residual_pixels: float = 1.0
    sdf_proof_renderer_weight: float = 2.50

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
    reactive_vram_dynamic_allocator_ceiling: bool = True
    reactive_vram_start_in_offload: bool = True

    # Host-memory safety for CPU-saved autograd activations. Offload is not
    # allowed to turn GPU pressure into Windows paging pressure.
    reactive_host_pause_free_fraction: float = 0.20
    reactive_host_resume_free_fraction: float = 0.25

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
            self.identity_epochs + self.residual_epochs + self.boundary_epochs
            + self.detail_epochs + self.physical_finetune_epochs
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
        for name in ("identity_epochs", "residual_epochs", "boundary_epochs", "detail_epochs"):
            setattr(self, name, max(1, min(int(getattr(self, name)), 100)))
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
        self.target_scale = 4; self.input_channels = 16
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
        self.implicit_sdf_hidden_channels = max(16, min(int(self.implicit_sdf_hidden_channels), 128))
        self.implicit_sdf_coordinate_scale = float(min(max(self.implicit_sdf_coordinate_scale, 0.1), 4.0))
        self.implicit_sdf_residual_pixels = float(min(max(self.implicit_sdf_residual_pixels, 0.25), 6.0))
        self.sdf_sign_gauge_invariant = bool(self.sdf_sign_gauge_invariant)
        self.sdf_metric_band_pixels = float(min(max(self.sdf_metric_band_pixels, 2.0), self.contour_sdf_max_distance_pixels))
        self.sdf_coarse_init_std = float(min(max(self.sdf_coarse_init_std, 0.0), 0.01))
        self.sdf_synthetic_validation_tiles = max(4, min(int(self.sdf_synthetic_validation_tiles), 64))
        self.sdf_zero_band_pixels = float(min(max(self.sdf_zero_band_pixels, 0.10), 1.50))
        self.sdf_bootstrap_residual_pixels = float(min(max(self.sdf_bootstrap_residual_pixels, 0.0), self.implicit_sdf_residual_pixels))
        self.sdf_proof_residual_pixels = float(min(max(self.sdf_proof_residual_pixels, self.sdf_bootstrap_residual_pixels), self.implicit_sdf_residual_pixels))
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
            "sdf_surface_weight", "sdf_sign_weight", "sdf_eikonal_weight",
            "sdf_gradient_alignment_weight", "sdf_metric_gradient_weight", "sdf_proof_renderer_weight",
            "coarse_sdf_surface_weight", "sdf_residual_l1_weight",
            "direct_flow_weight", "bootstrap_direct_flow_weight",
        ):
            setattr(self, name, float(min(max(getattr(self, name), 0.0), 20.0)))
        self.boundary_gate_need_scale = float(min(max(self.boundary_gate_need_scale, 1.0e-3), 1.0))
        self.boundary_gate_exact_floor = float(min(max(self.boundary_gate_exact_floor, 0.0), 1.0))
        self.tangent_variation_margin = float(min(max(self.tangent_variation_margin, 0.0), 0.5))
        self.curvature_variation_margin = float(min(max(self.curvature_variation_margin, 0.0), 0.5))
        self.flow_activity_threshold_source_pixels = float(min(max(self.flow_activity_threshold_source_pixels, 0.001), 0.50))
        self.synthetic_geometry_probability = float(min(max(self.synthetic_geometry_probability, 0.0), 0.95))
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
