from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import inspect
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools/nsamdr/neural"
if str(NEURAL) not in sys.path:
    sys.path.insert(0, str(NEURAL))


def text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_baseline_variant_is_exact_model_baseline_and_does_not_call_model_forward():
    from torch.nn import functional as F
    from v9.inference import infer_tiled
    from v9.model import FidelityResidualNetV9, UPSCALE_FACTOR

    class NeverForward(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                channels_last=False,
                amp_dtype="auto",
                appearance_enabled=True,
                detail_reconstruction_enabled=True,
            )

        def forward(self, _value):  # pragma: no cover - must never execute
            raise AssertionError("baseline variant called model.forward")

    rng = np.random.default_rng(117)
    value = rng.uniform(0.0, 1.0, (17, 12, 10)).astype(np.float32)
    # Exercise the normalization limiter; the old 1e-8 baseline epsilon differed
    # measurably from the model's canonical 1e-6 implementation here.
    value[3] = 0.8
    value[4] = 0.6
    value[5:8] = rng.uniform(0.0, 1.0, (3, 12, 10)).astype(np.float32)

    maps, diagnostics = infer_tiled(
        NeverForward(), value, "cpu", return_diagnostics=True,
        return_all_maps=True, output_variant="baseline",
    )

    source = torch.from_numpy(value).unsqueeze(0)
    expected_albedo = F.interpolate(
        source[:, 0:3].clamp(0.0, 1.0), scale_factor=UPSCALE_FACTOR,
        mode="bicubic", align_corners=False, antialias=True,
    ).clamp(0.0, 1.0)
    expected_normal = FidelityResidualNetV9._normalize_xy(F.interpolate(
        source[:, 3:5].clamp(-1.0, 1.0), scale_factor=UPSCALE_FACTOR,
        mode="bilinear", align_corners=False,
    ))
    expected_material = F.interpolate(
        source[:, 5:8].clamp(0.0, 1.0), scale_factor=UPSCALE_FACTOR, mode="nearest"
    )

    def nhwc(tensor):
        return tensor[0].permute(1, 2, 0).numpy()

    assert np.array_equal(maps["albedo"], nhwc(expected_albedo))
    assert np.array_equal(maps["normal_xy"], nhwc(expected_normal))
    assert np.array_equal(maps["material"], nhwc(expected_material))
    assert diagnostics["outputVariant"] == "baseline"
    assert diagnostics["candidateAuthority"] == "deterministic-4x-baseline"
    assert diagnostics["tileCount"] == 0
    assert "model.forward not called" in diagnostics["productionForward"]


def test_stage_variants_are_selected_before_final_selector():
    source = inspect.getsource(__import__("v9.inference", fromlist=["InferenceService"]).InferenceService.infer_tiled)
    assert 'variant == "structural"' in source
    assert 'output["boundary_pre_seam_albedo"]' in source
    assert 'variant == "seam"' in source
    assert 'output["boundary_reconstructed_albedo"]' in source
    assert 'variant == "detail"' in source
    assert 'output["detail_candidate_albedo"]' in source


def test_live_preview_is_authored_baseline_stage_not_raw_vs_final():
    live = text("tools/nsamdr/neural/live_preview_nsamdr_v9_training.py")
    types = text("trinityal/tests/nsamdr/NSAMDRPreviewTypes.h")
    processing = text("trinityal/tests/nsamdr/NSAMDRPreviewProcessing.cpp")
    render = text("trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp")
    panel = text("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
    assert 'return "structural"' in live
    assert 'output_variant="baseline"' in live
    assert 'output_variant=stage_variant' in live
    assert 'NSAMDR_DETERMINISTIC_4X_BASELINE' in live
    assert 'CandidateAssetGpu baseline;' in types
    assert 'pointer.baselineMaterials' in processing
    assert 'candidates.baseline = std::move(nextBaseline)' in processing
    assert 'A_AUTHORED_SOURCE_vs_B_DETERMINISTIC_4X_BASELINE_vs_C_CURRENT_TRAINED_STAGE' in processing
    assert 'finalCandidate=loaded provenance=verified' not in processing
    assert 'const bool threeWay = deterministicBaseline.available;' in render
    assert 'A AUTHORED SOURCE' in panel
    assert 'B 4X BASELINE' in panel
    assert 'C NSAMDR LIVE STAGE' in panel
    assert 'Current C state:' in panel
    assert 'Live epoch A/B/C comparison' in panel
    assert 'Swap A and C' in panel
    assert 'A/C texture resource isolation' in panel
    assert 'Live comparison is A AUTHORED SOURCE' in panel
    assert 'Current B state:' not in panel
    assert 'Live epoch A/B comparison' not in panel


def test_connected_spline_b1_optimizes_real_raven_and_keeps_synthetic_audit():
    from v9.training import TrainingService

    source = inspect.getsource(TrainingService.train_v9)
    assert 'authored_config.synthetic_geometry_probability = 0.0' in source
    assert 'structural_train_dataset = PhysicalTileDatasetV9(' in source
    assert 'epoch_loader = structural_train_loader' in source
    assert 'phase in {"sdf-bootstrap", "sdf-proof"}' in source
    assert 'SyntheticGeometryValidationDataset(' in source
    assert 'local_structure_train_dataset = ParametricPrimitiveTrainingDataset(' not in source
    assert 'local_structure_train_loader' not in source


def test_quick_first_b1b_is_authored_raven_smoke_and_cannot_promote():
    from v9.training import TrainingService

    source = inspect.getsource(TrainingService.train_v9)
    assert 'b1b_stage_epoch == 1' in source
    assert 'int(config.tiles_per_epoch) <= 64' in source
    assert 'structural_smoke_batch_limit = min(epoch_batch_count, 14)' in source
    assert 'B1b QUICK SMOKE' in source
    assert 'authored Raven' in source
    assert 'synthetic ladder remains validation-only' in source
    assert 'structural_smoke_epoch = structural_smoke_batch_limit is not None' in source
    assert 'not structural_smoke_epoch' in source
    assert 'B1/B2 promotion is disabled' in source
    assert 'not structural_smoke_epoch and integration_ready and hard_render_gate' in source

def test_v118_structural_candidate_is_exact_baseline_at_zero_gain():
    from v9.model import FidelityResidualNetV9, MODEL_SCHEMA

    locality = torch.ones((1, 1, 5, 7), dtype=torch.float32)
    zero_gain = torch.zeros_like(locality)
    weight = FidelityResidualNetV9._structural_residual_weight(
        locality, zero_gain, None
    )
    assert torch.equal(weight, torch.zeros_like(weight))

    baseline = torch.linspace(0.05, 0.95, 5 * 7 * 3, dtype=torch.float32).reshape(1, 3, 5, 7)
    proposal = torch.flip(baseline, dims=(-1,))
    candidate = baseline + weight * (proposal - baseline)
    assert torch.equal(candidate, baseline)
    assert MODEL_SCHEMA == "NSAMDR_RAVEN_PRODUCTION_B1A_IDENTITY_B1B_PRESEAM_RESIDUAL_SPLINE_GRAPH_4X_V11_10_0"


def test_v118_structural_residual_gain_is_zero_initialized_and_checkpointed():
    local = text("tools/nsamdr/neural/v9/local_boundary_production_contract.py")
    model = text("tools/nsamdr/neural/v9/model.py")
    assert "self.structural_residual_gain_head = nn.Sequential(" in local
    assert "nn.init.zeros_(self.structural_residual_gain_head[-1].weight)" in local
    assert "nn.init.zeros_(self.structural_residual_gain_head[-1].bias)" in local
    assert '"structural_residual_gain": structural_residual_gain' in local
    assert 'geometry.get("structural_residual_gain")' in model
    assert 'boundary_structural_residual_weight' in model
    assert 'structural_gate = candidate_locality' not in model


def test_v119_b1a_freezes_gain_and_b1b_unlocks_it():
    local = text("tools/nsamdr/neural/v9/local_boundary_production_contract.py")
    a = local.index("    def unlock_topology_for_bootstrap")
    b = local.index("    def lock_topology_for_proof")
    c = local.index("    def restore_locked_topology_parameters")
    b1a, b1b = local[a:b], local[b:c]
    assert "nn.init.zeros_(self.structural_residual_gain_head[-1].weight)" in b1a
    assert "self.structural_residual_gain_head.parameters()" in b1a
    assert "parameter.requires_grad_(False)" in b1a
    assert "self.structural_residual_gain_head.parameters()" in b1b
    assert "parameter.requires_grad_(True)" in b1b


def test_v119_quick_moves_strict_baseline_win_to_b1b():
    smoke = text("tools/nsamdr/neural/v9/application/baseline_relative_smoke.py")
    pipeline = text("tools/nsamdr/neural/v9/application/pipeline.py")
    assert "def safe_to_refine(" in smoke
    assert 'metrics["candidateMae"] <= metrics["baselineMae"] + tolerance' in smoke
    assert 'metrics["candidateMae"] < metrics["baselineMae"]' in smoke
    assert "def _run_quick_b1b_smoke(" in pipeline
    assert "PASS B1a: C preserved deterministic baseline B" in pipeline
    assert "PASS B1b: C now beats deterministic baseline B" in pipeline
    assert 'phase="sdf-proof-baseline-relative-smoke"' in pipeline


def test_v1110_structural_smoke_is_pre_seam_and_uses_explicit_metrics():
    losses = text("tools/nsamdr/neural/v9/losses.py")
    smoke = text("tools/nsamdr/neural/v9/application/baseline_relative_smoke.py")
    assert 'if phase in {"sdf-bootstrap", "sdf-proof"}:' in losses
    assert 'reconstructed_albedo = seam_source_albedo' in losses
    assert 'reconstructed_normal = seam_source_normal' in losses
    assert 'reconstructed_material = seam_source_material' in losses
    assert 'losses["structural_baseline_mae"]' in losses
    assert 'losses["structural_stage_mae"]' in losses
    assert 'losses["structural_relative_gain"]' in losses
    assert 'losses["structural_improvement_fraction"]' in losses
    assert 'losses["structural_regression_fraction"]' in losses
    assert 'value("structural_baseline_mae"' in smoke
    assert 'value("structural_stage_mae"' in smoke
    assert 'value("structural_relative_gain"' in smoke
    assert 'value("structural_improvement_fraction"' in smoke
    assert 'value("structural_regression_fraction"' in smoke
    assert 'value("sdf_stageb_renderer_mae"' not in smoke
