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


def test_quick_first_b1b_is_balanced_smoke_and_cannot_promote():
    from v9.dataset import ParametricPrimitiveTrainingDataset
    from v9.training import TrainingService

    source = inspect.getsource(TrainingService.train_v9)
    dataset_source = inspect.getsource(ParametricPrimitiveTrainingDataset.__getitem__)
    assert 'forced_class=int(index) % PRIMITIVE_COUNT' in dataset_source
    assert 'b1b_stage_epoch == 1' in source
    assert 'int(config.tiles_per_epoch) <= 64' in source
    assert 'PRIMITIVE_COUNT * 2' in source
    assert 'B1b QUICK SMOKE' in source
    assert 'structural_smoke_epoch = structural_smoke_batch_limit is not None' in source
    assert 'not structural_smoke_epoch' in source
    assert 'B1/B2 promotion is disabled' in source
    assert 'not structural_smoke_epoch and integration_ready and hard_render_gate' in source
    # The complete bank is unshuffled, so indices 0..13 are exactly two of each
    # class because the dataset maps index modulo primitive count to class.
    assert 'rolling_epoch_indices=False' in source
    assert 'local_structure_train_loader' in source
