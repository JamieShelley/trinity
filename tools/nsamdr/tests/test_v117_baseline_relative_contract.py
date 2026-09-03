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


def test_baseline_variant_is_deterministic_and_does_not_call_model_forward():
    from v9.inference import infer_tiled

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

    value = np.zeros((17, 12, 10), dtype=np.float32)
    value[0:3] = 0.37
    value[3] = 0.25
    value[4] = -0.15
    value[5] = 0.2
    value[6] = 0.4
    value[7] = 0.7
    maps, diagnostics = infer_tiled(
        NeverForward(), value, "cpu", return_diagnostics=True,
        return_all_maps=True, output_variant="baseline",
    )
    assert maps["albedo"].shape == (48, 40, 3)
    assert np.allclose(maps["albedo"], 0.37, atol=1.0e-5)
    assert np.allclose(maps["material"][..., 0], 0.2, atol=1.0e-6)
    assert diagnostics["outputVariant"] == "baseline"
    assert diagnostics["tileCount"] == 0


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
    assert 'CandidateAssetGpu baseline;' in types
    assert 'pointer.baselineMaterials' in processing
    assert 'candidates.baseline = std::move(nextBaseline)' in processing
    assert 'const bool threeWay = deterministicBaseline.available;' in render
    assert 'A AUTHORED SOURCE' in panel
    assert 'B 4X BASELINE' in panel
    assert 'C NSAMDR LIVE STAGE' in panel


def test_quick_first_b1b_is_small_complete_class_smoke_only():
    from v9.training import TrainingService

    source = inspect.getsource(TrainingService.train_v9)
    assert 'b1b_stage_epoch == 1' in source
    assert 'int(config.tiles_per_epoch) <= 64' in source
    assert 'PRIMITIVE_COUNT * 2' in source
    assert 'B1b QUICK SMOKE' in source
    assert 'structural_smoke_batch_limit' in source
    # Later epochs/full runs still use the canonical complete loader.
    assert 'local_structure_train_loader' in source
