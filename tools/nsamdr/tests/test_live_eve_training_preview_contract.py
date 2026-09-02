from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_live_eve_training_preview_is_end_to_end_and_non_authoritative():
    training = text("tools/nsamdr/neural/v9/training.py")
    workflow = text("tools/nsamdr/neural/run_nsamdr_v9_raven_tune_preview.py")
    live = text("tools/nsamdr/neural/live_preview_nsamdr_v9_training.py")
    gui = text("tools/nsamdr/gui/nsamdr_v9_workflow_gui.py")
    processing = text("trinityal/tests/nsamdr/NSAMDRPreviewProcessing.cpp")
    application = text("trinityal/tests/nsamdr/NSAMDRPreviewApplication.cpp")
    panel = text("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
    final_preview = text("tools/nsamdr/neural/preview_nsamdr_v9_experiment.py")

    assert "NSAMDR_LIVE_PREVIEW_CHECKPOINTS" in training
    assert "os.link(state_path, temporary_path)" in training
    assert '"authority": "training-intermediate"' in training
    assert "--live-preview-during-training" in workflow
    assert "live_preview_nsamdr_v9_training.py" in workflow
    assert 'training_env["NSAMDR_LIVE_PREVIEW_CHECKPOINTS"] = "1"' in workflow
    assert "Live EVE A/B preview while training" in gui
    assert '"1024", ("512", "1024", "2048")' in gui

    assert "eve.prepare_asset" in live
    assert "FidelityResidualNetV9" in live
    assert "StrategyCandidateGenerator" in live
    assert "NSAMDR_LIVE_CANDIDATE_POINTER_V1" in live
    assert '"qualified": False' in live
    assert '"authority": "training-intermediate"' in live
    assert "skipping stale publish" in live
    assert "eve.launch_preview" in live

    assert "RefreshLiveCandidate" in processing
    assert 'GetEnvironmentString("NSAMDR_LIVE_CANDIDATE_POINTER")' in processing
    assert "CandidateUsesSourceDrawRanges(resources, nextCandidate)" in processing
    assert "candidates.candidate = std::move(nextCandidate)" in processing
    assert "nextLiveCandidatePollSeconds" in application
    assert "elapsedSeconds + 0.5f" in application
    assert "LIVE TRAINING A/B" in application
    assert "LIVE TRAINING PREVIEW — UNQUALIFIED INTERMEDIATE" in panel
    assert "A RAW SOURCE stays fixed while B CURRENT NSAMDR EPOCH hot-reloads" in panel

    # Production final remains fail-closed and was not repurposed for live epochs.
    assert "preview requires a completed qualified experiment" in final_preview
    assert "require_qualified=True" in final_preview
    assert '"NSAMDR_PREVIEW_AUTHORITY": "production-final"' in final_preview
