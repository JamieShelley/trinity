"""Durable absence checks for removed NSAMDR product routes."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

CANONICAL_BATCH_FILES = {
    "nsamdr.bat",
    "run_nsamdr_obj_preview_dx11.bat",
    "run_nsamdr_v9_gui.bat",
    "setup_nsamdr_cpu.bat",
    "setup_nsamdr_cuda.bat",
}

REMOVED_PATHS = (
    "collect_nsamdr_failfast.py",
    "scripts/build/run_nsamdr_existing_preview.bat",
    "scripts/build/run_nsamdr_raven_architecture_locked.bat",
    "scripts/build/run_nsamdr_raven_capability_first.bat",
    "scripts/build/run_nsamdr_raven_capability_full_renderer.bat",
    "scripts/build/run_nsamdr_raven_capability_generalization.bat",
    "scripts/build/run_nsamdr_raven_capability_viewer.bat",
    "tools/nsamdr/gui/raven_capability_viewer.py",
    "tools/nsamdr/neural/V10_10_1_CAPABILITY_ROUTING_ACTIVE.txt",
    "tools/nsamdr/neural/audit_nsamdr_v9_geometry_checkpoint.py",
    "tools/nsamdr/neural/audit_nsamdr_v10_detail_checkpoint.py",
    "tools/nsamdr/neural/raven_capability_first.py",
    "tools/nsamdr/neural/raven_capability_full_asset_probe.py",
    "tools/nsamdr/neural/raven_capability_generalization.py",
    "tools/nsamdr/neural/recheck_saved_stage_b.py",
    "tools/nsamdr/neural/run_raven_capability_generalization_latest.py",
    "tools/nsamdr/neural/v9/boundary_specialist.py",
    "tools/nsamdr/neural/v9/edge_crossing_field.py",
    "tools/nsamdr/neural/v9/implicit_boundary.py",
    "tools/nsamdr/neural/v9/topology_anchored_field.py",
)


class TestNsamdrLegacyReferences:
    # Purpose: Implement test only canonical nsamdr batch entrypoints remain for TestNsamdrLegacyReferences.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_only_canonical_nsamdr_batch_entrypoints_remain(self) -> None:
        actual = {
            path.name
            for path in (ROOT / "scripts" / "build").glob("*nsamdr*.bat")
            if path.is_file()
        }
        assert actual == CANONICAL_BATCH_FILES

    # Purpose: Implement test removed alternate routes are absent for TestNsamdrLegacyReferences.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_removed_alternate_routes_are_absent(self) -> None:
        present = [relative for relative in REMOVED_PATHS if (ROOT / relative).exists()]
        assert present == []

    # Purpose: Implement test no version patch test files remain for TestNsamdrLegacyReferences.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_no_version_patch_test_files_remain(self) -> None:
        tests = ROOT / "tools" / "nsamdr" / "tests"
        assert list(tests.glob("test_v*.py")) == []

    # Purpose: Implement test real eve legacy pgs compatibility remains for TestNsamdrLegacyReferences.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def test_real_eve_legacy_pgs_compatibility_remains(self) -> None:
        types = (
            ROOT / "trinityal" / "tests" / "nsamdr" / "NSAMDRPreviewTypes.h"
        ).read_text(encoding="utf-8")
        processor = (
            ROOT / "trinityal" / "tests" / "nsamdr" / "NSAMDRAssetProcessor.cpp"
        ).read_text(encoding="utf-8")
        assert "LegacyPgs" in types
        assert "ShaderFamily::LegacyPgs" in processor
        assert "shaderFamily == ShaderFamily::LegacyPgs" in processor
