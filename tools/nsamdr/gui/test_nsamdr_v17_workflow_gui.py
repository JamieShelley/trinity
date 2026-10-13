import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.nsamdr.gui.nsamdr_v17_workflow_gui import (
    APP_TITLE,
    App,
    V17_PREVIEW_CHOICE,
    V17_GENERALIZATION_CHOICE,
    V17_V2_CHOICE,
    STAGES,
    _format_duration,
    _qualified_final,
)


class NSAMDRV17WorkflowGuiTests(unittest.TestCase):
    def test_v17_gui_identity(self) -> None:
        self.assertEqual(APP_TITLE, "NSAMDR V17 Workflow")
        self.assertEqual(V17_PREVIEW_CHOICE, "V17_SIBLING_LATEST")
        self.assertEqual(V17_GENERALIZATION_CHOICE, "V17_GENERALIZATION_LATEST")
        self.assertEqual(V17_V2_CHOICE, "V17_GENERALIZATION_V2_LATEST")
        self.assertIn("generalize", [stage.id for stage in STAGES])
        self.assertIn("generalize_v2", [stage.id for stage in STAGES])
        self.assertIn("generalize_v3", [stage.id for stage in STAGES])
        self.assertIn("comprehensive_af", [stage.id for stage in STAGES])
        self.assertIn("source_readiness", [stage.id for stage in STAGES])
        self.assertIn("diversity_proof", [stage.id for stage in STAGES])
        self.assertIn("matched_continuation", [stage.id for stage in STAGES])
        self.assertEqual(next(stage.number for stage in STAGES if stage.id=="preview"),"10")
        self.assertEqual(STAGES[-1].id, "preview")

    def test_completed_checkpoint_detection_does_not_require_tk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "proof_01/report.json"
            report_path.parent.mkdir(parents=True)
            checkpoint = report_path.parent / "stage_008192/checkpoint.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"saved")
            app = App.__new__(App)
            report = (report_path, {"stopUpdate": 8192})
            with patch.object(App, "_latest_v17_generalization_report", return_value=report):
                self.assertEqual(app._completed_v17_generalization_update(), 8192)
                checkpoint.unlink()
                self.assertIsNone(app._completed_v17_generalization_update())

    def test_v2_preview_finds_best_selected_validation_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / "proof_01/report.json"
            report.parent.mkdir()
            sheet = root / "selected_006144/validation_00_ABC.png"
            sheet.parent.mkdir()
            sheet.write_bytes(b"image")
            app = App.__new__(App)
            data = (report, {
                "validationPreviews": {"train_001": str(sheet)},
                "finalTestPreviews": {},
            })
            with patch.object(App, "_latest_v17_v2_report", return_value=data):
                self.assertEqual(app._latest_v17_v2_previews(), sheet.resolve())

    def test_4096_visual_folder_is_scoped_to_latest_v2_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / "proof_01/report.json"
            folder = root / "proof_01/resolution_4096/visual_01"
            folder.mkdir(parents=True)
            (folder / "report.json").write_text('{"schema":"4096"}')
            app = App.__new__(App)
            with patch.object(App, "_latest_v17_v2_report",
                              return_value=(report, {"schema": "v2"})):
                self.assertEqual(app._latest_v17_4096_visual_dir(), folder.resolve())
                (folder / "report.json").unlink()
                self.assertIsNone(app._latest_v17_4096_visual_dir())

    def test_v17_v3_gui_builds_controlled_match_command(self) -> None:
        app = App.__new__(App)
        app.vars = {}
        command = app._command_for_stage("generalize_v3")
        self.assertIsNotNone(command)
        self.assertIn("v17-spatial-generalization", command)
        self.assertIn("both", command)
        self.assertIn("12288,24576,36864,49152", command)
        self.assertIn("fresh", command)

    def test_v17_v3_report_discovery_requires_correct_schema(self) -> None:
        from tools.nsamdr.gui import nsamdr_v17_workflow_gui as gui
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "proof_01/report.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"schema":"NSAMDR_V17_SPATIAL_GENERALIZATION_PROOF_V1",
                                        "decision":"reject-independent-validation"}))
            app = App.__new__(App)
            with patch.object(gui, "V17_V3_ROOT", root):
                found = app._latest_v17_v3_report()
                self.assertIsNotNone(found)
                self.assertEqual(found[0], path.resolve())
                path.write_text('{"schema":"not-v17"}')
                self.assertIsNone(app._latest_v17_v3_report())

    def test_af_gui_builds_automatic_a_to_f_command(self) -> None:
        app = App.__new__(App)
        app.vars = {}
        command = app._command_for_stage("comprehensive_af")
        self.assertIsNotNone(command)
        self.assertIn("v17-comprehensive-af", command)
        self.assertIn("fresh", command)
        self.assertIn("cuda", command)
        self.assertNotIn("--audit-only", command)

    def test_af_gui_report_discovery_is_strict(self) -> None:
        from tools.nsamdr.gui import nsamdr_v17_workflow_gui as gui
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / "proof_01/report.json"
            report.parent.mkdir()
            report.write_text(json.dumps({
                "schema": "NSAMDR_V17_COMPREHENSIVE_AF_PROOF_V1",
                "decision": "reject-independent-validation",
                "finalTestEvaluated": False,
            }))
            app = App.__new__(App)
            with patch.object(gui, "V17_AF_ROOT", root):
                found = app._latest_v17_af_report()
                self.assertIsNotNone(found)
                self.assertEqual(found[0], report.resolve())
                report.write_text('{"schema":"not-a-f"}')
                self.assertIsNone(app._latest_v17_af_report())

    def test_source_readiness_gui_creates_cpu_only_command(self) -> None:
        app=App.__new__(App)
        app.vars={}
        command=app._command_for_stage("source_readiness")
        self.assertIsNotNone(command)
        self.assertIn("v17-source-readiness",command)
        self.assertIn("--device",command)
        self.assertIn("cpu",command)
        self.assertNotIn("--skip-conditioning",command)
        self.assertNotIn("v17-comprehensive-af",command)

    def test_source_readiness_strict_report_discovery(self) -> None:
        from tools.nsamdr.gui import nsamdr_v17_workflow_gui as gui
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            report=root/"proof_01/report.json"
            report.parent.mkdir()
            report.write_text(json.dumps({
                "schema":"NSAMDR_V17_5_SOURCE_READINESS_V1",
                "decision":"requires-verification-and-controlled-training",
                "finalTestEvaluated":False,
            }))
            app=App.__new__(App)
            with patch.object(gui,"V17_SOURCE_ROOT",root):
                latest=app._latest_v17_source_report()
                self.assertIsNotNone(latest)
                self.assertEqual(latest[0],report.resolve())
                report.write_text('{"schema":"other-proof"}')
                self.assertIsNone(app._latest_v17_source_report())

    def test_stage8_gui_command_preserves_matched_defaults(self) -> None:
        app=App.__new__(App)
        app.vars={}
        command=app._command_for_stage("diversity_proof")
        self.assertIn("v17-5-diversity-proof",command)
        self.assertIn("cuda",command)
        self.assertIn("fresh",command)
        self.assertIn("17301",command)
        self.assertNotIn("--selection-only",command)
        self.assertNotIn("--selection",command)

    def test_stage8_gui_can_run_selection_only_without_retraining(self) -> None:
        from unittest.mock import MagicMock
        app=App.__new__(App)
        app.vars={"selection_only":MagicMock()}
        app.vars["selection_only"].get.return_value=True
        command=app._command_for_stage("diversity_proof")
        self.assertIn("--selection-only",command)

    def test_stage8_report_discovery_requires_exact_schema(self) -> None:
        from tools.nsamdr.gui import nsamdr_v17_workflow_gui as gui
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            report=root/"proof_01/report.json"
            report.parent.mkdir()
            report.write_text(json.dumps({
                "schema":"NSAMDR_V17_5_NATIVE_AUTHORITY_DIVERSITY_PROOF_V1",
                "decision":"NO_DEMONSTRATED_BENEFIT","researchOnly":True,
                "finalTestEvaluated":False,
            }))
            app=App.__new__(App)
            with patch.object(gui,"V17_DIVERSITY_ROOT",root):
                self.assertEqual(app._latest_v17_diversity_report()[0],report.resolve())
                report.write_text('{"schema":"wrong"}')
                self.assertIsNone(app._latest_v17_diversity_report())

    def test_stage9_gui_defaults_to_read_only_provenance(self) -> None:
        from unittest.mock import MagicMock
        app=App.__new__(App)
        app.vars={"audit_only":MagicMock()}
        app.vars["audit_only"].get.return_value=True
        cmd=app._command_for_stage("matched_continuation")
        self.assertIn("v17-5-matched-continuation",cmd)
        self.assertIn("--audit-only",cmd)
        self.assertIn("--device",cmd)
        self.assertIn("cuda",cmd)
        self.assertNotIn("--selection",cmd)

    def test_stage9_gui_training_command_and_explicit_parent(self) -> None:
        from unittest.mock import MagicMock
        app=App.__new__(App)
        app.vars={"stage8_report":MagicMock(),"selection_only_report":MagicMock()}
        app.vars["stage8_report"].get.return_value=r"D:\\proof\\stage8.json"
        app.vars["selection_only_report"].get.return_value=r"D:\\proof\\preview.json"
        cmd=app._command_for_stage("matched_continuation")
        self.assertIn("--stage8-report",cmd)
        self.assertIn("--selection-only-report",cmd)
        self.assertNotIn("--audit-only",cmd)
        self.assertIn("fresh",cmd)

    def test_stage9_report_discovery_is_schema_strict(self) -> None:
        from tools.nsamdr.gui import nsamdr_v17_workflow_gui as gui
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            path=root/"proof_01/report.json"
            path.parent.mkdir()
            path.write_text(json.dumps({
                "schema":"NSAMDR_V17_5_MATCHED_CONTINUATION_PROOF_V1",
                "researchOnly":True,"finalTestEvaluated":False,
                "decision":"EXPOSURE_BENEFIT_SIGNAL"}))
            app=App.__new__(App)
            with patch.object(gui,"V17_CONTINUATION_ROOT",root):
                self.assertEqual(app._latest_v17_continuation_report()[0],path.resolve())
                path.write_text('{"schema":"unknown"}')
                self.assertIsNone(app._latest_v17_continuation_report())

    def test_format_duration(self) -> None:
        self.assertEqual(_format_duration(65), "1m 05s")
        self.assertEqual(_format_duration(3661), "1h 01m")

    def test_qualified_final_rejects_incomplete_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(_qualified_final(Path(root)))

    def test_qualified_final_accepts_complete_immutable_final(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            experiment = Path(root) / "EXP_0001"
            final_dir = experiment / "checkpoints/final"
            final_dir.mkdir(parents=True)
            checkpoint = final_dir / "nsamdr.pt"
            checkpoint.write_bytes(b"x")
            (experiment / "experiment.json").write_text(
                json.dumps({"qualified": True, "status": "completed"}),
                encoding="utf-8",
            )
            (experiment / "architecture_participation.json").write_text(
                json.dumps({"pass": True}),
                encoding="utf-8",
            )
            (experiment / "final_manifest.json").write_text(
                json.dumps(
                    {
                        "qualified": True,
                        "status": "completed",
                        "selectionKind": "production-final",
                        "checkpoint": {
                            "path": "checkpoints/final/nsamdr.pt",
                            "sha256": "a" * 64,
                            "immutable": True,
                            "selectionKind": "production-final",
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_qualified_final(experiment))


if __name__ == "__main__":
    unittest.main()
