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
