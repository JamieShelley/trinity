import json
import tempfile
import unittest
from pathlib import Path

from tools.nsamdr.gui.nsamdr_v17_workflow_gui import (
    APP_TITLE,
    V17_PREVIEW_CHOICE,
    _format_duration,
    _qualified_final,
)


class NSAMDRV17WorkflowGuiTests(unittest.TestCase):
    def test_v17_gui_identity(self) -> None:
        self.assertEqual(APP_TITLE, "NSAMDR V17 Workflow")
        self.assertEqual(V17_PREVIEW_CHOICE, "V17_SIBLING_LATEST")

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
