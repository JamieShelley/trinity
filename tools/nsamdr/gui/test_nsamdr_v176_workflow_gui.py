"""V17.6 GUI command routing tests (headless; no Tk display needed)."""
import tempfile
from pathlib import Path
import unittest

from tools.nsamdr.gui.nsamdr_v176_workflow_gui import (
    PHASES, latest_report, raven_command, training_command,
)


class V176OperatorGuiTests(unittest.TestCase):
    def test_train_comparison_is_real_optimisation(self):
        cmd = training_command("compare-features", "fresh", False, "auto")
        self.assertEqual(cmd[:3], ["v176-capacity-proof", "--phase", "compare-features"])
        self.assertNotIn("--audit-only", cmd)

    def test_attainable_is_separate_diagnostic(self):
        self.assertIn("attainable", PHASES)
        self.assertIn("attainable", training_command("attainable", "fresh", False, "auto"))

    def test_raven_selects_current_specialist(self):
        cmd = raven_command(r"C:\\CCP\\EVE", 1024, 10, "output10")
        self.assertEqual(cmd[0], "v176-raven-preview")
        self.assertEqual(cmd[-3:], ["10", "--mode", "output10"])

    def test_bad_selection_rejected(self):
        with self.assertRaises(ValueError):
            raven_command(r"C:\\CCP\\EVE", 8192, 10, "output10")
        with self.assertRaises(ValueError):
            training_command("v16-main", "fresh", False, "auto")

    def test_latest_report_ignores_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "proof_1"
            folder.mkdir()
            (folder / "report.json").write_text("{bad")
            self.assertIsNone(latest_report(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
