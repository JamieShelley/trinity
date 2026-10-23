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
        cmd = raven_command(r"C:\\CCP\\EVE", 10, "output10")
        self.assertEqual(cmd[0], "v176-raven-preview")
        self.assertEqual(cmd[-3:], ["10", "--mode", "output10"])
        self.assertNotIn("--target-size", cmd)

    def test_bad_selection_rejected(self):
        with self.assertRaises(ValueError):
            raven_command(r"C:\\CCP\\EVE", 11, "output10")
        with self.assertRaises(ValueError):
            training_command("v16-main", "fresh", False, "auto")

    def test_latest_report_ignores_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "proof_1"
            folder.mkdir()
            (folder / "report.json").write_text("{bad")
            self.assertIsNone(latest_report(Path(tmp)))

    def test_full_operator_stage_tree_restored(self):
        import inspect
        from tools.nsamdr.gui.nsamdr_v176_workflow_gui import App, STAGES
        self.assertEqual([s.id for s in STAGES], [
            "setup", "train", "compare", "attainable",
            "preview", "evidence", "tests", "quadtree",
        ])
        source = inspect.getsource(App._build)
        for item in ("ttk.Treeview", "ttk.Panedwindow", "controls_canvas",
                     "Command preview", "progress_value", "Stop current process",
                     "Detect artifacts", "preview_combo", "Runtime log"):
            self.assertIn(item, source)

    def test_current_stage_commands_are_wired_without_tk_instance(self):
        from tools.nsamdr.gui.nsamdr_v176_workflow_gui import App
        class Var:
            def __init__(self, value):
                self.value = value
            def get(self):
                return self.value
        app = App.__new__(App)
        app.vars = {
            "phase": Var("compare-features"), "resume": Var("fresh"),
            "amp": Var("auto"), "device": Var("cuda"), "audit": Var(False),
            "authority": Var("10"), "mode": Var("output10"),
            "cache": Var(r"C:\CCP\EVE"),
        }
        actual = app._command_for_stage("compare")
        self.assertIn("v176-capacity-proof", actual)
        self.assertIn("compare-features", actual)
        self.assertNotIn("attainable", actual)
        actual = app._command_for_stage("preview")
        self.assertIn("v176-raven-preview", actual)
        self.assertIn("output10", actual)
        self.assertIsNone(app._command_for_stage("quadtree"))

    def test_footer_source_matches_renderer_folder(self):
        from tools.nsamdr.gui.nsamdr_v176_workflow_gui import (
            PREVIEW_CHOICES, preview_label, App,
        )
        self.assertEqual(set(PREVIEW_CHOICES), {
            "TRAIN10_OUTPUT10", "TRAIN10_HR_FEATURES58",
            "TRAIN01_OUTPUT10", "TRAIN01_HR_FEATURES58",
        })
        for value in PREVIEW_CHOICES:
            authority = int(value[5:7])
            mode = value.split("_", 1)[1].lower()
            self.assertTrue(preview_label(authority, mode).endswith(value))


if __name__ == "__main__":
    unittest.main()
