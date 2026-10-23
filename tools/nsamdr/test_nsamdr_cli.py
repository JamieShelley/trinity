"""Only the V17.6 operator surface is supported."""
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.nsamdr.nsamdr_cli import NSAMDRCommandLineApplication


class V176CLITests(unittest.TestCase):
    def setUp(self):
        self.app = NSAMDRCommandLineApplication()
        self.parser = self.app.build_parser()

    def test_only_current_operator_commands(self):
        accepted = {action for action in
                    self.parser._subparsers._group_actions[0].choices}
        self.assertEqual(accepted, {
            "gui", "setup", "v176-detail-proof", "v176-capacity-proof",
            "v176-raven-preview", "test",
        })

    def test_v176_capacity_passes_phase_resume_and_preserves_audit(self):
        args = self.parser.parse_args([
            "v176-capacity-proof", "--phase", "attainable", "--resume", "auto",
            "--comparison-report", "comparison.json", "--audit-only",
        ])
        with patch.object(self.app, "_source_freshness_preflight", return_value=0), \
             patch.object(self.app, "_python", return_value=Path("python")), \
             patch.object(self.app, "_cuda_preflight", side_effect=AssertionError("audit must not preflight GPU")), \
             patch.object(self.app, "_run", return_value=0) as run:
            self.assertEqual(self.app._command_v176_capacity(args), 0)
        values = [str(v) for v in run.call_args.args[0]]
        self.assertIn("--audit-only", values)
        self.assertIn("attainable", values)
        self.assertIn("comparison.json", values)

    def test_training_requires_cuda_preflight(self):
        args = self.parser.parse_args([
            "v176-capacity-proof", "--phase", "compare-features",
        ])
        with patch.object(self.app, "_source_freshness_preflight", return_value=0), \
             patch.object(self.app, "_python", return_value=Path("python")), \
             patch.object(self.app, "_cuda_preflight", return_value=0) as preflight, \
             patch.object(self.app, "_run", return_value=0) as run:
            self.assertEqual(self.app._command_v176_capacity(args), 0)
        preflight.assert_called_once()
        self.assertIn("compare-features", run.call_args.args[0])

    def test_raven_requires_verified_source_and_accepts_mode(self):
        args = self.parser.parse_args([
            "v176-raven-preview", "--authority-index", "10",
            "--mode", "hr_features58", "--target-size", "1024",
        ])
        with patch.object(self.app, "_source_freshness_preflight", return_value=0), \
             patch.object(self.app, "_python", return_value=Path("python")), \
             patch.object(self.app, "_cuda_preflight", return_value=0), \
             patch.object(self.app, "_run", return_value=0) as run:
            self.assertEqual(self.app._command_raven(args), 0)
        values = [str(v) for v in run.call_args.args[0]]
        self.assertIn("render_nsamdr_v176_raven_preview.py", " ".join(values))
        self.assertIn("hr_features58", values)
        self.assertIn("10", values)

    def test_legacy_preview_no_longer_callable(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["render-preview", "EXP_0009"])
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["raven-quick"])


if __name__ == "__main__":
    unittest.main()
