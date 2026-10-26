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
            "v176-raven-preview", "v176-raven-channel-audit", "v177-enhancement-train",
            "v178-golden-patch", "v178-contour-train", "v178-golden-fit",
            "v178-golden-auto", "test",
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
            "--mode", "hr_features58",
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
        self.assertNotIn("--target-size", values)

    def test_channel_audit_does_not_preflight_gpu_or_launch_training(self):
        args = self.parser.parse_args([
            "v176-raven-channel-audit", "--crop-size", "128",
            "--max-textures", "4", "--crop-x", "0", "--crop-y", "16",
        ])
        with patch.object(self.app, "_source_freshness_preflight", return_value=0), \
             patch.object(self.app, "_python", return_value=Path("python")), \
             patch.object(self.app, "_cuda_preflight",
                               side_effect=AssertionError("channel audit must not run GPU preflight")), \
             patch.object(self.app, "_run", return_value=0) as run:
            self.assertEqual(self.app._command_raven_channel_audit(args), 0)
        line = [str(v) for v in run.call_args.args[0]]
        self.assertIn("audit_nsamdr_v176_raven_channels.py", " ".join(line))
        self.assertIn("128", line)
        self.assertIn("16", line)
        self.assertNotIn("v176-capacity-proof", line)

    def test_v177_train_and_checkpoint_preview_routing(self):
        args = self.parser.parse_args([
            "v177-enhancement-train", "--steps", "48", "--patch-size", "24",
            "--seed", "900", "--device", "cpu",
        ])
        with patch.object(self.app, "_source_freshness_preflight", return_value=0), \
             patch.object(self.app, "_python", return_value=Path("python")), \
             patch.object(self.app, "_cuda_preflight",
                               side_effect=AssertionError("CPU synthetic run must not preflight CUDA")), \
             patch.object(self.app, "_run", return_value=0) as run:
            self.assertEqual(self.app._command_v177_enhancement(args), 0)
        values = [str(v) for v in run.call_args.args[0]]
        self.assertIn("train_nsamdr_v177_enhancement.py", " ".join(values))
        self.assertIn("48", values)
        self.assertIn("900", values)
        arg = self.parser.parse_args([
            "v176-raven-preview", "--enhancer-checkpoint", "candidate.pt",
        ])
        with patch.object(self.app, "_source_freshness_preflight", return_value=0), \
             patch.object(self.app, "_python", return_value=Path("python")), \
             patch.object(self.app, "_cuda_preflight", return_value=0), \
             patch.object(self.app, "_run", return_value=0) as run:
            self.assertEqual(self.app._command_raven(arg), 0)
        values = [str(v) for v in run.call_args.args[0]]
        self.assertIn("--enhancer-checkpoint", values)
        self.assertIn("candidate.pt", values)

    def test_v178_golden_prepare_has_no_training_or_gpu_preflight(self):
        args = self.parser.parse_args([
            "v178-golden-patch", "--action", "prepare",
            "--crop-x", "320", "--crop-y", "448", "--size", "256",
        ])
        with patch.object(self.app, "_source_freshness_preflight", return_value=0), \
             patch.object(self.app, "_python", return_value=Path("python")), \
             patch.object(self.app, "_cuda_preflight", side_effect=AssertionError("no GPU for evidence")), \
             patch.object(self.app, "_run", return_value=0) as run:
            self.assertEqual(self.app._command_v178_golden(args),0)
        command = [str(v) for v in run.call_args.args[0]]
        self.assertIn("golden_nsamdr_v178.py", " ".join(command))
        self.assertIn("448",command)
        self.assertNotIn("--device",command)

    def test_v178_auto_uses_existing_cuda_runtime_without_trace(self):
        args = self.parser.parse_args([
            "v178-golden-auto", "--folder", "frozen_raven",
            "--checkpoint", "new_model.pt", "--device", "cpu", "--tile", "32",
        ])
        with patch.object(self.app, "_source_freshness_preflight", return_value=0), \
             patch.object(self.app, "_python", return_value=Path("python")), \
             patch.object(self.app, "_cuda_preflight",
                               side_effect=AssertionError("no GPU in CPU mode")), \
             patch.object(self.app, "_run", return_value=0) as run:
            self.assertEqual(self.app._command_v178_auto(args),0)
        command=[str(value) for value in run.call_args.args[0]]
        self.assertIn("preview_nsamdr_v178_automatic.py"," ".join(command))
        self.assertIn("frozen_raven",command)
        self.assertIn("new_model.pt",command)
        self.assertNotIn("trace.json",command)

    def test_v178_real_fit_needs_explicit_trace_folder(self):
        args = self.parser.parse_args([
            "v178-golden-fit", "--folder", "golden_patch", "--device", "cpu",
            "--steps", "12",
        ])
        with patch.object(self.app, "_source_freshness_preflight", return_value=0), \
             patch.object(self.app, "_python", return_value=Path("python")), \
             patch.object(self.app, "_cuda_preflight",
                               side_effect=AssertionError("CPU fit must not preflight CUDA")), \
             patch.object(self.app, "_run", return_value=0) as run:
            self.assertEqual(self.app._command_v178_real_fit(args),0)
        command=[str(v) for v in run.call_args.args[0]]
        self.assertIn("fit_nsamdr_v178_golden.py"," ".join(command))
        self.assertIn("golden_patch",command)
        self.assertIn("12",command)

    def test_v178_contour_uses_own_isolated_training_command(self):
        args = self.parser.parse_args([
            "v178-contour-train", "--steps", "30", "--device", "cpu",
        ])
        with patch.object(self.app, "_source_freshness_preflight", return_value=0), \
             patch.object(self.app, "_python", return_value=Path("python")), \
             patch.object(self.app, "_cuda_preflight", side_effect=AssertionError("no CUDA for cpu training")), \
             patch.object(self.app, "_run", return_value=0) as run:
            self.assertEqual(self.app._command_v178_contour(args),0)
        command = [str(v) for v in run.call_args.args[0]]
        self.assertIn("train_nsamdr_v178_contour.py"," ".join(command))
        self.assertIn("30",command)
        self.assertNotIn("v177-enhancement-train", command)

    def test_legacy_preview_no_longer_callable(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["render-preview", "EXP_0009"])
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["raven-quick"])


if __name__ == "__main__":
    unittest.main()
