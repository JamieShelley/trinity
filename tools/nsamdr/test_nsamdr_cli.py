import unittest

from tools.nsamdr.nsamdr_cli import NSAMDRCommandLineApplication


class NSAMDRCLITests(unittest.TestCase):
    def test_raven_quick_accepts_operator_gui_arguments(self) -> None:
        parser = NSAMDRCommandLineApplication().build_parser()
        args = parser.parse_args(
            [
                "raven-quick",
                "--shared-cache", r"C:\CCP\EVE",
                "--max-train-regions", "16",
                "--max-validation-regions", "4",
                "--experiment", "new",
                "--control", "auto",
                "--preview-target-size", "4096",
                "--preview-device", "cuda",
                "--performance-profile", "fast",
                "--workers", "4",
                "--prefetch-factor", "2",
                "--amp-precision", "auto",
            ]
        )
        self.assertEqual(args.shared_cache, r"C:\CCP\EVE")
        self.assertEqual(args.max_train_regions, 16)
        self.assertEqual(args.max_validation_regions, 4)
        self.assertEqual(args.preview_device, "cuda")
        self.assertEqual(args.amp_precision, "auto")


    def test_render_preview_accepts_research_source_and_watch(self) -> None:
        parser = NSAMDRCommandLineApplication().build_parser()
        args = parser.parse_args(
            [
                "render-preview",
                "EXP_0009",
                "--shared-cache", r"C:\CCP\EVE",
                "--target-size", "1024",
                "--device", "cuda",
                "--watch",
            ]
        )
        self.assertEqual(args.subject, "EXP_0009")
        self.assertEqual(args.target_size, 1024)
        self.assertEqual(args.device, "cuda")
        self.assertTrue(args.watch)

    def test_v17_sibling_proof_accepts_architecture_arguments(self) -> None:
        parser = NSAMDRCommandLineApplication().build_parser()
        args = parser.parse_args(
            [
                "v17-sibling-proof",
                "--device", "cuda",
                "--amp-precision", "auto",
                "--authority-id", "13006d2b807f89ac",
                "--stages", "896,1024,1280,1536",
                "--mid-channels", "64",
                "--detail-channels", "48",
                "--decoder-blocks", "4",
                "--resume", "auto",
            ]
        )
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.amp_precision, "auto")
        self.assertEqual(args.authority_id, "13006d2b807f89ac")
        self.assertEqual(args.stages, ["896,1024,1280,1536"])
        self.assertEqual(args.hr_size, 512)
        self.assertEqual(args.mid_channels, 64)
        self.assertEqual(args.detail_channels, 48)
        self.assertEqual(args.decoder_blocks, 4)
        self.assertEqual(args.resume, "auto")

    def test_v17_5_selection_only_defaults_and_no_gpu_preflight(self) -> None:
        from unittest.mock import patch
        from tools.nsamdr.nsamdr_cli import NSAMDRCommandLineApplication
        app=NSAMDRCommandLineApplication()
        args=app.build_parser().parse_args(["v17-5-diversity-proof","--selection-only"])
        self.assertEqual(args.seed,17301)
        self.assertEqual(args.resume,"fresh")
        self.assertTrue(args.selection_only)
        with patch.object(app,"_source_freshness_preflight",return_value=0), \
             patch.object(app,"_python",return_value="python"), \
             patch.object(app,"_cuda_preflight",side_effect=AssertionError("must not require CUDA")), \
             patch.object(app,"_run",return_value=0) as run:
            self.assertEqual(app._command_v17_5_diversity(args),0)
        self.assertIn("--selection-only",run.call_args.args[0])

    def test_v17_5_training_preflights_cuda_and_forwards_seed(self) -> None:
        from unittest.mock import patch
        from tools.nsamdr.nsamdr_cli import NSAMDRCommandLineApplication
        app=NSAMDRCommandLineApplication()
        args=app.build_parser().parse_args([
            "v17-5-diversity-proof","--resume","auto","--seed","17302",
            "--selection","frozen.json"])
        with patch.object(app,"_source_freshness_preflight",return_value=0), \
             patch.object(app,"_python",return_value="python"), \
             patch.object(app,"_cuda_preflight",return_value=0) as preflight, \
             patch.object(app,"_run",return_value=0) as run:
            self.assertEqual(app._command_v17_5_diversity(args),0)
        preflight.assert_called_once()
        command=run.call_args.args[0]
        self.assertIn("17302",command)
        self.assertIn("frozen.json",command)
        self.assertIn("auto",command)

    def test_stage9_audit_uses_cpu_only_preflight(self) -> None:
        from unittest.mock import patch
        from tools.nsamdr.nsamdr_cli import NSAMDRCommandLineApplication
        app=NSAMDRCommandLineApplication()
        args=app.build_parser().parse_args([
            "v17-5-matched-continuation","--audit-only",
            "--stage8-report","parent.json"])
        with patch.object(app,"_source_freshness_preflight",return_value=0), \
             patch.object(app,"_python",return_value="python"), \
             patch.object(app,"_cuda_preflight",side_effect=AssertionError("audit never preflights GPU")), \
             patch.object(app,"_run",return_value=0) as dispatch:
            self.assertEqual(app._command_v17_5_matched_continuation(args),0)
        command=dispatch.call_args.args[0]
        self.assertIn("--audit-only",command)
        self.assertIn("parent.json",command)

    def test_stage9_cuda_run_checks_gpu_and_forwards_original_precision(self) -> None:
        from unittest.mock import patch
        from tools.nsamdr.nsamdr_cli import NSAMDRCommandLineApplication
        app=NSAMDRCommandLineApplication()
        args=app.build_parser().parse_args([
            "v17-5-matched-continuation","--resume","auto",
            "--amp-precision","auto",
            "--selection-only-report","preview.json"])
        with patch.object(app,"_source_freshness_preflight",return_value=0), \
             patch.object(app,"_python",return_value="python"), \
             patch.object(app,"_cuda_preflight",return_value=0) as check, \
             patch.object(app,"_run",return_value=0) as dispatch:
            self.assertEqual(app._command_v17_5_matched_continuation(args),0)
        check.assert_called_once()
        command=dispatch.call_args.args[0]
        self.assertIn("--resume",command)
        self.assertIn("auto",command)
        self.assertIn("preview.json",command)
        self.assertNotIn("--audit-only",command)

    def test_v176_read_only_audit_bypasses_cuda_preflight(self) -> None:
        from unittest.mock import patch
        from tools.nsamdr.nsamdr_cli import NSAMDRCommandLineApplication
        app=NSAMDRCommandLineApplication()
        args=app.build_parser().parse_args([
            "v176-detail-proof","--audit-only","--stage9-report","stage9.json"])
        with patch.object(app,"_source_freshness_preflight",return_value=0), \
             patch.object(app,"_python",return_value="python"), \
             patch.object(app,"_cuda_preflight",side_effect=AssertionError("audit must not require CUDA")), \
             patch.object(app,"_run",return_value=0) as dispatch:
            self.assertEqual(app._command_v176_detail(args),0)
        command=dispatch.call_args.args[0]
        self.assertIn("--audit-only",command)
        self.assertIn("stage9.json",command)

    def test_v176_training_preflight_and_resume_dispatch(self) -> None:
        from unittest.mock import patch
        from tools.nsamdr.nsamdr_cli import NSAMDRCommandLineApplication
        app=NSAMDRCommandLineApplication()
        args=app.build_parser().parse_args([
            "v176-detail-proof","--resume","auto","--amp-precision","auto"])
        with patch.object(app,"_source_freshness_preflight",return_value=0), \
             patch.object(app,"_python",return_value="python"), \
             patch.object(app,"_cuda_preflight",return_value=0) as preflight, \
             patch.object(app,"_run",return_value=0) as dispatch:
            self.assertEqual(app._command_v176_detail(args),0)
        preflight.assert_called_once()
        command=dispatch.call_args.args[0]
        self.assertIn("auto",command)
        self.assertNotIn("--audit-only",command)

    def test_context_probe_accepts_bounded_arguments(self) -> None:
        parser = NSAMDRCommandLineApplication().build_parser()
        args = parser.parse_args(
            [
                "context-probe",
                "--device", "cuda",
                "--amp-precision", "auto",
                "--steps", "596",
                "--preview-samples", "4",
            ]
        )
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.amp_precision, "auto")
        self.assertEqual(args.steps, 596)
        self.assertEqual(args.preview_samples, 4)

    def test_main_train_accepts_operator_gui_arguments(self) -> None:
        parser = NSAMDRCommandLineApplication().build_parser()
        args = parser.parse_args(
            [
                "main-train",
                "--device", "cuda",
                "--amp-precision", "auto",
                "--epochs", "1",
                "--preview-samples", "4",
                "--shared-cache", r"C:\CCP\EVE",
            ]
        )
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.amp_precision, "auto")
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.d4_passes, 0)
        self.assertEqual(args.preview_samples, 4)
        self.assertEqual(args.shared_cache, r"C:\CCP\EVE")


if __name__ == "__main__":
    unittest.main()
