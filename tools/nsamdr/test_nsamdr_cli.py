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
                "--stages", "512,768,1024,1536",
            ]
        )
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.amp_precision, "auto")
        self.assertEqual(args.authority_id, "13006d2b807f89ac")
        self.assertEqual(args.stages, ["512,768,1024,1536"])
        self.assertEqual(args.hr_size, 512)

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
