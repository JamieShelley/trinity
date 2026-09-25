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


if __name__ == "__main__":
    unittest.main()
