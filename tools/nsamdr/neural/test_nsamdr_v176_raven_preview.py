"""V17.6 Raven loader contract: training source only, no target-assisted field."""
from __future__ import annotations
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from tools.nsamdr.neural import render_nsamdr_v176_raven_preview as raven


class V176RavenPreviewTests(unittest.TestCase):
    def test_research_only_schema_and_models(self):
        self.assertEqual(raven.SCHEMA, "NSAMDR_V176_RAVEN_RESEARCH_PREVIEW_V1")
        self.assertEqual(raven.STAGE, 6144)
        p = raven.parser()
        parsed = p.parse_args(["--authority-index", "1", "--mode", "hr_features58"])
        self.assertEqual((parsed.authority_index, parsed.mode), (1, "hr_features58"))

    def test_unavailable_comparison_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "No authenticated"):
                raven._resolve_comparison(Path(root), 10, "output10")

    def test_generic_renderer_requires_preverified_model(self):
        from tools.nsamdr.neural.v176 import raven_renderer
        with self.assertRaisesRegex(RuntimeError, "authenticated model"):
            with tempfile.TemporaryDirectory() as tmp:
                empty = Path(tmp) / "empty.pt"
                empty.write_bytes(b"test")
                source = raven_renderer.PreviewSource("V176", 0, "research", empty)
                raven_renderer._generate_candidate(
                    repo_root=Path(tmp), source=source, target_size=1024,
                    device=raven.torch.device("cpu"), obj_path=Path(tmp) / "x.obj",
                    materials=Path(tmp) / "materials.tsv",
                    model_override=None,
                )


if __name__ == "__main__":
    unittest.main()
