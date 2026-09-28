from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.nsamdr.neural.render_nsamdr_v16_training_preview import (
    LIVE_POINTER_SCHEMA,
    _pointer_text,
    _resolve_raven_source,
)


class NSAMDRV16RenderPreviewTests(unittest.TestCase):
    def test_live_pointer_contains_native_renderer_contract(self) -> None:
        report = {
            "source": "EXP_0009",
            "ordinal": 8,
            "phase": "sr-robust",
            "checkpointSha256": "a" * 64,
            "baselineObj": r"C:\preview\ship.obj",
            "baselineMaterials": r"C:\preview\baseline.materials.tsv",
            "candidateObj": r"C:\preview\ship.obj",
            "candidateMaterials": r"C:\preview\candidate.materials.tsv",
            "reportPath": r"C:\preview\candidate_manifest.json",
        }
        text = _pointer_text(report)
        self.assertTrue(text.startswith(LIVE_POINTER_SCHEMA + "\n"))
        self.assertIn("stageVariant=C-candidate", text)
        self.assertIn("baselineObj=", text)
        self.assertIn("baselineMaterials=", text)
        self.assertIn("candidateObj=", text)
        self.assertIn("candidateMaterials=", text)
        self.assertIn("authority=training-intermediate", text)
        self.assertIn("qualified=false", text)

    def test_raven_source_falls_back_to_latest_epoch_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            repo = Path(root)
            experiment = repo / "artifacts/nsamdr/experiments/EXP_0009"
            checkpoints = experiment / "checkpoints/candidate"
            checkpoints.mkdir(parents=True)
            (checkpoints / "epoch_0006.pt").write_bytes(b"six")
            latest = checkpoints / "epoch_0008.pt"
            latest.write_bytes(b"eight")
            pointer = experiment / "previews/live/checkpoint_ready.json"
            pointer.parent.mkdir(parents=True)
            pointer.write_text(
                json.dumps(
                    {
                        "schema": "NSAMDR_V16_LIVE_PREVIEW_V1",
                        "epoch": 8,
                        "phase": "sr-robust",
                    }
                ),
                encoding="utf-8",
            )

            source = _resolve_raven_source(repo, "EXP_0009")
            self.assertIsNotNone(source)
            assert source is not None
            self.assertEqual(source.ordinal, 8)
            self.assertEqual(source.phase, "sr-robust")
            self.assertEqual(source.checkpoint, latest.resolve())


if __name__ == "__main__":
    unittest.main()
