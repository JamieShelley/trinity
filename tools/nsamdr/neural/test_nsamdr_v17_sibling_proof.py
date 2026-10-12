from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.nsamdr.neural.probe_nsamdr_v17_sibling_architecture import (
    SCHEMA,
    _resolve_resume_checkpoint,
)


class V17SiblingProofResumeTests(unittest.TestCase):
    def test_auto_resume_selects_latest_compatible_v17_2_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            repo_root = Path(root)
            proof_root = (
                repo_root
                / "artifacts/nsamdr/diagnostics/v17_sibling_architecture"
            )
            authority = "13006d2b807f89ac"

            for index in (1, 2):
                run_dir = proof_root / f"proof_{index:02d}"
                run_dir.mkdir(parents=True)
                checkpoint = run_dir / "checkpoint.pt"
                checkpoint.write_bytes(f"checkpoint-{index}".encode("ascii"))
                report = {
                    "schema": SCHEMA,
                    "authorityId": authority,
                    "architecture": {"revision": "V17.2-proof"},
                    "siblingEvaluated": False,
                    "checkpoint": str(checkpoint.resolve()),
                }
                (run_dir / "report.json").write_text(
                    json.dumps(report),
                    encoding="utf-8",
                )
                checkpoint.touch()

            selected = _resolve_resume_checkpoint(
                repo_root,
                "auto",
                authority_id=authority,
            )
            self.assertIsNotNone(selected)
            self.assertEqual(selected, (proof_root / "proof_02/checkpoint.pt").resolve())

    def test_fresh_disables_resume(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            selected = _resolve_resume_checkpoint(
                Path(root),
                "fresh",
                authority_id="13006d2b807f89ac",
            )
            self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
