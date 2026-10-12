"""Guardrails for bounded V17.2 generalization research."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from tools.nsamdr.neural.probe_nsamdr_v17_generalization import (
    _augment_batch,
    _resume_path,
    _score,
    _select_pairs,
)


class V17GeneralizationProofTests(unittest.TestCase):
    def test_selects_distinct_training_and_heldout_crops(self) -> None:
        crops = [
            {
                "split": "train", "family_id": authority,
                "crop_id": f"{authority}_{number:03d}",
                "path": f"{authority}_{number}.npz",
            }
            for authority in ("anchor", "other", "third")
            for number in (0, 1)
        ]
        pairs = _select_pairs({"crops": list(reversed(crops))}, "anchor", 2)
        self.assertEqual([a["family_id"] for a, _ in pairs], ["anchor", "other"])
        self.assertEqual([a["crop_id"][-3:] for a, _ in pairs], ["000", "000"])
        self.assertEqual([b["crop_id"][-3:] for _, b in pairs], ["001", "001"])
        self.assertTrue(
            {a["crop_id"] for a, _ in pairs}.isdisjoint(
                {b["crop_id"] for _, b in pairs}
            )
        )

    def test_rejects_missing_sibling_or_duplicate_record(self) -> None:
        crops = [
            {"split": "train", "family_id": "anchor", "crop_id": "a0", "path": "a0"},
            {"split": "train", "family_id": "anchor", "crop_id": "a1", "path": "a1"},
            {"split": "train", "family_id": "bad", "crop_id": "b0", "path": "same"},
            {"split": "train", "family_id": "bad", "crop_id": "b1", "path": "same"},
        ]
        with self.assertRaises(RuntimeError):
            _select_pairs({"crops": crops}, "anchor", 2)

    def test_d4_identity_and_normal_transform(self) -> None:
        batch = {
            key: torch.ones((1, channels, 2, 2))
            for key, channels in (
                ("lr_albedo", 3), ("lr_normal", 2), ("lr_material", 3),
                ("target_albedo", 3), ("target_normal", 2),
                ("target_material", 3),
            )
        }
        for key in ("lr_normal", "target_normal"):
            batch[key][:, 1:2] = 0.0
        identity = _augment_batch(batch, 0)
        for key in batch:
            self.assertTrue(torch.equal(identity[key], batch[key]))
        rotated = _augment_batch(batch, 1)
        self.assertTrue(torch.all(rotated["target_normal"][:, 0:1] == 0.0))
        self.assertTrue(torch.all(rotated["target_normal"][:, 1:2] == 1.0))
        mirrored = _augment_batch(batch, 4)
        self.assertTrue(torch.all(mirrored["lr_normal"][:, 0:1] == -1.0))
        self.assertTrue(torch.all(mirrored["lr_normal"][:, 1:2] == 0.0))
        self.assertTrue(torch.all(batch["lr_normal"][:, 0:1] == 1.0))
        with self.assertRaises(ValueError):
            _augment_batch(batch, 8)

    def test_scoring_requires_each_crop_to_pass(self) -> None:
        metrics = {
            "global_recovery": 0.4, "edge_recovery": 0.5,
            "gradient_recovery": 0.3, "detail_recovery_1px": 0.1,
            "lattice_cell_excess": 0.1,
        }
        results = [
            {"metrics": metrics, "gateChecks": {"global": True, "edge": True}},
            {"metrics": metrics, "gateChecks": {"global": True, "edge": False}},
        ]
        self.assertFalse(_score(results, sibling=False)["allPass"])

    def test_auto_resume_excludes_completed_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proof = root / "artifacts/nsamdr/diagnostics/v17_generalization"
            completed = proof / "proof_01"
            interrupted = proof / "proof_02"
            for directory in (completed, interrupted):
                stage = directory / "stage_002048"
                stage.mkdir(parents=True)
                (stage / "checkpoint.pt").write_bytes(b"test")
            (completed / "report.json").write_text("{}")
            self.assertEqual(
                _resume_path(root, "auto"),
                interrupted / "stage_002048/checkpoint.pt",
            )


if __name__ == "__main__":
    unittest.main()
