"""Guardrails for bounded V17.2 generalization research."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.nsamdr.nsamdr_cli import _app
from tools.nsamdr.neural.probe_nsamdr_v17_generalization import (
    CHECKPOINT_SCHEMA,
    SCHEMA,
    _augment_batch,
    _checkpoint,
    _load_checkpoint,
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

    def test_managed_cli_exposes_isolated_generalization(self) -> None:
        args = _app.build_parser().parse_args(["v17-generalization-proof"])
        self.assertEqual(args.command, "v17-generalization-proof")
        self.assertEqual(args.authorities, 4)
        self.assertEqual(args.stages, ["2048,4096,6144,8192"])
        self.assertEqual(args.resume, "auto")

    def test_continue_latest_completed_uses_final_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proof = root / "artifacts/nsamdr/diagnostics/v17_generalization"
            older = proof / "proof_01"
            latest = proof / "proof_02"
            invalid = proof / "proof_03"
            for directory, update in ((older, 6144), (latest, 8192), (invalid, 12288)):
                stage = directory / f"stage_{update:06d}"
                stage.mkdir(parents=True)
                (stage / "checkpoint.pt").write_bytes(b"test")
                (directory / "report.json").write_text(json.dumps({
                    "schema": SCHEMA if directory != invalid else "other-proof",
                    "stopUpdate": update,
                    "decision": "reject-generalization",
                }))
            expected = latest / "stage_008192/checkpoint.pt"
            self.assertEqual(_resume_path(root, "Continue latest completed"), expected.resolve())
            self.assertEqual(_resume_path(root, "completed"), expected.resolve())
            self.assertIsNone(_resume_path(root, "fresh"))
            self.assertIsNone(_resume_path(root, "auto"))

    def test_completed_mode_does_not_silently_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "No completed"):
                _resume_path(Path(tmp), "Continue latest completed")

    def test_resume_restores_model_optimizer_and_historical_snapshots(self) -> None:
        class TinyModel(torch.nn.Linear):
            def architecture_contract(self) -> dict[str, str]:
                return {"revision": "test"}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "stage_008192/checkpoint.pt"
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            first = TinyModel(2, 2)
            optimizer = torch.optim.Adam(first.parameters(), lr=0.001)
            loss = first(torch.ones(1, 2)).square().sum()
            loss.backward()
            optimizer.step()
            original = [param.detach().clone() for param in first.parameters()]
            historical = [{"update": 2048}, {"update": 8192}]
            _checkpoint(checkpoint, first, optimizer, manifest,
                        ["train_000"], 8192, historical)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema"], CHECKPOINT_SCHEMA)
            second = TinyModel(2, 2)
            second_optimizer = torch.optim.Adam(second.parameters(), lr=0.001)
            update, snapshots = _load_checkpoint(
                checkpoint, second, second_optimizer,
                manifest, ["train_000"], torch.device("cpu"),
            )
            self.assertEqual(update, 8192)
            self.assertEqual(snapshots, historical)
            for expected, actual in zip(original, second.parameters()):
                self.assertTrue(torch.equal(expected, actual.detach()))
            original_state = optimizer.state_dict()["state"]
            resumed_state = second_optimizer.state_dict()["state"]
            self.assertEqual(set(original_state), set(resumed_state))
            for key in original_state:
                self.assertEqual(set(original_state[key]), set(resumed_state[key]))
                for name, value in original_state[key].items():
                    self.assertTrue(torch.equal(value, resumed_state[key][name]))
            self.assertEqual(
                optimizer.state_dict()["param_groups"],
                second_optimizer.state_dict()["param_groups"],
            )

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
