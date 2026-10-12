"""Sealed V17.2 V2 split, checkpoint and validation-selection contracts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from tools.nsamdr.nsamdr_cli import _app
from tools.nsamdr.neural.probe_nsamdr_v17_generalization_v2 import (
    CHECKPOINT_SCHEMA, PREVIOUS_VALIDATION_AUTHORITIES, SCHEMA,
    _ids, _promote_checkpoint, _resolve_resume, _restore, _save, _split_pairs, _validation_score,
)


def _manifest() -> dict[str, object]:
    authorities = [*PREVIOUS_VALIDATION_AUTHORITIES, *[f"extra_{i:02d}" for i in range(16)]]
    return {
        "crops": [
            {
                "split": "train", "family_id": authority,
                "crop_id": f"{authority}_{number:03d}",
                "path": f"/data/{authority}_{number}.npz",
            }
            for authority in authorities
            for number in (0, 1)
        ]
    }


class TinyModel(torch.nn.Linear):
    def architecture_contract(self) -> dict[str, str]:
        return {"revision": "tiny-v2"}


class V17GeneralizationV2Tests(unittest.TestCase):
    def test_partition_keeps_previously_examined_siblings_in_validation(self) -> None:
        splits = _split_pairs(
            _manifest(), anchor=PREVIOUS_VALIDATION_AUTHORITIES[0], authorities=12,
        )
        self.assertEqual([len(splits[key]) for key in ("training", "validation", "final_test")],
                         [12, 4, 8])
        self.assertEqual(
            _ids(splits["validation"]),
            [f"{x}_001" for x in PREVIOUS_VALIDATION_AUTHORITIES],
        )
        self.assertEqual(
            [x.endswith("_001") for x in _ids(splits["training"])], [False] * 12,
        )
        self.assertTrue(set(_ids(splits["final_test"])).isdisjoint(
            set(_ids(splits["validation"]))
        ))
        self.assertTrue(set(_ids(splits["final_test"])).isdisjoint(
            set(_ids(splits["training"]))
        ))
        self.assertTrue(
            all(not any(id.startswith(auth) for auth in PREVIOUS_VALIDATION_AUTHORITIES)
                for id in _ids(splits["final_test"]))
        )

    def test_training_only_residual_filter_replaces_bad_authority(self) -> None:
        screened: list[str] = []

        def eligible(record: dict[str, object]) -> bool:
            crop_id = str(record["crop_id"])
            screened.append(crop_id)
            self.assertTrue(crop_id.endswith("_000"))
            return crop_id != "extra_00_000"

        splits = _split_pairs(
            _manifest(), anchor=PREVIOUS_VALIDATION_AUTHORITIES[0],
            authorities=12, eligible_training=eligible,
        )
        training_ids = _ids(splits["training"])
        self.assertEqual(len(training_ids), 12)
        self.assertNotIn("extra_00_000", training_ids)
        self.assertIn("extra_08_000", training_ids)
        self.assertEqual(
            _ids(splits["validation"]),
            [f"{x}_001" for x in PREVIOUS_VALIDATION_AUTHORITIES],
        )
        self.assertEqual(len(splits["final_test"]), 8)
        self.assertNotIn("extra_00_001", _ids(splits["final_test"]))
        self.assertNotIn("extra_00_001", screened)
        self.assertTrue(all(id.endswith("_000") for id in screened))

    def test_training_filter_refuses_to_skip_development_anchor(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "required V1 development"):
            _split_pairs(
                _manifest(), anchor=PREVIOUS_VALIDATION_AUTHORITIES[0],
                authorities=12,
                eligible_training=lambda record: (
                    record["family_id"] != PREVIOUS_VALIDATION_AUTHORITIES[0]
                ),
            )

    def test_training_filter_reports_inadequate_available_corpus(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "eligible paired training authorities"):
            _split_pairs(
                _manifest(), anchor=PREVIOUS_VALIDATION_AUTHORITIES[0],
                authorities=12,
                eligible_training=lambda record: (
                    record["family_id"] in PREVIOUS_VALIDATION_AUTHORITIES
                ),
            )

    def test_partition_rejects_insufficient_corpus_or_duplicate_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 12"):
            _split_pairs(_manifest(), anchor=PREVIOUS_VALIDATION_AUTHORITIES[0], authorities=4)
        manifest = _manifest()
        manifest["crops"] = [
            x for x in manifest["crops"] if x["family_id"] != PREVIOUS_VALIDATION_AUTHORITIES[2]
        ]
        with self.assertRaisesRegex(RuntimeError, "development"):
            _split_pairs(manifest, anchor=PREVIOUS_VALIDATION_AUTHORITIES[0], authorities=12)
        manifest = _manifest()
        for row in manifest["crops"]:
            if row["crop_id"] == f"{PREVIOUS_VALIDATION_AUTHORITIES[0]}_001":
                row["path"] = "/data/extra_00_1.npz"
        with self.assertRaisesRegex(RuntimeError, "identities overlap"):
            _split_pairs(manifest, anchor=PREVIOUS_VALIDATION_AUTHORITIES[0], authorities=12)

    def test_validation_objective_prefers_edges_not_test_data(self) -> None:
        metrics = {
            "global_recovery": 0.3, "edge_recovery": 0.2,
            "gradient_recovery": 0.25, "detail_recovery_1px": 0.1,
            "lattice_cell_excess": 0.1,
        }
        first = _validation_score({"mean": metrics})
        better = dict(metrics, edge_recovery=0.3)
        self.assertAlmostEqual(_validation_score({"mean": better}) - first, 0.2)
        bad_lattice = dict(metrics, lattice_cell_excess=0.2)
        self.assertAlmostEqual(first - _validation_score({"mean": bad_lattice}), 0.1)

    def test_qualification_overrides_higher_nonqualified_score(self) -> None:
        self.assertTrue(_promote_checkpoint(
            qualified=True, previous_qualified=False, score=0.8, best_score=0.9,
            min_improvement=0.002, no_best=False,
        ))
        self.assertFalse(_promote_checkpoint(
            qualified=False, previous_qualified=True, score=2.0, best_score=1.0,
            min_improvement=0.002, no_best=False,
        ))
        self.assertFalse(_promote_checkpoint(
            qualified=False, previous_qualified=False, score=0.9001, best_score=0.9,
            min_improvement=0.002, no_best=False,
        ))

    def test_v1_checkpoints_cannot_be_loaded_as_v2(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            model = TinyModel(2, 2)
            opt = torch.optim.Adam(model.parameters(), lr=0.0002)
            partitions = _split_pairs(
                _manifest(), anchor=PREVIOUS_VALIDATION_AUTHORITIES[0], authorities=12,
            )
            checkpoint = root / "checkpoint.pt"
            torch.save({"schema": "NSAMDR_V17_GENERALIZATION_CHECKPOINT_V1"}, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "incompatible V2 checkpoint"):
                _restore(
                    checkpoint, model=model, optimizer=opt, manifest=manifest,
                    partitions=partitions, seed=17001, learning_rate=0.0002,
                    device=torch.device("cpu"),
                )

    def test_checkpoint_roundtrip_preserves_optimizer_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            partitions = _split_pairs(
                _manifest(), anchor=PREVIOUS_VALIDATION_AUTHORITIES[0], authorities=12,
            )
            model = TinyModel(2, 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
            loss = model(torch.ones(1, 2)).square().sum()
            loss.backward()
            optimizer.step()
            checkpoint = root / "stage_006144/checkpoint.pt"
            history = [{"update": 6144, "validationScore": 0.3, "newBest": True}]
            _save(
                checkpoint, model, optimizer, manifest=manifest, partitions=partitions,
                seed=17001, learning_rate=0.0002, update=6144, history=history,
                best_update=6144, best_score=0.3, stale=0, best_qualified=False,
            )
            clone = TinyModel(2, 2)
            clone_optimizer = torch.optim.Adam(clone.parameters(), lr=0.0002)
            payload = _restore(
                checkpoint, model=clone, optimizer=clone_optimizer,
                manifest=manifest, partitions=partitions, seed=17001,
                learning_rate=0.0002, device=torch.device("cpu"),
            )
            self.assertEqual(payload["schema"], CHECKPOINT_SCHEMA)
            self.assertEqual(payload["history"], history)
            self.assertEqual(payload["bestUpdate"], 6144)
            for before, after in zip(model.parameters(), clone.parameters()):
                self.assertTrue(torch.equal(before, after))
            orig_state = optimizer.state_dict()["state"]
            new_state = clone_optimizer.state_dict()["state"]
            self.assertEqual(set(orig_state), set(new_state))
            for param in orig_state:
                for name in orig_state[param]:
                    self.assertTrue(torch.equal(orig_state[param][name], new_state[param][name]))

    def test_auto_resume_ignores_completed_v2_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proof = root / "artifacts/nsamdr/diagnostics/v17_generalization_v2"
            finished, pending = proof / "proof_01", proof / "proof_02"
            for folder in (finished, pending):
                checkpoint = folder / "stage_006144/checkpoint.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"checkpoint")
            (finished / "report.json").write_text(json.dumps({"schema": SCHEMA}))
            self.assertEqual(
                _resolve_resume(root, "auto"),
                pending / "stage_006144/checkpoint.pt",
            )
            self.assertIsNone(_resolve_resume(root, "fresh"))

    def test_managed_cli_exposes_v2_defaults(self) -> None:
        options = _app.build_parser().parse_args(["v17-generalization-v2"])
        self.assertEqual(options.authorities, 12)
        self.assertEqual(options.stages, ["6144,12288,18432,24576,30720"])
        self.assertEqual(options.resume, "fresh")
        self.assertEqual(options.patience, 2)


if __name__ == "__main__":
    unittest.main()
