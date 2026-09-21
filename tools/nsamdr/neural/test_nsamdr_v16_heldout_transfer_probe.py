import tempfile
import unittest
from pathlib import Path

from tools.nsamdr.neural.probe_nsamdr_v16_heldout_transfer import (
    INTERFERENCE_CHECKPOINT_SCHEMA,
    _median_deltas,
    _validate_candidate_payload,
    _validation_authority_count,
)


class HeldoutTransferProbeTests(unittest.TestCase):
    def test_validation_authority_count_deduplicates(self) -> None:
        manifest = {
            "crops": [
                {"split": "validation", "family_id": "a"},
                {"split": "validation", "family_id": "a"},
                {"split": "validation", "family_id": "b"},
                {"split": "train", "family_id": "c"},
            ]
        }
        self.assertEqual(_validation_authority_count(manifest), 2)

    def test_median_deltas_use_candidate_minus_source(self) -> None:
        source = {
            "median_global_recovery": 0.10,
            "median_edge_recovery": 0.20,
            "median_gradient_recovery": 0.30,
            "median_normal_recovery": 0.40,
            "median_lattice_cell_excess": 0.20,
        }
        candidate = {
            "median_global_recovery": 0.15,
            "median_edge_recovery": 0.28,
            "median_gradient_recovery": 0.31,
            "median_normal_recovery": 0.45,
            "median_lattice_cell_excess": 0.12,
        }
        delta = _median_deltas(source, candidate)
        self.assertAlmostEqual(delta["median_global_recovery"], 0.05)
        self.assertAlmostEqual(delta["median_edge_recovery"], 0.08)
        self.assertAlmostEqual(delta["median_lattice_cell_excess"], -0.08)

    def test_candidate_payload_must_match_source_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            source = root_path / "source.pt"
            manifest = root_path / "manifest.json"
            source.touch()
            manifest.touch()
            payload = {
                "schema": INTERFERENCE_CHECKPOINT_SCHEMA,
                "sourceCheckpoint": str(source.resolve()),
                "sourceCheckpointStep": 596,
                "manifest": str(manifest.resolve()),
                "modelState": {"weight": object()},
            }
            _validate_candidate_payload(
                payload,
                source_checkpoint=source,
                source_step=596,
                manifest_path=manifest,
            )
            payload["sourceCheckpointStep"] = 595
            with self.assertRaisesRegex(RuntimeError, "source step mismatch"):
                _validate_candidate_payload(
                    payload,
                    source_checkpoint=source,
                    source_step=596,
                    manifest_path=manifest,
                )


if __name__ == "__main__":
    unittest.main()
