from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from tools.nsamdr.neural.probe_nsamdr_v16_interference import (
    _candidate_authority_order,
    _evaluate_all,
    _train_authority_ids,
)


class InterferenceProbeTests(unittest.TestCase):
    def test_fixed_train_authority_is_not_reported_as_held_out(self) -> None:
        evaluated = {
            "lossTerms": {"total": 0.1},
            "metrics": {
                "global_recovery": 0.5,
                "edge_recovery": 0.6,
                "gradient_recovery": 0.4,
                "normal_recovery": 0.2,
                "lattice_cell_excess": 0.1,
                "protected_preservation": 0.8,
                "heldout_sample": 1.0,
            },
            "residualDiagnostics": {
                "candidate_to_target_residual_ratio": 0.8,
                "residual_cosine_similarity": 0.7,
                "target_weighted_sign_agreement": 0.9,
                "least_squares_residual_gain": 1.0,
            },
        }
        config = SimpleNamespace(
            candidate_global_recovery_required=0.45,
            candidate_edge_recovery_required=0.60,
            candidate_gradient_recovery_required=0.35,
            candidate_lattice_cell_excess_max=0.15,
        )
        selected = [
            {
                "batch": {},
                "sample": {
                    "authorityId": "train-authority",
                    "cropId": "train-authority_000",
                },
            }
        ]
        with patch(
            "tools.nsamdr.neural.probe_nsamdr_v16_interference._evaluate_exact",
            return_value=evaluated,
        ):
            result = _evaluate_all(
                object(),
                selected,
                config,
                device=torch.device("cpu"),
                precision="auto",
            )
        row = result["perAuthority"][0]
        self.assertEqual(row["metrics"]["heldout_sample"], 0.0)
        self.assertEqual(
            row["evaluationRole"],
            "train-authority-fixed-fit-diagnostic",
        )

    def test_train_authority_ids_filters_and_sorts(self) -> None:
        manifest = {
            "crops": [
                {"family_id": "b", "split": "train"},
                {"family_id": "a", "split": "train"},
                {"family_id": "a", "split": "train"},
                {"family_id": "z", "split": "validation"},
            ]
        }
        self.assertEqual(_train_authority_ids(manifest), ["a", "b"])

    def test_candidate_order_keeps_anchor_first_and_is_deterministic(self) -> None:
        manifest = {
            "crops": [
                {"family_id": value, "split": "train"}
                for value in ("a", "b", "c", "d", "e")
            ]
        }
        first = _candidate_authority_order(
            manifest,
            anchor_authority="c",
            seed=123,
        )
        second = _candidate_authority_order(
            manifest,
            anchor_authority="c",
            seed=123,
        )
        self.assertEqual(first, second)
        self.assertEqual(first[0], "c")
        self.assertEqual(sorted(first), ["a", "b", "c", "d", "e"])

    def test_candidate_order_rejects_missing_anchor(self) -> None:
        manifest = {"crops": [{"family_id": "a", "split": "train"}]}
        with self.assertRaises(RuntimeError):
            _candidate_authority_order(
                manifest,
                anchor_authority="missing",
                seed=123,
            )


if __name__ == "__main__":
    unittest.main()
