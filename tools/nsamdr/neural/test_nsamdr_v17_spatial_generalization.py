"""CPU-only V17.3 spatial sampling, split sealing, checkpoint and CLI contracts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from tools.nsamdr.nsamdr_cli import _app
from tools.nsamdr.neural.probe_nsamdr_v17_spatial_generalization import (
    CHECKPOINT_SCHEMA, SCHEMA, _contract, _restore, _run_dir, _save,
)
from tools.nsamdr.neural.v17.spatial_sampling import (
    NativeAuthoritySampler, NativeFamily, _aligned_crop,
    _detail_integral, _fixed_origin, _id, _partitions,
)


def _manifest():
    families = []
    crops = []
    for split, count, prefix in (("train", 24, "a"), ("validation", 16, "b")):
        for i in range(count):
            authority = f"{prefix}{i:015x}"
            families.append({
                "familyId": authority,
                "albedo": {"width": 1024, "height": 1024, "path": f"/x/{authority}_a.dds"},
                "normal": {"width": 1024, "height": 1024, "path": f"/x/{authority}_n.dds"},
                "material": None,
            })
            for c in (0, 1):
                crops.append({
                    "family_id": authority, "split": split,
                    "crop_id": f"{authority}_{c:03d}",
                    "path": f"/x/{authority}_{c:03d}.npz",
                    "source_box": [0, 0, 500000, 500000],
                })
    return {"families": families, "crops": crops}


def _v2():
    return {
        "schema": "NSAMDR_V17_GENERALIZATION_V2_PROOF_V1",
        "trainingCropIds": ["a000000000000000_000", "a000000000000001_000"],
        "validationCropIds": ["a000000000000000_001"],
        "finalTestCropIds": ["a000000000000001_001"],
    }


class TinyModel(torch.nn.Linear):
    def architecture_contract(self):
        return {"revision": "tiny-V17.3"}


class SpatialGeneralizationTests(unittest.TestCase):
    def test_authority_split_is_disjoint_and_excludes_all_v2_sources(self):
        partitions = _partitions(_manifest(), _v2())
        self.assertEqual([len(partitions[k]) for k in
                          ("training", "validation", "final_test")], [12, 4, 8])
        ids = {k: {_id(record) for record in rows} for k, rows in partitions.items()}
        self.assertFalse(ids["training"] & ids["validation"])
        self.assertFalse(ids["training"] & ids["final_test"])
        self.assertFalse(ids["validation"] & ids["final_test"])
        self.assertNotIn("a000000000000000", ids["training"])
        self.assertNotIn("a000000000000001", ids["training"])
        self.assertTrue(all(x.startswith("b") for x in ids["validation"]))
        self.assertTrue(all(x.startswith("b") for x in ids["final_test"]))
        self.assertTrue(all(row["crop_id"].endswith("_000")
                            for rows in partitions.values() for row in rows))

    def test_only_training_candidates_are_screened_not_final_test(self):
        checked = []
        def eligible(record):
            checked.append(record["crop_id"])
            return record["family_id"] != "a000000000000002"
        subsets = _partitions(_manifest(), _v2(), eligible_training=eligible)
        self.assertNotIn("a000000000000002", {_id(x) for x in subsets["training"]})
        self.assertIn("a00000000000000e", {_id(x) for x in subsets["training"]})
        self.assertTrue(all(value.startswith("a") and value.endswith("_000")
                            for value in checked))

    def test_missing_partition_or_wrong_v2_schema_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "authoritative V2"):
            _partitions(_manifest(), {"schema": "unknown"})
        sparse = _manifest()
        sparse["crops"] = [x for x in sparse["crops"]
                          if x["split"] == "train"]
        with self.assertRaisesRegex(RuntimeError, "insufficient independent"):
            _partitions(sparse, _v2())

    def test_fixed_source_origin_and_aligned_crop(self):
        record = {"source_box": [500000, 500000, 562500, 562500]}
        self.assertEqual(_fixed_origin(record, (1024, 1024), 512), (288, 288))
        image = np.arange(32*32*3, dtype=np.uint32).reshape(32, 32, 3)
        tile = _aligned_crop(image, 8, 8, 16, (32, 32), nearest=False)
        self.assertEqual(tile.shape, (16, 16, 3))
        np.testing.assert_array_equal(tile, image[8:24, 8:24])

    def test_spatial_sampler_alternates_uniform_detail_and_changes_regions(self):
        sampler = NativeAuthoritySampler.__new__(NativeAuthoritySampler)
        sampler.hr_size, sampler.seed = 512, 17301
        authority = "a00000000000000a"
        rng = np.random.default_rng(34)
        a = rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
        n = np.full((1024, 1024, 2), 127, dtype=np.uint8)
        integral = _detail_integral(a, n)
        sampler.families = {authority: NativeFamily(
            authority, a, n, None, (128, 128), integral,
            "red-green-fallback", False,
        )}
        fixed, fixedmeta = sampler.sample(authority, 0, mode="fixed")
        self.assertEqual(fixedmeta["origin"], [128, 128])
        self.assertEqual(fixed["lr_albedo"].shape, (1, 3, 128, 128))
        self.assertEqual(fixed["target_albedo"].shape, (1, 3, 512, 512))
        b0, first = sampler.sample(authority, 0, mode="spatial")
        b1, second = sampler.sample(authority, 1, mode="spatial")
        self.assertEqual(first["samplingPolicy"], "uniform")
        self.assertEqual(second["samplingPolicy"], "detail")
        self.assertNotEqual(first["origin"], second["origin"])
        self.assertTrue(torch.isfinite(b0["target_normal"]).all())
        self.assertTrue(torch.isfinite(b1["target_normal"]).all())
        fixed_again, _ = sampler.sample(authority, 8, mode="fixed")
        self.assertTrue(torch.equal(fixed["target_albedo"],
                                    fixed_again["target_albedo"]))

    def test_checkpoint_roundtrip_and_wrong_arm_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixed/stage_000008/checkpoint.pt"
            model = TinyModel(2, 2)
            opt = torch.optim.Adam(model.parameters(), lr=2e-4)
            loss = model(torch.ones(1, 2)).square().sum()
            loss.backward()
            opt.step()
            contract = {"schema": SCHEMA, "seed": 17301}
            _save(path, model, opt, contract, "fixed", update=8,
                  history=[{"update": 8}], best_update=8,
                  best_score=1.0, stale=0, qualified=False)
            clone = TinyModel(2, 2)
            op2 = torch.optim.Adam(clone.parameters(), lr=2e-4)
            restored = _restore(path, clone, op2, contract, "fixed", torch.device("cpu"))
            self.assertEqual(restored["schema"], CHECKPOINT_SCHEMA)
            self.assertEqual(restored["update"], 8)
            for p, q in zip(model.parameters(), clone.parameters()):
                self.assertTrue(torch.equal(p, q))
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                _restore(path, clone, op2, contract, "spatial", torch.device("cpu"))

    def test_auto_resume_rejects_changed_manifest_or_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = {"schema": SCHEMA, "seed": 17301}
            folder = _run_dir(root, a, "fresh")
            self.assertEqual(_run_dir(root, a, "auto"), folder)
            with self.assertRaisesRegex(RuntimeError, "contract differs"):
                _run_dir(root, {"schema": SCHEMA, "seed": 12}, "auto")
            (folder / "report.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "no incomplete"):
                _run_dir(root, a, "auto")

    def test_contract_and_cli_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, report = root / "manifest.json", root / "v2.json"
            manifest.write_text("{}")
            report.write_text('{"schema":"V2"}')
            args = SimpleNamespace(
                arm="both", seed=17301, stages=["12288,24576,36864,49152"],
                learning_rate=0.0002, minimum_target_residual=0.01,
                patience=2, min_improvement=0.002,
                encoder_channels=96, mid_channels=64,
                detail_channels=48, decoder_blocks=4,
            )
            parts = _partitions(_manifest(), _v2())
            c = _contract(args, manifest, report, parts)
            self.assertEqual(c["arms"], ["fixed", "spatial"])
            self.assertEqual(c["stages"][-1], 49152)
            self.assertEqual(c["partitionCropIds"]["final_test"][0][-4:], "_000")
        parsed = _app.build_parser().parse_args(["v17-spatial-generalization"])
        self.assertEqual(parsed.arm, "both")
        self.assertEqual(parsed.authorities, 12)
        self.assertEqual(parsed.resume, "fresh")
        self.assertEqual(parsed.stages, ["12288,24576,36864,49152"])


if __name__ == "__main__":
    unittest.main()
