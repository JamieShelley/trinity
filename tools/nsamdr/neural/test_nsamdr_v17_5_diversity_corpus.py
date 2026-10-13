"""CPU-only V17.5 frozen diversity cohort provenance and selection tests."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.nsamdr.neural.v17 import diversity_corpus as d

class DiversityCorpusTests(unittest.TestCase):
    def test_fixed_contract(self):
        self.assertEqual((d.CONTROL_N,d.EXPANDED_N,d.ADDITIONAL_N),(12,64,52))
        self.assertEqual(d.SELECTION_SEED,17301)
        self.assertEqual(d.MIN_TARGET_RESIDUAL,.01)

    def test_canonical_logical_and_empty(self):
        self.assertEqual(d.canonical_logical(r" RES:\MODEL\SHIP\ABC "), "res:/model/ship/abc")
        with self.assertRaises(ValueError): d.canonical_logical("")

    def test_ship_only_stratum(self):
        self.assertEqual(d.ship_stratum("res:/dx9/MODEL/SHIP/caldari/battleship/c1_x_d.dds"),"caldari/battleship")
        with self.assertRaisesRegex(ValueError,"not-ship"):
            d.ship_stratum("res:/dx9/model/structure/station_d.dds")

    def test_crop_and_authority_require_ids(self):
        self.assertEqual(d.authority_id({"family_id":"abc"}),"abc")
        self.assertEqual(d.crop_id({"crop_id":"abc_000"}),"abc_000")
        with self.assertRaises(ValueError): d.crop_id({"crop_id":"abc_001"})
        with self.assertRaises(ValueError): d.authority_id({})

    def test_sha256_json_canonical(self):
        self.assertEqual(d.sha256_json({"b":2,"a":1}),d.sha256_json({"a":1,"b":2}))
        self.assertNotEqual(d.sha256_json({"a":1}),d.sha256_json({"a":2}))

    def test_sha256_stream_is_content_based(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"input"
            path.write_bytes(b"abcd"*300000)
            self.assertEqual(d.sha256_file(path),hashlib.sha256(path.read_bytes()).hexdigest())

    def test_duplicate_sha_and_reused_path_rejected(self):
        x={"albedo":{"logical":"a","sha256":"1","path":"a"},
           "normal":{"logical":"n","sha256":"2","path":"n"}}
        self.assertTrue(d._collision(x,[copy.deepcopy(x)]))
        y=copy.deepcopy(x)
        y["albedo"]["logical"]="b"
        y["normal"]["logical"]="m"
        self.assertTrue(d._collision(y,[x]))
        y["albedo"]["sha256"]="3"
        y["normal"]["sha256"]="4"
        self.assertTrue(d._collision(y,[x]))
        y["albedo"]["path"]="new-a"
        y["normal"]["path"]="new-n"
        self.assertFalse(d._collision(y,[x]))

    def test_dds_magic_and_format_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"hash-no-extension"
            p.write_bytes(b"DDS "+bytes(144))
            meta={"logical":"res:/model/ship/a/b/foo_d.dds","path":str(p),
                  "width":1024,"height":1024}
            reader=SimpleNamespace(parse_dds_header=lambda path:(1024,1024,11,"ATI1"))
            self.assertEqual(d._source(meta,"albedo",reader)["format"],"ATI1")
            with self.assertRaisesRegex(ValueError,"wrong-format-normal"):
                d._source(meta,"normal",reader)
            p.write_bytes(b"noDDS")
            with self.assertRaisesRegex(ValueError,"not-dds"):
                d._source(meta,"albedo",reader)

    def test_native_minimum_and_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"dds"
            p.write_bytes(b"DDS "+bytes(144))
            src={"logical":"res:/model/ship/a/b/foo_d.dds","path":str(p),
                 "width":1024,"height":1024}
            reader=SimpleNamespace(parse_dds_header=lambda path:(512,1024,11,"ATI1"))
            with self.assertRaisesRegex(ValueError,"dimension-mismatch"):
                d._source(src,"albedo",reader)
            src["height"]=512
            with self.assertRaisesRegex(ValueError,"small-albedo"):
                d._source(src,"albedo",reader)

    def test_missing_protected_v2_fails_closed_before_pixels(self):
        with self.assertRaisesRegex(RuntimeError,"V2 proof"):
            d.freeze_authorities(Path("."),{}, {},{},{"schema":"wrong"})

    def test_selection_requires_frozen_partition_sizes(self):
        v2={"schema":"NSAMDR_V17_GENERALIZATION_V2_PROOF_V1"}
        with self.assertRaisesRegex(RuntimeError,"partition"):
            d.freeze_authorities(Path("."),{}, {},{"training":[],"validation":[],"final_test":[]},v2)

    def test_frozen_selection_refuses_changed_roster(self):
        with self.assertRaisesRegex(RuntimeError,"roster|counts"):
            d.verify_frozen_selection({"schema":d.SELECTION_SCHEMA,"selectionSeed":17301},
                                      {},{}, {"training":[],"validation":[],"final_test":[]})

    def test_other_seed_cannot_change_source_selection(self):
        with self.assertRaisesRegex(ValueError,"locked"):
            d.freeze_authorities(Path("."),{}, {},{}, {},selection_seed=17302)

    def _fixture(self, root: Path, candidate_count: int = 80):
        def row(fid,split):
            p=root/(fid+".npz")
            p.write_bytes(b"training-crop-only")
            return {"family_id":fid,"crop_id":fid+"_000",
                    "split":split,"path":str(p)}
        training=[row(f"control{i:03d}","train") for i in range(12)]
        dev=[row(f"dev{i:03d}","validation") for i in range(4)]
        sealed=[row(f"sealed{i:03d}","validation") for i in range(8)]
        extra=[row(f"cand{i:03d}","train") for i in range(candidate_count)]
        manifest={"crops":training+dev+sealed+extra,
                  "families":[{"familyId":r["family_id"]}
                              for r in training+dev+sealed+extra]}
        for path in ("manifest","v2","v3"):
            (root/path).write_text("origin-"+path)
        v2={"schema":"NSAMDR_V17_GENERALIZATION_V2_PROOF_V1",
            "trainingCropIds":["cand000_000"],
            "validationCropIds":[],"finalTestCropIds":[]}
        source={"manifest":str(root/"manifest"),"v2Report":str(root/"v2"),
                "v3Report":str(root/"v3"),
                "v3ReportSha256":d.sha256_file(root/"v3")}
        parts={"training":training,"validation":dev,"final_test":sealed}
        return source,manifest,parts,v2

    def test_freeze_is_deterministic_and_never_selects_protected_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,manifest,parts,v2=self._fixture(root)
            def fake_pair(family,_reader):
                fid=family["familyId"]
                return {"albedo":{"logical":fid+"-a","path":fid+"-a",
                                  "sha256":fid+"-a"},
                        "normal":{"logical":fid+"-n","path":fid+"-n",
                                  "sha256":fid+"-n"},"stratum":"ship/class"}
            with patch.object(d,"_pair",side_effect=fake_pair):
                selection,audit=d.freeze_authorities(
                    root,source,manifest,parts,v2,suitability=lambda row:.02)
                second,_=d.freeze_authorities(
                    root,source,manifest,parts,v2,suitability=lambda row:.02)
            self.assertEqual(selection,second)
            self.assertEqual(selection["expandedTrainingCropIds"][:12],
                             [x["crop_id"] for x in parts["training"]])
            self.assertEqual(len(selection["expandedTrainingCropIds"]),64)
            self.assertEqual(len(set(selection["expandedTrainingCropIds"])),64)
            self.assertNotIn("cand000_000",selection["expandedTrainingCropIds"])
            self.assertEqual(selection["developmentCropIds"],
                             [x["crop_id"] for x in parts["validation"]])
            self.assertEqual(selection["sealedFinalCropIds"],
                             [x["crop_id"] for x in parts["final_test"]])
            self.assertEqual(audit["developmentPixelsOpened"],0)
            self.assertEqual(audit["sealedPixelsOpened"],0)
            self.assertEqual(audit["selectedAdditional"],52)

    def test_too_small_eligible_corpus_blocks_without_reducing_b64(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source,manifest,parts,v2=self._fixture(root,candidate_count=20)
            def fake_pair(family,_reader):
                fid=family["familyId"]
                return {"albedo":{"logical":fid+"-a","path":fid+"-a","sha256":fid+"-a"},
                        "normal":{"logical":fid+"-n","path":fid+"-n","sha256":fid+"-n"},
                        "stratum":"ship/class"}
            with patch.object(d,"_pair",side_effect=fake_pair):
                with self.assertRaisesRegex(RuntimeError,
                                            "BLOCKED_INSUFFICIENT_ELIGIBLE_TRAIN_AUTHORITIES"):
                    d.freeze_authorities(root,source,manifest,parts,v2,
                                         suitability=lambda row:.02)

    def test_unsupported_arm_rejected(self):
        with self.assertRaisesRegex(ValueError,"arm"):
            d.selected_rows({}, {}, "unsafe")

if __name__=="__main__":
    unittest.main()
