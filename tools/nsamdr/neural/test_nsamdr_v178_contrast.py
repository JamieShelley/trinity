"""The contrast diagnostic keeps fixed cases and authenticated provenance."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import torch

from tools.nsamdr.neural.benchmark_nsamdr_v178_contrast import (
    CASE_GROUPS, ROOT, authenticate, native_sources, parser, registered_batch,
)
from tools.nsamdr.neural.train_nsamdr_v178_contour import TRAIN_SCHEMA, source_hash
from tools.nsamdr.neural.v177.synthetic_geometry import make_batch
from tools.nsamdr.neural.v178.contour_model import SCHEMA, physical_edge_evidence
from tools.nsamdr.neural.v178.golden_patch import sha256


class FixedContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(2,torch.get_num_threads()))

    def test_registered_cases_preserve_geometry_and_are_independent_of_global_rng(self):
        seed=CASE_GROUPS["reported_failure"][0]
        images=make_batch(1,32,seed=seed)
        first=registered_batch(seed,torch.device("cpu"))
        torch.manual_seed(9183)
        repeated=registered_batch(seed,torch.device("cpu"))
        for before,after in zip(first,repeated):
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        for actual,expected in ((first[0],images[0]),(first[1],images[1]),
                                (first[3],images[5][:,:1])):
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        a,n,m,target=first
        original=[v.clone() for v in first]
        inverted=1.-a
        self.assertLess(float((physical_edge_evidence(inverted,n)-m[:,:1]).abs().max()),1.e-6)
        for before,after in zip(original,first):
            torch.testing.assert_close(before,after,rtol=0,atol=0)

    def test_primary_case_registration_is_exact_and_holdouts_do_not_overlap(self):
        self.assertEqual(CASE_GROUPS["reported_failure"],
                         tuple(17822+1000000+1723*i for i in range(12)))
        groups=[set(group) for group in CASE_GROUPS.values()]
        self.assertEqual(len(set.union(*groups)),36)
        self.assertGreaterEqual(min(groups[1]),2000000)
        self.assertGreaterEqual(min(groups[2]),3000000)

    def test_authentication_rejects_changed_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            checkpoint=folder/"candidate.pt"
            config={"seed":17832,"sourceHash":source_hash(ROOT)}
            torch.save({"schema":SCHEMA,"config":config,"steps":1,
                        "syntheticPass":False},checkpoint)
            report={"schema":TRAIN_SCHEMA,"modelSchema":SCHEMA,
                    "sourceHash":config["sourceHash"],"config":config,
                    "checkpointSha256":sha256(checkpoint),
                    "decision":"G2_SYNTHETIC_FAIL"}
            (folder/"report.json").write_text(json.dumps(report),encoding="utf-8")
            state,_=authenticate(checkpoint,ROOT)
            self.assertEqual(state["steps"],1)
            with checkpoint.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(RuntimeError,"authenticated"):
                authenticate(checkpoint,ROOT)

    def test_native_source_hash_is_checked_and_multiple_folders_are_accepted(self):
        args=parser().parse_args(["--checkpoint","candidate.pt","--output","new",
                                 "--native-folder","golden","--native-folder","unseen"])
        self.assertEqual(args.native_folder,[Path("golden"),Path("unseen")])
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"authored.png"
            source.write_bytes(b"registered native file")
            evidence={"assets":{"native_albedo.png":{
                "source":str(source),"sourceSha256":sha256(source)}}}
            self.assertEqual(native_sources(evidence)["native_albedo.png"]["sha256"],sha256(source))
            source.write_bytes(b"mutated native file")
            with self.assertRaisesRegex(RuntimeError,"source map changed"):
                native_sources(evidence)


if __name__=="__main__":
    unittest.main()
