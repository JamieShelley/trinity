"""V17.4 A-F unified proof contract tests (no GPU, no sealed final-test data)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from tools.nsamdr.neural import probe_nsamdr_v17_comprehensive as proof
from tools.nsamdr.neural.v17.spatial_sampling import NativeFamily, _detail_integral
from tools.nsamdr.nsamdr_cli import NSAMDRCommandLineApplication


class TinyModel(torch.nn.Linear):
    def architecture_contract(self):
        return {"revision": "af-toy"}


class AFProofTests(unittest.TestCase):
    def test_schema_and_cumulative_schedule(self):
        self.assertEqual(proof.STAGES, (61440,73728,86016))
        self.assertEqual(proof.INITIAL_UPDATE,49152)
        self.assertEqual(proof.FREQ_WEIGHTS["detailBand3"],0.10)
        args=proof.parser().parse_args([])
        self.assertEqual(args.resume,"fresh")
        self.assertFalse(args.audit_only)

    def test_origin_contract_requires_real_sealed_v3_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            manifest=root/"manifest.json"
            v2=root/"v2.json"
            pt=root/"checkpoint.pt"
            manifest.write_text("{}")
            v2.write_text('{"schema":"NSAMDR_V17_GENERALIZATION_V2_PROOF_V1"}')
            pt.write_bytes(b"checkpoint bytes for fingerprint only")
            contract={
                "manifest":str(manifest),"manifestSha256":proof._sha(manifest),
                "v2Report":str(v2),"v2ReportSha256":proof._sha(v2),
                "learningRate":0.0002,"seed":17301,
                "architecture":{
                    "encoderChannels":96,"decoderMidChannels":64,
                    "decoderDetailChannels":48,"decoderBlocksPerStage":4,
                },
                "partitionCropIds":{
                    "training":[f"a{i:015x}_000" for i in range(12)],
                    "validation":[f"b{i:015x}_000" for i in range(4)],
                    "final_test":[f"c{i:015x}_000" for i in range(8)],
                },
            }
            value={
                "schema":proof.V3_SCHEMA,"contract":contract,
                "decision":"reject-independent-validation",
                "finalTestEvaluated":False,"finalTestEvaluationCount":0,
                "armResults":{"spatial":{
                    "selectedCheckpoint":str(pt),
                    "bestValidationUpdate":49152,"bestValidationScore":1.23,
                }},
            }
            p=root/"v3.json"
            p.write_text(json.dumps(value))
            src=proof._origin(root,str(p))
            self.assertEqual(src["checkpointSha256"],hashlib.sha256(pt.read_bytes()).hexdigest())
            self.assertEqual(len(src["splits"]["final_test"]),8)
            value["finalTestEvaluated"]=True
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError,"sealed"):
                proof._origin(root,str(p))

    def test_new_proof_requires_frozen_authority_split_and_v2_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            manifest=root/"m.json"
            v2=root/"v2.json"
            v2.write_text(json.dumps({
                "schema":"NSAMDR_V17_GENERALIZATION_V2_PROOF_V1",
                "trainingCropIds":["a000000000000000_000"],
                "validationCropIds":[],"finalTestCropIds":[],
            }))
            roles={"training":[],"validation":[],"final_test":[]}
            for role,prefix,count,split in (
                ("training","a",12,"train"),
                ("validation","b",4,"validation"),
                ("final_test","c",8,"validation"),
            ):
                for i in range(count):
                    roles[role].append({
                        "crop_id":f"{prefix}{i:015x}_000",
                        "family_id":f"{prefix}{i:015x}",
                        "split":split,"path":f"/dummy/{i}.npz",
                    })
            manifest.write_text(json.dumps({"crops":[row for group in roles.values() for row in group]}))
            source={"manifest":str(manifest),"v2Report":str(v2),
                    "splits":{k:[r["crop_id"] for r in v] for k,v in roles.items()}}
            with self.assertRaisesRegex(RuntimeError,"protected"):
                proof._records(source)
            source["splits"]["training"][0]="a000000000000001_000"
            # A duplicate crop ID now fails explicit split/crop uniqueness.
            with self.assertRaises(RuntimeError):
                proof._records(source)

    def test_audit_b_reports_sampling_difference_without_mutating_sampler(self):
        sampler=SimpleNamespace(seed=17301)
        authority="abcde123456789ab"
        rng=np.random.default_rng(7)
        albedo=rng.integers(0,255,(768,768,3),dtype=np.uint8)
        normal=np.full((768,768,2),127,dtype=np.uint8)
        original=_detail_integral(albedo,normal)
        sampler.families={authority:NativeFamily(
            authority,albedo,normal,None,(0,0),original,
            "red-green-fallback",False,
        )}
        old=original.copy()
        result=proof._audit_b(sampler,seed=17301)
        self.assertIn(result["status"],("PASS","WARN"))
        self.assertFalse(result["samplingChangedInTraining"])
        self.assertEqual(result["trainingAuthoritiesAudited"],1)
        np.testing.assert_array_equal(sampler.families[authority].detail_integral,old)

    def test_audit_a_marks_uncertain_alpha_as_warning(self):
        sampler=SimpleNamespace(audit={
            "aaaaaaaaaaaaaaaa":{
                "albedoSize":[1024,1024],"normalSize":[1024,1024],
                "normalEncoding":"red-green-fallback",
                "materialSupervised":False,
            },
        })
        parts={
            "training":[{"family_id":"aaaaaaaaaaaaaaaa"}],
            "validation":[{"family_id":"bbbbbbbbbbbbbbbb"}],
            "final_test":[{"family_id":"cccccccccccccccc"}],
        }
        manifest={"families":[{
            "familyId":prefix*16,
            "albedo":{"path":f"/{prefix}_a.dds"},
            "normal":{"path":f"/{prefix}_n.dds"},
            "material":None,
        } for prefix in ("a","b","c")]}
        result=proof._audit_a(manifest,parts,sampler)
        self.assertEqual(result["status"],"WARN")
        self.assertEqual(result["untouchedSealedFinalAuthorities"],["cccccccccccccccc"])
        self.assertEqual(result["finalTestPixelsOpened"],0)
        self.assertEqual(result["normalEncodingCounts"]["red-green-fallback"],1)

    def test_frequency_loss_zero_if_perfect_and_has_gradient(self):
        target=torch.rand(1,3,24,24)
        candidate=target.detach().clone().requires_grad_()
        batch={"target_albedo":target}
        zero=proof._frequency_loss({"candidate_albedo":candidate},batch)
        self.assertLess(zero.item(),1e-7)
        shifted=(target*0.7).detach().clone().requires_grad_()
        loss=proof._frequency_loss({"candidate_albedo":shifted},batch)
        self.assertGreater(loss.item(),0)
        loss.backward()
        self.assertIsNotNone(shifted.grad)
        self.assertGreater(float(shifted.grad.abs().sum()),0)

    def test_resume_refuses_changed_seed_or_loss_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            repo=Path(folder)
            contract={"schema":proof.SCHEMA,"source":{"checkpointSha256":"abc"},
                      "frequencyLoss":{"detailBand3":0.10}}
            folder=proof._run_dir(repo,contract,"fresh")
            self.assertEqual(proof._run_dir(repo,contract,"auto"),folder)
            with self.assertRaisesRegex(RuntimeError,"contract mismatch"):
                proof._run_dir(repo,{
                    **contract,"frequencyLoss":{"detailBand3":0.15}
                },"auto")
            (folder/"report.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError,"No incomplete"):
                proof._run_dir(repo,contract,"auto")

    def test_arm_checkpoint_restores_weights_optimizer_and_rejects_wrong_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"D_continuation/stage_061440/checkpoint.pt"
            model=TinyModel(2,2)
            opt=torch.optim.Adam(model.parameters(),lr=0.0002)
            z=model(torch.ones(1,2)).square().mean();z.backward();opt.step()
            contract={"schema":proof.SCHEMA}
            proof._save_stage(path,model,opt,contract,"D_continuation",
                              61440,[],61440,2.2,0,False)
            clone=TinyModel(2,2)
            opt2=torch.optim.Adam(clone.parameters(),lr=0.0002)
            result=proof._load_stage(path,clone,opt2,contract,"D_continuation",
                                    torch.device("cpu"))
            self.assertEqual(result["update"],61440)
            for a,b in zip(model.parameters(),clone.parameters()):
                self.assertTrue(torch.equal(a,b))
            with self.assertRaisesRegex(RuntimeError,"provenance"):
                proof._load_stage(path,clone,opt2,contract,"E_frequency_loss",
                                  torch.device("cpu"))

    def test_matched_d_e_comparison_uses_only_common_update_checkpoints(self):
        def row(update, global_value, edge, detail):
            return {
                "update":update,
                "validationScore":global_value+2*edge,
                "validation":{"mean":{
                    "global_recovery":global_value,
                    "edge_recovery":edge,
                    "gradient_recovery":0.2,
                    "detail_recovery_1px":detail,
                    "lattice_cell_excess":0.1,
                }},
            }
        results={
            "D_continuation":{"history":[row(61440,0.2,0.25,0.1),row(73728,0.22,0.3,0.11)]},
            "E_frequency_loss":{"history":[row(61440,0.24,0.28,0.17)]},
        }
        paired=proof._matched_comparison(results)
        self.assertEqual(len(paired),1)
        self.assertEqual(paired[0]["update"],61440)
        self.assertAlmostEqual(paired[0]["delta_edge_recovery"],0.03)
        self.assertAlmostEqual(paired[0]["delta_detail_recovery_1px"],0.07)
        self.assertEqual(proof._matched_comparison({}),[])

    def test_cli_command_and_test_suite_registered(self):
        app=NSAMDRCommandLineApplication()
        parsed=app.build_parser().parse_args(["v17-comprehensive-af","--audit-only"])
        self.assertTrue(parsed.audit_only)
        self.assertEqual(parsed.resume,"fresh")
        self.assertEqual(parsed.patience,2)
        parsed=app.build_parser().parse_args(["v17-comprehensive-af","--resume","auto"])
        self.assertEqual(parsed.resume,"auto")


if __name__=="__main__":
    unittest.main()
