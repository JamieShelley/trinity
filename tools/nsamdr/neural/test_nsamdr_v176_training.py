"""V17.6 Stage10A CPU tests: authenticated parent and specialist-only resume."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools.nsamdr.neural import probe_nsamdr_v176_detail as proof
from tools.nsamdr.neural.test_nsamdr_v176_detail import _lr, _model
from tools.nsamdr.neural.v17.diversity_corpus import sha256_file


class V176TrainingProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls)->None:
        torch.set_num_threads(min(torch.get_num_threads(),2))

    def test_schedule_and_research_contract(self):
        self.assertEqual(proof.STAGES,(0,1536,3072,6144))
        self.assertEqual(proof.LR,.0002)
        self.assertEqual(proof.DETAIL_SEED,17610)
        self.assertEqual(proof.SCHEMA,"NSAMDR_V176_STAGE10A_DETAIL_SPECIALIST_PROOF_V1")

    def test_cli_audit_defaults_no_background_training(self):
        args=proof.parser().parse_args(["--audit-only"])
        self.assertTrue(args.audit_only)
        self.assertEqual(args.resume,"fresh")
        self.assertFalse(args.stage9_report)

    def test_no_completed_parent_proof_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,"not found"):
                proof._parent_report(Path(tmp),"")

    def test_explicit_nonexistent_parent_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError,"missing"):
            proof._parent_report(Path("/unused"),"/definitely/not/here/report.json")

    def test_checkpoint_roundtrip_preserves_specialist_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=_model()
            opt=torch.optim.Adam(model.trainable_parameters(),lr=proof.LR)
            model.train()
            result=model(*_lr())
            loss=result["fine_detail_highpass"].square().mean()+(
                result["candidate_albedo"]-0.42).square().mean()
            loss.backward()
            opt.step()
            state=copy.deepcopy(model.specialist.state_dict())
            torch.manual_seed(101)
            path=Path(tmp)/"stage_001536/checkpoint.pt"
            history=[{"update":0},{"update":1536}]
            proof._save(path,model,opt,{"parent":"sha"},1536,history,1536,1.5,False)
            expected=torch.rand(4)
            with torch.no_grad():
                model.specialist.head.weight.zero_()
            restored=proof._restore(path,model,opt,{"parent":"sha"},torch.device("cpu"))
            self.assertEqual(restored["update"],1536)
            self.assertTrue(torch.equal(expected,torch.rand(4)))
            self.assertTrue(torch.equal(model.specialist.head.weight,
                                        state["head.weight"]))
            self.assertEqual(opt.param_groups[0]["lr"],proof.LR)

    def test_checkpoint_refuses_wrong_parent_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"ckpt.pt"
            model=_model()
            opt=torch.optim.Adam(model.trainable_parameters(),lr=proof.LR)
            proof._save(path,model,opt,{"parent":"original"},1536,
                        [{"update":0},{"update":1536}],1536,1.5,False)
            with self.assertRaisesRegex(RuntimeError,"identity"):
                proof._restore(path,model,opt,{"parent":"other"},torch.device("cpu"))

    def test_checkpoint_refuses_wrong_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"ckpt.pt"
            model=_model()
            opt=torch.optim.Adam(model.trainable_parameters(),lr=proof.LR)
            proof._save(path,model,opt,{},1536,[{"update":1536}],1536,1.5,False)
            with self.assertRaisesRegex(RuntimeError,"identity"):
                proof._restore(path,model,opt,{},torch.device("cpu"))

    def test_stage_zero_baseline_is_restorable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"stage_000000/checkpoint.pt"
            model=_model()
            opt=torch.optim.Adam(model.trainable_parameters(),lr=proof.LR)
            proof._save(path,model,opt,{},0,[{"update":0}],0,0.,False)
            state=proof._restore(path,model,opt,{},torch.device("cpu"))
            self.assertEqual(state["update"],0)

    def test_optimization_updates_only_specialist(self):
        model=_model()
        original={name:part.detach().clone()
                  for name,part in model.backbone.state_dict().items()}
        output=model(*_lr())
        lr=_lr()
        with torch.no_grad():
            target=(output["candidate_albedo"]+
                    torch.randn_like(output["candidate_albedo"])*.1).clamp(0,1)
        batch={"lr_albedo":lr[0],"lr_normal":lr[1],"lr_material":lr[2],
               "target_albedo":target}
        opt=torch.optim.Adam(model.trainable_parameters(),lr=proof.LR)
        proof._train_one(model,opt,batch,torch.device("cpu"),"auto")
        self.assertTrue(all(torch.equal(part,model.backbone.state_dict()[name])
                            for name,part in original.items()))
        self.assertTrue(any(p.grad is not None for p in model.trainable_parameters()))

    def test_stage9_parent_checkpoint_authentication_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stage9=root/"stage9"
            stage9.mkdir()
            stage8=root/"stage8.json"
            stage8.write_text("{}")
            sources=root/"manifest.json"
            sources.write_text("{}")
            parent={
                "schema":proof.PARENT_SCHEMA,"stages":[36864,49152],
                "trainingSeed":17301,"sealedFinalEnabled":False,
                "trainedSelectionSha256":"selectionhash",
                "sourceManifestSha256":sha256_file(sources),
                "parentStage8ReportSha256":"stage8hash",
                "parentStage8ManifestSha256":"stage8manifesthash",
            }
            selection={"controlTrainingCropIds":[f"test{i}_000" for i in range(12)],
                       "expandedTrainingCropIds":[f"test{i}_000" for i in range(64)],
                       "developmentCropIds":[f"dev{i}_000" for i in range(4)],
                       "sealedFinalCropIds":[f"sealed{i}_000" for i in range(8)]}
            report={
                "schema":proof.PARENT_SCHEMA,"researchOnly":True,
                "finalTestEvaluated":False,"finalTestEvaluationCount":0,
                "productionPromotionAllowed":False,
                "qualificationGatesModified":False,
                "additionalOptimizerUpdates":49152,
                "armResults":{"A12":{"atFinalUpdate":{"update":49152},
                                     "trainingAuthorities":[f"test{i}" for i in range(12)]},
                              "B64":{}},
                "runContract":parent,"trainedSelectionSha256":"selectionhash",
                "stage8SourceReport":str(stage8),
            }
            path=stage9/"report.json"
            path.write_text(json.dumps(report))
            (stage9/"run_manifest.json").write_text(json.dumps(parent))
            model=_model().backbone
            ckpt=stage9/"A12/stage_049152/checkpoint.pt"
            ckpt.parent.mkdir(parents=True)
            state={
                "schema":proof.PARENT_CHECKPOINT_SCHEMA,"contract":parent,
                "arm":"A12","update":49152,
                "history":[{"update":36864},{"update":49152}],
                "architecture":model.architecture_contract(),
                "modelState":model.state_dict(),
                "optimizerState":{"param_groups":[]},
                "torchRngState":torch.get_rng_state(),
                "cudaRngState":[torch.zeros(20,dtype=torch.uint8)],
            }
            torch.save(state,ckpt)
            audit8={"trainedSelectionSha256":"selectionhash",
                    "stage8ReportSha256":"stage8hash",
                    "stage8RunManifestSha256":"stage8manifesthash"}
            fake_manifest={"crops":[{"crop_id":f"test{i}_000","family_id":f"test{i}"}
                                    for i in range(64)]}
            with patch.object(proof,"verify_stage8",
                              return_value=(selection,{},audit8)):
                obtained,audit=proof.verify_parent(path,
                    {"manifest":str(sources)},fake_manifest,{},model)
                self.assertEqual(len(audit["trainingAuthorityIds"]),12)
                self.assertEqual(audit["finalTestPixelsOpened"],0)
                self.assertEqual(obtained,selection)
                state["update"]=36864
                torch.save(state,ckpt)
                with self.assertRaisesRegex(RuntimeError,"identity"):
                    proof.verify_parent(path,
                        {"manifest":str(sources)},fake_manifest,{},model)

    def test_sealed_final_cannot_be_used_in_training(self):
        body=Path(proof.__file__).read_text(encoding="utf-8")
        self.assertNotIn('parts["final_test"]',body)
        self.assertNotIn('role="final_test"',body)
        self.assertIn('parts["validation"]',body)
        self.assertIn('"finalTestEvaluated":False',body)

    def test_auto_resume_requires_one_unfinished_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/proof.ROOT
            root.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError,"one unfinished"):
                proof._run_folder(Path(tmp),"auto")
            for index in (1,2):
                p=root/f"proof_{index}"
                p.mkdir()
                (p/"run_manifest.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError,"got 2"):
                proof._run_folder(Path(tmp),"auto")


if __name__=="__main__":
    unittest.main()
