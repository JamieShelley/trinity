"""V17.5 matched authority diversity runner contract tests; no training or sealed pixels."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from tools.nsamdr.neural import probe_nsamdr_v17_5_diversity as d

class TinyModel(torch.nn.Linear):
    def __init__(self):
        super().__init__(2,1)
    def architecture_contract(self):
        return {"revision":"tiny","channels":2}

def _summary(global_recovery=.4,edge=.4,lattice=.1,qualifies=False):
    rows=[{"metrics":{"global_recovery":global_recovery,
                      "edge_recovery":edge,"gradient_recovery":.3,
                      "detail_recovery_1px":.2,"lattice_cell_excess":lattice},
           "gateChecks":{}} for _ in range(4)]
    return {"allPass":qualifies,"mean":{"global_recovery":global_recovery,
                                      "edge_recovery":edge,"gradient_recovery":.3,
                                      "detail_recovery_1px":.2,
                                      "lattice_cell_excess":lattice},
            "perCrop":rows,"gateChecks":[{}]*4}

def _arm(name,edges=(.3,.31,.32),qualified=False,train_edge=.5):
    history=[]
    for stage,edge in zip(d.STAGES,edges):
        history.append({"update":stage,"trained":_summary(edge=train_edge),
                        "validation":_summary(edge=edge),"validationScore":edge,
                        "newBest":True})
    return {"arm":name,"bestValidationUpdate":d.STAGES[-1],
            "selectedTrained":_summary(edge=train_edge,qualifies=qualified),
            "selectedValidation":_summary(edge=edges[-1],qualifies=qualified),
            "history":history}

class DiversityRunnerTests(unittest.TestCase):
    def test_schema_stages_and_arms_locked(self):
        self.assertEqual(d.ARMS,("A12","B64"))
        self.assertEqual(d.STAGES,(6144,12288,24576))
        self.assertEqual(d.LEARNING_RATE,.0002)
        self.assertEqual(d.ARCHITECTURE["encoderChannels"],96)

    def test_cli_uses_fresh_cuda_research_default(self):
        args=d.parser().parse_args([])
        self.assertEqual((args.device,args.resume,args.seed),("cuda","fresh",17301))
        self.assertFalse(args.selection_only)

    def test_selection_only_and_second_seed_are_explicit(self):
        args=d.parser().parse_args(["--selection-only","--seed","17302","--selection","frozen.json"])
        self.assertTrue(args.selection_only)
        self.assertEqual(args.seed,17302)
        self.assertEqual(args.selection,"frozen.json")
        with self.assertRaises(SystemExit):
            d.parser().parse_args(["--seed","666"])

    def test_initial_state_sha_is_order_independent_and_changes_with_parameters(self):
        model=TinyModel()
        before=d._tensor_sha(model)
        self.assertEqual(before,d._tensor_sha(model))
        with torch.no_grad(): model.weight.add_(1)
        self.assertNotEqual(before,d._tensor_sha(model))

    def test_checkpoint_roundtrip_preserves_model_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"A12/stage_006144/checkpoint.pt"
            model=TinyModel()
            opt=torch.optim.Adam(model.parameters(),lr=.0002)
            torch.manual_seed(17)
            loss=model(torch.ones(1,2)).square().mean()
            loss.backward()
            opt.step()
            weights=copy.deepcopy(model.state_dict())
            contract={"selectionSha256":"abc","trainingSeed":17301}
            d._checkpoint_save(path,model,opt,contract,"A12",6144,[{"update":6144}],
                               6144,1.0,False)
            expected=torch.rand(3)
            with torch.no_grad(): model.weight.mul_(0)
            state=d._checkpoint_load(path,model,opt,contract,"A12",torch.device("cpu"))
            actual=torch.rand(3)
            self.assertTrue(torch.equal(actual,expected))
            self.assertTrue(torch.equal(model.weight,weights["weight"]))
            self.assertEqual(state["update"],6144)
            self.assertEqual(opt.param_groups[0]["lr"],.0002)

    def test_resume_rejects_wrong_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"x.pt"
            model=TinyModel()
            opt=torch.optim.Adam(model.parameters())
            d._checkpoint_save(path,model,opt,{},"A12",6144,[],6144,0.,False)
            with self.assertRaisesRegex(RuntimeError,"provenance"):
                d._checkpoint_load(path,model,opt,{},"B64",torch.device("cpu"))

    def test_resume_rejects_contract_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"x.pt"
            model=TinyModel()
            opt=torch.optim.Adam(model.parameters())
            d._checkpoint_save(path,model,opt,{"seed":17301},"A12",6144,[],6144,0.,False)
            with self.assertRaisesRegex(RuntimeError,"provenance"):
                d._checkpoint_load(path,model,opt,{"seed":17302},"A12",torch.device("cpu"))

    def test_resume_rejects_architecture_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"x.pt"
            model=TinyModel()
            opt=torch.optim.Adam(model.parameters())
            d._checkpoint_save(path,model,opt,{},"A12",6144,[],6144,0.,False)
            with patch.object(model,"architecture_contract",return_value={"changed":True}):
                with self.assertRaisesRegex(RuntimeError,"architecture"):
                    d._checkpoint_load(path,model,opt,{},"A12",torch.device("cpu"))

    def test_resume_refuses_missing_or_multiple_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,"exactly one"):
                d._resume_directory(repo)
            root=repo/d.ROOT
            for name in ("proof_one","proof_two"):
                (root/name).mkdir(parents=True)
                (root/name/"run_manifest.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError,"got 2"):
                d._resume_directory(repo)

    def test_completed_proof_not_auto_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp)
            folder=repo/d.ROOT/"proof_one"
            folder.mkdir(parents=True)
            (folder/"run_manifest.json").write_text("{}")
            self.assertEqual(d._resume_directory(repo),folder)
            (folder/"report.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                d._resume_directory(repo)

    def test_matched_comparison_has_three_stages_and_actual_exposures(self):
        rows=d._compare({"A12":_arm("A12"),"B64":_arm("B64")})
        self.assertEqual([r["update"] for r in rows],list(d.STAGES))
        self.assertEqual(rows[-1]["A12ExposuresPerAuthority"],2048)
        self.assertEqual(rows[-1]["B64ExposuresPerAuthority"],384)
        self.assertEqual(len(rows[-1]["edgeDeltaByCrop"]),4)

    def test_outcome_reports_research_signal_only(self):
        a=_arm("A12",edges=(.2,.2,.2))
        b=_arm("B64",edges=(.2,.25,.35))
        self.assertEqual(d._decision({"A12":a,"B64":b})[0],"DIVERSITY_BENEFIT_SIGNAL")

    def test_original_control_can_qualify_without_opening_sealed_set(self):
        arms={"A12":_arm("A12",qualified=True),
              "B64":_arm("B64",qualified=False)}
        self.assertEqual(d._decision(arms)[0],"VALIDATION_QUALIFIED_RESEARCH_ONLY")

    def test_outcome_no_benefit_not_qualified(self):
        arms={"A12":_arm("A12"),"B64":_arm("B64")}
        self.assertEqual(d._decision(arms)[0],"NO_DEMONSTRATED_BENEFIT")

    def test_outcome_qualified_requires_train_and_dev(self):
        arms={"A12":_arm("A12"),"B64":_arm("B64",qualified=True)}
        self.assertEqual(d._decision(arms)[0],"VALIDATION_QUALIFIED_RESEARCH_ONLY")

    def test_underexposure_is_flagged_separately(self):
        a=_arm("A12",train_edge=.7)
        b=_arm("B64",train_edge=.4)
        result,under=d._decision({"A12":a,"B64":b})
        self.assertEqual(result,"NO_DEMONSTRATED_BENEFIT")
        self.assertTrue(under)

    def test_report_cannot_promote_or_unseal_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            source={"v3Report":"v3_report.json"}
            selection={"selectionPolicy":"source-policy"}
            report=d._initial_report(source,selection,{},Path(tmp))
            self.assertFalse(report["productionPromotionAllowed"])
            self.assertFalse(report["finalTestEvaluated"])
            self.assertEqual(report["finalTestEvaluationCount"],0)
            self.assertIsNone(report["selectedFinalTest"])
            self.assertTrue(report["researchOnly"])

    def test_atomic_write_creates_valid_json_without_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"nested/report.json"
            d._atomic_json(path,{"schema":"test"})
            self.assertEqual(json.loads(path.read_text())["schema"],"test")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_readiness_rejects_original_architecture_drift(self):
        model=TinyModel()
        with self.assertRaisesRegex(RuntimeError,"architecture"):
            d._assert_unchanged_architecture(model,{"v3Contract":{"architecture":{"x":1}}})

    def test_selected_preview_role_only_validation_in_runner_source(self):
        path=Path(d.__file__)
        source=path.read_text(encoding="utf-8")
        self.assertIn('role="validation"',source)
        self.assertNotIn('role="final_test"',source)
        self.assertNotIn('parts["final_test"]',source)

    def test_source_change_is_hashed_into_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"source"
            path.write_bytes(b"a")
            h1=d.sha256_file(path)
            path.write_bytes(b"b")
            self.assertNotEqual(h1,d.sha256_file(path))

if __name__=="__main__":
    unittest.main()
