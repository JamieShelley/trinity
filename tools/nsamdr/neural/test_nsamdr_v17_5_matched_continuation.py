"""Stage 9 exact-resume, provenance, diagnostics, and sealed-test tests (CPU)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools.nsamdr.neural import probe_nsamdr_v17_5_matched_continuation as runner
from tools.nsamdr.neural.v17 import diversity_continuation as contract
from tools.nsamdr.neural.v17.diversity_corpus import sha256_json,sha256_file


class Tiny(torch.nn.Linear):
    def __init__(self):
        super().__init__(2,1)
    def architecture_contract(self):
        return {"revision":"unit-test"}

def snapshot(edge:float=.30,lattice:float=.10,qualified:bool=False,
             count:int=4) -> dict:
    crops = []
    for i in range(count):
        crops.append({
            "metrics":{
                "edge_recovery":edge+i*.001,
                "global_recovery":.32,
                "gradient_recovery":.27,
                "detail_recovery_1px":.08,
                "lattice_cell_excess":lattice,
            },
            "gateChecks":{"edge":qualified,"global":qualified,
                          "gradient":qualified,"lattice":qualified},
            "residualDiagnostics":{
                "residual_cosine_similarity":.55+i*.01,
                "candidate_phase_energy_spread":.35,
                "target_phase_energy_spread":.23,
                "candidate_to_target_residual_ratio":.7,
                "target_weighted_sign_agreement":.8,
            },
        })
    return {"allPass":qualified,"perCrop":crops,"mean":{
        "edge_recovery":edge,"global_recovery":.32,
        "gradient_recovery":.27,"detail_recovery_1px":.08,
        "lattice_cell_excess":lattice}}


def selection_fixture() -> dict:
    def item(i):
        return {"familyId":f"family{i}","cropId":f"family{i}_000",
                "cropBundleSha256":f"bundle{i}",
                "albedo":{"sha256":f"albedo{i}"},
                "normal":{"sha256":f"normal{i}"}}
    return {
        "controlTrainingCropIds":[f"family{i}_000" for i in range(12)],
        "expandedTrainingCropIds":[f"family{i}_000" for i in range(64)],
        "developmentCropIds":[f"dev{i}_000" for i in range(4)],
        "sealedFinalCropIds":[f"sealed{i}_000" for i in range(8)],
        "selectedAuthoritySources":[item(i) for i in range(64)],
        "manifestSha256":"manifest-hash",
    }


class MatchedContinuationTests(unittest.TestCase):
    def test_stage_schema_and_exact_budgets(self):
        self.assertEqual(contract.UPDATES,(36864,49152))
        self.assertEqual(contract.ARMS,("A12","B64"))
        self.assertEqual(runner.START_UPDATE,24576)
        self.assertEqual(runner.CHECKPOINT_SCHEMA,
                         "NSAMDR_V17_5_MATCHED_CONTINUATION_CHECKPOINT_V1")

    def test_cli_audit_defaults_and_cuda_only_training(self):
        args=runner.parser().parse_args(["--audit-only"])
        self.assertTrue(args.audit_only)
        self.assertEqual((args.resume,args.device),("fresh","cuda"))
        with self.assertRaises(SystemExit):
            runner.parser().parse_args(["--device","cpu"])

    def test_stage9_optimizer_model_and_rng_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            model=Tiny()
            optimizer=torch.optim.Adam(model.parameters(),lr=.0002)
            torch.manual_seed(100)
            loss=model(torch.ones(1,2)).square().mean()
            loss.backward()
            optimizer.step()
            weights=copy.deepcopy(model.state_dict())
            path=Path(temp)/"A12/stage_036864/checkpoint.pt"
            runner._stage_save(path,model,optimizer,{"x":"source"},
                               "A12",36864,[],36864,1.2,False)
            expected=torch.rand(3)
            with torch.no_grad():
                model.weight.mul_(0)
            state=runner._stage_load(path,model,optimizer,{"x":"source"},
                                     "A12",torch.device("cpu"))
            self.assertEqual(state["update"],36864)
            self.assertTrue(torch.equal(torch.rand(3),expected))
            self.assertTrue(torch.equal(weights["weight"],model.weight))
            self.assertEqual(optimizer.param_groups[0]["lr"],.0002)

    def test_rejects_cross_arm_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            model=Tiny()
            opt=torch.optim.Adam(model.parameters())
            path=Path(temp)/"stage.pt"
            runner._stage_save(path,model,opt,{},"A12",36864,[],36864,0.,False)
            with self.assertRaisesRegex(RuntimeError,"mismatch"):
                runner._stage_load(path,model,opt,{},"B64",torch.device("cpu"))

    def test_rejects_changed_run_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            model=Tiny()
            opt=torch.optim.Adam(model.parameters())
            path=Path(temp)/"stage.pt"
            runner._stage_save(path,model,opt,{"seed":17301},
                               "A12",36864,[],36864,0.,False)
            with self.assertRaisesRegex(RuntimeError,"mismatch"):
                runner._stage_load(path,model,opt,{"seed":17302},
                                   "A12",torch.device("cpu"))

    def test_stage9_rejects_invalid_checkpoint_update(self):
        model=Tiny()
        opt=torch.optim.Adam(model.parameters())
        with self.assertRaisesRegex(RuntimeError,"invalid"):
            runner._stage_save(Path("unused"),model,opt,{},"A12",24576,[],0,0.,False)

    def test_probe_groups_exact_common12_and_additional52(self):
        data=snapshot(count=64)
        info=contract.probe_groups(data,"B64")
        self.assertEqual((info["all"]["count"],info["commonA12"]["count"],
                          info["additional52"]["count"]),(64,12,52))
        self.assertAlmostEqual(info["commonA12"]["mean"]["edge_recovery"],.3055)
        self.assertEqual(contract.probe_groups(snapshot(count=12),"A12")
                         ["commonA12"]["count"],12)
        with self.assertRaises(RuntimeError):
            contract.probe_groups(snapshot(count=12),"B64")

    def test_phase_residual_diagnostics_per_crop(self):
        report=contract.diagnostic_summary(snapshot(count=4))
        self.assertEqual(len(report["perCrop"]),4)
        self.assertAlmostEqual(
            report["perCrop"][2]["residual_cosine_similarity"],.57)
        self.assertAlmostEqual(report["perCrop"][0]["latticeExcess"],.1)

    def test_preview_comparison_metadata_only_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            trained=selection_fixture()
            preview=copy.deepcopy(trained)
            preview["baselineResidualComputedOn"]="cpu"
            trained["baselineResidualComputedOn"]="cuda"
            first=base/"proof_01"
            second=base/"proof_02"
            first.mkdir();second.mkdir()
            report=first/"report.json"
            report.write_text(json.dumps({"schema":contract.STAGE8_SCHEMA,
                "selectionOnly":True,"runContract":{
                    "sourceManifestSha256":"manifest-hash"}}))
            (first/"frozen_selection.json").write_text(json.dumps(preview))
            later=second/"report.json"
            later.write_text("{}")
            result=contract.compare_selection_preview(trained,later,
                                                       specified=str(report))
            self.assertEqual(result["status"],"MATCH")
            self.assertTrue(result["metadataOnlyDifference"])

    def test_preview_different_roster_is_visible_not_silent(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            train=selection_fixture()
            prev=copy.deepcopy(train)
            prev["expandedTrainingCropIds"][13]="different_000"
            preview=base/"proof_01"
            preview.mkdir()
            (preview/"report.json").write_text(json.dumps({
                "schema":contract.STAGE8_SCHEMA,"selectionOnly":True}))
            (preview/"frozen_selection.json").write_text(json.dumps(prev))
            training=base/"proof_02"
            training.mkdir()
            (training/"report.json").write_text("{}")
            result=contract.compare_selection_preview(
                train,training/"report.json",
                specified=str(preview/"report.json"))
            self.assertEqual(result["status"],"MISMATCH_ROSTER_OR_SOURCE")
            self.assertIn("expandedIds",result["differingIdentityFields"])

    def test_missing_explicit_preview_fails(self):
        with self.assertRaisesRegex(RuntimeError,"missing"):
            contract.compare_selection_preview(selection_fixture(),
                                               Path("stage8/report.json"),
                                               specified="does-not-exist.json")

    def test_matched_checkpoints_are_not_cherry_picked(self):
        base={"armResults":{}}
        for arm in contract.ARMS:
            base["armResults"][arm]={"history":[{
                "trained":snapshot(count=12 if arm=="A12" else 64),
                "validation":snapshot(edge=.20)}]}
        def arm(name,edge):
            records=[]
            for i,update in enumerate(contract.UPDATES):
                records.append({
                    "update":update,
                    "trained":snapshot(count=12 if name=="A12" else 64),
                    "validation":snapshot(edge=edge+i*.03)})
            return {"history":records}
        matched=runner._compare(arm("A12",.25),arm("B64",.24),base)
        self.assertEqual([m["update"] for m in matched],
                         [24576,36864,49152])
        self.assertEqual(matched[-1]["A12ExposuresPerAuthority"],4096)
        self.assertEqual(matched[-1]["B64ExposuresPerAuthority"],768)
        self.assertEqual(len(matched[-1]["edgeDeltaByCrop"]),4)

    def test_quality_decision_uses_unmodified_gates(self):
        a={"atFinalUpdate":{"trained":snapshot(count=12),
                            "validation":snapshot()}}
        b={"atFinalUpdate":{"trained":snapshot(count=64),
                            "validation":snapshot()}}
        history=[{"B64WorstEdge":.2,"A12WorstEdge":.2,
                  "B64MeanEdge":.25,"A12MeanEdge":.25,
                  "B64MaxLattice":.1,"globalDeltaByCrop":[0]*4},
                 {"B64WorstEdge":.28,"A12WorstEdge":.24,
                  "B64MeanEdge":.34,"A12MeanEdge":.29,
                  "B64MaxLattice":.14,"globalDeltaByCrop":[0]*4}]
        self.assertEqual(contract.decide(history,{"A12":a,"B64":b}),
                         "EXPOSURE_BENEFIT_SIGNAL")
        b["atFinalUpdate"]["trained"]["allPass"]=True
        b["atFinalUpdate"]["validation"]["allPass"]=True
        self.assertEqual(contract.decide(history,{"A12":a,"B64":b}),
                         "VALIDATION_QUALIFIED_RESEARCH_ONLY")

    def test_lattice_regression_not_misreported_as_benefit(self):
        a={"atFinalUpdate":{"trained":snapshot(count=12),
                            "validation":snapshot()}}
        b={"atFinalUpdate":{"trained":snapshot(count=64),
                            "validation":snapshot()}}
        history=[{"B64MeanEdge":.20,"B64MaxLattice":.12},
                 {"B64WorstEdge":.28,"A12WorstEdge":.24,
                  "B64MeanEdge":.34,"A12MeanEdge":.29,
                  "B64MaxLattice":.17,"globalDeltaByCrop":[0]*4}]
        self.assertEqual(contract.decide(history,{"A12":a,"B64":b}),
                         "RECOVERY_LATTICE_CONFLICT")

    def test_sealed_pixels_never_loaded_by_runner(self):
        body=Path(runner.__file__).read_text(encoding="utf-8")
        self.assertNotIn('parts["final_test"]',body)
        self.assertNotIn('role="final_test"',body)
        self.assertIn('parts["validation"]',body)

    def test_auto_resume_refuses_ambiguous_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            repo=Path(temp)
            root=repo/runner.ROOT
            root.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError,"one unfinished"):
                runner._new_directory(repo,"auto")
            for i in (1,2):
                folder=root/f"proof_{i}"
                folder.mkdir()
                (folder/"run_manifest.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError,"found 2"):
                runner._new_directory(repo,"auto")

    def test_report_contract_binds_exact_parent_hashes(self):
        parent={"device":"cuda","ampPrecision":"auto",
                "sourceManifestSha256":"abc","trainingSeed":17301,
                "initialModelSha256":"def","architecture":{"x":1},
                "learningRate":.0002,"sampling":"spatial",
                "augmentation":"D4","loss":"V17.2"}
        audit={"stage8ReportSha256":"report","stage8RunManifestSha256":"manifest",
               "trainedSelectionSha256":"selection",
               "stage8CheckpointSha256":{"A12":"one","B64":"two"},
               "stage8CheckpointPaths":{"A12":"a.pt","B64":"b.pt"}}
        result=runner._contract(audit,parent,device="cuda",amp="auto")
        self.assertEqual(result["startUpdate"],24576)
        self.assertEqual(result["stages"],[36864,49152])
        self.assertFalse(result["sealedFinalEnabled"])
        self.assertEqual(result["parentCheckpointsSha256"]["B64"],"two")


if __name__=="__main__":
    unittest.main()
