"""CPU tests for the training-only attainable-recovery diagnostic."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools.nsamdr.neural import probe_nsamdr_v176_capacity as proof
from tools.nsamdr.neural.test_nsamdr_v176_capacity import _backbone, _batch
from tools.nsamdr.neural.v176 import attainable_recovery as oracle
from tools.nsamdr.neural.v176.detail_capacity import create_matched_variants
from tools.nsamdr.neural.v176.stability import NumericalStabilityError, STABILITY_POLICY
from tools.nsamdr.neural.v14.qualification import sample_metrics


def _synthetic():
    baseline=torch.full((1,3,24,24),.5)
    original={"baseline_albedo":baseline,"candidate_albedo":baseline,
              "candidate_raw_residual_albedo":torch.zeros_like(baseline),
              "baseline_normal":torch.zeros(1,2,24,24),
              "candidate_normal":torch.zeros(1,2,24,24),
              "baseline_material":torch.zeros_like(baseline),
              "candidate_material":torch.zeros_like(baseline)}
    raw=torch.full((1,1,24,24),.15)
    raw[...,::2,1::2]=-.15
    raw[...,1::2,::2]=-.15
    target=oracle.scalar_candidate(original,raw,.4)["candidate_albedo"].detach()
    return original,{"target_albedo":target,"target_normal":original["candidate_normal"],
                     "target_material":original["candidate_material"]}


def _comparison_fixture(root,backbone):
    rows=[{"family_id":str(i),"crop_id":f"crop_{i}"} for i in range(12)]
    contract={"schema":proof.SCHEMA,"phase":"compare-features","seed":proof.SEED,
              "stages":list(proof.STAGES),"alternateExposures":list(proof.ALT_EXPOSURES),
              "learningRate":proof.LEARNING_RATE,"loss":"unchanged Stage10A stationary Haar",
              "numericalStability":dict(STABILITY_POLICY),"frozenBackbone":True,
              "developmentPixelsOpened":0,"finalTestPixelsOpened":0,"finalTestEnabled":False,
              "parentStage9CheckpointSha256":"stage9","parentBackboneTensorSha256":proof._tensor_sha(backbone),
              "parentStage10AReportSha256":"stage10a","parentStage10AStagesSha256":{"6144":"stage10a-final"},
              "sourceManifestSha256":"manifest","selectionSha256":"selection",
              "trainingAuthorities":{str(i):{"authorityId":str(i),"cropId":f"crop_{i}"} for i in (1,10)}}
    report={"schema":proof.SCHEMA,"phase":"compare-features","runContract":contract,
            "decision":"NO_PROVEN_HR_FEATURE_BENEFIT","researchOnly":True,
            "developmentPixelsOpened":0,"finalTestEvaluated":False,"finalTestEvaluationCount":0,
            "productionPromotionAllowed":False,"qualificationGatesModified":False,"armResults":{}}
    for i in (1,10):
        for mode,model in zip(proof.MODES,create_matched_variants(backbone,seed=proof.SEED+i)):
            arm=proof._arm_name(i,mode)
            history=[{"update":n,**({"backboneSha256":proof._tensor_sha(backbone)} if n else {})}
                     for n in proof.STAGES]
            report["armResults"][arm]={"arm":arm,"mode":mode,"authorityIndex":i,"authorityId":str(i),
                                       "startUpdate":0,"stopUpdate":6144,"history":history,
                                       "runContractSha256":proof.sha256_json(contract),
                                       "alternateProbeOrigins":[{"origin":[n,n]} for n in range(4)]}
            path=root/arm/"stage_006144/checkpoint.pt"
            path.parent.mkdir(parents=True)
            torch.save({"schema":proof.CHECKPOINT_SCHEMA,"contract":contract,"arm":arm,
                        "update":6144,"history":history,"architecture":model.architecture_contract(),
                        "detailState":model.specialist.state_dict()},path)
    proof._atomic_json(root/"run_manifest.json",contract)
    proof._atomic_json(root/"report.json",report)
    return rows,contract,report


class V176AttainableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(torch.get_num_threads(),2))

    def test_direct_fit_recovers_known_feasible_detail_with_both_objectives(self):
        original,batch=_synthetic()
        before={k:v.clone() for k,v in original.items()}
        seed=None
        for objective in oracle.ATTAINABLE_POLICY["objectives"]:
            with self.subTest(objective=objective):
                outputs,fit=oracle.fit_scalar_field(original,batch,.4,objective,
                                                   initial_raw=seed,stages=(0,16,64))
                self.assertLess(fit["bestObjective"],fit["zeroFieldObjective"]*.10)
                self.assertLessEqual(fit["bestObjective"],fit["lastObjective"])
                self.assertEqual(fit["updates"],64)
                self.assertFalse(fit["globalOptimalityProven"])
                self.assertTrue(torch.equal(outputs["candidate_normal"],original["candidate_normal"]))
                self.assertTrue(torch.equal(outputs["candidate_material"],original["candidate_material"]))
                self.assertFalse(outputs["candidate_albedo"].requires_grad)
                self.assertLessEqual(float(outputs["predicted_residual_albedo"].abs().max()),.4)
                if objective=="edge":
                    seed=outputs["fine_detail_raw"]
        self.assertTrue(all(torch.equal(before[k],v) for k,v in original.items()))

    def test_zero_target_retains_exact_zero_field_control(self):
        original,batch=_synthetic()
        batch["target_albedo"]=original["candidate_albedo"].clone()
        outputs,fit=oracle.fit_scalar_field(original,batch,.4,"stage10a",stages=(0,3))
        self.assertEqual(fit["bestUpdate"],0)
        self.assertEqual(fit["bestObjective"],0)
        self.assertTrue(torch.equal(outputs["candidate_albedo"],original["candidate_albedo"]))
        self.assertEqual(int(torch.count_nonzero(outputs["fine_detail_raw"])),0)

    def test_edge_objective_is_the_existing_recovery_metric_numerator(self):
        original,batch=_synthetic()
        control=oracle.scalar_candidate(original,torch.zeros(1,1,24,24),.4)
        metrics=sample_metrics(control,batch)
        error=oracle.edge_objective(control["candidate_albedo"],batch["target_albedo"])
        baseline=oracle.edge_objective(original["baseline_albedo"],batch["target_albedo"])
        self.assertAlmostEqual(metrics["edge_recovery"],float(1-error/baseline),places=6)

    def test_rgb_relaxation_bounds_both_recovery_metrics_for_feasible_fields(self):
        original,batch=_synthetic()
        torch.manual_seed(33)
        batch["target_albedo"]=torch.rand_like(batch["target_albedo"])
        relaxed={**original,"candidate_albedo":oracle.rgb_projection_relaxation(
            original,batch["target_albedo"],.4)}
        upper=sample_metrics(relaxed,batch)
        for _ in range(3):
            feasible=oracle.scalar_candidate(original,torch.randn(1,1,24,24)*.2,.4)
            measured=sample_metrics(feasible,batch)
            for metric in ("edge_recovery","global_recovery"):
                self.assertGreaterEqual(upper[metric]+1e-6,measured[metric])

    def test_only_relaxation_failure_can_establish_projection_range_limit(self):
        original,batch=_synthetic()
        batch["target_albedo"]=torch.full_like(batch["target_albedo"],.99)
        relaxed={**original,"candidate_albedo":oracle.rgb_projection_relaxation(
            original,batch["target_albedo"],.02)}
        measured=proof._measure_attainable(relaxed,batch,_backbone().config,with_loss=False)
        self.assertFalse(measured["gateChecks"]["edge"])
        self.assertEqual(proof._attainable_evidence({"rgbProjectionRelaxation":measured}),
                         "PROJECTION_RANGE_LIMIT_ON_EDGE_OR_GLOBAL")

    def test_failed_local_fits_are_reported_as_unresolved(self):
        bad={"measurement":{"gateChecks":{"edge":False,"global":True,"gradient":True,"lattice":True}}}
        region={"rgbProjectionRelaxation":{"gateChecks":{"edge":True,"global":True}},
                "fits":{"stage10a":copy.deepcopy(bad),"edge":copy.deepcopy(bad)}}
        self.assertIn("UNRESOLVED_FINITE_BUDGET",proof._attainable_evidence(region))
        region["fits"]["edge"]["measurement"]["gateChecks"]["edge"]=True
        self.assertEqual(proof._attainable_evidence(region),
                         "FEASIBLE_EDGE_FIT_OBJECTIVE_OR_OPTIMIZATION_UNRESOLVED")
        region["fits"]["stage10a"]["measurement"]["gateChecks"]["edge"]=True
        self.assertEqual(proof._attainable_evidence(region),
                         "FEASIBLE_SAME_LOSS_FIT_NETWORK_LEARNING_UNRESOLVED")

    def test_feasibility_backtracking_preserves_the_real_physical_path(self):
        before=torch.zeros(1,1,24,24)
        proposal=before.clone()
        proposal[...,::2]=32
        proposal[...,1::2]=-32
        count=oracle._accept_feasible_step(proposal,before)
        self.assertGreater(count,0)
        self.assertLessEqual(float(oracle.gaussian_highpass(proposal).abs().max()),4.)
        self.assertEqual(float(before.abs().max()),0.)
        with self.assertRaises(NumericalStabilityError):
            oracle._accept_feasible_step(torch.full_like(before,float("nan")),before)

    def test_fit_rejects_unknown_objective_bad_stages_and_nonfinite_target(self):
        original,batch=_synthetic()
        with self.assertRaises(ValueError):
            oracle.fit_scalar_field(original,batch,.4,"unknown",stages=(0,1))
        with self.assertRaises(ValueError):
            oracle.fit_scalar_field(original,batch,.4,"edge",stages=(0,4,2))
        batch["target_albedo"][...,0,0]=float("nan")
        with self.assertRaises(NumericalStabilityError):
            oracle.fit_scalar_field(original,batch,.4,"edge",stages=(0,1))

    def test_comparison_provenance_authenticates_all_four_checkpoint_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _,contract,report=_comparison_fixture(root,_backbone())
            actual,audit=proof._verify_comparison(root/"report.json",contract)
            self.assertEqual(actual,report)
            self.assertEqual(len(audit["comparisonCheckpointSha256"]),4)
            (root/"train_10_hr_features58/stage_006144/checkpoint.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                proof._verify_comparison(root/"report.json",contract)

    def test_comparison_rejects_changed_source_policy_and_incomplete_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            _,contract,report=_comparison_fixture(root,_backbone())
            wrong=copy.deepcopy(contract)
            wrong["sourceManifestSha256"]="other-source"
            with self.assertRaisesRegex(RuntimeError,"lineage"):
                proof._verify_comparison(root/"report.json",wrong)
            wrong=copy.deepcopy(contract)
            wrong["numericalStability"]["filterPrecision"]="bf16"
            with self.assertRaisesRegex(RuntimeError,"lineage"):
                proof._verify_comparison(root/"report.json",wrong)
            report["armResults"]["train_01_output10"]["history"].pop()
            proof._atomic_json(root/"report.json",report)
            with self.assertRaisesRegex(RuntimeError,"checkpoint differs"):
                proof._verify_comparison(root/"report.json",contract)

    def test_runner_uses_original_regions_preserves_weights_and_resumes_completed_regions(self):
        backbone=_backbone()
        plain,_=create_matched_variants(backbone)
        batch=_batch(plain)
        with torch.no_grad():
            original=backbone(batch["lr_albedo"],batch["lr_normal"],batch["lr_material"])
        batch["target_normal"]=original["candidate_normal"]
        batch["target_material"]=original["candidate_material"]
        before={k:v.clone() for k,v in backbone.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows,lineage,comparison=_comparison_fixture(root/"comparison",backbone)
            _,audit=proof._verify_comparison(root/"comparison/report.json",lineage)
            contract={**lineage,"phase":"attainable","comparisonCheckpointSha256":audit["comparisonCheckpointSha256"]}
            origins=[{"origin":[n,n]} for n in range(4)]
            def samples(_sampler,authority,_device):
                self.assertIn(authority,("1","10"))
                return batch,[batch]*4,origins
            policy={**oracle.ATTAINABLE_POLICY,"stages":[0,2]}
            with patch.object(proof,"_samples",side_effect=samples), \
                 patch.dict(oracle.ATTAINABLE_POLICY,policy), \
                 patch("builtins.print"):
                result=proof._run_attainable(root/"probe",backbone,root/"comparison/report.json",
                    comparison,contract,None,rows,backbone.config,torch.device("cpu"))
                self.assertEqual(set(result["attainableRecovery"]),{"1","10"})
                self.assertEqual(sum(len(x["regions"]) for x in result["attainableRecovery"].values()),10)
                self.assertFalse(result["networkWeightsUpdated"])
                self.assertFalse(result["globalOptimalityProven"])
                self.assertEqual(len(list((root/"probe").rglob("*.json"))),10)
                self.assertFalse(list((root/"probe").rglob("checkpoint.pt")))
                with patch.object(proof,"fit_scalar_field",side_effect=AssertionError("completed regions must be reused")):
                    resumed=proof._run_attainable(root/"probe",backbone,root/"comparison/report.json",
                        comparison,contract,None,rows,backbone.config,torch.device("cpu"))
                self.assertEqual(result,resumed)
                path=root/"probe/authority_01/fixed.json"
                invalid=json.loads(path.read_text())
                invalid["runContractSha256"]="other-contract"
                proof._atomic_json(path,invalid)
                with self.assertRaisesRegex(RuntimeError,"resume identity"):
                    proof._run_attainable(root/"probe",backbone,root/"comparison/report.json",
                        comparison,contract,None,rows,backbone.config,torch.device("cpu"))
        self.assertTrue(all(torch.equal(v,backbone.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))

    def test_probe_parser_accepts_phase_and_explicit_comparison(self):
        args=proof.parser().parse_args(["--phase","attainable","--comparison-report","matched.json"])
        self.assertEqual(args.phase,"attainable")
        self.assertEqual(args.comparison_report,"matched.json")


if __name__=="__main__":
    unittest.main()
