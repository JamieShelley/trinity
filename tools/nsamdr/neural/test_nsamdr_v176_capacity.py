"""Stage 10A.1 CPU regression suite: provenance, matched evidence, probes, resume."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from tools.nsamdr.neural import probe_nsamdr_v176_capacity as proof
from tools.nsamdr.neural.test_nsamdr_v17_model import _config
from tools.nsamdr.neural.v17.model import NSAMDRV17
from tools.nsamdr.neural.v176.detail_capacity import (
    HARD_TRAINING_INDICES, MODES, HRFeatureDetail, create_matched_variants,
    diagnostic_batch, gradient_statistics, tensor_statistics,
)
from tools.nsamdr.neural.v176.wavelet_supervision import detail_supervision_loss


def _backbone():
    return NSAMDRV17(_config(),encoder_channels=16,mid_channels=12,
                     detail_channels=48,decoder_blocks=1)


def _lr():
    torch.manual_seed(10)
    return torch.rand(1,3,8,8),torch.rand(1,2,8,8)*.2,torch.rand(1,3,8,8)


def _batch(model):
    a,n,m=_lr()
    with torch.no_grad():
        out=model.backbone(a,n,m)
        target=out["candidate_albedo"].clone()
        target[:,:,:,12:15]=(target[:,:,:,12:15]+.15).clamp(0,1)
    return {"lr_albedo":a,"lr_normal":n,"lr_material":m,
            "target_albedo":target}


class V176CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(torch.get_num_threads(),2))

    def test_locked_authorities_modes_and_stages(self):
        self.assertEqual(HARD_TRAINING_INDICES,(1,10))
        self.assertEqual(MODES,("output10","hr_features58"))
        self.assertEqual(proof.STAGES,(0,1536,3072,6144))
        self.assertEqual(proof.ALT_EXPOSURES,(20000,20001,20002,20003))
        self.assertEqual(proof.LEARNING_RATE,.0002)

    def test_parser_defaults_to_nondestructive_diagnostics(self):
        args=proof.parser().parse_args([])
        self.assertEqual(args.phase,"diagnose")
        self.assertFalse(args.audit_only)
        self.assertEqual(args.resume,"fresh")
        with self.assertRaises(SystemExit):
            proof.parser().parse_args(["--phase","unsafe"])

    def test_authenticated_feature_tap_extracts_frozen_48_channels(self):
        _,rich=create_matched_variants(_backbone())
        self.assertIsInstance(rich,HRFeatureDetail)
        self.assertEqual(rich.specialist.stem.in_channels,58)
        self.assertEqual(rich.specialist.head.out_channels,1)
        with torch.no_grad():
            out=rich(*_lr())
        self.assertEqual(tuple(out["fine_detail_raw"].shape),(1,1,32,32))
        self.assertIsNone(rich._feature)
        self.assertTrue(all(not p.requires_grad for p in rich.backbone.parameters()))

    def test_matched_models_have_identical_initial_candidate_pixels(self):
        plain,rich=create_matched_variants(_backbone(),seed=17611)
        with torch.no_grad():
            p=plain(*_lr())
            r=rich(*_lr())
        for key in ("candidate_albedo","candidate_normal","candidate_material",
                    "predicted_residual_albedo","fine_detail_raw"):
            self.assertTrue(torch.equal(p[key],r[key]),key)
        self.assertEqual(float(p["fine_detail_raw"].abs().max()),0.)

    def test_rich_common_stem_and_blocks_copied_exactly(self):
        plain,rich=create_matched_variants(_backbone())
        self.assertTrue(torch.equal(
            plain.specialist.stem.weight,
            rich.specialist.stem.weight[:,:10]))
        self.assertTrue(torch.count_nonzero(rich.specialist.stem.weight[:,10:])==0)
        for a,b in zip(plain.specialist.blocks.parameters(),
                       rich.specialist.blocks.parameters()):
            self.assertTrue(torch.equal(a,b))

    def test_feature_arm_gets_gradient_while_backbone_stays_frozen(self):
        plain,rich=create_matched_variants(_backbone())
        terms=detail_supervision_loss(rich(*_lr()),_batch(rich))
        terms["total"].backward()
        self.assertIsNotNone(rich.specialist.head.weight.grad)
        self.assertGreater(float(rich.specialist.head.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in rich.backbone.parameters()))
        self.assertGreater(gradient_statistics(rich)["totalL2"],0.)

    def test_diagnostic_has_band_losses_raw_highpass_grad_norms(self):
        plain,_=create_matched_variants(_backbone())
        result=diagnostic_batch(plain,_batch(plain))
        self.assertEqual(len(result["bandRelativeL1"]),6)
        self.assertIn("meanAbs",result["rawDetail"])
        self.assertIn("rms",result["highpassDetail"])
        self.assertIn("totalL2",result["gradient"])
        self.assertIn("head.weight",result["gradient"]["selectedParameterGradL2"])
        self.assertTrue(result["normalUnchanged"])
        self.assertTrue(all(p.grad is None for p in plain.backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in plain.specialist.parameters()))

    def test_diagnostics_never_change_specialist_or_backbone_weights(self):
        plain,_=create_matched_variants(_backbone())
        initial={name:value.detach().clone()
                 for name,value in plain.state_dict().items()}
        diagnostic_batch(plain,_batch(plain))
        self.assertTrue(all(torch.equal(value,plain.state_dict()[name])
                            for name,value in initial.items()))

    def test_training_step_updates_specialist_not_backbone(self):
        plain,_=create_matched_variants(_backbone())
        optimizer=torch.optim.Adam(plain.trainable_parameters(),lr=proof.LEARNING_RATE)
        before={name:value.detach().clone()
                for name,value in plain.backbone.state_dict().items()}
        head=plain.specialist.head.weight.detach().clone()
        metrics=proof._train_step(plain,optimizer,_batch(plain),
                                  torch.device("cpu"),"auto",collect=True)
        self.assertIn("totalL2",metrics["gradient"])
        self.assertFalse(torch.equal(head,plain.specialist.head.weight))
        self.assertTrue(all(torch.equal(value,plain.backbone.state_dict()[name])
                            for name,value in before.items()))

    def test_tensor_statistic_threshold_is_one_lsb(self):
        result=tensor_statistics(torch.tensor([[[[0.0,1.0/255,2/255]]]]))
        self.assertAlmostEqual(result["aboveOneLsbFraction"],2/3)

    def test_invalid_authority_index_or_ablation_rejected(self):
        with self.assertRaises(ValueError):
            proof._arm_name(0,"output10")
        with self.assertRaises(ValueError):
            proof._arm_name(1,"unknown")
        self.assertEqual(proof._arm_name(10,"hr_features58"),
                         "train_10_hr_features58")

    def test_checkpoint_roundtrip_restores_rng_optimizer_and_identity(self):
        plain,_=create_matched_variants(_backbone())
        opt=torch.optim.Adam(plain.trainable_parameters(),lr=.0002)
        original=copy.deepcopy(plain.specialist.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"stage_001536/checkpoint.pt"
            history=[{"update":0},{"update":1536}]
            torch.manual_seed(27)
            proof._save(path,plain,opt,{"cohort":"hash"},
                        "train_01_output10",1536,history)
            expected=torch.rand(4)
            with torch.no_grad():
                plain.specialist.stem.bias.add_(1)
            out=proof._restore(path,plain,opt,{"cohort":"hash"},
                               "train_01_output10",torch.device("cpu"))
            self.assertEqual(out["update"],1536)
            self.assertTrue(torch.equal(plain.specialist.stem.bias,
                                        original["stem.bias"]))
            self.assertTrue(torch.equal(torch.rand(4),expected))

    def test_checkpoint_rejects_cross_arm_and_bad_contract(self):
        plain,_=create_matched_variants(_backbone())
        opt=torch.optim.Adam(plain.trainable_parameters(),lr=.0002)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"checkpoint.pt"
            proof._save(path,plain,opt,{"f":"original"},"train_01_output10",
                        1536,[{"update":0},{"update":1536}])
            with self.assertRaisesRegex(RuntimeError,"mismatch"):
                proof._restore(path,plain,opt,{"f":"other"},
                               "train_01_output10",torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError,"mismatch"):
                proof._restore(path,plain,opt,{"f":"original"},
                               "train_10_output10",torch.device("cpu"))

    def test_outcome_is_training_only_and_requires_both_hard_authorities(self):
        def arm(index,mode,edge,lattice):
            return {"arm":proof._arm_name(index,mode),
                    "qualification":"TRAIN_REGIONS_UNQUALIFIED",
                    "history":[{"fixed":{"metrics":{"edge_recovery":edge,
                                                      "global_recovery":.35,
                                                      "lattice_cell_excess":lattice},
                                          "gateChecks":{"edge":False,"lattice":lattice<=.15}},
                                "alternateMeanEdge":edge,
                                "alternateMaxLattice":lattice}]}
        data={}
        for index in HARD_TRAINING_INDICES:
            data[proof._arm_name(index,"output10")]=arm(index,"output10",.35,.10)
            data[proof._arm_name(index,"hr_features58")]=arm(index,"hr_features58",.38,.10)
        result=proof._decision(data,"compare-features")
        self.assertEqual(result["decision"],"RICH_HR_FEATURE_SIGNAL_TRAIN_ONLY")
        data[proof._arm_name(10,"hr_features58")]["history"][-1]["alternateMaxLattice"]=.20
        self.assertEqual(proof._decision(data,"compare-features")["decision"],
                         "NO_PROVEN_HR_FEATURE_BENEFIT")

    def test_stage10a_authenticity_rejects_incomplete_proof(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p=root/"report.json"
            p.write_text('{"schema":"wrong"}')
            (root/"run_manifest.json").write_text("{}")
            (root/"provenance.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError,"lineage"):
                proof.verify_stage10a_report(p,{
                    "stage9BackboneTensorSha256":"a",
                    "stage9A12CheckpointSha256":"b",
                    "trainingAuthorityIds":[],"sealedFinalCropIds":[]})

    def test_phase_resume_refuses_ambiguous_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,"one unfinished"):
                proof._folder(Path(tmp),"auto","capacity")
            root=Path(tmp)/proof.ROOT
            for i in (1,2):
                folder=root/f"proof_{i}"
                folder.mkdir(parents=True)
                (folder/"run_manifest.json").write_text('{"phase":"capacity"}')
            with self.assertRaisesRegex(RuntimeError,"got 2"):
                proof._folder(Path(tmp),"auto","capacity")

    def test_training_source_only_no_dev_or_final_decoder(self):
        body=Path(proof.__file__).read_text(encoding="utf-8")
        self.assertNotIn('parts["validation"]',body)
        self.assertNotIn('parts["final_test"]',body)
        self.assertNotIn('role="final_test"',body)
        self.assertIn('selected_rows(selection,manifest,"A12")',body)
        self.assertIn("HARD_TRAINING_INDICES",body)


if __name__=="__main__":
    unittest.main()
