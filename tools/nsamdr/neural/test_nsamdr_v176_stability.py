"""Numerical stability regressions without authored development/test pixels."""
from __future__ import annotations

from contextlib import nullcontext
import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from tools.nsamdr.neural import probe_nsamdr_v176_capacity as proof
from tools.nsamdr.neural.test_nsamdr_v176_capacity import _backbone, _batch
from tools.nsamdr.neural.test_nsamdr_v176_detail import _lr, _model
from tools.nsamdr.neural.v176.detail_capacity import create_matched_variants
from tools.nsamdr.neural.v176.physical_composition import compose_albedo_detail, gaussian_highpass
from tools.nsamdr.neural.v176.stability import (
    NumericalStabilityError, STABILITY_POLICY, guarded_train_step, tensor_statistics,
)
from tools.nsamdr.neural.v176.wavelet_supervision import detail_supervision_loss, stationary_haar_details


class V176StabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(torch.get_num_threads(), 2))

    def test_constant_highpass_and_gradient_ignore_enclosing_autocast(self):
        for value in (0.25, 1.234567, 16.0):
            with self.subTest(value=value):
                image = torch.full((1, 1, 16, 16), value, requires_grad=True)
                expected = gaussian_highpass(image)
                gradient = torch.autograd.grad(expected.abs().mean(), image)[0]
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    actual = gaussian_highpass(image)
                    actual_gradient = torch.autograd.grad(actual.abs().mean(), image)[0]
                self.assertEqual(actual.dtype, torch.float32)
                self.assertTrue(torch.equal(expected, actual))
                self.assertTrue(torch.equal(gradient, actual_gradient))
                self.assertLess(float(actual.detach().abs().max()), 1e-5)

    def test_wavelet_bands_and_gradients_ignore_enclosing_autocast(self):
        image = torch.rand(1, 1, 16, 16, requires_grad=True)
        expected = stationary_haar_details(image)
        gradient = torch.autograd.grad(sum(v.square().mean() for v in expected.values()), image)[0]
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = stationary_haar_details(image)
            actual_gradient = torch.autograd.grad(sum(v.square().mean() for v in actual.values()), image)[0]
        for name in expected:
            self.assertEqual(actual[name].dtype, torch.float32)
            self.assertTrue(torch.equal(expected[name], actual[name]))
        self.assertTrue(torch.equal(gradient, actual_gradient))

    def test_fixed_filters_disable_tf32_and_restore_backend_settings(self):
        before = (torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
                  torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32)
        convolution = torch.nn.functional.conv2d
        observed = []
        def record(*args, **kwargs):
            observed.append(torch.backends.cudnn.allow_tf32)
            return convolution(*args, **kwargs)
        with patch("torch.nn.functional.conv2d", side_effect=record):
            image = torch.rand(1, 1, 16, 16)
            gaussian_highpass(image)
            stationary_haar_details(image)
        self.assertTrue(observed)
        self.assertFalse(any(observed))
        after = (torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
                 torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32)
        self.assertEqual(before, after)

    def test_giant_finite_and_nonfinite_raw_outputs_fail_before_filter(self):
        for value in (1e12, 1e17, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(NumericalStabilityError) as caught:
                    gaussian_highpass(torch.full((1, 1, 16, 16), value))
                self.assertEqual(caught.exception.diagnostic["tensor"], "fine_detail_raw")
                self.assertIn("fine_detail_raw", caught.exception.tensors)

    def test_excessive_finite_highpass_fails_after_filter(self):
        image = torch.zeros(1, 1, 16, 16)
        image[..., ::2] = 32
        image[..., 1::2] = -32
        with self.assertRaises(NumericalStabilityError) as caught:
            gaussian_highpass(image)
        self.assertEqual(caught.exception.diagnostic["tensor"], "fine_detail_highpass")
        self.assertIn("fine_detail_raw", caught.exception.tensors)

    def test_compositor_rejects_giant_correction_before_final_tanh(self):
        model = _model()
        with torch.no_grad():
            original = model.backbone(*_lr())
        for value in (1e12, 1e17):
            with self.subTest(value=value), self.assertRaises(NumericalStabilityError):
                compose_albedo_detail(original, torch.full((1, 1, 32, 32), value),
                                      albedo_residual_cap=model.backbone.config.albedo_residual_cap)

    def test_zero_identity_and_physical_caps_with_autocast(self):
        for model in create_matched_variants(_backbone()):
            with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
                original = model.backbone(*_lr())
                result = model(*_lr())
            for name in ("candidate_albedo", "candidate_normal", "candidate_material"):
                self.assertTrue(torch.equal(original[name], result[name]), name)
            correction = torch.full((1, 1, 32, 32), 3.5)
            correction[..., 1::2] *= -1
            bounded = compose_albedo_detail(original, correction,
                                           albedo_residual_cap=model.backbone.config.albedo_residual_cap)
            self.assertLessEqual(float(bounded["predicted_residual_albedo"].abs().max()),
                                 model.backbone.config.albedo_residual_cap + 1e-6)
            self.assertGreaterEqual(float(bounded["candidate_albedo"].min()), 0)
            self.assertLessEqual(float(bounded["candidate_albedo"].max()), 1)
            self.assertTrue(torch.equal(bounded["candidate_normal"], original["candidate_normal"]))
            self.assertTrue(torch.equal(bounded["candidate_material"], original["candidate_material"]))

    def test_one_pixel_detail_and_gradient_remain_recoverable(self):
        image = torch.zeros(1, 1, 24, 24, requires_grad=True)
        with torch.no_grad():
            image[..., 12, 12] = 0.1
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = gaussian_highpass(image)
        self.assertGreater(float(result[..., 12, 12].detach()), 0.08)
        result[..., 12, 12].sum().backward()
        self.assertGreater(float(image.grad[..., 12, 12]), 0.8)
        self.assertTrue(bool(torch.isfinite(image.grad).all()))

    def test_perfect_target_has_zero_wavelet_loss_under_autocast(self):
        model = _model()
        outputs = model(*_lr())
        with torch.autocast("cpu", dtype=torch.bfloat16):
            terms = detail_supervision_loss(outputs, {"target_albedo": outputs["candidate_albedo"].detach()})
        self.assertEqual(float(terms["wavelet"].detach()), 0)
        self.assertEqual(float(terms["total"].detach()), 0)

    def test_giant_finite_gradient_stops_without_optimizer_update_and_saves_evidence(self):
        model, _ = create_matched_variants(_backbone())
        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=proof.LEARNING_RATE)
        before = copy.deepcopy(model.specialist.state_dict())
        handle = model.specialist.head.weight.register_hook(lambda gradient: gradient + 1e12)
        context = {"update": 1793, "authorityId": "training-source", "cropId": "original-crop",
                   "sourceRegion": {"origin": [8, 12]}}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "failure_tensors.pt"
                with self.assertRaises(NumericalStabilityError) as caught:
                    proof._train_step(model, optimizer, _batch(model), torch.device("cpu"), "auto",
                                      collect=False, failure_path=path, context=context)
                self.assertEqual(caught.exception.diagnostic["tensor"], "gradient/totalL2")
                saved = torch.load(path, weights_only=True)
                self.assertEqual(saved["context"], context)
                self.assertIn("gradient/head.weight", saved["tensors"])
                self.assertIn("batch/target_albedo", saved["tensors"])
                self.assertFalse(list(Path(tmp).rglob("checkpoint.pt")))
        finally:
            handle.remove()
        self.assertEqual(len(optimizer.state), 0)
        self.assertTrue(all(torch.equal(value, model.specialist.state_dict()[name])
                            for name, value in before.items()))

    def test_nonfinite_gradient_is_not_silently_clipped(self):
        model, _ = create_matched_variants(_backbone())
        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=proof.LEARNING_RATE)
        handle = model.specialist.head.weight.register_hook(lambda gradient: gradient * float("nan"))
        try:
            with self.assertRaises(NumericalStabilityError):
                guarded_train_step(model, optimizer, _batch(model), nullcontext())
        finally:
            handle.remove()
        self.assertEqual(len(optimizer.state), 0)

    def test_corrupt_optimizer_moment_fails_before_another_step(self):
        model, _ = create_matched_variants(_backbone())
        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=proof.LEARNING_RATE)
        batch = _batch(model)
        guarded_train_step(model, optimizer, batch, nullcontext())
        next(iter(optimizer.state.values()))["exp_avg_sq"].fill_(1e12)
        with patch.object(optimizer, "step", wraps=optimizer.step) as step:
            with self.assertRaisesRegex(NumericalStabilityError, "optimizer/"):
                guarded_train_step(model, optimizer, batch, nullcontext())
        step.assert_not_called()

    def test_corrupt_post_step_parameters_fail_before_checkpoint(self):
        model, _ = create_matched_variants(_backbone())
        optimizer = torch.optim.Adam(model.trainable_parameters(), lr=proof.LEARNING_RATE)
        batch = _batch(model)
        actual_step = optimizer.step
        def bad_step():
            actual_step()
            with torch.no_grad():
                model.specialist.head.bias.fill_(1e12)
        with patch.object(optimizer, "step", side_effect=bad_step):
            with self.assertRaisesRegex(NumericalStabilityError, "parameter/head.bias"):
                guarded_train_step(model, optimizer, batch, nullcontext())

    def test_large_finite_diagnostic_rms_does_not_overflow(self):
        stats = tensor_statistics(torch.full((1, 1, 8, 8), 1e20))
        self.assertTrue(math.isfinite(stats["rms"]))
        self.assertAlmostEqual(stats["rms"] / 1e20, 1, places=6)

    def test_stability_trial_covers_reported_failure_interval(self):
        self.assertEqual(proof.parser().parse_args(["--phase", "stability"]).phase, "stability")
        self.assertEqual(proof.STABILITY_STAGES[-1], 3072)
        self.assertIn(2048, proof.STABILITY_STAGES)
        self.assertIn(2816, proof.STABILITY_STAGES)
        self.assertEqual(proof.STAGES[-1], 6144)
        self.assertFalse(STABILITY_POLICY["physicalCapsChanged"])
        self.assertFalse(STABILITY_POLICY["correctionClipped"])

    def test_snapshots_never_acquire_future_telemetry(self):
        model, other = create_matched_variants(_backbone())
        batch = _batch(model)
        batch["target_normal"] = torch.zeros(1, 2, 32, 32)
        batch["target_material"] = torch.zeros(1, 3, 32, 32)
        sampler = Mock()
        sampler.sample.return_value = (batch, {"origin": [0, 0]})
        contract = {"parentBackboneTensorSha256": "hash", "stages": [0, 1, 2]}
        def snapshot(*args):
            update = args[-1]
            return {"update": update, "fixed": {"metrics": {"edge_recovery": 0.1},
                    "gateChecks": {"edge": False, "lattice": False}},
                    "alternateRegions": [], "alternateMeanEdge": 0.1}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(proof, "_samples", return_value=(batch, [], [])), \
             patch.object(proof, "_tensor_sha", return_value="hash"), \
             patch.object(proof, "_snapshot", side_effect=snapshot), \
             patch.object(proof, "_train_step", return_value={"gradient": {"totalL2": 0.1}}):
            result = proof._run_arm(Path(tmp), (model, other), contract, "training-source", 1,
                                    "output10", sampler, None, torch.device("cpu"), "auto")
        for entry in result["history"]:
            self.assertTrue(all(row["update"] <= entry["update"]
                                for row in entry.get("recentTrainGradients", [])))

    def test_failed_proof_is_not_automatically_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / proof.ROOT / "proof_failed"
            folder.mkdir(parents=True)
            (folder / "run_manifest.json").write_text('{"phase":"stability"}')
            (folder / "failure.json").write_text('{"decision":"BLOCKED_NUMERICAL_STABILITY"}')
            with self.assertRaisesRegex(RuntimeError, "got 0"):
                proof._folder(root, "auto", "stability")

    def test_interrupted_proof_can_resume_without_reusing_failed_numerical_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / proof.ROOT / "proof_interrupted"
            folder.mkdir(parents=True)
            (folder / "run_manifest.json").write_text('{"phase":"stability"}')
            (folder / "failure.json").write_text('{"decision":"BLOCKED","reason":"KeyboardInterrupt"}')
            self.assertEqual(proof._folder(root, "auto", "stability"), folder)


if __name__ == "__main__":
    unittest.main()
