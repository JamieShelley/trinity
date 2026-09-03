from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one match in {path}: {old[:100]!r}; got {text.count(old)}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1. Baseline-only inference must be mathematically identical to the baseline
# computed inside FidelityResidualNetV9._forward_impl().
replace_once(
    "tools/nsamdr/neural/v9/inference.py",
    """            normal_length = torch.sqrt(\n                baseline_normal.square().sum(dim=1, keepdim=True) + 1.0e-8\n            )\n""",
    """            normal_length = torch.sqrt(\n                baseline_normal.square().sum(dim=1, keepdim=True) + 1.0e-6\n            )\n""",
)

# 2. Smoke validation is useful evidence, but it must never promote B1/B2.
replace_once(
    "tools/nsamdr/neural/v9/training.py",
    """                self._status(\n                    f\"  B1b QUICK SMOKE: {structural_smoke_batch_limit} batch(es) \"\n                    \"(2/class) before the full connected-spline bank.\"\n                )\n            epoch_workers = workers\n""",
    """                self._status(\n                    f\"  B1b QUICK SMOKE: {structural_smoke_batch_limit} batch(es) \"\n                    \"(2/class) before the full connected-spline bank; promotion disabled.\"\n                )\n            structural_smoke_epoch = structural_smoke_batch_limit is not None\n            epoch_workers = workers\n""",
)
replace_once(
    "tools/nsamdr/neural/v9/training.py",
    """                    integration_ready = bool(b1b_parameters_qualified)\n                    select_structure = integration_ready and structure_rank < best_structure_rank\n""",
    """                    integration_ready = bool(b1b_parameters_qualified)\n                    if structural_smoke_epoch:\n                        self._status(\n                            \"  B1b QUICK SMOKE validation only: B1/B2 promotion is disabled \"\n                            \"until a full-bank sdf-proof epoch.\"\n                        )\n                    select_structure = (\n                        not structural_smoke_epoch\n                        and integration_ready\n                        and structure_rank < best_structure_rank\n                    )\n""",
)
replace_once(
    "tools/nsamdr/neural/v9/training.py",
    """                    if integration_ready and hard_structure_gate and not structure_qualified:\n""",
    """                    if (\n                        not structural_smoke_epoch\n                        and integration_ready\n                        and hard_structure_gate\n                        and not structure_qualified\n                    ):\n""",
)
replace_once(
    "tools/nsamdr/neural/v9/training.py",
    """                    if integration_ready and hard_render_gate:\n""",
    """                    if not structural_smoke_epoch and integration_ready and hard_render_gate:\n""",
)

# 3. Make baseline provenance explicit instead of labelling it as a learned
# training-intermediate physical candidate.
replace_once(
    "tools/nsamdr/neural/live_preview_nsamdr_v9_training.py",
    '"""Live EVE A/B preview of completed, unqualified NSAMDR training epochs."""',
    '"""Live EVE A/B/C preview of deterministic baseline and completed NSAMDR stages."""',
)
replace_once(
    "tools/nsamdr/neural/live_preview_nsamdr_v9_training.py",
    """        replacements: Mapping[Path, Path],\n        source_dir: Path,\n    ) -> None:\n""",
    """        replacements: Mapping[Path, Path],\n        source_dir: Path,\n        physical_candidate: str = \"NSAMDR_TRAINING_INTERMEDIATE_UNQUALIFIED\",\n    ) -> None:\n""",
)
replace_once(
    "tools/nsamdr/neural/live_preview_nsamdr_v9_training.py",
    '            handle.write("# PHYSICAL_CANDIDATE NSAMDR_TRAINING_INTERMEDIATE_UNQUALIFIED\\n")\n',
    '            handle.write(f"# PHYSICAL_CANDIDATE {physical_candidate}\\n")\n',
)
replace_once(
    "tools/nsamdr/neural/live_preview_nsamdr_v9_training.py",
    """        self._write_material_manifest(\n            helper, baseline_materials, fields, rows, comments,\n            baseline_replacements, materials.parent,\n        )\n""",
    """        self._write_material_manifest(\n            helper, baseline_materials, fields, rows, comments,\n            baseline_replacements, materials.parent,\n            physical_candidate=\"NSAMDR_DETERMINISTIC_4X_BASELINE\",\n        )\n""",
)

replace_once(
    "trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp",
    """    // Scientific comparison is deliberately only two panes:\n    // A) untouched source, and B) the NSAMDR candidate. Both use the exact\n    // same high-quality sampler, camera, lighting, geometry and material shader.\n""",
    """    // Scientific comparison is A authored source / B deterministic 4x baseline /\n    // C current learned stage during live training, while immutable final preview\n    // remains A source / C final. Every pane uses the same sampler, camera,\n    // lighting, geometry and material shader.\n""",
)

# 4. Strengthen V11.7 regression tests with numerical baseline equivalence,
# class-balance proof, and explicit no-promotion checks.
test_path = ROOT / "tools/nsamdr/tests/test_v117_baseline_relative_contract.py"
test_text = test_path.read_text(encoding="utf-8")
old_test = '''def test_baseline_variant_is_deterministic_and_does_not_call_model_forward():\n    from v9.inference import infer_tiled\n\n    class NeverForward(torch.nn.Module):\n        def __init__(self) -> None:\n            super().__init__()\n            self.config = SimpleNamespace(\n                channels_last=False,\n                amp_dtype="auto",\n                appearance_enabled=True,\n                detail_reconstruction_enabled=True,\n            )\n\n        def forward(self, _value):  # pragma: no cover - must never execute\n            raise AssertionError("baseline variant called model.forward")\n\n    value = np.zeros((17, 12, 10), dtype=np.float32)\n    value[0:3] = 0.37\n    value[3] = 0.25\n    value[4] = -0.15\n    value[5] = 0.2\n    value[6] = 0.4\n    value[7] = 0.7\n    maps, diagnostics = infer_tiled(\n        NeverForward(), value, "cpu", return_diagnostics=True,\n        return_all_maps=True, output_variant="baseline",\n    )\n    assert maps["albedo"].shape == (48, 40, 3)\n    assert np.allclose(maps["albedo"], 0.37, atol=1.0e-5)\n    assert np.allclose(maps["material"][..., 0], 0.2, atol=1.0e-6)\n    assert diagnostics["outputVariant"] == "baseline"\n    assert diagnostics["tileCount"] == 0\n'''
new_test = '''def test_baseline_variant_is_exact_model_baseline_and_does_not_call_model_forward():\n    from torch.nn import functional as F\n    from v9.inference import infer_tiled\n    from v9.model import FidelityResidualNetV9, UPSCALE_FACTOR\n\n    class NeverForward(torch.nn.Module):\n        def __init__(self) -> None:\n            super().__init__()\n            self.config = SimpleNamespace(\n                channels_last=False,\n                amp_dtype="auto",\n                appearance_enabled=True,\n                detail_reconstruction_enabled=True,\n            )\n\n        def forward(self, _value):  # pragma: no cover - must never execute\n            raise AssertionError("baseline variant called model.forward")\n\n    rng = np.random.default_rng(117)\n    value = rng.uniform(0.0, 1.0, (17, 12, 10)).astype(np.float32)\n    # Exercise the normalization limiter; the old 1e-8 baseline epsilon differed\n    # measurably from the model's canonical 1e-6 implementation here.\n    value[3] = 0.8\n    value[4] = 0.6\n    value[5:8] = rng.uniform(0.0, 1.0, (3, 12, 10)).astype(np.float32)\n\n    maps, diagnostics = infer_tiled(\n        NeverForward(), value, "cpu", return_diagnostics=True,\n        return_all_maps=True, output_variant="baseline",\n    )\n\n    source = torch.from_numpy(value).unsqueeze(0)\n    expected_albedo = F.interpolate(\n        source[:, 0:3].clamp(0.0, 1.0), scale_factor=UPSCALE_FACTOR,\n        mode="bicubic", align_corners=False, antialias=True,\n    ).clamp(0.0, 1.0)\n    expected_normal = FidelityResidualNetV9._normalize_xy(F.interpolate(\n        source[:, 3:5].clamp(-1.0, 1.0), scale_factor=UPSCALE_FACTOR,\n        mode="bilinear", align_corners=False,\n    ))\n    expected_material = F.interpolate(\n        source[:, 5:8].clamp(0.0, 1.0), scale_factor=UPSCALE_FACTOR, mode="nearest"\n    )\n\n    def nhwc(tensor):\n        return tensor[0].permute(1, 2, 0).numpy()\n\n    assert np.array_equal(maps["albedo"], nhwc(expected_albedo))\n    assert np.array_equal(maps["normal_xy"], nhwc(expected_normal))\n    assert np.array_equal(maps["material"], nhwc(expected_material))\n    assert diagnostics["outputVariant"] == "baseline"\n    assert diagnostics["candidateAuthority"] == "deterministic-4x-baseline"\n    assert diagnostics["tileCount"] == 0\n    assert "model.forward not called" in diagnostics["productionForward"]\n'''
if test_text.count(old_test) != 1:
    raise RuntimeError("expected original baseline V11.7 test exactly once")
test_text = test_text.replace(old_test, new_test, 1)

old_smoke = '''def test_quick_first_b1b_is_small_complete_class_smoke_only():\n    from v9.training import TrainingService\n\n    source = inspect.getsource(TrainingService.train_v9)\n    assert 'b1b_stage_epoch == 1' in source\n    assert 'int(config.tiles_per_epoch) <= 64' in source\n    assert 'PRIMITIVE_COUNT * 2' in source\n    assert 'B1b QUICK SMOKE' in source\n    assert 'structural_smoke_batch_limit' in source\n    # Later epochs/full runs still use the canonical complete loader.\n    assert 'local_structure_train_loader' in source\n'''
new_smoke = '''def test_quick_first_b1b_is_balanced_smoke_and_cannot_promote():\n    from v9.dataset import ParametricPrimitiveTrainingDataset\n    from v9.training import TrainingService\n\n    source = inspect.getsource(TrainingService.train_v9)\n    dataset_source = inspect.getsource(ParametricPrimitiveTrainingDataset.__getitem__)\n    assert 'forced_class=int(index) % PRIMITIVE_COUNT' in dataset_source\n    assert 'b1b_stage_epoch == 1' in source\n    assert 'int(config.tiles_per_epoch) <= 64' in source\n    assert 'PRIMITIVE_COUNT * 2' in source\n    assert 'B1b QUICK SMOKE' in source\n    assert 'structural_smoke_epoch = structural_smoke_batch_limit is not None' in source\n    assert 'not structural_smoke_epoch' in source\n    assert 'B1/B2 promotion is disabled' in source\n    assert 'not structural_smoke_epoch and integration_ready and hard_render_gate' in source\n    # The complete bank is unshuffled, so indices 0..13 are exactly two of each\n    # class because the dataset maps index modulo primitive count to class.\n    assert 'rolling_epoch_indices=False' in source\n    assert 'local_structure_train_loader' in source\n'''
if test_text.count(old_smoke) != 1:
    raise RuntimeError("expected original smoke V11.7 test exactly once")
test_text = test_text.replace(old_smoke, new_smoke, 1)

test_text = test_text.replace(
    "    assert 'output_variant=stage_variant' in live\n",
    "    assert 'output_variant=stage_variant' in live\n    assert 'NSAMDR_DETERMINISTIC_4X_BASELINE' in live\n",
    1,
)
test_path.write_text(test_text, encoding="utf-8")

# Clarify the design doc contract now that promotion is mechanically blocked.
replace_once(
    "tools/nsamdr/NSAMDR_BASELINE_RELATIVE_DESIGN.md",
    "The first Quick B1b epoch is a two-examples-per-primitive smoke pass. It is not a promotion proof. If C is visibly/quantitatively worse than B, stop there. Later B1b epochs retain the complete training bank and all existing hard qualification gates.",
    "The first Quick B1b epoch is a two-examples-per-primitive smoke pass. It is validation-only and cannot promote B1/B2 even if its held-out metrics happen to pass. If C is visibly/quantitatively worse than B, stop there. Later B1b epochs retain the complete training bank and all existing hard qualification gates.",
)

print("V11.7 baseline-relative corrections applied")
