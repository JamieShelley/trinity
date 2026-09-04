from pathlib import Path

training_path = Path("tools/nsamdr/neural/v9/training.py")
source = training_path.read_text(encoding="utf-8")

old_dataset = '''        # V10.8.0 downstream appearance learning is deliberately real-Raven only.
        # Structural synthetic primitives remain useful for B1, but seam/detail/
        # selector stages must see the authored Raven crops they are expected to
        # improve in the instrumented preview.
        downstream_config = copy.deepcopy(config)
        downstream_config.synthetic_geometry_probability = 0.0
        downstream_train_dataset = PhysicalTileDatasetV9(
            manifest, downstream_config, "train",
            int(getattr(config, "raven_downstream_tiles_per_epoch", 24)),
            seed=config.seed + 80_813,
        )
        downstream_train_loader = self._build_loader(
            downstream_train_dataset,
            batch_size=config.batch_size, device=device, workers=workers,
            prefetch_factor=config.data_loader_prefetch_factor,
            persistent_workers=config.data_loader_persistent_workers,
            rolling_epoch_indices=True,
        )
'''
new_dataset = '''        # The deployable connected-spline structure must learn on the authored
        # domain it is judged on. Synthetic primitives remain representation and
        # topology audits, but they are not optimizer authority for production B1.
        authored_config = copy.deepcopy(config)
        authored_config.synthetic_geometry_probability = 0.0
        structural_train_dataset = PhysicalTileDatasetV9(
            manifest, authored_config, "train", config.tiles_per_epoch,
            seed=config.seed + 71_337,
        )
        structural_train_loader = self._build_loader(
            structural_train_dataset,
            batch_size=config.batch_size, device=device, workers=workers,
            prefetch_factor=config.data_loader_prefetch_factor,
            persistent_workers=config.data_loader_persistent_workers,
            rolling_epoch_indices=True,
        )
        # Downstream appearance/seam learning is likewise authored-Raven only,
        # but retains its independent smaller work budget.
        downstream_train_dataset = PhysicalTileDatasetV9(
            manifest, authored_config, "train",
            int(getattr(config, "raven_downstream_tiles_per_epoch", 24)),
            seed=config.seed + 80_813,
        )
        downstream_train_loader = self._build_loader(
            downstream_train_dataset,
            batch_size=config.batch_size, device=device, workers=workers,
            prefetch_factor=config.data_loader_prefetch_factor,
            persistent_workers=config.data_loader_persistent_workers,
            rolling_epoch_indices=True,
        )
'''
if source.count(old_dataset) != 1:
    raise RuntimeError(f"expected one downstream dataset block, found {source.count(old_dataset)}")
source = source.replace(old_dataset, new_dataset, 1)
source = source.replace('manifest, downstream_config, "train",', 'manifest, authored_config, "train",')

old_local_bank = '''        # V11.4 uses the fixed complete-teacher analytic curriculum at the
        # current structural epoch budget. Round up only to preserve complete
        # seven-class balance; the legacy compact B1b bank size is not a floor for
        # this full production graph. The permanent 29-case ladder remains held out.
        local_structure_train_tiles = (
            (int(config.tiles_per_epoch) + PRIMITIVE_COUNT - 1)
            // PRIMITIVE_COUNT
        ) * PRIMITIVE_COUNT
        local_structure_train_dataset = ParametricPrimitiveTrainingDataset(
            config, local_structure_train_tiles, seed=config.seed + 71_337
        )
        local_structure_train_loader = self._build_loader(
            local_structure_train_dataset,
            batch_size=config.batch_size,
            device=device,
            workers=workers,
            prefetch_factor=config.data_loader_prefetch_factor,
            persistent_workers=config.data_loader_persistent_workers,
            rolling_epoch_indices=False,
        )
'''
if source.count(old_local_bank) != 1:
    raise RuntimeError(f"expected one synthetic local structure bank, found {source.count(old_local_bank)}")
source = source.replace(old_local_bank, "", 1)

old_phase = '''            model.set_phase(phase)
            local_structure_phase = bool(
                phase == "sdf-proof"
                and hasattr(model.geometry_net, "production_structure")
            )
'''
new_phase = '''            model.set_phase(phase)
            production_structure_phase = bool(
                phase in {"sdf-bootstrap", "sdf-proof"}
                and hasattr(model.geometry_net, "production_structure")
            )
            local_structure_phase = bool(
                phase == "sdf-proof" and production_structure_phase
            )
'''
if source.count(old_phase) != 1:
    raise RuntimeError(f"expected one structural phase block, found {source.count(old_phase)}")
source = source.replace(old_phase, new_phase, 1)

old_loader = '''            if phase == "sdf-proof":
                # V11.4 local-boundary proof needs the fixed complete-teacher
                # analytic bank that its held-out B1/B2 ladder evaluates, but the
                # full production graph must stay at the canonical batch size. The
                # legacy compact field keeps its configured micro-batch loader.
                epoch_loader = (
                    local_structure_train_loader
                    if local_structure_phase
                    else parametric_train_loader
                )
            elif phase == "seam-proof":
'''
new_loader = '''            if production_structure_phase:
                # B1a and B1b optimize the connected-spline production path on
                # authored Raven crops. The fixed synthetic ladder remains a
                # separate held-out capability/topology audit below.
                epoch_loader = structural_train_loader
            elif phase == "sdf-proof":
                # Compatibility path for the retired compact primitive field.
                epoch_loader = parametric_train_loader
            elif phase == "seam-proof":
'''
if source.count(old_loader) != 1:
    raise RuntimeError(f"expected one structural loader selection block, found {source.count(old_loader)}")
source = source.replace(old_loader, new_loader, 1)

old_smoke = '''            # Quick's first connected-spline geometry epoch is a complete-class
            # smoke proof, not a promotion epoch. Two examples per primitive family
            # give an early A/B/C result before paying for the full 70-tile bank.
            # Full runs and every later B1b epoch retain the complete training bank.
            structural_smoke_batch_limit = None
            if (
                local_structure_phase
                and b1b_stage_epoch == 1
                and int(config.tiles_per_epoch) <= 64
            ):
                structural_smoke_batch_limit = min(
                    epoch_batch_count, max(PRIMITIVE_COUNT * 2, PRIMITIVE_COUNT)
                )
                epoch_batch_count = structural_smoke_batch_limit
                epoch_tile_count = structural_smoke_batch_limit * epoch_batch_size
                self._status(
                    f"  B1b QUICK SMOKE: {structural_smoke_batch_limit} batch(es) "
                    "(2/class) before the full connected-spline bank; promotion disabled."
                )
'''
new_smoke = '''            # Quick's first connected-spline refinement epoch is a bounded
            # authored-Raven smoke pass, not a promotion epoch. The permanent
            # synthetic ladder remains validation-only; Full and later B1b work
            # retain the complete authored structural bank.
            structural_smoke_batch_limit = None
            if (
                local_structure_phase
                and b1b_stage_epoch == 1
                and int(config.tiles_per_epoch) <= 64
            ):
                structural_smoke_batch_limit = min(epoch_batch_count, 14)
                epoch_batch_count = structural_smoke_batch_limit
                epoch_tile_count = structural_smoke_batch_limit * epoch_batch_size
                self._status(
                    f"  B1b QUICK SMOKE: {structural_smoke_batch_limit} authored Raven "
                    "batch(es); synthetic ladder remains validation-only; promotion disabled."
                )
'''
if source.count(old_smoke) != 1:
    raise RuntimeError(f"expected one Quick structural smoke block, found {source.count(old_smoke)}")
source = source.replace(old_smoke, new_smoke, 1)

old_status = '''                    self._status(
                        "  B1 local-boundary production proof: full production "
                        "geometry + same-renderer losses have authority; "
                        f"analytic complete-teacher bank={len(local_structure_train_dataset)} "
                        f"batch={int(local_structure_train_loader.batch_size or 1)}"
                    )
'''
new_status = '''                    self._status(
                        "  B1 local-boundary production proof: full production "
                        "geometry + same-renderer losses have authority; "
                        f"authored Raven structural bank={len(structural_train_dataset)} "
                        f"batch={int(structural_train_loader.batch_size or 1)}; "
                        "synthetic ladder=validation-only"
                    )
'''
if source.count(old_status) != 1:
    raise RuntimeError(f"expected one local structure status block, found {source.count(old_status)}")
source = source.replace(old_status, new_status, 1)
training_path.write_text(source, encoding="utf-8")

test_path = Path("tools/nsamdr/tests/test_v117_baseline_relative_contract.py")
test_source = test_path.read_text(encoding="utf-8")
old_test = '''def test_quick_first_b1b_is_balanced_smoke_and_cannot_promote():
    from v9.dataset import ParametricPrimitiveTrainingDataset
    from v9.training import TrainingService

    source = inspect.getsource(TrainingService.train_v9)
    dataset_source = inspect.getsource(ParametricPrimitiveTrainingDataset.__getitem__)
    assert 'forced_class=int(index) % PRIMITIVE_COUNT' in dataset_source
    assert 'b1b_stage_epoch == 1' in source
    assert 'int(config.tiles_per_epoch) <= 64' in source
    assert 'PRIMITIVE_COUNT * 2' in source
    assert 'B1b QUICK SMOKE' in source
    assert 'structural_smoke_epoch = structural_smoke_batch_limit is not None' in source
    assert 'not structural_smoke_epoch' in source
    assert 'B1/B2 promotion is disabled' in source
    assert 'not structural_smoke_epoch and integration_ready and hard_render_gate' in source
    # The complete bank is unshuffled, so indices 0..13 are exactly two of each
    # class because the dataset maps index modulo primitive count to class.
    assert 'rolling_epoch_indices=False' in source
    assert 'local_structure_train_loader' in source
'''
new_test = '''def test_connected_spline_b1_optimizes_real_raven_and_keeps_synthetic_audit():
    from v9.training import TrainingService

    source = inspect.getsource(TrainingService.train_v9)
    assert 'authored_config.synthetic_geometry_probability = 0.0' in source
    assert 'structural_train_dataset = PhysicalTileDatasetV9(' in source
    assert 'epoch_loader = structural_train_loader' in source
    assert 'phase in {"sdf-bootstrap", "sdf-proof"}' in source
    assert 'SyntheticGeometryValidationDataset(' in source
    assert 'local_structure_train_dataset = ParametricPrimitiveTrainingDataset(' not in source
    assert 'local_structure_train_loader' not in source


def test_quick_first_b1b_is_authored_raven_smoke_and_cannot_promote():
    from v9.training import TrainingService

    source = inspect.getsource(TrainingService.train_v9)
    assert 'b1b_stage_epoch == 1' in source
    assert 'int(config.tiles_per_epoch) <= 64' in source
    assert 'structural_smoke_batch_limit = min(epoch_batch_count, 14)' in source
    assert 'B1b QUICK SMOKE' in source
    assert 'authored Raven' in source
    assert 'synthetic ladder remains validation-only' in source
    assert 'structural_smoke_epoch = structural_smoke_batch_limit is not None' in source
    assert 'not structural_smoke_epoch' in source
    assert 'B1/B2 promotion is disabled' in source
    assert 'not structural_smoke_epoch and integration_ready and hard_render_gate' in source
'''
if test_source.count(old_test) != 1:
    raise RuntimeError(f"expected one old B1b smoke test, found {test_source.count(old_test)}")
test_path.write_text(test_source.replace(old_test, new_test, 1), encoding="utf-8")

design_path = Path("tools/nsamdr/NSAMDR_BASELINE_RELATIVE_DESIGN.md")
design = design_path.read_text(encoding="utf-8")
old_design = "The first Quick B1b epoch is a two-examples-per-primitive smoke pass. It is validation-only and cannot promote B1/B2 even if its held-out metrics happen to pass. If C is visibly/quantitatively worse than B, stop there. Later B1b epochs retain the complete training bank and all existing hard qualification gates."
new_design = "Production B1a/B1b optimization uses authored Raven crops with synthetic geometry disabled. Synthetic line/circle/ring cases remain representation and topology audits; they do not replace real-domain optimizer evidence. The first Quick B1b epoch is a bounded 14-batch authored-Raven structural refinement smoke pass. It cannot promote B1/B2 even if its held-out metrics happen to pass. If C is visibly/quantitatively worse than B, stop there. Later B1b epochs retain the complete authored structural bank and all existing hard qualification gates."
if design.count(old_design) != 1:
    raise RuntimeError(f"expected one old Quick feedback paragraph, found {design.count(old_design)}")
design_path.write_text(design.replace(old_design, new_design, 1), encoding="utf-8")

for cleanup in (
    Path("tools/nsamdr/_temp_real_raven_structure_patch.py"),
    Path(".github/workflows/nsamdr_v117_real_raven_structure_temp.yml"),
    Path(".github/workflows/nsamdr_v117_real_raven_structure_retry.yml"),
):
    cleanup.unlink(missing_ok=True)
