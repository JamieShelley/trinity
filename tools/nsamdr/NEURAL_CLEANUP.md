# NSAMDR neural cleanup — 30 October 2026

## Verified remote scope

Remote `NSAMDR` base: `7f4c0b3a7348f48076b1d06a8379661498352407`.
Cleanup commits build on isolated `cleanup/nsamdr-cache-audit-20261030`.
The Windows checkout was not reachable; only GitHub remote tracked files were changed.

## Deleted from tracked neural root

These V16/V17 standalone historical experiments and their dedicated regression tests were removed from current HEAD. Their source remains recoverable at the original base commit (no Git history rewrite).

- `tools/nsamdr/neural/probe_nsamdr_v16_augmented_two_crop_fit.py`
- `tools/nsamdr/neural/probe_nsamdr_v16_heldout_transfer.py`
- `tools/nsamdr/neural/probe_nsamdr_v16_interference.py`
- `tools/nsamdr/neural/probe_nsamdr_v16_pyramid_context.py`
- `tools/nsamdr/neural/probe_nsamdr_v16_sibling_crop_transfer.py`
- `tools/nsamdr/neural/probe_nsamdr_v16_structure_conditioning.py`
- `tools/nsamdr/neural/probe_nsamdr_v16_two_crop_fit.py`
- `tools/nsamdr/neural/test_nsamdr_v16_augmented_two_crop_fit_probe.py`
- `tools/nsamdr/neural/test_nsamdr_v16_heldout_transfer_probe.py`
- `tools/nsamdr/neural/test_nsamdr_v16_interference_probe.py`
- `tools/nsamdr/neural/test_nsamdr_v16_sibling_crop_transfer_probe.py`
- `tools/nsamdr/neural/test_nsamdr_v16_structure_probe_ladder.py`
- `tools/nsamdr/neural/test_nsamdr_v16_two_crop_fit_probe.py`
- `tools/nsamdr/neural/probe_nsamdr_v17_4096_visual.py`
- `tools/nsamdr/neural/probe_nsamdr_v17_5_matched_continuation.py`
- `tools/nsamdr/neural/probe_nsamdr_v17_source_readiness.py`
- `tools/nsamdr/neural/test_nsamdr_v17_4096_visual.py`
- `tools/nsamdr/neural/test_nsamdr_v17_5_matched_continuation.py`
- `tools/nsamdr/neural/test_nsamdr_v17_source_readiness.py`

## Dependency review

- CLI (`nsamdr_cli.py`), full GUI, Windows CI workflow and README contain no references to these files.
- Active V17.6 training still imports functions from `probe_nsamdr_v16_full_broad.py`, `probe_nsamdr_v16_memorization.py`, `probe_nsamdr_v17_comprehensive.py`, `probe_nsamdr_v17_generalization.py`, `probe_nsamdr_v17_generalization_v2.py`, `probe_nsamdr_v17_sibling_architecture.py` and `probe_nsamdr_v17_5_diversity.py`; all remain intact.
- Historical `train_nsamdr_v16_main.py` and its tests still depend on `probe_nsamdr_v16_full_broad.py`; it is retained.
- Remaining V16/V17 module and V17.6/17.7/17.8 test imports were checked for the removed modules; none found outside the removed paired tests.
- Active V5 training source-hash list names only V17.7 synthetic geometry, V178 contour model and V5 training files, all unchanged.
- Frozen V17.6 evidence, archived checkpoints and generated outputs were not altered.
- Tracked `gr2_converter/node_modules` is not removable safely: it contains a private/source runtime dependency; preserve it until the converter is packaged and validated.

## Additional safe local cleanup

`scripts/maintenance/cleanup_nsamdr_caches.py` is a dry-run-first tool for ignored Python caches. It is optional and is **not** needed for this tracked cleanup.

## Verification and limits

- GitHub compare/tree inspection confirms the deleted paths and unchanged active code.
- Static import/reference review is not a substitute for running the Windows CLI/GUI, DX11, CUDA or complete regression suites.
- Hosted GitHub Actions is disabled. The unreachable local Windows working tree may contain additional untracked files; no such files were touched.
- Production/default model is still V5 and all V6–V8 candidates remain unqualified.

Historical experiments may be recovered from `7f4c0b3a` if a specific forensic re-run needs them. No change to `main`.
