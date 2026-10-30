# V178 contrast-transfer iteration — 30 October 2026

V5 repairs the measured RGB-inversion failure but does **not** satisfy the
README/EXAMPLE visual target. Original-polarity precision and IoU regress;
authentic Raven and another asset still contain fragmented thin contours.
Physical-map reconstruction and native-render qualification remain outstanding.

Implementation: branch `NSAMDR`, commit
`e56a5731aa31f13bbcaf0124eaa2f8ce1dcf7c6b`, based on
`92afe9cd1f76965ca4b717ae0ca920092e52e5b5`. The Windows checkout at
`D:\REPOS\trinity` was used directly. No cloud checkout, reset, force-push,
main merge, historical checkpoint replacement or V17.6 proof change occurred.

## Diagnosis and authenticated baseline

The seven registered Golden Patch assets and five original native source-map
hashes match `evidence.json`. The transferred `golden_patch_v4_evidence.zip`
contains native evidence but an old, blank V2 seed-17812 preview and no
checkpoint. Existing seed-17802/V1, seed-17812/V2 and seed-17822/V3 checkpoints
match their report hashes but are stale against V4 sources.

No completed V4 seed-17832 checkpoint existed. One 400-update GPU run through
the existing CLI created the baseline before implementation edits. Its raw
source files and hashes were frozen under
`artifacts/nsamdr/diagnostics/v178_contrast_transfer/v4_source_92afe9c/`.
The old V2 preview remains untouched.

The previous executor's V4 CPU-trained weights were not present. This local
baseline was trained on CUDA, so its CPU-evaluated native mean (0.128251) is not
numerically identical to the previously reported 0.122787. The controlled
inversion failure and exact CPU dense/tiled agreement reproduce; weight-level
or metric-level identity with that unavailable checkpoint is not claimed.

| Identity | V4 baseline | V5 experiment |
| --- | --- | --- |
| Seed / updates / batch / native patch | 17832 / 400 / 4 / 32 | 17842 / 400 / 4 / 32 |
| Checkpoint schema suffix | `V4_AUXILIARY_INDEPENDENT` | `V5_RGB_CONTRAST_BALANCED` |
| Checkpoint SHA256 | `09feeec945ea486a3f58f158e2d851a746fbab9820798368dfcda71f15d34864` | `e12ddf6d0ff351838672d80495160dfc6a4caf5fd052758ec7f3d54e1986c4ef` |
| Training source SHA256 | `45c4fb1c0d7eecb0f2a1cc37421b94e95903074ffa9af07b6a15de76fd4ca406` | `e7071000bc5451614139cd864fde7f64a681fceba024f30cc87451d813bd283d` |
| Existing synthetic gate | Pass, synthetic only | Pass, synthetic only |

Checkpoints, resume states and training reports are preserved in the respective
`artifacts/nsamdr/diagnostics/v178_golden_contour/seed_*_patch_32_batch_4/`
directories. Neither valid candidate was retrained. The separate two-update
CPU smoke candidate, seed 812244, failed closed and is also preserved.

## Implementation and exact change surface

- `neural/train_nsamdr_v178_contour.py:56`: training-only, independently seeded
  per-sample 50% `RGB = 1 - RGB`. The 400-update experiment inverted 798 of
  1600 samples. Normals, original edge observations, analytic target coverage
  and V4's independent auxiliary fields are preserved exactly. Evaluation
  stays unaugmented by default; extra inverted evaluation is diagnostic.
  Report/configuration record the RGB policy and new schema/source identity.
- `neural/v178/contour_model.py:12`: new V5 checkpoint schema only. Network,
  geometric/foreground loss and numerical behavior are otherwise unchanged.
  The shared `v177/synthetic_geometry.py` was not changed.
- `neural/benchmark_nsamdr_v178_contrast.py`: fixed original/inverted cases,
  checkpoint/report/source authentication, tensor identities, original native
  source-map rehashing, four-region inference, dense/tiled comparisons and
  hashed PNG/float outputs. It refuses to overwrite an output directory.
- `neural/test_nsamdr_v178_golden.py:206` and
  `neural/test_nsamdr_v178_contrast.py`: deterministic augmentation, global-RNG
  independence, unchanged geometry/observations, fixed-case registration,
  checkpoint/source tamper rejection and multi-region parsing.
- `nsamdr_cli.py`, `gui/nsamdr_v176_workflow_gui.py`,
  `neural/preview_nsamdr_v178_automatic.py`, `neural/v178/auto_preview.py`:
  consistent seed 17842, checkpoint paths and stale-schema guidance. Existing
  source/provenance checks remain enforced. CLI/GUI tests and automatic-preview
  tests verify defaults and rejection of old V4 candidates.
- `.github/workflows/nsamdr-v176-operator-ci.yml`: compile/test coverage and
  path triggers for the diagnostic. `README.md` and `GOLDEN_PATCH.md` describe
  the new contract and retain the V4 correction/history.

## Fixed contrast experiment

The primary 12 cases are exactly `_batch(1, 32,
17822 + 1000000 + 1723*i, device)`, `i=0..11`, with V4's unaugmented input
semantics frozen in the diagnostic. Two independent 12-case groups begin at
2017832 and 3017832. All 36 tensor identities match between V4/V5 reports;
all registered seeds are disjoint from both training sequences. Only RGB is
inverted: normals, edge observations, independent auxiliaries and targets stay
fixed. Recomputed edge magnitudes differ by at most `3.576279e-7`.

| Fixed group / polarity | Recall V4 → V5 | Precision V4 → V5 | IoU V4 → V5 |
| --- | --- | --- | --- |
| Primary / original | 80.27% → 85.04% | 76.57% → 63.97% | 65.79% → 56.99% |
| Primary / inverted | 1.16% → 84.25% | 0.50% → 65.09% | 0.35% → 57.87% |
| Independent A / original | 84.32% → 87.65% | 80.82% → 67.38% | 71.29% → 61.64% |
| Independent A / inverted | 1.24% → 85.90% | 1.43% → 67.73% | 0.66% → 61.25% |
| Independent B / original | 66.78% → 74.43% | 68.54% → 59.51% | 54.54% → 52.52% |
| Independent B / inverted | 1.37% → 73.05% | 0.83% → 60.09% | 0.52% → 52.14% |

Reports: `v178_contrast_transfer/v4_seed17832/report.json` and
`v5_seed17842/report.json` under the diagnostics root. The original-polarity
IoU/precision regressions are retained; V5 does not dominate V4.

The within-model RGB-only intervention establishes V4 photometric sensitivity.
Full inversion changes both brightness and contrast polarity. The required
new V5 seed also changes initialization and training geometry, so this is
supporting evidence for the augmentation, not a matched-seed causal ablation
or an explanation of every real artifact. The preserved synthetic gate uses
original polarity; its pass alone says nothing about inverted robustness.

## Authentic native results and numerical equivalence

Each input below is an authenticated native 256×256 region, producing
1024×1024 contours. Native auxiliary crops retain their own dimensions, with
explicit UV alignment. None entered training.

| Asset / native ROI | Candidate mean V4 → V5 | Active fraction V4 → V5 |
| --- | --- | --- |
| Raven cb1_t1, x320/y448 | 0.128248 → 0.137720 | 13.64% → 14.31% |
| Raven cb1_t1, x64/y64 | 0.114318 → 0.135201 | 12.15% → 14.03% |
| Raven cb1_t1, x640/y640 | 0.103717 → 0.124929 | 11.17% → 13.09% |
| Other asset cb2_t1, x320/y448 | 0.048203 → 0.078416 | 5.38% → 8.40% |

Activity is a sanity measurement, not accuracy or continuity truth. Actual
pixels show some stronger long panel outlines, but beaded thin lines,
uneven parallel trim, disconnected diagonal joins and incomplete curves remain.
Protected-detail preservation and unseen-asset quality are not established.

See the [all-region comparison](../../artifacts/nsamdr/diagnostics/v178_contrast_transfer/comparison_v4_v5/all_regions_overview.png),
[Golden Patch fixed crops](../../artifacts/nsamdr/diagnostics/v178_contrast_transfer/comparison_v4_v5/cb1_t1_a_000_x320_y448_s256_fixed_crops.png)
and [other-asset fixed crops](../../artifacts/nsamdr/diagnostics/v178_contrast_transfer/comparison_v4_v5/cb2_t1_a_000_x320_y448_s256_fixed_crops.png).
Panels use a fixed 0..1 mask range, without thresholding or contrast adjustment.
These are raw contour diagnostics, not enhanced physical textures.

The new CUDA-default dense/tiled maximum was 0.001547396 for V4 and
0.001606822 for V5 on the Golden Patch. Controlled probes identify cuDNN TF32
rounding: disabling only cuDNN TF32 gives exact dense/tiled equality for V4
and V5, with V5 verified on all four regions. CPU Golden Patch also matches
exactly. Default-CUDA threshold masks at 0.30 are identical, and maximum errors
occur inside tiles. Cross-device CPU/CUDA outputs are not bit-identical.
No tiling changes are justified. Probe scripts, backend settings and reports
are preserved in `v4_backend_numerics/` and `v5_backend_numerics/`.

## Additional analytic stress diagnosis

An artifact-only script, `v178_contrast_transfer/probe_analytic_stress.py`,
evaluated both authenticated models on 28 independently constructed analytic
cases × two polarities. It adds genuine parallel segments, diagonals, circles,
T/X junctions, background and protected appearance, using area/point native
sampling and known clean 4× geometry. CUDA TF32 was disabled for both models.
No training, source generator, loss or acceptance gate was changed.

| Geometry (four cases each) | Original IoU V4 → V5 | Inverted IoU V4 → V5 |
| --- | --- | --- |
| Diagonal | 0.7000 → 0.6131 | 0.0008 → 0.6469 |
| Genuine parallel | 0.6480 → 0.5224 | 0.0003 → 0.5006 |
| Circle | 0.6379 → 0.5444 | 0.0035 → 0.5524 |
| T junction | 0.5966 → 0.5314 | 0.0015 → 0.5581 |
| X junction | 0.7037 → 0.5381 | 0.0002 → 0.5615 |

The existing generator's supposed second parallel line is identically zero:
`signed >= -0.70`, added offset `>= 1.10`, so
`.5 - 4*(signed + offset) <= -1.10` before clamping. There is also no explicit
T/X composition. Aggregate scores from that generator cannot establish those
capabilities. The shared historical generator remains untouched.

Thin parallel predictions still fragment. One point-sampled T strongly
attenuates its vertical arm; weak native RGB/normal evidence remains, with
edge observations around 0.00454–0.00833. V5's longest missing centerline run
is about 12.80 native pixels. This cannot be attributed solely to missing
junction training. The diagnostic uses fixed ±0.5-native-pixel bands around
known analytic paths and does not measure real geometry or RGB preservation.
Four examples per family are bounded diagnosis, not broad generalisation.

## Verification, environment and CI

- `scripts\build\nsamdr.bat test`: **141 tests passed**.
- Automatic-preview suite: **6 passed**; material-coverage suite: **11 passed**.
  Total: **158 distinct regressions passed**. Targeted contour/diagnostic tests
  passed before the broader run.
- Two-update CPU smoke emits authenticated V5 checkpoint/report and correctly
  returns `G2_SYNTHETIC_FAIL`; failed candidate retained.
- Changed Python sources compile; `git diff --check` passes. Windows SDK FXC
  compiles `PSMain` (`ps_5_0`). Existing warning X4000 at shader line 156 remains.
- No CI run exists for baseline 92afe9c or implementation e56a5731a. Explicit
  dispatch to `JamieShelley/trinity` returned HTTP 422:
  `Actions has been disabled for this repository.` Local checks are not
  represented as successful hosted CI.

Environment: Windows x64, managed Python 3.10.11, torch 2.11.0+cu128, CUDA 12.8,
RTX 5080 (sm_120), driver 617.42, OpenCV 5.0.0.93, numpy 2.2.6, Pillow 12.3.0.
Native tooling is CMake/MSBuild, VS 18 2026, MSVC 19.51.36252.0; presets require
CMake 3.31 and the existing viewer targets x64 TrinityDev/DX11. This is not a
Unity project. Training/backend precision policy was unchanged between runs.

To verify without retraining existing valid candidates:

```powershell
& scripts/build/nsamdr.bat test
& artifacts/nsamdr/python-env/Scripts/python.exe -m unittest tools.nsamdr.neural.test_nsamdr_v178_automatic tools.nsamdr.neural.test_nsamdr_v177_material_coverage
& artifacts/nsamdr/python-env/Scripts/python.exe -u tools/nsamdr/neural/benchmark_nsamdr_v178_contrast.py --checkpoint artifacts/nsamdr/diagnostics/v178_golden_contour/seed_17842_patch_32_batch_4/candidate.pt --output artifacts/nsamdr/diagnostics/v178_contrast_transfer/v5_review_repeat --native-folder artifacts/nsamdr/golden_patch/cb1_t1_a_000_x320_y448_s256 --device cuda
```

The last command requires an unused output directory and current authenticated
sources. Archived numerical/stress/panel scripts reproduce the supplemental
diagnostics. Checkpoints, raw source snapshots, reports, native evidence and
images are preserved in the
[local evidence archive](../../artifacts/nsamdr/v178_contrast_iteration_20261030.zip).

## Planning review before the next contract change

**Gate:** G0 passes; existing G2 synthetic gate passes; RGB inversion sensitivity
is substantially reduced; real structural/physical qualification still fails.
The assignment's final completion criteria have not been achieved.

Proposed smallest next experiment (not implemented):

1. Keep the V5 head, representation, loss, RGB augmentation and original gates.
   Introduce a V178-only analytic training path that composes genuine parallel
   grooves and T/X junctions before rasterization. Keep historical V17.7 and
   V17.6 files/evidence frozen. Add invariants for positive second-line area,
   intended separation, junction connectivity and protected appearance.
2. Review/register the extended supervision and typed diagnostic contract
   explicitly: known-path gaps, false structures, parallel splits/merges,
   junction survival and original/inverted precision alongside the existing
   scores. Keep these fixed stress cases out of training. No relaxed threshold
   or replacement of real-domain evaluation is proposed.
3. Only after that review, run one bounded experiment under a fresh schema/seed.
   Expected outcome to test is fewer supported gaps/merges without the current
   original-polarity precision/IoU regressions. No numerical gain is promised.
   If evidence is erased by point sampling, expose the uncertainty rather than
   inventing a branch. Use paired sampling controls to distinguish missing
   observations from failed inference.

Only credible real structure should advance to physical reconstruction.
`NativeEnhancer4x` and the Raven exporter were inspected: the current DX11 C1
is historical V17.6 checkpoint
`456c4a2c98b3974c1fb0b266d0758970a404e467074f8ac6f4e1f34ef02e63d5`,
not V178. No V178 contour-to-physical bridge exists. Such a bridge requires
separate physical supervision and planning review, followed by full-native
1024→4096 albedo, own-resolution aligned auxiliaries, map preservation,
dense/tiled/seam tests and actual controlled DX11 original/baseline/new-map
renders. Those outputs and comparisons have not been fabricated or claimed.
