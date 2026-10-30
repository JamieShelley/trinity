# NSAMDR V178 — V7/V8 offline diagnostic continuation (30 October 2026)

## Authority and scope

This work was executed from the uploaded authenticated V4/V5 evidence and the prior V6 offline archive under a separate Linux CPU working directory. It **did not** modify `D:\REPOS\trinity` or create a Git commit. Windows CUDA, DX11 and original full-resolution native source-map hashes were unavailable. These experiments are **not** promoted models or native-render qualification. The previous V4, V5 and V6 source files and checkpoints remain unmodified.

## New evidence: native-observation distribution mismatch

Measurements from four archived native EVE crops and 32 synthetic examples per distribution:

| Mean | V5 synthetic | V178 composed | Raven centre | Raven other regions | EVE second asset |
|---|---:|---:|---:|---:|---:|
| RGB intensity | 0.306 | 0.306 | 0.7796 | 0.735–0.759 | 0.7395 |
| Absolute native normal XY | 0.0095 | 0.0078 | 0.1604 | 0.135–0.141 | 0.1563 |
| Calibrated native edge evidence | 0.0477 | 0.0468 | 0.2430 | 0.216–0.221 | 0.1211 |

**Brightness control:** Adding +0.42 to registered synthetic RGB while retaining fixed geometry, normals, observations and targets did not materially degrade V5: primary original IoU 56.99% → 58.14%, independent A 61.64% → 62.25%, independent B 52.52% → 52.75%. Some clipping occurred (per-case maximum 3.7–5.7%) and derived RGB edge changes were nonzero, so this is an intervention rather than exact scene simulation. Brightness alone is not a sufficient cause.

**Normal stress:** Add an independent, reproducible background-normal perturbation at native resolution and recompute its physical edge observation. This preserves the RGB, clean HR geometry target and auxiliary maps. Smooth perturbations make synthetic mean absolute normal approximately 0.16 and edge evidence approximately 0.17–0.18; independent pixel noise gives normal ~0.16 and edge ~0.25–0.26. These ranges overlap native aggregate magnitudes but do not reproduce the actual spatial statistics of EVE normals.

| Fixed original-polarity IoU | V4 original | V5 original | V6 original |
|---|---:|---:|---:|
| Primary clean | 65.80% | 56.99% | 53.53% |
| Primary + independent noise | 15.8% | 10.18% | 10.6% |
| Primary + smooth noise | 23.2% | 13.91% | 13.7% |

This demonstrates model sensitivity to nuisance normal evidence. It does not establish a physically faithful native training distribution or the accuracy of actual EVE contours.

## V7 — V5-initialised historical replay and composed-geometry continuation

V7 was trained from the authenticated V5 model for **120 CPU updates** (fresh seed 17862, 3 historical examples + 1 composed example per four-sample batch, unchanged model/loss, LR 0.0001). Completed checkpoint/report are provenance-bound.

| Fixed 36-case original IoU | V5 | V6 | V7 |
|---|---:|---:|---:|
| Primary | 56.99% | 53.53% | 57.46% |
| Independent A | 61.64% | 59.10% | 62.72% |
| Independent B | 52.52% | 49.90% | 51.85% |
| Composed parallel | 35.53% | 48.54% | **20.55%** |

V7 partially recovers original holdout results but damages parallel continuity. **Rejected for promotion.**

## V8 — normal-background augmentation experiment

V8 was trained from the unchanged authenticated V5 weights for **160 CPU updates** (seed 17872, LR 0.00015). Each batch used V7's three historical + one composed examples, with 2/4 receiving independently seeded *smooth* native normal perturbations and consistently recomputed edge observations. Geometry, targets, albedo, aux maps, network and loss stayed fixed.

| Primary fixed set, IoU | V5 | V8 |
|---|---:|---:|
| Clean original | **56.99%** | 50.84% |
| Clean inverted | **57.88%** | 53.26% |
| Smooth-noise original | 13.91% | **35.20%** |
| Smooth-noise inverted | 14.83% | **44.25%** |
| Independent-noise original | 10.18% | **21.11%** |

V8's clean original IoU also drops on independent A (61.64% → 56.23%) and independent B (52.52% → 42.11%). It fails most composed geometries, including T junctions (55.34% → 48.92%), X junctions (46.59% → 40.42%) and diagonals (43.28% → 27.83%). Thus improved nuisance robustness is purchased at unacceptable structural fidelity loss. **Rejected for promotion.**

## Archived native diagnostic

The offline archive provided 256×256 crop inputs; the **full original source-map hashes were not rechecked**. Authenticated V5 and V8 weights produced 1024×1024 contour fields and **exact CPU dense/tiled correspondence (max difference 0.0)** with the existing 96-pixel tile/20-pixel halo.

| Archived crop | V5 active fraction | V8 active fraction |
|---|---:|---:|
| Raven central | 14.31% | 9.13% |
| Second EVE asset | 8.40% | 2.82% |

Visual comparisons show V8 suppresses substantial plausible structural detail; this is not a qualified improvement. Activity fractions alone are not accuracy metrics. The native outputs are raw *contour diagnostics*, **not enhanced physical albedo, normals or materials** and **not DX11 renders**.

## Exact added files

- `tools/nsamdr/neural/train_nsamdr_v178_replay.py` — isolated V7 training and source-bound replay checkpoint.
- `tools/nsamdr/neural/benchmark_nsamdr_v178_replay.py` — all fixed original/inverted controls and composed cases.
- `tools/nsamdr/neural/diagnose_nsamdr_v178_brightness.py` — registered photometric-shift intervention.
- `tools/nsamdr/neural/diagnose_nsamdr_v178_normal_domain.py` — deterministic normal perturbations, observation consistency and stress benchmark.
- `tools/nsamdr/neural/train_nsamdr_v178_normal_variation.py` — isolated source-bound V8 experiment.
- `tools/nsamdr/neural/benchmark_nsamdr_v178_normal_variation.py` — fixed clean, smooth-noise, independent-noise and composed comparisons.
- `tools/nsamdr/neural/inspect_nsamdr_v178_v8_archived.py` — archive-crop-only visual and dense/tile diagnostic.
- `tools/nsamdr/neural/test_nsamdr_v178_domain.py` — deterministic/provenance/geometry-preservation focused tests.

All archived original project sources remain byte-for-byte unchanged. The original V5 training-source hash still matches its report. This package includes new V7/V8 scripts, immutable offline checkpoints, matched reports, visuals and SHA256 manifest, **not a complete repository**.

## Verification

- **28 targeted tests passed**, comprising 9 new normal-domain tests, 9 existing V6 geometry tests, 4 contrast tests and 6 automatic-preview tests.
- Source/parent SHA256 authentication passed for V7/V8; checkpoint/report SHA256 checks passed.
- Native V5/V8 CPU dense-versus-tiled max difference: **0.0** for both archived regions examined.
- Automatic-preview legacy tests passed; separate archived material-coverage module is absent and cannot be run in this environment.
- **Not verified:** `D:\REPOS\trinity` Git state, full repository test suite, CUDA, original full native source SHA256, hosted CI, true HR native references, physical-map reconstruction and DX11 comparison.

## Gate / planning review

**V4/V5 stay the historical baselines; V6, V7 and V8 are failed experiments.** None is promoted. The latest evidence rules out a simple brightness-only correction and shows that independent normal noise strongly changes predictions, while naive smooth-noise augmentation loses clean geometry.

Before changing the network, loss, input representation or evaluation contract, review a more physically grounded native-observation model: separate authored high-frequency normal/material texture from observable geometric boundaries, preserve evidence for thin supported paths, and evaluate against independent native assets. Use a reproducible *unlabelled* domain-statistics study or independently generated physically based synthetic observations; do not train on the held-out native crops or construct fake native HR targets. Keep the existing clean/inverted/stressed fixed cases and actual native visual comparisons as separate gates.

The outstanding objective remains continuous, faithful real EVE geometry and actual coherent 1024→4096 physical map reconstruction. No claim of real-domain completion is justified.
