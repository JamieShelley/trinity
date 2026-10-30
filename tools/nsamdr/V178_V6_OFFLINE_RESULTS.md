# V178 structural-continuity experiment — offline CPU execution

## Scope and provenance

This is an **archive-based implementation and CPU evaluation**, **not** a modification
of `D:\REPOS\trinity`. That Windows checkout, `AGENTS.md`, remote HEAD, Windows CUDA
runtime, hosted CI, the original full EVE maps and DX11 renderer were inaccessible
from this session. No Git commit or push was made. Do not represent these
outputs as a repository commit or native-render qualification.

The uploaded 30 October V4/V5 archive supplied source-matched V4/V5 checkpoints,
original V5 source files, diagnostic scripts, verified archive crop assets and
prior comparison results. Original V5 source hash remained
`e7071000bc5451614139cd864fde7f64a681fceba024f30cc87451d813bd283d`.
All original archived project source files were left unchanged.

## Implementation

New only (no existing project source modifications):

- `tools/nsamdr/neural/v178/composed_geometry.py` — continuous HR SDF
  compositions for real paired grooves, T/X junctions, diagonals, curves,
  circles and protected texture. Native area/point sampling, independent
  auxiliary maps and seeded RGB inversion. Mixes V5 original and V178
  composed cases during training. Existing `v177` generator untouched.
- `tools/nsamdr/neural/train_nsamdr_v178_continuity.py` — separate
  experimental V6 schema and seed 17852; original V5 CLI defaults and
  authentication unchanged. Same architecture and loss; refuses existing
  candidate/report overwrite, source-hash-bound resume and training report.
- `tools/nsamdr/neural/test_nsamdr_v178_continuity.py` — 9 focused geometry,
  provenance and determinism regressions.
- `tools/nsamdr/neural/benchmark_nsamdr_v178_continuity.py` — fixed 28 composed
  analytic geometries under paired area/point sampling and both RGB polarities,
  total 56 input cases; scores V4/V5/V6 on the same cases. Diagnostic only.
- `tools/nsamdr/neural/benchmark_nsamdr_v178_v6_fixed.py` — the original 36
  registered original/inverted holdouts, all V4/V5/V6 on identical cases.
- `tools/nsamdr/neural/inspect_nsamdr_v178_v6_archived_native.py` — explicit
  archive-crop-only V6 inference/visual comparisons with original source-map
  verification marked **unavailable**, never promoted to authentic full-source
  validation.

### Experiment

Model schema: `NSAMDR_V178_CONTOUR_HEAD_V6_COMPOSED_GEOMETRY_EXPERIMENT`.
Seed: 17852. Batch: 4. Native patch: 32. Total updates: **400**, CPU only,
with resumed checkpoints at 250 and 350 and final step 400. The final report's
`trainingUpdates=50` is the *last resumed invocation*, not the total trained
horizon. Checkpoint identity and training report SHA256 are verified by the
supplied diagnostic scripts. The historical synthetic gate passes, but real
qualification remains false.

### Fixed original 36-case benchmark — IoU

| Group | V4 orig | V5 orig | V6 orig | V4 inverted | V5 inverted | V6 inverted |
|---|---:|---:|---:|---:|---:|---:|
| Primary 12 | 65.80% | 56.99% | **53.53%** | 0.35% | 57.88% | **54.20%** |
| Independent A 12 | 71.29% | 61.64% | **59.10%** | 0.66% | 61.25% | **58.57%** |
| Independent B 12 | 54.54% | 52.52% | **49.90%** | 0.52% | 52.14% | **49.81%** |

V6 **regresses against V5 in all six registered comparisons**. This rejects
promotion even though the additional composed-case tests showed local gains.

### Additional composed-case IoU (original polarity)

| Geometry | V4 | V5 | V6 |
|---|---:|---:|---:|
| Parallel | 52.52% | 35.53% | 48.50% |
| T junction | 70.03% | 55.34% | 57.90% |
| X junction | 57.73% | 46.59% | 49.90% |
| Curve | 65.82% | 53.88% | 52.40% |

These additional scores are synthetic; path-band survival is only a diagnostic
and does not prove native topology, especially on curves built from multiple
short sample segments. Circle paths lack explicit continuity metadata in this
iteration. Neither threshold nor historical acceptance contract changed.

### Native archived crops

All four native **crop-asset** hashes were validated against their evidence
registrations. The original full authored source maps were not supplied and
therefore their registered SHA256 values could **not** be independently
verified. This is not a bypass or substitute for the live provenance gate.

V6 256x256 crop inference yielded 1024x1024 output, with CPU dense/tiled max
absolute difference **0.0** on each of four regions. Active fractions at the
unchanged 0.30 threshold were 17.24%, 16.77%, 17.23% and 11.92%. These are
activity measurements, not structural accuracy. Visual comparisons show
fragmentation/line thickening; no faithful physical reconstruction demonstrated.
V4/V5 native float baselines were archived CUDA outputs, whereas V6 here used CPU.

## Tests and limitations

- **19/19** focused V178 continuity, contrast and automatic-preview tests passed
  under the container's CPU-only PyTorch/Python environment.
- An expanded 29-test attempted run passed 28 and failed 1 because the uploaded
  archive lacks `tools/nsamdr/neural/golden_nsamdr_v178.py`; this is a missing
  archive dependency, not a verified regression.
- 400 CPU training updates completed and final checkpoint/report hashes match.
- The existing V5 source hash is unchanged.
- GPU/CUDA training, original full-source-map authentication, complete repo
  regression suite, CI and DX11 rendering are **not verified**.

**Decision:** Preserve V6 as a **failed experimental candidate**. Do not switch
CLI/GUI defaults or construct physical maps. Next investigation should measure
why original-polarity and fixed holdout precision/IoU regress, distinguishing
unobservable thin geometry from training-distribution compromise. Any
representation, loss or evaluation-contract change needs separate review.

## Offline reproducer

Install the new source files into an independently verified working copy
*without overwriting any existing files*. Confirm the V5 project source files
still match their original hash. Then, using the existing managed Python,
run relevant tests and the two diagnostic scripts. Do **not** run V6 `fresh`
against a pre-existing candidate; the training script intentionally refuses it.

Example from the repository root:

```powershell
& artifacts/nsamdr/python-env/Scripts/python.exe -m unittest tools.nsamdr.neural.test_nsamdr_v178_continuity
& artifacts/nsamdr/python-env/Scripts/python.exe tools/nsamdr/neural/benchmark_nsamdr_v178_v6_fixed.py --device cpu --candidate artifacts/nsamdr/diagnostics/v178_golden_contour/seed_17852_patch_32_batch_4/candidate.pt --output artifacts/nsamdr/diagnostics/v178_continuity_review_fixed.json
```

Reproducing a fresh 400-update **GPU** run requires a **new unused seed and
schema/source provenance**. Do not overwrite the supplied CPU candidate or
silently reclassify it as a CUDA run.
