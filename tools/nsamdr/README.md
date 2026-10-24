# NSAMDR — V17.7 physical-baseline research, V17.6 neural evidence

**Canonical operator interface: one full-featured GUI.** Raven preview now
uses the V17.7 semantic B baseline and four independently rendered A/B/C0/C1
views. Neural C0/C1 are still the authenticated, unqualified V17.6 experiment;
the active enhancement model has **not** been trained. Historic proof
implementations and physical-map utilities remain internal immutable
dependencies for checkpoint provenance, not alternative operator GUIs.

## V17.7 M1 + M2: four-way native Raven research comparison

The current active **preview** exports four independent material manifests from
one full-native, authenticated neural inference pass:

- **A:** original authored ship textures (no neural reconstruction).
- **B:** V17.7 conservative 4× semantic baseline. RGB bicubic, normal XY
  bilinear + unit-length bound, legacy coverage masks bilinear, V5 categorical
  material selectors nearest to preserve IDs, glow/roughness continuous. An
  unknown shader policy fails safe to categorical. Untouched packed channels
  use nearest instead of the former blanket Lanczos reconstruction.
- **C0:** the actual frozen V17.2 candidate from
  `base_candidate_albedo`, with original C0 normal/material outputs.
- **C1:** the authenticated **trained V17.6** specialist output. This is
  **not** a new trained V17.7 enhancement model and is still unqualified.

All four have exactly 4× native albedo dimensions, independent texture files,
the same ship mesh and draw ranges, one shared camera/light/shader/sampler, and
a 2×2 comparison in the existing full-featured DX11 viewer. No downsampling
of the full albedo source is performed. Where original auxiliary maps have
different dimensions, their mapping to albedo-space remains an explicit
interpolation (categorical nearest, continuous area-down/linear-up). Direct
multiresolution neural feature input is future V17.7 work.

Run from the existing GUI **Raven 3D A/B/C0/C1 preview**, or:

```bat
scripts\build\nsamdr.bat v176-raven-preview --authority-index 10 --mode output10
```

**Before judging visual quality, build the revised DX11 host.** The normal
preview command automatically detects and rebuilds an out-of-date viewer.
Build-only is available via:

```bat
set NSAMDR_BUILD_ONLY=1
scripts\build\run_nsamdr_obj_preview_dx11.bat
set NSAMDR_BUILD_ONLY=
```

Check `candidate_manifest.json`: it must contain `fourWayComparison: true`,
`backboneMaterials`, `baselineMaterials`, `candidateMaterials`, and
`baselinePolicyBySource`. If the native C++ build fails, do not interpret an
old three-pane renderer as the current result.

**This is M1/M2 groundwork, not quality qualification.** The categorical V5
selector still has a fundamental one-scalar representation limit: it cannot
express arbitrary mixed material identity/coverage. A true antialiased selector
requires a dedicated coverage representation and shader semantics, planned for
the next enhancement model. The immutable historical V17.6 proof loss and
backbone are untouched. Follow-up M3/M4 will supply clean 4× target geometry
and learn a bounded, multichannel refinement specialist.

## Launch

From `D:\\REPOS\\trinity`:

```bat
scripts\build\nsamdr.bat gui
```

This opens the **NSAMDR — V17.7 Baseline / V17.6 Neural Research** GUI, with the
restored stage tree, scrollable controls, runtime log, progress, stop, command
preview, artifact detection, evidence navigation and Raven actions. The old V16
and V17 operator GUIs have been retired.

## Raven: see a V17.6 neural preview now

In the left workflow stage tree, select **Raven 3D visual preview**:

- Authority: `10` (or `1`)
- Specialist: `output10` (or `hr_features58` for the ablation)
- Texture resolution: native input, exact 4× output (fixed)
- EVE cache: `C:\\CCP\\EVE`
- Click **Run selected** or use **Render Raven (DX11)** in the footer

It reads the latest authenticated, completed `compare-features` checkpoint
for that arm, loads the frozen Stage 9 A12 backbone, then generates temporary
physical maps and opens the existing DX11 ship renderer. It **does not** load
a V16 checkpoint and **does not** apply the unattainable target-assisted pixel
fit at inference. Preview files are under
`artifacts/nsamdr/render_preview/V176_RAVEN_TRAIN10_OUTPUT10/`.
The candidate is research-only and unqualified.

Equivalent CLI command:

```bat
scripts\build\nsamdr.bat v176-raven-preview --authority-index 10 --mode output10
```

## Train a new specialist and refresh Raven

In the left workflow stage tree, select **Capacity and features**:

- Phase: `compare-features`
- Resume: `fresh`
- Audit-only: **unchecked**
- AMP: `auto`
- Click **Run selected** and confirm the process exits successfully

This runs the original two frozen A12 training authorities with the two
ten-channel and 58-channel V17.6 specialists, with unchanged physical caps,
Stage10A loss, provenance checks, and sealed final-test boundary.

Then return to **Raven 3D visual preview** and render again. This picks the
newest authenticated completed checkpoint, never an incomplete fit. To keep
using a specific run, the CLI also accepts `--comparison-report PATH`.

The `attainable` phase is **not neural training**: it fits disposable
target-assisted per-pixel fields for diagnosis, and updates no model weights.

## Regression checks

```bat
scripts\build\nsamdr.bat test
```

This runs the supported V17.6 CLI/GUI/render tests and Stage10A specialist,
capacity, stability, and attainable-fit tests.

## Architecture constraints

- 4× reconstruction from aligned low-resolution albedo, normal XY, material.
- Frozen V17.2 physical backbone from authenticated Stage 9 proof.
- Current V17.6 output: a bounded single-channel grayscale albedo detail
  correction before the original physical projection.
- Normal XY and material pass through unchanged.
- Training-only stationary Haar supervision; authored HR reference must
  never enter inference evidence.
- No production promotion, final-test exposure, or modified quality gates.

**Quadtree routing is not implemented in the current repository.** The Raven
preview currently uses overlapping dense tiles and cannot demonstrate
quadtree performance. Add full-density equivalence and seam tests before
switching to adaptive routing.

The original historic proof data and sealed partitions remain immutable. The
next architecture gate is successful specialist learning, not a preview alone.

## Full-native inference and quality objective

Raven inference reads **every native source albedo pixel** (no global preview
downsampling) and reconstructs at **4× native dimensions**. A 2048×2048
albedo is therefore passed to the network as 2048×2048 evidence and requires
8192×8192 candidate output. Execution uses overlapping 128-LR tiles. GPU
inference is tile-wise; full-resolution output accumulation is stored in
temporary **disk-backed float32 buffers**, and material textures are flushed
after their last use. Large native sources require substantial **free disk
space** for these temporary maps, with output images potentially reaching
16384×16384. If a native source would require an output exceeding the DX11 16384-pixel
texture limit, the renderer **fails explicitly** instead of silently
discarding input detail.

**Important: this corrects a preview regression, not neural learning.** The
frozen Stage10A/10A.1 training diagnostics still take 512×512 native authored
*patches* and synthesize aligned 128×128 degraded inputs for the registered
4× paired supervision. That design is valid for testing reconstruction, but
does not teach the specialist to surpass the native authored target or
straighten the target's aliased edges. Achieving that stated NSAMDR visual
goal requires a separate, explicitly registered native-source → enhanced-4×
training objective with spatial/edge continuity and artifact controls.
Do not alter the old frozen manifests, archived checkpoints or sealed test.

## Diagnose jagged Raven markings before changing training

In the full V17.6 GUI choose **Raven 3D visual preview**, then click
**Audit native texture channels (no training)** and **Open latest channel audit**.

Or run:

```bat
scripts\build\nsamdr.bat v176-raven-channel-audit --crop-size 128 --max-textures 6
```

This prepares the Raven's authentic texture material manifest, identifies
high-contrast **native source-coordinate** ROIs, and writes labelled PNG
comparisons for albedo, normal XY, material, glow, roughness and the
unchanged paint, dirt and ambient-occlusion masks. Each sheet
shows **nearest 4×**, **bilinear 4×** and their absolute difference. The report
records the source texture dimensions, ROI coordinates and per-channel
interpolation difference. For a known defect you can target a specific
native-coordinate ROI with `--crop-x X --crop-y Y`.

Output:
`artifacts/nsamdr/diagnostics/v176_raven_channel_audit/audit_*/report.json`

This is a **visualization crop, not a network input crop**. The model and
standard full-native Raven renderer are unchanged. Bilinear filtering is an
ablation, **not** an automatic fix for categorical paint/material masks.
The audit identifies which semantics cause block edges so a subsequent
spatial/semantic-aware reconstruction can be designed safely.

## V17.7 M3/M4: first learned enhancement over B (synthetic only)

The new `NativeEnhancer4x` is **separate from the frozen V17.6 proof**.
It consumes native-resolution albedo RGB, normal XY and three semantic
channels, predicts bounded 4× RGB/normal/material residuals over the semantic
B baseline, and initialises to **zero correction**. It can fall back to B
exactly (subject to normal floating-point precision). V5 categorical
material selector identity is preserved and cannot be corrected by this
initial architecture. Continuous legacy masks, glow and roughness may receive
bounded corrections; normals are renormalised.

Training uses clean **analytic 4× geometric references** (diagonal grooves,
circles, rounded panel outlines and curved lines) paired with simulated
native rasterisation. This is training-only synthetic geometry, not EVE's
sealed native/final-test data and not evidence of real-Raven quality.

From `D:\\REPOS\\trinity`:

```bat
scripts\build\nsamdr.bat v177-enhancement-train --steps 300 --batch-size 4 --patch-size 32 --seed 177 --device cuda --resume fresh
```

Or choose **V17.7 native enhancement** in the *existing full GUI*.
The trainer writes a report and synthetic-only checkpoint under
`artifacts/nsamdr/diagnostics/v177_native_enhancement/seed_177_patch_32_batch_4/`.
It will exit **2** and report `SYNTHETIC_LEARNING_UNPROVEN` unless
held-out synthetic geometry error beats B by at least 5%, with an actual
reduction against the untrained network. Continue an interrupted or
insufficient run with `--resume auto --steps <higher total>`; do not
reset or silently overwrite existing proof outputs.

Only if the decision is `SYNTHETIC_IMPROVEMENT_ONLY`, use:

```bat
scripts\build\nsamdr.bat v176-raven-preview --authority-index 10 --mode output10 --enhancer-checkpoint "artifacts\nsamdr\diagnostics\v177_native_enhancement\seed_177_patch_32_batch_4\candidate.pt"
```

Or select **Raven 3D A/B/C0/C1 preview**, C1 source `v177` and
enter the absolute candidate.pt path. It checks the report decision,
checkpoint hash, model schema and current source-code SHA, refusing to
load an unvalidated or stale candidate.

The viewer's C1 must be labelled **V17.7 SYNTHETIC TRAINED —
UNQUALIFIED**. C0 is still the real frozen historical model. A and B
are unchanged. A synthetic pass is only a prerequisite for actual EVE
native-domain training and visual validation; it is **not** proof that C1
surpasses the authored A. Quadtree remains unimplemented.

## Further V17.7 native-domain enhancement work

The long-term quality requirement is **better than the original authored
texture**, not only reproducing that texture from a degraded 128×128 input.
Examples include smoothing stair-stepped panel lines without shifting UV
boundaries, preserving mechanical geometry, reconstructing high-frequency
texture consistently across tiles, and refining supported normal detail.

The current V17.6 `compare-features` phase is a **historical reconstruction
proof**, not this enhancement training objective. Existing reports and sealed
holdouts must remain unchanged. The new enhancement training must sample
native-resolution regions, expose every available input physical channel,
then learn a 4× refined output against verified high-quality/anti-aliased
references or explicitly confidence-gated geometry-based supervision.
Without a credible better-than-native reference, optimizing against the
unaltered original would reproduce its original aliasing. The revised
objective must measure edge regularity, topology preservation, texture
content retention, boundary seams, normals, and false-detail rate.
