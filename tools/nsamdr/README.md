# NSAMDR — active V17.6 texture reconstruction

**Canonical operator interface: V17.6 only.** The Windows launcher, Python
GUI, training commands, tests, and Raven research preview all target the same
current dense specialist architecture. Historic proof implementations and
shared physical-map utilities remain internally necessary for checkpoint
provenance; they are not operator-selectable training engines.

## Launch

From `D:\\REPOS\\trinity`:

```bat
scripts\build\nsamdr.bat gui
```

This opens the **NSAMDR V17.6 — Training and Raven Preview** GUI, with the
restored stage tree, scrollable controls, runtime log, progress, stop, command
preview, artifact detection, evidence navigation and Raven actions. The old V16
and V17 operator GUIs have been retired.

## Raven: see a V17.6 neural preview now

In the left workflow stage tree, select **Raven 3D visual preview**:

- Authority: `10` (or `1`)
- Specialist: `output10` (or `hr_features58` for the ablation)
- Target size: `1024`
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
scripts\build\nsamdr.bat v176-raven-preview --authority-index 10 --mode output10 --target-size 1024
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
