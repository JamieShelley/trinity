# NSAMDR — Golden Patch G2 automatic contour inference (current)

Manual contour tracing is **not the standard workflow**. A hand-drawn trace
introduces bias and scales poorly. The canonical Golden Patch work now has
three steps:

1. **G0 native Raven evidence**: `v178-golden-patch --action prepare`
   extracts and hashes native albedo, normal and other physical-map crops.
2. **G2 synthetic structural training V2**: train an analytic reference
   network using **native RGB and normal edge evidence** rather than a
   categorical material selector or manual traced label. This makes the
   training-time observation semantics match the real Raven inference path.
3. **Automatic G2 real-patch preview**: apply the trained synthetic candidate
   to the frozen native Raven patch without trace.json. Compare its inferred
   contour field with native and interpolated appearance. This produces a
   *hypothesis* about continuous geometry, not a verified real clean target.

## Commands

After updating NSAMDR, from the repository root:

~~~bat
scripts\build\nsamdr.bat v178-contour-train --steps 400 --batch-size 4 --patch-size 32 --seed 17822 --device cuda --resume fresh
~~~

This is **a new model schema / input encoding**. The previous
seed-17812 checkpoint cannot be reused as a source-equivalent candidate.

After a `SYNTHETIC_CONTOUR_PASS_ONLY` report (>= 20% held-out synthetic
contour recovery), run the new **Automatic real Raven contour preview**
stage in the full NSAMDR GUI. Or run:

~~~bat
scripts\build\nsamdr.bat v178-golden-auto --folder "artifacts\nsamdr\golden_patch\cb1_t1_a_000_x320_y448_s256" --device cuda
~~~

The exact folder name is grounded by the G0 preparation output and may vary.

Outputs under `automatic_contour_preview`:

- `native_edge_observation.png`: authentic source RGB/normal native-edge evidence
- `baseline_observation_4x.png`: interpolated observation; **not ground truth**
- `candidate_contour_4x.png`: automatic learned 4× contour proposal
- `automatic_contour_comparison.png`: A native, B interpolated, C candidate
  contour overlay, D model-versus-observation diagnostic difference
- `report.json`: provenance, actual dimensions and explicit **no real-EVE
  quality qualification** decision

The CPU GUI does not import OpenCV or load the contour model. The inference
stage executes in the managed CUDA environment; `cv2` is required there.
Do **not** install Python packages in the GUI CPU environment merely to
display a manual trace editor.

**Important limitation:** no real 4× target exists just because a smooth
contour was generated. Distinct structural evidence, physical-render
consistency and real held-out evaluation are still needed before enhanced
albedo/normal maps can be qualified. The synthetic pass alone is insufficient.

---

# Historical/manual single-patch annotation experiments (not recommended)

The following earlier G1 human tracing and G2.R same-patch fitting
commands are retained for experimental reproducibility, not required for
automatic inference or GUI operation. They are not the production objective.


## The reference is the project goal, not the latest training loss

The original [README](README.md), [EXAMPLE.png](EXAMPLE.png) and
[NSAMDR_FULL_SYSTEM_ARCHITECTURE.png](NSAMDR_FULL_SYSTEM_ARCHITECTURE.png)
remain the canonical project references. The goal is continuous contours and
legitimate mechanical detail **better than the jagged native authored raster**.
A larger output image or a lower synthetic reconstruction loss is not success.

| Gate | Real input | Expected evidence | Claim permitted |
| --- | --- | --- | --- |
| G0 | Authentic Raven albedo and physical maps | Frozen native 256×256 crop, hashes, own-resolution auxiliary maps | Provenance established |
| G1 | Independently human-traced native contours | Clean 4× spline reference and protected-detail mask | Supervision available |
| G2.S | Aliased synthetic observations | Analytic clean 4× geometry in unseen synthetic holdout | Synthetic-only learning |
| G2.R | Real Raven G0 and human G1 | Capacity fit and visual comparisons against traced geometry | Single-patch overfit only |

No model prediction, interpolation, or sharpening result may be registered
as an independently curated clean target. G2.R deliberately sees the target
during training; that is **not** a real-world generalisation test.

## Execute G0

From D:\REPOS\trinity, run:

~~~bat
scripts\build\nsamdr.bat v178-golden-patch --action prepare --crop-x 320 --crop-y 448 --size 256
~~~

Golden source images go to the directory
artifacts\nsamdr\golden_patch\cb1_t1_a_000_x320_y448_s256
(actual folder prefix depends on the prepared native albedo filename).

Inspect native_albedo.png, native_nearest4x.png,
baseline_bicubic4x.png and evidence.json. Other physical-map crops retain
their own native resolutions; coordinates record the same source UV rectangle.
Input and source hashes are checked when this directory is reopened.
None of these visuals is a clean target.

If this ROI lacks useful panel lines, use different crop coordinates and
keep a separate immutable evidence directory.

## Execute G1 — human clean geometry reference

Open the **existing full-featured GUI**:

~~~bat
scripts\build\nsamdr.bat gui
~~~

Select **Raven Golden Patch**, then
**Open latest Golden Patch / trace editor**. Supply an author. Click along
the *intended continuous* panel centreline, not every square-pixel step.
Set its width in native pixels. Click **Finish line** and repeat as needed,
then **Save curated reference**.

The independent trace is rendered as smooth interpolating Catmull–Rom
geometry, not recovered by an edge detector. It creates:

- trace.json — author, native control points, immutable source SHA
- curated_contours_4x.png — clean human-defined geometry at exact 4×
- protected_regions_4x.png — regions whose original details must be retained
- curated_contours.provenance.json — source/trace/result hashes

The trace author must inspect and approve centreline placement.
A synthetically smooth line at the wrong position is not a correct reference.

For a hand-edited trace.json, run:

~~~bat
scripts\build\nsamdr.bat v178-golden-patch --action render-trace --folder "artifacts\nsamdr\golden_patch\cb1_t1_a_000_x320_y448_s256"
~~~

## Execute G2.S — synthetic structural proof

~~~bat
scripts\build\nsamdr.bat v178-contour-train --steps 400 --batch-size 4 --patch-size 32 --seed 17802 --device cuda --resume fresh
~~~

This independently trained 4× contour head predicts structural coverage,
not an RGB sharpening residual. The supervised loss includes geometric
coverage, tangent matching and curvature matching. Holdout samples are unseen
analytic geometry, never the real Raven Golden Patch. The synthetic pass gate
is at least 20% relative contour-loss improvement over the deterministic
observation baseline, with improvement over the untrained candidate.

Report: artifacts\nsamdr\diagnostics\v178_golden_contour\seed_17802_patch_32_batch_4\report.json

A synthetic pass never qualifies the full Raven.

## Execute G2.R — real single-patch capacity fit

After completing a human trace, select **Real Golden Patch capacity**
in the same GUI, or run:

~~~bat
scripts\build\nsamdr.bat v178-golden-fit --folder "artifacts\nsamdr\golden_patch\cb1_t1_a_000_x320_y448_s256" --device cuda --steps 300 --patch-size 48 --seed 17803 --resume fresh
~~~

The fit uses the entire native albedo region and UV-aligned features from
the other authentic native-resolution physical-map crops. A packed V5
material selector is an identity label, **not an authoritative geometric
contour**: the fit report records that semantic domain mismatch.

Under the Golden Patch directory, fit_runs contains:

- report.json — IoU, contour hit rate and false positives in protected areas
- golden_real_fit_comparison.png — A native, B colour baseline, C learnt
  contour overlay, T independently human-traced contour overlay
- candidate.pt — same-patch capacity checkpoint; not a production model

The provisional fit gate is contour IoU ≥ 0.65, hit rate ≥ 0.80 and protected
false-positive rate ≤ 0.02. It does not establish appearance enhancement
or generalisation to an unseen EVE texture, even if it passes.

## Stop conditions

1. Reject missing, mutated, or wrong-hash original physical-map evidence.
2. Refuse training on G1 until a real human trace is saved and verified.
3. Do not promote a synthetic-only success to a claim about Raven visuals.
4. If G2.R cannot fit one patch, fix the geometry representation and its
   supervision before extending the architecture.
5. Even after passing G2.R, **do not** proceed to full-ship neural
   qualification until several unseen authentic EVE patches have separately
   curated references and pass held-out evaluation.
6. Do not replace these gates with PSNR/SSIM, shader tweaks or quadtree work.

Frozen V17.6 evidence, historical checkpoints and sealed test partitions
remain unchanged.
