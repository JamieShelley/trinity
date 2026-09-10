# NSAMDR baseline-relative reconstruction contract

## Non-negotiable comparison

Every learned stage is judged against the deterministic reconstruction available from
the same degraded LR evidence.

- **A — authored source:** held-out HR target / real EVE authored texture.
- **B — deterministic 4x baseline:** bicubic albedo, normalized bilinear normal XY,
  nearest physical material channels.
- **C — current learned stage:** the exact stage candidate being trained, before a
  downstream selector can hide it.

Training is useful only when C improves on B while moving toward A. Production-final
remains a separate fail-closed authority.

## V12.4 baseline-centred specialist contract

The architectural anchor is B. Structure, seam/profile and appearance/detail are
bounded specialists around that anchor rather than an obligation to pass through one
serial repainting chain.

```text
                         +-- structure: B + delta geometry --+
LR authored maps -> B ---+-- seam/profile: B + delta seam ---+--> bounded fusion
                         +-- detail: B + delta detail --------+
                                                               |
                                                               v
                                                        BenefitSelector
                                                               |
                                                               v
                                                             FINAL
```

The responsibilities are intentionally different:

- structure corrects contour position/connectivity and downsampling stair-steps;
- seam/profile corrects fuzzy, over-wide, ringing or phase-damaged transitions;
- detail restores non-parametric high-frequency appearance and physical-map detail;
- BenefitSelector applies final local safety between B and the complete useful
  candidate.

A specialist that is unsupported, has zero authority or is not independently useful
contributes identity. It is not permitted to poison a different specialist that has
already demonstrated useful capacity.

V12.3 establishes this rule for detail in production: the existing detail network
produces a bounded direct residual over deterministic B, and learned geometry/seam
state cannot alter that direct-detail candidate. Structure and seam remain separately
auditable until they earn equivalent baseline-relative authority.

## Protected preservation

A reconstruction system should not pay for local improvement by repainting regions B
already reconstructs correctly. V12.4 therefore defines an explicit protected-region
metric for authored training/qualification data.

A pixel is **protected** when every albedo channel of B is within `2/255` of A. For
those protected pixels:

```text
protectedPreservationRate >= 0.990
candidateDriftTolerance   = 1/255 from B
```

The rate is the fraction of protected pixels that remain within that drift tolerance.
Mean and maximum protected drift are recorded separately so the remaining <=1% cannot
hide a large unbounded corruption.

A is used only to label protected pixels while training or qualifying. Production
inference never receives authored HR. Detail and final-selector training receive an
excess-drift penalty on those target-known protected pixels; the checkpoint topology is
unchanged.

## Historical contracts still in force

### V11.8 structural identity

The structural redraw is baseline-relative: an unearned structural correction must
reduce to B. The old signed structural gain may not invert a bad geometry proposal to
manufacture an apparent improvement.

### V11.9 B1 authority

B1a establishes topology with structural residual authority at identity. B1b freezes
that topology and must earn strict improvement over B using the actual structural
candidate. Equality is safe during topology bootstrap but is not a B1b success.

### V11.10 structural-stage consumer

B1 is evaluated on the pre-seam structural output. Frozen downstream seam/detail
components cannot contribute to B1 acceptance evidence. A structural failure cannot be
hidden by a later appearance stage.

### V12.0 estimator/refiner separation

Continuous geometry is not treated as a one-shot raster prediction. A neural branch
proposes fixed connected topology plus initial continuous crossing/tangent parameters.
A separate parameter-free explicit refiner optimizes only those continuous parameters
against observed LR structural evidence while remaining bounded around the proposal.
Topology is immutable in that refinement step and authored HR is not an inference
input.

### V12.2 authority alignment

Training and deployment must supervise the exact outputs that the corresponding heads
control. The explicit refiner uses a first-order differentiable training path so loss
on the refined rendered geometry reaches the neural initializer. PhaseAwareSeamSR uses
one learned seam authority. BenefitSelector is the final residual authority rather than
a product of unrelated veto gates.

### V12.2.4 B1 production objective

B1b SGD is driven by the actual rendered structural candidate in source-observable
support. Legacy point/tangent/proxy objectives remain telemetry if they conflict with
rendered production quality. HR-only appearance detail is not a geometry target.

### V12.3 independent direct detail

The detail candidate is generated from deterministic B rather than the serial
geometry/seam candidate. Existing parameters and state-dict keys are retained. This
prevents a weak structural specialist from destroying a detail path that independently
beats B.

## Qualification ladder

Use the cheapest proof that can invalidate the current hypothesis:

1. identity and protected preservation;
2. direct detail capacity on fixed real Raven evidence;
3. structural capacity on exact rendered C versus B;
4. seam/profile capacity independent of unqualified upstream specialists;
5. bounded specialist fusion and retention of each demonstrated gain;
6. Raven Quick using the complete production model;
7. Full Training only after the earlier gates are stable.

Synthetic geometry, teacher geometry and oracle authorities are useful diagnostic tools.
They establish representation capacity but never count as production qualification.

## Training/inference boundary

Training may use A for loss, teacher signals and qualification metrics. The public
production call remains:

```python
outputs = model(inputs)
```

Production callers cannot provide A, replace structural geometry, force gates, inject
cached intermediate tensors or choose a different Raven-only model. A selected
checkpoint must strict-load into the production model and survive a fresh direct
forward with the same output contract.

## Literature rationale carried forward

1. Residual SR systems such as VDSR, LapSRN and SwinIR preserve a known low-frequency
   path and learn missing correction rather than repainting the complete output.
2. Deep Vectorization of Technical Drawings separates neural estimation from explicit
   geometric optimization; this remains the right structural decomposition when final
   vector parameters need refinement.
3. End-to-End Line Drawing Vectorization supports representing connectivity explicitly
   rather than relying on raster penalties to recover it.
4. DiffVG demonstrates differentiable anti-aliased vector rasterization but does not
   solve discrete topology changes.
5. LIVE reinforces that low raster error alone is not a topology guarantee.
6. Long smoothing B-splines motivate derivative/curve priors only where they do not
   erase intentional manufactured corners, junctions, bevels or kinks.

## Design rule

Do not relax a qualification gate merely because a serial dependency makes a later
candidate look poor. First ask whether each specialist independently improves the same
baseline B. Only independently useful corrections are eligible for fusion.
