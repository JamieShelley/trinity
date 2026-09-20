# NSAMDR

**Neural Structure-Aware Material Detail Reconstruction** reconstructs aligned EVE ship material textures at **4x** while preserving authored structure and avoiding unsupported invention.

[![Visual target](./EXAMPLE.png)](./EXAMPLE.png)

`EXAMPLE.png` is the visual target. `NSAMDR_FULL_SYSTEM_ARCHITECTURE.png` is the long-term production-system target; experimental branches must still earn their place through held-out evidence.

## Production contract

```text
A = authored HR target, training/qualification only
B = deterministic 4x baseline
C = learned physical-map reconstruction over B
F = BenefitSelector result choosing locally between B and C

C = project(B + predicted residual)
F = B + selector * (C - B)
```

NSAMDR must recover supported seams, panel boundaries, contours and manufactured detail; keep albedo/normal/material aligned; avoid unsupported invention; preserve correct deterministic regions; and fall back to B when C is worse.

Final protected-region requirement:

```text
protectedPreservationRate >= 0.990
```

## Current evidence

The V16 Swin reconstruction body is **locally capable but not multi-family qualified**.

Capacity passes the unchanged gates:

```text
global recovery   >= 45%
edge recovery     >= 60%
gradient recovery >= 35%
lattice excess    <= 15%
```

Four-family Stage 2 fails held-out Raven/Golem/Rattlesnake generalisation while Rokh is much stronger. Earlier recoverability, context and loss experiments did not remove that generalisation gap.

### Rejected structural formulations

The first structure audit reported about 99.9% of B->A error inside a generously dilated boundary band and about 99.1% LR boundary support. That result was useful but not sufficient because it did not report how much image area the band occupied. `audit_nsamdr_v16_structure_support.py` is now V3 and measures boundary area, threshold sweeps and **error enrichment** instead of treating raw in-band error fraction as decisive.

Two explicit local reconstruction ideas have failed held-out tests:

```text
geometry-class residual oracle        median recovery  -5.5%
BoundaryProfile oracle                median recovery -23.0%
BoundaryProfile edge recovery                         -12.6%
```

Therefore `BoundaryProfileNet` is retired. Its constants and CPU audit remain only to preserve negative evidence and reproducibility.

### Broad authored authority

The EVE ship-corpus census found **345 independent authored albedo+normal authorities**:

```text
native 2048 tier   16
native 1024 tier  320
native  512 tier    8
native  256 tier    1
--------------------
total              345
```

The current broad-prior corpus uses all **336 authorities at >=1K** and stores two native-authored 512x512 HR crops per authority:

```text
train authorities / crops       298 / 596
held-out authorities / crops      38 / 76
split unit                        complete authored authority
crop leakage                      forbidden
```

This gives genuine `128 -> 512` 4x proof data across hundreds of authorities. The 16 >=2K authorities can later support genuine `512 -> 2048`. The current ship corpus contains no genuine >=4K-square authority, so it cannot provide authored-pixel ground truth for `1024 -> 4096`.

Material semantics are not claimed by the fast census. Broad-prior qualification is therefore driven by authored albedo+normal; material remains telemetry until complete SOF/material authority is resolved.

## Active V16.2 experiment

The active experiment no longer renders an explicit geometry/profile intermediate. Structure is used only as learned conditioning:

```text
LR authored maps (8 channels)
        |
        +-------------------------------> deterministic baseline B
        |
        +------> appearance context ------------------+
        |                                            |
        +------> StructureConditioningEncoder         |
                   + analytic albedo edge evidence    |
                   + analytic normal edge evidence    |
                   + analytic material edge evidence  |
                   + learned continuity/context       |
                   + phase-neutral bilinear resize    |
                                                      v
                                            feature fusion
                                                      |
                                                      v
                                         existing V16 Swin body
                                                      |
                                                      v
                                                candidate C
                                                      |
                                                      v
                                  Confidence / Regret -> BenefitSelector -> F
```

The structure branch emits **features only**. It has no physical-map pixel head, no profile renderer, no external SDF authority, no PixelShuffle and no transposed convolution. Its fusion projection is zero-initialized so the conditioned model begins as the exact control graph and must learn any structural contribution.

## Broad-prior probe evidence

The authority-balanced reduced-model proof is complete through 4096 updates across all 38 held-out authorities. It establishes two facts:

```text
broad authored prior        useful
structure conditioning      useful but modest
```

At 4096 updates:

```text
control global / edge        10.22% /  8.47%
conditioned global / edge    11.43% /  8.94%
conditioned - control        +1.20pp / +0.46pp
conditioned gradient          9.15%
conditioned normal           14.00%
conditioned lattice excess   24.90%
```

The conditioned model remains better than the control on global, edge, gradient, normal and lattice metrics, but the extra conditioning advantage is no longer growing consistently. The reduced proof has therefore served its purpose and should not be extended indefinitely.

The production-size V16 body is now trained through **596 updates**: 96 HR channels, 6 residual-Swin groups x 6 blocks, 6 heads, 8x8 windows, full LR context, and structure conditioning.

Current held-out result at 596:

```text
global recovery      5.38%
edge recovery        4.89%
gradient recovery    4.53%
normal recovery     11.86%
lattice excess      52.01%
```

The 596 diagnostic now shows that seen and held-out authorities fail at nearly the same level:

```text
                         seen       held-out
global recovery           4.68%        5.38%
edge recovery             4.47%        4.89%
gradient recovery         4.16%        4.53%
normal recovery          10.43%       11.86%
lattice excess           53.64%       52.01%
```

This is not primarily a held-out generalisation gap at the current checkpoint. The production candidate is not fitting the broad seen-authority mapping either.

Residual telemetry shows a median applied/target albedo residual ratio of about 0.25-0.27 while residual-cap saturation is 0%. A no-training unit-slope ablation, `cap*tanh(raw/cap)`, increased residual amplitude but reduced global recovery on both seen and held-out authorities. It improved gradient recovery only. Therefore simple residual amplification is rejected.

The residual-alignment diagnostic is complete. Median residual cosine alignment is only about 0.32 seen / 0.37 held-out, with target-weighted sign agreement about 65% / 67%. A target-informed per-sample least-squares scalar oracle does not rescue global reconstruction: seen global recovery falls from 4.68% to 3.95%, and held-out falls from 5.38% to 4.73%. Therefore residual amplitude calibration is not the blocker; the learned residual direction/spatial support is wrong.

The exact single-authority memorization probe is complete and passes all candidate gates on authority `13006d2b807f89ac`. At update 320 it reaches 59.98% global, 61.44% edge, 50.53% gradient and 5.09% lattice excess; at update 384 it improves to 62.82% global, 64.37% edge, 53.83% gradient and 3.82% lattice excess. Residual cosine alignment reaches 0.92 and target-weighted sign agreement 94.90%, ruling out a fundamental single-target representation/capacity failure.

The fixed 4-authority interference probe is complete through 448 updates per authority. This is a **train-authority fixed-fit diagnostic**, not held-out qualification: it uses one deterministic center crop from each selected train authority, disables augmentation, and asks whether mixing authorities prevents the same model/objective from fitting them.

The median trajectory is:

```text
updates/authority    global     edge     gradient   lattice
0                     2.70%     1.83%      2.76%    53.96%
64                   21.62%    17.91%     17.30%    20.81%
128                  33.65%    31.11%     27.90%    13.77%
256                  45.95%    47.01%     38.41%     6.63%
320                  53.56%    54.50%     43.98%     4.40%
384                  PASS median candidate gates
448                  60.33%    61.96%     50.62%     4.78%
```

At 448, all four median candidate gates pass. All four authorities pass global, gradient and lattice; three of four pass the 60% edge threshold. The aggregate median edge is 61.96%, global 60.33%, gradient 50.62%, and lattice 4.78%. This proves the current full-size model/objective can fit four distinct authored authorities simultaneously under deterministic fixed-crop training.

The known anchor remains the controlled interference comparison because the single-authority and mixed-authority probes use the same source checkpoint, exact fixed crop, loss, metrics and fresh-Adam start. By 448 mixed updates per authority the anchor reaches 60.55% global / 62.89% edge / 52.33% gradient / 4.65% lattice, essentially converging on the successful single-authority control. This supports **optimization slowdown with authority mixing**, not a hard four-authority capacity limit. It still does not prove held-out generalisation.

The 16-authority fixed-fit scaling probe is complete through 256 updates per authority and remains healthy. Median recovery progresses as follows:

```text
updates/authority    global     edge     gradient   lattice
64                   25.60%    28.63%     25.87%    15.36%
128                  38.55%    40.12%     33.53%     9.10%
192                  46.75%    48.57%     41.96%     6.52%
256                  51.26%    53.71%     46.51%     7.11%
```

At 256, median global, gradient and lattice all pass. Thirteen of sixteen authorities pass global, thirteen pass gradient, all sixteen pass lattice, and edge is the only remaining median gate. The best edge result is 59.93%, within 0.08 percentage points of the 60% threshold.

The shared anchor also continues improving: at 256 mixed-authority updates it reaches 53.71% global / 56.21% edge / 45.72% gradient / 5.57% lattice, with residual cosine 0.88 and target-weighted sign agreement 92.12%. The residual direction remains coherent and the authority mix is not erasing the anchor.

The 16-authority train-fit problem is now specifically an edge-recovery convergence question rather than a general reconstruction-capacity problem. Continue the same saved 16-authority checkpoint to 320 and 384 updates per authority. Do not change the architecture, increase authority count, or resume broad training until this edge-only question is closed.

Each full-capacity checkpoint saves fixed held-out visual samples under `previews/step_NNNNNN/`. `--preview-only --resume <checkpoint>` now reports current seen/held-out metrics plus the unit-slope residual-bound ablation.

## Active qualification ladder

```text
1. V16 Capacity                                      PASS
2. Four-family Stage 2                              FAIL generalisation
3. Raw recoverability/context/loss diagnostics      FAIL to solve gap
4. Full EVE authored corpus census                  PASS
5. Geometry-class residual oracle                   FAIL
6. Boundary Profile oracle                          FAIL
7. Structure Support Audit V3                       supporting CPU evidence
8. Broad authored-prior corpus                      PASS
   -> 298 train authorities
   -> 38 complete held-out authorities
9. Reduced authority-balanced conditioning proof    PASS as direction evidence
   -> complete through 4096
   -> conditioned remains better than control
   -> conditioning advantage is modest / plateauing
10. Full-capacity broad-authority V16.2 proof        CURRENT
    -> production-size 96ch / 6x6 Swin candidate
    -> current checkpoint 596 updates
    -> held-out reconstruction remains about 5%
    -> lattice excess remains about 52%
    -> seen and held-out recovery are both about 5%
    -> candidate applies only about one quarter of target residual magnitude
    -> unit-slope residual amplification rejected
    -> residual alignment low: about 0.32 seen / 0.37 held-out cosine
    -> per-sample scalar oracle cannot rescue global recovery
    -> exact one-authority memorization PASS
    -> 320 updates: global 60.0%, edge 61.4%, gradient 50.5%, lattice 5.1%
    -> 384 updates: global 62.8%, edge 64.4%, gradient 53.8%, lattice 3.8%
    -> residual cosine 0.92, weighted sign agreement 94.9%
    -> fundamental single-target representation/capacity failure ruled out
    -> fixed 4-authority train-fit interference diagnostic PASS
    -> median candidate gates pass by 384 and remain passed at 448
    -> median @448: global 60.3%, edge 62.0%, gradient 50.6%, lattice 4.8%
    -> 4/4 global PASS, 4/4 gradient PASS, 4/4 lattice PASS, 3/4 edge PASS
    -> four-authority hard capacity limit ruled out
    -> authority mixing mainly slows optimization at this scale
    -> held-out generalisation remains unproven
    -> 16-authority fixed-fit scaling remains healthy through 256 updates/authority
    -> median @256: global 51.3% PASS, edge 53.7% MISS, gradient 46.5% PASS, lattice 7.1% PASS
    -> 13/16 global PASS, 13/16 gradient PASS, 16/16 lattice PASS
    -> best edge 59.93%; edge is the only remaining median train-fit gate
    -> anchor @256: global 53.7%, edge 56.2%, gradient 45.7%, lattice 5.6%
    -> no evidence of destructive authority mixing or hard capacity collapse
    -> next: continue same 16-authority checkpoint to 320/384 updates per authority
    -> do not expand authority count or resume broad training to 894 yet
11. Full Stage 2 qualification
12. BenefitSelector qualification
13. Raven / production Quick
14. Highest-native-resolution renderer proof
```

Do not extend the reduced proof to 8192. Do not resume the full-capacity model to 894. The unit-slope and scalar-oracle diagnostics are rejected as fixes. Exact single-authority memorization and fixed 4-authority train-fit pass the candidate gate set. The 16-authority fixed-fit scaling probe now passes median global, gradient and lattice through 256 updates per authority; edge is the only remaining median gate at 53.71%, with the best authority already at 59.93%. Continue the same saved 16-authority checkpoint to 320/384 updates per authority to close this edge-only train-fit question. This remains train-fit evidence and must not be substituted for later complete-authority held-out qualification.

## Candidate qualification gates

These remain unchanged:

```text
global recovery   >= 45%
edge recovery     >= 60%
gradient recovery >= 35%
lattice excess    <= 15%
minimum held-out samples = 4
```

BenefitSelector trains only after C qualifies and must retain:

```text
edge recovery retention   >= 90%
global recovery retention >= 90%
protected preservation    >= 99%
```

## Data contract

Only genuine authored resolution counts as supervision:

```text
native 4096 -> genuine 1024 -> 4096
native 2048 -> genuine  512 -> 2048
native 1024 -> genuine  256 -> 1024
```

The broad diagnostic corpus stores 512px native-authored crops and can train smaller genuine 4x sub-crops such as `128 -> 512` without inventing HR truth.

## Source layout

The proven V16 reconstruction body remains under `tools/nsamdr/neural/v14/` for active checkpoint/import compatibility.

Current V16.2 code:

```text
tools/nsamdr/neural/v16/
    structure.py       analytic + learned structure conditioning
    conditioning.py    identity-initialized structure-conditioned V16 wrapper
    broad_prior.py     complete-authority balanced sampling
    stage2_data.py     shared Stage 2 diagnostic data/baseline access
    profiles.py        constants only for rejected profile audit
```

Active evidence/data tools:

```text
scan_eve_authored_corpus.py
prepare_nsamdr_v16_authored_prior_corpus.py
audit_nsamdr_v16_structure_support.py
audit_nsamdr_v16_boundary_profiles.py
probe_nsamdr_v16_structure_conditioning.py
probe_nsamdr_v16_full_broad.py
probe_nsamdr_v16_memorization.py
probe_nsamdr_v16_interference.py
```

Historical V9-V13 model/training implementations are retired. A minimal `v9/` compatibility package remains only because active authored-data preparation still imports its manifest/config names.

## Commands

```bat
scripts\build\nsamdr.bat gui
scripts\build\nsamdr.bat eve-census
scripts\build\nsamdr.bat structure-audit
scripts\build\nsamdr.bat boundary-profile-audit
scripts\build\nsamdr.bat authored-prior-corpus
scripts\build\nsamdr.bat structure-conditioning-probe --device cuda
scripts\build\nsamdr.bat full-broad-probe --device cuda
scripts\build\nsamdr.bat full-broad-probe --device cuda --preview-only --resume <checkpoint>
scripts\build\nsamdr.bat memorization-probe --device cuda --resume <checkpoint> --authority-id <train-authority>
scripts\build\nsamdr.bat memorization-probe --device cuda --resume <checkpoint> --authority-id <train-authority> --stages 64,128,256
scripts\build\nsamdr.bat memorization-probe --device cuda --resume <checkpoint> --continue-from <memorization-checkpoint> --authority-id <train-authority> --stages 128,256
scripts\build\nsamdr.bat interference-probe --device cuda --resume <checkpoint> --authority-count 4 --stages-per-authority 64 128
scripts\build\nsamdr.bat interference-probe --device cuda --resume <checkpoint> --authority-count 16 --stages-per-authority 64
scripts\build\nsamdr.bat interference-probe --device cuda --resume <checkpoint> --continue-from <16-authority-checkpoint> --authority-count 16 --stages-per-authority 128
scripts\build\nsamdr.bat interference-probe --device cuda --resume <checkpoint> --continue-from <16-authority-checkpoint> --authority-count 16 --stages-per-authority 192 256
scripts\build\nsamdr.bat interference-probe --device cuda --resume <checkpoint> --continue-from <16-authority-checkpoint> --authority-count 16 --stages-per-authority 320 384
scripts\build\nsamdr.bat stage2-status
scripts\build\nsamdr.bat stage2-probe
scripts\build\nsamdr.bat stage2-summary
```

The reduced structure-conditioning probe remains available for reproducibility. The active full-capacity broad checkpoint is 596. Residual alignment and scalar-oracle diagnostics are complete and do not justify more broad training. Exact single-authority memorization and fixed 4-authority train-fit pass the candidate gate set. The 16-authority fixed-fit probe is healthy through 256 updates per authority, with only median edge recovery still below gate. Continue only the same saved 16-authority checkpoint to 320/384. Do not resume the 894-step broad run.

## Final visual proof

Qualification is not metric-only. With the same ship, camera, lighting, LOD and shader, the final system must show cleaner continuous seams and contours, sharper supported panel boundaries, restored narrow manufactured features, reduced stair-stepping, aligned normal/material detail, no obvious LR-grid blocks and no invented generic AI texture.

If that renderer-level improvement is not visible, the production goal is not complete.
