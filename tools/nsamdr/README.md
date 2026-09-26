# NSAMDR

**Neural Structure-Aware Material Detail Reconstruction** reconstructs aligned EVE ship material textures at **4x** while preserving authored structure and avoiding unsupported invention.

[![Visual target](./EXAMPLE.png)](./EXAMPLE.png)

## Quick start / operator guide

Run all commands from the Trinity repository root:

```bat
D:\REPOS\trinity
```

### Launch the current GUI

The canonical launcher is:

```bat
scripts\build\nsamdr.bat gui
```

This opens the restored operator workflow GUI:

```text
tools/nsamdr/gui/nsamdr_v16_workflow_gui.py
```

The operator GUI again presents the normal workflow as **environment -> Raven
Quick -> Main V16 Training -> Preview**, with stage controls, command preview,
runtime progress/log output, experiment detection and qualified-preview
selection.

The previous structure-only V16.2 GUI is retained as an advanced diagnostics
surface:

```text
tools/nsamdr/gui/nsamdr_v16_structure_workflow_gui.py
```

Use **Advanced diagnostics** in the operator GUI for the Stage 2 runtime,
structure-support audit, boundary-profile audit and corpus-census controls.

Main V16 Training is intentionally shown but locked until the current
broad-authority D4 recipe is promoted into the canonical workflow. The GUI will
not silently route that stage to the obsolete V9 full-training implementation.

### Validate the active checkout

```bat
scripts\build\nsamdr.bat validate
```

Layout-only check:

```bat
scripts\build\nsamdr.bat validate --layout-only
```

### Environment setup

CUDA environment:

```bat
scripts\build\nsamdr.bat setup cuda
```

CPU environment:

```bat
scripts\build\nsamdr.bat setup cpu
```

Force rebuild of either environment:

```bat
scripts\build\nsamdr.bat setup cuda --force
scripts\build\nsamdr.bat setup cpu --force
```

### Raven development workflow

Prepare/index the deterministic Raven dataset:

```bat
scripts\build\nsamdr.bat index raven
```

Rebuild it explicitly:

```bat
scripts\build\nsamdr.bat index raven --rebuild
```

Run the Raven Quick qualification baseline:

```bat
scripts\build\nsamdr.bat raven-quick
```

The CLI performs source-freshness and CUDA preflight checks before training.

Raven Quick is currently the **legacy deterministic V16.0 Raven baseline**. It
does not yet include the D4 + broad-authority recipe being qualified in the
active V16.2 diagnostics. Exit code `2` after all SR epochs can therefore mean
**candidate qualification rejected**, not a software/runtime failure. The
operator GUI reports that state as `rejected` and can open the latest
diagnostic `A/B/C/F` training preview even when the candidate is not
production-qualified.

Preview an existing experiment:

```bat
scripts\build\nsamdr.bat preview EXP_####
```

Optional preview controls:

```bat
scripts\build\nsamdr.bat preview EXP_#### --device cuda
scripts\build\nsamdr.bat preview EXP_#### --target-size 2048 --device cuda
```

### Native preview utilities

Build the isolated native preview target:

```bat
scripts\build\nsamdr.bat native build
```

Launch the OBJ preview bridge with forwarded arguments:

```bat
scripts\build\nsamdr.bat native obj <arguments>
```

Prepare and launch the default EVE asset preview:

```bat
scripts\build\nsamdr.bat native eve
```

### Active V16 diagnostic commands

The commands used by the current V16 evidence ladder are:

```bat
scripts\build\nsamdr.bat eve-census
scripts\build\nsamdr.bat structure-audit
scripts\build\nsamdr.bat boundary-profile-audit
scripts\build\nsamdr.bat authored-prior-corpus
scripts\build\nsamdr.bat structure-conditioning-probe
scripts\build\nsamdr.bat full-broad-probe
scripts\build\nsamdr.bat memorization-probe
scripts\build\nsamdr.bat interference-probe
scripts\build\nsamdr.bat heldout-transfer-probe
scripts\build\nsamdr.bat sibling-crop-transfer-probe
scripts\build\nsamdr.bat two-crop-fit-probe
scripts\build\nsamdr.bat augmented-two-crop-fit-probe
scripts\build\nsamdr.bat stage2-status
scripts\build\nsamdr.bat stage2-probe
scripts\build\nsamdr.bat stage2-summary
```

Most diagnostic commands require their own arguments/checkpoint paths; the
specific active command is recorded beside the relevant experiment in this
README.

### Current GUI/production status

The historical README described the older V9 production implementation and a
`full-train` command. The operator-style GUI has now been restored for V16.2,
but the old V9 full-training command has **not** been revived. Main V16 Training
remains locked until the current broad-authority/D4 recipe is promoted into the
canonical training workflow.

The intended end state remains the same operator flow:

```text
GUI
 -> prepare/select authored data
 -> train the production candidate
 -> qualify C
 -> train/qualify BenefitSelector
 -> generate immutable preview artifacts
 -> launch A RAW SOURCE vs B NSAMDR FINAL
```

The current work is closing the training recipe before that main GUI path is
re-enabled.

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

The 16-authority fixed-fit scaling probe is complete through 448 updates per authority and **passes all median candidate gates**. Median recovery progresses as follows:

```text
updates/authority    global     edge     gradient   lattice
64                   25.60%    28.63%     25.87%    15.36%
128                  38.55%    40.12%     33.53%     9.10%
192                  46.75%    48.57%     41.96%     6.52%
256                  51.26%    53.71%     46.51%     7.11%
320                  55.89%    59.39%     49.71%     5.78%
384                  56.16%    59.94%     51.38%     5.65%
448                  59.81%    64.86%     54.48%     3.47%
```

At 448, all sixteen authorities pass global, gradient and lattice; thirteen of sixteen pass the 60% edge gate. Median global is 59.81%, median edge 64.86%, median gradient 54.48%, and median lattice excess 3.47%. The anchor also passes every candidate gate at 60.83% global / 63.39% edge / 51.57% gradient / 3.98% lattice.

The train-fit capacity question is now closed at 1, 4 and 16 authorities. The model can learn the required authored residual direction across a non-trivial mixed authority set without lattice collapse. This still does not prove production generalisation.

The **independent held-out transfer** diagnostic is complete on all 38 complete validation authorities, comparing the original step-596 broad checkpoint against the 448-update 16-authority train-fit checkpoint with no further training.

```text
held-out median             source @596   16-authority @448   delta
global recovery                5.38%            7.73%          +2.34pp
edge recovery                  4.89%            9.44%          +4.55pp
gradient recovery              4.53%           13.64%          +9.11pp
normal recovery               11.86%           17.16%          +5.30pp
lattice excess                52.01%            8.30%         -43.71pp
detail recovery 1px            2.07%           -7.12%          -9.20pp
```

This is real positive transfer, but it is far below candidate qualification. The learned 16-authority model strongly suppresses lattice structure and improves global/edge/gradient/normal recovery on unseen authorities, while the finest 1px detail regresses. Residual magnitude transfer becomes much stronger (candidate/target ratio 0.76 versus 0.27 at the source), but residual cosine is still only about 0.47 and sign agreement about 72.5%. A target-informed scalar oracle prefers a median gain around 0.56 and improves held-out global/edge only to about 12.5% / 12.7%, so simple amplitude tuning still cannot close the gap.

The **same-authority unseen sibling-crop transfer** diagnostic is complete and shows substantial crop-specific overfit. On the exact trained first crop, the 16-authority checkpoint retains the successful 59.81% global / 64.86% edge / 54.48% gradient / 3.47% lattice medians. On the unseen second crop from those same authorities, recovery falls to 14.49% global / 17.47% edge / 22.35% gradient / 8.25% lattice. The unseen sibling crop still improves materially over the source checkpoint (5.14% global / 4.35% edge / 3.41% gradient / 55.08% lattice), so some structure transfers within an authority, but most of the train-fit gain is crop-specific.

The finest detail confirms the same failure mode: median 1px detail recovery is about +50.83% on the trained crop, +1.67% on the source sibling crop, and -0.87% on the candidate sibling crop. This is directly relevant to thin seams and panel-boundary sharpness: the model learns them strongly on the trained spatial sample but does not yet transfer that finest structure reliably to another authored region of the same ship.

The bounded **two fixed crops per authority** diagnostic is complete through the matched 7168-update budget.

```text
                         112/crop       224/crop
total updates               3584           7168

32 fixed train crops:
global recovery            31.84%         41.81%
edge recovery              31.14%         43.77%
gradient recovery          30.82%         38.43%
1px detail recovery        19.99%         29.24%
lattice excess              7.43%          3.77%

38 held-out authorities:
global recovery             9.10%          8.37%
edge recovery               6.60%          6.17%
gradient recovery          12.69%         12.62%
1px detail recovery        -4.30%         -7.74%
lattice excess              8.91%          8.06%
```

The fixed train crops continue improving strongly from 112 to 224 updates/crop, but held-out recovery does not. Global, edge and gradient all fall slightly, 1px detail degrades from -4.30% to -7.74%, and only lattice improves. Train residual cosine rises from about 0.72 to 0.81 while held-out cosine falls from about 0.48 to 0.46; train residual magnitude approaches the target while held-out alignment worsens. This is direct evidence that simply fitting the same 32 fixed spatial samples harder increases crop-specific overfit.

At the matched 7168-update budget, two fixed crops do not materially beat the prior one-crop @448 transfer. Two-crop global is slightly higher (8.37% vs 7.73%) and lattice slightly lower (8.06% vs 8.30%), but edge (6.17% vs 9.44%), gradient (12.62% vs 13.64%) and 1px detail (-7.74% vs -7.12%) are worse. Do not extend the fixed two-crop run.

The **D4-augmented two-crop diagnostic** is complete through the matched 7168-update budget on the same 16 authorities and two authored crops. Each crop cycles through all eight rotation/reflection variants with normal XY vectors transformed consistently.

```text
                         112/crop       224/crop
total updates               3584           7168

32 fixed-orientation train evaluations:
global recovery            21.78%         28.83%
edge recovery              19.77%         31.34%
gradient recovery          19.84%         27.46%
1px detail recovery        10.95%         17.60%
lattice excess             12.59%         11.70%

38 held-out authorities:
global recovery            15.85%         15.73%
edge recovery              13.61%         15.76%
gradient recovery          14.38%         15.22%
1px detail recovery         2.67%          1.48%
lattice excess             13.36%         10.58%
```

D4 remains substantially better than fixed-crop training at the matched budget. Relative to the fixed two-crop 224 result, held-out global improves from 8.37% to 15.73%, edge from 6.17% to 15.76%, gradient from 12.62% to 15.22%, and 1px detail from -7.74% to +1.48%. Median lattice is higher than the fixed run (10.58% vs 8.06%) but remains below the 15% median diagnostic threshold.

From D4 112 -> 224, train recovery continues rising while held-out global is flat, edge and gradient improve modestly, 1px detail falls from +2.67% to +1.48%, and median lattice improves from 13.36% to 10.58%. Held-out residual cosine rises from about 0.52 to 0.54, candidate/target residual ratio from about 0.54 to 0.66, and sign agreement to about 76.1%. This is not the severe fixed-crop collapse, but the 16-authority D4 run is now plateauing.

Production qualification is still far away. The held-out global/edge/gradient medians remain well below 45/60/35%, production-style maximum lattice is about 29.25% (>15%), and worst global recovery is about -53.9% (< -10%). Do not extend the 16-authority D4 checkpoint further.

The **32-authority D4 diversity diagnostic** is complete through 112 updates per crop (7168 total updates). It preserves the original 16-authority set as the deterministic prefix, adds 16 independent train authorities, keeps two authored crops per authority and the same eight D4 transforms.

```text
matched total budget: 7168 updates

                         16 auth D4 @224   32 auth D4 @112
held global                   15.73%            17.39%
held edge                     15.76%            18.09%
held gradient                 15.22%            16.85%
held 1px detail                1.48%             5.92%
held 2px detail                8.19%            12.81%
held 4px detail               18.04%            19.15%
held normal                   23.62%            24.03%
held lattice                  10.58%            16.70%
max lattice                   29.25%            27.85%
```

At identical total compute, increasing authority diversity improves every held-out reconstruction/detail metric, especially edge (+2.33pp) and 1px detail (+4.44pp). Positive fractions also strengthen: global and edge reach about 92.1%, gradient about 97.4%, and 1px detail about 81.6%. The worst global sample improves from about -53.9% to -43.9%, but remains far outside production tolerance.

The 32-authority continuation from 56 -> 112 updates/crop also keeps improving rather than collapsing: global rises about 16.10 -> 17.39, edge 17.24 -> 18.09, gradient 14.96 -> 16.85 and median lattice improves about 18.61 -> 16.70. Residual cosine rises to about 0.553 and weighted sign agreement to about 77.0%. This confirms that the broader-authority model is still optimization-limited rather than showing the severe fixed-crop overfit seen earlier.

The remaining trade-off is lattice and absolute qualification distance. Median lattice is still 16.70% and production-style maximum lattice is about 27.85%; global/edge/gradient are still far below 45/60/35%, and worst global remains about -43.9%. The 64 fixed-orientation train crops are also still under-fit at about 19.4% global / 20.2% edge / 17.6% gradient, with median lattice about 18.2%.

This closes the bounded authority-diversity decision. **D4 augmentation plus broader independent authority exposure is the training direction to promote into the canonical V16 preview-training workflow.** Do not spend another cycle on 16/32-authority fixed-budget microprobes. The next engineering task is to wire the proven augmentation/authority-balanced recipe into Main V16 Training and produce a research preview while keeping production qualification gates unchanged.

Matched held-out preview panels should still be inspected before any claim that 1px panel-boundary quality is visually solved.

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
    -> 16-authority fixed-fit scaling PASS at 448 updates/authority
    -> median @448: global 59.8% PASS, edge 64.9% PASS, gradient 54.5% PASS, lattice 3.47% PASS
    -> 16/16 global PASS, 16/16 gradient PASS, 16/16 lattice PASS, 13/16 edge PASS
    -> anchor @448: global 60.8%, edge 63.4%, gradient 51.6%, lattice 3.98%
    -> train-fit capacity demonstrated at 1, 4 and 16 authorities
    -> independent held-out transfer on 38 authorities: positive but insufficient
    -> source -> 16-fit held-out: global 5.38 -> 7.73, edge 4.89 -> 9.44, gradient 4.53 -> 13.64
    -> lattice excess improves 52.01 -> 8.30
    -> 1px detail regresses 2.07 -> -7.12
    -> residual cosine improves 0.37 -> 0.47; sign agreement 66.8% -> 72.5%
    -> same-authority sibling-crop transfer confirms strong crop-specific overfit
    -> trained crop: global 59.81, edge 64.86, gradient 54.48, lattice 3.47
    -> unseen sibling: global 14.49, edge 17.47, gradient 22.35, lattice 8.25
    -> sibling source: global 5.14, edge 4.35, gradient 3.41, lattice 55.08
    -> 1px detail: trained +50.83, sibling source +1.67, sibling candidate -0.87
    -> two-crop fixed diagnostic complete through 224/crop = 7168 total updates
    -> train 112 -> 224: global 31.84 -> 41.81, edge 31.14 -> 43.77, gradient 30.82 -> 38.43
    -> held 112 -> 224: global 9.10 -> 8.37, edge 6.60 -> 6.17, gradient 12.69 -> 12.62
    -> held 1px detail worsens -4.30 -> -7.74 while lattice improves 8.91 -> 8.06
    -> fixed-crop overfit confirmed: train fit rises while held-out alignment/recovery stagnates or regresses
    -> matched budget does not beat one-crop @448 overall
    -> D4 two-crop @112/crop: held global 15.85, edge 13.61, gradient 14.38, 1px +2.67, lattice 13.36
    -> D4 two-crop @224/crop: held global 15.73, edge 15.76, gradient 15.22, 1px +1.48, lattice 10.58
    -> D4 clearly beats fixed two-crop at the matched 7168-update budget
    -> 112 -> 224: global flat, edge/gradient modestly higher, 1px lower, lattice better
    -> held residual cosine ~0.54; candidate/target ratio ~0.66; sign agreement ~76.1%
    -> production-style max lattice still fails at ~29.25%; worst global ~-53.9%
    -> 16-authority D4 is plateauing; do not extend it further
    -> 32-authority D4 @56/crop = 3584 total: diversity improves edge/1px at matched compute
    -> 32-authority D4 @112/crop = 7168 total complete
    -> vs 16-authority D4 @224 at same budget:
       global 15.73 -> 17.39, edge 15.76 -> 18.09
       gradient 15.22 -> 16.85, 1px +1.48 -> +5.92
       median lattice 10.58 -> 16.70, max lattice 29.25 -> 27.85
    -> global/edge positive fractions ~92.1%, gradient ~97.4%, 1px ~81.6%
    -> 64 train crops remain under-fit at ~19-20% global/edge and ~18.2% lattice
    -> authority diversity decision CLOSED: broader authority exposure is beneficial
    -> next: promote D4 + authority-balanced training into canonical Main V16 Training
    -> produce research preview without weakening production qualification gates
    -> do not resume broad596 -> 894 or change architecture/loss first
11. Full Stage 2 qualification
12. BenefitSelector qualification
13. Raven / production Quick
14. Highest-native-resolution renderer proof
```

Do not extend the reduced proof to 8192. Do not resume the full-capacity model from 596 to 894. Capacity diagnostics are closed. D4 augmentation and increased authority diversity are both now supported by the matched 3584- and 7168-update comparisons. The 32-authority @112 result improves held-out global/edge/gradient and fine-detail transfer over 16-authority D4 at identical total compute, while remaining under-fit and still failing absolute recovery/lattice gates. The bounded diversity question is therefore closed. Promote D4 + authority-balanced exposure into the canonical Main V16 Training / research-preview path next; keep production gates unchanged. Material semantics and BenefitSelector remain unresolved.

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
probe_nsamdr_v16_heldout_transfer.py
probe_nsamdr_v16_sibling_crop_transfer.py
probe_nsamdr_v16_two_crop_fit.py
probe_nsamdr_v16_augmented_two_crop_fit.py
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
scripts\build\nsamdr.bat interference-probe --device cuda --resume <checkpoint> --continue-from <16-authority-checkpoint> --authority-count 16 --stages-per-authority 448
scripts\build\nsamdr.bat heldout-transfer-probe --device cuda --resume <checkpoint> --candidate-checkpoint <16-authority-checkpoint>
scripts\build\nsamdr.bat sibling-crop-transfer-probe --device cuda --resume <checkpoint> --candidate-checkpoint <16-authority-checkpoint>
scripts\build\nsamdr.bat two-crop-fit-probe --device cuda --resume <checkpoint> --authority-reference <16-authority-checkpoint> --authority-count 16 --crops-per-authority 2 --stages-per-crop 112 224
scripts\build\nsamdr.bat augmented-two-crop-fit-probe --device cuda --resume <checkpoint> --authority-reference <16-authority-checkpoint> --authority-count 16 --crops-per-authority 2 --stages-per-crop 112 224
scripts\build\nsamdr.bat stage2-status
scripts\build\nsamdr.bat stage2-probe
scripts\build\nsamdr.bat stage2-summary
```

The reduced structure-conditioning probe remains available for reproducibility. The active full-capacity broad checkpoint is 596. Capacity diagnostics are closed. The matched-budget fixed two-crop run confirms crop-specific overfit: train recovery improves while held-out recovery and 1px detail regress. Run the D4-augmented two-crop probe on the same 16 authorities and same 7168-update budget next. Do not increase authority count or resume the 894-step broad run yet.

## Final visual proof

Qualification is not metric-only. With the same ship, camera, lighting, LOD and shader, the final system must show cleaner continuous seams and contours, sharper supported panel boundaries, restored narrow manufactured features, reduced stair-stepping, aligned normal/material detail, no obvious LR-grid blocks and no invented generic AI texture.

If that renderer-level improvement is not visible, the production goal is not complete.
