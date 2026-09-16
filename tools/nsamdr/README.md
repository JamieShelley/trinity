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

The first 256-update broad-prior comparison used 298 train authorities and 38 held-out authorities, but random short-run sampling provided less than one update per authority on average. It showed no early conditioning benefit:

```text
control global / edge       +1.06% / +0.55%
conditioned global / edge   +0.37% / +0.33%
conditioned - control       -0.69pp / -0.21pp
```

This rejects the claim of an immediate 256-step advantage, but it does **not** adequately test broad-prior learning.

The current probe therefore uses authority-balanced sampling and a cumulative checkpoint ladder:

```text
512 updates   evaluate control and conditioned
1024 updates  evaluate control and conditioned
2048 updates  run only if held-out conditioning signal remains positive or marginal
```

Every training cycle visits every authority before repeating one. Validation is also authority-balanced and, by default, evaluates one sample from every held-out authority. If conditioning still has no held-out benefit at the second checkpoint, the 2048 stage is skipped.

## Active qualification ladder

```text
1. V16 Capacity                                         PASS
2. Four-family Stage 2                                 FAIL generalisation
3. Raw recoverability/context/loss diagnostics         FAIL to solve gap
4. Full EVE authored corpus census                     PASS
5. Geometry-class residual oracle                      FAIL
6. Boundary Profile oracle                             FAIL
7. Structure Support Audit V3                          CURRENT CPU evidence
   -> threshold sweep
   -> occupied boundary area
   -> error enrichment
   -> precision / recall
8. Broad authored-prior corpus                         PASS
   -> 298 train authorities
   -> 38 complete held-out authorities
9. Authority-balanced structure-conditioning ladder   CURRENT GPU proof
   -> 512 / 1024 / conditional 2048
10. Full Stage 2 qualification                         only after broad proof
11. BenefitSelector qualification
12. Raven / production Quick
13. Highest-native-resolution renderer proof
```

No long production-scale V16.2 run is justified until the authority-balanced broad-prior proof shows a held-out advantage.

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
scripts\build\nsamdr.bat stage2-status
scripts\build\nsamdr.bat stage2-probe
scripts\build\nsamdr.bat stage2-summary
```

To reproduce the old single-checkpoint probe, pass `--steps N`. The default structure-conditioning command now runs the staged `512,1024,2048` ladder with automatic stop before the long stage when the second checkpoint still shows no benefit.

## Final visual proof

Qualification is not metric-only. With the same ship, camera, lighting, LOD and shader, the final system must show cleaner continuous seams and contours, sharper supported panel boundaries, restored narrow manufactured features, reduced stair-stepping, aligned normal/material detail, no obvious LR-grid blocks and no invented generic AI texture.

If that renderer-level improvement is not visible, the production goal is not complete.
