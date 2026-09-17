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

The next proof uses the production-size V16 body: 96 HR channels, 6 residual-Swin groups x 6 blocks, 6 heads, 8x8 windows, full LR context, and the structure-conditioning branch. It trains on the same 298-authority broad prior and validates against the same 38 complete held-out authorities. The first run is intentionally bounded at 256 updates and writes a resumable model+optimizer checkpoint before any longer stage.

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
    -> 256 updates first
    -> resumable checkpoint
    -> extend to 512 / 1024 / 2048 only from evidence
11. Full Stage 2 qualification
12. BenefitSelector qualification
13. Raven / production Quick
14. Highest-native-resolution renderer proof
```

Do not extend the reduced proof to 8192. The current question is whether the full-capacity candidate can combine the already-proven local reconstruction capacity with broad-authority generalisation.

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
scripts\build\nsamdr.bat stage2-status
scripts\build\nsamdr.bat stage2-probe
scripts\build\nsamdr.bat stage2-summary
```

The reduced structure-conditioning probe remains available for reproducibility. The active full-capacity proof defaults to one bounded 256-update stage and writes `resume_checkpoint.pt`. Continue later stages with `--resume <checkpoint> --stages "512"`, then `1024` and `2048` only when the held-out curve justifies more GPU work.

## Final visual proof

Qualification is not metric-only. With the same ship, camera, lighting, LOD and shader, the final system must show cleaner continuous seams and contours, sharper supported panel boundaries, restored narrow manufactured features, reduced stair-stepping, aligned normal/material detail, no obvious LR-grid blocks and no invented generic AI texture.

If that renderer-level improvement is not visible, the production goal is not complete.
