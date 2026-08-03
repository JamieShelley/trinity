# NSAMDR V5.34 — tile-context material reconstruction

## V5.34 converter dependency resolution correction

The converter now validates the two JavaScript entry points it actually imports instead of assuming npm hoisted every transitive package into a particular physical `node_modules` directory. npm may legally nest or hoist `runtime-utils` and the `core-math` compatibility package. A successful Node ESM import probe is the readiness contract; failures now print the real module-resolution diagnostic.

## V5.26 mode selector and icon correction

- Mode 1 remains the startup default.
- Modes 1, 2 and 3 are permanently visible in a fixed selector at the top of the controls panel.
- Mode 3 remains selectable even when its candidate is unavailable; it displays an explicit error rather than disappearing or falling back to Mode 1.
- The generated NSAMDR icon is embedded as the executable's primary icon and applied to both the window and its Win32 class for title-bar and taskbar use.


## V5.25 historical mode and icon changes — superseded by V5.26

- V5.25 temporarily started the viewer in Mode 3 with split comparison enabled; V5.26 restored Mode 1 as the startup default.
- Mode 3 is always present in the UI, even before the first V4 training run.
- Before training, Mode 3 uses a deterministic prepared-4K bootstrap candidate; retraining replaces it with the tile-context result.
- Missing Mode 3 resources never fall back to Mode 1 under a Mode 3 label. The pane displays an explicit unavailable state instead.
- A bundled multi-resolution NSAMDR icon is embedded in the executable and applied to both the window and Windows taskbar icon.


## Build-state and repository-marker repair

CMake performs its normal incremental configure and build. If the tracked `trinityal/CMakeLists.txt` marker is missing, the verifier restores only that missing file from the current working-tree `HEAD`; it does not create a commit.

## Build dependency correction

The NSAMDR launcher now disables automatic vcpkg manifest installation for this already-populated build directory and resolves `fxc.exe` from the installed Windows SDK. This prevents the preview build from trying to redownload CCP's private `fxc-v10.1.zip` package. It does not enable Granny and does not use a private overlay.


## V5.33 public-source converter dependency repair

V5.32 requested `@carbonenginejs/format-gr2` and `@carbonenginejs/runtime-resource` as npm-registry versions, but those package names are not published there. V5.33 installs the readers from their public GitHub source archives instead. It deletes the failed V5.32 `package-lock.json` and partial `node_modules` tree before the one-time install.

The archived GR2 reader still imports its former `@carbonenginejs/core-math` subpaths. A checked-in compatibility package with that exact package name and compatible version re-exports `mesh`, `num`, `tangent`, and `vec3` from the maintained public `@carbonenginejs/runtime-utils` source. No npm-registry copy of the unpublished CarbonEngineJS packages is required.

## V5.32 converter-source packaging repair

The complete source override now includes `tools/nsamdr/gr2_converter` rather than merely referencing it. The bridge uses `@carbonenginejs/format-gr2` for Granny-free `.gr2` geometry decoding and `@carbonenginejs/runtime-resource` for DDS decoding. On the first EVE-asset test run, npm installs those two open-source readers locally beneath the converter directory. No native Granny library or private CCP SDK is used.

The converter writes every mesh bound by the selected highest-detail GR2 model to OBJ, converts DDS material maps to PNG, converts DDS cube environments to an equirectangular PNG, and decodes the selected hull/faction material definition from SOF `data.black`. The tint-only path remains only as an explicit failure fallback.

## V5.31 source-tree repair

The verifier preserves the upstream `trinityal/scripts`, `trinityal/tools` and `trinityal/trinityal` directories. It restores only tracked TrinityAL files that are absent from the working tree, excludes the override-owned `trinityal/tests/nsamdr` directory, and never overwrites an existing modified file. The launcher also requires the Windows SDK **x64** `fxc.exe`; ARM64 and x86 tools are rejected.

## Quick start

V5.24 replaces the old per-pixel V3 network with a fully convolutional tile-context model. **Retraining is mandatory** because the old V3 checkpoint is incompatible.

From Command Prompt in the repository root:

```bat
call scripts\build\verify_and_clean_nsamdr_layout.bat
call scripts\build\setup_nsamdr_cuda.bat
call scripts\build\train_nsamdr.bat --source-root "D:\actual\EVE\SharedCache" --device cuda
call scripts\build\test_nsamdr.bat
call scripts\build\test_nsamdr_real_eve_asset.bat
```

Use the real SharedCache or extracted texture directory. Do not use the literal placeholder `D:\actual\EVE\SharedCache`.

For CPU training:

```bat
call scripts\build\setup_nsamdr_cpu.bat
call scripts\build\train_nsamdr.bat --source-root "D:\actual\EVE\SharedCache" --device cpu
```

The RTX 5080 profile uses PyTorch 2.11.0 with the CUDA 12.8 wheel and verifies `sm_120` before training.

## Comparison contract

The viewer is a strict material-cleanup A/B test:

| Pane | Content |
|---|---|
| **Mode 1 — original source, no cleanup** | Original extracted source material. |
| **Mode 2 — UV/stretch diagnostics** | Stretch and sampling-pressure evidence. |
| **Mode 3 — tile-context cleanup** | V4 reconstructed albedo baked into the Mode 3 material manifest. |

Mode 1 and Mode 3 use the **same mesh, camera, lighting, environment and shader**. The intended difference is the albedo resource selected by each material manifest. Mode 3 does not receive hidden exposure, normal, specular, roughness or lighting advantages.

This remains Granny-free:

```text
WITH_GRANNY=OFF
```

The preview applies the bundled NSAMDR window and taskbar icon to the title bar and Windows taskbar while it is active.

## Why V4

The V3 model evaluated one pixel from a sparse local feature vector. It could soften isolated stair steps, but it could not maintain a panel contour over a long distance or distinguish coherent trim from repeated texture damage.

V4 uses broad spatial context. It can see complete corners, parallel trim lines, repeated stair-step patterns and surrounding material evidence before deciding how to reconstruct a pixel.

## Network architecture

Schema:

```text
NSAMDR_TILE_CONTEXT_MATERIAL_V4
```

Default model:

```text
8-channel material tile
    -> two 3×3 stem convolutions
    -> 8 dilated residual blocks
       dilation: 1, 2, 4, 8, 8, 4, 2, 1
    -> continuous flow XY
    -> bounded RGB residual
    -> confidence gate
```

Inputs:

1. albedo RGB;
2. authored normal XY;
3. material selector;
4. paint support;
5. roughness.

Outputs:

1. continuous source offset X/Y;
2. RGB residual;
3. confidence.

The default model has approximately 160,000 parameters and a **125-pixel receptive field**. It is large enough to understand extended panel structure while remaining practical for offline GPU inference.

The model reconstructs material colour, not the final lit image. Lighting therefore remains controlled by the common preview renderer.

## Offline overlapping inference

Mode 3 generation runs the model over **overlapping 512×512 tiles** with a **64-pixel overlap**. A tapered blend window combines neighbouring predictions to prevent tile seams.

Data flow:

```text
original EVE material maps
    -> deterministic 4K preparation
    -> 8-channel semantic material tensor
    -> overlapping V4 tile inference
    -> continuous colour transport + residual + confidence
    -> baked Mode 3 albedo
    -> common live preview shader
```

There is **No runtime neural compute shader**. There is no generated HLSL weight include and no transient neural-albedo resource. This removes the previous per-pixel runtime path and makes the A/B boundary explicit.

Generated model files:

```text
artifacts\nsamdr\neural\nsamdr_tile_context.pt
artifacts\nsamdr\neural\nsamdr_tile_context.json
```

Generated Mode 3 files:

```text
artifacts\nsamdr\eve_assets\<asset>\strategy_candidates_4096\mode3_nsamdr_neural
```

## Training data behaviour

Synthetic training creates clean panel materials and then degrades the input with combinations of:

- low-resolution resampling;
- diagonal staircase contours;
- broken bright trim;
- compression-like blocks;
- anisotropic blur;
- halo and edge fuzz;
- local normal/material disagreement.

The clean material remains the target. Normal, material, paint and roughness maps provide structural guidance.

Real source textures are also sampled for identity-preservation training. They are not silently treated as perfect contour truth. The identity loss discourages changes to valid flat panels and existing authored detail.

Training losses cover:

- reconstructed colour;
- Sobel contour agreement;
- identity preservation;
- confidence supervision;
- flow and residual smoothness.

## Default training configuration

File:

```text
tools\nsamdr\neural\default_training_config.json
```

| Key | Default | Purpose |
|---|---:|---|
| `epochs` | 24 | Complete passes over generated tiles. |
| `tilesPerEpoch` | 2048 | Training tiles generated per epoch. |
| `batchSize` | 8 | Tiles per optimiser step. |
| `baseChannels` | 32 | Feature width. |
| `residualBlocks` | 8 | Dilated context blocks. |
| `tileSize` | 96 | Training crop size. |
| `maxOffsetPixels` | 8 | Maximum learned source transport. |
| `maxResidual` | 0.25 | Maximum bounded RGB residual. |
| `edgeWeight` | 1.5 | Contour fidelity weight. |
| `identityWeight` | 0.65 | Valid-region preservation weight. |
| `inferenceTileSize` | 512 | Candidate-generation tile size. |
| `inferenceOverlap` | 64 | Seam-prevention overlap. |

The training panel writes an override JSON and launches:

```text
scripts\build\retrain_nsamdr_and_preview.bat
```

That process trains, validates, regenerates the Mode 3 candidate, rebuilds and relaunches the viewer.

## Model validation

Run:

```bat
call scripts\build\test_nsamdr.bat
```

The validator checks:

- V4 schema;
- model hash;
- 8 input and 6 output channels;
- parameter count between 100,000 and 500,000;
- output dimensions and bounded values;
- receptive field of at least 125 pixels;
- stored validation metrics.

Candidate generation rejects an old V3 checkpoint instead of silently producing an identity comparison.

## C++ architecture

The preview remains ordinary `.h` and `.cpp` composition:

```text
PreviewApplication
    +-- CameraController
    +-- MeshProcessor
    +-- InputController
    +-- StrategyModes
    +-- NSAMDRPipeline
    +-- AssetProcessor
    +-- NSAMDRTrainingController
    +-- SceneController
    +-- RenderPipeline
    +-- PreviewRenderer
    +-- PreviewProcessing
    +-- PreviewPanel
```

The former C++ runtime-compute subsystem is removed. Mode 3 preparation is now owned by the Python candidate generator, while C++ only loads and displays the resulting material resources.

The only inheritance is the GoogleTest fixture required by the TrinityAL test host. `.inl` implementation files are forbidden.

## Source ownership

| Responsibility | File |
|---|---|
| V4 model, training and tiled inference | `tools/nsamdr/neural/train_nsamdr_kernel.py` |
| V4 checkpoint validator | `tools/nsamdr/neural/test_nsamdr_kernel.py` |
| Mode 3 4K candidate generation | `tools/nsamdr/generate_strategy_candidates.py` |
| Real EVE asset preparation and launch | `tools/nsamdr/eve_asset_test.py` |
| CUDA environment setup | `scripts/build/setup_nsamdr_cuda.bat` |
| Training entry point | `scripts/build/train_nsamdr.bat` |
| Full retrain/build/relaunch | `scripts/build/retrain_nsamdr_and_preview.bat` |
| Layout and stale-runtime cleanup | `scripts/build/verify_and_clean_nsamdr_layout.bat` |
| Same-renderer A/B draw path | `trinityal/tests/nsamdr/NSAMDRRenderPipeline.cpp` |
| Common material shader | `trinityal/tests/nsamdr/NSAMDRPreview.hlsl` |
| UI and training controls | `trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp` |

## Acceptance criteria

A V4 result is useful only when:

1. long panel contours become continuous;
2. bright trim loses staircase and fuzzy halo artefacts;
3. valid surface grain and authored markings remain;
4. no tile seams appear;
5. Mode 1 remains unchanged when the model or candidate changes;
6. lighting, geometry and shader state remain identical between panes.
