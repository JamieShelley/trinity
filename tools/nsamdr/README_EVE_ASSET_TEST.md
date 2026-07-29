# NSAMDR real EVE asset inspection tool

NSAMDR means **Neural Stretch-Aware Material Detail Reconstruction**.

This test reads real ship geometry, minimal SOF visual data, material maps and a space environment from the installed EVE Online SharedCache. It does not require the proprietary Granny SDK. CarbonEngineJS converts the highest-detail `.gr2` mesh, decodes `data.black`, resolves faction texture substitutions and emits per-area material draws for the TrinityAL DX11 viewer.

## Run

From the Trinity repository root:

```bat
scripts\build\test_nsamdr_real_eve_asset.bat
```

No cache path should normally be supplied. The helper searches Windows Installed Apps and App Paths, EVE Launcher configuration/logs, previous verified locations, previous NSAMDR manifests, and common CCP/EVE/Steam install folders.

## Named and grouped ship selection

The first run downloads and locally caches the official EVE JSONL Static Data Export. It joins ship `typeID`/name/group information to graphics and SharedCache model paths so the render window can show names such as **Raven**, **Archon** and **Apocalypse** rather than only internal names such as `cb1_t1`.

The selector groups raw model files under a named ship and chooses the preferred highest-detail model by default. Enable **Show raw LOD / asset variants** to inspect the individual cached resources.

The generated grouped catalog is stored at:

```text
artifacts\nsamdr\eve_assets\ship_catalog.tsv
```

The SDE download is cached at:

```text
%LOCALAPPDATA%\NSAMDR\sde
```

If the SDE cannot be reached and no cached copy exists, the tool remains usable with clearly marked cache-path fallback names.

## Granny-free SOF visuals

The extractor keeps only rendering data required by the inspection viewer:

- highest-detail hull mesh; lower LOD meshes are excluded rather than stacked
- opaque, decal, transparent and additive area ranges
- faction-specific texture substitution
- area albedo, normal, material and glow maps
- faction/material tint, roughness, specular and alpha approximations
- sRGB colour sampling, linear data maps and generated mipmaps

Animation, skeletons, attachments, damage states and other Granny/runtime metadata are not loaded. The generated visual files are:

```text
<asset>.sof-visuals.json
ship.materials.tsv
materials\*.png
```

If a faction-specific resource is indexed but not downloaded locally, the area keeps its SOF tint and may reuse the selected hull's `_d`/`_n` maps only as geometry-aligned surface detail. Those fallback maps no longer determine the area's material identity. If SOF extraction itself fails, the manifest remains clearly labelled and the same hull-local detail fallback prevents the flat tint-only result shown by the earlier build.

## EVE environment lighting

The extractor now indexes every locally available EVE universe cubemap/background and places the race-appropriate environment first. Use the **Background** dropdown, the previous/next buttons, or `[` and `]` to cycle them. Backgrounds are loaded on demand so only one large environment texture remains resident on the GPU.

Each cubemap is converted to an equirectangular PNG for the lightweight viewer. The selected environment is used for:

- the visible space background
- diffuse ambient contribution
- reflective/specular contribution

If the required environment resource has not been downloaded by the EVE client, the viewer uses a procedural EVE-like fallback. The lighting presets are **Game-like**, **Studio**, **Harsh inspection**, and **Dark silhouette**.

## Inspection camera

- Right mouse: orbit around the current focus point
- Middle mouse or left+right mouse together: pan in the camera plane
- Shift + either pan gesture: fine pan
- Mouse wheel: zoom toward the surface under the cursor
- Ctrl + mouse wheel: fine zoom
- Double-click a hull panel: make that surface point the orbit/zoom focus
- `F` or `Home`: frame the complete ship
- `R`: reset orientation and frame the complete ship
- `V`: flip texture V

Near/far clipping, orbit sensitivity, pan speed and zoom speed are adjustable in the render window.

The swap chain, depth buffer, camera projection and ship viewport now follow the client size. Enlarging the window adds real render area instead of stretching the previous frame; the control panel remains on the left and the ship is centred in the remaining viewport.

## Generated output

Converted assets remain under:

```text
artifacts\nsamdr\eve_assets\<asset-code>
```

Do not include extracted EVE assets, downloaded SDE data, generated manifests or screenshots in the eventual pull request.

## Scope

This revision corrects the visual baseline used by the test harness. It does not add Granny runtime dependencies and does not change the NSAMDR reconstruction or UV-damage algorithms.

### Validate the visual baseline

After preparing a ship, validate the generated report before evaluating any repair mode:

```bat
scripts\build\validate_nsamdr_baseline.bat artifacts\nsamdr\eve_assets\<ship>\ship.materials.report.json
```

The command reports unresolved shader families, packed/separate semantic inputs, material-slot parameters and unmapped draw ranges. Modes 3–5 remain disabled until the same report is complete and all declared textures load successfully in the viewer.
