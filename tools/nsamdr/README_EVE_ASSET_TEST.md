# NSAMDR real EVE asset inspection tool

NSAMDR means **Neural Stretch-Aware Material Detail Reconstruction**.

This test reads real ship geometry, material maps and a space environment from the installed EVE Online SharedCache. It does not require the proprietary Granny SDK. The selected `.gr2` model and DDS resources are converted into local inspection files and opened in the TrinityAL DX11 viewer.

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

## EVE environment lighting

For each model, the extractor looks for the race-appropriate EVE universe scene and a locally available referenced nebula/cubemap DDS. A cubemap is converted to an equirectangular PNG for the lightweight viewer. The same environment is used for:

- the visible space background
- diffuse ambient contribution
- reflective/specular contribution

If the required environment resource has not been downloaded by the EVE client, the viewer uses a procedural EVE-like fallback. The lighting presets are **Game-like**, **Studio**, **Harsh inspection**, and **Dark silhouette**.

## Inspection camera

- Right mouse: orbit around the current focus point
- Middle mouse: pan in the camera plane
- Shift + middle mouse: fine pan
- Mouse wheel: zoom toward the surface under the cursor
- Ctrl + mouse wheel: fine zoom
- Double-click a hull panel: make that surface point the orbit/zoom focus
- `F` or `Home`: frame the complete ship
- `R`: reset orientation and frame the complete ship
- `V`: flip texture V

Near/far clipping, orbit sensitivity, pan speed and zoom speed are adjustable in the render window.

## Generated output

Converted assets remain under:

```text
artifacts\nsamdr\eve_assets\<asset-code>
```

Do not include extracted EVE assets, downloaded SDE data, generated manifests or screenshots in the eventual pull request.

## Scope

This revision improves the test harness only: named ship selection, LOD grouping, environment lighting and inspection navigation. It deliberately does not change the NSAMDR reconstruction or UV-damage algorithms yet.
