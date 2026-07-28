# NSAMDR Real EVE Ship Material Viewer

This tool is the full-Trinity replacement for the procedural TrinityAL block-ship preview.

It mounts the installed EVE SharedCache, loads the real SOF database, lists every SOF ship hull, faction/material set and race, then builds the selected ship through `EveSOF`. The resulting scene therefore uses the ship's real `.gr2` geometry, textures, decals and Trinity material effects.

The viewer is injected only when its launcher supplies `NSAMDRProjectInclude.cmake`. Normal Carbon Trinity configure and build commands are unchanged.

## Run

From the Trinity repository root:

```bat
scripts\build\run_nsamdr_realship_dx11.bat
```

The older `run_nsamdr_preview_dx11.bat` name is retained only as a compatibility alias and redirects to this real-ship viewer. It no longer builds or launches `TrinityALTest_dx11`.

When the EVE SharedCache is not in a common location, pass it explicitly:

```bat
scripts\build\run_nsamdr_realship_dx11.bat "D:\EVE\SharedCache"
```

Alternatively set:

```bat
set EVE_SHARED_CACHE=D:\EVE\SharedCache
```

The folder must contain `ResFiles` and a `resfileindex.txt` somewhere below it.

## Controls

| Input | Action |
|---|---|
| Right mouse drag | Orbit around the current focus point |
| Middle mouse drag | Pan the focus point |
| Mouse wheel | Zoom |
| `R` | Frame the loaded ship and reset orbit |
| `Space` | Toggle original/last enabled NSAMDR mode |
| `0`–`4` | Select NSAMDR mode |
| `Esc` | Exit |

## Ship selection

The UI exposes searchable lists for:

- SOF ship hulls
- faction/material sets
- races
- direct DNA strings

The first matching Raven/Caldari entries are selected when available. The actual available names come from the locally installed SOF data rather than a bundled hard-coded catalog.

## Lighting

The scene uses the real EVE material lighting inputs as a studio rig: directional key light, cool ambient fill, environment-reflection/rim contribution and an adjustable dark-blue background. `Studio` and `Harsh inspection` presets are included.

## NSAMDR shader hook

The viewer publishes a global Trinity variable named:

```text
NSAMDRSettings = (mode, strength, exposure, reserved)
```

This establishes the control path on the real EVE material. The EVE material effect override still needs to consume this variable before modes 1–4 alter the real shader. Until that effect patch is added, mode 0 is a real-material baseline and modes 1–4 change the published hook only. The UI states this explicitly to avoid presenting a post-process approximation as a material comparison.

## Files and isolation

The viewer does not copy EVE assets into the repository. It reads the user's existing SharedCache through Blue's remote-cache resource filesystem and its `resfileindex.txt` mapping.

The launcher enables:

```text
BUILD_DX11=ON
WITH_GRANNY=ON
BUILD_TESTING=OFF
```

and builds only `NSAMDRRealShipViewer` plus its required dependencies.
