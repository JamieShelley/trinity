# NSAMDR TrinityAL DX11 Preview

This test renders a procedural Raven-like battleship in TrinityAL and provides an ImGui panel for comparing the original stretched-detail result with NSAMDR modes.

## Install

Extract the overlay ZIP into the root of the Trinity repository and allow the included files to overwrite the existing versions.

Example repository location:

```text
D:\REPOS\trinity
```

## Run

From the repository root:

```bat
scripts\build\run_nsamdr_preview_dx11.bat
```

The script configures and builds `TrinityALTest_dx11`, then launches only:

```text
NSAMDRRendering.ShipPreview
```

## Controls

| Input | Action |
|---|---|
| `0` | Original stretched-UV rendering |
| `1` | Full NSAMDR |
| `2` | Damage-mask view |
| `3` | Stochastic reconstruction |
| `4` | Neural-residual prototype |
| `Space` | Toggle original/last enabled mode |
| `R` | Reset view |
| `F9` | Save a DDS screenshot |
| `Esc` | Exit |

The same settings are available in the ImGui panel.

Screenshots are written to:

```text
artifacts\nsamdr
```

The neural mode currently uses fixed prototype weights. It is intended to validate the rendering structure and comparison workflow before trained weights are introduced.
