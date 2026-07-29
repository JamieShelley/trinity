# NSAMDR real EVE asset test (automatic path)

The preferred test path now extracts a real EVE ship directly from the installed SharedCache and does **not** require you to locate or manually convert source assets.

Run the build GUI:

```bat
scripts\build\run_build_gui.bat
```

In **NSAMDR real EVE asset test (Granny SDK-free)**, leave the default Raven resource selected and click **Extract Raven + convert + run**.

Command-line equivalent:

```bat
scripts\build\run_nsamdr_eve_asset_dx11.bat
```

The helper reads `tq\resfileindex.txt`, copies the real `.gr2` and matching DDS textures from `ResFiles`, converts them through the open-source CarbonEngineJS reader, and launches this OBJ preview. Local extracted assets are placed under `artifacts/nsamdr/eve_assets` and are not intended for Git.

See `tools/nsamdr/README_EVE_ASSET_TEST.md` for the full workflow.

---

# NSAMDR real-OBJ preview (Granny-free)

This test loads a real ship mesh from Wavefront OBJ and runs the NSAMDR material-detail comparison in TrinityAL DX11 without enabling Carbon's Granny dependency.

No EVE assets, Granny files or third-party conversion binaries are included.

## What this path proves

- real converted ship geometry rather than the procedural block proxy
- original OBJ UV coordinates
- per-triangle UV stretch and anisotropy analysis
- screen-space stretch evidence
- damage-mask display
- original material, validation, UV diagnostics, structure reconstruction and full NSAMDR comparison
- interactive inspection lighting, orbit, pan and zoom
- optional source albedo texture

It does not recreate the complete EveSOF faction/material stack. It is an isolated visual test for the stretched-detail problem while the native Granny build is unavailable.

## Run an existing OBJ

From the Trinity repository root:

```bat
scripts\build\run_nsamdr_obj_preview_dx11.bat "D:\Models\raven.obj"
```

The first run builds `TrinityALTest_dx11` in a dedicated Granny-free directory and then launches only:

```text
NSAMDRRendering.RealObjShipPreview
```

The compatibility command now uses the same real-OBJ path:

```bat
scripts\build\run_nsamdr_preview_dx11.bat "D:\Models\raven.obj"
```

The viewer refuses to fall back to procedural geometry. An OBJ with positions, faces and UV coordinates is required.

## Add an albedo texture

Supply a PNG, JPG, BMP or TIFF as the second argument:

```bat
scripts\build\run_nsamdr_obj_preview_dx11.bat ^
  "D:\Models\raven.obj" ^
  "D:\Models\raven_albedo.png"
```

DDS is not loaded directly by this lightweight path. Convert a DDS locally with Microsoft's `texconv`, for example:

```bat
texconv -ft png "D:\Models\raven_d.dds"
```

The viewer includes a **Flip texture V** checkbox and `V` shortcut because OBJ exporters differ in texture-coordinate orientation.

## Convert a GR2 automatically

The launcher accepts a `.gr2` file when a local `evegr2toobj` installation is present at:

```text
tools\nsamdr\evegr2toobj\evegr2toobj.exe
tools\nsamdr\evegr2toobj\granny2.dll
```

Then run:

```bat
scripts\build\run_nsamdr_obj_preview_dx11.bat "D:\EVEAssets\raven.gr2"
```

The launcher calls the converter as:

```text
evegr2toobj.exe <source.gr2> <destination.obj>
```

Converted files are written to:

```text
artifacts\nsamdr\converted
```

The converter and Granny runtime must be obtained and used legally. They are deliberately not downloaded or redistributed by this overlay. An OBJ converted by another application works equally well.

## Build without launching

```bat
scripts\build\build_nsamdr_obj_preview_dx11.bat
```

## Controls

| Input | Action |
|---|---|
| Right mouse | Orbit around the current focus point |
| Middle mouse | Pan in the camera plane |
| Shift + middle mouse | Fine pan |
| Mouse wheel | Zoom toward the surface under the cursor |
| Ctrl + mouse wheel | Fine zoom |
| Double-click | Focus the clicked hull surface |
| `F` / `Home` | Frame the complete ship |
| `R` | Reset orientation and frame the ship |
| `V` | Flip texture V |
| `0` | Original EVE material |
| `1` | Material / input validation |
| `2` | UV texel-density and stretch diagnostics |
| `3` | Structure-preserving reconstruction |
| `4` | Full NSAMDR result |
| `5` | Difference / reconstructed contribution |
| `Space` | Toggle original/last enabled mode |
| `F9` | Save DDS screenshot |
| `Esc` | Exit |

The real-EVE launcher also builds a named, grouped ship selector from the official EVE SDE and the installed SharedCache. Raw LOD/model variants are hidden by default but can be exposed in the render window.

When present locally, an EVE universe nebula/cubemap is used as the background, ambient environment and reflection source. The viewer provides Game-like, Studio, Harsh inspection and Dark silhouette presets.

Screenshots are written to:

```text
artifacts\nsamdr
```

## Expected build banner

```text
NSAMDR REAL OBJ SHIP PREVIEW - GRANNY FREE DX11
Target     : TrinityALTest_dx11
Granny     : OFF
```

This path does not download or install the `granny` vcpkg package.

## v8 reconstruction-tool additions

This revision changes the test workflow from noise-only enhancement to a two-stage inspection and reconstruction pass. Mode 1 validates the source inputs with split albedo/checker, normal and PGS-or-mip views. Mode 2 visualises UV stretch direction, estimated mip pressure and the damage heatmap. Mode 3 performs structure-preserving reconstruction of broad panel detail before any microdetail is added. Mode 4 applies the full NSAMDR result, and Mode 5 isolates only the reconstructed contribution.
