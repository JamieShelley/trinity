# Retired proprietary converter path

The NSAMDR preview no longer uses `evegr2toobj.exe`, `granny2.dll`, or any Granny runtime component.

GR2 geometry is converted by the open-source CarbonEngineJS reader under:

```text
tools/nsamdr/gr2_converter/
```

The conversion path selects the highest-detail render mesh, preserves its mesh-area groups, and emits a Wavefront OBJ plus a draw-range summary. Minimal ship visuals are then reconstructed from `data.black` and SharedCache textures by `tools/nsamdr/eve_asset_test.py`.

This directory remains only to make the removal of the previous local proprietary-converter workflow explicit. Do not place Granny binaries here.
