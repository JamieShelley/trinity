# NSAMDR EVE asset converter

Checked-in source for the EVE SharedCache comparison path.

- GR2 conversion uses the public CarbonEngineJS `format-gr2` reader.
- Every mesh bound by the highest-detail GR2 model is exported, not only the largest mesh.
- GR2 material indices are retained separately from sequential OBJ draw-group indices.
- DDS/BC1-BC7 conversion uses the public CarbonEngineJS `runtime-resource` reader.
- EVE `data.black` SOF data is decoded with the public `black-reader-js` project.
- The SOF projection selects the requested hull and faction, then exports area passes, textures, faction colours, material-library parameters and instanced-mesh diagnostics.
- Dependencies are installed from public GitHub source archives.
- No Granny SDK, `granny2.dll`, native addon, or private CCP package is used.

The resulting preview remains a standalone approximation of Trinity's production shader stack, but it now uses the real SOF material definition instead of a generic tint-only fallback whenever `data.black` is readable.
