# Local GR2 converter location

Place a legally obtained local converter installation here only when launching the NSAMDR preview directly from a `.gr2` file:

```text
evegr2toobj.exe
granny2.dll
```

The launcher invokes:

```text
evegr2toobj.exe <source.gr2> <destination.obj>
```

Neither binary is included in this repository overlay. They must not be committed or redistributed without the relevant permission.

You may instead convert the mesh elsewhere and pass the resulting `.obj` directly to `run_nsamdr_obj_preview_dx11.bat`.
