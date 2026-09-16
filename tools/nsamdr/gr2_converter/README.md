# NSAMDR EVE asset converter

This directory contains the JavaScript helper used by `tools/nsamdr/eve_asset_test.py` to read EVE resource formats needed by NSAMDR asset preparation.

## Active source

```text
convert_eve_asset.mjs
    GR2 / DDS / SOF-related conversion helper used by the Python asset pipeline

vendor/core-math-compat/
    local compatibility package required by the converter dependency graph

node_modules/
    vendored runtime dependencies required by the current converter
```

The vendored dependency tree is intentionally retained for now. `@carbonenginejs/runtime-resource` in this branch is a private/source package rather than a clean reproducible npm dependency, so deleting `node_modules/` currently breaks the converter. Do not treat that directory as historical output until the converter dependency graph has been packaged reproducibly.

## Ownership

The converter is an implementation detail of EVE asset preparation. Training and diagnostics should call the Python NSAMDR tools rather than invoke this file directly.

Canonical entry points:

```bat
scripts\build\nsamdr.bat gui
scripts\build\nsamdr.bat eve-census
```

The authored-corpus census itself reads DDS headers directly and does not run the converter. Dataset preparation may run this converter when geometry, SOF or material semantics must be resolved.
