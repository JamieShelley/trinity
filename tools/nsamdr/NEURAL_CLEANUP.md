# NSAMDR tracked-source cleanup audit — 30 October 2026

Audited the remote NSAMDR tree at commit 7f4c0b3a7348f48076b1d06a8379661498352407. The Windows working tree and ignored/local files were not accessible.

## Findings

- tools/nsamdr/neural: 163 tracked files (~1.66 MiB): 60 versioned module files, 44 tests, 20 probes, 6 trainers, 6 dataset-preparation scripts, 5 benchmarks, 3 audits, 3 inspectors and 16 other files.
- tools/nsamdr/gr2_converter/node_modules: 476 tracked files (~2.42 MiB). The converter README explicitly says these contain private/source dependencies not reproducibly installable from npm. Preserve until packaging replacement and full converter tests are available.
- artifacts/nsamdr is already ignored. Checkpoints, evidence and generated native previews are not disposable cache files.
- V6–V8 scripts are provenance-bearing experimental source files. Moving or editing them without a provenance migration would invalidate source-matched checkpoints.
- Zero static references alone does not demonstrate that a historical probe is dead: Python CLI, GUI, documentation and external command execution must be checked.

## Safe local-only cleanup

From a verified D:\REPOS\trinity checkout, run a dry-run:

    python scripts/maintenance/cleanup_nsamdr_caches.py

After inspecting all printed paths, run:

    python scripts/maintenance/cleanup_nsamdr_caches.py --apply

Optional --scope repo expands the same strict policy to the full repository. Only ignored, untracked cache files are eligible. Tracked files, environments, checkpoint/evidence paths, vendored dependencies and symlinks are skipped. No Git reset or history rewrite occurs.

## Next tracked-cleanup gate

Trace CLI/GUI, dynamic import, test, documentation and checkpoint-source references for each historical script. Remove only a provably obsolete group on an isolated branch, with Windows regression and converter validation before merging. Preserve frozen V17.6 evidence.

**No tracked source is deleted by this cleanup tool.**
