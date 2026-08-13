# NSAMDR V9.8.3 Operator guide

NSAMDR uses one supported command surface:

```bat
scripts\build\nsamdr.bat --help
```

The launcher resolves the repository root and Python environment, then delegates
to `tools\nsamdr\nsamdr_cli.py`. Commands work from any current directory and
return the underlying tool's exit code.

`run_nsamdr_v9_gui.bat` remains only as a double-click GUI shim. The CUDA and
CPU setup BATs remain bootstrap entry points. The native DX11 OBJ launcher is
an internal bridge used by the dispatcher and preview tooling; call it through
`nsamdr.bat native ...` for normal use.

## Command surface

| Capability | Command |
|---|---|
| GUI | `nsamdr.bat gui` |
| Environment | `nsamdr.bat setup cuda`; `nsamdr.bat setup cpu` |
| Guarded Raven workflow | `nsamdr.bat tune [options]` |
| Raven dataset | `nsamdr.bat index raven [options]` |
| Low-level Raven trainer | `nsamdr.bat train preview [options]` |
| Experiment preview | `nsamdr.bat preview EXP_#### [options]` |
| Compare | `nsamdr.bat compare EXP_#### EXP_#### [EXP_####]` |
| Promote | `nsamdr.bat promote EXP_####` |
| Full EVE dataset | `nsamdr.bat index eve [options]` |
| Full production training | `nsamdr.bat train full [options]` |
| Full production preview | `nsamdr.bat preview production [options]` |
| Promoted production pipeline | `nsamdr.bat run` |
| Validation | `nsamdr.bat validate`; `nsamdr.bat test ...` |
| Candidate/native preview | `nsamdr.bat candidate`; `nsamdr.bat native ...` |
| Maintenance | `nsamdr.bat cleanup`; `nsamdr.bat integrate` |

Use a leaf command's `--help` for its owned options. Backend-specific options
are forwarded without a shell, so existing V9 option names remain valid.

## Current workflow

The supported tuning path is:

```text
Stage A: parameter-free renderer preflight
  -> sign-gauge metric-SDF training
  -> Stage B: predicted-SDF proof with the learned gate blocked
  -> gate training only after PASS
  -> synthetic acceptance gate
  -> Raven audit and native preview
```

Run that complete guarded sequence with:

```bat
call scripts\build\nsamdr.bat tune
```

The GUI uses this same dispatcher route:

```bat
call scripts\build\nsamdr.bat gui
```

Expected title: `NSAMDR V9 Workflow Controller 4.9.3`.

### `tune` versus `train preview`

`tune` is the normal operator command. It prepares or reuses the deterministic
Raven data, preserves the Stage A/SDF/Stage B/gate ordering, enforces the hard
acceptance gates, audits the result, and launches the Raven preview only after
PASS.

`train preview` is a lower-level experiment trainer. It does not build the
Raven dataset, orchestrate both proof stages, or launch the final preview. Use
it only for isolated recovery or debugging.

Quick tuning uses 11 epochs and 96 training/16 validation tiles per epoch. It
is previewable and comparable but cannot be promoted. A Full proof uses the
complete 24-epoch phase schedule with 128 training/32 validation tiles per
epoch and is the only promotion-eligible mode.

## Raven tuning and experiments

Build the fixed, non-overlapping Raven Navy Issue dataset explicitly when
needed:

```bat
call scripts\build\nsamdr.bat index raven ^
  --shared-cache "C:\CCP\EVE" ^
  --train-crops 12 ^
  --validation-crops 4
```

Start a Quick experiment:

```bat
call scripts\build\nsamdr.bat tune ^
  --experiment new ^
  --training-mode quick
```

For a promotion proof, use `--training-mode full`. Experiments are immutable;
resume an existing `EXP_####` only with its stored mode and semantic settings.

Preview, compare, and promote completed experiments:

```bat
call scripts\build\nsamdr.bat preview EXP_0001
call scripts\build\nsamdr.bat compare EXP_0001 EXP_0002
call scripts\build\nsamdr.bat promote EXP_0002
```

Configuration promotion copies the exact accepted semantic hyperparameters
into a production-scoped config. It does not copy Raven weights into the
production model.

## Production

Index the authored EVE dataset using the selected promoted config:

```bat
call scripts\build\nsamdr.bat index eve ^
  --config artifacts\nsamdr\promoted\EXP_0002\v9_fidelity_full.json ^
  --shared-cache "C:\CCP\EVE"
```

Run Full production training:

```bat
call scripts\build\nsamdr.bat train full ^
  --config artifacts\nsamdr\promoted\EXP_0002\v9_fidelity_full.json ^
  --shared-cache "C:\CCP\EVE" ^
  --control auto
```

The command performs CUDA preflight, optional indexing, manifest validation,
training, checkpoint validation, and candidate-cache invalidation in that
order. `--resume` and `--restart` retain their existing meanings.

Preview the validated production checkpoint:

```bat
call scripts\build\nsamdr.bat preview production ^
  --checkpoint-dir artifacts\nsamdr\neural_v9 ^
  --force-candidate
```

After a Full Raven proof has been promoted, the entire production sequence is:

The all-assets production run uses the promoted configuration for indexing,
full training, checkpoint validation, and the native production preview.

```bat
call scripts\build\nsamdr.bat run --shared-cache "C:\CCP\EVE"
```

The final checkpoint is `artifacts\nsamdr\neural_v9\nsamdr_v9_fidelity.pt`.

## Validation

Run the capability/layout check, semantic contract, and architecture smoke test:

```bat
call scripts\build\nsamdr.bat validate --device cpu
```

Individual checks are available as:

```bat
call scripts\build\nsamdr.bat test contract
call scripts\build\nsamdr.bat test architecture --device cpu
call scripts\build\nsamdr.bat test checkpoint
```

The checkpoint command expects a completed production checkpoint unless a
different config is supplied.

Current invariants:

- schema: `NSAMDR_SIGN_GAUGE_METRIC_SDF_RENDERER_4X_V9_8_3`
- upscale factor: 4
- model input channels: 16
- production parameter count: 7,915,282
- production semantic config hash:
  `ce04236d056f41a5376a167d15232083d9e7ffdf1965205ffb170bb4e1bc05a0`

The Raven tuning model uses the same production architecture. Dataset scope and
work volume differ; architecture and semantic gates do not.

## Native preview

Build the specialized TrinityAL DX11 preview without launching it:

```bat
call scripts\build\nsamdr.bat native build
```

Open a local OBJ or GR2 asset:

```bat
call scripts\build\nsamdr.bat native obj "D:\assets\ship.obj" "D:\assets\ship.png"
```

Extract and preview an installed EVE asset:

```bat
call scripts\build\nsamdr.bat native eve --shared-cache "C:\CCP\EVE"
```

The retained `run_nsamdr_obj_preview_dx11.bat` owns the specialized incremental
CMake configuration and launch. It is intentionally not another public workflow
surface.

## Trinity integration

The upstream renderer files do not contain the local NSAMDR quality override by
default. Apply the idempotent anchor-based integration with:

```bat
call scripts\build\nsamdr.bat integrate
```

Dry-run and verify anchors without writing:

```bat
call scripts\build\nsamdr.bat integrate --check
```

The exposed policy remains Off/Balanced/High/Ultra with Off as the default.

## Cleanup and artifacts

With no artifact flags, cleanup removes only Python caches. Artifact deletion is
explicit and repository-contained. Always inspect a dry run first:

```bat
call scripts\build\nsamdr.bat cleanup --all-artifacts --dry-run
```

Then select only the intended roots, for example `--tuning-dataset`,
`--experiments`, `--promotion`, `--dataset`, or `--production`.

Important artifact roots:

```text
artifacts\nsamdr\training_v9_preview_raven   fixed Raven dataset
artifacts\nsamdr\experiments                immutable experiments
artifacts\nsamdr\promoted                   selected configuration
artifacts\nsamdr\training_v9                full authored dataset
artifacts\nsamdr\neural_v9                  production checkpoint/state
```

Cleanup is never run automatically by the GUI or training commands.
