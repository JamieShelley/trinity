# Carbon Trinity Windows Build Guide

## Quick start — install CMake, then run the GUI

### 1. Install CMake

Download the current **Windows x64 Installer** from:

https://cmake.org/download/

During installation, select the option to add CMake to the system `PATH`.

Carbon Trinity requires **CMake 3.31 or newer**.

Confirm the installation in Command Prompt:

```bat
cmake --version
```

### 2. Install the other required tools

Install:

- **Visual Studio 2022 Build Tools** or Visual Studio 2022
  - Desktop development with C++
  - MSVC x64/x86 build tools
  - Windows 10 or Windows 11 SDK
- **Git for Windows**
- **Python 3**
  - Keep Tkinter enabled
  - Add Python to `PATH`, or install the standard Python launcher

Confirm them:

```bat
git --version
python --version
```

`py -3 --version` is also accepted by the GUI launcher.

### 3. Put these files in the Trinity repository

The resulting structure must be:

```text
trinity/
├── CMakeLists.txt
├── CMakePresets.json
├── scripts/
│   └── build/
│       ├── build_gui.py
│       ├── run_build_gui.bat
│       ├── build_all.bat
│       └── ...
└── vendor/
```

These scripts must be inside a proper Git clone, not a source-code ZIP download,
because Carbon Trinity uses Git submodules.

### 4. Initialise dependencies

Run once:

```bat
scripts\build\setup_dependencies.bat
```

This initialises:

- Microsoft's `vcpkg`
- Carbon Engine's `vcpkg-registry`
- nested Git submodules

### 5. Launch the build GUI

Double-click:

```text
scripts\build\run_build_gui.bat
```

Or run it from Command Prompt:

```bat
scripts\build\run_build_gui.bat
```

The batch file finds `py -3` or `python` and launches the Tkinter GUI.

### 6. Build everything

In the GUI, press:

```text
Run build_all.bat
```

This creates a TrinityDev build containing:

- DX11
- DX12
- Trinity stub
- TrinityAL tests
- ShaderCompiler
- ShaderCompiler tests

Build output appears in the embedded console.

## Recommended development build

For normal NSAMDR development, build DX11 first:

```bat
scripts\build\build_trinitydev_dx11.bat
```

Then run its tests:

```bat
scripts\build\run_tests_trinitydev_dx11.bat
```

After DX11 is stable, test DX12:

```bat
scripts\build\build_trinitydev_dx12.bat
scripts\build\run_tests_trinitydev_dx12.bat
```

## GUI controls

The GUI provides:

- selection of individual batch files
- selection of normal build scripts
- sequential execution
- live combined standard output and error output
- current-script status
- final exit code
- process-tree termination through **Stop**
- direct execution of `build_all.bat`
- embedded console clearing

Scripts run sequentially. Execution stops when one returns a non-zero exit code.

## Automatic clean-build protection

All build and test batch files that use `_build_config.bat` now compare the
current repository state with the state recorded after the last successful
configure or build. The standalone real-ship viewer performs the same check.

A build directory is removed before CMake runs when any of these change:

- the current Git commit
- staged or unstaged tracked-file contents
- untracked-file contents
- an initialized submodule commit or its local changes
- build-defining options such as backend, configuration, tests, ShaderCompiler
  support or Granny support

When the complete source fingerprint exactly matches the last successful build,
the existing build directory is retained and CMake performs a normal incremental
build. The explicit `rebuild` action always removes the matching build directory.

Only the build directory selected by the current script is removed. Source files,
other configurations and other backend build directories are not touched.

The state check is implemented by:

```text
scripts\build\nsamdr\SourceBuildState.ps1
```

## Generic command-line build

The generic build driver is:

```bat
scripts\build\build.bat <config> <backend> [target] [action] [shader]
```

### Configurations

```text
debug
release
internal
trinitydev
```

### Backends

```text
dx11
dx12
both
stub
```

### Actions

```text
build
configure
rebuild
```

### Examples

Build all TrinityDev DX11 targets:

```bat
scripts\build\build.bat trinitydev dx11 ALL build
```

Rebuild the DX12 test executable:

```bat
scripts\build\build.bat debug dx12 TrinityALTest_dx12 rebuild
```

Build the ShaderCompiler:

```bat
scripts\build\build.bat trinitydev stub ShaderCompiler build shader
```

## Batch script reference

All paths below are relative to `scripts\build`.

### Core helpers

| Batch file | Description |
|---|---|
| `_build_config.bat` | Internal CMake driver used by the build wrappers. Selects the configuration, backend, target, action and optional ShaderCompiler support, and removes the selected build directory when its recorded source fingerprint is stale. |
| `_run_tests.bat` | Internal test driver. Builds the selected TrinityAL test executable, locates it and runs it normally, interactively, in screenshot mode or in comparison mode. |
| `build.bat` | Public command-line entry point for `_build_config.bat`; accepts configuration, backend, target, action and shader arguments. |
| `run_tests.bat` | Public command-line entry point for `_run_tests.bat`; accepts configuration, backend, GoogleTest filter and run mode. |

### TrinityDev builds

| Batch file | Description |
|---|---|
| `build_all.bat` | Builds all TrinityDev DX11 and DX12 targets, tests and the ShaderCompiler in one combined build directory. |
| `build_everything_trinitydev.bat` | Performs the same complete TrinityDev DX11, DX12 and ShaderCompiler build as `build_all.bat`; retained as a descriptive alias. |
| `build_trinitydev_dx11.bat` | Builds all TrinityDev targets with DX11 enabled and DX12 disabled. |
| `build_trinitydev_dx12.bat` | Builds all TrinityDev targets with DX12 enabled and DX11 disabled. |
| `build_trinitydev_both.bat` | Builds all TrinityDev targets with both DX11 and DX12 enabled, without enabling the ShaderCompiler explicitly. |
| `rebuild_trinitydev_dx11.bat` | Removes the TrinityDev DX11 build directory, configures it again and rebuilds all targets. |
| `rebuild_trinitydev_dx12.bat` | Removes the TrinityDev DX12 build directory, configures it again and rebuilds all targets. |
| `build_stub_trinitydev.bat` | Builds the TrinityDev `trinity_stub` target without a DX11 or DX12 backend. |
| `build_shadercompiler_trinitydev.bat` | Configures ShaderCompiler support and builds the TrinityDev `ShaderCompiler` target. |
| `build_shadercompiler_tests_trinitydev.bat` | Configures ShaderCompiler support and builds the TrinityDev `ShaderCompilerTest` target. |
| `list_targets_trinitydev_dx11.bat` | Configures the TrinityDev DX11 build and prints the CMake targets available in that build directory. |

### Other configurations

| Batch file | Description |
|---|---|
| `build_debug_dx11.bat` | Builds all Debug targets with DX11 enabled. |
| `build_debug_dx12.bat` | Builds all Debug targets with DX12 enabled. |
| `build_release_dx11.bat` | Builds all Release targets with DX11 enabled. |
| `build_release_dx12.bat` | Builds all Release targets with DX12 enabled. |
| `build_internal_dx11.bat` | Builds all Internal targets with DX11 enabled. |
| `build_internal_dx12.bat` | Builds all Internal targets with DX12 enabled. |
| `build_all_configs.bat` | Sequentially builds Debug, Release, Internal and TrinityDev for both DX11 and DX12; stops at the first failure. |

### Tests and NSAMDR tools

| Batch file | Description |
|---|---|
| `run_tests_trinitydev_dx11.bat` | Builds and runs the complete TrinityAL GoogleTest suite using the TrinityDev DX11 backend. |
| `run_tests_trinitydev_dx12.bat` | Builds and runs the complete TrinityAL GoogleTest suite using the TrinityDev DX12 backend. |
| `build_nsamdr_realship_dx11.bat` | Builds only the isolated full-Trinity DX11 real-ship viewer with Granny support. It performs a clean build whenever the exact repository/source state differs from the last successful build. |
| `run_nsamdr_realship_dx11.bat` | Builds the full-Trinity DX11 real-ship viewer, mounts the local EVE SharedCache, and launches the real SOF ship/material browser with studio lighting and orbit/pan/zoom controls. |
| `build_nsamdr_preview_dx11.bat` | Compatibility alias that redirects to `build_nsamdr_realship_dx11.bat`; it no longer builds the obsolete procedural proxy. |
| `run_nsamdr_preview_dx11.bat` | Compatibility alias that redirects to `run_nsamdr_realship_dx11.bat`; it cannot launch `TrinityALTest_dx11`. |
| `capture_nsamdr_screenshots_dx11.bat` | Runs DX11 tests matching `*StretchAwareDetail*` and writes their first-frame screenshots under `artifacts\screenshots`. |
| `capture_nsamdr_screenshots_dx12.bat` | Runs DX12 tests matching `*StretchAwareDetail*` and writes their first-frame screenshots under `artifacts\screenshots`. |
| `compare_nsamdr_screenshots_dx11.bat` | Runs DX11 tests matching `*StretchAwareDetail*` and compares their output with the stored reference screenshots. |
| `compare_nsamdr_screenshots_dx12.bat` | Runs DX12 tests matching `*StretchAwareDetail*` and compares their output with the stored reference screenshots. |

The screenshot scripts expect a GoogleTest whose name contains:

```text
StretchAwareDetail
```

### Setup, GUI and cleanup

| Batch file | Description |
|---|---|
| `setup_dependencies.bat` | Synchronises and initialises the vcpkg, Carbon registry and nested Git submodules, temporarily translating GitHub SSH URLs to HTTPS. |
| `run_build_gui.bat` | Finds Python 3 and launches `build_gui.py`; pauses when startup fails so the error remains visible. |
| `clean_generated_builds.bat` | After the user types `CLEAN`, removes only repository-root directories matching `.cmake-build-x64-windows-*`. |

## Build directories

Each configuration and backend uses a separate directory to prevent stale CMake
settings from one backend affecting another.

Examples:

```text
.cmake-build-x64-windows-trinitydev-dx11
.cmake-build-x64-windows-trinitydev-dx12
.cmake-build-x64-windows-trinitydev-both
.cmake-build-x64-windows-trinitydev-stub
```

## Cleaning generated builds

Run:

```bat
scripts\build\clean_generated_builds.bat
```

It asks for the exact confirmation word:

```text
CLEAN
```

It removes only directories matching:

```text
.cmake-build-x64-windows-*
```

## Common failures

### `cmake.exe was not found in PATH`

Install CMake from:

https://cmake.org/download/

Re-run the installer and enable the option to add CMake to `PATH`, or add the
CMake `bin` directory manually. Open a new Command Prompt afterward.

### Repository is not a Git clone

The dependency setup script requires `.git` metadata. Clone the project instead
of downloading GitHub's source ZIP:

```bat
git clone --recursive https://github.com/carbonengine/trinity.git
```

### Visual Studio compiler not found

Open Visual Studio Installer and add:

```text
Desktop development with C++
MSVC x64/x86 build tools
Windows SDK
```

### Python or Tkinter not found

Install the normal Windows Python distribution with Tcl/Tk enabled. Test it:

```bat
python -c "import tkinter; tkinter._test()"
```

### Submodule authentication failure

`setup_dependencies.bat` temporarily rewrites GitHub SSH submodule URLs to HTTPS
for that command. It does not modify the user's global Git configuration.

### Build folder contains stale data

Use the matching rebuild script or remove generated build folders:

```bat
scripts\build\clean_generated_builds.bat
```

Then configure and build again.
