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

## Main wrapper scripts

### Engine builds

```text
build_trinitydev_dx11.bat
build_trinitydev_dx12.bat
build_trinitydev_both.bat
build_debug_dx11.bat
build_debug_dx12.bat
build_release_dx11.bat
build_release_dx12.bat
build_internal_dx11.bat
build_internal_dx12.bat
```

### Complete builds

```text
build_all.bat
build_everything_trinitydev.bat
build_all_configs.bat
```

`build_all.bat` is the normal single-command complete development build.

`build_all_configs.bat` builds all four configurations for DX11 and DX12. It
can use substantial time and disk space.

### Other targets

```text
build_stub_trinitydev.bat
build_shadercompiler_trinitydev.bat
build_shadercompiler_tests_trinitydev.bat
list_targets_trinitydev_dx11.bat
```

### Test and screenshot scripts

```text
run_tests_trinitydev_dx11.bat
run_tests_trinitydev_dx12.bat
capture_nsamdr_screenshots_dx11.bat
capture_nsamdr_screenshots_dx12.bat
compare_nsamdr_screenshots_dx11.bat
compare_nsamdr_screenshots_dx12.bat
```

The NSAMDR scripts currently expect a Google Test whose name contains:

```text
StretchAwareDetail
```

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


## NSAMDR real EVE asset test

Launch the GUI and use the dedicated **NSAMDR real EVE asset test** panel:

```bat
run_build_gui.bat
```

The normal direct command requires no cache path:

```bat
run_nsamdr_eve_asset_dx11.bat
```

The launcher discovers the installed EVE SharedCache automatically through Windows installation and launcher metadata. A manual cache path remains available only as a diagnostic override.

The default query is the Raven hull:

```text
res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2
```

This path extracts from the local EVE SharedCache, converts GR2 and DDS through CarbonEngineJS, and runs the Granny-free OBJ viewer. The render window builds a named ship catalog from the official EVE SDE, groups LOD/model variants, loads an EVE nebula environment when present, and provides close inspection camera controls.
