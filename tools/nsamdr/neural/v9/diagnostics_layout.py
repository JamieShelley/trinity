from __future__ import annotations

"""Shared filesystem contract for non-promotable NSAMDR diagnostics.

Historical runners used three sibling roots beneath ``artifacts/nsamdr``. Their
large diagnostic implementations remain byte-for-byte unchanged; thin entry
points expose one canonical hierarchy instead:

    artifacts/nsamdr/diagnostics/
        direct_residual/
        parallel_detail/
        micro/

All three category directories are created together, even before their first run,
so the diagnostic suite has one obvious filesystem home. Before execution, old
local folders are migrated. During execution the runner's historical path is
temporarily aliased to the canonical category directory, so new evidence is
written into the unified tree rather than copied there later. The alias is
removed afterwards. If alias creation is unavailable, the wrapper falls back to
lossless post-run migration, including partial failed-run evidence.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Iterator, Sequence


DIAGNOSTICS_ROOT = Path("artifacts/nsamdr/diagnostics")
DIAGNOSTIC_CATEGORIES = (
    "direct_residual",
    "parallel_detail",
    "micro",
)
LEGACY_DIAGNOSTIC_FOLDERS = {
    "micro_diagnostics": "micro",
    "direct_residual_diagnostics": "direct_residual",
    "parallel_detail_diagnostics": "parallel_detail",
}


# Purpose: Resolve the repository root using the same --repo-root convention as diagnostic CLIs.
# Called by: run_consolidated_diagnostic.
# Calls: pathlib only.
def _repo_root_from_argv(argv: Sequence[str] | None) -> Path:
    arguments = list(sys.argv[1:] if argv is None else argv)
    for index, value in enumerate(arguments):
        if value == "--repo-root" and index + 1 < len(arguments):
            return Path(arguments[index + 1]).expanduser().resolve()
        if value.startswith("--repo-root="):
            return Path(value.split("=", 1)[1]).expanduser().resolve()
    return Path.cwd().resolve()


# Purpose: Create the complete three-category diagnostic hierarchy under one root.
# Called by: consolidate_legacy_diagnostics and run_consolidated_diagnostic.
# Calls: pathlib directory creation only.
def ensure_diagnostic_layout(repo_root: Path) -> Path:
    diagnostics_root = Path(repo_root).resolve() / DIAGNOSTICS_ROOT
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    for category in DIAGNOSTIC_CATEGORIES:
        (diagnostics_root / category).mkdir(parents=True, exist_ok=True)
    return diagnostics_root


# Purpose: Test whether a historical root is already an alias of its canonical target.
# Called by: _remove_existing_alias and _move_legacy_entries.
# Calls: os.path.samefile when both paths exist.
def _same_directory(left: Path, right: Path) -> bool:
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


# Purpose: Remove only a temporary symlink/junction without deleting its canonical target.
# Called by: _remove_existing_alias and _diagnostic_alias.
# Calls: os.unlink on POSIX or cmd rmdir for Windows junctions.
def _remove_alias(alias: Path) -> None:
    if not alias.exists() and not alias.is_symlink():
        return
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/d", "/c", "rmdir", str(alias)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise OSError(
                f"unable to remove diagnostic junction {alias}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
    else:
        alias.unlink()


# Purpose: Clear a stale alias left by a previously interrupted diagnostic process.
# Called by: _move_legacy_entries and _diagnostic_alias.
# Calls: _same_directory and _remove_alias.
def _remove_existing_alias(alias: Path, target: Path) -> bool:
    if _same_directory(alias, target):
        _remove_alias(alias)
        return True
    return False


# Purpose: Choose a collision-safe destination without replacing prior diagnostic evidence.
# Called by: _move_legacy_entries.
# Calls: pathlib existence checks.
def _deduplicated_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    for index in range(2, 10_000):
        candidate = destination.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"unable to allocate diagnostic destination for {destination}")


# Purpose: Move one historical real directory into its canonical diagnostic subfolder.
# Called by: consolidate_legacy_diagnostics.
# Calls: _remove_existing_alias, _deduplicated_destination and shutil.move.
def _move_legacy_entries(
    repo_root: Path,
    legacy_folder: str,
    category: str,
) -> list[Path]:
    legacy_root = repo_root / "artifacts/nsamdr" / legacy_folder
    destination_root = repo_root / DIAGNOSTICS_ROOT / category
    if _remove_existing_alias(legacy_root, destination_root):
        return []
    if not legacy_root.is_dir():
        return []

    destination_root.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    for source in sorted(legacy_root.iterdir(), key=lambda path: path.name.lower()):
        destination = _deduplicated_destination(destination_root / source.name)
        shutil.move(str(source), str(destination))
        moved.append(destination)

    try:
        legacy_root.rmdir()
    except OSError:
        pass
    return moved


# Purpose: Collapse every known historical diagnostic root into one canonical hierarchy.
# Called by: diagnostic wrappers before/after execution; tests may call it directly.
# Calls: ensure_diagnostic_layout and _move_legacy_entries.
def consolidate_legacy_diagnostics(repo_root: Path) -> list[Path]:
    root = Path(repo_root).resolve()
    ensure_diagnostic_layout(root)
    moved: list[Path] = []
    for legacy_folder, category in LEGACY_DIAGNOSTIC_FOLDERS.items():
        moved.extend(_move_legacy_entries(root, legacy_folder, category))
    return moved


# Purpose: Create a temporary filesystem alias so unchanged runners write directly to the canonical tree.
# Called by: run_consolidated_diagnostic.
# Calls: Windows directory junction or POSIX directory symlink creation and _remove_alias.
@contextmanager
def _diagnostic_alias(
    repo_root: Path,
    legacy_folder: str,
    category: str,
) -> Iterator[bool]:
    legacy_root = repo_root / "artifacts/nsamdr" / legacy_folder
    destination_root = repo_root / DIAGNOSTICS_ROOT / category
    destination_root.mkdir(parents=True, exist_ok=True)
    legacy_root.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_alias(legacy_root, destination_root)
    if legacy_root.exists():
        yield False
        return

    created = False
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/d", "/c", "mklink", "/J", str(legacy_root), str(destination_root)],
            capture_output=True,
            text=True,
            check=False,
        )
        created = completed.returncode == 0 and _same_directory(legacy_root, destination_root)
    else:
        try:
            legacy_root.symlink_to(destination_root, target_is_directory=True)
            created = _same_directory(legacy_root, destination_root)
        except OSError:
            created = False

    try:
        yield created
    finally:
        if created:
            _remove_alias(legacy_root)


# Purpose: Translate asynchronous result-viewer launches away from the temporary legacy alias.
# Called by: run_consolidated_diagnostic while the implementation is running.
# Calls: os.startfile on Windows or subprocess.Popen for open/xdg-open on other desktops.
@contextmanager
def _canonical_result_launcher(
    repo_root: Path,
    legacy_folder: str,
    category: str,
) -> Iterator[None]:
    legacy_root = repo_root / "artifacts/nsamdr" / legacy_folder
    canonical_root = repo_root / DIAGNOSTICS_ROOT / category

    def canonicalize(path_value: object) -> object:
        try:
            candidate = Path(os.fspath(path_value))
        except TypeError:
            return path_value
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        try:
            relative = candidate.relative_to(legacy_root)
        except ValueError:
            return path_value
        return canonical_root / relative

    original_startfile = getattr(os, "startfile", None)
    original_popen = subprocess.Popen

    if original_startfile is not None:
        def redirected_startfile(path: object, *args: object, **kwargs: object) -> object:
            return original_startfile(canonicalize(path), *args, **kwargs)

        os.startfile = redirected_startfile  # type: ignore[attr-defined,assignment]

    def redirected_popen(args: object, *popen_args: object, **popen_kwargs: object) -> object:
        command = args
        if isinstance(args, (list, tuple)) and len(args) >= 2:
            executable = str(args[0])
            if executable in {"open", "xdg-open"}:
                rewritten = list(args)
                rewritten[1] = os.fspath(canonicalize(rewritten[1]))
                command = rewritten
        return original_popen(command, *popen_args, **popen_kwargs)

    subprocess.Popen = redirected_popen  # type: ignore[assignment]
    try:
        yield
    finally:
        subprocess.Popen = original_popen  # type: ignore[assignment]
        if original_startfile is not None:
            os.startfile = original_startfile  # type: ignore[attr-defined,assignment]


# Purpose: Execute one unchanged diagnostic implementation under the canonical artifact hierarchy.
# Called by: Micro, Direct Residual and Parallel Detail compatibility entry points.
# Calls: _repo_root_from_argv, ensure_diagnostic_layout, consolidate_legacy_diagnostics,
#        _diagnostic_alias and _canonical_result_launcher.
def run_consolidated_diagnostic(
    implementation_main: Callable[[list[str] | None], int],
    argv: list[str] | None = None,
    *,
    legacy_folder: str,
    category: str,
) -> int:
    expected_category = LEGACY_DIAGNOSTIC_FOLDERS.get(legacy_folder)
    if expected_category != category:
        raise ValueError(
            f"unknown diagnostic layout mapping {legacy_folder!r} -> {category!r}"
        )

    repo_root = _repo_root_from_argv(argv)
    diagnostics_root = ensure_diagnostic_layout(repo_root)
    pre_moved = consolidate_legacy_diagnostics(repo_root)
    if pre_moved:
        print(
            f"[diagnostics] migrated {len(pre_moved)} existing item(s) under "
            f"{diagnostics_root}",
            flush=True,
        )

    result: int
    try:
        with _diagnostic_alias(repo_root, legacy_folder, category) as direct_write:
            if not direct_write:
                print(
                    "[diagnostics] canonical alias unavailable; using safe post-run migration",
                    flush=True,
                )
            with _canonical_result_launcher(repo_root, legacy_folder, category):
                result = int(implementation_main(argv))
    except BaseException:
        moved = consolidate_legacy_diagnostics(repo_root)
        if moved:
            print(
                f"[diagnostics] preserved {len(moved)} partial item(s) under "
                f"{diagnostics_root}",
                flush=True,
            )
        raise

    moved = consolidate_legacy_diagnostics(repo_root)
    print(f"[diagnostics] canonical root: {diagnostics_root}", flush=True)
    print(f"[diagnostics] current category: {diagnostics_root / category}", flush=True)
    if moved:
        print(f"[diagnostics] consolidated {len(moved)} fallback item(s)", flush=True)
    return result
