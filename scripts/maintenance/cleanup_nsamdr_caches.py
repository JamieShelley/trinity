#!/usr/bin/env python3
"""Delete only Git-ignored Python cache files, never NSAMDR evidence or sources.

Dry-run by default. --apply deletes only paths printed by the dry run.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import subprocess
import sys

CACHE_DIRS = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
CACHE_EXTS = frozenset({".pyc", ".pyo"})
PRESERVED_DIRS = frozenset({
    "artifacts", "node_modules", "vendor", "python-env", "python-env-cpu",
    ".venv", "venv", "checkpoints", "evidence",
})


def git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def verify_root(root: Path) -> Path:
    resolved = root.resolve()
    try:
        actual = Path(os.fsdecode(git(resolved, "rev-parse", "--show-toplevel")).strip()).resolve()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"not a readable Git checkout: {resolved}") from exc
    if actual != resolved:
        raise ValueError(f"expected repository root {resolved}; found {actual}")
    if not (resolved / "tools" / "nsamdr").is_dir():
        raise ValueError("repository does not contain tools/nsamdr")
    return resolved


def candidate_paths(root: Path, scope: str) -> list[tuple[Path, int]]:
    pathspec = ["--", "tools/nsamdr"] if scope == "nsamdr" else []
    ignored = git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z", *pathspec)
    tracked = set(git(root, "ls-files", "-z", "--", "tools/nsamdr").split(b"\0"))
    found: list[tuple[Path, int]] = []
    for raw in ignored.split(b"\0"):
        if not raw or raw in tracked:
            continue
        rel = Path(os.fsdecode(raw))
        if rel.is_absolute() or ".." in rel.parts:
            continue
        parts = rel.parts
        if set(parts).intersection(PRESERVED_DIRS):
            continue
        cache_parts = set(parts).intersection(CACHE_DIRS)
        if not (cache_parts or rel.suffix.lower() in CACHE_EXTS):
            continue
        # Do not erase arbitrary user data placed in a Python bytecode directory.
        if cache_parts == {"__pycache__"} and rel.suffix.lower() not in CACHE_EXTS:
            continue
        path = root / rel
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root):
            continue
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode):
            continue
        if not path.resolve().is_relative_to(root):
            continue
        found.append((rel, info.st_size))
    return sorted(found, key=lambda item: str(item[0]).lower())


def remove_empty_cache_dirs(root: Path, files: list[tuple[Path, int]]) -> None:
    folders: set[Path] = set()
    for rel, _ in files:
        parent = (root / rel).parent
        while parent != root and parent.is_relative_to(root):
            folders.add(parent)
            parent = parent.parent
    for folder in sorted(folders, key=lambda p: len(p.parts), reverse=True):
        if not any(part in CACHE_DIRS for part in folder.relative_to(root).parts):
            continue
        try:
            folder.rmdir()
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--scope", choices=("nsamdr", "repo"), default="nsamdr")
    parser.add_argument("--apply", action="store_true", help="delete only listed ignored cache files")
    args = parser.parse_args(argv)
    try:
        root = verify_root(args.root)
        files = candidate_paths(root, args.scope)
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    total = sum(size for _, size in files)
    print(f"Scope: {args.scope}; ignored cache files: {len(files)}; bytes: {total}")
    for rel, size in files:
        print(f"  {rel.as_posix()} ({size} bytes)")
    if not args.apply:
        print("Dry run only. Pass --apply to delete the listed cache files.")
        return 0
    for rel, _ in files:
        path = root / rel
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root):
            raise RuntimeError(f"symlink detected; aborting: {rel}")
        if not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError(f"not a regular file; aborting: {rel}")
        path.unlink()
    remove_empty_cache_dirs(root, files)
    print(f"Deleted {len(files)} ignored cache files ({total} bytes). No tracked files touched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
