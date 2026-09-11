#!/usr/bin/env python3
"""Compatibility entry point for the V13.1 SR-first Raven visual-quality diagnostic."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import TextIO

from run_nsamdr_v13_1_raven_sr_diagnostic import main as _v131_main


class _CanonicalDiagnosticStream:
    """Rewrite the temporary legacy junction path to the canonical diagnostics path."""

    def __init__(self, stream: TextIO, legacy_root: Path, canonical_root: Path) -> None:
        self._stream = stream
        self._legacy = str(legacy_root)
        self._canonical = str(canonical_root)

    def write(self, text: str) -> int:
        return self._stream.write(text.replace(self._legacy, self._canonical))

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def _replace_legacy_default(arguments: list[str], flag: str, old: str, new: str) -> None:
    try:
        index = arguments.index(flag)
    except ValueError:
        arguments.extend((flag, new))
        return
    if index + 1 < len(arguments) and arguments[index + 1] == old:
        arguments[index + 1] = new


def _repo_root(arguments: list[str]) -> Path:
    for index, value in enumerate(arguments):
        if value == "--repo-root" and index + 1 < len(arguments):
            return Path(arguments[index + 1]).expanduser().resolve()
        if value.startswith("--repo-root="):
            return Path(value.split("=", 1)[1]).expanduser().resolve()
    return Path.cwd().resolve()


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Treat only exact legacy GUI defaults as compatibility sentinels. Explicit
    # non-default user values remain untouched.
    _replace_legacy_default(arguments, "--required-edge-recovery", "0.50", "0.70")
    _replace_legacy_default(arguments, "--required-global-recovery", "0.25", "0.50")
    _replace_legacy_default(arguments, "--required-retention", "0.85", "0.90")
    if "--required-gradient-recovery" not in arguments:
        arguments.extend(("--required-gradient-recovery", "0.40"))

    root = _repo_root(arguments)
    legacy_root = root / "artifacts/nsamdr/micro_diagnostics"
    canonical_root = root / "artifacts/nsamdr/diagnostics/micro"
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = _CanonicalDiagnosticStream(stdout, legacy_root, canonical_root)
    sys.stderr = _CanonicalDiagnosticStream(stderr, legacy_root, canonical_root)
    try:
        return _v131_main(arguments)
    finally:
        sys.stdout = stdout
        sys.stderr = stderr


if __name__ == "__main__":
    raise SystemExit(main())
