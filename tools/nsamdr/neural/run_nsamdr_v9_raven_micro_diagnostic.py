#!/usr/bin/env python3
"""Compatibility entry point for the V13.1 SR-first Raven visual-quality diagnostic."""
from __future__ import annotations

import sys

import _run_nsamdr_v9_raven_micro_diagnostic_impl as _implementation
from run_nsamdr_v13_1_raven_sr_diagnostic import main as _v131_main

# Preserve historical helper exports for tests and local tooling that import the
# compatibility module directly. Test 3 execution itself is now V13.1 SR-first.
for _name in dir(_implementation):
    if not _name.startswith("__") and _name != "main":
        globals()[_name] = getattr(_implementation, _name)


def _replace_legacy_default(arguments: list[str], flag: str, old: str, new: str) -> None:
    try:
        index = arguments.index(flag)
    except ValueError:
        arguments.extend((flag, new))
        return
    if index + 1 < len(arguments) and arguments[index + 1] == old:
        arguments[index + 1] = new


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # The existing GUI still emits the V13.0 defaults. Treat only those exact legacy
    # values as compatibility sentinels; explicit user overrides remain untouched.
    _replace_legacy_default(arguments, "--required-edge-recovery", "0.50", "0.70")
    _replace_legacy_default(arguments, "--required-global-recovery", "0.25", "0.50")
    _replace_legacy_default(arguments, "--required-retention", "0.85", "0.90")
    if "--required-gradient-recovery" not in arguments:
        arguments.extend(("--required-gradient-recovery", "0.40"))
    return _v131_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())