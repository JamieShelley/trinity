#!/usr/bin/env python3
"""Compatibility entry point for the V13.1 SR-first Raven visual-quality diagnostic."""
from __future__ import annotations

import sys

from run_nsamdr_v13_1_raven_sr_diagnostic import main as _v131_main


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
    # Treat only exact legacy GUI defaults as compatibility sentinels. Explicit
    # non-default user values remain untouched.
    _replace_legacy_default(arguments, "--required-edge-recovery", "0.50", "0.70")
    _replace_legacy_default(arguments, "--required-global-recovery", "0.25", "0.50")
    _replace_legacy_default(arguments, "--required-retention", "0.85", "0.90")
    if "--required-gradient-recovery" not in arguments:
        arguments.extend(("--required-gradient-recovery", "0.40"))
    return _v131_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
