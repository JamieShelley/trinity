#!/usr/bin/env python3
"""Compatibility entry point for the V12.11 hard-gated staged Micro diagnostic."""
from __future__ import annotations

import _run_nsamdr_v9_raven_micro_diagnostic_impl as _implementation
from run_nsamdr_v9_raven_micro_diagnostic_v8 import main

for _name in dir(_implementation):
    if not _name.startswith("__") and _name != "main":
        globals()[_name] = getattr(_implementation, _name)


if __name__ == "__main__":
    raise SystemExit(main())
