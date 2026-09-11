#!/usr/bin/env python3
"""Compatibility CLI for the current SR-first NSAMDR training application."""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v9.application import main


if __name__ == "__main__":
    raise SystemExit(main())
