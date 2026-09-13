#!/usr/bin/env python3
"""Compatibility command name for V14 immutable-checkpoint preview baking."""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v14.preview import main


if __name__ == "__main__":
    raise SystemExit(main())
