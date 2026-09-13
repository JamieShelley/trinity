#!/usr/bin/env python3
"""Compatibility command name for the V14 HR-first Raven workflow.

The public CLI still calls this historical filename so existing batch files and the GUI
keep working. There is no V9/V13 model or trainer behind this entry point.
"""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v14.workflow import main


if __name__ == "__main__":
    raise SystemExit(main())
