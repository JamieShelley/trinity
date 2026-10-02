#!/usr/bin/env python3
"""Compatibility launcher for the canonical NSAMDR V17 operator GUI.

The active operator workflow moved to nsamdr_v17_workflow_gui.py. This module
remains only so historical scripts/imports that name the V16 GUI path do not
silently launch the rejected V16 Main workflow.
"""
from __future__ import annotations

try:
    from .nsamdr_v17_workflow_gui import *  # noqa: F401,F403
except ImportError:  # pragma: no cover - direct script execution
    from nsamdr_v17_workflow_gui import *  # type: ignore # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(main())
