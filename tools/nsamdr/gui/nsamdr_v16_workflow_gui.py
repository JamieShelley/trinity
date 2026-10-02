#!/usr/bin/env python3
"""Compatibility launcher for the canonical NSAMDR V17 operator GUI.

The active operator workflow moved to nsamdr_v17_workflow_gui.py. This module
remains only so historical scripts/imports that name the V16 GUI path do not
silently launch the rejected V16 Main workflow.
"""
from __future__ import annotations

try:
    from . import nsamdr_v17_workflow_gui as _v17
except ImportError:  # pragma: no cover - direct script execution
    import nsamdr_v17_workflow_gui as _v17  # type: ignore

App = _v17.App
Stage = _v17.Stage
STAGES = _v17.STAGES
BY_ID = _v17.BY_ID
APP_TITLE = _v17.APP_TITLE
STATE_SCHEMA = _v17.STATE_SCHEMA
V17_PREVIEW_CHOICE = _v17.V17_PREVIEW_CHOICE
MAIN_PREVIEW_CHOICE = _v17.MAIN_PREVIEW_CHOICE
_format_duration = _v17._format_duration
_qualified_final = _v17._qualified_final
_read_json = _v17._read_json
main = _v17.main


if __name__ == "__main__":
    raise SystemExit(main())
