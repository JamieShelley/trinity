#!/usr/bin/env python3
"""V16 GUI entry point with the Stage 2 multi-family diagnostic enabled."""
from __future__ import annotations

import sys

import nsamdr_v16_workflow_gui as v16


base = v16.base
_original_dispatcher_argv = base.App._dispatcher_argv
_original_select_multiregion = v16._select_multiregion


def _dispatcher_argv(
    self: base.App,
    command: tuple[str, ...],
    args: list[str],
) -> list[str]:
    if command == ("v14-mini-multiregion",):
        return [
            sys.executable,
            str(
                self.repo
                / "tools/nsamdr/neural/v14/multifamily_multiregion_diagnostic.py"
            ),
            "--repo-root",
            str(self.repo),
            *args,
        ]
    return _original_dispatcher_argv(self, command, args)


def _select_multiregion(self: base.App, stage: base.Stage, status: str) -> None:
    _original_select_multiregion(self, stage, status)
    self._label_row(
        "Authored families",
        "Stage 2 requires and balances at least two distinct native Raven-hull texture families.",
    )
    self._label_row(
        "Family split",
        "Each authored family gets its own hard pixel-disjoint train/held-out spatial domains.",
    )
    self._label_row(
        "Region selection",
        "Training and held-out regions are balanced across authored families before the existing round-robin optimizer schedule.",
    )


base.App._dispatcher_argv = _dispatcher_argv
v16.legacy._select_multiregion = _select_multiregion


if __name__ == "__main__":
    raise SystemExit(base.main())
