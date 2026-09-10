#!/usr/bin/env python3
"""Compatibility entry point for Direct Residual diagnostics with canonical storage."""
from __future__ import annotations

import _run_nsamdr_v9_raven_direct_residual_diagnostic_impl as _implementation
from v9.diagnostics_layout import run_consolidated_diagnostic

# Re-export the implementation helpers because the parallel integration proof uses
# Direct Residual's metric/render helpers as its reference implementation.
for _name in dir(_implementation):
    if not _name.startswith("__") and _name != "main":
        globals()[_name] = getattr(_implementation, _name)


# Purpose: Run Direct Residual under the unified diagnostic artifact hierarchy.
# Called by: CLI and diagnostic integrations.
# Calls: the unchanged Direct Residual implementation and run_consolidated_diagnostic.
def main(argv: list[str] | None = None) -> int:
    return run_consolidated_diagnostic(
        _implementation.main,
        argv,
        legacy_folder="direct_residual_diagnostics",
        category="direct_residual",
    )


if __name__ == "__main__":
    raise SystemExit(main())
