#!/usr/bin/env python3
"""Compatibility entry point for Parallel Detail diagnostics with canonical storage."""
from __future__ import annotations

import _run_nsamdr_v9_raven_parallel_detail_diagnostic_v2_impl as _implementation
from v9.diagnostics_layout import run_consolidated_diagnostic

for _name in dir(_implementation):
    if not _name.startswith("__") and _name != "main":
        globals()[_name] = getattr(_implementation, _name)


# Purpose: Run the parallel-detail proof under the unified diagnostic hierarchy.
# Called by: compatibility launcher and CLI.
# Calls: the unchanged V2 implementation and run_consolidated_diagnostic.
def main(argv: list[str] | None = None) -> int:
    return run_consolidated_diagnostic(
        _implementation.main,
        argv,
        legacy_folder="parallel_detail_diagnostics",
        category="parallel_detail",
    )


if __name__ == "__main__":
    raise SystemExit(main())
