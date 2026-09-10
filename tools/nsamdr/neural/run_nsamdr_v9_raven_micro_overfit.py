#!/usr/bin/env python3
"""Compatibility entry point for Micro capacity diagnostics with canonical storage."""
from __future__ import annotations

import _run_nsamdr_v9_raven_micro_overfit_impl as _implementation
from v9.diagnostics_layout import run_consolidated_diagnostic

# Other Raven diagnostics import Micro's fixed-patch/config helpers. Preserve that
# compatibility surface while keeping the historical implementation unchanged.
for _name in dir(_implementation):
    if not _name.startswith("__") and _name != "main":
        globals()[_name] = getattr(_implementation, _name)


# Purpose: Run the Micro capacity proof under the unified diagnostic hierarchy.
# Called by: CLI and other Raven diagnostic entry points.
# Calls: the unchanged Micro implementation and run_consolidated_diagnostic.
def main(argv: list[str] | None = None) -> int:
    return run_consolidated_diagnostic(
        _implementation.main,
        argv,
        legacy_folder="micro_diagnostics",
        category="micro",
    )


if __name__ == "__main__":
    raise SystemExit(main())
