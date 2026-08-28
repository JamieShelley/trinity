#!/usr/bin/env python3
"""Compatibility CLI for the composition-oriented canonical NSAMDR application.

Implementation lives under ``v9/application/``. This file deliberately remains
small so the executable entrypoint no longer mixes configuration, lifecycle,
gates, training stages, evolutionary recovery, and persistence.
"""
from __future__ import annotations

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v9.application import (
    CANONICAL_SEMANTIC_OVERRIDES,
    DATASET_SCOPE_FIELDS,
    FULL_MINIMUM_WORK_BUDGET,
    QUICK_WORK_BUDGET,
    ConfigResolver,
    QualificationGates,
    StagePlan,
    build_parser,
    main,
)

# Compatibility aliases for old internal imports during staged refactoring.
_parser = build_parser
_canonical_overrides = ConfigResolver().resolve_overrides
_gate_candidate_passed = QualificationGates().production_gate
_local_geometry_metrics = QualificationGates().local_geometry_metrics
_local_geometry_gate = QualificationGates().local_geometry_gate
_PASS_DRIVEN_STAGE_PLAN = StagePlan(QualificationGates()).legacy_tuple()


if __name__ == "__main__":
    raise SystemExit(main())
