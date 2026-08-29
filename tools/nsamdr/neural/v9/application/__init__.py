"""Composition-oriented application layer for canonical NSAMDR training."""
from .cli import build_parser, parse_options
from .configuration import (
    CANONICAL_SEMANTIC_OVERRIDES,
    DATASET_SCOPE_FIELDS,
    FULL_MINIMUM_WORK_BUDGET,
    QUICK_WORK_BUDGET,
    ConfigResolver,
)
from .gates import QualificationGates, StagePlan
from .runner import TrainingApplication


class InitService:
    def main(self, argv: list[str] | None = None) -> int:
        """Run the canonical NSAMDR training application.

        Purpose:
            Preserve the historic script entrypoint through the new composition root.
        Called by:
            train_nsamdr_v9_preview_experiment.py and direct package callers.
        Calls:
            parse_options(), TrainingApplication.run().
        """
        return TrainingApplication(parse_options(argv)).run()

_init_service = InitService()
main = _init_service.main


__all__ = [
    "CANONICAL_SEMANTIC_OVERRIDES",
    "DATASET_SCOPE_FIELDS",
    "FULL_MINIMUM_WORK_BUDGET",
    "QUICK_WORK_BUDGET",
    "ConfigResolver",
    "QualificationGates",
    "StagePlan",
    "TrainingApplication",
    "build_parser",
    "main",
    "parse_options",
]
