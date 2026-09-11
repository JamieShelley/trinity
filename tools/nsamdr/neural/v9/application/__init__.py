"""Composition-oriented application layer for current NSAMDR training."""
from .cli import build_parser, parse_options
from .configuration import (
    CANONICAL_SEMANTIC_OVERRIDES,
    DATASET_SCOPE_FIELDS,
    QUICK_WORK_BUDGET,
    ConfigResolver,
    assert_quick_stop_phase,
    assert_sr_first_quick_config,
    is_sr_first_quick_config,
)
from .runner import TrainingApplication


class InitService:
    def main(self, argv: list[str] | None = None) -> int:
        """Run the canonical NSAMDR training application.

        Purpose:
            Preserve the script entrypoint through the application composition root.
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
    "QUICK_WORK_BUDGET",
    "ConfigResolver",
    "TrainingApplication",
    "assert_quick_stop_phase",
    "assert_sr_first_quick_config",
    "build_parser",
    "is_sr_first_quick_config",
    "main",
    "parse_options",
]
