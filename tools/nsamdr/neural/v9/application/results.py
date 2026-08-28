"""Result-file persistence service for canonical NSAMDR orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ResultWriter:
    """Write optional CLI result JSON without coupling workflow services to paths."""

    def _resolve_path(self, requested: Path | None) -> Path | None:
        """Resolve an optional result path relative to the repository.

        Purpose:
            Preserve the original result-file path semantics.
        Called by:
            ResultWriter.__init__().
        Calls:
            No project functions.
        """
        if requested is None:
            return None
        return requested if requested.is_absolute() else self.repo_root / requested

    def __init__(self, repo_root: Path, requested: Path | None) -> None:
        """Resolve and retain the optional result path.

        Purpose:
            Own result-file destination state for one application run.
        Called by:
            TrainingApplication.run().
        Calls:
            ResultWriter._resolve_path().
        """
        self.repo_root = Path(repo_root).resolve()
        self.path = self._resolve_path(requested)

    def write(self, payload: dict[str, Any]) -> None:
        """Write one JSON result object when a destination was requested.

        Purpose:
            Centralise optional caller-facing result persistence.
        Called by:
            ExperimentService lifecycle methods and TrainingApplication.
        Calls:
            json.dumps(), Path.write_text().
        """
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
