"""Clock service used by experiment lifecycle metadata."""
from __future__ import annotations

from datetime import datetime, timezone


class UtcClock:
    """Provide UTC timestamps without scattering datetime calls through orchestration."""

    def now(self) -> str:
        """Return an ISO-8601 UTC timestamp rounded to seconds.

        Purpose:
            Supply deterministic-format lifecycle timestamps.
        Called by:
            ExperimentService and ExperimentRunSession.
        Calls:
            datetime.now().
        """
        return datetime.now(timezone.utc).isoformat(timespec="seconds")
