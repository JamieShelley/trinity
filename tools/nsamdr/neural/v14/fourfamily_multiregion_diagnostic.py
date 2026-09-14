#!/usr/bin/env python3
"""V16 Stage 2 diagnostic requiring four authored battleship families."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
NEURAL_ROOT = HERE.parent
if str(NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURAL_ROOT))

from v14 import multifamily_multiregion_diagnostic as multi


REQUIRED_AUTHORED_FAMILIES = 4
_original_records = multi._records


def _records(
    self: multi.base.MultiRegionDiagnostic,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, validation = _original_records(self, manifest)

    train_counts = Counter(str(record.get("family_id") or "") for record in train)
    validation_counts = Counter(
        str(record.get("family_id") or "") for record in validation
    )
    if len(train_counts) < REQUIRED_AUTHORED_FAMILIES:
        raise RuntimeError(
            "V16 Stage 2 requires four authored families in training; "
            f"found {dict(train_counts)}"
        )
    if len(validation_counts) < REQUIRED_AUTHORED_FAMILIES:
        raise RuntimeError(
            "V16 Stage 2 requires four authored families in held-out evaluation; "
            f"found {dict(validation_counts)}"
        )

    if int(self.args.train_regions) == 8 and any(count != 2 for count in train_counts.values()):
        raise RuntimeError(
            "V16 controlled four-family test requires 2 train regions per family: "
            f"{dict(train_counts)}"
        )
    if int(self.args.validation_regions) == 4 and any(
        count != 1 for count in validation_counts.values()
    ):
        raise RuntimeError(
            "V16 controlled four-family test requires 1 held-out region per family: "
            f"{dict(validation_counts)}"
        )

    print(
        "[v16.0-fourfamily] controlled coverage confirmed: "
        f"train={dict(train_counts)} held-out={dict(validation_counts)}",
        flush=True,
    )
    return train, validation


multi.base.MultiRegionDiagnostic._records = _records


def main(argv: list[str] | None = None) -> int:
    return multi.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
