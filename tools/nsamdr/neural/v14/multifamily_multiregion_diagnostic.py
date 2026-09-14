#!/usr/bin/env python3
"""V16 Stage 2 entry point with multi-family authored data authority.

The underlying V16 model, losses, qualification gates, optimizer schedule, and
reporting stay unchanged.  This wrapper changes only Stage 2 dataset preparation
and region selection so both train and held-out evaluation cover distinct native
authored texture families.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
NEURAL_ROOT = HERE.parent
if str(NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURAL_ROOT))

from v14 import multiregion_diagnostic as base


def _prepare_multifamily_dataset(args: Any, repo_root: Path) -> None:
    script = repo_root / "tools/nsamdr/neural/prepare_nsamdr_v16_multifamily_dataset.py"
    command = [
        sys.executable,
        "-u",
        str(script),
        "--repo-root",
        str(repo_root),
        "--shared-cache",
        str(args.shared_cache),
        "--train-crops",
        str(args.prepare_train_regions),
        "--validation-crops",
        str(args.prepare_validation_regions),
    ]
    if bool(args.rebuild_dataset):
        command.append("--rebuild")
    print(
        "[v16.0-multiregion] prepare multi-family authored dataset: "
        + subprocess.list2cmdline(command),
        flush=True,
    )
    result = subprocess.run(command, cwd=repo_root, check=False)
    if result.returncode:
        raise RuntimeError(
            "V16 Stage 2 multi-family dataset preparation failed with exit code "
            f"{result.returncode}"
        )


def _balanced_family_subset(
    records: list[dict[str, Any]],
    requested: int,
) -> list[dict[str, Any]]:
    count = min(len(records), max(1, int(requested)))
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_family.setdefault(str(record.get("family_id") or "unknown"), []).append(record)
    for family_records in by_family.values():
        family_records.sort(key=base.detail_score, reverse=True)

    family_order = sorted(
        by_family,
        key=lambda family: (
            -base.detail_score(by_family[family][0]),
            family,
        ),
    )
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for family in family_order:
            queue = by_family[family]
            if not queue or len(selected) >= count:
                continue
            selected.append(queue.pop(0))
            progressed = True
        if not progressed:
            break
    return selected


def _records(
    self: base.MultiRegionDiagnostic,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_pool = [
        record for record in manifest["crops"] if record.get("split") == "train"
    ]
    validation_pool = [
        record for record in manifest["crops"] if record.get("split") == "validation"
    ]
    train = _balanced_family_subset(train_pool, int(self.args.train_regions))
    validation = _balanced_family_subset(
        validation_pool,
        int(self.args.validation_regions),
    )

    train_families = sorted({str(record.get("family_id") or "") for record in train})
    validation_families = sorted(
        {str(record.get("family_id") or "") for record in validation}
    )
    if len(train_families) < 2 or len(validation_families) < 2:
        raise RuntimeError(
            "V16 Stage 2 requires at least two authored families in both training "
            "and held-out region selections"
        )
    print(
        f"[v16.0-multiregion] balanced authored families: "
        f"train={len(train_families)} held-out={len(validation_families)}",
        flush=True,
    )
    for split, records in (("train", train), ("held-out", validation)):
        counts: dict[str, int] = {}
        names: dict[str, str] = {}
        for record in records:
            family = str(record.get("family_id") or "unknown")
            counts[family] = counts.get(family, 0) + 1
            names[family] = str(record.get("source_asset_name") or family)
        detail = ", ".join(
            f"{names[family]}={counts[family]}" for family in sorted(counts)
        )
        print(f"[v16.0-multiregion] {split} regions: {detail}", flush=True)
    return train, validation


base.prepare_raven_dataset = _prepare_multifamily_dataset
base.MultiRegionDiagnostic._records = _records


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
