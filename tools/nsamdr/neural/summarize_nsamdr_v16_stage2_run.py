#!/usr/bin/env python3
"""Summarize a V16 Stage 2 run before changing training behavior."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def _latest_run(root: Path) -> Path | None:
    runs = [path for path in root.glob("multiregion_*") if path.is_dir()]
    return max(runs, key=lambda path: path.stat().st_mtime) if runs else None


def _pct(report: dict[str, Any], key: str) -> str:
    try:
        return f"{float(report[key]) * 100:+.2f}%"
    except (KeyError, TypeError, ValueError):
        return "?"


def _line(label: str, report: dict[str, Any]) -> str:
    return (
        f"{label:<12} global={_pct(report, 'medianGlobalRecovery')} "
        f"edge={_pct(report, 'medianEdgeRecovery')} "
        f"grad={_pct(report, 'medianGradientRecovery')} "
        f"normal={_pct(report, 'medianNormalRecovery')} "
        f"material={_pct(report, 'medianMaterialRecovery')} "
        f"lattice={_pct(report, 'maxLatticeCellExcess')} "
        f"qualified={'YES' if report.get('passed') else 'NO'}"
    )


def _score(report: dict[str, Any]) -> float:
    try:
        g = float(report.get("medianGlobalRecovery") or 0.0)
        e = float(report.get("medianEdgeRecovery") or 0.0)
        d = float(report.get("medianGradientRecovery") or 0.0)
        l = float(report.get("maxLatticeCellExcess") or 1.0)
    except (TypeError, ValueError):
        return -1.0e9
    return g + e + d - max(0.0, l - 0.15)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize V16 Stage 2 evidence")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.repo_root.resolve() / "artifacts/nsamdr/diagnostics/v16_mini"
    run_dir = args.run.resolve() if args.run else _latest_run(root)
    if run_dir is None or not run_dir.is_dir():
        print("[v16-stage2-summary] no Stage 2 run found")
        return 1

    curve_path = run_dir / "generalisation_curve.json"
    curve = _read_json(curve_path)
    if not isinstance(curve, list):
        print(f"[v16-stage2-summary] no generalisation curve: {curve_path}")
        return 1
    points = [item for item in curve if isinstance(item, dict)]
    if not points:
        print("[v16-stage2-summary] curve contains no validation points")
        return 1

    best = max(points, key=lambda item: _score(dict(item.get("validation") or {})))
    final = points[-1]
    report = _read_json(run_dir / "report.json")
    report = report if isinstance(report, dict) else {}

    print(f"[v16-stage2-summary] run       : {run_dir}")
    print(f"[v16-stage2-summary] points    : {len(points)}")
    print(f"[v16-stage2-summary] final     : step {final.get('step', '?')} diagnosis={final.get('diagnosis', '?')}")
    print(_line("final train", dict(final.get("train") or {})))
    print(_line("final held", dict(final.get("validation") or {})))
    print(f"[v16-stage2-summary] best      : step {best.get('step', '?')} diagnosis={best.get('diagnosis', '?')}")
    print(_line("best train", dict(best.get("train") or {})))
    print(_line("best held", dict(best.get("validation") or {})))

    if report:
        print(
            f"[v16-stage2-summary] selected  : step {report.get('selectedStep', '?')} "
            f"completed={report.get('completedSteps', '?')} "
            f"result={'PASS' if report.get('passed') else 'FAIL'} "
            f"diagnosis={report.get('diagnosis', '?')}"
        )

    per_family = dict(best.get("validation") or {}).get("perFamily")
    if isinstance(per_family, dict) and per_family:
        print("[v16-stage2-summary] best held-out per family:")
        for family_id, family_report in sorted(per_family.items()):
            if isinstance(family_report, dict):
                name = str(family_report.get("sourceAssetName") or family_report.get("familyName") or family_id)
                print("  " + _line(name[:12], family_report))

    first_val = dict(points[0].get("validation") or {})
    final_train = dict(final.get("train") or {})
    final_val = dict(final.get("validation") or {})
    best_val = dict(best.get("validation") or {})
    try:
        best_global = float(best_val.get("medianGlobalRecovery") or 0.0)
        final_global = float(final_val.get("medianGlobalRecovery") or 0.0)
        final_train_global = float(final_train.get("medianGlobalRecovery") or 0.0)
        first_global = float(first_val.get("medianGlobalRecovery") or 0.0)
    except (TypeError, ValueError):
        best_global = final_global = final_train_global = first_global = 0.0

    if final_train_global > best_global + 0.20 and final_global < best_global - 0.03:
        reading = "clear train/held-out divergence; generalisation/overfit remains the dominant failure"
    elif best_global > first_global + 0.10 and final_global >= best_global - 0.03:
        reading = "held-out recovery is improving or broadly retained; inspect gates before changing data/model"
    else:
        reading = "mixed/weak movement; inspect per-family curves and probes before changing training behavior"
    print(f"[v16-stage2-summary] reading   : {reading}")

    for label, path in (
        ("probe", run_dir / "probes/latest/ALL_FAMILIES.png"),
        ("probe-all", run_dir / "probes/latest/ALL_FAMILIES_DETAIL.png"),
    ):
        if path.is_file():
            print(f"[v16-stage2-summary] {label:<9}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
