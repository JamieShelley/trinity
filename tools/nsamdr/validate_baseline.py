#!/usr/bin/env python3
"""Validate an NSAMDR material baseline report and print exact blockers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"ERROR: baseline report does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ERROR: invalid baseline report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"ERROR: baseline report root must be an object: {path}")
    return value


def validate(report: dict[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    areas = report.get("areas")
    if not isinstance(areas, list) or not areas:
        return False, ["report contains no material areas"]

    for index, area in enumerate(areas):
        if not isinstance(area, dict):
            blockers.append(f"area[{index}] is not an object")
            continue
        group = area.get("group", index)
        name = str(area.get("areaName") or area.get("areaType") or "unnamed")
        prefix = f"group {group} ({name})"
        family = str(area.get("shaderFamily") or "unknown")
        if family == "unknown":
            blockers.append(f"{prefix}: unknown shader family")
        missing = area.get("missingSemantics")
        if isinstance(missing, list) and missing:
            blockers.append(f"{prefix}: missing semantics: {', '.join(map(str, missing))}")
        if not bool(area.get("semanticComplete")):
            blockers.append(f"{prefix}: semantic texture inputs incomplete")
        if not bool(area.get("parameterComplete")):
            material_names = area.get("materialNames") if isinstance(area.get("materialNames"), list) else []
            matches = area.get("materialLibraryMatches") if isinstance(area.get("materialLibraryMatches"), list) else []
            unresolved_parameters = area.get("unresolvedParameters") if isinstance(area.get("unresolvedParameters"), list) else []
            if not any(str(name).strip() for name in material_names):
                blockers.append(f"{prefix}: no SOF material names resolved for Mtl1-Mtl4")
            else:
                missing_materials = [
                    str(name) for slot, name in enumerate(material_names)
                    if str(name).strip() and slot < len(matches) and not bool(matches[slot])
                ]
                if missing_materials:
                    blockers.append(f"{prefix}: material library entries not found: {', '.join(missing_materials)}")
            if unresolved_parameters:
                blockers.append(f"{prefix}: unresolved parameters: {', '.join(map(str, unresolved_parameters))}")
            blockers.append(f"{prefix}: material-slot parameters incomplete")
        if not bool(area.get("baselineComplete")):
            blockers.append(f"{prefix}: baseline incomplete")

    return bool(report.get("complete")) and not blockers, blockers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path, help="ship.materials.report.json")
    parser.add_argument("--json", action="store_true", help="emit machine-readable validation output")
    args = parser.parse_args()

    report = _load(args.report)
    complete, blockers = validate(report)
    result = {
        "complete": complete,
        "reportedComplete": bool(report.get("complete")),
        "unresolvedCount": int(report.get("unresolvedCount") or 0),
        "blockers": blockers,
    }
    diagnostics = report.get("extractionDiagnostics") if isinstance(report.get("extractionDiagnostics"), dict) else {}
    if args.json:
        result["extractionDiagnostics"] = diagnostics
        print(json.dumps(result, indent=2))
    else:
        print("NSAMDR visual baseline: " + ("COMPLETE" if complete else "INCOMPLETE"))
        print(f"Unresolved count: {result['unresolvedCount']}")
        for blocker in blockers:
            print(f"  - {blocker}")
        if diagnostics:
            print("SOF extraction diagnostics:")
            print(f"  graph mode: {diagnostics.get('blackGraphMode', 'unknown')}")
            print(f"  faction area types: {', '.join(map(str, diagnostics.get('factionAreaTypes', []))) or '(none)'}")
            print(f"  primary materials: {', '.join(map(str, diagnostics.get('factionPrimaryMaterialNames', []))) or '(none)'}")
            print(f"  material usage: {diagnostics.get('factionMaterialUsage', [])}")
            reports = diagnostics.get("blackReports")
            if isinstance(reports, list) and reports:
                print(f"  Black reader reports: {len(reports)}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
