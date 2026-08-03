from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CandidateBuildContext:
    source_obj: Path
    source_materials: Path
    output_root: Path
    target_size: int


@dataclass(frozen=True)
class CandidateArtifact:
    mode: int
    label: str
    obj: Path
    materials: Path
    status: str = "ready"
    metadata: dict[str, Any] = field(default_factory=dict)

    def report_entry(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "label": self.label,
            "status": self.status,
            "obj": str(self.obj.resolve()),
            "materials": str(self.materials.resolve()),
            **self.metadata,
        }


class StrategyManifest:
    """Owns the registry-shaped report while retaining legacy flat keys."""

    def __init__(self, schema: str, target_size: int, baseline_obj: Path, baseline_materials: Path) -> None:
        self.schema = schema
        self.target_size = target_size
        self.baseline_obj = baseline_obj
        self.baseline_materials = baseline_materials
        self._artifacts: dict[int, CandidateArtifact] = {}
        self._extra_entries: dict[int, dict[str, Any]] = {}
        self.notes: list[str] = []

    def add(self, artifact: CandidateArtifact) -> None:
        self._artifacts[artifact.mode] = artifact

    def add_entry(self, mode: int, entry: dict[str, Any]) -> None:
        self._extra_entries[mode] = dict(entry)

    def to_report(self) -> dict[str, Any]:
        strategies: dict[str, dict[str, Any]] = {}
        report: dict[str, Any] = {
            "schema": self.schema,
            "targetSize": self.target_size,
            "baselineObj": str(self.baseline_obj.resolve()),
            "baselineMaterials": str(self.baseline_materials.resolve()),
        }
        for mode in sorted(self._artifacts):
            artifact = self._artifacts[mode]
            strategies[str(mode)] = artifact.report_entry()
            # Preserve the launcher-compatible flat fields.
            report[f"mode{mode}Obj"] = str(artifact.obj.resolve())
            report[f"mode{mode}Materials"] = str(artifact.materials.resolve())
            report[f"mode{mode}Status"] = artifact.status
            for key, value in artifact.metadata.items():
                report[f"mode{mode}{key[0].upper()}{key[1:]}"] = value
        for mode, entry in sorted(self._extra_entries.items()):
            strategies[str(mode)] = dict(entry)
            report[f"mode{mode}Status"] = str(entry.get("status", "unknown"))
        report["strategies"] = strategies
        report["notes"] = list(self.notes)
        return report
