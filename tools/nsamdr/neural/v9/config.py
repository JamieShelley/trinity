"""Minimal source-dataset configuration retained for V16 compatibility."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, ClassVar


@dataclass
class V9Config:
    """Fields still consumed by the native Raven source-data builder.

    The class name is retained to avoid changing persisted source-builder imports.
    Historical V9 model/training configuration no longer lives in this package.
    """

    dataset_manifest: str = "artifacts/nsamdr/training_v9_preview_raven/dataset_manifest.json"
    dataset_root: str = "artifacts/nsamdr/training_v9_preview_raven"
    source_crop_size: int = 512
    tile_size: int = 128
    target_scale: int = 4
    seed: int = 1337

    _KEYS: ClassVar[dict[str, str]] = {
        "datasetManifest": "dataset_manifest",
        "datasetRoot": "dataset_root",
        "sourceCropSize": "source_crop_size",
        "tileSize": "tile_size",
        "targetScale": "target_scale",
        "seed": "seed",
    }

    @classmethod
    def load(cls, path: Path | None = None) -> "V9Config":
        if path is None or not Path(path).is_file():
            value = cls()
            value.validate()
            return value
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"source dataset config must be a JSON object: {path}")
        values: dict[str, Any] = {}
        for key, field_name in cls._KEYS.items():
            if key in payload:
                values[field_name] = payload[key]
        value = cls(**values)
        value.validate()
        return value

    def validate(self) -> None:
        if self.source_crop_size <= 0 or self.tile_size <= 0 or self.target_scale <= 0:
            raise ValueError("source crop, tile size and target scale must be positive")
        if self.source_crop_size != self.tile_size * self.target_scale:
            raise ValueError("source_crop_size must equal tile_size * target_scale")
        if not self.dataset_manifest or not self.dataset_root:
            raise ValueError("dataset paths must be non-empty")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
