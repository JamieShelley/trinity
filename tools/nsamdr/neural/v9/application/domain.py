"""Application-layer value objects for canonical NSAMDR experiment training."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import V9Config


@dataclass(frozen=True)
class TrainingOptions:
    """Parsed immutable command-line choices for one training application run."""

    repo_root: Path
    base_config: Path
    dataset_config: Path | None
    experiment: str
    control: str
    training_mode: str
    preset: str
    tiles_per_epoch: int | None
    validation_tiles: int | None
    performance_profile: str
    workers: int
    prefetch_factor: int
    amp_precision: str
    device: str
    early_stop_patience: int
    early_stop_min_delta: float
    result_file: Path | None
    allocate_only: bool
    stop_after_phase: str | None


@dataclass
class ExperimentContext:
    """Resolved mutable experiment context shared by composed application services."""

    experiment_id: str
    directory: Path
    config: V9Config
    resume: bool
    asset_name: str
    asset_query: str
    selection_key: str


@dataclass(frozen=True)
class StageDefinition:
    """One pass-driven production stage and its promotion gate."""

    phase: str
    gate_label: str
    gate: Any
