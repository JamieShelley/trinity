"""Quick/Full configuration resolution for one canonical production model."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ..config import V9Config
from .domain import TrainingOptions


DATASET_SCOPE_FIELDS = (
    "dataset_manifest",
    "dataset_root",
    "max_families",
    "crops_per_family",
    "source_crop_size",
    "min_source_dimension",
    "min_auxiliary_dimension",
    "validation_fraction",
    "require_complete_pbr_family",
)

CANONICAL_SEMANTIC_OVERRIDES: dict[str, Any] = {
    "appearance_enabled": True,
    "detail_reconstruction_enabled": True,
    "seam_directional_enabled": True,
    "raven_full_pipeline_preview_enabled": True,
    "raven_representative_preview_enabled": False,
    "raven_train_only_enabled": False,
    "preview_allow_unqualified_downstream": False,
}

QUICK_WORK_BUDGET: dict[str, int] = {
    "identity_epochs": 3,
    "residual_epochs": 1,
    "seam_proof_epochs": 1,
    "seam_authority_epochs": 1,
    "boundary_epochs": 1,
    "detail_epochs": 1,
    "physical_finetune_epochs": 1,
    "tiles_per_epoch": 64,
    "validation_tiles": 8,
    "raven_downstream_tiles_per_epoch": 16,
    "parametric_primitive_train_tiles_per_epoch": 14,
}

FULL_MINIMUM_WORK_BUDGET: dict[str, int] = {
    "identity_epochs": 12,
    "residual_epochs": 1,
    "seam_proof_epochs": 3,
    "seam_authority_epochs": 2,
    "boundary_epochs": 2,
    "detail_epochs": 5,
    "physical_finetune_epochs": 3,
    "raven_downstream_tiles_per_epoch": 128,
}


class ConfigResolver:
    """Resolve Quick/Full work budgets while preserving production semantics."""

    def _set_values(self, config: V9Config, values: dict[str, Any]) -> None:
        """Assign validated named fields to one mutable V9Config.

        Purpose:
            Centralise guarded config mutation used by canonical override resolution.
        Called by:
            ConfigResolver.resolve_overrides().
        Calls:
            setattr().
        """
        for key, value in values.items():
            if not hasattr(config, key):
                raise RuntimeError(f"canonical workflow references unknown config field: {key}")
            setattr(config, key, value)

    def _full_budget(self, resolved: V9Config) -> dict[str, int]:
        """Build the Full minimum-work overlay from the production config.

        Purpose:
            Preserve larger production budgets while retaining the retired one-slot residual minimum.
        Called by:
            ConfigResolver.resolve_overrides().
        Calls:
            No project functions.
        """
        values = {
            field: max(int(getattr(resolved, field)), minimum)
            for field, minimum in FULL_MINIMUM_WORK_BUDGET.items()
        }
        values["residual_epochs"] = 1
        return values

    def resolve_overrides(
        self,
        options: TrainingOptions,
        base: V9Config,
        dataset_config: V9Config,
    ) -> dict[str, Any]:
        """Calculate the immutable experiment override set for Quick or Full.

        Purpose:
            Reproduce the canonical entry script's exact semantic/work-budget resolution.
        Called by:
            ExperimentService.allocate_or_resume().
        Calls:
            ConfigResolver._set_values(), ConfigResolver._full_budget(),
            V9Config.apply_performance_profile(), V9Config.validate().
        """
        resolved = copy.deepcopy(base)
        if options.training_mode == "quick":
            self._set_values(
                resolved,
                {field: getattr(dataset_config, field) for field in DATASET_SCOPE_FIELDS},
            )
        self._set_values(resolved, CANONICAL_SEMANTIC_OVERRIDES)

        if options.training_mode == "quick":
            self._set_values(resolved, QUICK_WORK_BUDGET)
        else:
            self._set_values(resolved, self._full_budget(resolved))

        resolved.parametric_primitive_train_tiles_per_epoch = max(
            int(getattr(resolved, "parametric_primitive_batch_size", 14)),
            14,
        )
        if options.tiles_per_epoch is not None:
            resolved.tiles_per_epoch = max(1, int(options.tiles_per_epoch))
        if options.validation_tiles is not None:
            resolved.validation_tiles = max(1, int(options.validation_tiles))

        resolved.apply_performance_profile(options.performance_profile)
        resolved.data_loader_workers = max(0, int(options.workers))
        resolved.data_loader_prefetch_factor = max(1, int(options.prefetch_factor))
        resolved.amp_dtype = options.amp_precision
        resolved.validate()

        base_payload = base.to_dict()
        resolved_payload = resolved.to_dict()
        return {
            key: value
            for key, value in resolved_payload.items()
            if base_payload.get(key) != value
        }

    def resolve_path(self, repo_root: Path, requested: Path) -> Path:
        """Resolve a CLI path against the repository root.

        Purpose:
            Keep path normalisation out of high-level application flow.
        Called by:
            TrainingApplication._load_configs().
        Calls:
            Path.resolve().
        """
        path = requested if requested.is_absolute() else repo_root / requested
        return path.resolve()
