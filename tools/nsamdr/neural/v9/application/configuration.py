"""Current SR-first Quick configuration resolution."""
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

# The only supported training workflow is V13.3 SR-first Quick:
# deterministic B -> learned multi-map SR candidate C -> BenefitSelector -> F.
# detail_learning_rate is now the ACTUAL V13.3 detail-body LR; V13.3 owns that
# phase directly, so historical 3x/16x wrappers cannot reinterpret it. The selector
# compatibility value remains 1e-3/20 because the unchanged legacy selector phase
# adapter still resolves it to the proven effective 1e-3.
QUICK_WORK_BUDGET: dict[str, int | float] = {
    "identity_epochs": 0,
    "residual_epochs": 0,
    "seam_proof_epochs": 0,
    "seam_authority_epochs": 0,
    "boundary_epochs": 0,
    "detail_epochs": 8,
    "physical_finetune_epochs": 3,
    "tiles_per_epoch": 384,
    "validation_tiles": 32,
    "raven_downstream_tiles_per_epoch": 384,
    "tile_size": 32,
    "batch_size": 1,
    "detail_albedo_max_delta": 0.40,
    "detail_recovery_required": 0.40,
    "detail_gradient_recovery_required": 0.30,
    "detail_win_fraction_required": 0.60,
    "detail_regression_fraction_max": 0.25,
    "detail_learning_rate": 1.0e-3,
    "finetune_learning_rate": 1.0e-3 / 20.0,
    "weight_decay": 0.0,
}

_ALLOWED_QUICK_STOP_PHASES = {None, "detail-reconstruction"}


def is_sr_first_quick_config(config: V9Config) -> bool:
    """Return whether a config has the current SR-first Quick stage schedule.

    Purpose:
        Identify the only training schedule that the current application supports.
    Called by:
        assert_sr_first_quick_config(), representative validation.
    Calls:
        No project functions.
    """
    return bool(
        int(config.identity_epochs) == 0
        and int(config.residual_epochs) == 0
        and int(config.seam_proof_epochs) == 0
        and int(config.seam_authority_epochs) == 0
        and int(config.boundary_epochs) == 0
        and int(config.detail_epochs) > 0
        and int(config.physical_finetune_epochs) > 0
        and float(config.detail_albedo_max_delta) >= 0.40 - 1.0e-8
    )


def assert_sr_first_quick_config(config: V9Config) -> None:
    """Fail closed when a persisted experiment predates the current Quick contract.

    Purpose:
        Prevent old geometry/profile/seam schedules or stale optimizer scales from resuming.
    Called by:
        TrainingApplication.run().
    Calls:
        is_sr_first_quick_config().
    """
    if not is_sr_first_quick_config(config):
        raise RuntimeError(
            "Pre-V13.3 Quick schedules are retired; allocate a new SR-first Quick experiment."
        )

    expected = {
        "tile_size": 32,
        "batch_size": 1,
        "detail_learning_rate": 1.0e-3,
        "finetune_learning_rate": 1.0e-3 / 20.0,
        "weight_decay": 0.0,
    }
    mismatches: list[str] = []
    for name, value in expected.items():
        actual = getattr(config, name)
        if isinstance(value, float):
            if abs(float(actual) - value) > 1.0e-12:
                mismatches.append(f"{name}={actual!r} expected {value!r}")
        elif int(actual) != int(value):
            mismatches.append(f"{name}={actual!r} expected {value!r}")
    if mismatches:
        raise RuntimeError(
            "Quick experiment does not use the proven V13.3 SR training regime: "
            + "; ".join(mismatches)
        )


def assert_quick_stop_phase(phase: str | None) -> None:
    """Reject hidden requests for retired geometry/profile/seam training phases.

    Purpose:
        Keep the command-line diagnostic surface aligned with the active SR-only trainer.
    Called by:
        TrainingApplication._run_diagnostic_stage().
    Calls:
        No project functions.
    """
    if phase not in _ALLOWED_QUICK_STOP_PHASES:
        raise RuntimeError(
            f"stop-after phase {phase!r} was retired by V13.3; "
            "only detail-reconstruction is available."
        )


class ConfigResolver:
    """Resolve the one supported SR-first Quick work budget."""

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

    def resolve_overrides(
        self,
        options: TrainingOptions,
        base: V9Config,
        dataset_config: V9Config,
    ) -> dict[str, Any]:
        """Calculate the immutable override set for current SR-first Quick training.

        Purpose:
            Remove retired Quick/Full curricula from application configuration.
        Called by:
            ExperimentService._allocate_new().
        Calls:
            ConfigResolver._set_values(), V9Config.apply_performance_profile(), V9Config.validate().
        """
        if str(options.training_mode).lower() != "quick":
            raise RuntimeError(
                "Full Training is disabled until it is converted to the V13 SR-first authority."
            )

        resolved = copy.deepcopy(base)
        self._set_values(
            resolved,
            {field: getattr(dataset_config, field) for field in DATASET_SCOPE_FIELDS},
        )
        self._set_values(resolved, CANONICAL_SEMANTIC_OVERRIDES)

        resolved.apply_performance_profile(options.performance_profile)
        self._set_values(resolved, QUICK_WORK_BUDGET)

        if options.tiles_per_epoch is not None:
            resolved.tiles_per_epoch = max(1, int(options.tiles_per_epoch))
        if options.validation_tiles is not None:
            resolved.validation_tiles = max(32, int(options.validation_tiles))

        resolved.data_loader_workers = max(0, int(options.workers))
        resolved.data_loader_prefetch_factor = max(1, int(options.prefetch_factor))
        resolved.amp_dtype = options.amp_precision
        resolved.validate()
        assert_sr_first_quick_config(resolved)

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
