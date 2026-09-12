from __future__ import annotations

"""Permit the explicit V13.3 SR-first Quick schedule through legacy config validation.

The historical V9 validator clamps identity/residual epochs to at least one because
those stages used to be mandatory. V13.3 deliberately requests them as zero after
G/profile/seam execution was retired from the active production path. Preserve zero
only for that unmistakable SR-first schedule; every other config keeps historical bounds.

Quick stays at the proven 32x32 LR tile/batch-1 capacity scale. The active production
path is deterministic B -> learned multi-map SR candidate C -> BenefitSelector F.
"""

from typing import Any

from .config import V9Config


SR_FIRST_QUICK_CONFIG_REVISION = "V13.3"
SR_FIRST_QUICK_TILE_SIZE = 32
SR_FIRST_QUICK_BATCH_SIZE = 1
_INSTALLED = False
_ORIGINAL_VALIDATE: Any = None


def _requested_sr_first_quick(config: V9Config) -> bool:
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


def _validate_with_sr_first_quick(self: V9Config) -> None:
    if _ORIGINAL_VALIDATE is None:
        raise RuntimeError("V13.3 Quick config contract installed without base validator")
    preserve_sr_first_schedule = _requested_sr_first_quick(self)
    _ORIGINAL_VALIDATE(self)
    if preserve_sr_first_schedule:
        self.identity_epochs = 0
        self.residual_epochs = 0
        self.seam_proof_epochs = 0
        self.seam_authority_epochs = 0
        self.boundary_epochs = 0
        self.tile_size = SR_FIRST_QUICK_TILE_SIZE
        self.batch_size = SR_FIRST_QUICK_BATCH_SIZE


def install_sr_first_quick_config_contract() -> None:
    global _INSTALLED, _ORIGINAL_VALIDATE
    if _INSTALLED:
        return
    _ORIGINAL_VALIDATE = V9Config.validate
    V9Config.validate = _validate_with_sr_first_quick
    _INSTALLED = True
