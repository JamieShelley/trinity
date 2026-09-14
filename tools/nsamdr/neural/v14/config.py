from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


MODEL_SCHEMA = "NSAMDR_HR_FIRST_MULTI_MAP_SR_4X_V16_0"


@dataclass
class V16Config:
    """Configuration for the V16.0 SwinIR-style HR-first candidate."""

    schema: str = MODEL_SCHEMA
    scale: int = 4
    train_hr_size: int = 512
    train_lr_size: int = 128
    validation_hr_size: int = 512
    validation_lr_size: int = 128
    native_validation_max_families: int = 4
    minimum_heldout_samples: int = 4
    tiles_per_epoch: int = 192
    validation_tiles: int = 32
    clean_epochs: int = 4
    robust_epochs: int = 4
    selector_epochs: int = 3

    # Keep the conservative Adam learning rate used by the stable V15 candidate.
    sr_learning_rate: float = 2.0e-4
    selector_learning_rate: float = 2.0e-4
    weight_decay: float = 0.0

    # LR evidence remains phase-neutral context only.
    lr_context_channels: int = 32
    lr_blocks: int = 5

    # V16.0 candidate backbone. The learned candidate stays at one HR spatial scale.
    # The 6 x 6 residual-Swin layout and window size 8 follow the medium SwinIR pattern.
    # The embedding width is reduced because NSAMDR performs attention at 512 HR pixels.
    hr_channels: int = 96
    swin_groups: int = 6
    swin_blocks_per_group: int = 6
    swin_num_heads: int = 6
    swin_window_size: int = 8
    swin_mlp_ratio: float = 2.0
    map_tail_blocks: int = 2
    tail_residual_scale: float = 0.10
    use_gradient_checkpointing: bool = True

    selector_channels: int = 24
    albedo_residual_cap: float = 0.40
    normal_residual_cap: float = 0.20
    material_residual_cap: float = 0.25
    dataset_manifest: str = "artifacts/nsamdr/training_v9_preview_raven/dataset_manifest.json"
    seed: int = 14001

    candidate_edge_recovery_required: float = 0.60
    candidate_global_recovery_required: float = 0.45
    candidate_gradient_recovery_required: float = 0.35
    candidate_positive_edge_fraction_required: float = 0.75
    candidate_positive_global_fraction_required: float = 0.75
    candidate_worst_recovery_min: float = -0.10
    candidate_lattice_cell_excess_max: float = 0.15

    selector_edge_retention_required: float = 0.90
    selector_global_retention_required: float = 0.90
    protected_preservation_required: float = 0.99

    production_tile_lr: int = 128
    production_overlap_lr: int = 16

    @property
    def sr_epochs(self) -> int:
        return int(self.clean_epochs + self.robust_epochs)

    @property
    def swin_depth(self) -> int:
        return int(self.swin_groups * self.swin_blocks_per_group)

    def validate(self) -> None:
        if self.schema != MODEL_SCHEMA:
            raise ValueError(f"config schema must be {MODEL_SCHEMA}")
        if self.scale != 4:
            raise ValueError("V16.0 currently supports exactly 4x reconstruction")
        if self.train_hr_size != self.train_lr_size * self.scale:
            raise ValueError("train_hr_size must equal train_lr_size * scale")
        if self.validation_hr_size != self.validation_lr_size * self.scale:
            raise ValueError("validation_hr_size must equal validation_lr_size * scale")
        if self.production_overlap_lr * 2 >= self.production_tile_lr:
            raise ValueError("production overlap must be less than half the LR tile")

        architecture_values = (
            self.lr_context_channels,
            self.lr_blocks,
            self.hr_channels,
            self.swin_groups,
            self.swin_blocks_per_group,
            self.swin_num_heads,
            self.swin_window_size,
            self.map_tail_blocks,
            self.selector_channels,
        )
        if min(architecture_values) < 1:
            raise ValueError("all V16.0 architecture dimensions and counts must be positive")
        if self.hr_channels % self.swin_num_heads != 0:
            raise ValueError("hr_channels must be divisible by swin_num_heads")
        if self.swin_window_size < 2 or self.swin_window_size % 2 != 0:
            raise ValueError("swin_window_size must be an even integer >= 2")
        if self.swin_mlp_ratio <= 0.0:
            raise ValueError("swin_mlp_ratio must be positive")
        if not 0.0 < float(self.tail_residual_scale) <= 1.0:
            raise ValueError("tail_residual_scale must be in (0, 1]")

        work_values = (
            self.tiles_per_epoch,
            self.validation_tiles,
            self.minimum_heldout_samples,
            self.sr_epochs,
            self.selector_epochs,
        )
        if min(work_values) < 1:
            raise ValueError("training and qualification work budgets must be positive")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["sr_epochs"] = self.sr_epochs
        payload["swin_depth"] = self.swin_depth
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "V16Config":
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = cls.__dataclass_fields__
        kwargs = {key: value for key, value in payload.items() if key in fields}
        config = cls(**kwargs)
        config.validate()
        return config


# Compatibility aliases for the existing package path and helper modules.
V15Config = V16Config
V14Config = V16Config
