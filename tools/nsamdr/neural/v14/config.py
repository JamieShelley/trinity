from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


MODEL_SCHEMA = "NSAMDR_HR_FIRST_MULTI_MAP_SR_4X_V14_3"


@dataclass
class V14Config:
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
    sr_learning_rate: float = 2.0e-4
    selector_learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-5

    lr_context_channels: int = 32
    lr_blocks: int = 5

    hr_channels: int = 48
    half_channels: int = 64
    quarter_channels: int = 96
    hr_encoder_blocks: int = 4
    half_encoder_blocks: int = 4
    quarter_encoder_blocks: int = 6
    bottleneck_blocks: int = 6
    half_decoder_blocks: int = 4
    hr_decoder_blocks: int = 4
    map_tail_blocks: int = 2
    attention_reduction: int = 8
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

    def validate(self) -> None:
        if self.schema != MODEL_SCHEMA:
            raise ValueError(f"config schema must be {MODEL_SCHEMA}")
        if self.scale != 4:
            raise ValueError("V14.3 currently supports exactly 4x reconstruction")
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
            self.half_channels,
            self.quarter_channels,
            self.hr_encoder_blocks,
            self.half_encoder_blocks,
            self.quarter_encoder_blocks,
            self.bottleneck_blocks,
            self.half_decoder_blocks,
            self.hr_decoder_blocks,
            self.map_tail_blocks,
            self.attention_reduction,
            self.selector_channels,
        )
        if min(architecture_values) < 1:
            raise ValueError("all V14.3 architecture dimensions and block counts must be positive")

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
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "V14Config":
        payload = json.loads(path.read_text(encoding="utf-8"))
        fields = cls.__dataclass_fields__
        kwargs = {key: value for key, value in payload.items() if key in fields}
        config = cls(**kwargs)
        config.validate()
        return config
