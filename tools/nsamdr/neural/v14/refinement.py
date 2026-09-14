from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ZeroHead(nn.Conv2d):
    """Zero-initialized map head so candidate C starts exactly at baseline B."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(in_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class EDSRResidualBlock(nn.Module):
    """EDSR-style residual block: Conv-ReLU-Conv with residual scaling and no BN."""

    def __init__(self, channels: int, residual_scale: float) -> None:
        super().__init__()
        if not 0.0 < float(residual_scale) <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=False)
        self.residual_scale = float(residual_scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.act(self.conv1(value)))
        return value + residual * self.residual_scale


class CheckpointedResidualBody(nn.Module):
    """Run one single-resolution residual body with optional block-segment checkpointing."""

    def __init__(
        self,
        channels: int,
        blocks: int,
        residual_scale: float,
        *,
        checkpoint_segment_blocks: int,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        if blocks < 1:
            raise ValueError("blocks must be positive")
        if checkpoint_segment_blocks < 1 or blocks % checkpoint_segment_blocks != 0:
            raise ValueError("checkpoint_segment_blocks must divide blocks")
        self.blocks = nn.ModuleList(
            EDSRResidualBlock(channels, residual_scale) for _ in range(blocks)
        )
        self.segment_blocks = int(checkpoint_segment_blocks)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

    def _run_range(self, value: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for index in range(start, end):
            value = self.blocks[index](value)
        return value

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for start in range(0, len(self.blocks), self.segment_blocks):
            end = start + self.segment_blocks
            if self.use_gradient_checkpointing and self.training and value.requires_grad:
                run = partial(self._run_range, start=start, end=end)
                value = checkpoint(run, value, use_reentrant=False)
            else:
                value = self._run_range(value, start, end)
        return value


class PhysicalMapTail(nn.Module):
    """Map-specific residual refinement followed by a zero-initialized raw head."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        blocks: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(EDSRResidualBlock(channels, residual_scale) for _ in range(blocks))
        )
        self.head = ZeroHead(channels, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(value))


class SingleScaleHRRefinementTrunk(nn.Module):
    """V15.0 single-resolution HR residual backbone.

    The trunk follows the EDSR pattern: shallow feature extraction, a deep residual body,
    one long feature skip, and reconstruction heads. It never changes spatial resolution.
    """

    def __init__(
        self,
        *,
        baseline_channels: int,
        context_channels: int,
        channels: int,
        blocks: int,
        map_tail_blocks: int,
        residual_scale: float,
        checkpoint_segment_blocks: int,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(
            baseline_channels + context_channels,
            channels,
            3,
            padding=1,
        )
        self.body = CheckpointedResidualBody(
            channels,
            blocks,
            residual_scale,
            checkpoint_segment_blocks=checkpoint_segment_blocks,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.body_tail = nn.Conv2d(channels, channels, 3, padding=1)
        self.albedo_tail = PhysicalMapTail(
            channels,
            3,
            map_tail_blocks,
            residual_scale,
        )
        self.normal_tail = PhysicalMapTail(
            channels,
            2,
            map_tail_blocks,
            residual_scale,
        )
        self.material_tail = PhysicalMapTail(
            channels,
            3,
            map_tail_blocks,
            residual_scale,
        )

    def forward(
        self,
        baseline: torch.Tensor,
        context_hr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        stem = self.stem(torch.cat((baseline.float(), context_hr.float()), dim=1))
        value = self.body(stem)
        value = stem + self.body_tail(value)
        return {
            "albedo": self.albedo_tail(value),
            "normal": self.normal_tail(value),
            "material": self.material_tail(value),
        }
