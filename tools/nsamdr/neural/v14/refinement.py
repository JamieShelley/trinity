from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ZeroHead(nn.Conv2d):
    """Zero-initialized physical-map head so candidate C starts exactly at baseline B."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(in_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class ResidualLimiter(nn.Module):
    """Clamp a residual in the forward pass and pass gradients straight through."""

    def __init__(self, cap: float) -> None:
        super().__init__()
        if cap <= 0.0:
            raise ValueError("residual cap must be positive")
        self.cap = float(cap)

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        value = raw.float()
        clipped = value.clamp(-self.cap, self.cap)
        return value + (clipped - value).detach()


class ChannelAttention(nn.Module):
    """Channel-only attention. It does not assign spatial pixel authority."""

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.net(self.pool(value))


class RCAB(nn.Module):
    """Residual channel-attention block with an identity-safe initial state."""

    def __init__(
        self,
        channels: int,
        attention_reduction: int,
        residual_scale: float = 0.20,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()
        self.attention = ChannelAttention(channels, attention_reduction)
        self.residual_scale = float(residual_scale)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.act(self.conv1(value)))
        residual = self.attention(residual)
        return value + residual * self.residual_scale


class ResidualGroup(nn.Module):
    """RCAB group with an identity skip, group scaling, and optional checkpointing."""

    def __init__(
        self,
        channels: int,
        blocks: int,
        attention_reduction: int,
        *,
        residual_scale: float,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        if not 0.0 < float(residual_scale) <= 1.0:
            raise ValueError("ResidualGroup residual_scale must be in (0, 1]")
        self.blocks = nn.Sequential(
            *(RCAB(channels, attention_reduction) for _ in range(blocks))
        )
        self.tail = nn.Conv2d(channels, channels, 3, padding=1)
        self.residual_scale = float(residual_scale)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def _forward_impl(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.tail(self.blocks(value))
        return value + residual * self.residual_scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training and value.requires_grad:
            return checkpoint(self._forward_impl, value, use_reentrant=False)
        return self._forward_impl(value)


class DownsampleFeatures(nn.Module):
    """Reduce spatial size while increasing feature capacity."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(value))


class UpsampleAndFuse(nn.Module):
    """Phase-neutral bilinear resize followed by HR convolution and skip fusion."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.adapter = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.skip_adapter = (
            nn.Identity()
            if skip_channels == out_channels
            else nn.Conv2d(skip_channels, out_channels, 1)
        )
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)

    def forward(self, value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(
            value,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        value = F.gelu(self.adapter(value))
        skip = self.skip_adapter(skip)
        return F.gelu(self.fuse(torch.cat((value, skip), dim=1)))


class PhysicalMapTail(nn.Module):
    """Map-specific refinement followed by a raw zero-initialized residual head."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        blocks: int,
        attention_reduction: int,
        *,
        residual_group_scale: float,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.refinement = ResidualGroup(
            channels,
            blocks,
            attention_reduction,
            residual_scale=residual_group_scale,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.head = ZeroHead(channels, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.refinement(value))


class MultiScaleHRRefinementTrunk(nn.Module):
    """Deep phase-neutral multi-scale reconstruction trunk for candidate C."""

    def __init__(
        self,
        *,
        baseline_channels: int,
        context_channels: int,
        hr_channels: int,
        half_channels: int,
        quarter_channels: int,
        hr_encoder_blocks: int,
        half_encoder_blocks: int,
        quarter_encoder_blocks: int,
        bottleneck_blocks: int,
        half_decoder_blocks: int,
        hr_decoder_blocks: int,
        map_tail_blocks: int,
        attention_reduction: int,
        residual_group_scale: float,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv2d(
            baseline_channels + context_channels,
            hr_channels,
            3,
            padding=1,
        )

        group_kwargs = {
            "attention_reduction": attention_reduction,
            "residual_scale": residual_group_scale,
            "use_gradient_checkpointing": use_gradient_checkpointing,
        }
        self.hr_encoder = ResidualGroup(
            hr_channels,
            hr_encoder_blocks,
            **group_kwargs,
        )
        self.down_half = DownsampleFeatures(hr_channels, half_channels)
        self.half_encoder = ResidualGroup(
            half_channels,
            half_encoder_blocks,
            **group_kwargs,
        )
        self.down_quarter = DownsampleFeatures(half_channels, quarter_channels)
        self.quarter_encoder = ResidualGroup(
            quarter_channels,
            quarter_encoder_blocks,
            **group_kwargs,
        )
        self.bottleneck = ResidualGroup(
            quarter_channels,
            bottleneck_blocks,
            **group_kwargs,
        )

        self.up_half = UpsampleAndFuse(
            quarter_channels,
            half_channels,
            half_channels,
        )
        self.half_decoder = ResidualGroup(
            half_channels,
            half_decoder_blocks,
            **group_kwargs,
        )
        self.up_hr = UpsampleAndFuse(
            half_channels,
            hr_channels,
            hr_channels,
        )
        self.hr_decoder = ResidualGroup(
            hr_channels,
            hr_decoder_blocks,
            **group_kwargs,
        )

        tail_kwargs = {
            "blocks": map_tail_blocks,
            "attention_reduction": attention_reduction,
            "residual_group_scale": residual_group_scale,
            "use_gradient_checkpointing": use_gradient_checkpointing,
        }
        self.albedo_tail = PhysicalMapTail(hr_channels, 3, **tail_kwargs)
        self.normal_tail = PhysicalMapTail(hr_channels, 2, **tail_kwargs)
        self.material_tail = PhysicalMapTail(hr_channels, 3, **tail_kwargs)

    def forward(
        self,
        baseline: torch.Tensor,
        context_hr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        stem = F.gelu(
            self.stem(torch.cat((baseline.float(), context_hr.float()), dim=1))
        )

        hr_skip = self.hr_encoder(stem)
        half_skip = self.half_encoder(self.down_half(hr_skip))
        quarter = self.quarter_encoder(self.down_quarter(half_skip))
        value = self.bottleneck(quarter)

        value = self.half_decoder(self.up_half(value, half_skip))
        value = self.hr_decoder(self.up_hr(value, hr_skip))
        value = value + stem

        return {
            "albedo": self.albedo_tail(value),
            "normal": self.normal_tail(value),
            "material": self.material_tail(value),
        }
