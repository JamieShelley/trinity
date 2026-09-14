from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ZeroHead(nn.Conv2d):
    """Zero-initialized map head so candidate C starts exactly at baseline B."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(in_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)


class ConvResidualBlock(nn.Module):
    """Small EDSR-style residual block for map-specific reconstruction tails."""

    def __init__(self, channels: int, residual_scale: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=False)
        self.residual_scale = float(residual_scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.act(self.conv1(value)))
        return value + residual * self.residual_scale


class MLP(nn.Module):
    """Transformer feed-forward network used inside one Swin layer."""

    def __init__(self, channels: int, ratio: float) -> None:
        super().__init__()
        hidden = max(1, int(round(channels * float(ratio))))
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(value)))


def window_partition(value: torch.Tensor, window_size: int) -> torch.Tensor:
    """Convert BHWC features into non-overlapping flattened windows."""

    batch, height, width, channels = value.shape
    if height % window_size or width % window_size:
        raise ValueError("window_partition requires dimensions divisible by window_size")
    value = value.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    value = value.permute(0, 1, 3, 2, 4, 5).contiguous()
    return value.view(-1, window_size * window_size, channels)


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Restore flattened windows to BHWC features."""

    windows_per_image = (height // window_size) * (width // window_size)
    if windows_per_image < 1 or windows.shape[0] % windows_per_image:
        raise ValueError("window count does not match target geometry")
    batch = windows.shape[0] // windows_per_image
    channels = windows.shape[-1]
    value = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        channels,
    )
    value = value.permute(0, 1, 3, 2, 4, 5).contiguous()
    return value.view(batch, height, width, channels)


class WindowAttention(nn.Module):
    """Swin window self-attention with learnable relative position bias."""

    def __init__(self, channels: int, window_size: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = int(channels)
        self.window_size = int(window_size)
        self.num_heads = int(num_heads)
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5

        table_size = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size, num_heads)
        )

        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        flattened = coordinates.flatten(1)
        relative = flattened[:, :, None] - flattened[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window_size - 1
        relative[:, :, 1] += window_size - 1
        relative[:, :, 0] *= 2 * window_size - 1
        index = relative.sum(-1)
        self.register_buffer("relative_position_index", index, persistent=False)

        self.qkv = nn.Linear(channels, channels * 3, bias=True)
        self.proj = nn.Linear(channels, channels)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_windows, tokens, channels = value.shape
        qkv = self.qkv(value).reshape(
            batch_windows,
            tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, val = qkv.unbind(0)
        query = query * self.scale

        attention = query @ key.transpose(-2, -1)
        bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        bias = bias.view(tokens, tokens, self.num_heads).permute(2, 0, 1)
        attention = attention + bias.unsqueeze(0).to(attention.dtype)

        if attention_mask is not None:
            window_count = attention_mask.shape[0]
            if batch_windows % window_count:
                raise ValueError("attention mask window count does not match batch")
            batch = batch_windows // window_count
            attention = attention.view(
                batch,
                window_count,
                self.num_heads,
                tokens,
                tokens,
            )
            attention = attention + attention_mask.unsqueeze(0).unsqueeze(2).to(
                attention.dtype
            )
            attention = attention.view(
                batch_windows,
                self.num_heads,
                tokens,
                tokens,
            )

        probabilities = torch.softmax(attention.float(), dim=-1).to(val.dtype)
        result = probabilities @ val
        result = result.transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj(result)


class SwinTransformerLayer(nn.Module):
    """One shifted-window Swin Transformer layer at fixed HR resolution."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        if shift_size not in {0, window_size // 2}:
            raise ValueError("shift_size must be 0 or half the window size")
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.norm1 = nn.LayerNorm(channels)
        self.attention = WindowAttention(channels, window_size, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = MLP(channels, mlp_ratio)

    def _shift_mask(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None

        mask = torch.zeros((1, height, width, 1), device=device)
        window = self.window_size
        shift = self.shift_size
        height_slices = (
            slice(0, -window),
            slice(-window, -shift),
            slice(-shift, None),
        )
        width_slices = (
            slice(0, -window),
            slice(-window, -shift),
            slice(-shift, None),
        )
        region = 0
        for hs in height_slices:
            for ws in width_slices:
                mask[:, hs, ws, :] = region
                region += 1

        mask_windows = window_partition(mask, window).squeeze(-1)
        difference = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return difference.masked_fill(difference != 0, -100.0).masked_fill(
            difference == 0,
            0.0,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        _, _, height, width = value.shape
        if height % self.window_size or width % self.window_size:
            raise ValueError("SwinTransformerLayer requires padded window geometry")

        shortcut = value
        normalized = self.norm1(value.permute(0, 2, 3, 1))
        if self.shift_size:
            shifted = torch.roll(
                normalized,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted = normalized

        windows = window_partition(shifted, self.window_size)
        attended = self.attention(
            windows,
            self._shift_mask(height, width, device=value.device),
        )
        merged = window_reverse(attended, self.window_size, height, width)
        if self.shift_size:
            merged = torch.roll(
                merged,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )

        value = shortcut + merged.permute(0, 3, 1, 2).contiguous()
        mlp_input = self.norm2(value.permute(0, 2, 3, 1))
        mlp_output = self.mlp(mlp_input).permute(0, 3, 1, 2).contiguous()
        return value + mlp_output


class ResidualSwinTransformerGroup(nn.Module):
    """SwinIR-style residual group with alternating regular and shifted windows."""

    def __init__(
        self,
        channels: int,
        blocks: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            SwinTransformerLayer(
                channels=channels,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if index % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
            )
            for index in range(blocks)
        )
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        for layer in self.layers:
            residual = layer(residual)
        return value + self.conv(residual)


class PhysicalMapTail(nn.Module):
    """Map-specific convolutional refinement followed by a zero residual head."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        blocks: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *(
                ConvResidualBlock(channels, residual_scale)
                for _ in range(blocks)
            )
        )
        self.head = ZeroHead(channels, out_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(value))


class SwinIRHRRefinementTrunk(nn.Module):
    """V16.0 SwinIR-style deep feature extractor around deterministic HR baseline B."""

    def __init__(
        self,
        *,
        baseline_channels: int,
        context_channels: int,
        channels: int,
        groups: int,
        blocks_per_group: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        map_tail_blocks: int,
        tail_residual_scale: float,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.stem = nn.Conv2d(
            baseline_channels + context_channels,
            channels,
            3,
            padding=1,
        )
        self.groups = nn.ModuleList(
            ResidualSwinTransformerGroup(
                channels=channels,
                blocks=blocks_per_group,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(groups)
        )
        self.body_tail = nn.Conv2d(channels, channels, 3, padding=1)
        self.albedo_tail = PhysicalMapTail(
            channels,
            3,
            map_tail_blocks,
            tail_residual_scale,
        )
        self.normal_tail = PhysicalMapTail(
            channels,
            2,
            map_tail_blocks,
            tail_residual_scale,
        )
        self.material_tail = PhysicalMapTail(
            channels,
            3,
            map_tail_blocks,
            tail_residual_scale,
        )
        self.apply(self._init_transformer_weights)

    @staticmethod
    def _init_transformer_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _run_group(
        self,
        group: ResidualSwinTransformerGroup,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training and value.requires_grad:
            return checkpoint(group, value, use_reentrant=False)
        return group(value)

    def _pad_to_windows(self, value: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        height, width = value.shape[-2:]
        pad_h = (-height) % self.window_size
        pad_w = (-width) % self.window_size
        if pad_h or pad_w:
            value = F.pad(value, (0, pad_w, 0, pad_h), mode="replicate")
        return value, height, width

    def forward(
        self,
        baseline: torch.Tensor,
        context_hr: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        stem = self.stem(torch.cat((baseline.float(), context_hr.float()), dim=1))
        padded, original_h, original_w = self._pad_to_windows(stem)

        value = padded
        for group in self.groups:
            value = self._run_group(group, value)
        value = padded + self.body_tail(value)
        value = value[..., :original_h, :original_w]

        return {
            "albedo": self.albedo_tail(value),
            "normal": self.normal_tail(value),
            "material": self.material_tail(value),
        }
