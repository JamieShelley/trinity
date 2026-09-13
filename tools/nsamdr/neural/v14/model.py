from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .baseline import Baseline4x, normalize_xy
from .config import V14Config


MODEL_SCHEMA = "NSAMDR_HR_FIRST_MULTI_MAP_SR_4X_V14_0"


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x))) * 0.20


class ZeroHead(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, *, bias: float = 0.0) -> None:
        super().__init__(in_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.weight)
        nn.init.constant_(self.bias, bias)


class LRContextEncoder(nn.Module):
    """Encode LR physical evidence and learn sub-pixel phase context with PixelShuffle."""

    def __init__(self, channels: int, blocks: int, scale: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(8, channels, 3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.phase = nn.Conv2d(channels, channels * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, lr_maps: torch.Tensor) -> torch.Tensor:
        x = self.body(F.gelu(self.stem(lr_maps.float())))
        return self.shuffle(self.phase(x))


class HRRefinementTrunk(nn.Module):
    """Generate the missing signal while reasoning in final-output coordinates."""

    def __init__(self, context_channels: int, channels: int, blocks: int) -> None:
        super().__init__()
        self.stem = nn.Conv2d(8 + context_channels, channels, 3, padding=1)
        self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))
        self.albedo_head = ZeroHead(channels, 3)
        self.normal_head = ZeroHead(channels, 2)
        self.material_head = ZeroHead(channels, 3)

    def forward(self, baseline: torch.Tensor, context_hr: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.body(F.gelu(self.stem(torch.cat((baseline.float(), context_hr.float()), dim=1))))
        return {
            "albedo": torch.tanh(self.albedo_head(x)),
            "normal": torch.tanh(self.normal_head(x)),
            "material": torch.tanh(self.material_head(x)),
        }


def _gray_gradient(value: torch.Tensor) -> torch.Tensor:
    gray = value.float().mean(dim=1, keepdim=True)
    dx = F.pad((gray[..., :, 1:] - gray[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((gray[..., 1:, :] - gray[..., :-1, :]).abs(), (0, 0, 0, 1))
    return dx + dy


class BenefitSelector(nn.Module):
    """Production-visible local safety selector. It cannot see authored HR targets."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(12, channels, 3, padding=1),
            nn.GELU(),
            ResidualBlock(channels),
            ResidualBlock(channels),
            nn.Conv2d(channels, 1, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.2)

    def forward(
        self,
        baseline_albedo: torch.Tensor,
        candidate_albedo: torch.Tensor,
        lr_albedo: torch.Tensor,
    ) -> torch.Tensor:
        delta = (candidate_albedo.float() - baseline_albedo.float()).abs()
        bgrad = _gray_gradient(baseline_albedo)
        cgrad = _gray_gradient(candidate_albedo)
        lr_edge = _gray_gradient(lr_albedo)
        lr_edge = F.interpolate(lr_edge, size=baseline_albedo.shape[-2:], mode="bilinear", align_corners=False)
        features = torch.cat((baseline_albedo.float(), candidate_albedo.float(), delta, bgrad, cgrad, lr_edge), dim=1)
        return self.net(features)


class NSAMDRV14(nn.Module):
    """Clean V14 production graph: B -> HR-first C -> BenefitSelector F."""

    def __init__(self, config: V14Config | None = None) -> None:
        super().__init__()
        self.config = config or V14Config()
        self.config.validate()
        self.baseline = Baseline4x(self.config.scale)
        self.context_encoder = LRContextEncoder(self.config.lr_context_channels, self.config.lr_blocks, self.config.scale)
        self.hr_refiner = HRRefinementTrunk(self.config.lr_context_channels, self.config.hr_channels, self.config.hr_blocks)
        self.selector = BenefitSelector(self.config.selector_channels)

    def forward(
        self,
        lr_albedo: torch.Tensor,
        lr_normal: torch.Tensor,
        lr_material: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b_albedo, b_normal, b_material = self.baseline(lr_albedo, lr_normal, lr_material)
        lr_maps = torch.cat((lr_albedo.float(), lr_normal.float(), lr_material.float()), dim=1)
        context_hr = self.context_encoder(lr_maps)
        baseline_maps = torch.cat((b_albedo, b_normal, b_material), dim=1)
        residual = self.hr_refiner(baseline_maps, context_hr)

        c_albedo = (b_albedo + residual["albedo"] * self.config.albedo_residual_cap).clamp(0.0, 1.0)
        c_normal = normalize_xy(b_normal + residual["normal"] * self.config.normal_residual_cap)
        c_material = (b_material + residual["material"] * self.config.material_residual_cap).clamp(0.0, 1.0)

        selector_logits = self.selector(b_albedo, c_albedo, lr_albedo)
        gate = torch.sigmoid(selector_logits.float())
        f_albedo = (b_albedo * (1.0 - gate) + c_albedo * gate).clamp(0.0, 1.0)
        f_normal = normalize_xy(b_normal * (1.0 - gate) + c_normal * gate)
        f_material = (b_material * (1.0 - gate) + c_material * gate).clamp(0.0, 1.0)

        return {
            "baseline_albedo": b_albedo,
            "baseline_normal": b_normal,
            "baseline_material": b_material,
            "candidate_albedo": c_albedo,
            "candidate_normal": c_normal,
            "candidate_material": c_material,
            "candidate_residual_albedo": c_albedo - b_albedo,
            "candidate_residual_normal": c_normal - b_normal,
            "candidate_residual_material": c_material - b_material,
            "selector_logits": selector_logits,
            "selector_probability": gate,
            "albedo": f_albedo,
            "normal": f_normal,
            "material": f_material,
        }

    def set_candidate_training(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.context_encoder, self.hr_refiner):
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def set_selector_training(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.selector.parameters():
            parameter.requires_grad_(True)

    def architecture_contract(self) -> dict[str, object]:
        return {
            "schema": MODEL_SCHEMA,
            "revision": "V14",
            "scale": self.config.scale,
            "productionForward": "LR -> deterministic B + learned subpixel context -> HR refinement C -> BenefitSelector F",
            "candidateIdentityAtInitialization": "C == B",
            "activeComponents": ("baseline", "context_encoder", "hr_refiner", "selector"),
            "retiredComponents": (),
            "geometryPixelAuthority": False,
            "seamPixelAuthority": False,
            "profilePixelAuthority": False,
        }
