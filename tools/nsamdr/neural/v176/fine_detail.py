"""V17.6 Stage 10A: frozen V17.2 + phase-neutral high-frequency specialist.

The pre-existing V17.2 model is unchanged and remains frozen, including
batch/statistics behaviour. Only the new head trains. Source information is
limited to LR physical maps and tensors emitted by the frozen backbone.
No wavelet target, validation pixel or quadtree oracle enters inference.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..v17.model import NSAMDRV17, PhysicalMapEncoder
from .physical_composition import compose_albedo_detail, gaussian_highpass


class FineResidualBlock(nn.Module):
    """Spatially shared, dilation-controlled refinement without phase encodings."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        if channels < 1 or dilation < 1:
            raise ValueError("channels and dilation must be positive")
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 0.20 * self.conv2(self.act(self.conv1(value)))


class FineDetailSpecialist(nn.Module):
    """10-channel HR evidence -> one scalar BC4 fine residual (pre-highpass)."""

    INPUT_CHANNELS = 10
    DILATIONS = (1, 2, 4, 1)

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        if channels < 8:
            raise ValueError("detail width must be at least eight channels")
        self.channels = int(channels)
        self.stem = nn.Conv2d(self.INPUT_CHANNELS, self.channels, 3, padding=1)
        self.blocks = nn.Sequential(
            *(FineResidualBlock(self.channels, d) for d in self.DILATIONS)
        )
        self.head = nn.Conv2d(self.channels, 1, 3, padding=1)
        # Adding the specialist to an authenticated checkpoint changes no pixels.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 4 or evidence.shape[1] != self.INPUT_CHANNELS:
            raise ValueError("detail specialist expects N x 10 x HR x HR evidence")
        return self.head(self.blocks(F.gelu(self.stem(evidence.float()))))


class NSAMDRV176Detail(nn.Module):
    """Stage 10A proof wrapper. The original V17.2 object is never modified."""

    REVISION = "V17.6-detail-10A-research"
    EVIDENCE_CHANNELS = 10

    def __init__(self, backbone: NSAMDRV17, *, detail_channels: int = 32) -> None:
        super().__init__()
        if not isinstance(backbone, NSAMDRV17):
            raise TypeError("Stage 10A requires an authentic V17.2 backbone")
        if backbone.architecture_contract().get("revision") != "V17.2-proof":
            raise ValueError("Stage 10A requires the unchanged V17.2 architecture")
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.specialist = FineDetailSpecialist(channels=detail_channels)

    def train(self, mode: bool = True) -> NSAMDRV176Detail:
        super().train(mode)
        # Never update backbone buffers or enable stochastic train behaviour.
        self.backbone.eval()
        return self

    @staticmethod
    def _evidence(
        lr_albedo: torch.Tensor,
        lr_normal: torch.Tensor,
        lr_material: torch.Tensor,
        original: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if (lr_albedo.ndim != 4 or lr_albedo.shape[1] != 3 or
            lr_normal.shape[:2] != (lr_albedo.shape[0], 2) or
            lr_material.shape[:2] != (lr_albedo.shape[0], 3) or
            lr_normal.shape[-2:] != lr_albedo.shape[-2:] or
            lr_material.shape[-2:] != lr_albedo.shape[-2:]):
            raise ValueError("expected aligned 3/2/3-channel LR physical maps")

        out_size = original["candidate_albedo"].shape[-2:]
        analytic = PhysicalMapEncoder.analytic_edges(
            torch.cat((lr_albedo, lr_normal, lr_material), dim=1)
        )[:, :2]
        analytic = F.interpolate(
            analytic.float(), size=out_size, mode="bilinear", align_corners=False
        )
        def gray(value: torch.Tensor) -> torch.Tensor:
            return value.float().mean(dim=1, keepdim=True)

        # Channel accounting: 1 B albedo + 2 B normal + 1 C0 albedo
        # + 2 C0 normal + 1 existing mid band + 1 detail band + 2 LR
        # analytic gradients = 10. No absolute UV or subpixel phase fields.
        evidence = torch.cat((
            gray(original["baseline_albedo"]),
            original["baseline_normal"].float(),
            gray(original["candidate_albedo"]),
            original["candidate_normal"].float(),
            gray(original["decoder_mid_residual_hr"][:, 0:3]),
            gray(original["decoder_detail_residual"][:, 0:3]),
            analytic,
        ), dim=1)
        if evidence.shape[1] != NSAMDRV176Detail.EVIDENCE_CHANNELS:
            raise RuntimeError("Stage 10A evidence channel accounting changed")
        return evidence

    def forward(
        self,
        lr_albedo: torch.Tensor,
        lr_normal: torch.Tensor,
        lr_material: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self.backbone.eval()
        with torch.no_grad():
            original = self.backbone(lr_albedo, lr_normal, lr_material)
            evidence = self._evidence(lr_albedo,lr_normal,lr_material,original)
        raw = self.specialist(evidence)
        highpass = gaussian_highpass(raw, sigma=1.25, radius=4)
        result = compose_albedo_detail(
            original, highpass,
            albedo_residual_cap=self.backbone.config.albedo_residual_cap,
        )
        result["fine_detail_raw"] = raw
        return result

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.specialist.parameters() if p.requires_grad]

    def architecture_contract(self) -> dict[str, object]:
        return {
            "revision":self.REVISION,
            "backboneRevision":"V17.2-proof",
            "backboneFrozen":True,
            "detailEvidenceChannels":self.EVIDENCE_CHANNELS,
            "detailChannels":self.specialist.channels,
            "detailBlocks":len(self.specialist.blocks),
            "detailDilations":list(FineDetailSpecialist.DILATIONS),
            "highpassSigma":1.25,
            "highpassRadius":4,
            "outputChannels":1,
            "albedoOnly":True,
            "waveletTargetUsedAtInference":False,
            "quadtreeEnabled":False,
            "absoluteCoordinatesUsed":False,
            "subpixelPhaseCoordinatesUsed":False,
            "pixelShuffleUsed":False,
            "transposedConvolutionUsed":False,
            "normalAndMaterialPassThrough":True,
            "candidateBoundedBeforeProjection":True,
            "candidateIdentityAtInitialization":True,
        }
