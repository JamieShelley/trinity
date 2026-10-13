"""V17.6 Stage 10A.1: controlled HR feature evidence and detail diagnostics.

This module does not alter the original V17.2 backbone or the original
ten-channel Stage 10A model. The experimental 58-channel arm adds ONLY the
frozen 48-channel HR decoder features, with zero-initialized extra stem
weights. Both branches start with bit-identical C0 candidate outputs.
"""
from __future__ import annotations

import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

from .fine_detail import NSAMDRV176Detail, FineDetailSpecialist, FineResidualBlock
from .physical_composition import compose_albedo_detail, gaussian_highpass
from .wavelet_supervision import detail_supervision_loss, stationary_haar_details

MODES = ("output10", "hr_features58")
HARD_TRAINING_INDICES = (1, 10)
DIAGNOSTIC_BANDS = ("L1_LH", "L1_HL", "L1_HH", "L2_LH", "L2_HL", "L2_HH")


class HRFeatureSpecialist(nn.Module):
    """Matched head with direct 48-channel frozen 4x decoder evidence."""

    def __init__(self, feature_channels: int = 48, width: int = 32) -> None:
        super().__init__()
        if feature_channels != 48:
            raise ValueError("Stage 10A.1 only tests the authenticated 48 HR decoder features")
        self.channels = width
        self.stem = nn.Conv2d(10 + feature_channels, width, 3, padding=1)
        self.blocks = nn.Sequential(
            *(FineResidualBlock(width, d) for d in (1, 2, 4, 1))
        )
        self.head = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        if evidence.ndim != 4 or evidence.shape[1] != 58:
            raise ValueError("rich specialist requires N x 58 x HR x HR evidence")
        return self.head(self.blocks(F.gelu(self.stem(evidence.float()))))


class HRFeatureDetail(NSAMDRV176Detail):
    """Read frozen decoder.detail_blocks features through a transient hook.

    The hook captures the *pre-head* learned 4x HR feature plane, never an
    authored reference. The pre-existing V17.2 weights, original output
    projection, and training loss are unchanged.
    """

    def __init__(self, backbone, *, detail_channels: int = 32) -> None:
        super().__init__(backbone, detail_channels=detail_channels)
        if self.backbone.decoder.detail_channels != 48:
            raise ValueError("the matched HR-feature experiment requires 48 features")
        self.specialist = HRFeatureSpecialist(width=detail_channels)
        self._feature: torch.Tensor | None = None
        self._capture_handle = self.backbone.decoder.detail_blocks.register_forward_hook(
            self._capture
        )

    def _capture(self, _module, _inputs, output) -> None:
        if not isinstance(output, torch.Tensor) or output.shape[1] != 48:
            raise RuntimeError("V17.2 4x learned features changed")
        self._feature = output.detach()

    def forward(self, lr_albedo, lr_normal, lr_material) -> dict[str, torch.Tensor]:
        self.backbone.eval()
        self._feature = None
        with torch.no_grad():
            original = self.backbone(lr_albedo, lr_normal, lr_material)
            evidence = self._evidence(lr_albedo,lr_normal,lr_material,original)
        features = self._feature
        self._feature = None
        if features is None or features.shape[-2:] != evidence.shape[-2:]:
            raise RuntimeError("frozen 4x decoder features were not captured")
        enriched = torch.cat((evidence,features), dim=1)
        raw = self.specialist(enriched)
        delta = gaussian_highpass(raw,sigma=1.25,radius=4)
        result=compose_albedo_detail(
            original,delta,albedo_residual_cap=self.backbone.config.albedo_residual_cap
        )
        result["fine_detail_raw"] = raw
        return result

    def architecture_contract(self) -> dict[str, object]:
        result = dict(super().architecture_contract())
        result.update({
            "revision":"V17.6-detail-10A1-feature-ablation",
            "detailEvidenceChannels":58,
            "evidenceMode":"hr_features58",
            "frozen4xDecoderFeatures":48,
            "featureTap":"backbone.decoder.detail_blocks",
            "additionalStemChannelsInitiallyZero":True,
        })
        return result


def create_matched_variants(backbone, seed: int = 17611):
    """Same backbone, common ten-channel parameters, zero extra feature weights.

    Returned models have independent copies of all backbone buffers and
    specialist parameters. Their initial 4x candidate outputs are identical.
    """
    torch.manual_seed(seed)
    plain = NSAMDRV176Detail(copy.deepcopy(backbone),detail_channels=32)
    torch.manual_seed(seed)
    rich = HRFeatureDetail(copy.deepcopy(backbone),detail_channels=32)
    with torch.no_grad():
        rich.specialist.stem.weight.zero_()
        rich.specialist.stem.weight[:,:10].copy_(plain.specialist.stem.weight)
        rich.specialist.stem.bias.copy_(plain.specialist.stem.bias)
        rich.specialist.blocks.load_state_dict(plain.specialist.blocks.state_dict())
        rich.specialist.head.load_state_dict(plain.specialist.head.state_dict())
    return plain,rich


def tensor_statistics(value: torch.Tensor) -> dict[str,float]:
    v=value.detach().float()
    return {
        "meanAbs":float(v.abs().mean()),
        "rms":float(v.square().mean().sqrt()),
        "maxAbs":float(v.abs().amax()),
        "aboveOneLsbFraction":float((v.abs() >= (1/255)).float().mean()),
    }


def gradient_statistics(model: NSAMDRV176Detail) -> dict[str, float]:
    norms = {}
    squared=0.0
    count=0
    for name,param in model.specialist.named_parameters():
        if param.grad is None:
            continue
        norm=float(param.grad.detach().float().norm())
        squared+=norm*norm
        count+=1
        if name in ("head.weight","head.bias","stem.weight","stem.bias"):
            norms[name]=norm
    if any(param.grad is not None for param in model.backbone.parameters()):
        raise RuntimeError("frozen Stage 9 backbone received gradients")
    return {"totalL2":math.sqrt(squared),"tensorsWithGrad":count,
            "selectedParameterGradL2":norms}


def diagnostic_batch(model: NSAMDRV176Detail, batch: dict) -> dict:
    """Training-only loss gradient probe; performs NO optimizer update."""
    model.train()
    model.zero_grad(set_to_none=True)
    outputs=model(batch["lr_albedo"],batch["lr_normal"],batch["lr_material"])
    terms=detail_supervision_loss(outputs,batch)
    terms["total"].backward()
    grads=gradient_statistics(model)
    target=batch["target_albedo"].detach().float().mean(dim=1,keepdim=True)
    b=outputs["baseline_albedo"].detach().float().mean(dim=1,keepdim=True)
    candidate=outputs["candidate_albedo"].detach().float().mean(dim=1,keepdim=True)
    target_bands=stationary_haar_details(target-b)
    predicted_bands=stationary_haar_details(candidate-b)
    ratios={}
    for key in DIAGNOSTIC_BANDS:
        target_energy=target_bands[key].abs().mean().clamp_min(0.001)
        ratios[key]=float(((predicted_bands[key]-target_bands[key]).abs().mean()/
                           target_energy))
    result={
        "lossTerms":{k:float(v.detach().float()) for k,v in terms.items()},
        "bandRelativeL1":ratios,
        "rawDetail":tensor_statistics(outputs["fine_detail_raw"]),
        "highpassDetail":tensor_statistics(outputs["fine_detail_highpass"]),
        "appliedAlbedoChange":tensor_statistics(
            outputs["candidate_albedo"]-outputs["base_candidate_albedo"]),
        "gradient":grads,
    }
    # Validate physical pass-through without constructing a second autograd
    # graph. A hook on the rich model must not retain a 48xHR feature tensor.
    with torch.no_grad():
        original=model.backbone(
            batch["lr_albedo"],batch["lr_normal"],batch["lr_material"]
        )
        result["normalUnchanged"]=bool(torch.equal(
            outputs["candidate_normal"],original["candidate_normal"]
        ))
        result["materialUnchanged"]=bool(torch.equal(
            outputs["candidate_material"],original["candidate_material"]
        ))
    if isinstance(model,HRFeatureDetail):
        model._feature=None
    model.zero_grad(set_to_none=True)
    return result
