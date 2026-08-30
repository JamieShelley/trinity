"""Checkpoint loading and overlapping 4x inference for NSAMDR V9."""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .config import V9Config
from .model import INPUT_CHANNELS, MODEL_SCHEMA, UPSCALE_FACTOR, FidelityResidualNetV9


class InferenceService:
    # Purpose: Implement cuda install hint for InferenceService.
    # Called by: resolve_device
    # Calls: No same-class helper methods.
    def cuda_install_hint(self) -> str:
        return "Run scripts\\build\\nsamdr.bat setup cuda; V9 fidelity reconstruction requires CUDA for production inference."

    # Purpose: Implement resolve device for InferenceService.
    # Called by: External callers and the owning workflow.
    # Calls: cuda_install_hint
    def resolve_device(self, config: V9Config, requested_device: str | None = None) -> torch.device:
        requested = (requested_device or config.device).strip().lower()
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is not available. {self.cuda_install_hint()}")
            device = torch.device(f"cuda:{config.cuda_device_index}")
            torch.cuda.set_device(device)
            torch.set_float32_matmul_precision(config.matmul_precision)
            return device
        return torch.device("cpu")

    # Purpose: Implement load trained model for InferenceService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def load_trained_model(
        self,
        checkpoint_path: Path,
        device: torch.device | str = "cpu",
    ) -> tuple[FidelityResidualNetV9, V9Config, dict[str, Any]]:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise RuntimeError("V9 checkpoint root is not a dictionary")
        if str(checkpoint.get("schema", "")) != MODEL_SCHEMA:
            raise RuntimeError(
                f"checkpoint schema {checkpoint.get('schema')!r} does not match {MODEL_SCHEMA!r}; older checkpoint schemas are incompatible")
        payload = checkpoint.get("config")
        if not isinstance(payload, dict):
            raise RuntimeError("V9 checkpoint config is missing")
        config = V9Config()
        for key, value in payload.items():
            if hasattr(config, key):
                if key in {"widths", "blocks_per_level", "decoder_blocks"}:
                    value = tuple(int(item) for item in value)
                setattr(config, key, value)
        config.validate()
        model = FidelityResidualNetV9(config).to(device)
        if torch.device(device).type == "cuda" and config.channels_last:
            converter = getattr(torch.nn.utils, "convert_conv2d_weight_memory_format", None)
            model = converter(model, torch.channels_last) if converter is not None else model.to(memory_format=torch.channels_last)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        selection_kind = str(checkpoint.get("selection_kind", ""))
        # Selection labels are provenance only. Every schema-compatible checkpoint
        # has one strict state topology and one production forward graph.
        model._checkpoint_selection_kind = selection_kind
        model.eval()
        return model, config, checkpoint

    # Purpose: Implement blend window for InferenceService.
    # Called by: infer_tiled
    # Calls: No same-class helper methods.
    def _blend_window(self, tile_size: int, overlap: int, device: torch.device) -> torch.Tensor:
        edge = max(1, min(overlap, tile_size // 2))
        axis = torch.ones(tile_size, device=device, dtype=torch.float32)
        ramp = torch.linspace(0.0, 1.0, edge + 2, device=device, dtype=torch.float32)[1:-1]
        axis[:edge] = torch.sin(ramp * math.pi * 0.5).square()
        axis[-edge:] = axis[:edge].flip(0)
        return torch.outer(axis, axis).view(1, 1, tile_size, tile_size).clamp_min(1.0e-4)

    # Purpose: Implement infer tiled for InferenceService.
    # Called by: External callers and the owning workflow.
    # Calls: _blend_window
    @torch.no_grad()
    def infer_tiled(
        self,
        model: FidelityResidualNetV9,
        model_input: np.ndarray | torch.Tensor,
        device: torch.device | str,
        tile_size: int = 128,
        overlap: int = 24,
        return_diagnostics: bool = False,
        return_all_maps: bool = False,
    ):
        if isinstance(model_input, np.ndarray):
            tensor = torch.from_numpy(model_input)
        else:
            tensor = model_input.detach().cpu()
        if tensor.ndim != 3 or tensor.shape[0] != INPUT_CHANNELS:
            raise ValueError(f"V9 model input must be {INPUT_CHANNELS}xHxW, got {tuple(tensor.shape)}")
        device = torch.device(device)
        tensor = tensor.to(device=device, dtype=torch.float32).unsqueeze(0)
        if device.type == "cuda" and model.config.channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        _, _, input_height, input_width = tensor.shape
        tile_size = max(64, min(int(tile_size), max(input_height, input_width)))
        tile_size -= tile_size % 16
        overlap = max(8, min(int(overlap), tile_size // 3))
        stride = max(16, tile_size - overlap * 2)
        pad_bottom = max(0, tile_size - input_height)
        pad_right = max(0, tile_size - input_width)
        if pad_bottom or pad_right:
            tensor = F.pad(tensor, (0, pad_right, 0, pad_bottom), mode="replicate")
        padded_height, padded_width = tensor.shape[-2:]
        ys = list(range(0, max(1, padded_height - tile_size + 1), stride))
        xs = list(range(0, max(1, padded_width - tile_size + 1), stride))
        if not ys or ys[-1] != padded_height - tile_size:
            ys.append(max(0, padded_height - tile_size))
        if not xs or xs[-1] != padded_width - tile_size:
            xs.append(max(0, padded_width - tile_size))

        output_tile = tile_size * UPSCALE_FACTOR
        output_height = padded_height * UPSCALE_FACTOR
        output_width = padded_width * UPSCALE_FACTOR
        # albedo3, normal2, roughness1, emissive1, material3, sdf1,
        # orientation2, confidence1, edge1, hardness1, boundaryGate1,
        # source-compatible SDF1, contourNormalOffsetPixels1 = 19 channels.
        accumulator = torch.zeros((1, 19, output_height, output_width), device=device, dtype=torch.float32)
        weights = torch.zeros((1, 1, output_height, output_width), device=device, dtype=torch.float32)
        window = self._blend_window(output_tile, overlap * UPSCALE_FACTOR, device)
        use_amp = device.type == "cuda"
        if use_amp and model.config.amp_dtype == "bf16":
            amp_dtype = torch.bfloat16
        elif use_amp and model.config.amp_dtype == "auto" and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
        if use_amp:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for y in ys:
            for x in xs:
                tile = tensor[:, :, y:y + tile_size, x:x + tile_size]
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    output = model(tile)
                packed = torch.cat((
                    output["albedo"], output["normal_xy"], output["roughness"], output["emissive"],
                    output["material"], output["sdf"], output["orientation"], output["confidence"],
                    torch.sigmoid(output["edge_logits"]), output["hardness"], output["boundary_gate"],
                    output["coarse_sdf"], output["contour_normal_offset_pixels"],
                ), dim=1).float()
                oy, ox = y * UPSCALE_FACTOR, x * UPSCALE_FACTOR
                accumulator[:, :, oy:oy + output_tile, ox:ox + output_tile] += packed * window
                weights[:, :, oy:oy + output_tile, ox:ox + output_tile] += window

        if use_amp:
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        packed = accumulator / weights.clamp_min(1.0e-5)
        packed = packed[:, :, :input_height * UPSCALE_FACTOR, :input_width * UPSCALE_FACTOR]
        packed_np = packed.squeeze(0).permute(1, 2, 0).cpu().numpy()
        maps = {
            "albedo": np.clip(packed_np[..., 0:3], 0.0, 1.0),
            "normal_xy": np.clip(packed_np[..., 3:5], -1.0, 1.0),
            "roughness": np.clip(packed_np[..., 5:6], 0.0, 1.0),
            "emissive": np.clip(packed_np[..., 6:7], 0.0, 1.0),
            "material": np.clip(packed_np[..., 7:10], 0.0, 1.0),
            "sdf": np.clip(packed_np[..., 10:11], -1.0, 1.0),
            "orientation": np.clip(packed_np[..., 11:13], -1.0, 1.0),
            "confidence": np.clip(packed_np[..., 13:14], 0.0, 1.0),
            "edge": np.clip(packed_np[..., 14:15], 0.0, 1.0),
            "hardness": np.clip(packed_np[..., 15:16], 0.0, 1.0),
            "boundary_gate": np.clip(packed_np[..., 16:17], 0.0, 1.0),
            "coarse_sdf": np.clip(packed_np[..., 17:18], -1.0, 1.0),
            "contour_normal_offset_pixels": packed_np[..., 18:19].astype(np.float32),
        }
        # Compatibility maps for old preview/audit consumers. V9.9.3 has no free
        # renderer-facing raster redistance. Geometry is queried continuously; the
        # scalar normal-offset map below remains compatibility telemetry only.
        maps["sdf_residual_pixels"] = np.zeros_like(maps["contour_normal_offset_pixels"], dtype=np.float32)
        maps["displacement"] = np.zeros(
            (maps["sdf"].shape[0], maps["sdf"].shape[1], 2), dtype=np.float32
        )
        maps["displacement_gate"] = maps["boundary_gate"]
        diagnostics = {
            "schema": MODEL_SCHEMA,
            "upscaleFactor": UPSCALE_FACTOR,
            "inputSize": [input_width, input_height],
            "outputSize": [input_width * UPSCALE_FACTOR, input_height * UPSCALE_FACTOR],
            "tileSize": tile_size,
            "overlap": overlap,
            "tileCount": len(xs) * len(ys),
            "inferenceMilliseconds": elapsed_ms,
            "device": str(device),
            "precision": ("bf16" if amp_dtype == torch.bfloat16 else "fp16") if use_amp else "fp32",
            "confidenceMean": float(maps["confidence"].mean()),
            "edgeMean": float(maps["edge"].mean()),
            "boundaryGateMean": float(maps["boundary_gate"].mean()),
            "boundaryGateP95": float(np.percentile(maps["boundary_gate"], 95)),
            "hardnessMean": float(maps["hardness"].mean()),
            "displacementRmsPixels": 0.0,
            "displacementMaxAbsPixels": 0.0,
            "appearanceEnabled": bool(model.config.appearance_enabled),
            "detailReconstructionEnabled": bool(getattr(model.config, "detail_reconstruction_enabled", True)),
            "checkpointSelectionKind": str(getattr(model, "_checkpoint_selection_kind", "")),
            "productionForward": "FidelityResidualNetV9.forward(inputs)",
            "externalGeometryAuthority": False,
            "sourceCompatibleSdfMeanAbs": float(np.mean(np.abs(maps["coarse_sdf"]))),
            "contourNormalOffsetRmsPixels": float(np.sqrt(np.mean(maps["contour_normal_offset_pixels"] ** 2))),
            "contourNormalOffsetMaxAbsPixels": float(np.max(np.abs(maps["contour_normal_offset_pixels"]))),
            "sdfResidualRmsPixels": 0.0,
            "reconstructionPrimitive": (
                "learned-parametric-analytic-sdf-plus-boundary-profile-phase-seam-detail-selector"
            ),
        }
        if not return_all_maps:
            maps = {key: maps[key] for key in ("albedo", "normal_xy", "material")}
        if return_diagnostics:
            return maps, diagnostics
        return maps

_inference_service = InferenceService()
cuda_install_hint = _inference_service.cuda_install_hint
resolve_device = _inference_service.resolve_device
load_trained_model = _inference_service.load_trained_model
_blend_window = _inference_service._blend_window
infer_tiled = _inference_service.infer_tiled
