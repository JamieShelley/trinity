from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .config import V14Config


class CapacityArtifactWriter:
    """Write V14.2 capacity telemetry images without affecting pass or fail."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir

    @staticmethod
    def _gray_u8(value: torch.Tensor) -> np.ndarray:
        image = (
            value.detach()
            .float()
            .clamp(0.0, 1.0)[0, 0]
            .cpu()
            .numpy()
        )
        return np.round(image * 255.0).astype(np.uint8)

    @staticmethod
    def _edge(value: torch.Tensor) -> torch.Tensor:
        gray = value.float().mean(dim=1, keepdim=True)
        dx = F.pad(
            (gray[..., :, 1:] - gray[..., :, :-1]).abs(),
            (0, 1, 0, 0),
        )
        dy = F.pad(
            (gray[..., 1:, :] - gray[..., :-1, :]).abs(),
            (0, 0, 0, 1),
        )
        edge = dx + dy
        maximum = edge.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)
        return edge / maximum

    @staticmethod
    def _detail_band(value: torch.Tensor, radius: int) -> torch.Tensor:
        kernel = radius * 2 + 1
        smooth = F.avg_pool2d(
            value.float(),
            kernel_size=kernel,
            stride=1,
            padding=radius,
        )
        detail = (value.float() - smooth).abs().mean(dim=1, keepdim=True)
        maximum = detail.amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8)
        return detail / maximum

    @staticmethod
    def _label(panel: np.ndarray, label: str) -> np.ndarray:
        result = panel.copy()
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(
            result,
            (0, 0),
            (result.shape[1], 38),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            result,
            label,
            (10, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return result

    def _write_panels(
        self,
        name: str,
        panels: list[tuple[str, np.ndarray]],
    ) -> Path:
        rendered = [
            self._label(panel, label)
            for label, panel in panels
        ]
        canvas = np.concatenate(rendered, axis=1)
        path = self.run_dir / name
        cv2.imwrite(str(path), canvas)
        return path

    def write_edge_comparison(
        self,
        target: torch.Tensor,
        baseline: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Path:
        return self._write_panels(
            "edge_comparison.png",
            [
                ("A EDGE", self._gray_u8(self._edge(target))),
                ("B EDGE", self._gray_u8(self._edge(baseline))),
                ("C EDGE", self._gray_u8(self._edge(candidate))),
            ],
        )

    def write_error_comparison(
        self,
        target: torch.Tensor,
        baseline: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Path:
        baseline_error = (
            (baseline.float() - target.float())
            .abs()
            .mean(dim=1, keepdim=True)
        )
        candidate_error = (
            (candidate.float() - target.float())
            .abs()
            .mean(dim=1, keepdim=True)
        )
        maximum = torch.maximum(
            baseline_error.amax(),
            candidate_error.amax(),
        ).clamp_min(1.0e-8)
        return self._write_panels(
            "error_comparison.png",
            [
                (
                    "|A-B|",
                    self._gray_u8(baseline_error / maximum),
                ),
                (
                    "|A-C|",
                    self._gray_u8(candidate_error / maximum),
                ),
            ],
        )

    def write_detail_band_comparison(
        self,
        target: torch.Tensor,
        baseline: torch.Tensor,
        candidate: torch.Tensor,
    ) -> Path:
        panels: list[tuple[str, np.ndarray]] = []
        for radius, label in ((1, "1PX"), (2, "2PX"), (4, "4PX")):
            target_band = self._detail_band(target, radius)
            baseline_band = self._detail_band(baseline, radius)
            candidate_band = self._detail_band(candidate, radius)
            baseline_error = (baseline_band - target_band).abs()
            candidate_error = (candidate_band - target_band).abs()
            maximum = torch.maximum(
                baseline_error.amax(),
                candidate_error.amax(),
            ).clamp_min(1.0e-8)
            panels.extend(
                [
                    (
                        f"B {label} ERROR",
                        self._gray_u8(baseline_error / maximum),
                    ),
                    (
                        f"C {label} ERROR",
                        self._gray_u8(candidate_error / maximum),
                    ),
                ]
            )
        return self._write_panels(
            "detail_band_comparison.png",
            panels,
        )

    def write_all(
        self,
        batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, str]:
        target = batch["target_albedo"]
        baseline = outputs["baseline_albedo"]
        candidate = outputs["candidate_albedo"]
        paths = {
            "edgeComparison": self.write_edge_comparison(
                target,
                baseline,
                candidate,
            ),
            "errorComparison": self.write_error_comparison(
                target,
                baseline,
                candidate,
            ),
            "detailBandComparison": self.write_detail_band_comparison(
                target,
                baseline,
                candidate,
            ),
        }
        return {
            key: str(path.resolve())
            for key, path in paths.items()
        }

    @staticmethod
    def residual_cap_saturation(
        outputs: dict[str, torch.Tensor],
        config: V14Config,
    ) -> dict[str, float]:
        thresholds = {
            "albedo": float(config.albedo_residual_cap) * 0.98,
            "normal": float(config.normal_residual_cap) * 0.98,
            "material": float(config.material_residual_cap) * 0.98,
        }
        tensors = {
            "albedo": outputs["candidate_residual_albedo"].float(),
            "normal": outputs["candidate_residual_normal"].float(),
            "material": outputs["candidate_residual_material"].float(),
        }
        return {
            key: float(
                (tensors[key].abs() >= threshold)
                .float()
                .mean()
                .item()
            )
            for key, threshold in thresholds.items()
        }
