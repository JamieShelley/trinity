"""Deterministic geometry auditor and pairwise learned geometry critic for NSAMDR V9.8.

The auditor is intentionally independent from the reconstruction loss. It measures
whether a candidate improves boundary geometry without buying that improvement
through blur, halos, double edges, topology damage, or off-edge repainting.

The optional critic is supplementary: it is trained on synthetic analytic
geometry and ranks two boundary patches. It never becomes an acceptance gate
unless explicitly requested and its synthetic validation accuracy is calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import html
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

import cv2
import numpy as np

AUDIT_SCHEMA = "NSAMDR_BOUNDARY_GEOMETRY_AUDIT_V2"
CRITIC_SCHEMA = "NSAMDR_GEOMETRY_PAIR_CRITIC_V1"


@dataclass(frozen=True)
class AuditOptions:
    evidence_regions: int = 12
    max_analysis_dimension: int = 1536
    patch_size: int = 112
    patch_stride: int = 72
    critic_mode: str = "auto"  # off | auto | required
    policy: str = "report"     # report | strict


class GeometryAuditService:
    # Purpose: Implement gray for GeometryAuditService.
    # Called by: _render_evidence, audit_pair
    # Calls: No same-class helper methods.
    def _gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            result = image.astype(np.float32)
        else:
            bgr = image[..., :3]
            result = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if result.max(initial=0.0) > 1.5:
            result /= 255.0
        return np.clip(result, 0.0, 1.0)

    # Purpose: Implement resize for analysis for GeometryAuditService.
    # Called by: audit_pair
    # Calls: No same-class helper methods.
    def _resize_for_analysis(self, image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        scale = min(1.0, float(max_dim) / max(h, w))
        if scale >= 0.999:
            return image, 1.0
        resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        return resized, scale

    # Purpose: Implement gradient for GeometryAuditService.
    # Called by: _edge_mask, _general_edge_width_proxy, audit_pair
    # Calls: No same-class helper methods.
    def _gradient(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(gx, gy)
        return gx, gy, mag

    # Purpose: Implement edge mask for GeometryAuditService.
    # Called by: audit_pair
    # Calls: _gradient
    def _edge_mask(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        _gx, _gy, mag = self._gradient(gray)
        nonzero = mag[mag > 1e-5]
        if nonzero.size == 0:
            return np.zeros_like(gray, dtype=np.uint8), mag
        threshold = max(float(np.percentile(nonzero, 82.0)), float(nonzero.mean() * 1.25), 0.015)
        mask = (mag >= threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return mask, mag

    # Purpose: Implement dilate for GeometryAuditService.
    # Called by: _boundary_gate_metrics, _flow_metrics, _general_edge_width_proxy, _ringing_proxy, audit_pair
    # Calls: No same-class helper methods.
    def _dilate(self, mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return mask.astype(bool)
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    # Purpose: Implement fit straightness for GeometryAuditService.
    # Called by: _patch_metrics
    # Calls: No same-class helper methods.
    def _fit_straightness(self, edge: np.ndarray, mag: np.ndarray) -> dict[str, float] | None:
        ys, xs = np.nonzero(edge)
        if len(xs) < 24:
            return None
        points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        centre = points.mean(axis=0)
        centred = points - centre
        cov = np.cov(centred, rowvar=False)
        if not np.all(np.isfinite(cov)):
            return None
        values, vectors = np.linalg.eigh(cov)
        order = np.argsort(values)[::-1]
        major = float(max(values[order[0]], 1e-8))
        minor = float(max(values[order[1]], 1e-8))
        anisotropy = major / minor
        if anisotropy < 3.2:
            return None
        tangent = vectors[:, order[0]].astype(np.float32)
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        dist = centred @ normal
        weights = mag[ys, xs].astype(np.float32) + 1e-4
        rms = float(np.sqrt(np.average(dist * dist, weights=weights)))
        # Robust width of gradient support around the fitted centre line.
        width = float(np.sqrt(np.average((dist - np.average(dist, weights=weights)) ** 2, weights=weights)))
        return {
            "straightnessRms": rms,
            "edgeWidthProxy": width,
            "anisotropy": anisotropy,
            "tangentX": float(tangent[0]),
            "tangentY": float(tangent[1]),
        }

    # Purpose: Implement curvature noise for GeometryAuditService.
    # Called by: _patch_metrics
    # Calls: No same-class helper methods.
    def _curvature_noise(self, edge: np.ndarray) -> float | None:
        contours, _ = cv2.findContours((edge * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = [c.reshape(-1, 2).astype(np.float32) for c in contours if len(c) >= 24]
        if not contours:
            return None
        contour = max(contours, key=len)
        step = max(1, len(contour) // 96)
        p = contour[::step]
        if len(p) < 12:
            return None
        d = np.diff(p, axis=0)
        length = np.linalg.norm(d, axis=1)
        valid = length > 0.5
        d = d[valid]
        if len(d) < 10:
            return None
        angle = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        turn = np.diff(angle)
        if len(turn) < 5:
            return None
        # Median absolute second difference penalises alternating pixel staircases
        # while being tolerant of genuine smooth curvature.
        second = np.diff(turn)
        return float(np.median(np.abs(second)))

    # Purpose: Implement ringing proxy for GeometryAuditService.
    # Called by: _patch_metrics, audit_pair
    # Calls: _dilate
    def _ringing_proxy(self, gray: np.ndarray, edge: np.ndarray) -> float:
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        near = self._dilate(edge, 6)
        core = self._dilate(edge, 1)
        band = near & ~core
        if not np.any(band):
            return 0.0
        return float(np.mean(lap[band]))

    # Purpose: Implement topology components for GeometryAuditService.
    # Called by: _patch_metrics, audit_pair
    # Calls: No same-class helper methods.
    def _topology_components(self, edge: np.ndarray) -> int:
        count, _labels = cv2.connectedComponents(edge.astype(np.uint8), connectivity=8)
        return max(0, int(count) - 1)

    # Purpose: Implement symmetric edge chamfer for GeometryAuditService.
    # Called by: audit_pair
    # Calls: No same-class helper methods.
    def _symmetric_edge_chamfer(self, edge_a: np.ndarray, edge_b: np.ndarray) -> float:
        if not np.any(edge_a) or not np.any(edge_b):
            return float("inf")
        # distanceTransform gives distance to zeros, so invert edge maps.
        da = cv2.distanceTransform((1 - edge_a.astype(np.uint8)), cv2.DIST_L2, 3)
        db = cv2.distanceTransform((1 - edge_b.astype(np.uint8)), cv2.DIST_L2, 3)
        return 0.5 * (float(np.mean(db[edge_a.astype(bool)])) + float(np.mean(da[edge_b.astype(bool)])))

    # Purpose: Implement candidate patches for GeometryAuditService.
    # Called by: audit_pair
    # Calls: No same-class helper methods.
    def _candidate_patches(self, edge: np.ndarray, mag: np.ndarray, patch: int, stride: int, limit: int = 48) -> list[tuple[int, int, int, int, float]]:
        h, w = edge.shape
        patch = max(48, min(int(patch), h, w))
        stride = max(24, min(int(stride), patch))
        regions: list[tuple[int, int, int, int, float]] = []
        for y in range(0, max(1, h - patch + 1), stride):
            for x in range(0, max(1, w - patch + 1), stride):
                e = edge[y:y + patch, x:x + patch]
                density = float(e.mean())
                if density < 0.012 or density > 0.42:
                    continue
                score = float(np.mean(mag[y:y + patch, x:x + patch]) * (0.5 + density))
                regions.append((x, y, patch, patch, score))
        regions.sort(key=lambda r: r[4], reverse=True)
        # Suppress strongly overlapping patches.
        selected: list[tuple[int, int, int, int, float]] = []
        for region in regions:
            x, y, pw, ph, _ = region
            keep = True
            for sx, sy, sw, sh, _ in selected:
                ix = max(0, min(x + pw, sx + sw) - max(x, sx))
                iy = max(0, min(y + ph, sy + sh) - max(y, sy))
                if (ix * iy) / float(pw * ph) > 0.45:
                    keep = False
                    break
            if keep:
                selected.append(region)
            if len(selected) >= limit:
                break
        return selected

    # Purpose: Implement metric delta for GeometryAuditService.
    # Called by: audit_pair
    # Calls: No same-class helper methods.
    def _metric_delta(self, before: float | None, after: float | None) -> float | None:
        if before is None or after is None or not math.isfinite(before) or not math.isfinite(after):
            return None
        return float(after - before)

    # Purpose: Implement relative improvement for GeometryAuditService.
    # Called by: audit_pair
    # Calls: No same-class helper methods.
    def _relative_improvement(self, before: float | None, after: float | None) -> float | None:
        if before is None or after is None or not math.isfinite(before) or not math.isfinite(after):
            return None
        denom = max(abs(before), 1e-5)
        return float((before - after) / denom)

    # Purpose: Implement flow metrics for GeometryAuditService.
    # Called by: audit_pair
    # Calls: _dilate
    def _flow_metrics(self, flow: np.ndarray | None, edge: np.ndarray, gx: np.ndarray, gy: np.ndarray, scale: float) -> dict[str, Any]:
        if flow is None:
            return {"available": False}
        f = np.asarray(flow, dtype=np.float32)
        if scale != 1.0:
            f = cv2.resize(f, (edge.shape[1], edge.shape[0]), interpolation=cv2.INTER_AREA) * scale
        elif f.shape[:2] != edge.shape:
            sy = edge.shape[0] / max(1, f.shape[0])
            sx = edge.shape[1] / max(1, f.shape[1])
            f = cv2.resize(f, (edge.shape[1], edge.shape[0]), interpolation=cv2.INTER_AREA)
            f[..., 0] *= sx
            f[..., 1] *= sy
        magnitude = np.sqrt(np.sum(f * f, axis=-1))
        band = self._dilate(edge, 4)
        outside = ~self._dilate(edge, 8)
        active = magnitude > 0.05 * max(scale, 0.25)
        grad_len = np.sqrt(gx * gx + gy * gy) + 1e-6
        nx, ny = gx / grad_len, gy / grad_len
        tx, ty = -ny, nx
        normal_motion = np.abs(f[..., 0] * nx + f[..., 1] * ny)
        tangent_motion = np.abs(f[..., 0] * tx + f[..., 1] * ty)
        edge_motion = magnitude[band] if np.any(band) else magnitude.reshape(-1)
        off_motion = magnitude[outside] if np.any(outside) else np.zeros((1,), np.float32)
        tangent_fraction = float(np.mean(tangent_motion[band]) / max(np.mean(magnitude[band]), 1e-6)) if np.any(band) else 0.0
        normal_fraction = float(np.mean(normal_motion[band]) / max(np.mean(magnitude[band]), 1e-6)) if np.any(band) else 0.0
        return {
            "available": True,
            "rmsPixels": float(np.sqrt(np.mean(magnitude * magnitude))),
            "maxPixels": float(np.max(magnitude)),
            "activeFraction": float(np.mean(active)),
            "edgeBandRmsPixels": float(np.sqrt(np.mean(edge_motion * edge_motion))) if edge_motion.size else 0.0,
            "offEdgeRmsPixels": float(np.sqrt(np.mean(off_motion * off_motion))) if off_motion.size else 0.0,
            "activeOnEdgeFraction": float(np.mean(active & band) / max(np.mean(active), 1e-8)) if np.any(active) else 0.0,
            "normalMotionFraction": normal_fraction,
            "tangentMotionFraction": tangent_fraction,
        }

    # Purpose: Implement boundary gate metrics for GeometryAuditService.
    # Called by: audit_pair
    # Calls: _dilate
    def _boundary_gate_metrics(
        self,
        gate: np.ndarray | None,
        edge: np.ndarray,
        scale: float,
    ) -> dict[str, Any]:
        if gate is None:
            return {"available": False}
        g = np.asarray(gate, dtype=np.float32)
        if g.ndim == 3:
            g = g[..., 0]
        if g.shape[:2] != edge.shape:
            g = cv2.resize(
                g,
                (edge.shape[1], edge.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        g = np.clip(g, 0.0, 1.0)
        edge_band = self._dilate(edge, 5)
        off_edge = ~self._dilate(edge, 10)
        edge_values = g[edge_band] if np.any(edge_band) else g.reshape(-1)
        off_values = g[off_edge] if np.any(off_edge) else np.zeros((1,), np.float32)
        active = g > 0.20
        return {
            "available": True,
            "mean": float(np.mean(g)),
            "p95": float(np.percentile(g, 95)),
            "edgeMean": float(np.mean(edge_values)) if edge_values.size else 0.0,
            "edgeP95": float(np.percentile(edge_values, 95)) if edge_values.size else 0.0,
            "offEdgeMean": float(np.mean(off_values)) if off_values.size else 0.0,
            "activeFraction": float(np.mean(active)),
            "activeOnEdgeFraction": (
                float(np.mean(active & edge_band) / max(np.mean(active), 1.0e-8))
                if np.any(active) else 0.0
            ),
        }

    # Purpose: Implement general edge width proxy for GeometryAuditService.
    # Called by: _patch_metrics
    # Calls: _dilate, _gradient
    def _general_edge_width_proxy(self, gray: np.ndarray, edge: np.ndarray) -> float | None:
        """Shape-agnostic fuzz proxy from gradient energy leaking away from edge cores."""
        if gray.size == 0 or not np.any(edge):
            return None
        _gx, _gy, mag = self._gradient(gray)
        core = self._dilate(edge, 1)
        broad = self._dilate(edge, 5)
        outer = broad & ~core
        total = float(np.sum(mag[broad])) if np.any(broad) else 0.0
        if total <= 1.0e-8:
            return None
        outer_fraction = float(np.sum(mag[outer]) / total) if np.any(outer) else 0.0
        # Map the leakage fraction into an intuitive width-like quantity. Only
        # relative before/after comparisons are used by the audit.
        return 1.0 + 6.0 * max(0.0, min(1.0, outer_fraction))

    # Purpose: Implement patch metrics for GeometryAuditService.
    # Called by: audit_pair
    # Calls: _curvature_noise, _fit_straightness, _general_edge_width_proxy, _ringing_proxy, _topology_components
    def _patch_metrics(self, gray: np.ndarray, edge: np.ndarray, mag: np.ndarray, region: tuple[int, int, int, int, float]) -> dict[str, Any]:
        x, y, w, h, energy = region
        g = gray[y:y + h, x:x + w]
        e = edge[y:y + h, x:x + w]
        m = mag[y:y + h, x:x + w]
        straight = self._fit_straightness(e, m)
        general_width = self._general_edge_width_proxy(g, e)
        straight_width = None if straight is None else straight["edgeWidthProxy"]
        edge_width = straight_width if straight_width is not None else general_width
        return {
            "x": x, "y": y, "width": w, "height": h, "edgeEnergy": energy,
            "edgeDensity": float(e.mean()),
            "straightnessRms": None if straight is None else straight["straightnessRms"],
            "edgeWidthProxy": edge_width,
            "generalFuzzProxy": general_width,
            "anisotropy": None if straight is None else straight["anisotropy"],
            "curvatureNoise": self._curvature_noise(e),
            "ringingProxy": self._ringing_proxy(g, e),
            "componentCount": self._topology_components(e),
        }

    # Purpose: Implement critic classes for GeometryAuditService.
    # Called by: _critic_scores, ensure_geometry_critic
    # Calls: No same-class helper methods.
    def _critic_classes(self):
        import torch
        from torch import nn

        class GeometryCritic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(2, 24, 5, stride=2, padding=2), nn.GELU(),
                    nn.Conv2d(24, 40, 3, stride=2, padding=1), nn.GELU(),
                    nn.Conv2d(40, 64, 3, stride=2, padding=1), nn.GELU(),
                    nn.Conv2d(64, 80, 3, stride=2, padding=1), nn.GELU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.head = nn.Linear(80, 1)

            def forward(self, gray):
                gx = gray[..., :, 1:] - gray[..., :, :-1]
                gy = gray[..., 1:, :] - gray[..., :-1, :]
                gx = torch.nn.functional.pad(gx, (0, 1, 0, 0))
                gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))
                grad = torch.sqrt(gx * gx + gy * gy + 1e-8)
                x = torch.cat((gray, grad), dim=1)
                return self.head(self.features(x).flatten(1)).squeeze(1)

        return GeometryCritic

    # Purpose: Implement synthetic critic batch for GeometryAuditService.
    # Called by: ensure_geometry_critic
    # Calls: No same-class helper methods.
    def _synthetic_critic_batch(self, batch: int, size: int, device, generator) -> tuple[Any, Any, Any]:
        import torch
        import torch.nn.functional as F

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, size, device=device),
            torch.linspace(-1.0, 1.0, size, device=device), indexing="ij")
        xx = xx[None].expand(batch, -1, -1)
        yy = yy[None].expand(batch, -1, -1)
        shape_type = torch.randint(0, 3, (batch,), device=device, generator=generator)
        theta = torch.rand(batch, device=device, generator=generator) * math.pi
        offset = (torch.rand(batch, device=device, generator=generator) - 0.5) * 0.8
        width = 0.025 + torch.rand(batch, device=device, generator=generator) * 0.08
        line_dist = torch.abs(xx * torch.cos(theta)[:, None, None] + yy * torch.sin(theta)[:, None, None] - offset[:, None, None])
        cx = (torch.rand(batch, device=device, generator=generator) - 0.5) * 0.6
        cy = (torch.rand(batch, device=device, generator=generator) - 0.5) * 0.6
        radius = 0.25 + torch.rand(batch, device=device, generator=generator) * 0.55
        circle_dist = torch.abs(torch.sqrt((xx-cx[:,None,None])**2 + (yy-cy[:,None,None])**2 + 1e-8) - radius[:,None,None])
        ax = 0.25 + torch.rand(batch, device=device, generator=generator) * 0.65
        ay = 0.18 + torch.rand(batch, device=device, generator=generator) * 0.55
        ellipse_dist = torch.abs(torch.sqrt(((xx-cx[:,None,None])/ax[:,None,None])**2 + ((yy-cy[:,None,None])/ay[:,None,None])**2 + 1e-8) - 1.0) * torch.minimum(ax, ay)[:,None,None]
        dist = torch.where(shape_type[:,None,None] == 0, line_dist, torch.where(shape_type[:,None,None] == 1, circle_dist, ellipse_dist))
        clean = torch.sigmoid((width[:,None,None] - dist) * (90.0 + torch.rand(batch, device=device, generator=generator)[:,None,None]*80.0))
        bg = 0.1 + torch.rand(batch, device=device, generator=generator)[:,None,None] * 0.35
        fg = 0.6 + torch.rand(batch, device=device, generator=generator)[:,None,None] * 0.4
        clean = (bg + (fg-bg) * clean).unsqueeze(1)

        severity_a = 0.15 + torch.rand(batch, device=device, generator=generator) * 1.05
        severity_b = 0.15 + torch.rand(batch, device=device, generator=generator) * 1.05

        def degrade(severity):
            # Low-resolution staircase, blur, and bounded ringing scale with severity.
            low_side = torch.clamp((size / (1.6 + severity * 2.8)).round().long(), min=12, max=size)
            outputs = []
            for i in range(batch):
                side = int(low_side[i].item())
                low = F.interpolate(clean[i:i+1], size=(side, side), mode="area")
                up = F.interpolate(low, size=(size, size), mode="bicubic", align_corners=False)
                blur = F.avg_pool2d(F.pad(up, (1,1,1,1), mode="replicate"), 3, 1)
                ring = (up - blur) * (0.12 + 0.34 * severity[i])
                outputs.append((up + ring).clamp(0,1))
            return torch.cat(outputs, dim=0)

        a = degrade(severity_a)
        b = degrade(severity_b)
        label = (severity_b < severity_a).float()  # 1 means B should score higher.
        return a, b, label

    # Purpose: Implement ensure geometry critic for GeometryAuditService.
    # Called by: audit_pair
    # Calls: _critic_classes, _synthetic_critic_batch
    def ensure_geometry_critic(self, checkpoint: Path, *, device: str = "cpu", steps: int = 180, seed: int = 98173) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        checkpoint = checkpoint.resolve()
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        GeometryCritic = self._critic_classes()
        if checkpoint.is_file():
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(checkpoint, map_location="cpu")
            if isinstance(payload, dict) and payload.get("schema") == CRITIC_SCHEMA:
                return {
                    "checkpoint": str(checkpoint),
                    "validationAccuracy": float(payload.get("validation_accuracy", 0.0)),
                    "trainingSteps": int(payload.get("training_steps", 0)),
                    "calibrated": float(payload.get("validation_accuracy", 0.0)) >= 0.75,
                    "reused": True,
                }

        dev = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        model = GeometryCritic().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
        gen = torch.Generator(device=dev).manual_seed(seed)
        model.train()
        for _ in range(max(40, int(steps))):
            a, b, label = self._synthetic_critic_batch(32, 64, dev, gen)
            score = model(b) - model(a)
            loss = F.binary_cross_entropy_with_logits(score, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for _ in range(20):
                a, b, label = self._synthetic_critic_batch(32, 64, dev, gen)
                pred = ((model(b) - model(a)) > 0).float()
                correct += int((pred == label).sum().item())
                total += int(label.numel())
        accuracy = correct / max(total, 1)
        payload = {
            "schema": CRITIC_SCHEMA,
            "state_dict": model.state_dict(),
            "validation_accuracy": accuracy,
            "training_steps": max(40, int(steps)),
            "seed": seed,
        }
        torch.save(payload, checkpoint)
        return {
            "checkpoint": str(checkpoint),
            "validationAccuracy": accuracy,
            "trainingSteps": max(40, int(steps)),
            "calibrated": accuracy >= 0.75,
            "reused": False,
        }

    # Purpose: Implement critic scores for GeometryAuditService.
    # Called by: audit_pair
    # Calls: _critic_classes
    def _critic_scores(self, checkpoint: Path, baseline: np.ndarray, candidate: np.ndarray, regions: Iterable[tuple[int,int,int,int,float]], device: str) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F
        GeometryCritic = self._critic_classes()
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        model = GeometryCritic()
        model.load_state_dict(payload["state_dict"], strict=True)
        dev = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
        model.to(dev).eval()
        ba, ca = [], []
        for x,y,w,h,_ in regions:
            bp = baseline[y:y+h, x:x+w]
            cp = candidate[y:y+h, x:x+w]
            if bp.size == 0 or cp.size == 0:
                continue
            bp = cv2.resize(bp, (64,64), interpolation=cv2.INTER_AREA)
            cp = cv2.resize(cp, (64,64), interpolation=cv2.INTER_AREA)
            ba.append(bp); ca.append(cp)
        if not ba:
            return {"available": False}
        bt = torch.from_numpy(np.stack(ba)[:,None]).float().to(dev)
        ct = torch.from_numpy(np.stack(ca)[:,None]).float().to(dev)
        with torch.no_grad():
            sb = model(bt)
            sc = model(ct)
            delta = sc - sb
            win = torch.sigmoid(delta)
        return {
            "available": True,
            "baselineScoreMean": float(sb.mean().cpu()),
            "candidateScoreMean": float(sc.mean().cpu()),
            "scoreDeltaMean": float(delta.mean().cpu()),
            "candidateWinProbabilityMean": float(win.mean().cpu()),
            "candidateWinFraction": float((delta > 0).float().mean().cpu()),
            "patchCount": int(len(ba)),
        }

    # Purpose: Implement render evidence for GeometryAuditService.
    # Called by: audit_pair
    # Calls: _gray
    def _render_evidence(self, path: Path, baseline: np.ndarray, candidate: np.ndarray, b_edge: np.ndarray, c_edge: np.ndarray,
                         region: tuple[int,int,int,int,float], flow: np.ndarray | None, gate: np.ndarray | None,
                         title: str, metrics: dict[str, Any]) -> None:
        x,y,w,h,_ = region
        def crop_rgb(img):
            c = img[y:y+h, x:x+w]
            if c.ndim == 2:
                c = cv2.cvtColor(np.uint8(np.clip(c * 255, 0, 255)), cv2.COLOR_GRAY2BGR)
            elif c.ndim == 3 and c.shape[2] == 1:
                c = cv2.cvtColor(c[:, :, 0], cv2.COLOR_GRAY2BGR)
            elif c.ndim == 3 and c.shape[2] == 4:
                # Candidate textures are BGRA internally. Evidence canvases are BGR.
                # Keeping alpha here caused a 4-channel -> 3-channel broadcast failure
                # after resize (for example 196x260x4 into 196x260x3).
                c = cv2.cvtColor(c, cv2.COLOR_BGRA2BGR)
            elif c.ndim == 3 and c.shape[2] > 3:
                c = c[:, :, :3]
            return np.ascontiguousarray(c.copy())
        b = crop_rgb(baseline)
        c = crop_rgb(candidate)
        be = cv2.cvtColor((b_edge[y:y+h,x:x+w]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ce = cv2.cvtColor((c_edge[y:y+h,x:x+w]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        bg = self._gray(b); cg = self._gray(c)
        diff = np.abs(cg-bg)
        diff = cv2.applyColorMap(np.uint8(np.clip(diff / max(float(np.percentile(diff,99)),1e-4)*255,0,255)), cv2.COLORMAP_TURBO)
        if gate is not None:
            g = np.asarray(gate, dtype=np.float32)
            if g.ndim == 3:
                g = g[..., 0]
            if g.shape[:2] != baseline.shape[:2]:
                g = cv2.resize(g, (baseline.shape[1], baseline.shape[0]), interpolation=cv2.INTER_AREA)
            gm = np.clip(g[y:y+h, x:x+w], 0.0, 1.0)
            action_img = cv2.applyColorMap(np.uint8(np.round(gm * 255.0)), cv2.COLORMAP_TURBO)
            action_label = "boundary gate"
        elif flow is not None:
            f = flow
            if f.shape[:2] != baseline.shape[:2]:
                f = cv2.resize(f, (baseline.shape[1], baseline.shape[0]), interpolation=cv2.INTER_AREA)
            fm = np.sqrt(np.sum(f[y:y+h,x:x+w]**2, axis=-1))
            action_img = cv2.applyColorMap(np.uint8(np.clip(fm/max(float(np.percentile(fm,99)),0.05)*255,0,255)), cv2.COLORMAP_TURBO)
            action_label = "flow"
        else:
            action_img = np.zeros_like(b)
            action_label = "boundary gate"
        panels = [("baseline",b),("candidate",c),("baseline edge",be),("candidate edge",ce),("abs delta",diff),(action_label,action_img)]
        cell_w, cell_h = 260, 220
        canvas = np.zeros((cell_h*2+70, cell_w*3, 3), np.uint8)
        for i,(label,img) in enumerate(panels):
            img = cv2.resize(img, (cell_w, cell_h-24), interpolation=cv2.INTER_NEAREST if "edge" in label else cv2.INTER_AREA)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.ndim == 3 and img.shape[2] > 3:
                img = img[:, :, :3]
            row,col = divmod(i,3)
            oy,ox = row*cell_h,col*cell_w
            canvas[oy+24:oy+cell_h,ox:ox+cell_w] = img
            cv2.putText(canvas,label,(ox+6,oy+17),cv2.FONT_HERSHEY_SIMPLEX,0.48,(235,235,235),1,cv2.LINE_AA)
        cv2.putText(canvas,title,(8,cell_h*2+24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1,cv2.LINE_AA)
        summary = f"straight {metrics.get('straightImprovement')}  curvature {metrics.get('curvatureImprovement')}  score {metrics.get('proxyScore')}"
        cv2.putText(canvas,summary[:120],(8,cell_h*2+50),cv2.FONT_HERSHEY_SIMPLEX,0.46,(220,220,220),1,cv2.LINE_AA)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 4])

    # Purpose: Implement audit pair for GeometryAuditService.
    # Called by: External callers and the owning workflow.
    # Calls: _boundary_gate_metrics, _candidate_patches, _critic_scores, _dilate, _edge_mask, _flow_metrics, _gradient, _gray, _metric_delta, _patch_metrics, _relative_improvement, _render_evidence, _resize_for_analysis, _ringing_proxy, _symmetric_edge_chamfer, _topology_components, ensure_geometry_critic
    def audit_pair(
        self,
        baseline_image: np.ndarray,
        candidate_image: np.ndarray,
        *,
        source_name: str,
        output_dir: Path,
        options: AuditOptions = AuditOptions(),
        flow: np.ndarray | None = None,
        gate: np.ndarray | None = None,
        target_image: np.ndarray | None = None,
        critic_checkpoint: Path | None = None,
        critic_device: str = "cpu",
    ) -> dict[str, Any]:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        baseline0, scale = self._resize_for_analysis(baseline_image, options.max_analysis_dimension)
        candidate0 = cv2.resize(candidate_image, (baseline0.shape[1], baseline0.shape[0]), interpolation=cv2.INTER_AREA) if candidate_image.shape[:2] != baseline0.shape[:2] else candidate_image
        bg = self._gray(baseline0); cg = self._gray(candidate0)
        be, bmag = self._edge_mask(bg); ce, cmag = self._edge_mask(cg)
        bgx,bgy,_ = self._gradient(bg)
        regions = self._candidate_patches(be, bmag, options.patch_size, options.patch_stride)
        rows: list[dict[str, Any]] = []
        for index, region in enumerate(regions):
            bm = self._patch_metrics(bg, be, bmag, region)
            cm = self._patch_metrics(cg, ce, cmag, region)
            straight_imp = self._relative_improvement(bm["straightnessRms"], cm["straightnessRms"])
            curvature_imp = self._relative_improvement(bm["curvatureNoise"], cm["curvatureNoise"])
            width_delta = self._metric_delta(bm["edgeWidthProxy"], cm["edgeWidthProxy"])
            ring_imp = self._relative_improvement(bm["ringingProxy"], cm["ringingProxy"])
            terms = [v for v in (straight_imp, curvature_imp, ring_imp) if v is not None]
            proxy = float(np.mean(terms)) if terms else 0.0
            if width_delta is not None and bm["edgeWidthProxy"] not in (None,0):
                width_ratio = float(cm["edgeWidthProxy"] / max(bm["edgeWidthProxy"],1e-5))
                if width_ratio > 1.15:
                    proxy -= min(0.5, width_ratio - 1.15)
            else:
                width_ratio = None
            topology_change = abs(int(cm["componentCount"]) - int(bm["componentCount"]))
            proxy -= min(0.3, topology_change * 0.04)
            rows.append({
                "region": index,
                "x": region[0], "y": region[1], "width": region[2], "height": region[3],
                "straightBefore": bm["straightnessRms"], "straightAfter": cm["straightnessRms"],
                "straightImprovement": straight_imp,
                "curvatureBefore": bm["curvatureNoise"], "curvatureAfter": cm["curvatureNoise"],
                "curvatureImprovement": curvature_imp,
                "edgeWidthBefore": bm["edgeWidthProxy"], "edgeWidthAfter": cm["edgeWidthProxy"],
                "edgeWidthRatio": width_ratio,
                "generalFuzzBefore": bm["generalFuzzProxy"], "generalFuzzAfter": cm["generalFuzzProxy"],
                "ringingBefore": bm["ringingProxy"], "ringingAfter": cm["ringingProxy"],
                "ringingImprovement": ring_imp,
                "componentBefore": bm["componentCount"], "componentAfter": cm["componentCount"],
                "topologyComponentDelta": topology_change,
                "proxyScore": proxy,
            })

        band = self._dilate(be, 8)
        off = ~band
        delta = np.abs(cg - bg)
        off_identity_rms = float(np.sqrt(np.mean(delta[off] ** 2)) * 255.0) if np.any(off) else 0.0
        edge_delta_rms = float(np.sqrt(np.mean(delta[band] ** 2)) * 255.0) if np.any(band) else 0.0
        topology_before = self._topology_components(be)
        topology_after = self._topology_components(ce)
        ringing_before = self._ringing_proxy(bg, be)
        ringing_after = self._ringing_proxy(cg, ce)
        flow_report = self._flow_metrics(flow, be, bgx, bgy, scale)
        boundary_report = self._boundary_gate_metrics(gate, be, scale)

        valid_straight = [r for r in rows if r["straightImprovement"] is not None]
        valid_curve = [r for r in rows if r["curvatureImprovement"] is not None]
        straight_mean = float(np.mean([r["straightImprovement"] for r in valid_straight])) if valid_straight else None
        straight_win = float(np.mean([r["straightImprovement"] > 0 for r in valid_straight])) if valid_straight else None
        curve_mean = float(np.mean([r["curvatureImprovement"] for r in valid_curve])) if valid_curve else None
        curve_win = float(np.mean([r["curvatureImprovement"] > 0 for r in valid_curve])) if valid_curve else None
        valid_width = [r for r in rows if r["edgeWidthRatio"] is not None and math.isfinite(float(r["edgeWidthRatio"]))]
        width_ratio_mean = float(np.mean([r["edgeWidthRatio"] for r in valid_width])) if valid_width else None
        fuzz_reduction_mean = float(np.mean([1.0 - r["edgeWidthRatio"] for r in valid_width])) if valid_width else None
        width_win = float(np.mean([r["edgeWidthRatio"] < 0.98 for r in valid_width])) if valid_width else None
        width_regression = float(np.mean([r["edgeWidthRatio"] > 1.10 for r in valid_width])) if valid_width else None
        proxy_mean = float(np.mean([r["proxyScore"] for r in rows])) if rows else 0.0

        exact: dict[str, Any] = {"available": False}
        if target_image is not None:
            target0 = cv2.resize(target_image, (baseline0.shape[1], baseline0.shape[0]), interpolation=cv2.INTER_AREA) if target_image.shape[:2] != baseline0.shape[:2] else target_image
            tg = self._gray(target0); te,_ = self._edge_mask(tg)
            cb = self._symmetric_edge_chamfer(be, te); cc = self._symmetric_edge_chamfer(ce, te)
            mae_b = float(np.mean(np.abs(bg-tg))); mae_c = float(np.mean(np.abs(cg-tg)))
            exact = {
                "available": True,
                "edgeChamferBefore": cb,
                "edgeChamferAfter": cc,
                "edgeChamferImprovement": self._relative_improvement(cb, cc),
                "lumaMaeBefore": mae_b,
                "lumaMaeAfter": mae_c,
                "lumaMaeImprovement": self._relative_improvement(mae_b, mae_c),
            }

        critic: dict[str, Any] = {"enabled": options.critic_mode != "off", "available": False}
        if options.critic_mode != "off" and critic_checkpoint is not None:
            calibration = self.ensure_geometry_critic(critic_checkpoint, device=critic_device)
            critic.update(calibration)
            scores = self._critic_scores(critic_checkpoint, bg, cg, regions[:24], critic_device)
            critic.update(scores)
            critic["status"] = "calibrated" if calibration["calibrated"] else "calibration-low-confidence"

        # Hard safety tests first; improvement tests only classify PASS vs NEUTRAL.
        reasons: list[str] = []
        fail = False
        if off_identity_rms > 1.25:
            fail = True; reasons.append(f"off-edge identity RMS {off_identity_rms:.3f} > 1.25 levels")
        if ringing_before > 1e-6 and ringing_after / ringing_before > 1.22:
            fail = True; reasons.append("ringing proxy increased >22%")
        if rows and not exact.get("available"):
            width_regressions = [r for r in rows if r["edgeWidthRatio"] is not None and r["edgeWidthRatio"] > 1.25]
            if len(width_regressions) / len(rows) > 0.20:
                fail = True; reasons.append("edge-width proxy regressed in >20% of audited regions")
            topology_regressions = [r for r in rows if r["topologyComponentDelta"] >= 3]
            if len(topology_regressions) / len(rows) > 0.15:
                fail = True; reasons.append("topology component count changed strongly in >15% of regions")
        if flow_report.get("available") and float(flow_report.get("offEdgeRmsPixels",0.0)) > 0.18 * max(scale,0.25):
            fail = True; reasons.append("learned flow is too active away from detected boundaries")
        if boundary_report.get("available") and float(boundary_report.get("offEdgeMean", 0.0)) > 0.12:
            fail = True; reasons.append("boundary renderer gate is too active away from detected boundaries")

        exact_improvement = exact.get("edgeChamferImprovement") if exact.get("available") else None
        meaningful = False
        if exact_improvement is not None:
            meaningful = bool(exact_improvement > 0.025)
        else:
            meaningful = bool(proxy_mean > 0.02 and (straight_win is None or straight_win >= 0.52))
        flow_collapse = bool(
            flow_report.get("available")
            and float(flow_report.get("rmsPixels", 0.0)) < 0.030 * max(scale, 0.25)
        )
        boundary_inactive = bool(
            boundary_report.get("available")
            and float(boundary_report.get("edgeMean", 0.0)) < 0.05
        )
        if flow_collapse and not boundary_report.get("available"):
            meaningful = False
            reasons.append("flow is near identity / actuator-gate collapse")
        if boundary_inactive:
            meaningful = False
            reasons.append("implicit boundary renderer is inactive on detected edges")
        if options.critic_mode == "required":
            if not critic.get("calibrated"):
                fail = True; reasons.append("required critic is not calibrated")
            elif float(critic.get("candidateWinProbabilityMean",0.5)) < 0.50:
                fail = True; reasons.append("required critic ranks candidate below baseline")

        verdict = (
            "FAIL" if fail
            else "PASS" if meaningful
            else "NEUTRAL_BOUNDARY_INACTIVE" if boundary_inactive
            else "NEUTRAL_FLOW_COLLAPSE" if flow_collapse and not boundary_report.get("available")
            else "NEUTRAL_NO_NET_GAIN"
        )
        if not reasons and verdict == "PASS":
            reasons.append("geometry proxies improved without triggering safety regressions")
        elif not reasons:
            reasons.append("no material regression detected, but geometry improvement is below the proof threshold")

        # Evidence: strongest improvements and regressions.
        evidence_dir = output_dir / "evidence"
        evidence: list[dict[str, Any]] = []
        count = max(0, int(options.evidence_regions))
        if rows and count:
            ordered = sorted(rows, key=lambda r: r["proxyScore"])
            selected_rows = ordered[: count//2] + list(reversed(ordered[-(count-count//2):]))
            seen: set[int] = set()
            for rank, row in enumerate(selected_rows):
                idx = int(row["region"])
                if idx in seen or idx >= len(regions):
                    continue
                seen.add(idx)
                kind = "regression" if row["proxyScore"] < 0 else "improvement"
                filename = f"{source_name}_{kind}_{rank:02d}_r{idx:03d}.png".replace(" ", "_")
                path = evidence_dir / filename
                self._render_evidence(path, baseline0, candidate0, be, ce, regions[idx], flow, gate, f"{source_name} | {kind} | region {idx}", row)
                evidence.append({"kind": kind, "region": idx, "proxyScore": row["proxyScore"], "path": str(path)})

        report = {
            "schema": AUDIT_SCHEMA,
            "source": source_name,
            "analysisScale": scale,
            "analysisSize": [int(bg.shape[1]), int(bg.shape[0])],
            "verdict": verdict,
            "reasons": reasons,
            "summary": {
                "regionCount": len(rows),
                "proxyGeometryImprovementMean": proxy_mean,
                "straightnessImprovementMean": straight_mean,
                "straightnessWinFraction": straight_win,
                "curvatureImprovementMean": curve_mean,
                "curvatureWinFraction": curve_win,
                "edgeWidthRatioMean": width_ratio_mean,
                "fuzzReductionMean": fuzz_reduction_mean,
                "edgeWidthWinFraction": width_win,
                "edgeWidthRegressionFraction": width_regression,
                "offEdgeIdentityRms8bit": off_identity_rms,
                "edgeBandDeltaRms8bit": edge_delta_rms,
                "ringingBefore": ringing_before,
                "ringingAfter": ringing_after,
                "topologyComponentsBefore": topology_before,
                "topologyComponentsAfter": topology_after,
            },
            "flow": flow_report,
            "boundary": boundary_report,
            "exactGroundTruth": exact,
            "critic": critic,
            "regions": rows,
            "evidence": evidence,
        }
        report_path = output_dir / f"{source_name}_geometry_audit.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["reportPath"] = str(report_path)
        return report

    # Purpose: Implement write audit bundle for GeometryAuditService.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def write_audit_bundle(self, output_dir: Path, reports: list[dict[str, Any]]) -> dict[str, Any]:
        output_dir = output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
        verdicts = [str(r.get("verdict","NEUTRAL")) for r in reports]
        sources = [str(r.get("source") or "") for r in reports]
        unique_sources = sorted({source for source in sources if source})
        pbr_alignment_pass = len(unique_sources) == len(reports)
        verdict = (
            "PBR_ALIGNMENT_FAIL" if reports and not pbr_alignment_pass
            else "FAIL" if "FAIL" in verdicts
            else "PASS" if verdicts and all(v == "PASS" for v in verdicts)
            else "NEUTRAL_BOUNDARY_INACTIVE" if any(v == "NEUTRAL_BOUNDARY_INACTIVE" for v in verdicts)
            else "NEUTRAL_NO_NET_GAIN"
        )
        proxy_values = [float(r.get("summary",{}).get("proxyGeometryImprovementMean",0.0)) for r in reports]
        summary = {
            "schema": "NSAMDR_GEOMETRY_AUDIT_BUNDLE_V2",
            "verdict": verdict,
            "textureCount": len(reports),
            "uniqueTextureSourceCount": len(unique_sources),
            "pbrAlignmentAuditPass": pbr_alignment_pass,
            "pbrAlignmentReason": "distinct physical-map audits" if pbr_alignment_pass else "duplicate audit source detected; physical maps were not independently verified",
            "passCount": verdicts.count("PASS"),
            "neutralCount": sum(1 for value in verdicts if value.startswith("NEUTRAL")),
            "failCount": verdicts.count("FAIL"),
            "proxyGeometryImprovementMean": float(np.mean(proxy_values)) if proxy_values else 0.0,
            "reports": [{
                "source": r.get("source"), "verdict": r.get("verdict"), "reasons": r.get("reasons"),
                "summary": r.get("summary"), "flow": r.get("flow"), "boundary": r.get("boundary"), "exactGroundTruth": r.get("exactGroundTruth"),
                "critic": r.get("critic"), "reportPath": r.get("reportPath"), "evidence": r.get("evidence"),
            } for r in reports],
        }
        json_path = output_dir / "geometry_audit.json"
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        csv_path = output_dir / "geometry_audit_regions.csv"
        rows = []
        for report in reports:
            for row in report.get("regions",[]):
                rows.append({"source":report.get("source"), "verdict":report.get("verdict"), **row})
        if rows:
            columns = sorted({k for r in rows for k in r})
            with csv_path.open("w",newline="",encoding="utf-8") as h:
                w=csv.DictWriter(h,fieldnames=columns); w.writeheader(); w.writerows(rows)
        html_path = output_dir / "geometry_audit.html"
        parts = ["<html><head><meta charset='utf-8'><title>NSAMDR geometry audit</title>",
                 "<style>body{font-family:Segoe UI,Arial;background:#111;color:#ddd}table{border-collapse:collapse}td,th{border:1px solid #444;padding:5px}img{max-width:980px;margin:8px 0}.PASS{color:#7fda7f}.FAIL{color:#ff7777}.NEUTRAL,.NEUTRAL_FLOW_COLLAPSE,.NEUTRAL_BOUNDARY_INACTIVE,.NEUTRAL_NO_NET_GAIN{color:#ffd36a}</style></head><body>",
                 f"<h1>NSAMDR geometry audit: <span class='{verdict}'>{verdict}</span></h1>",
                 f"<p>Textures: {len(reports)} | proxy improvement mean: {summary['proxyGeometryImprovementMean']:.4f}</p>"]
        for report in reports:
            src=html.escape(str(report.get("source"))); v=html.escape(str(report.get("verdict")))
            parts.append(f"<h2>{src}: <span class='{v}'>{v}</span></h2><pre>{html.escape(json.dumps(report.get('summary',{}),indent=2))}</pre>")
            for ev in report.get("evidence",[]):
                p=Path(str(ev.get("path","")))
                try: rel=p.relative_to(output_dir)
                except ValueError: rel=p
                parts.append(f"<div>{html.escape(str(ev.get('kind')))} score={float(ev.get('proxyScore',0)):.4f}<br><img src='{html.escape(str(rel).replace(chr(92),'/'))}'></div>")
        parts.append("</body></html>")
        html_path.write_text("\n".join(parts),encoding="utf-8")
        summary.update({"jsonPath":str(json_path),"csvPath":str(csv_path),"htmlPath":str(html_path)})
        return summary

_geometry_audit_service = GeometryAuditService()
_gray = _geometry_audit_service._gray
_resize_for_analysis = _geometry_audit_service._resize_for_analysis
_gradient = _geometry_audit_service._gradient
_edge_mask = _geometry_audit_service._edge_mask
_dilate = _geometry_audit_service._dilate
_fit_straightness = _geometry_audit_service._fit_straightness
_curvature_noise = _geometry_audit_service._curvature_noise
_ringing_proxy = _geometry_audit_service._ringing_proxy
_topology_components = _geometry_audit_service._topology_components
_symmetric_edge_chamfer = _geometry_audit_service._symmetric_edge_chamfer
_candidate_patches = _geometry_audit_service._candidate_patches
_metric_delta = _geometry_audit_service._metric_delta
_relative_improvement = _geometry_audit_service._relative_improvement
_flow_metrics = _geometry_audit_service._flow_metrics
_boundary_gate_metrics = _geometry_audit_service._boundary_gate_metrics
_general_edge_width_proxy = _geometry_audit_service._general_edge_width_proxy
_patch_metrics = _geometry_audit_service._patch_metrics
_critic_classes = _geometry_audit_service._critic_classes
_synthetic_critic_batch = _geometry_audit_service._synthetic_critic_batch
ensure_geometry_critic = _geometry_audit_service.ensure_geometry_critic
_critic_scores = _geometry_audit_service._critic_scores
_render_evidence = _geometry_audit_service._render_evidence
audit_pair = _geometry_audit_service.audit_pair
write_audit_bundle = _geometry_audit_service.write_audit_bundle


# ----------------------------- learned critic -----------------------------
