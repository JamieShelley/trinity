"""Canonical 29-case synthetic geometry proof ladder for NSAMDR V10.6.

Training-time Stage-B selection and the final geometry audit both consume this
module.  Case geometry, low-resolution degradation and teacher SDF therefore
cannot silently diverge between the selector and the acceptance audit.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ProofCase:
    name: str
    target_rgb: np.ndarray
    target_normal: np.ndarray
    target_material: np.ndarray
    target_sdf: np.ndarray
    target_orientation: np.ndarray
    target_edge: np.ndarray
    teacher_gate: np.ndarray
    low_rgb: np.ndarray
    low_normal: np.ndarray
    low_material: np.ndarray
    kind: int  # 1=line, 2=circle, 0=other
    stress: bool


def _canvas(draw_fn, size: int = 512) -> np.ndarray:
    ss = 4
    s = size * ss
    image = np.full((s, s), 36, np.uint8)
    draw_fn(image, ss)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def proof_case_images(size: int = 512) -> list[tuple[str, np.ndarray]]:
    """Return the permanent G0-G5 target images in canonical order."""
    def line(angle_deg: float, width: int = 5):
        def draw(im, ss):
            c = np.array([im.shape[1] / 2, im.shape[0] / 2])
            r = im.shape[0] * 0.62
            a = np.deg2rad(angle_deg)
            d = np.array([np.cos(a), np.sin(a)]) * r
            p0 = tuple(np.round(c - d).astype(int)); p1 = tuple(np.round(c + d).astype(int))
            cv2.line(im, p0, p1, 225, width * ss, cv2.LINE_AA)
        return draw

    def circle(radius: float, width: int = 5):
        return lambda im, ss: cv2.circle(
            im, (im.shape[1] // 2, im.shape[0] // 2), int(radius * ss),
            225, width * ss, cv2.LINE_AA,
        )

    def ellipse(ax: float, ay: float, angle: float, width: int = 5):
        return lambda im, ss: cv2.ellipse(
            im, (im.shape[1] // 2, im.shape[0] // 2), (int(ax * ss), int(ay * ss)),
            angle, 0, 360, 225, width * ss, cv2.LINE_AA,
        )

    def corner(angle_deg: float):
        def draw(im, ss):
            c = np.array([im.shape[1] / 2, im.shape[0] / 2])
            r = im.shape[0] * 0.31
            a = np.deg2rad(angle_deg * 0.5)
            for sign in (-1.0, 1.0):
                d = np.array([np.cos(sign * a), np.sin(sign * a)]) * r
                cv2.line(im, tuple(np.round(c).astype(int)), tuple(np.round(c + d).astype(int)),
                         225, 5 * ss, cv2.LINE_AA)
        return draw

    def rounded_box(im, ss):
        x0, y0, x1, y1 = 92 * ss, 128 * ss, 420 * ss, 382 * ss
        r = 46 * ss; w = 5 * ss
        cv2.line(im, (x0 + r, y0), (x1 - r, y0), 225, w, cv2.LINE_AA)
        cv2.line(im, (x0 + r, y1), (x1 - r, y1), 225, w, cv2.LINE_AA)
        cv2.line(im, (x0, y0 + r), (x0, y1 - r), 225, w, cv2.LINE_AA)
        cv2.line(im, (x1, y0 + r), (x1, y1 - r), 225, w, cv2.LINE_AA)
        cv2.ellipse(im, (x0 + r, y0 + r), (r, r), 180, 0, 90, 225, w, cv2.LINE_AA)
        cv2.ellipse(im, (x1 - r, y0 + r), (r, r), 270, 0, 90, 225, w, cv2.LINE_AA)
        cv2.ellipse(im, (x1 - r, y1 - r), (r, r), 0, 0, 90, 225, w, cv2.LINE_AA)
        cv2.ellipse(im, (x0 + r, y1 - r), (r, r), 90, 0, 90, 225, w, cv2.LINE_AA)

    def parallel(im, ss):
        for off in (-20, 20):
            c = np.array([im.shape[1] / 2, im.shape[0] / 2 + off * ss])
            a = np.deg2rad(27); d = np.array([np.cos(a), np.sin(a)]) * im.shape[0] * 0.55
            cv2.line(im, tuple(np.round(c - d).astype(int)), tuple(np.round(c + d).astype(int)),
                     225, 3 * ss, cv2.LINE_AA)

    def ring(im, ss):
        c = (im.shape[1] // 2, im.shape[0] // 2)
        cv2.circle(im, c, 132 * ss, 225, 3 * ss, cv2.LINE_AA)
        cv2.circle(im, c, 118 * ss, 225, 3 * ss, cv2.LINE_AA)

    def junction(im, ss):
        c = (im.shape[1] // 2, im.shape[0] // 2)
        for a in (15, 135, 255):
            rad = np.deg2rad(a)
            end = (int(c[0] + 170 * ss * np.cos(rad)), int(c[1] + 170 * ss * np.sin(rad)))
            cv2.line(im, c, end, 225, 4 * ss, cv2.LINE_AA)

    cases: list[tuple[str, np.ndarray]] = []
    for a in (1, 3, 7, 11, 19, 33, 45, 67, 83, 89):
        cases.append((f"G0_line_{a:02d}deg", _canvas(line(a, 5), size)))
    cases.extend([
        ("G0_thin_33deg", _canvas(line(33, 2), size)),
        ("G0_wide_19deg", _canvas(line(19, 8), size)),
        ("G1_circle_r92", _canvas(circle(92), size)),
        ("G1_circle_r157", _canvas(circle(157, 4), size)),
        ("G1_ellipse_150x72", _canvas(ellipse(150, 72, 23), size)),
        ("G1_ellipse_118x165", _canvas(ellipse(118, 165, -17, 4), size)),
        ("G1_rounded_box", _canvas(rounded_box, size)),
        ("G2_corner_45", _canvas(corner(45), size)),
        ("G2_corner_90", _canvas(corner(90), size)),
        ("G2_corner_135", _canvas(corner(135), size)),
        ("G3_parallel_lines", _canvas(parallel, size)),
        ("G3_concentric_ring", _canvas(ring, size)),
        ("G3_junction", _canvas(junction, size)),
    ])
    for name, img in list(cases)[::6][:4]:
        luma = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        low = (0.16 + luma[..., None] * 0.11 + np.asarray([0.015, 0.005, 0.0], np.float32)).clip(0, 1)
        cases.append(("G4_lowcontrast_" + name.split('_', 1)[1], np.uint8(np.round(low * 255))))
    cases.append(("G5_degrade_blur_line33", _canvas(line(33, 5), size)))
    cases.append(("G5_degrade_halo_circle", _canvas(circle(118, 5), size)))
    return cases


def pbr_companions(target_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = (
        target_rgb[..., 0] * 0.2126
        + target_rgb[..., 1] * 0.7152
        + target_rgb[..., 2] * 0.0722
    ).astype(np.float32)
    lo = float(np.percentile(gray, 10)); hi = float(np.percentile(gray, 90))
    mask = np.clip((gray - lo) / max(hi - lo, 1.0e-5), 0.0, 1.0)[..., None]
    normal_a = np.asarray([-0.16, 0.10], dtype=np.float32)
    normal_b = np.asarray([0.20, -0.13], dtype=np.float32)
    normal = normal_a.reshape(1, 1, 2) * (1.0 - mask) + normal_b.reshape(1, 1, 2) * mask
    material_a = np.asarray([0.18, 0.02, 0.72], dtype=np.float32)
    material_b = np.asarray([0.76, 0.07, 0.28], dtype=np.float32)
    material = material_a.reshape(1, 1, 3) * (1.0 - mask) + material_b.reshape(1, 1, 3) * mask
    return np.ascontiguousarray(normal), np.ascontiguousarray(material)


def teacher_field(target_rgb: np.ndarray, max_distance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized SDF, orientation, edge and forced-gate teacher."""
    gray = (
        target_rgb[..., 0] * 0.2126
        + target_rgb[..., 1] * 0.7152
        + target_rgb[..., 2] * 0.0722
    ).astype(np.float32)
    lo = float(np.min(gray)); hi = float(np.max(gray))
    coverage_fraction = np.clip((gray - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0).astype(np.float32)
    inside = (coverage_fraction >= 0.5).astype(np.uint8)
    inside_distance = cv2.distanceTransform(inside, cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
    signed = np.where(
        inside > 0,
        -np.maximum(inside_distance - 0.5, 0.0),
        np.maximum(outside_distance - 0.5, 0.0),
    ).astype(np.float32)
    transition = (coverage_fraction > 1.0e-4) & (coverage_fraction < 1.0 - 1.0e-4)
    signed[transition] = 0.5 - coverage_fraction[transition]
    sdf = np.clip(signed / max(float(max_distance), 1.0), -1.0, 1.0)[..., None].astype(np.float32)
    edge = np.exp(-0.5 * (signed / 0.72) ** 2)[..., None].astype(np.float32)
    gy, gx = np.gradient(signed)
    norm = np.sqrt(gx * gx + gy * gy + 1.0e-8)
    orientation = np.stack((gx / norm, gy / norm), axis=-1).astype(np.float32)
    radius = 9.0
    distance = np.abs(signed).astype(np.float32)
    outside = np.maximum(distance - radius, 0.0)
    gate = np.where(distance <= radius, 1.0, np.exp(-outside / 0.25))[..., None].astype(np.float32)
    return (
        np.ascontiguousarray(sdf), np.ascontiguousarray(orientation),
        np.ascontiguousarray(edge), np.ascontiguousarray(np.clip(gate, 0.0, 1.0)),
    )


def build_proof_case(index: int, *, size: int, max_distance: float) -> ProofCase:
    images = proof_case_images(size)
    name, target_rgb_u8 = images[int(index) % len(images)]
    target_rgb = target_rgb_u8.astype(np.float32) / 255.0
    target_normal, target_material = pbr_companions(target_rgb)
    sdf, orientation, edge, gate = teacher_field(target_rgb, max_distance)
    low_size = (size // 4, size // 4)
    low_rgb = cv2.resize(target_rgb, low_size, interpolation=cv2.INTER_AREA)
    low_normal = cv2.resize(target_normal, low_size, interpolation=cv2.INTER_LINEAR)
    low_material = cv2.resize(target_material, low_size, interpolation=cv2.INTER_AREA)
    if "G5_degrade_blur" in name:
        low_rgb = cv2.GaussianBlur(low_rgb, (5, 5), 1.15)
        low_material = cv2.GaussianBlur(low_material, (5, 5), 0.85)
    elif "G5_degrade_halo" in name:
        blur = cv2.GaussianBlur(low_rgb, (0, 0), 1.0)
        low_rgb = np.clip(low_rgb + (low_rgb - blur) * 0.42, 0.0, 1.0)
        low_rgb = cv2.GaussianBlur(low_rgb, (3, 3), 0.55)
    kind = 1 if ("line_" in name or name.startswith("G0_line") or name.startswith("G4_lowcontrast_line")) else (2 if "circle" in name else 0)
    return ProofCase(
        name=name,
        target_rgb=np.ascontiguousarray(target_rgb),
        target_normal=np.ascontiguousarray(target_normal),
        target_material=np.ascontiguousarray(target_material),
        target_sdf=sdf,
        target_orientation=orientation,
        target_edge=edge,
        teacher_gate=gate,
        low_rgb=np.ascontiguousarray(low_rgb.astype(np.float32)),
        low_normal=np.ascontiguousarray(low_normal.astype(np.float32)),
        low_material=np.ascontiguousarray(low_material.astype(np.float32)),
        kind=kind,
        stress=name.startswith("G5_"),
    )


PROOF_CASE_COUNT = 29
