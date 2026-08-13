#!/usr/bin/env python3
"""Exact analytic PBR boundary proof for NSAMDR V9.8.3 sign-gauge metric-SDF convergence.

Creates known continuous HR geometry, rasterises it to LR, runs GeometryNet, and
measures whether the learned warp moves the reconstruction closer to the exact
HR boundary. This is the objective counterpart to the real-Raven proxy audit.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v9.geometry_audit import AuditOptions, audit_pair, ensure_geometry_critic
from v9.inference import load_trained_model, resolve_device
from v9.config import V9Config
from v9.model import BoundaryRenderer, build_model_input


def _canvas(draw_fn, size: int = 512) -> np.ndarray:
    ss = 4
    s = size * ss
    image = np.full((s, s), 36, np.uint8)
    draw_fn(image, ss)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return bgr


def _cases(size: int = 512) -> list[tuple[str, np.ndarray]]:
    """Permanent G0-G5 geometry ladder used before any Raven judgement."""
    def line(angle_deg: float, width: int = 5):
        def draw(im, ss):
            c = np.array([im.shape[1]/2, im.shape[0]/2])
            r = im.shape[0] * 0.62
            a = np.deg2rad(angle_deg)
            d = np.array([np.cos(a), np.sin(a)]) * r
            p0 = tuple(np.round(c-d).astype(int)); p1 = tuple(np.round(c+d).astype(int))
            cv2.line(im, p0, p1, 225, width*ss, cv2.LINE_AA)
        return draw
    def circle(radius: float, width: int = 5):
        return lambda im,ss: cv2.circle(im,(im.shape[1]//2,im.shape[0]//2),int(radius*ss),225,width*ss,cv2.LINE_AA)
    def ellipse(ax: float, ay: float, angle: float, width: int = 5):
        return lambda im,ss: cv2.ellipse(im,(im.shape[1]//2,im.shape[0]//2),(int(ax*ss),int(ay*ss)),angle,0,360,225,width*ss,cv2.LINE_AA)
    def corner(angle_deg: float):
        def draw(im,ss):
            c=np.array([im.shape[1]/2,im.shape[0]/2])
            r=im.shape[0]*0.31
            a=np.deg2rad(angle_deg*0.5)
            for sign in (-1.0,1.0):
                d=np.array([np.cos(sign*a),np.sin(sign*a)])*r
                cv2.line(im,tuple(np.round(c).astype(int)),tuple(np.round(c+d).astype(int)),225,5*ss,cv2.LINE_AA)
        return draw
    def rounded_box(im,ss):
        x0,y0,x1,y1 = 92*ss,128*ss,420*ss,382*ss
        r=46*ss; w=5*ss
        cv2.line(im,(x0+r,y0),(x1-r,y0),225,w,cv2.LINE_AA); cv2.line(im,(x0+r,y1),(x1-r,y1),225,w,cv2.LINE_AA)
        cv2.line(im,(x0,y0+r),(x0,y1-r),225,w,cv2.LINE_AA); cv2.line(im,(x1,y0+r),(x1,y1-r),225,w,cv2.LINE_AA)
        cv2.ellipse(im,(x0+r,y0+r),(r,r),180,0,90,225,w,cv2.LINE_AA)
        cv2.ellipse(im,(x1-r,y0+r),(r,r),270,0,90,225,w,cv2.LINE_AA)
        cv2.ellipse(im,(x1-r,y1-r),(r,r),0,0,90,225,w,cv2.LINE_AA)
        cv2.ellipse(im,(x0+r,y1-r),(r,r),90,0,90,225,w,cv2.LINE_AA)
    def parallel(im,ss):
        for off in (-20,20):
            c=np.array([im.shape[1]/2,im.shape[0]/2+off*ss]); a=np.deg2rad(27); d=np.array([np.cos(a),np.sin(a)])*im.shape[0]*0.55
            cv2.line(im,tuple(np.round(c-d).astype(int)),tuple(np.round(c+d).astype(int)),225,3*ss,cv2.LINE_AA)
    def ring(im,ss):
        c=(im.shape[1]//2,im.shape[0]//2)
        cv2.circle(im,c,132*ss,225,3*ss,cv2.LINE_AA)
        cv2.circle(im,c,118*ss,225,3*ss,cv2.LINE_AA)
    def junction(im,ss):
        c=(im.shape[1]//2,im.shape[0]//2)
        for a in (15,135,255):
            rad=np.deg2rad(a); end=(int(c[0]+170*ss*np.cos(rad)),int(c[1]+170*ss*np.sin(rad)))
            cv2.line(im,c,end,225,4*ss,cv2.LINE_AA)

    cases=[]
    # G0: arbitrary-angle lines, including almost-axis-aligned cases.
    for a in (1,3,7,11,19,33,45,67,83,89):
        cases.append((f"G0_line_{a:02d}deg",_canvas(line(a,5),size)))
    cases.extend([
        ("G0_thin_33deg",_canvas(line(33,2),size)),
        ("G0_wide_19deg",_canvas(line(19,8),size)),
        ("G1_circle_r92",_canvas(circle(92),size)),
        ("G1_circle_r157",_canvas(circle(157,4),size)),
        ("G1_ellipse_150x72",_canvas(ellipse(150,72,23),size)),
        ("G1_ellipse_118x165",_canvas(ellipse(118,165,-17,4),size)),
        ("G1_rounded_box",_canvas(rounded_box,size)),
        ("G2_corner_45",_canvas(corner(45),size)),
        ("G2_corner_90",_canvas(corner(90),size)),
        ("G2_corner_135",_canvas(corner(135),size)),
        ("G3_parallel_lines",_canvas(parallel,size)),
        ("G3_concentric_ring",_canvas(ring,size)),
        ("G3_junction",_canvas(junction,size)),
    ])
    # G4: low-contrast dark-hull variants of representative geometry.
    for name,img in list(cases)[::6][:4]:
        f=img.astype(np.float32)/255.0
        luma=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
        low=(0.16 + luma[...,None]*0.11 + np.asarray([0.015,0.005,0.0],np.float32)).clip(0,1)
        cases.append(("G4_lowcontrast_"+name.split('_',1)[1],np.uint8(np.round(low*255))))
    # G5 names trigger an additional LR degradation in main().
    cases.append(("G5_degrade_blur_line33",_canvas(line(33,5),size)))
    cases.append(("G5_degrade_halo_circle",_canvas(circle(118,5),size)))
    return cases


def _pbr_companions(target_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build aligned companion maps so the proof exercises the deployment input."""
    gray = (
        target_rgb[..., 0] * 0.2126
        + target_rgb[..., 1] * 0.7152
        + target_rgb[..., 2] * 0.0722
    ).astype(np.float32)
    lo = float(np.percentile(gray, 10))
    hi = float(np.percentile(gray, 90))
    mask = np.clip((gray - lo) / max(hi - lo, 1.0e-5), 0.0, 1.0)[..., None]

    normal_a = np.asarray([-0.16, 0.10], dtype=np.float32)
    normal_b = np.asarray([0.20, -0.13], dtype=np.float32)
    normal = normal_a.reshape(1, 1, 2) * (1.0 - mask) + normal_b.reshape(1, 1, 2) * mask

    material_a = np.asarray([0.18, 0.02, 0.72], dtype=np.float32)
    material_b = np.asarray([0.76, 0.07, 0.28], dtype=np.float32)
    material = material_a.reshape(1, 1, 3) * (1.0 - mask) + material_b.reshape(1, 1, 3) * mask
    return np.ascontiguousarray(normal), np.ascontiguousarray(material)



def _teacher_field(
    target_rgb: np.ndarray,
    _target_normal: np.ndarray,
    _target_material: np.ndarray,
    config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact HR region SDF for the controlled black/bright shapes.

    Unlike the general texture contour extractor, this keeps both sides of thin
    lines/stripes.  A centreline SDF would be the wrong geometry for those cases.
    """
    gray=(
        target_rgb[...,0]*0.2126
        +target_rgb[...,1]*0.7152
        +target_rgb[...,2]*0.0722
    ).astype(np.float32)
    # Synthetic cases can contain extremely sparse bright lines, so percentile
    # thresholds collapse to the background.  The audit owns these controlled
    # targets; use their full dynamic range to recover the exact region mask.
    lo=float(np.min(gray))
    hi=float(np.max(gray))
    threshold=(lo+hi)*0.5
    coverage_fraction=np.clip(
        (gray-lo)/max(hi-lo,1.0e-6),0.0,1.0
    ).astype(np.float32)
    inside=(coverage_fraction>=0.5).astype(np.uint8)
    inside_distance=cv2.distanceTransform(inside,cv2.DIST_L2,5)
    outside_distance=cv2.distanceTransform(1-inside,cv2.DIST_L2,5)
    # Pixel-centre signed distance: nearest pure centres sit at +/-0.5 px.
    # For anti-aliased synthetic pixels the authored fractional coverage gives
    # the exact subpixel zero crossing, matching coverage=clip(0.5-SDF,0,1).
    signed=np.where(
        inside>0,
        -np.maximum(inside_distance-0.5,0.0),
        np.maximum(outside_distance-0.5,0.0),
    ).astype(np.float32)
    transition=(coverage_fraction>1.0e-4)&(coverage_fraction<1.0-1.0e-4)
    signed[transition]=0.5-coverage_fraction[transition]
    max_distance=float(config.contour_sdf_max_distance_pixels)
    sdf=np.clip(signed/max(max_distance,1.0),-1.0,1.0)[...,None].astype(np.float32)
    edge=np.exp(-0.5*(signed/0.72)**2)[...,None].astype(np.float32)
    # Stage A/B use full reconstruction authority across the known contaminated
    # band. EXP_0003 showed a soft teacher retained part of the degraded edge.
    radius=max(7.0,min(float(config.boundary_renderer_band_pixels)+5.0,9.0))
    distance=np.abs(signed).astype(np.float32)
    outside=np.maximum(distance-radius,0.0)
    gate=np.where(distance<=radius,1.0,np.exp(-outside/0.25))[...,None].astype(np.float32)
    return np.ascontiguousarray(sdf),np.ascontiguousarray(np.clip(gate,0.0,1.0)),np.ascontiguousarray(edge)


def _sdf_metrics(predicted: np.ndarray, target: np.ndarray, edge: np.ndarray) -> dict[str, float]:
    pred_px = predicted[..., 0].astype(np.float32)
    tgt_px = target[..., 0].astype(np.float32)
    edge_mask = edge[..., 0] > 0.35
    mae = float(np.mean(np.abs(pred_px - tgt_px)))
    zero_rms = float(np.sqrt(np.mean(pred_px[edge_mask] ** 2))) if np.any(edge_mask) else float("nan")
    gy, gx = np.gradient(pred_px)
    grad = np.sqrt(gx * gx + gy * gy + 1.0e-8)
    eik = float(np.mean(np.abs(grad - 1.0)))
    tgy, tgx = np.gradient(tgt_px)
    pnorm = np.stack((gx, gy), axis=-1) / np.maximum(grad[..., None], 1.0e-5)
    tnorm_len = np.sqrt(tgx * tgx + tgy * tgy + 1.0e-8)
    tnorm = np.stack((tgx, tgy), axis=-1) / np.maximum(tnorm_len[..., None], 1.0e-5)
    alignment = np.abs(np.sum(pnorm * tnorm, axis=-1))
    tangent_error = float(np.mean(1.0 - np.clip(alignment[edge_mask], 0.0, 1.0))) if np.any(edge_mask) else float("nan")
    return {
        "maePixels": mae,
        "zeroSetRmsPixels": zero_rms,
        "eikonalError": eik,
        "normalAlignmentError": tangent_error,
    }


def _profile_metrics(
    image_rgb: np.ndarray,
    target_rgb: np.ndarray,
    target_sdf_pixels: np.ndarray,
) -> dict[str, float]:
    """Measure actual transition concentration around the analytic zero-set."""
    gray=(
        image_rgb[...,0]*0.2126 + image_rgb[...,1]*0.7152 + image_rgb[...,2]*0.0722
    ).astype(np.float32)
    gy,gx=np.gradient(gray)
    grad=np.sqrt(gx*gx+gy*gy+1.0e-10)
    distance=np.abs(target_sdf_pixels[...,0].astype(np.float32))
    band=distance<=6.0
    weight=grad*band.astype(np.float32)
    total=float(np.sum(weight))
    if total<=1.0e-8:
        width=float("inf"); outer=1.0
    else:
        width=float(np.sqrt(np.sum(weight*distance*distance)/total))
        outer=float(np.sum(weight*((distance>1.5)&(distance<=6.0)))/total)
    target_lo=np.min(target_rgb,axis=(0,1),keepdims=True).astype(np.float32)
    target_hi=np.max(target_rgb,axis=(0,1),keepdims=True).astype(np.float32)
    below=np.maximum(target_lo-image_rgb,0.0)
    above=np.maximum(image_rgb-target_hi,0.0)
    halo=float(max(float(np.max(below)),float(np.max(above)))*255.0)
    return {
        "widthRmsPixels":width,
        "outerGradientFraction":outer,
        "haloOvershoot8bit":halo,
    }


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(
        np.uint8(np.round(np.clip(rgb, 0.0, 1.0) * 255.0)),
        cv2.COLOR_RGB2BGR,
    )


def _stage_contact(
    path: Path,
    baseline: np.ndarray,
    oracle: np.ndarray,
    sdf_forced: np.ndarray,
    final: np.ndarray,
    labels: tuple[str, str, str, str],
) -> None:
    panels = []
    for image, label in zip((baseline, oracle, sdf_forced, final), labels):
        bgr = _to_bgr(image)
        panel = bgr.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 34), (16, 16, 16), -1)
        cv2.putText(panel, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
        panels.append(panel)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(panels, axis=1))



def _oracle_render_from_low(
    low_rgb: np.ndarray,
    target_sdf: np.ndarray,
    teacher_gate: np.ndarray,
    config: V9Config,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parameter-free Stage-A render.

    The display baseline remains bicubic, while plateau evidence is a monotonic
    bilinear upscale of the authored LR samples. This isolates renderer quality
    from GeometryNet and prevents bicubic overshoot from becoming a plateau.
    """
    source = torch.from_numpy(low_rgb.transpose(2, 0, 1)).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    baseline = F.interpolate(
        source, scale_factor=4, mode="bicubic", align_corners=False, antialias=True
    ).clamp(0.0, 1.0)
    evidence = F.interpolate(
        source, scale_factor=4, mode="bilinear", align_corners=False
    ).clamp(0.0, 1.0)
    sdf_t = torch.from_numpy(target_sdf.transpose(2, 0, 1)).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    gate_t = torch.from_numpy(teacher_gate.transpose(2, 0, 1)).unsqueeze(0).to(
        device=device, dtype=torch.float32
    )
    zeros = torch.zeros_like(sdf_t)
    ones = torch.ones_like(sdf_t)
    renderer = BoundaryRenderer(config).to(device).eval()
    with torch.no_grad():
        rendered, _ = renderer(
            baseline,
            sdf_t,
            zeros,
            zeros,
            zeros,
            ones,
            enabled=True,
            plateau_evidence=evidence,
            source_value_lr=source,
            forced_gate=gate_t,
            forced_hardness=ones,
        )
    baseline_np = np.clip(
        baseline[0].permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0
    )
    evidence_np = np.clip(
        evidence[0].permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0
    )
    rendered_np = np.clip(
        rendered[0].permute(1, 2, 0).float().cpu().numpy(), 0.0, 1.0
    )
    return baseline_np, evidence_np, rendered_np


def _synthetic_region_components(image_rgb: np.ndarray, target_rgb: np.ndarray) -> int:
    gray = (
        image_rgb[..., 0] * 0.2126
        + image_rgb[..., 1] * 0.7152
        + image_rgb[..., 2] * 0.0722
    ).astype(np.float32)
    target_gray = (
        target_rgb[..., 0] * 0.2126
        + target_rgb[..., 1] * 0.7152
        + target_rgb[..., 2] * 0.0722
    ).astype(np.float32)
    threshold = 0.5 * (float(np.min(target_gray)) + float(np.max(target_gray)))
    mask = (gray >= threshold).astype(np.uint8)
    count, _labels = cv2.connectedComponents(mask, connectivity=8)
    return max(0, int(count) - 1)


def _topology_mismatch(image_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
    return float(
        _synthetic_region_components(image_rgb, target_rgb)
        != _synthetic_region_components(target_rgb, target_rgb)
    )


def _synthetic_region_boundary(image_rgb: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    """Extract the material-region midpoint contour for controlled synthetic data.

    Gradient-threshold Chamfer is intentionally not used for promotion here:
    making an edge sharper changes its gradient distribution and can make a
    visually/exactly better contour look worse.  The synthetic target has known
    two-side plateaus, so its 50% material split is the physically meaningful
    zero-set comparison.
    """
    gray = (
        image_rgb[..., 0] * 0.2126
        + image_rgb[..., 1] * 0.7152
        + image_rgb[..., 2] * 0.0722
    ).astype(np.float32)
    target_gray = (
        target_rgb[..., 0] * 0.2126
        + target_rgb[..., 1] * 0.7152
        + target_rgb[..., 2] * 0.0722
    ).astype(np.float32)
    threshold = 0.5 * (float(np.min(target_gray)) + float(np.max(target_gray)))
    region = (gray >= threshold).astype(np.uint8)
    if not np.any(region):
        return np.zeros_like(region)
    eroded = cv2.erode(region, np.ones((3, 3), np.uint8), iterations=1)
    return (region != eroded).astype(np.uint8)


def _synthetic_region_chamfer(image_rgb: np.ndarray, target_rgb: np.ndarray) -> float:
    candidate = _synthetic_region_boundary(image_rgb, target_rgb)
    target = _synthetic_region_boundary(target_rgb, target_rgb)
    if not np.any(candidate) or not np.any(target):
        return float("inf")
    candidate_distance = cv2.distanceTransform(
        1 - candidate, cv2.DIST_L2, 3
    )
    target_distance = cv2.distanceTransform(
        1 - target, cv2.DIST_L2, 3
    )
    return 0.5 * (
        float(np.mean(target_distance[candidate.astype(bool)]))
        + float(np.mean(candidate_distance[target.astype(bool)]))
    )


def _synthetic_region_chamfer_improvement(
    baseline_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    target_rgb: np.ndarray,
) -> tuple[float, float, float]:
    before = _synthetic_region_chamfer(baseline_rgb, target_rgb)
    after = _synthetic_region_chamfer(candidate_rgb, target_rgb)
    if np.isfinite(after) and not np.isfinite(before):
        improvement = 1.0
    elif not np.isfinite(after):
        improvement = -1.0
    else:
        improvement = float((before - after) / max(abs(before), 1.0e-5))
    return float(before), float(after), improvement


def _topology_regression_fraction(reports: list[dict]) -> float:
    changes: list[float] = []
    for report in reports:
        summary = report.get("summary", {}) if isinstance(report, dict) else {}
        if not isinstance(summary, dict):
            continue
        before = summary.get("topologyComponentsBefore")
        after = summary.get("topologyComponentsAfter")
        if before is not None and after is not None:
            changes.append(float(int(before) != int(after)))
    return float(np.mean(changes)) if changes else 0.0


def _improvement(report: dict) -> float:
    value = report.get("exactGroundTruth", {}).get("edgeChamferImprovement")
    return float(value) if value is not None and np.isfinite(value) else 0.0


def main() -> int:
    parser=argparse.ArgumentParser(description="Sign-gauge metric-SDF geometry proof for NSAMDR V9.8.3")
    parser.add_argument("--repo-root",type=Path,default=Path.cwd())
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--config",type=Path)
    parser.add_argument("--oracle-only",action="store_true")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--device",choices=("cuda","cpu","auto"),default="cuda")
    parser.add_argument("--critic",choices=("off","auto","required"),default="auto")
    parser.add_argument("--critic-checkpoint",type=Path)
    parser.add_argument("--evidence-regions",type=int,default=6)
    parser.add_argument("--critic-steps",type=int,default=120)
    args=parser.parse_args()

    root=args.repo_root.resolve(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    if args.oracle_only:
        if args.config is None:
            raise RuntimeError("--oracle-only requires --config")
        config_path=args.config if args.config.is_absolute() else root/args.config
        config=V9Config.load(config_path.resolve())
        device=resolve_device(config,args.device)
        model=None
    else:
        if args.checkpoint is None:
            raise RuntimeError("checkpoint is required unless --oracle-only is used")
        checkpoint=args.checkpoint if args.checkpoint.is_absolute() else root/args.checkpoint
        model, config, _payload = load_trained_model(checkpoint.resolve(), "cpu")
        device=resolve_device(config,args.device)
        model=model.to(device).eval()
    critic_checkpoint=(args.critic_checkpoint or (root/"artifacts/nsamdr/geometry_critic/geometry_pair_critic.pt")).resolve()
    critic_calibration={"enabled":args.critic!="off" and not args.oracle_only}
    if args.critic!="off" and not args.oracle_only:
        critic_calibration.update(ensure_geometry_critic(critic_checkpoint,device=str(device),steps=args.critic_steps))
        print(f"[geometry-audit] critic accuracy={critic_calibration['validationAccuracy']:.3f} calibrated={critic_calibration['calibrated']}",flush=True)

    rows=[]; oracle_reports=[]; sdf_reports=[]; final_reports=[]
    for index,(name,target_bgr) in enumerate(_cases()):
        target_rgb=cv2.cvtColor(target_bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        target_normal,target_material=_pbr_companions(target_rgb)
        target_sdf, teacher_gate, target_edge = _teacher_field(
            target_rgb,target_normal,target_material,config
        )

        low_size=(target_rgb.shape[1]//4,target_rgb.shape[0]//4)
        low=cv2.resize(target_rgb,low_size,interpolation=cv2.INTER_AREA)
        low_normal=cv2.resize(target_normal,low_size,interpolation=cv2.INTER_LINEAR)
        low_material=cv2.resize(target_material,low_size,interpolation=cv2.INTER_AREA)
        if "G5_degrade_blur" in name:
            low=cv2.GaussianBlur(low,(5,5),1.15)
            low_material=cv2.GaussianBlur(low_material,(5,5),0.85)
        elif "G5_degrade_halo" in name:
            blur=cv2.GaussianBlur(low,(0,0),1.0)
            low=np.clip(low+(low-blur)*0.42,0.0,1.0)
            low=cv2.GaussianBlur(low,(3,3),0.55)
        baseline,evidence,oracle=_oracle_render_from_low(
            low,target_sdf,teacher_gate,config,device
        )
        target_sdf_pixels=target_sdf*float(config.contour_sdf_max_distance_pixels)
        target_profile=_profile_metrics(target_rgb,target_rgb,target_sdf_pixels)
        baseline_profile=_profile_metrics(baseline,target_rgb,target_sdf_pixels)
        evidence_profile=_profile_metrics(evidence,target_rgb,target_sdf_pixels)
        oracle_profile=_profile_metrics(oracle,target_rgb,target_sdf_pixels)

        if args.oracle_only:
            sdf_forced=baseline
            final=baseline
            final_gate=np.zeros(target_sdf.shape[:2]+(1,),np.float32)
            sdf_quality={
                "maePixels":float("nan"),
                "zeroSetRmsPixels":float("nan"),
                "eikonalError":float("nan"),
                "normalAlignmentError":float("nan"),
            }
            sdf_profile=baseline_profile
            final_profile=baseline_profile
        else:
            model_input=build_model_input(low,low_normal,low_material,degradation_level=1.0)
            tensor=torch.from_numpy(model_input).unsqueeze(0).to(device=device,dtype=torch.float32)
            gate_t=torch.from_numpy(teacher_gate.transpose(2,0,1)).unsqueeze(0).to(device=device,dtype=torch.float32)
            hard_t=torch.ones_like(gate_t)
            assert model is not None
            with torch.no_grad():
                # Stage B must evaluate the exact SDF policy that was trained.
                # V9.8.2 loaded checkpoints in full inference mode (±2 px residual)
                # although sdf-proof was trained at ±1 px.
                model.set_phase("sdf-proof")
                model.eval()
                sdf_out=model(
                    tensor,
                    gate_override=gate_t,
                    hardness_override=hard_t,
                    renderer_enabled_override=True,
                )
                model.set_inference_mode()
                model.eval()
                final_out=model(tensor)
            sdf_forced=np.clip(sdf_out["boundary_reconstructed_albedo"][0].permute(1,2,0).float().cpu().numpy(),0,1)
            final=np.clip(final_out["boundary_reconstructed_albedo"][0].permute(1,2,0).float().cpu().numpy(),0,1)
            final_gate=final_out.get("boundary_gate_prediction",final_out["boundary_gate"])[0].permute(1,2,0).float().cpu().numpy()
            pred_sdf_pixels=final_out["predicted_sdf_pixels"][0].permute(1,2,0).float().cpu().numpy()
            sdf_quality=_sdf_metrics(pred_sdf_pixels,target_sdf_pixels,target_edge)
            sdf_profile=_profile_metrics(sdf_forced,target_rgb,target_sdf_pixels)
            final_profile=_profile_metrics(final,target_rgb,target_sdf_pixels)

        base_bgr=_to_bgr(baseline)
        oracle_report=audit_pair(
            base_bgr,_to_bgr(oracle),source_name=f"stageA_{index:02d}_{name}",output_dir=out/"stageA_oracle_renderer",
            options=AuditOptions(evidence_regions=0,critic_mode="off"),
            target_image=target_bgr,
        )
        oracle_reports.append(oracle_report)
        if args.oracle_only:
            sdf_report={"exactGroundTruth":{"edgeChamferImprovement":0.0},"summary":{}}
            final_report={"exactGroundTruth":{"edgeChamferImprovement":0.0},"summary":{}}
        else:
            sdf_report=audit_pair(
                base_bgr,_to_bgr(sdf_forced),source_name=f"stageB_{index:02d}_{name}",output_dir=out/"stageB_predicted_sdf",
                options=AuditOptions(evidence_regions=0,critic_mode="off"),
                target_image=target_bgr,
            )
            sdf_reports.append(sdf_report)
            final_report=audit_pair(
                base_bgr,_to_bgr(final),source_name=f"stageC_{index:02d}_{name}",output_dir=out/"stageC_full",
                options=AuditOptions(evidence_regions=args.evidence_regions,critic_mode=args.critic),
                gate=final_gate,target_image=target_bgr,
                critic_checkpoint=critic_checkpoint if args.critic!="off" else None,
                critic_device=str(device),
            )
            final_reports.append(final_report)

        _stage_contact(
            out/"staged_evidence"/f"{index:02d}_{name}_stages.png",
            baseline,oracle,sdf_forced,final,
            ("1 baseline","2 GT SDF + forced gate","3 predicted SDF + forced gate","4 predicted SDF + predicted gate"),
        )

        renderer_chamfer_before, renderer_chamfer_after, renderer_chamfer_gain = (
            _synthetic_region_chamfer_improvement(baseline, oracle, target_rgb)
        )
        if args.oracle_only:
            sdf_chamfer_gain = 0.0
            final_chamfer_gain = 0.0
        else:
            _sdf_before, _sdf_after, sdf_chamfer_gain = (
                _synthetic_region_chamfer_improvement(baseline, sdf_forced, target_rgb)
            )
            _final_before, _final_after, final_chamfer_gain = (
                _synthetic_region_chamfer_improvement(baseline, final, target_rgb)
            )

        row={
            "case":name,
            "rendererChamferImprovement":renderer_chamfer_gain,
            "rendererRegionChamferBefore":renderer_chamfer_before,
            "rendererRegionChamferAfter":renderer_chamfer_after,
            "rendererGradientChamferImprovement":_improvement(oracle_report),
            "sdfForcedChamferImprovement":sdf_chamfer_gain,
            "sdfGradientChamferImprovement":_improvement(sdf_report),
            "finalChamferImprovement":final_chamfer_gain,
            "finalGradientChamferImprovement":_improvement(final_report),
            "finalBoundaryGateEdgeMean":final_report.get("boundary",{}).get("edgeMean"),
            "finalBoundaryGateOffEdgeMean":final_report.get("boundary",{}).get("offEdgeMean"),
            "criticWinProbability":final_report.get("critic",{}).get("candidateWinProbabilityMean"),
            "targetProfileWidthRmsPixels":target_profile["widthRmsPixels"],
            "baselineProfileWidthRatio":baseline_profile["widthRmsPixels"] / max(target_profile["widthRmsPixels"],1.0e-6),
            "baselineHaloOvershoot8bit":baseline_profile["haloOvershoot8bit"],
            "evidenceProfileWidthRatio":evidence_profile["widthRmsPixels"] / max(target_profile["widthRmsPixels"],1.0e-6),
            "evidenceHaloOvershoot8bit":evidence_profile["haloOvershoot8bit"],
            "rendererProfileWidthRmsPixels":oracle_profile["widthRmsPixels"],
            "rendererProfileWidthRatio":oracle_profile["widthRmsPixels"] / max(target_profile["widthRmsPixels"],1.0e-6),
            "rendererOuterGradientFraction":oracle_profile["outerGradientFraction"],
            "rendererHaloOvershoot8bit":oracle_profile["haloOvershoot8bit"],
            "rendererTopologyMismatch":_topology_mismatch(oracle,target_rgb),
            "sdfTopologyMismatch":_topology_mismatch(sdf_forced,target_rgb) if not args.oracle_only else 0.0,
            "finalTopologyMismatch":_topology_mismatch(final,target_rgb) if not args.oracle_only else 0.0,
            "sdfProfileWidthRatio":sdf_profile["widthRmsPixels"] / max(target_profile["widthRmsPixels"],1.0e-6),
            "sdfOuterGradientFraction":sdf_profile["outerGradientFraction"],
            "sdfHaloOvershoot8bit":sdf_profile["haloOvershoot8bit"],
            "finalProfileWidthRatio":final_profile["widthRmsPixels"] / max(target_profile["widthRmsPixels"],1.0e-6),
            "finalOuterGradientFraction":final_profile["outerGradientFraction"],
            "finalHaloOvershoot8bit":final_profile["haloOvershoot8bit"],
            "targetOuterGradientFraction":target_profile["outerGradientFraction"],
            **{f"sdf{k[0].upper()+k[1:]}":v for k,v in sdf_quality.items()},
        }
        rows.append(row)
        print(
            f"[geometry-audit] {name:18s} "
            f"renderer={row['rendererChamferImprovement']:+.2%} "
            f"sdf={row['sdfForcedChamferImprovement']:+.2%} "
            f"final={row['finalChamferImprovement']:+.2%} "
            + (f"zero={sdf_quality['zeroSetRmsPixels']:.3f}px" if not args.oracle_only else "oracle-only"),
            flush=True,
        )

    def values(key):
        return [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(r[key])]

    renderer_vals=values("rendererChamferImprovement")
    sdf_vals=values("sdfForcedChamferImprovement")
    final_vals=values("finalChamferImprovement")
    renderer_mean=float(np.mean(renderer_vals)) if renderer_vals else 0.0
    sdf_mean=float(np.mean(sdf_vals)) if sdf_vals else 0.0
    final_mean=float(np.mean(final_vals)) if final_vals else 0.0
    renderer_win=float(np.mean(np.asarray(renderer_vals)>0.0)) if renderer_vals else 0.0
    sdf_win=float(np.mean(np.asarray(sdf_vals)>0.0)) if sdf_vals else 0.0
    final_win=float(np.mean(np.asarray(final_vals)>0.0)) if final_vals else 0.0
    gate_values=[float(r["finalBoundaryGateEdgeMean"]) for r in rows if r.get("finalBoundaryGateEdgeMean") is not None]
    gate_edge_mean=float(np.mean(gate_values)) if gate_values else 0.0

    renderer_regress=float(np.mean(np.asarray(renderer_vals)<-0.02)) if renderer_vals else 1.0
    sdf_regress=float(np.mean(np.asarray(sdf_vals)<-0.02)) if sdf_vals else 1.0
    zero_mean=float(np.mean(values("sdfZeroSetRmsPixels"))) if values("sdfZeroSetRmsPixels") else float("inf")
    eikonal_mean=float(np.mean(values("sdfEikonalError"))) if values("sdfEikonalError") else float("inf")

    renderer_width_ratio=float(np.mean(values("rendererProfileWidthRatio"))) if values("rendererProfileWidthRatio") else float("inf")
    renderer_outer=float(np.mean(values("rendererOuterGradientFraction"))) if values("rendererOuterGradientFraction") else 1.0
    target_outer=float(np.mean(values("targetOuterGradientFraction"))) if values("targetOuterGradientFraction") else 0.0
    renderer_halo=max(values("rendererHaloOvershoot8bit"),default=float("inf"))
    core_rows=[r for r in rows if not str(r.get("case","")).startswith("G5_")]
    stress_rows=[r for r in rows if str(r.get("case","")).startswith("G5_")]
    renderer_core_halo=max(
        [float(r["rendererHaloOvershoot8bit"]) for r in core_rows],
        default=float("inf"),
    )
    renderer_stress_halo=max(
        [float(r["rendererHaloOvershoot8bit"]) for r in stress_rows],
        default=0.0,
    )
    stress_halo_ratios=[]
    for r in stress_rows:
        before=float(r.get("baselineHaloOvershoot8bit") or 0.0)
        after=float(r.get("rendererHaloOvershoot8bit") or 0.0)
        if before > 1.0:
            stress_halo_ratios.append(after / max(before, 1.0e-6))
    renderer_stress_halo_ratio=max(stress_halo_ratios,default=0.0)
    sdf_width_ratio=float(np.mean(values("sdfProfileWidthRatio"))) if values("sdfProfileWidthRatio") else float("inf")
    final_width_ratio=float(np.mean(values("finalProfileWidthRatio"))) if values("finalProfileWidthRatio") else float("inf")
    final_outer=float(np.mean(values("finalOuterGradientFraction"))) if values("finalOuterGradientFraction") else 1.0
    final_halo=max(values("finalHaloOvershoot8bit"),default=float("inf"))
    final_core_halo=max(
        [float(r["finalHaloOvershoot8bit"]) for r in core_rows],
        default=float("inf"),
    )
    final_stress_halo=max(
        [float(r["finalHaloOvershoot8bit"]) for r in stress_rows],
        default=0.0,
    )
    final_stress_halo_ratios=[]
    for r in stress_rows:
        before=float(r.get("baselineHaloOvershoot8bit") or 0.0)
        after=float(r.get("finalHaloOvershoot8bit") or 0.0)
        if before > 1.0:
            final_stress_halo_ratios.append(after / max(before, 1.0e-6))
    final_stress_halo_ratio=max(final_stress_halo_ratios,default=0.0)
    oracle_topology_regression=float(np.mean(values("rendererTopologyMismatch"))) if values("rendererTopologyMismatch") else 0.0
    sdf_topology_regression=float(np.mean(values("sdfTopologyMismatch"))) if values("sdfTopologyMismatch") else 0.0
    final_topology_regression=float(np.mean(values("finalTopologyMismatch"))) if values("finalTopologyMismatch") else 0.0

    # V9.8.3 promotion gates deliberately match the illustrated target rather than
    # accepting a merely positive delta. Exact geometry must already yield a
    # tight clean renderer before the network is allowed to claim progress.
    renderer_ok=(
        renderer_mean >= 0.50
        and renderer_win >= 0.90
        and renderer_regress <= 0.05
        and renderer_width_ratio <= 1.30
        and renderer_outer <= target_outer + 0.05
        # G0-G4 are clean authored controls and must be essentially halo-free.
        and renderer_core_halo <= 1.0
        # G5 deliberately injects blur/ringing; require strong removal rather
        # than pretending the contaminated input has the same absolute contract.
        and renderer_stress_halo <= 5.0
        and renderer_stress_halo_ratio <= 0.70
        and oracle_topology_regression == 0.0
    )
    sdf_required=max(0.25,renderer_mean*0.75)
    sdf_ok=(
        sdf_mean >= sdf_required
        and sdf_win >= 0.80
        and sdf_regress <= 0.10
        and zero_mean <= 0.50
        and eikonal_mean <= 0.20
        and sdf_width_ratio <= 1.45
        and sdf_topology_regression == 0.0
    )
    final_required=max(0.20,sdf_mean*0.90)
    final_geometry_ok=(
        final_mean >= final_required
        and final_win >= 0.80
        and gate_edge_mean >= 0.05
    )

    final_summaries=[r.get("summary",{}) for r in final_reports if isinstance(r.get("summary"),dict)]
    def summary_mean(key, default=0.0):
        vals=[float(item[key]) for item in final_summaries if item.get(key) is not None and np.isfinite(item.get(key))]
        return float(np.mean(vals)) if vals else float(default)

    edge_width_ratio_mean=summary_mean("edgeWidthRatioMean",1.0)
    fuzz_reduction_mean=summary_mean("fuzzReductionMean",0.0)
    ringing_before=summary_mean("ringingBefore",0.0)
    ringing_after=summary_mean("ringingAfter",0.0)
    topology_regression_fraction=final_topology_regression

    topology_ok=topology_regression_fraction == 0.0
    fuzz_ok=final_width_ratio <= 1.30 and final_outer <= target_outer + 0.08
    halo_ok=(
        final_core_halo <= 1.0
        and final_stress_halo <= 5.0
        and final_stress_halo_ratio <= 0.70
        and ringing_after <= (ringing_before * 1.02 + 1.0e-4)
    )
    double_edge_ok=final_outer <= target_outer + 0.08

    if args.oracle_only:
        verdict="PASS" if renderer_ok else "RENDERER_FAIL"
    elif not renderer_ok:
        verdict="RENDERER_FAIL"
    elif not sdf_ok:
        verdict="SDF_FAIL"
    elif not final_geometry_ok:
        verdict="GATE_FAIL"
    elif not topology_ok:
        verdict="TOPOLOGY_FAIL"
    elif not fuzz_ok:
        verdict="FUZZ_FAIL"
    elif not halo_ok:
        verdict="HALO_FAIL"
    elif not double_edge_ok:
        verdict="DOUBLE_EDGE_FAIL"
    else:
        verdict="PASS"

    summary={
        "schema":"NSAMDR_METRIC_SDF_GEOMETRY_PROOF_V3",
        "verdict":verdict,
        "caseCount":len(rows),
        "rendererProof":{"chamferMetric":"material-midpoint-region-boundary","chamferImprovementMean":renderer_mean,"winFraction":renderer_win,"regressionFraction":renderer_regress,"profileWidthRatioMean":renderer_width_ratio,"outerGradientFractionMean":renderer_outer,"haloOvershoot8bitMax":renderer_halo,"coreHaloOvershoot8bitMax":renderer_core_halo,"stressHaloOvershoot8bitMax":renderer_stress_halo,"stressHaloRatioMax":renderer_stress_halo_ratio,"topologyRegressionFraction":oracle_topology_regression,"pass":renderer_ok},
        "sdfProof":{"chamferImprovementMean":sdf_mean,"requiredChamferImprovement":sdf_required,"winFraction":sdf_win,"regressionFraction":sdf_regress,"profileWidthRatioMean":sdf_width_ratio,"zeroSetRmsPixelsMean":zero_mean,"eikonalErrorMean":eikonal_mean,"topologyRegressionFraction":sdf_topology_regression,"pass":sdf_ok},
        "gateProof":{"chamferImprovementMean":final_mean,"requiredChamferImprovement":final_required,"winFraction":final_win,"boundaryGateEdgeMean":gate_edge_mean,"profileWidthRatioMean":final_width_ratio,"outerGradientFractionMean":final_outer,"haloOvershoot8bitMax":final_halo,"pass":final_geometry_ok},
        "finalQuality":{
            "edgeWidthRatioMean":edge_width_ratio_mean,
            "fuzzReductionMean":fuzz_reduction_mean,
            "ringingBeforeMean":ringing_before,
            "ringingAfterMean":ringing_after,
            "topologyRegressionFraction":topology_regression_fraction,
            "topologyPass":topology_ok,
            "fuzzPass":fuzz_ok,
            "haloPass":halo_ok,
            "doubleEdgePass":double_edge_ok,
            "profileWidthRatioToExactMean":final_width_ratio,
            "outerGradientFractionMean":final_outer,
            "haloOvershoot8bitMax":final_halo,
            "coreHaloOvershoot8bitMax":final_core_halo,
            "stressHaloOvershoot8bitMax":final_stress_halo,
            "stressHaloRatioMax":final_stress_halo_ratio,
        },
        "sdfMetrics":{
            "maePixelsMean":float(np.mean(values("sdfMaePixels"))) if values("sdfMaePixels") else None,
            "zeroSetRmsPixelsMean":float(np.mean(values("sdfZeroSetRmsPixels"))) if values("sdfZeroSetRmsPixels") else None,
            "eikonalErrorMean":float(np.mean(values("sdfEikonalError"))) if values("sdfEikonalError") else None,
            "normalAlignmentErrorMean":float(np.mean(values("sdfNormalAlignmentError"))) if values("sdfNormalAlignmentError") else None,
        },
        "plateauEvidenceDiagnostics":{
            "baselineHaloOvershoot8bitMax":max(values("baselineHaloOvershoot8bit"),default=None),
            "evidenceHaloOvershoot8bitMax":max(values("evidenceHaloOvershoot8bit"),default=None),
            "baselineProfileWidthRatioMean":float(np.mean(values("baselineProfileWidthRatio"))) if values("baselineProfileWidthRatio") else None,
            "evidenceProfileWidthRatioMean":float(np.mean(values("evidenceProfileWidthRatio"))) if values("evidenceProfileWidthRatio") else None,
        },
        "oracleOnly":bool(args.oracle_only),
        "critic":critic_calibration,
        "cases":rows,
        "stageEvidenceDirectory":str(out/"staged_evidence"),
        "reports":[r.get("reportPath") for r in final_reports],
    }
    json_path=out/"synthetic_geometry_audit.json"
    json_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    csv_path=out/"synthetic_geometry_audit.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)

    print("="*72,flush=True)
    print(f"V9.8.3 SIGN-GAUGE METRIC-SDF GEOMETRY PROOF : {verdict}",flush=True)
    print(f"A renderer / GT SDF       : {renderer_mean:+.2%} | wins={renderer_win:.1%}",flush=True)
    print(f"B predicted SDF / GT gate : {sdf_mean:+.2%} | wins={sdf_win:.1%}",flush=True)
    print(f"C predicted SDF + gate    : {final_mean:+.2%} | wins={final_win:.1%}",flush=True)
    print(f"Final gate edge mean      : {gate_edge_mean:.4f}",flush=True)
    print(f"Oracle profile width      : {renderer_width_ratio:.3f}x exact | core halo={renderer_core_halo:.3f}/255 | G5 stress halo={renderer_stress_halo:.3f}/255 | stress ratio={renderer_stress_halo_ratio:.3f}x | topology={oracle_topology_regression:.1%}",flush=True)
    print(f"Predicted SDF quality     : zero={zero_mean:.3f}px | eikonal={eikonal_mean:.3f} | width={sdf_width_ratio:.3f}x",flush=True)
    print(f"Final profile quality     : width={final_width_ratio:.3f}x | core halo={final_core_halo:.3f}/255 | G5 stress halo={final_stress_halo:.3f}/255 | stress ratio={final_stress_halo_ratio:.3f}x",flush=True)
    print(f"Fuzz / edge width         : {fuzz_reduction_mean:+.2%} / {edge_width_ratio_mean:.3f}x",flush=True)
    print(f"Topology regression       : {topology_regression_fraction:.1%}",flush=True)
    print(f"Ringing before -> after   : {ringing_before:.5f} -> {ringing_after:.5f}",flush=True)
    print(f"Evidence/report           : {json_path}",flush=True)
    print("="*72,flush=True)
    return 2 if args.critic=="required" and not critic_calibration.get("calibrated") else 0


if __name__=="__main__":
    raise SystemExit(main())
