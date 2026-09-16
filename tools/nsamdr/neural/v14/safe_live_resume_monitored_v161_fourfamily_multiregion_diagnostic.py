#!/usr/bin/env python3
"""V16.1 Stage 2 loss laboratory on the frozen V16.0 architecture.

This wrapper keeps the V16.0 model, authored four-family dataset, round-robin
sampling, optimizer, learning rate, qualification gates, safe GPU pacing, resume,
and live probes unchanged.  It adds controlled loss-side experiments motivated by
the completed family-difficulty audit:

- bounded baseline-relative map normalisation,
- optional wavelet or focal-Fourier high-frequency supervision,
- optional tangent-space angular normal supervision,
- low-cost per-family gradient-magnitude/direction telemetry.

The default contract is the controlled V16.1A experiment: normalisation ON,
frequency auxiliary OFF, XY-L1 normals unchanged, conflict telemetry ON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
NEURAL_ROOT = HERE.parent
if str(NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURAL_ROOT))

from v14 import safe_live_resume_monitored_fourfamily_multiregion_diagnostic as monitored

resume = monitored.resume
safe = monitored.safe
base = monitored.base

_ORIGINAL_PARSER = safe.parser
_ORIGINAL_RUN = safe.SafeFourFamilyDiagnostic.run
_ORIGINAL_RECORDS = safe.SafeFourFamilyDiagnostic._records
_ORIGINAL_SAFE_VALUES = safe.SafeFourFamilyDiagnostic._safe_values
_ORIGINAL_COMPATIBLE_RESUME = safe.SafeFourFamilyDiagnostic._compatible_resume
_ORIGINAL_MAKE_RUN_DIRECTORY = base.make_run_directory
_ORIGINAL_CANDIDATE_LOSS = base.candidate_loss
_ORIGINAL_CLIP_GRAD_NORM = safe.torch.nn.utils.clip_grad_norm_

_ACTIVE_OPTIONS: dict[str, Any] = {}
_ACTIVE_RUN_DIR: Path | None = None
_ACTIVE_STEP = 0
_REGION_FAMILIES: list[dict[str, str]] = []
_LAST_LOSS_CONTEXT: dict[str, Any] | None = None
_GRADIENT_SKETCHES: dict[str, torch.Tensor] = {}
_PENDING_TELEMETRY: list[dict[str, Any]] = []
_TELEMETRY_AGGREGATE: dict[str, dict[str, float]] = {}


def _parser() -> argparse.ArgumentParser:
    p = _ORIGINAL_PARSER()
    p.description = "NSAMDR V16.1 loss-balanced four-family Stage 2 diagnostic"
    p.add_argument(
        "--loss-normalization",
        choices=("off", "baseline-relative-bounded"),
        default="baseline-relative-bounded",
        help="Balance map/family optimisation pressure using detached baseline difficulty.",
    )
    p.add_argument(
        "--frequency-loss",
        choices=("off", "wavelet", "focal-fourier"),
        default="off",
        help="Optional high-frequency auxiliary objective for V16.1 ablations.",
    )
    p.add_argument("--frequency-loss-weight", type=float, default=0.05)
    p.add_argument(
        "--normal-loss",
        choices=("xy-l1", "angular+l1"),
        default="xy-l1",
        help="Normal-map supervision; angular mode reconstructs tangent-space Z and adds cosine loss.",
    )
    p.add_argument("--angular-normal-weight", type=float, default=0.25)
    p.add_argument(
        "--gradient-conflict",
        choices=("off", "telemetry-only"),
        default="telemetry-only",
        help="Record a small deterministic gradient sketch by family; no gradient surgery is applied.",
    )
    p.add_argument("--normalization-min-weight", type=float, default=0.50)
    p.add_argument("--normalization-max-weight", type=float, default=2.50)
    p.add_argument("--normalization-reference-albedo", type=float, default=0.040)
    p.add_argument("--normalization-reference-normal", type=float, default=0.060)
    p.add_argument("--normalization-reference-material", type=float, default=0.020)
    p.add_argument("--gradient-sketch-values", type=int, default=4096)
    return p


def _options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    minimum = float(args.normalization_min_weight)
    maximum = float(args.normalization_max_weight)
    if not 0.0 < minimum <= maximum:
        raise ValueError("normalization weights must satisfy 0 < min <= max")
    frequency_weight = float(args.frequency_loss_weight)
    angular_weight = float(args.angular_normal_weight)
    if frequency_weight < 0.0 or angular_weight < 0.0:
        raise ValueError("V16.1 auxiliary loss weights must be non-negative")
    references = {
        "albedo": float(args.normalization_reference_albedo),
        "normal": float(args.normalization_reference_normal),
        "material": float(args.normalization_reference_material),
    }
    if min(references.values()) <= 0.0:
        raise ValueError("V16.1 baseline reference errors must be positive")
    return {
        "schema": "NSAMDR_V16_1_LOSS_LAB_V1",
        "normalization": str(args.loss_normalization),
        "normalizationMinWeight": minimum,
        "normalizationMaxWeight": maximum,
        "normalizationReference": references,
        "frequencyLoss": str(args.frequency_loss),
        "frequencyLossWeight": frequency_weight,
        "normalLoss": str(args.normal_loss),
        "angularNormalWeight": angular_weight,
        "gradientConflict": str(args.gradient_conflict),
        "gradientSketchValues": max(256, int(args.gradient_sketch_values)),
    }


def _loss_contract() -> dict[str, Any]:
    return dict(_ACTIVE_OPTIONS)


def _safe_values(self: safe.SafeFourFamilyDiagnostic) -> dict[str, Any]:
    values = _ORIGINAL_SAFE_VALUES(self)
    options = _options_from_args(self.args)
    values["v161LossContract"] = options
    return values


def _records(
    self: safe.SafeFourFamilyDiagnostic,
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    global _REGION_FAMILIES
    train, validation = _ORIGINAL_RECORDS(self, manifest)
    _REGION_FAMILIES = [
        {
            "familyId": str(record.get("family_id") or "unknown"),
            "family": str(record.get("source_asset_name") or record.get("family_id") or "unknown"),
        }
        for record in train
    ]
    return train, validation


def _write_contract(run_dir: Path) -> None:
    try:
        safe._atomic_json(  # noqa: SLF001 - companion diagnostic metadata
            run_dir / "v161_loss_contract.json",
            {
                **_loss_contract(),
                "architectureChanged": False,
                "datasetChanged": False,
                "optimizerChanged": False,
                "qualificationGatesChanged": False,
            },
        )
    except OSError:
        pass


def _make_run_directory(repo_root: Path, mode: str) -> Path:
    global _ACTIVE_RUN_DIR
    path = _ORIGINAL_MAKE_RUN_DIRECTORY(repo_root, mode)
    if mode == "multiregion":
        _ACTIVE_RUN_DIR = path
        _write_contract(path)
    return path


def _compatible_resume(
    self: safe.SafeFourFamilyDiagnostic,
    *,
    manifest: dict[str, Any],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    max_steps: int,
) -> tuple[Path, dict[str, Any]] | None:
    global _ACTIVE_RUN_DIR, _ACTIVE_STEP
    result = _ORIGINAL_COMPATIBLE_RESUME(
        self,
        manifest=manifest,
        train_records=train_records,
        validation_records=validation_records,
        max_steps=max_steps,
    )
    if result is None:
        return None
    checkpoint, payload = result
    state = dict(payload.get("trainingState") or {})
    safe_runtime = dict(state.get("safeRuntime") or {})
    saved = safe_runtime.get("v161LossContract")
    current = _options_from_args(self.args)
    if saved != current:
        explicit = getattr(self.args, "resume_checkpoint", None) is not None
        message = (
            "Stage 2 resume loss contract differs from V16.1; "
            "the optimizer state cannot be reused across this ablation"
        )
        if explicit:
            raise RuntimeError(message)
        print(f"[v16.1-stage2] {message}; starting fresh", flush=True)
        return None
    _ACTIVE_RUN_DIR = Path(str(state.get("runDir") or checkpoint.parent)).resolve()
    _ACTIVE_STEP = int(state.get("step") or 0)
    _write_contract(_ACTIVE_RUN_DIR)
    return checkpoint, payload


def _bounded_weight(reference: float, baseline_error: torch.Tensor) -> torch.Tensor:
    if _ACTIVE_OPTIONS.get("normalization") != "baseline-relative-bounded":
        return baseline_error.new_tensor(1.0)
    minimum = float(_ACTIVE_OPTIONS["normalizationMinWeight"])
    maximum = float(_ACTIVE_OPTIONS["normalizationMaxWeight"])
    return (
        baseline_error.detach().new_tensor(float(reference))
        / baseline_error.detach().clamp_min(1.0e-6)
    ).clamp(minimum, maximum)


def _gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gray = value.float().mean(dim=1, keepdim=True)
    dx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    return dx, dy


def _laplacian(value: torch.Tensor) -> torch.Tensor:
    gray = value.float().mean(dim=1, keepdim=True)
    kernel = gray.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    ).view(1, 1, 3, 3)
    return F.conv2d(gray, kernel, padding=1)


def _pyramid_l1(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    values = [
        F.l1_loss(
            F.avg_pool2d(candidate.float(), scale, scale),
            F.avg_pool2d(target.float(), scale, scale),
        )
        for scale in (2, 4)
    ]
    return sum(values) / len(values)


def _wavelet_detail_loss(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def bands(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = value.float()
        a = value[..., 0::2, 0::2]
        b = value[..., 0::2, 1::2]
        c = value[..., 1::2, 0::2]
        d = value[..., 1::2, 1::2]
        horizontal = (a - b + c - d) * 0.5
        vertical = (a + b - c - d) * 0.5
        diagonal = (a - b - c + d) * 0.5
        return horizontal, vertical, diagonal

    cb = bands(candidate)
    tb = bands(target)
    return sum(F.l1_loss(left, right) for left, right in zip(cb, tb)) / 3.0


def _focal_fourier_loss(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = torch.fft.rfft2(
        candidate.float() - target.float(),
        dim=(-2, -1),
        norm="ortho",
    ).abs()
    detached = error.detach()
    focal = (detached / detached.mean().clamp_min(1.0e-8)).clamp(0.25, 4.0)
    return (error * focal).mean()


def _normal_xyz(value: torch.Tensor) -> torch.Tensor:
    xy = value.float().clamp(-1.0, 1.0)
    z = torch.sqrt((1.0 - (xy * xy).sum(dim=1, keepdim=True)).clamp(0.0, 1.0))
    xyz = torch.cat((xy, z), dim=1)
    return F.normalize(xyz, p=2.0, dim=1, eps=1.0e-8)


def _angular_normal_loss(candidate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    candidate_xyz = _normal_xyz(candidate)
    target_xyz = _normal_xyz(target)
    cosine = (candidate_xyz * target_xyz).sum(dim=1).clamp(-1.0, 1.0)
    return (1.0 - cosine).mean()


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().item())


def _v161_candidate_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Any,
) -> dict[str, torch.Tensor]:
    global _ACTIVE_STEP, _LAST_LOSS_CONTEXT

    if not _ACTIVE_OPTIONS:
        return _ORIGINAL_CANDIDATE_LOSS(outputs, batch, config)

    _ACTIVE_STEP += 1
    ca = outputs["candidate_albedo"].float()
    cn = outputs["candidate_normal"].float()
    cm = outputs["candidate_material"].float()
    ba = outputs["baseline_albedo"].detach().float()
    bn = outputs["baseline_normal"].detach().float()
    bm = outputs["baseline_material"].detach().float()
    ta = batch["target_albedo"].float()
    tn = batch["target_normal"].float()
    tm = batch["target_material"].float()

    baseline_a = F.l1_loss(ba, ta)
    baseline_n = F.l1_loss(bn, tn)
    baseline_m = F.l1_loss(bm, tm)
    refs = dict(_ACTIVE_OPTIONS["normalizationReference"])
    weight_a = _bounded_weight(float(refs["albedo"]), baseline_a)
    weight_n = _bounded_weight(float(refs["normal"]), baseline_n)
    weight_m = _bounded_weight(float(refs["material"]), baseline_m)

    reconstruction = F.l1_loss(ca, ta)
    cgx, cgy = _gradient(ca)
    tgx, tgy = _gradient(ta)
    gradient = (cgx - tgx).abs().mean() + (cgy - tgy).abs().mean()
    laplacian = F.l1_loss(_laplacian(ca), _laplacian(ta))
    pyramid = _pyramid_l1(ca, ta)

    target_residual_a = (ta - ba).clamp(-config.albedo_residual_cap, config.albedo_residual_cap)
    target_residual_n = (tn - bn).clamp(-config.normal_residual_cap, config.normal_residual_cap)
    target_residual_m = (tm - bm).clamp(-config.material_residual_cap, config.material_residual_cap)
    residual_a = F.l1_loss(outputs["predicted_residual_albedo"].float(), target_residual_a)
    residual_n = F.l1_loss(outputs["predicted_residual_normal"].float(), target_residual_n)
    residual_m = F.l1_loss(outputs["predicted_residual_material"].float(), target_residual_m)

    normal_l1 = F.l1_loss(cn, tn)
    angular = _angular_normal_loss(cn, tn)
    material = F.l1_loss(cm, tm)

    frequency = ca.new_tensor(0.0)
    frequency_mode = str(_ACTIVE_OPTIONS["frequencyLoss"])
    if frequency_mode == "wavelet":
        frequency = _wavelet_detail_loss(ca, ta)
    elif frequency_mode == "focal-fourier":
        frequency = _focal_fourier_loss(ca, ta)

    normal_objective = normal_l1
    if _ACTIVE_OPTIONS["normalLoss"] == "angular+l1":
        normal_objective = normal_objective + float(_ACTIVE_OPTIONS["angularNormalWeight"]) * angular

    albedo_objective = (
        reconstruction
        + 0.50 * gradient
        + 0.25 * laplacian
        + 0.25 * pyramid
        + residual_a
    )
    if frequency_mode != "off":
        albedo_objective = (
            albedo_objective
            + float(_ACTIVE_OPTIONS["frequencyLossWeight"]) * frequency
        )

    total = (
        weight_a * albedo_objective
        + 0.25 * weight_n * residual_n
        + 0.25 * weight_m * residual_m
        + 0.25 * weight_n * normal_objective
        + 0.25 * weight_m * material
    )

    original_scale_total = (
        reconstruction
        + 0.50 * gradient
        + 0.25 * laplacian
        + 0.25 * pyramid
        + residual_a
        + 0.25 * residual_n
        + 0.25 * residual_m
        + 0.25 * normal_l1
        + 0.25 * material
    )

    family = {"familyId": "unknown", "family": "unknown"}
    if _REGION_FAMILIES:
        index = (_ACTIVE_STEP - 1) % len(_REGION_FAMILIES)
        family = _REGION_FAMILIES[index]
    _LAST_LOSS_CONTEXT = {
        "schema": "NSAMDR_V16_1_LOSS_TELEMETRY_V1",
        "step": int(_ACTIVE_STEP),
        **family,
        "total": _scalar(total),
        "originalScaleTotal": _scalar(original_scale_total),
        "baselineAlbedoMae": _scalar(baseline_a),
        "baselineNormalMae": _scalar(baseline_n),
        "baselineMaterialMae": _scalar(baseline_m),
        "weightAlbedo": _scalar(weight_a),
        "weightNormal": _scalar(weight_n),
        "weightMaterial": _scalar(weight_m),
        "reconstruction": _scalar(reconstruction),
        "gradient": _scalar(gradient),
        "laplacian": _scalar(laplacian),
        "pyramid": _scalar(pyramid),
        "residualAlbedo": _scalar(residual_a),
        "residualNormal": _scalar(residual_n),
        "residualMaterial": _scalar(residual_m),
        "normalL1": _scalar(normal_l1),
        "normalAngular": _scalar(angular),
        "material": _scalar(material),
        "frequency": _scalar(frequency),
        "frequencyMode": frequency_mode,
        "normalMode": str(_ACTIVE_OPTIONS["normalLoss"]),
    }

    return {
        "total": total,
        "reconstruction": reconstruction,
        "gradient": gradient,
        "laplacian": laplacian,
        "pyramid": pyramid,
        "residual": residual_a + 0.25 * residual_n + 0.25 * residual_m,
        "normal": normal_objective,
        "material": material,
        "frequency": frequency,
        "normal_angular": angular,
        "weight_albedo": weight_a,
        "weight_normal": weight_n,
        "weight_material": weight_m,
    }


def _gradient_sketch(parameters: Iterable[torch.nn.Parameter], maximum: int) -> torch.Tensor | None:
    params = [parameter for parameter in parameters if parameter.grad is not None]
    if not params:
        return None
    per_parameter = max(1, int(maximum) // len(params))
    pieces: list[torch.Tensor] = []
    remaining = int(maximum)
    for parameter in params:
        if remaining <= 0:
            break
        flat = parameter.grad.detach().float().reshape(-1)
        budget = min(per_parameter, remaining, flat.numel())
        if budget <= 0:
            continue
        stride = max(1, int(math.ceil(flat.numel() / float(budget))))
        sample = flat[::stride][:budget].cpu()
        pieces.append(sample)
        remaining -= int(sample.numel())
    if not pieces:
        return None
    vector = torch.cat(pieces)
    norm = torch.linalg.vector_norm(vector)
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 1.0e-12:
        return None
    return vector / norm


def _flush_telemetry() -> None:
    global _PENDING_TELEMETRY
    if _ACTIVE_RUN_DIR is None or not _PENDING_TELEMETRY:
        return
    path = _ACTIVE_RUN_DIR / "v161_loss_telemetry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as handle:
            for row in _PENDING_TELEMETRY:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
    finally:
        _PENDING_TELEMETRY = []


def _update_aggregate(row: dict[str, Any]) -> None:
    family = str(row.get("family") or row.get("familyId") or "unknown")
    aggregate = _TELEMETRY_AGGREGATE.setdefault(
        family,
        {
            "count": 0.0,
            "lossSum": 0.0,
            "gradNormSum": 0.0,
            "negativeCosines": 0.0,
            "cosineCount": 0.0,
            "cosineSum": 0.0,
        },
    )
    aggregate["count"] += 1.0
    aggregate["lossSum"] += float(row.get("total") or 0.0)
    aggregate["gradNormSum"] += float(row.get("gradientNorm") or 0.0)
    cosines = dict(row.get("pairwiseGradientCosine") or {})
    for value in cosines.values():
        scalar = float(value)
        aggregate["cosineCount"] += 1.0
        aggregate["cosineSum"] += scalar
        if scalar < 0.0:
            aggregate["negativeCosines"] += 1.0


def _write_gradient_summary() -> None:
    if _ACTIVE_RUN_DIR is None:
        return
    families: dict[str, Any] = {}
    for family, values in sorted(_TELEMETRY_AGGREGATE.items()):
        count = max(1.0, values["count"])
        cosine_count = max(1.0, values["cosineCount"])
        families[family] = {
            "samples": int(values["count"]),
            "meanLoss": values["lossSum"] / count,
            "meanGradientNorm": values["gradNormSum"] / count,
            "meanCrossFamilyGradientCosine": (
                values["cosineSum"] / cosine_count
                if values["cosineCount"] > 0
                else None
            ),
            "negativeCrossFamilyCosineFraction": (
                values["negativeCosines"] / cosine_count
                if values["cosineCount"] > 0
                else None
            ),
        }
    safe._atomic_json(  # noqa: SLF001
        _ACTIVE_RUN_DIR / "v161_gradient_summary.json",
        {
            "schema": "NSAMDR_V16_1_GRADIENT_SUMMARY_V1",
            "contract": _loss_contract(),
            "gradientSketch": (
                "deterministic sparse parameter-gradient sketch; diagnostic only, "
                "not used to alter optimizer updates"
            ),
            "families": families,
        },
    )


def _clip_grad_norm_(parameters: Iterable[torch.nn.Parameter], max_norm: float, *args: Any, **kwargs: Any):
    global _LAST_LOSS_CONTEXT
    parameter_list = list(parameters)
    pairwise: dict[str, float] = {}
    current_family_id = "unknown"
    sketch: torch.Tensor | None = None
    if _LAST_LOSS_CONTEXT is not None:
        current_family_id = str(_LAST_LOSS_CONTEXT.get("familyId") or "unknown")
    if _ACTIVE_OPTIONS.get("gradientConflict") == "telemetry-only":
        sketch = _gradient_sketch(
            parameter_list,
            int(_ACTIVE_OPTIONS.get("gradientSketchValues", 4096)),
        )
        if sketch is not None:
            for family_id, previous in _GRADIENT_SKETCHES.items():
                if family_id == current_family_id:
                    continue
                count = min(int(sketch.numel()), int(previous.numel()))
                if count > 0:
                    pairwise[family_id] = float(torch.dot(sketch[:count], previous[:count]).item())

    grad_norm = _ORIGINAL_CLIP_GRAD_NORM(parameter_list, max_norm, *args, **kwargs)
    if sketch is not None:
        _GRADIENT_SKETCHES[current_family_id] = sketch

    if _LAST_LOSS_CONTEXT is not None:
        row = dict(_LAST_LOSS_CONTEXT)
        try:
            row["gradientNorm"] = float(grad_norm.detach().float().item())
        except AttributeError:
            row["gradientNorm"] = float(grad_norm)
        row["pairwiseGradientCosine"] = pairwise
        _PENDING_TELEMETRY.append(row)
        _update_aggregate(row)
        if len(_PENDING_TELEMETRY) >= 32:
            _flush_telemetry()
    _LAST_LOSS_CONTEXT = None
    return grad_norm


def _run(self: safe.SafeFourFamilyDiagnostic) -> tuple[int, Path]:
    global _ACTIVE_OPTIONS, _ACTIVE_RUN_DIR, _ACTIVE_STEP
    global _LAST_LOSS_CONTEXT, _GRADIENT_SKETCHES, _PENDING_TELEMETRY
    global _TELEMETRY_AGGREGATE
    _ACTIVE_OPTIONS = _options_from_args(self.args)
    _ACTIVE_RUN_DIR = None
    _ACTIVE_STEP = 0
    _LAST_LOSS_CONTEXT = None
    _GRADIENT_SKETCHES = {}
    _PENDING_TELEMETRY = []
    _TELEMETRY_AGGREGATE = {}
    print("[v16.1-stage2] loss contract: " + json.dumps(_ACTIVE_OPTIONS, sort_keys=True), flush=True)
    try:
        return _ORIGINAL_RUN(self)
    finally:
        _flush_telemetry()
        if _ACTIVE_RUN_DIR is not None:
            _write_contract(_ACTIVE_RUN_DIR)
            _write_gradient_summary()


safe.parser = _parser
safe.SafeFourFamilyDiagnostic._safe_values = _safe_values
safe.SafeFourFamilyDiagnostic._records = _records
safe.SafeFourFamilyDiagnostic._compatible_resume = _compatible_resume
safe.SafeFourFamilyDiagnostic.run = _run
base.make_run_directory = _make_run_directory
base.candidate_loss = _v161_candidate_loss
safe.torch.nn.utils.clip_grad_norm_ = _clip_grad_norm_


def main(argv: list[str] | None = None) -> int:
    return monitored.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
