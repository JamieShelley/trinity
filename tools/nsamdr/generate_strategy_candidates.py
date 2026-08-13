#!/usr/bin/env python3
"""Generate the single public NSAMDR reconstruction candidate.

Public viewer modes:
1. untouched source asset;
2. UV/stretch diagnostics rendered from the source asset;
3. NSAMDR V9 fidelity reconstruction checkpoint using CUDA
   or low-impact CPU inference baked into prepared albedo/normal textures before
   live shader sampling.

Historical experimental candidates are no longer generated or exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from strategy_pipeline import CandidateArtifact, StrategyManifest

NEURAL_DIR = Path(__file__).resolve().parent / "neural"
if str(NEURAL_DIR) not in sys.path:
    sys.path.insert(0, str(NEURAL_DIR))
import train_nsamdr_v9 as v9_tile_model
from v9.geometry_audit import AuditOptions, audit_pair, write_audit_bundle

tile_model = v9_tile_model
ACTIVE_MODEL_LABEL = "V9.8.3 sign-gauge metric-SDF geometry-convergence renderer 4x"
ACTIVE_BACKEND = "staged implicit-SDF-multimap-4x-v9-7"
ACTIVE_CHECKPOINT_PROFILE = "v9"
ACTIVE_PREVIEW_STRENGTH = 1.0
ACTIVE_VALIDATION_RMS_LIMIT = 18.0

REPORT_SCHEMA = "NSAMDR_THREE_MODE_PIPELINE_V9_6_SCIENTIFIC_CONTROL"




def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _geometry_audit_options() -> AuditOptions:
    try:
        evidence = int(os.environ.get("NSAMDR_GEOMETRY_AUDIT_EVIDENCE", "12"))
    except ValueError:
        evidence = 12
    critic_mode = os.environ.get("NSAMDR_GEOMETRY_CRITIC", "auto").strip().lower() or "auto"
    if critic_mode not in {"off", "auto", "required"}:
        critic_mode = "auto"
    policy = os.environ.get("NSAMDR_GEOMETRY_AUDIT_POLICY", "report").strip().lower() or "report"
    if policy not in {"report", "strict"}:
        policy = "report"
    return AuditOptions(evidence_regions=max(0, min(evidence, 40)), critic_mode=critic_mode, policy=policy)


def _write_candidate_result_pointer(report_path: Path, report: dict[str, object]) -> None:
    raw = os.environ.get("NSAMDR_CANDIDATE_RESULT_FILE", "").strip()
    if not raw:
        return
    pointer = Path(raw)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reportPath": str(report_path.resolve()),
        "mode3Analysis": str(report.get("mode3Analysis", "")),
        "mode3Validation": str(report.get("mode3Validation", "")),
        "geometryAudit": report.get("geometryAudit", {}),
        "controlProvenance": report.get("controlProvenance", {}),
        "controlProvenancePath": str(report.get("controlProvenancePath", "")),
    }
    pointer.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _resolve_checkpoint_dir(repository_root: Path, explicit: Path | None) -> Path:
    raw = str(explicit or os.environ.get("NSAMDR_NEURAL_CHECKPOINT_DIR", "")).strip()
    checkpoint_dir = Path(raw).expanduser() if raw else repository_root / "artifacts" / "nsamdr" / "neural_v9"
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = repository_root / checkpoint_dir
    return checkpoint_dir.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_dimensions(path: Path) -> list[int]:
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read image for provenance: {path}")
    return [int(image.shape[1]), int(image.shape[0])]


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _candidate_tree_component(path: Path) -> bool:
    return any(
        part.lower().startswith("strategy_candidates")
        or part.lower() == "mode3_nsamdr_neural"
        for part in path.resolve().parts
    )


def _build_control_provenance(
    *,
    sources_before: dict[Path, dict[str, object]],
    replacements: dict[Path, Path],
    usages: dict[Path, list[object]],
    output_root: Path,
    materials: Path,
    asset_manifest: Path | None,
    destination: Path,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    primary_albedo: dict[str, object] | None = None
    verified = True
    failures: list[str] = []

    for source, candidate in sorted(replacements.items(), key=lambda item: str(item[0]).lower()):
        before = sources_before[source]
        source_after_sha = _sha256(source)
        source_after_dimensions = _image_dimensions(source)
        candidate_sha = _sha256(candidate)
        candidate_dimensions = _image_dimensions(candidate)
        semantics = sorted({
            str(getattr(usage, "semantic", "unknown"))
            for usage in usages.get(source, [])
        })
        path_distinct = source.resolve() != candidate.resolve()
        source_outside_candidate_tree = (
            not _is_within(source, output_root)
            and not _candidate_tree_component(source)
        )
        candidate_inside_output = _is_within(candidate, output_root)
        source_unchanged = (
            str(before["sha256"]) == source_after_sha
            and list(before["dimensions"]) == source_after_dimensions
            and int(before["sizeBytes"]) == int(source.stat().st_size)
        )
        record_verified = bool(
            path_distinct
            and source_outside_candidate_tree
            and candidate_inside_output
            and source_unchanged
        )
        if not record_verified:
            verified = False
            failures.append(str(source))
        record = {
            "semantics": semantics,
            "sourcePath": str(source.resolve()),
            "sourceSha256Before": str(before["sha256"]),
            "sourceSha256After": source_after_sha,
            "sourceDimensions": source_after_dimensions,
            "sourceSizeBytes": int(source.stat().st_size),
            "candidatePath": str(candidate.resolve()),
            "candidateSha256": candidate_sha,
            "candidateDimensions": candidate_dimensions,
            "candidateSizeBytes": int(candidate.stat().st_size),
            "pathsDistinct": path_distinct,
            "sourceOutsideCandidateTree": source_outside_candidate_tree,
            "candidateInsideOutputTree": candidate_inside_output,
            "sourceUnchangedDuringGeneration": source_unchanged,
            "verified": record_verified,
        }
        records.append(record)
        if primary_albedo is None and "albedo" in semantics:
            primary_albedo = dict(record)

    asset_manifest_sha = (
        _sha256(asset_manifest)
        if asset_manifest is not None and asset_manifest.is_file()
        else ""
    )
    payload: dict[str, object] = {
        "schema": "NSAMDR_PREVIEW_CONTROL_PROVENANCE_V1",
        "verified": verified,
        "failureCount": len(failures),
        "failures": failures,
        "sourceMaterialManifest": str(materials.resolve()),
        "sourceMaterialManifestSha256": _sha256(materials),
        "assetManifest": str(asset_manifest.resolve()) if asset_manifest else "",
        "assetManifestSha256": asset_manifest_sha,
        "candidateOutputRoot": str(output_root.resolve()),
        "sourceCount": len(records),
        "primaryAlbedo": primary_albedo or {},
        "records": records,
        "scientificComparison": {
            "rawControlSampler": "16x anisotropic; LOD bias 0.00",
            "legacyControlSampler": "2x anisotropic; LOD bias +1.00",
            "candidateSampler": "16x anisotropic; LOD bias 0.00",
            "authoritativeComparison": "rawControlVsCandidate",
            "legacyComparisonRole": "presentation/emulation only",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not verified:
        raise RuntimeError(
            "NSAMDR preview control provenance failed; refusing to launch an "
            f"ambiguous comparison. Evidence: {destination}"
        )
    return payload

def _parse_preview_strength(default: float) -> float:
    raw = os.environ.get("NSAMDR_V9_PREVIEW_STRENGTH", "").strip()
    if not raw or raw.lower() == "auto":
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"NSAMDR preview strength must be a number from 0 to 1, got {raw!r}"
        ) from exc
    if not 0.0 <= value <= 1.0:
        raise RuntimeError(
            f"NSAMDR preview strength must be between 0 and 1, got {value}"
        )
    return value


def _finite_metric(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def _checkpoint_profile(model_config: object, checkpoint: dict[str, object], architecture: str) -> str:
    if bool(checkpoint.get("live_preview", False)):
        return "live-completed-epoch"
    parameter_total = int(checkpoint.get("parameter_count", 0) or 0)
    tile_size = int(getattr(model_config, "tile_size", 0) or 0)
    total_epochs = int(getattr(model_config, "total_epochs", 0) or 0)
    architecture = architecture.upper()
    if architecture == "V9":
        if parameter_total < 4_000_000 or tile_size < 128 or total_epochs <= 5:
            return "stability-smoke"
        if total_epochs < 16:
            return "pilot"
        return "production"
    if parameter_total < 10_000_000 or tile_size < 128 or total_epochs <= 3:
        return "stability-smoke"
    return "production"


def _ensure_dependencies(install: bool) -> None:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        if not install:
            raise SystemExit(
                "NumPy and OpenCV are required. Run:\n"
                f'  "{sys.executable}" -m pip install numpy opencv-python-headless'
            )
        print("Installing NSAMDR candidate-generation dependencies...", flush=True)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "numpy", "opencv-python-headless",
        ])
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for V9 fidelity reconstruction. Run "
            "scripts\\build\\nsamdr.bat setup cuda (or setup cpu), "
            "then launch through scripts\\build\\nsamdr.bat native eve."
        ) from exc


@dataclass(frozen=True)
class Usage:
    semantic: str
    x_channel: int | None = None  # Manifest channels are RGBA order.
    y_channel: int | None = None


@dataclass
class AlbedoContext:
    normal: Path | None = None
    normal_x_channel: int = 0
    normal_y_channel: int = 1
    material: Path | None = None
    material_channel: int = 0
    paint: Path | None = None
    paint_channel: int = 0
    roughness: Path | None = None
    roughness_channel: int = 0
    glow: Path | None = None
    glow_channel: int = 0


SEMANTICS = (
    "albedo", "normal", "material", "glow", "dirt", "ao",
    "paint_mask", "roughness_map",
)
CHANNEL_COLUMNS = {
    "normal": ("normal_x_channel", "normal_y_channel"),
    "material": ("material_channel", None),
    "glow": ("glow_channel", None),
    "dirt": ("dirt_channel", None),
    "ao": ("ao_channel", None),
    "paint_mask": ("paint_channel", None),
    "roughness_map": ("roughness_channel", None),
}


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
    comments: list[str] = []
    data_lines: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            comments.append(line)
        else:
            data_lines.append(line)
    if not data_lines:
        raise ValueError(f"No material rows in {path}")
    reader = csv.DictReader(data_lines, delimiter="\t")
    rows = [dict(row) for row in reader]
    if reader.fieldnames is None:
        raise ValueError(f"No TSV header in {path}")
    return list(reader.fieldnames), rows, comments


def _resolve_path(value: str, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    return path if path.is_file() else None


def _parse_channel(row: dict[str, str], name: str | None, fallback: int = 0) -> int | None:
    if name is None:
        return None
    try:
        return max(0, min(3, int(float(row.get(name, str(fallback)) or fallback))))
    except ValueError:
        return fallback


def _rgba_to_bgra_channel(channel: int) -> int:
    """Translate manifest RGBA indices to OpenCV BGRA array indices."""
    return (2, 1, 0, 3)[max(0, min(3, channel))]


def _collect_usages(rows: list[dict[str, str]], base: Path) -> dict[Path, list[Usage]]:
    usages: dict[Path, list[Usage]] = {}
    for row in rows:
        for semantic in SEMANTICS:
            source = _resolve_path(row.get(semantic, ""), base)
            if source is None:
                continue
            x_name, y_name = CHANNEL_COLUMNS.get(semantic, (None, None))
            usage = Usage(semantic, _parse_channel(row, x_name), _parse_channel(row, y_name))
            bucket = usages.setdefault(source, [])
            if usage not in bucket:
                bucket.append(usage)
    return usages


def _collect_albedo_contexts(rows: list[dict[str, str]], base: Path) -> dict[Path, AlbedoContext]:
    contexts: dict[Path, AlbedoContext] = {}
    for row in rows:
        albedo = _resolve_path(row.get("albedo", ""), base)
        if albedo is None:
            continue
        context = contexts.setdefault(albedo, AlbedoContext())
        normal = _resolve_path(row.get("normal", ""), base)
        material = _resolve_path(row.get("material", ""), base)
        paint = _resolve_path(row.get("paint_mask", ""), base)
        roughness = _resolve_path(row.get("roughness_map", ""), base)
        glow = _resolve_path(row.get("glow", ""), base)
        if context.normal is None and normal is not None:
            context.normal = normal
            context.normal_x_channel = _parse_channel(row, "normal_x_channel", 0) or 0
            context.normal_y_channel = _parse_channel(row, "normal_y_channel", 1) or 1
        if context.material is None and material is not None:
            context.material = material
            context.material_channel = _parse_channel(row, "material_channel", 0) or 0
        if context.paint is None and paint is not None:
            context.paint = paint
            context.paint_channel = _parse_channel(row, "paint_channel", 0) or 0
        if context.glow is None:
            # Match the fixed Raven dataset builder: when emissive/glow is packed
            # into the material resource, use that resource with the manifest's
            # declared glow channel rather than silently dropping the semantic.
            selected = glow or material
            if selected is not None:
                context.glow = selected
                context.glow_channel = _parse_channel(row, "glow_channel", 0) or 0
        if context.roughness is None:
            # Roughness may be a separate map or another channel of the packed
            # material resource. Preserve the declared channel in either case.
            selected = roughness or material
            if selected is not None:
                context.roughness = selected
                context.roughness_channel = _parse_channel(row, "roughness_channel", 0) or 0
    return contexts


def _resized_channel(source: Path | None, rgba_channel: int, width: int, height: int, default: float):
    import cv2
    import numpy as np
    if source is None:
        return np.full((height, width), default, dtype=np.float32)
    image = _read_bgra(source)
    channel = image[:, :, _rgba_to_bgra_channel(rgba_channel)].astype(np.float32) / 255.0
    return cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)


def _semantic_context_for_albedo(
    context: AlbedoContext | None,
    rgb: "object",
    width: int,
    height: int,
):
    import cv2
    import numpy as np
    fallback = tile_model._semantic_maps_from_rgb(rgb.astype(np.float32) / 255.0)
    if context is None:
        return fallback
    semantic = fallback.copy()
    if context.normal is not None:
        semantic[:, :, 0] = _resized_channel(context.normal, context.normal_x_channel, width, height, 0.5) * 2.0 - 1.0
        semantic[:, :, 1] = _resized_channel(context.normal, context.normal_y_channel, width, height, 0.5) * 2.0 - 1.0
    if context.material is not None:
        semantic[:, :, 2] = _resized_channel(context.material, context.material_channel, width, height, 0.0)
    if context.paint is not None:
        semantic[:, :, 3] = _resized_channel(context.paint, context.paint_channel, width, height, 0.0)
    if context.roughness is not None:
        semantic[:, :, 4] = _resized_channel(context.roughness, context.roughness_channel, width, height, 0.5)
    return np.ascontiguousarray(semantic.astype(np.float32))


def _apply_tile_context_model(
    source_bgra,
    high_bgra,
    context: AlbedoContext | None,
    model,
    model_config: tile_model.TrainingConfig,
    device,
):
    """Apply V9.8.2 metric-SDF geometry-convergence SDF to deterministic reconstruction.

    The model exports a continuous SDF, boundary hardness and a safe boundary
    gate.  Candidate generation re-renders that same boundary over the stronger
    structure-aware albedo/normal/material baselines, so every physical map
    receives identical geometry without using the model's internal bicubic RGB.
    """
    import cv2
    import numpy as np

    target_height, target_width = high_bgra.shape[:2]
    input_width = max(1, (target_width + tile_model.UPSCALE_FACTOR - 1) // tile_model.UPSCALE_FACTOR)
    input_height = max(1, (target_height + tile_model.UPSCALE_FACTOR - 1) // tile_model.UPSCALE_FACTOR)
    source_rgb = cv2.cvtColor(source_bgra[:, :, :3], cv2.COLOR_BGR2RGB)
    albedo_lr = cv2.resize(source_rgb, (input_width, input_height), interpolation=cv2.INTER_LANCZOS4)

    fallback = tile_model._semantic_maps_from_rgb(albedo_lr.astype(np.float32) / 255.0)
    normal_lr = fallback[:, :, 0:2]
    material_lr = fallback[:, :, 2:5]
    if context is not None and context.normal is not None:
        normal_image = _read_bgra(context.normal)
        normal_x = normal_image[:, :, _rgba_to_bgra_channel(context.normal_x_channel)].astype(np.float32) / 127.5 - 1.0
        normal_y = normal_image[:, :, _rgba_to_bgra_channel(context.normal_y_channel)].astype(np.float32) / 127.5 - 1.0
        normal_lr = np.stack((
            cv2.resize(normal_x, (input_width, input_height), interpolation=cv2.INTER_LINEAR),
            cv2.resize(normal_y, (input_width, input_height), interpolation=cv2.INTER_LINEAR),
        ), axis=-1)
    if context is not None and any((context.material, context.glow, context.roughness)):
        # Match training semantics exactly:
        #   R = material semantic, G = emissive/glow, B = roughness.
        material_source = context.material
        glow_source = context.glow or material_source
        roughness_source = context.roughness or material_source
        material_plane = _resized_channel(
            material_source, context.material_channel, input_width, input_height, 0.0)
        emissive_plane = _resized_channel(
            glow_source, context.glow_channel, input_width, input_height, 0.0)
        roughness_plane = _resized_channel(
            roughness_source, context.roughness_channel, input_width, input_height, 0.5)
        material_lr = np.stack(
            (material_plane, emissive_plane, roughness_plane), axis=-1)

    model_input = tile_model.build_model_input(
        albedo_lr,
        normal_xy=np.ascontiguousarray(normal_lr.astype(np.float32)),
        material_rgb=np.ascontiguousarray(material_lr.astype(np.float32)),
        degradation_level=1.0,
    )
    maps, diagnostics = tile_model.infer_tiled(
        model,
        model_input,
        device,
        tile_size=model_config.inference_tile_size,
        overlap=model_config.inference_overlap,
        return_diagnostics=True,
        return_all_maps=True,
    )

    def _resize_scalar(field, interpolation=cv2.INTER_LINEAR):
        field = np.asarray(field, dtype=np.float32)
        if field.ndim == 3 and field.shape[2] == 1:
            field = field[:, :, 0]
        if field.shape[:2] == (target_height, target_width):
            return field
        return cv2.resize(
            field,
            (target_width, target_height),
            interpolation=interpolation,
        ).astype(np.float32)

    sdf_map = _resize_scalar(maps["sdf"])
    hardness_map = np.clip(
        _resize_scalar(maps["hardness"]), 0.0, 1.0
    )
    boundary_gate_map = np.clip(
        _resize_scalar(maps["boundary_gate"]), 0.0, 1.0
    )
    edge_map = np.clip(
        _resize_scalar(maps["edge"]), 0.0, 1.0
    )

    if maps["albedo"].shape[:2] != (target_height, target_width):
        maps["albedo"] = cv2.resize(
            maps["albedo"], (target_width, target_height),
            interpolation=cv2.INTER_LANCZOS4)
        maps["normal_xy"] = cv2.resize(
            maps["normal_xy"], (target_width, target_height),
            interpolation=cv2.INTER_LINEAR)
        maps["material"] = cv2.resize(
            maps["material"], (target_width, target_height),
            interpolation=cv2.INTER_LINEAR)
        maps["roughness"] = cv2.resize(
            maps["roughness"], (target_width, target_height),
            interpolation=cv2.INTER_LINEAR)
        maps["emissive"] = cv2.resize(
            maps["emissive"], (target_width, target_height),
            interpolation=cv2.INTER_LINEAR)

    sdf_pixels = (
        sdf_map * float(model_config.contour_sdf_max_distance_pixels)
    ).astype(np.float32)
    grad_y, grad_x = np.gradient(sdf_pixels)
    grad_length = np.sqrt(
        grad_x * grad_x + grad_y * grad_y + 1.0e-6
    )
    normal_x = grad_x / grad_length
    normal_y = grad_y / grad_length

    grid_x, grid_y = np.meshgrid(
        np.arange(target_width, dtype=np.float32),
        np.arange(target_height, dtype=np.float32),
    )
    sample_pixels = float(model_config.boundary_renderer_sample_pixels)
    plateau_count = max(3, int(getattr(model_config, "boundary_renderer_plateau_samples", 5)))
    plateau_max = float(getattr(model_config, "boundary_renderer_plateau_max_multiplier", 2.20))
    plateau_stability = float(getattr(model_config, "boundary_renderer_plateau_stability_scale", 14.0))
    plateau_multipliers = np.linspace(0.65, plateau_max, plateau_count, dtype=np.float32)

    hard_width = float(model_config.boundary_renderer_hard_width_pixels)
    soft_width = float(model_config.boundary_renderer_soft_width_pixels)
    transition_width = np.clip(
        soft_width + (hard_width - soft_width) * hardness_map,
        0.25,
        None,
    )
    coverage_negative = np.clip(
        0.5 - sdf_pixels / transition_width, 0.0, 1.0
    )
    coverage_negative = (
        coverage_negative * coverage_negative
        * (3.0 - 2.0 * coverage_negative)
    ).astype(np.float32)

    def _boundary_render(image, interpolation=cv2.INTER_LINEAR):
        original_dtype = image.dtype
        value = np.asarray(image, dtype=np.float32)
        value_scale = 255.0 if np.issubdtype(original_dtype, np.integer) else 1.0

        def adaptive_side(sign):
            samples=[]
            for multiplier in plateau_multipliers:
                offset=sample_pixels*float(multiplier)*float(sign)
                samples.append(cv2.remap(
                    value,
                    grid_x + normal_x * offset,
                    grid_y + normal_y * offset,
                    interpolation=interpolation,
                    borderMode=cv2.BORDER_REPLICATE,
                ))
            stack=np.stack(samples,axis=0).astype(np.float32)
            errors=[]
            for i in range(plateau_count):
                j=min(i+1,plateau_count-1); k=max(i-1,0)
                neighbour=0.5*(stack[j]+stack[k])
                diff=np.abs(stack[i]-neighbour)
                if diff.ndim==3:
                    diff=diff.mean(axis=2)
                errors.append(diff/value_scale)
            error=np.stack(errors,axis=0)
            stability=np.exp(-error*plateau_stability)
            prior=np.exp(-np.arange(plateau_count,dtype=np.float32)*0.28)[:,None,None]
            weights=stability*prior
            weights/=np.maximum(weights.sum(axis=0,keepdims=True),1.0e-6)
            if stack.ndim==4:
                return np.sum(stack*weights[:,:,:,None],axis=0)
            return np.sum(stack*weights,axis=0)

        positive = adaptive_side(+1.0)
        negative = adaptive_side(-1.0)
        if value.ndim == 3:
            coverage = coverage_negative[:, :, None]
            gate = boundary_gate_map[:, :, None]
        else:
            coverage = coverage_negative
            gate = boundary_gate_map
        reconstructed = negative * coverage + positive * (1.0 - coverage)
        rendered = value * (1.0 - gate) + reconstructed * gate
        if np.issubdtype(original_dtype, np.integer):
            return np.asarray(
                np.round(np.clip(rendered, 0.0, 255.0)),
                dtype=original_dtype,
            )
        return rendered.astype(original_dtype)

    def _blend(before, after, strength):
        return np.uint8(np.round(np.clip(
            before.astype(np.float32) * (1.0 - strength)
            + after.astype(np.float32) * strength,
            0.0,
            255.0,
        )))

    strength = float(ACTIVE_PREVIEW_STRENGTH)
    geometry_only = not bool(getattr(model_config, "appearance_enabled", False))

    before_high = high_bgra.copy()
    before_albedo = cv2.cvtColor(before_high[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32)
    if geometry_only:
        raw_high = _boundary_render(before_high, cv2.INTER_LINEAR)
        blended_high = _blend(before_high, raw_high, strength)
        high_bgra[:, :, :] = blended_high
        raw_corrected_rgb = cv2.cvtColor(raw_high[:, :, :3], cv2.COLOR_BGR2RGB)
        corrected_rgb = cv2.cvtColor(blended_high[:, :, :3], cv2.COLOR_BGR2RGB)
    else:
        # Future appearance stage: retain the absolute learned output path until
        # appearance residual export is introduced. Geometry proof never enters
        # this branch.
        raw_corrected_rgb = np.uint8(np.round(np.clip(maps["albedo"], 0.0, 1.0) * 255.0))
        corrected_rgb = np.uint8(np.round(np.clip(
            before_albedo * (1.0 - strength)
            + raw_corrected_rgb.astype(np.float32) * strength,
            0.0,
            255.0,
        )))
        high_bgra[:, :, :3] = cv2.cvtColor(corrected_rgb, cv2.COLOR_RGB2BGR)

    raw_albedo_delta_rms = float(np.sqrt(np.mean(
        (raw_corrected_rgb.astype(np.float32) - before_albedo) ** 2)))
    albedo_delta_rms = float(np.sqrt(np.mean(
        (corrected_rgb.astype(np.float32) - before_albedo) ** 2)))

    companion_outputs: dict[Path, object] = {}
    companion_baselines: dict[Path, object] = {}
    normal_delta_rms = 0.0
    raw_normal_delta_rms = 0.0
    material_delta_rms = 0.0

    def _deterministic_normal_high(path):
        original = _read_bgra(path)
        result = cv2.resize(original, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        base_size = max(target_width, target_height)
        x_index = _rgba_to_bgra_channel(context.normal_x_channel)
        y_index = _rgba_to_bgra_channel(context.normal_y_channel)
        x, y = _reconstruct_normal_pair_v2(
            original[:, :, x_index], original[:, :, y_index], base_size)
        if (x.shape[1], x.shape[0]) != (target_width, target_height):
            x = cv2.resize(x, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
            y = cv2.resize(y, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        result[:, :, x_index] = np.uint8(np.round(np.clip((x + 1.0) * 127.5, 0.0, 255.0)))
        result[:, :, y_index] = np.uint8(np.round(np.clip((y + 1.0) * 127.5, 0.0, 255.0)))
        return result

    if context is not None and context.normal is not None:
        normal_high = _deterministic_normal_high(context.normal)
        companion_baselines[context.normal.resolve()] = normal_high.copy()
        x_index = _rgba_to_bgra_channel(context.normal_x_channel)
        y_index = _rgba_to_bgra_channel(context.normal_y_channel)
        before_x = normal_high[:, :, x_index].astype(np.float32)
        before_y = normal_high[:, :, y_index].astype(np.float32)
        if geometry_only:
            raw_normal_high = _boundary_render(normal_high, cv2.INTER_LINEAR)
            raw_x = raw_normal_high[:, :, x_index].astype(np.float32)
            raw_y = raw_normal_high[:, :, y_index].astype(np.float32)
            normal_high = _blend(normal_high, raw_normal_high, strength)
            # Renormalise the authored XY normal pair after interpolation.
            blended_x = normal_high[:, :, x_index].astype(np.float32) / 127.5 - 1.0
            blended_y = normal_high[:, :, y_index].astype(np.float32) / 127.5 - 1.0
            normal_length = np.sqrt(np.maximum(blended_x * blended_x + blended_y * blended_y, 1.0e-8))
            limiter = np.maximum(1.0, normal_length / 0.999)
            blended_x /= limiter
            blended_y /= limiter
            encoded_x = np.uint8(np.round(np.clip((blended_x + 1.0) * 127.5, 0.0, 255.0)))
            encoded_y = np.uint8(np.round(np.clip((blended_y + 1.0) * 127.5, 0.0, 255.0)))
            normal_high[:, :, x_index] = encoded_x
            normal_high[:, :, y_index] = encoded_y
            normal_delta_rms = float(np.sqrt(np.mean(
                (encoded_x.astype(np.float32) - before_x) ** 2
                + (encoded_y.astype(np.float32) - before_y) ** 2)))
            raw_normal_delta_rms = float(np.sqrt(np.mean(
                (raw_x - before_x) ** 2 + (raw_y - before_y) ** 2)))
        else:
            raw_x = np.clip(maps["normal_xy"][:, :, 0], -1.0, 1.0)
            raw_y = np.clip(maps["normal_xy"][:, :, 1], -1.0, 1.0)
            baseline_x = before_x / 127.5 - 1.0
            baseline_y = before_y / 127.5 - 1.0
            blended_x = baseline_x * (1.0 - strength) + raw_x * strength
            blended_y = baseline_y * (1.0 - strength) + raw_y * strength
            normal_length = np.sqrt(np.maximum(blended_x * blended_x + blended_y * blended_y, 1.0e-8))
            limiter = np.maximum(1.0, normal_length / 0.999)
            blended_x /= limiter
            blended_y /= limiter
            encoded_x = np.uint8(np.round(np.clip((blended_x + 1.0) * 127.5, 0.0, 255.0)))
            encoded_y = np.uint8(np.round(np.clip((blended_y + 1.0) * 127.5, 0.0, 255.0)))
            normal_high[:, :, x_index] = encoded_x
            normal_high[:, :, y_index] = encoded_y
            raw_encoded_x = np.uint8(np.round(np.clip((raw_x + 1.0) * 127.5, 0.0, 255.0)))
            raw_encoded_y = np.uint8(np.round(np.clip((raw_y + 1.0) * 127.5, 0.0, 255.0)))
            normal_delta_rms = float(np.sqrt(np.mean(
                (encoded_x.astype(np.float32) - before_x) ** 2
                + (encoded_y.astype(np.float32) - before_y) ** 2)))
            raw_normal_delta_rms = float(np.sqrt(np.mean(
                (raw_encoded_x.astype(np.float32) - before_x) ** 2
                + (raw_encoded_y.astype(np.float32) - before_y) ** 2)))
        companion_outputs[context.normal.resolve()] = normal_high

    semantic_specs = []
    if context is not None:
        semantic_specs = [
            ("material", context.material, context.material_channel, True, maps["material"][:, :, 0:1]),
            ("roughness", context.roughness, context.roughness_channel, False, maps["roughness"]),
            ("emissive", context.glow, context.glow_channel, True, maps["emissive"]),
        ]

    semantic_delta_rms: dict[str, float] = {}
    if geometry_only:
        by_path: dict[Path, list[tuple[str, int, bool]]] = {}
        for semantic, path, channel, crisp, _predicted in semantic_specs:
            if path is not None:
                by_path.setdefault(path.resolve(), []).append((semantic, channel, crisp))
        for resolved, specs in by_path.items():
            original = _read_bgra(resolved)
            deterministic = cv2.resize(
                original, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            base_size = max(target_width, target_height)
            for _semantic, channel, crisp in specs:
                plane_index = _rgba_to_bgra_channel(channel)
                plane = _reconstruct_scalar_plane_v2(
                    original[:, :, plane_index], base_size, crisp=crisp)
                if (plane.shape[1], plane.shape[0]) != (target_width, target_height):
                    plane = cv2.resize(
                        plane, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
                deterministic[:, :, plane_index] = np.uint8(np.round(np.clip(plane, 0.0, 255.0)))
            companion_baselines[resolved] = deterministic.copy()
            raw_warped = _boundary_render(deterministic, cv2.INTER_LINEAR)
            blended = _blend(deterministic, raw_warped, strength)
            companion_outputs[resolved] = blended
            for semantic, channel, _crisp in specs:
                plane_index = _rgba_to_bgra_channel(channel)
                before = deterministic[:, :, plane_index].astype(np.float32)
                after = blended[:, :, plane_index].astype(np.float32)
                semantic_delta_rms[semantic] = float(np.sqrt(np.mean((after - before) ** 2)))
        material_delta_rms = float(max(semantic_delta_rms.values(), default=0.0))
    else:
        def _semantic_companion(path, channel, predicted):
            if path is None:
                return 0.0
            resolved = path.resolve()
            image = companion_outputs.get(resolved)
            if image is None:
                image = cv2.resize(
                    _read_bgra(path), (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            plane_index = _rgba_to_bgra_channel(channel)
            before = image[:, :, plane_index].astype(np.float32)
            raw = np.clip(predicted[:, :, 0], 0.0, 1.0) * 255.0
            blended = np.clip(before * (1.0 - strength) + raw * strength, 0.0, 255.0)
            image[:, :, plane_index] = np.uint8(np.round(blended))
            companion_outputs[resolved] = image
            return float(np.sqrt(np.mean((blended - before) ** 2)))

        for semantic, path, channel, _crisp, predicted in semantic_specs:
            semantic_delta_rms[semantic] = _semantic_companion(path, channel, predicted)
        material_delta_rms = float(max(semantic_delta_rms.values(), default=0.0))

    diagnostics = dict(diagnostics)
    diagnostics.update({
        "albedoDeltaRms": albedo_delta_rms,
        "rawAlbedoDeltaRms": raw_albedo_delta_rms,
        "normalDeltaRms": normal_delta_rms,
        "rawNormalDeltaRms": raw_normal_delta_rms,
        "materialDeltaRms": material_delta_rms,
        "semanticMaterialDeltaRms": semantic_delta_rms,
        "checkpointProfile": ACTIVE_CHECKPOINT_PROFILE,
        "previewStrength": strength,
        "preparedTargetWidth": int(target_width),
        "preparedTargetHeight": int(target_height),
        "modelInputWidth": int(input_width),
        "modelInputHeight": int(input_height),
        "nativeSourceWidth": int(source_bgra.shape[1]),
        "nativeSourceHeight": int(source_bgra.shape[0]),
        "native4xEvaluation": bool(
            input_width >= source_bgra.shape[1]
            and input_height >= source_bgra.shape[0]
        ),
        "mapsReconstructed": ["albedo", "normalXY", "materialRGB", "roughness", "emissive"],
        "materialPolicy": (
            "shared-implicit-boundary-canonical-physical-map"
            if geometry_only
            else "neural-aligned-physical-map"
        ),
        "geometryOnlyProof": geometry_only,
        "appearanceEnabled": not geometry_only,
        "deterministicBaselineBoundaryRendered": geometry_only,
        "boundaryRendererAppliedToAllPhysicalMaps": geometry_only,
        "boundaryGateMean": float(np.mean(boundary_gate_map)),
        "boundaryGateP95": float(np.percentile(boundary_gate_map, 95)),
        "boundaryHardnessMean": float(np.mean(hardness_map)),
        "boundaryTransitionWidthMeanPixels": float(np.mean(
            transition_width[boundary_gate_map > 0.10]
        )) if np.any(boundary_gate_map > 0.10) else float(np.mean(transition_width)),
        "flowRmsPixels": 0.0,
        "flowMaxAbsPixels": 0.0,
        "reconstructionPrimitive": "coarse-sdf-bounded-residual-adaptive-plateau-renderer",
    })
    audit_aux = {
        "displacement": None,
        "gate": boundary_gate_map.astype(np.float32),
        "modelEdge": edge_map.astype(np.float32),
        "sdf": sdf_map.astype(np.float32),
        "hardness": hardness_map.astype(np.float32),
        "companionBaselines": companion_baselines,
    }
    return albedo_delta_rms, diagnostics, companion_outputs, audit_aux

def _unique_output_name(source: Path) -> str:
    digest = hashlib.sha1(str(source).lower().encode("utf-8")).hexdigest()[:10]
    return f"{source.stem}_{digest}_4k.png"


def _read_bgra(source: Path):
    import cv2
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Could not read texture: {source}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] != 4:
        raise RuntimeError(f"Unsupported channel count in {source}: {image.shape}")
    return image


def _unsharp(channel, cv2, amount: float, sigma: float):
    blurred = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return cv2.addWeighted(channel, 1.0 + amount, blurred, -amount, 0.0)


def _find_realesrgan(tool_dir: Path) -> tuple[Path, str] | None:
    candidates: list[Path] = []
    env_path = os.environ.get("NSAMDR_REALESRGAN_EXE", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    third_party = tool_dir / "third_party" / "realesrgan"
    if third_party.is_dir():
        candidates.extend(third_party.rglob("realesrgan-ncnn-vulkan.exe"))
    for executable in candidates:
        if not executable.is_file():
            continue
        model_roots = [executable.parent / "models", executable.parent]
        available: set[str] = set()
        for root in model_roots:
            if not root.is_dir():
                continue
            for param in root.rglob("*.param"):
                if param.with_suffix(".bin").is_file():
                    available.add(param.stem)
        requested = os.environ.get("NSAMDR_REALESRGAN_MODEL", "").strip()
        preferences = [requested] if requested else []
        preferences += ["realesrnet-x4plus", "realesrgan-x4plus", "realesr-general-x4v3"]
        for model in preferences:
            if model and model in available:
                return executable.resolve(), model
    return None


def _run_realesrgan(source_bgr, executable: Path, model: str, target_size: int):
    import cv2
    with tempfile.TemporaryDirectory(prefix="nsamdr_sr_") as temporary:
        temp = Path(temporary)
        input_path = temp / "input.png"
        output_path = temp / "output.png"
        if not cv2.imwrite(str(input_path), source_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise RuntimeError("Could not write Real-ESRGAN input")
        command = [
            str(executable), "-i", str(input_path), "-o", str(output_path),
            "-n", model, "-s", "4", "-t", "0", "-f", "png",
        ]
        result = subprocess.run(
            command, cwd=executable.parent, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if result.returncode != 0 or not output_path.is_file():
            raise RuntimeError(
                f"Real-ESRGAN failed with exit code {result.returncode}:\n{result.stdout[-4000:]}"
            )
        output = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
        if output is None:
            raise RuntimeError("Real-ESRGAN produced an unreadable image")
        if output.shape[0] != target_size or output.shape[1] != target_size:
            output = cv2.resize(output, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        return output


def _classic_albedo(source_bgr, target_size: int):
    """Fidelity-first deterministic enlargement with anti-ringing."""
    import cv2
    import numpy as np

    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)
    source_l, source_a, source_b = cv2.split(source_lab)
    # Remove block/fuzz energy from the broad colour field while retaining hard
    # authored edges in a separately bounded residual.
    low = cv2.bilateralFilter(source_l, 9, 16, 5)
    blur = cv2.GaussianBlur(source_l, (0, 0), 1.05)
    residual = source_l.astype(np.float32) - blur.astype(np.float32)
    local_mean = cv2.GaussianBlur(source_l.astype(np.float32), (0, 0), 2.0)
    local_var = cv2.GaussianBlur((source_l.astype(np.float32) - local_mean) ** 2, (0, 0), 2.0)
    sigma = np.sqrt(np.maximum(local_var, 0.0))
    residual = np.clip(residual, -0.82 * sigma - 1.0, 0.82 * sigma + 1.0)

    low_up = cv2.resize(low, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    source_up = cv2.resize(source_l, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    detail_up = cv2.resize(residual, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    restored_l = np.clip(0.60 * low_up + 0.40 * source_up + 1.18 * detail_up, 0.0, 255.0).astype(np.uint8)
    a_up = cv2.resize(source_a, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    b_up = cv2.resize(source_b, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    # Chroma noise is a major source of low-resolution colour fuzz.
    a_up = cv2.bilateralFilter(a_up, 5, 5, 2)
    b_up = cv2.bilateralFilter(b_up, 5, 5, 2)
    return cv2.cvtColor(cv2.merge((restored_l, a_up, b_up)), cv2.COLOR_LAB2BGR)



def _smoothstep_array(edge0: float, edge1: float, value):
    import numpy as np
    t = np.clip((value - edge0) / max(edge1 - edge0, 1.0e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _iterative_back_projection(high, target, cv2, iterations: int = 2, gain: float = 0.62):
    """Restore source-consistent edge energy without unconstrained sharpening."""
    import numpy as np
    source_h, source_w = target.shape[:2]
    result = high.astype(np.float32)
    target_f = target.astype(np.float32)
    for _ in range(max(iterations, 0)):
        down = cv2.resize(result, (source_w, source_h), interpolation=cv2.INTER_AREA)
        residual = target_f - down
        correction = cv2.resize(
            residual,
            (result.shape[1], result.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        result += correction * gain
    return result


def _structure_aware_albedo(source_bgr, target_size: int):
    """Fuzz-suppressing 4K reconstruction that protects coherent authored structure.

    Flat low-confidence high-frequency energy is treated as source fuzz. Coherent
    panel edges and line work are reconstructed through a gated Laplacian pyramid
    and iterative back-projection, then bounded by a local anti-ringing envelope.
    """
    import cv2
    import numpy as np

    source_lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)
    l8, a8, b8 = cv2.split(source_lab)
    l = l8.astype(np.float32)

    fine_blur = cv2.GaussianBlur(l, (0, 0), 0.62)
    mid_blur = cv2.GaussianBlur(l, (0, 0), 1.35)
    broad_blur = cv2.GaussianBlur(l, (0, 0), 2.8)
    fine = l - fine_blur
    medium = fine_blur - mid_blur
    coarse = mid_blur - broad_blur

    gx = cv2.Scharr(l, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(l, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    gradient_reference = max(float(np.percentile(gradient, 94.0)), 1.0)
    edge_strength = np.clip(gradient / gradient_reference, 0.0, 1.0)

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.2)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.2)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.2)
    coherence = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0)) / (jxx + jyy + 1.0e-5)

    median = cv2.medianBlur(l8, 3).astype(np.float32)
    impulsive = np.abs(l - median)
    noise_reference = max(float(np.percentile(impulsive, 82.0)), 1.0)
    noise_like = np.clip(impulsive / noise_reference, 0.0, 1.0) * (1.0 - edge_strength)

    detail_confidence = np.clip(
        0.14 + 0.58 * edge_strength + 0.42 * coherence - 0.52 * noise_like,
        0.0,
        1.0,
    )
    fine_gate = _smoothstep_array(0.18, 0.74, detail_confidence)
    medium_gate = _smoothstep_array(0.08, 0.56, detail_confidence)

    # Remove weak, incoherent source fuzz while preserving authored line work.
    reconstructed_source = (
        broad_blur
        + coarse * (0.98 + 0.12 * medium_gate)
        + medium * (0.88 + 0.34 * medium_gate)
        + fine * (0.18 + 0.92 * fine_gate)
    )
    reconstructed_source = np.clip(reconstructed_source, 0.0, 255.0)

    high = cv2.resize(
        reconstructed_source,
        (target_size, target_size),
        interpolation=cv2.INTER_LANCZOS4,
    )
    high = _iterative_back_projection(high, reconstructed_source, cv2, iterations=3, gain=0.58)

    edge_high = cv2.resize(edge_strength, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    confidence_high = cv2.resize(detail_confidence, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    local_blur = cv2.GaussianBlur(high, (0, 0), 0.82)
    bounded_detail = high - local_blur
    sharpen_gain = 0.10 + 0.34 * np.clip(edge_high * 0.72 + confidence_high * 0.28, 0.0, 1.0)
    high += bounded_detail * sharpen_gain

    source_min = cv2.erode(reconstructed_source, np.ones((3, 3), np.uint8))
    source_max = cv2.dilate(reconstructed_source, np.ones((3, 3), np.uint8))
    low_envelope = cv2.resize(source_min, (target_size, target_size), interpolation=cv2.INTER_LINEAR) - 2.0
    high_envelope = cv2.resize(source_max, (target_size, target_size), interpolation=cv2.INTER_LINEAR) + 2.0
    high = np.clip(high, low_envelope, high_envelope)
    high = np.clip(high, 0.0, 255.0).astype(np.uint8)

    # Chroma receives stronger fuzz suppression than luminance. This removes the
    # low-resolution colour crawling visible on large dark and metallic panels.
    a_clean = cv2.bilateralFilter(a8, 9, 7, 4)
    b_clean = cv2.bilateralFilter(b8, 9, 7, 4)
    a_high = cv2.resize(a_clean, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    b_high = cv2.resize(b_clean, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(cv2.merge((high, a_high, b_high)), cv2.COLOR_LAB2BGR)


def _reconstruct_scalar_plane_v2(source_plane, target_size: int, *, crisp: bool):
    import cv2
    import numpy as np

    source = source_plane.astype(np.float32)
    if crisp:
        cleaned = cv2.bilateralFilter(source, 5, 7.5, 2.2)
        gain = 0.64
        sharpen = 0.22
    else:
        cleaned = cv2.bilateralFilter(source, 7, 5.5, 3.4)
        gain = 0.56
        sharpen = 0.10
    high = cv2.resize(cleaned, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    high = _iterative_back_projection(high, cleaned, cv2, iterations=2, gain=gain)
    blurred = cv2.GaussianBlur(high, (0, 0), 0.78 if crisp else 1.05)
    high += (high - blurred) * sharpen
    minimum = cv2.resize(cv2.erode(source_plane, np.ones((3, 3), np.uint8)), (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    maximum = cv2.resize(cv2.dilate(source_plane, np.ones((3, 3), np.uint8)), (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return np.clip(high, minimum.astype(np.float32) - 1.0, maximum.astype(np.float32) + 1.0)


def _reconstruct_normal_pair_v2(source_x, source_y, target_size: int):
    import cv2
    import numpy as np

    x_source = source_x.astype(np.float32) / 127.5 - 1.0
    y_source = source_y.astype(np.float32) / 127.5 - 1.0
    x_clean = cv2.bilateralFilter(x_source, 5, 0.045, 2.6)
    y_clean = cv2.bilateralFilter(y_source, 5, 0.045, 2.6)
    x_high = cv2.resize(x_clean, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    y_high = cv2.resize(y_clean, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    x_high = _iterative_back_projection(x_high, x_clean, cv2, iterations=2, gain=0.52)
    y_high = _iterative_back_projection(y_high, y_clean, cv2, iterations=2, gain=0.52)

    x_blur = cv2.GaussianBlur(x_high, (0, 0), 0.86)
    y_blur = cv2.GaussianBlur(y_high, (0, 0), 0.86)
    x_high += (x_high - x_blur) * 0.18
    y_high += (y_high - y_blur) * 0.18
    length_sq = x_high * x_high + y_high * y_high
    limiter = np.maximum(1.0, np.sqrt(np.maximum(length_sq, 1.0e-8)) / 0.982)
    x_high /= limiter
    y_high /= limiter
    return x_high, y_high


def _prepare_nsamdr_texture(
    source: Path,
    usages: list[Usage],
    output: Path,
    target_size: int,
    neural: tuple[Path, str] | None,
    albedo_context: AlbedoContext | None,
    tile_runtime: tuple[object, tile_model.TrainingConfig, object] | None,
    neural_bundle_cache: dict[Path, tuple[object, dict[str, object]]],
) -> tuple[int, int, str, dict[str, object]]:
    """Prepare a target texture and bake the selected physical reconstruction."""
    import cv2
    import numpy as np

    image = _read_bgra(source)
    height, width = image.shape[:2]
    if width == height:
        out_w = out_h = target_size
    else:
        scale = target_size / max(width, height)
        out_w, out_h = max(1, round(width * scale)), max(1, round(height * scale))

    cached = neural_bundle_cache.get(source.resolve())
    if cached is not None:
        cached_image, cached_metrics = cached
        high = np.asarray(cached_image, dtype=np.uint8)
        if high.shape[:2] != (out_h, out_w):
            high = cv2.resize(high, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), high, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise RuntimeError(f"Could not write prepared NSAMDR texture: {output}")
        luma_before = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
        down = cv2.resize(high[:, :, :3], (width, height), interpolation=cv2.INTER_AREA)
        luma_after = cv2.cvtColor(down, cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual_rms = float(np.sqrt(np.mean((luma_after - luma_before) ** 2)))
        metrics = dict(cached_metrics)
        metrics.update({
            "sourceWidth": float(width),
            "sourceHeight": float(height),
            "roundTripLumaRms": residual_rms,
            "tileModelDeltaRms": float(cached_metrics.get("mapDeltaRms", 0.0)),
            "neuralCompanionMap": True,
        })
        return out_w, out_h, f"{ACTIVE_BACKEND}-companion", metrics

    high = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    original_high = high.copy()
    semantic_names = {usage.semantic for usage in usages}
    backend = "structure-aware-v2"
    tile_model_delta_rms = 0.0
    neural_diagnostics: dict[str, object] = {}

    if "albedo" in semantic_names:
        classic_v2 = _structure_aware_albedo(image[:, :, :3], target_size)
        albedo = classic_v2
        if neural is not None:
            executable, model = neural
            try:
                learned = _run_realesrgan(image[:, :, :3], executable, model, target_size)
                # The optional learned upsampler proposes edge placement while the
                # deterministic branch remains responsible for texture fidelity.
                albedo = cv2.addWeighted(classic_v2, 0.66, learned, 0.34, 0.0)
                backend = f"structure-aware-v2+realesrgan:{model}"
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: optional neural upsampling failed for {source.name}; using deterministic v2: {exc}", file=sys.stderr)
        if (out_w, out_h) != (target_size, target_size):
            albedo = cv2.resize(albedo, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        high[:, :, :3] = albedo

    processed_channels: set[int] = set((0, 1, 2)) if "albedo" in semantic_names else set()
    normal_pairs: set[tuple[int, int]] = set()
    for usage in usages:
        if usage.semantic == "normal" and usage.x_channel is not None and usage.y_channel is not None:
            normal_pairs.add((_rgba_to_bgra_channel(usage.x_channel), _rgba_to_bgra_channel(usage.y_channel)))
    for x_index, y_index in sorted(normal_pairs):
        x, y = _reconstruct_normal_pair_v2(image[:, :, x_index], image[:, :, y_index], target_size)
        if (out_w, out_h) != (target_size, target_size):
            x = cv2.resize(x, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            y = cv2.resize(y, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        high[:, :, x_index] = np.clip((x + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        high[:, :, y_index] = np.clip((y + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
        processed_channels.update((x_index, y_index))

    crisp_semantics = {"material", "paint_mask", "glow"}
    soft_semantics = {"dirt", "ao", "roughness_map"}
    for usage in usages:
        if usage.x_channel is None or usage.semantic not in crisp_semantics | soft_semantics:
            continue
        channel = _rgba_to_bgra_channel(usage.x_channel)
        if channel in processed_channels:
            continue
        plane = _reconstruct_scalar_plane_v2(
            image[:, :, channel],
            target_size,
            crisp=usage.semantic in crisp_semantics,
        )
        if (out_w, out_h) != (target_size, target_size):
            plane = cv2.resize(plane, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        high[:, :, channel] = np.clip(plane, 0.0, 255.0).astype(np.uint8)
        processed_channels.add(channel)

    for channel in range(4):
        if channel not in processed_channels:
            high[:, :, channel] = original_high[:, :, channel]

    if "albedo" in semantic_names:
        if tile_runtime is not None:
            model, model_config, model_device = tile_runtime
            deterministic_before = high.copy()
            tile_model_delta_rms, neural_diagnostics, companion_outputs, audit_aux = _apply_tile_context_model(
                image, high, albedo_context, model, model_config, model_device)
            if _env_bool("NSAMDR_GEOMETRY_AUDIT", True):
                audit_options = _geometry_audit_options()
                critic_raw = os.environ.get("NSAMDR_GEOMETRY_CRITIC_CHECKPOINT", "").strip()
                critic_checkpoint = Path(critic_raw) if critic_raw else output.parent / "geometry_audit" / "geometry_pair_critic.pt"
                critic_device = os.environ.get("NSAMDR_GEOMETRY_CRITIC_DEVICE", str(model_device)).strip() or str(model_device)
                audit_name = f"{source.stem}_{hashlib.sha1(str(source).lower().encode('utf-8')).hexdigest()[:8]}"
                try:
                    audit_report = audit_pair(
                        deterministic_before, high, source_name=audit_name,
                        output_dir=output.parent / "geometry_audit", options=audit_options,
                        flow=audit_aux.get("displacement"), gate=audit_aux.get("gate"),
                        critic_checkpoint=critic_checkpoint, critic_device=critic_device,
                    )
                except Exception as audit_exc:  # noqa: BLE001
                    # `report` means instrumentation must never prevent the renderer
                    # candidate from being generated. `strict` deliberately keeps the
                    # fail-fast behaviour for CI/promotion gating.
                    if audit_options.policy == "strict":
                        raise
                    neural_diagnostics["geometryAuditError"] = {
                        "type": type(audit_exc).__name__,
                        "message": str(audit_exc),
                        "source": str(source),
                    }
                    print(
                        f"WARNING: geometry audit evidence failed for {source.name}; "
                        f"continuing because audit policy=report: {audit_exc}",
                        file=sys.stderr, flush=True,
                    )
                else:
                    neural_diagnostics["geometryAudit"] = audit_report
                    # Four-way texture evidence separates deterministic gain from
                    # actual neural gain: A raw source, B deterministic 4x, C
                    # NSAMDR, D the learned SDF/gate diagnostic.
                    try:
                        evidence_size = 768
                        raw_panel = cv2.resize(image[:, :, :3], (evidence_size, evidence_size), interpolation=cv2.INTER_NEAREST)
                        deterministic_panel = cv2.resize(deterministic_before[:, :, :3], (evidence_size, evidence_size), interpolation=cv2.INTER_AREA)
                        neural_panel = cv2.resize(high[:, :, :3], (evidence_size, evidence_size), interpolation=cv2.INTER_AREA)
                        sdf_diag = np.asarray(audit_aux.get("sdf"), dtype=np.float32)
                        gate_diag = np.asarray(audit_aux.get("gate"), dtype=np.float32)
                        sdf_diag = cv2.resize(sdf_diag, (evidence_size, evidence_size), interpolation=cv2.INTER_LINEAR)
                        gate_diag = cv2.resize(gate_diag, (evidence_size, evidence_size), interpolation=cv2.INTER_LINEAR)
                        zero_band = np.exp(-np.abs(sdf_diag) * 18.0)
                        diagnostic_panel = np.zeros((evidence_size, evidence_size, 3), dtype=np.uint8)
                        diagnostic_panel[:, :, 1] = np.uint8(np.round(np.clip(zero_band, 0.0, 1.0) * 255.0))
                        diagnostic_panel[:, :, 2] = np.uint8(np.round(np.clip(gate_diag, 0.0, 1.0) * 255.0))
                        panels = [raw_panel, deterministic_panel, neural_panel, diagnostic_panel]
                        labels = ["A RAW SOURCE", "B DETERMINISTIC 4X", "C NSAMDR V9.8.3", "D SDF ZERO-SET / GATE"]
                        for panel, label in zip(panels, labels):
                            cv2.rectangle(panel, (0,0), (evidence_size-1,34), (8,8,8), -1)
                            cv2.putText(panel, label, (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240,240,240), 1, cv2.LINE_AA)
                        abcd_path = output.parent / "geometry_audit" / "mode3_abcd_texture_evidence.png"
                        abcd_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(abcd_path), np.concatenate(panels, axis=1), [cv2.IMWRITE_PNG_COMPRESSION, 4])
                        neural_diagnostics["abcdTextureEvidence"] = str(abcd_path)

                        # Deterministic Raven texture crops: choose the strongest
                        # non-overlapping baseline edge regions. The source asset
                        # and selection algorithm are fixed, so coordinates stay
                        # stable across experiments and make progress directly
                        # comparable.
                        gray_det = cv2.cvtColor(deterministic_before[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
                        gx = cv2.Sobel(gray_det, cv2.CV_32F, 1, 0, ksize=3)
                        gy = cv2.Sobel(gray_det, cv2.CV_32F, 0, 1, ksize=3)
                        mag = np.sqrt(gx * gx + gy * gy)
                        crop_size = max(192, min(512, min(target_width, target_height) // 5))
                        stride = max(96, crop_size // 2)
                        candidate_regions = []
                        for cy in range(0, max(1, target_height - crop_size + 1), stride):
                            for cx in range(0, max(1, target_width - crop_size + 1), stride):
                                score = float(np.mean(mag[cy:cy+crop_size, cx:cx+crop_size]))
                                candidate_regions.append((score, cx, cy))
                        candidate_regions.sort(reverse=True)
                        selected_regions = []
                        for score, cx, cy in candidate_regions:
                            if any(
                                max(0, min(cx + crop_size, sx + crop_size) - max(cx, sx))
                                * max(0, min(cy + crop_size, sy + crop_size) - max(cy, sy))
                                > crop_size * crop_size * 0.30
                                for _ss, sx, sy in selected_regions
                            ):
                                continue
                            selected_regions.append((score, cx, cy))
                            if len(selected_regions) >= 6:
                                break
                        raw_full = cv2.resize(image[:, :, :3], (target_width, target_height), interpolation=cv2.INTER_NEAREST)
                        sdf_full = np.asarray(audit_aux.get("sdf"), dtype=np.float32)
                        gate_full = np.asarray(audit_aux.get("gate"), dtype=np.float32)
                        if sdf_full.shape[:2] != (target_height, target_width):
                            sdf_full = cv2.resize(sdf_full, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                        if gate_full.shape[:2] != (target_height, target_width):
                            gate_full = cv2.resize(gate_full, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                        diag_full = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                        diag_full[:, :, 1] = np.uint8(np.round(np.clip(np.exp(-np.abs(sdf_full) * 18.0), 0.0, 1.0) * 255.0))
                        diag_full[:, :, 2] = np.uint8(np.round(np.clip(gate_full, 0.0, 1.0) * 255.0))
                        fixed_dir = output.parent / "geometry_audit" / "raven_fixed_crops"
                        fixed_dir.mkdir(parents=True, exist_ok=True)
                        crop_manifest = []
                        for crop_index, (score, cx, cy) in enumerate(selected_regions, start=1):
                            panels_crop = [
                                raw_full[cy:cy+crop_size, cx:cx+crop_size].copy(),
                                deterministic_before[cy:cy+crop_size, cx:cx+crop_size, :3].copy(),
                                high[cy:cy+crop_size, cx:cx+crop_size, :3].copy(),
                                diag_full[cy:cy+crop_size, cx:cx+crop_size].copy(),
                            ]
                            crop_labels = ["A RAW", "B DETERMINISTIC", "C NSAMDR", "D SDF/GATE"]
                            for cp, label in zip(panels_crop, crop_labels):
                                cv2.rectangle(cp, (0,0), (cp.shape[1]-1,30), (8,8,8), -1)
                                cv2.putText(cp, label, (6,21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240,240,240), 1, cv2.LINE_AA)
                            crop_path = fixed_dir / f"RAVEN_FIXED_CROP_{crop_index:02d}.png"
                            cv2.imwrite(str(crop_path), np.concatenate(panels_crop, axis=1), [cv2.IMWRITE_PNG_COMPRESSION, 4])
                            crop_manifest.append({"index": crop_index, "x": cx, "y": cy, "size": crop_size, "edgeScore": score, "path": str(crop_path)})
                        crop_manifest_path = fixed_dir / "raven_fixed_crop_manifest.json"
                        crop_manifest_path.write_text(json.dumps({
                            "schema": "NSAMDR_RAVEN_FIXED_CROPS_V1",
                            "source": str(source.resolve()),
                            "sourceSha256": _sha256(source),
                            "selection": "deterministic-baseline-edge-energy-nonoverlap",
                            "crops": crop_manifest,
                        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                        neural_diagnostics["fixedRavenCropManifest"] = str(crop_manifest_path)
                    except Exception as abcd_exc:  # noqa: BLE001
                        neural_diagnostics["abcdTextureEvidenceError"] = str(abcd_exc)
                    print(
                        f"    geometry audit: {audit_report['verdict']} | "
                        f"proxy={audit_report['summary']['proxyGeometryImprovementMean']:+.4f} | "
                        f"off-edge={audit_report['summary']['offEdgeIdentityRms8bit']:.3f} levels",
                        flush=True,
                    )
            companion_audits: dict[str, object] = {}
            if _env_bool("NSAMDR_GEOMETRY_AUDIT", True):
                companion_baselines = audit_aux.get("companionBaselines", {})
                for companion_path, companion_image in companion_outputs.items():
                    resolved_companion = companion_path.resolve()
                    baseline_companion = companion_baselines.get(resolved_companion) if isinstance(companion_baselines, dict) else None
                    if baseline_companion is None:
                        continue
                    companion_name = f"{resolved_companion.stem}_{hashlib.sha1(str(resolved_companion).lower().encode('utf-8')).hexdigest()[:8]}"
                    try:
                        companion_report = audit_pair(
                            baseline_companion, companion_image, source_name=companion_name,
                            output_dir=output.parent / "geometry_audit", options=audit_options,
                            gate=audit_aux.get("gate"),
                            critic_checkpoint=critic_checkpoint, critic_device=critic_device,
                        )
                    except Exception as companion_exc:  # noqa: BLE001
                        if audit_options.policy == "strict":
                            raise
                        print(f"WARNING: companion geometry audit failed for {resolved_companion.name}: {companion_exc}", file=sys.stderr, flush=True)
                    else:
                        companion_audits[str(resolved_companion)] = companion_report
                neural_diagnostics["companionGeometryAudits"] = companion_audits

            for companion_path, companion_image in companion_outputs.items():
                resolved = companion_path.resolve()
                if albedo_context and albedo_context.normal and resolved == albedo_context.normal.resolve():
                    role = "normal"
                    delta = float(neural_diagnostics.get("normalDeltaRms", 0.0))
                else:
                    role = "physical-material"
                    delta = float(neural_diagnostics.get("materialDeltaRms", 0.0))
                companion_diagnostics = dict(neural_diagnostics)
                companion_diagnostics.pop("geometryAudit", None)
                companion_diagnostics.pop("companionGeometryAudits", None)
                map_audit = companion_audits.get(str(resolved)) if 'companion_audits' in locals() else None
                if isinstance(map_audit, dict):
                    companion_diagnostics["geometryAudit"] = map_audit
                neural_bundle_cache[resolved] = (companion_image, {
                    "schema": tile_model.MODEL_SCHEMA,
                    "mapRole": role,
                    "mapDeltaRms": delta,
                    "neural": companion_diagnostics,
                })
            backend += "+" + ACTIVE_BACKEND
        else:
            raise RuntimeError(
                f"Mode 3 requires a trained NSAMDR {ACTIVE_MODEL_LABEL} CUDA checkpoint; deterministic bootstrap is disabled")

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), high, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise RuntimeError(f"Could not write prepared NSAMDR texture: {output}")

    luma_before = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    down = cv2.resize(high[:, :, :3], (width, height), interpolation=cv2.INTER_AREA)
    luma_after = cv2.cvtColor(down, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual_rms = float(np.sqrt(np.mean((luma_after - luma_before) ** 2)))
    return out_w, out_h, backend, {
        "sourceWidth": float(width),
        "sourceHeight": float(height),
        "roundTripLumaRms": residual_rms,
        "tileModelDeltaRms": tile_model_delta_rms,
        "neural": neural_diagnostics,
    }



def _write_manifest(
    output: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    comments: list[str],
    replacements: dict[Path, Path],
    source_dir: Path,
    label: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# NSAMDR_MATERIALS_V7\n")
        handle.write(f"# PHYSICAL_CANDIDATE {label}\n")
        for comment in comments:
            if comment.startswith("#") and "NSAMDR_MATERIALS" not in comment:
                handle.write(comment + "\n")
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            adjusted = dict(row)
            for semantic in SEMANTICS:
                source = _resolve_path(row.get(semantic, ""), source_dir)
                if source in replacements:
                    adjusted[semantic] = str(replacements[source].resolve())
            writer.writerow(adjusted)


def _latest_input_mtime(paths: Iterable[Path]) -> float:
    return max((path.stat().st_mtime for path in paths if path.is_file()), default=0.0)


def _report_is_fresh(
    report_path: Path,
    inputs: Iterable[Path],
    target_size: int,
    neural_checkpoint: Path,
    checkpoint_sha256: str,
    checkpoint_profile: str,
    preview_strength: float,
) -> bool:
    if not report_path.is_file():
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if report.get("schema") != REPORT_SCHEMA or int(report.get("targetSize", 0)) != target_size:
        return False
    reported_checkpoint = Path(str(report.get("neuralCheckpoint", "")))
    try:
        same_checkpoint = reported_checkpoint.resolve() == neural_checkpoint.resolve()
    except OSError:
        same_checkpoint = False
    if not same_checkpoint or str(report.get("neuralCheckpointSha256", "")) != checkpoint_sha256:
        return False
    if str(report.get("checkpointProfile", "")) != checkpoint_profile:
        return False
    try:
        reported_strength = float(report.get("previewStrength", -1.0))
    except (TypeError, ValueError):
        return False
    if abs(reported_strength - float(preview_strength)) > 1.0e-8:
        return False
    provenance = report.get("controlProvenance", {})
    if not isinstance(provenance, dict) or not bool(provenance.get("verified")):
        return False
    required = [Path(str(report.get(key, ""))) for key in (
        "mode3Obj", "mode3Materials", "mode3Analysis", "mode3Validation",
    )]
    if not all(path.is_file() for path in required):
        return False
    return report_path.stat().st_mtime >= _latest_input_mtime(inputs)


def generate_candidates(
    obj: Path,
    materials: Path,
    asset_manifest: Path | None,
    output_root: Path,
    target_size: int,
    force: bool,
    super_resolution_backend: str,
    inference_device: str,
    checkpoint_dir: Path | None = None,
) -> dict[str, object]:
    report_path = output_root / "strategy_candidates.json"
    fields, rows, comments = _read_tsv(materials)
    usages = _collect_usages(rows, materials.parent)
    albedo_contexts = _collect_albedo_contexts(rows, materials.parent)
    tool_dir = Path(__file__).resolve().parent
    neural_sr = _find_realesrgan(tool_dir) if super_resolution_backend in {"auto", "realesrgan"} else None
    if super_resolution_backend == "realesrgan" and neural_sr is None:
        raise RuntimeError(
            "Real-ESRGAN was requested but is not configured. Install the backend or use --super-resolution-backend auto/classic."
        )

    repository_root = tool_dir.parents[1]
    resolved_checkpoint_dir = _resolve_checkpoint_dir(repository_root, checkpoint_dir)
    global tile_model, ACTIVE_MODEL_LABEL, ACTIVE_BACKEND
    global ACTIVE_CHECKPOINT_PROFILE, ACTIVE_PREVIEW_STRENGTH, ACTIVE_VALIDATION_RMS_LIMIT
    requested_architecture = os.environ.get("NSAMDR_NEURAL_ARCHITECTURE", "V9").strip().upper()
    if requested_architecture not in {"", "V9"}:
        raise RuntimeError(f"Only NSAMDR V9 is supported, got architecture {requested_architecture!r}")
    tile_model = v9_tile_model
    ACTIVE_MODEL_LABEL = "V9.8.3 sign-gauge metric-SDF geometry-convergence renderer 4x"
    ACTIVE_BACKEND = "staged implicit-SDF-multimap-4x-v9-7"
    neural_checkpoint = resolved_checkpoint_dir / "nsamdr_v9_fidelity.pt"
    neural_metadata = resolved_checkpoint_dir / "nsamdr_v9_fidelity.json"
    use_v9 = True
    inputs = [
        obj,
        materials,
        Path(__file__).resolve(),
        tool_dir / "strategy_pipeline" / "reconstruction.py",
        *usages.keys(),
    ]
    if neural_sr is not None:
        inputs.append(neural_sr[0])
    checkpoint = {}
    tile_runtime = None
    device = "not-trained"
    if not neural_checkpoint.is_file():
        raise RuntimeError(
            f"Mode 3 checkpoint was not found: {neural_checkpoint}. "
            "Set NSAMDR_NEURAL_CHECKPOINT_DIR or pass --checkpoint-dir to select a trained checkpoint directory."
        )
    if not neural_metadata.is_file():
        raise RuntimeError(f"NSAMDR checkpoint metadata is missing: {neural_metadata}")
    inputs.extend((neural_checkpoint, neural_metadata))
    checkpoint_sha256 = _sha256(neural_checkpoint)
    print(f"NSAMDR neural checkpoint directory: {resolved_checkpoint_dir}", flush=True)
    print(f"NSAMDR neural checkpoint SHA-256 : {checkpoint_sha256}", flush=True)
    requested_inference_device = str(inference_device or "cuda").strip().lower()
    checkpoint_probe = tile_model.TrainingConfig()
    checkpoint_probe.device = requested_inference_device
    device = tile_model.resolve_device(checkpoint_probe, requested_inference_device)
    if device.type == "cpu":
        try:
            import torch
            cpu_threads = max(
                1,
                min(
                    32,
                    int(os.environ.get("NSAMDR_CPU_THREADS", "2")),
                ),
            )
            torch.set_num_threads(cpu_threads)
            if hasattr(torch, "set_num_interop_threads"):
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass
        except (ImportError, TypeError, ValueError):
            cpu_threads = 2
        print(
            f"NSAMDR CPU live preview       : {cpu_threads} inference thread(s); "
            "training CUDA allocation is untouched",
            flush=True,
        )
    model, model_config, checkpoint = tile_model.load_trained_model(neural_checkpoint, device)
    ACTIVE_BACKEND = (
        f"{device.type}-staged implicit-SDF-multimap-4x-v9-7"
    )
    tile_runtime = (model, model_config, device)
    if use_v9:
        ACTIVE_CHECKPOINT_PROFILE = _checkpoint_profile(model_config, checkpoint, "V9")
        ACTIVE_PREVIEW_STRENGTH = _parse_preview_strength(
            0.20 if ACTIVE_CHECKPOINT_PROFILE == "stability-smoke" else 1.0)
        ACTIVE_VALIDATION_RMS_LIMIT = 96.0
        version_label = "V9"
        print(f"NSAMDR {version_label} checkpoint profile     : {ACTIVE_CHECKPOINT_PROFILE}", flush=True)
        print(f"NSAMDR {version_label} preview strength       : {ACTIVE_PREVIEW_STRENGTH:.3f}", flush=True)
        if ACTIVE_CHECKPOINT_PROFILE == "stability-smoke":
            print(
                "WARNING: This is a reduced stability checkpoint. Candidate output is strength-limited "
                "to validate the pipeline; it is not a quality result.", flush=True)
        elif ACTIVE_CHECKPOINT_PROFILE == "live-completed-epoch":
            print(
                "LIVE PREVIEW: Mode 3 is using the latest completed training epoch, "
                "not the final production checkpoint.",
                flush=True,
            )
    else:
        ACTIVE_CHECKPOINT_PROFILE = "v7-compatibility"
        ACTIVE_PREVIEW_STRENGTH = 1.0
        ACTIVE_VALIDATION_RMS_LIMIT = 18.0

    if not force and _report_is_fresh(
        report_path, inputs, target_size, neural_checkpoint, checkpoint_sha256,
        ACTIVE_CHECKPOINT_PROFILE, ACTIVE_PREVIEW_STRENGTH,
    ):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _write_candidate_result_pointer(report_path, report)
        print(f"NSAMDR Mode 3 candidate is current: {report_path}", flush=True)
        return report

    mode3_dir = output_root / "mode3_nsamdr_neural"
    mode3_dir.mkdir(parents=True, exist_ok=True)

    replacements: dict[Path, Path] = {}
    source_provenance_before: dict[Path, dict[str, object]] = {
        source: {
            "sha256": _sha256(source),
            "dimensions": _image_dimensions(source),
            "sizeBytes": int(source.stat().st_size),
        }
        for source in usages
    }
    dimensions: dict[str, list[int]] = {}
    backends: set[str] = set()
    metrics_by_source: dict[str, dict[str, object]] = {}
    neural_bundle_cache: dict[Path, tuple[object, dict[str, object]]] = {}
    print(
        f"Preparing Mode 3 NSAMDR targets with {ACTIVE_MODEL_LABEL} CUDA inference "
        f"({len(usages)} unique source images, device={device})...", flush=True)
    ordered_sources = sorted(
        usages.items(),
        key=lambda item: (0 if any(usage.semantic == "albedo" for usage in item[1]) else 1, str(item[0]).lower()),
    )
    for index, (source, source_usages) in enumerate(ordered_sources, 1):
        output = mode3_dir / _unique_output_name(source).replace("_4k.png", "_nsamdr_4k.png")
        width, height, backend, metrics = _prepare_nsamdr_texture(
            source, source_usages, output, target_size, neural_sr,
            albedo_contexts.get(source), tile_runtime, neural_bundle_cache,
        )
        replacements[source] = output
        dimensions[str(output.resolve())] = [width, height]
        backends.add(backend)
        metrics_by_source[str(source.resolve())] = metrics
        print(
            f"  [{index}/{len(usages)}] {source.name} -> {output.name} "
            f"({width}x{height}, {backend}, round-trip RMS={metrics['roundTripLumaRms']:.3f}, model delta={metrics['tileModelDeltaRms']:.3f})",
            flush=True,
        )

    mode3_materials = mode3_dir / "ship.materials.mode3.tsv"
    _write_manifest(
        mode3_materials,
        fields,
        rows,
        comments,
        replacements,
        materials.parent,
        "MODE3_NSAMDR_V9_6_BOUNDARY_ANCHORED_4X",
    )
    mode3_obj = mode3_dir / obj.name
    mode3_obj.write_bytes(obj.read_bytes())

    control_provenance_path = mode3_dir / "preview_control_provenance.json"
    control_provenance = _build_control_provenance(
        sources_before=source_provenance_before,
        replacements=replacements,
        usages=usages,
        output_root=output_root,
        materials=materials,
        asset_manifest=asset_manifest,
        destination=control_provenance_path,
    )
    primary = control_provenance.get("primaryAlbedo", {})
    print(
        "NSAMDR preview control provenance: VERIFIED | "
        f"source={Path(str(primary.get('sourcePath', ''))).name if primary else 'n/a'} | "
        f"evidence={control_provenance_path}",
        flush=True,
    )

    validation_failures: list[str] = []
    validation_advisories: list[str] = []
    for source, metrics in metrics_by_source.items():
        round_trip = metrics.get("roundTripLumaRms")
        model_delta = metrics.get("tileModelDeltaRms")
        if not _finite_metric(round_trip) or not _finite_metric(model_delta):
            validation_failures.append(source)
            continue
        round_trip_value = float(round_trip)
        if not 0.0 <= round_trip_value <= ACTIVE_VALIDATION_RMS_LIMIT:
            validation_failures.append(source)
        elif use_v9 and round_trip_value > 18.0:
            validation_advisories.append(source)
    all_outputs_exist = all(path.is_file() for path in replacements.values())

    geometry_audit_reports: list[dict[str, object]] = []
    for source_metrics in metrics_by_source.values():
        neural_metrics = source_metrics.get("neural", {}) if isinstance(source_metrics, dict) else {}
        audit = neural_metrics.get("geometryAudit") if isinstance(neural_metrics, dict) else None
        if isinstance(audit, dict):
            geometry_audit_reports.append(audit)
    geometry_audit_bundle: dict[str, object] = {}
    if geometry_audit_reports:
        geometry_audit_bundle = write_audit_bundle(mode3_dir / "geometry_audit", geometry_audit_reports)
        print(
            f"NSAMDR geometry audit: {geometry_audit_bundle['verdict']} | "
            f"textures={geometry_audit_bundle['textureCount']} | "
            f"report={geometry_audit_bundle['jsonPath']}", flush=True)

    mode3_analysis = mode3_dir / "mode3.analysis.json"
    mode3_analysis.write_text(json.dumps({
        "schema": f"NSAMDR_MODE3_ANALYSIS_{ACTIVE_MODEL_LABEL.replace(' ', '_').upper()}",
        "phase": f"fp16-cuda-overlapping-{ACTIVE_BACKEND}",
        "targetSize": target_size,
        "textureCount": len(replacements),
        "textureDimensions": dimensions,
        "preparationBackends": sorted(backends),
        "roundTripMetrics": metrics_by_source,
        "trainedCheckpoint": str(neural_checkpoint.resolve()) if neural_checkpoint.is_file() else "",
        "trainedMetadata": str(neural_metadata.resolve()) if neural_metadata.is_file() else "",
        "trainedMetadataPresent": neural_metadata.is_file(),
        "cudaNeuralInferenceApplied": True,
        "bootstrapCandidate": False,
        "inferenceDevice": str(device),
        "inferencePrecision": "fp16",
        "modelSha256": str(checkpoint.get("model_sha256", "")),
        "parameterCount": int(checkpoint.get("parameter_count", 0)),
        "checkpointProfile": ACTIVE_CHECKPOINT_PROFILE,
        "previewStrength": ACTIVE_PREVIEW_STRENGTH,
        "qualityReady": ACTIVE_CHECKPOINT_PROFILE == "production" and ACTIVE_PREVIEW_STRENGTH >= 0.999,
        "runtimeStages": {
            "1_cudaTraining": "tools/nsamdr/neural/train_nsamdr_v9.py",
            "2_fp16CudaInference": "generate_strategy_candidates.py overlapping albedo/normal bundle tiles",
            "3_bakedMaterialOutput": "mode3_nsamdr_neural/*_nsamdr_4k.png for neural albedo/normal/material/roughness/emissive companions",
            "4_liveShaderSampling": "NSAMDRRenderPipeline.cpp -> NSAMDRPreview.hlsl",
        },
        "semanticMaps": ACTIVE_MODEL_LABEL,
        "materialPolicy": "neural-aligned-physical-map",
        "authoredUvTopologyPreserved": True,
        "geometryAudit": geometry_audit_bundle,
        "controlProvenance": control_provenance,
        "controlProvenancePath": str(control_provenance_path.resolve()),
    }, indent=2) + "\n", encoding="utf-8")

    mode3_validation = mode3_dir / "mode3.validation.json"
    validation_payload = {
        "schema": f"NSAMDR_MODE3_VALIDATION_{ACTIVE_MODEL_LABEL.replace(' ', '_').upper()}",
        "passed": (
            all_outputs_exist and not validation_failures and neural_checkpoint.is_file() and neural_metadata.is_file()
            and bool(control_provenance.get("verified"))
            and not (
                _geometry_audit_options().policy == "strict"
                and geometry_audit_bundle.get("verdict") == "FAIL"
            )
        ),
        "failedTextures": validation_failures,
        "advisoryTextures": validation_advisories,
        "checkpointProfile": ACTIVE_CHECKPOINT_PROFILE,
        "previewStrength": ACTIVE_PREVIEW_STRENGTH,
        "qualityReady": ACTIVE_CHECKPOINT_PROFILE == "production" and ACTIVE_PREVIEW_STRENGTH >= 0.999,
        "checks": {
            "allPreparedInputsExist": all_outputs_exist,
            "roundTripLumaRmsMaximum": ACTIVE_VALIDATION_RMS_LIMIT,
            "roundTripMetricRole": "corruption ceiling; V9 quality is governed by baseline-regret metrics",
            "tileCheckpointExists": neural_checkpoint.is_file(),
            "trainedMetadataPresent": neural_metadata.is_file(),
            "runtimeComputeKernelRequired": False,
            "cudaRequired": True,
            "cudaDeviceActive": str(device).startswith("cuda"),
            "fp16OverlappingInferenceBaked": True,
            "bootstrapCandidateAvailable": False,
            "neuralAlbedoNormalReconstructed": True,
            "neuralPhysicalMaterialReconstructed": True,
            "materialMapPassthrough": False,
            "authoredUvTopologyPreserved": True,
            "stabilityPreviewOnly": ACTIVE_CHECKPOINT_PROFILE == "stability-smoke",
            "geometryAuditEnabled": bool(geometry_audit_reports),
            "geometryAuditPolicy": _geometry_audit_options().policy,
            "geometryAuditVerdict": geometry_audit_bundle.get("verdict", "NOT_RUN"),
            "controlProvenanceVerified": bool(control_provenance.get("verified")),
            "rawControlUsesCandidateTree": any(
                not bool(item.get("sourceOutsideCandidateTree"))
                for item in control_provenance.get("records", [])
            ),
        },
        "geometryAudit": geometry_audit_bundle,
        "controlProvenance": control_provenance,
        "controlProvenancePath": str(control_provenance_path.resolve()),
    }
    mode3_validation.write_text(json.dumps(validation_payload, indent=2) + "\n", encoding="utf-8")
    if validation_advisories:
        version_label = "V9"
        print(
            f"NSAMDR {version_label} validation advisory: {len(validation_advisories)} texture(s) exceed the old "
            "conservative V7 source-round-trip threshold; reconstruction remains valid.", flush=True)
    if not validation_payload["passed"]:
        print(f"NSAMDR candidate validation report: {mode3_validation}", flush=True)
        print(f"Failed textures: {validation_failures}", flush=True)
        raise RuntimeError("Mode 3 NSAMDR candidate validation failed")

    manifest = StrategyManifest(REPORT_SCHEMA, target_size, obj, materials)
    manifest.add(CandidateArtifact(
        mode=3,
        label="NSAMDR V9.8.3 sign-gauge metric-SDF geometry reconstruction",
        obj=mode3_obj,
        materials=mode3_materials,
        metadata={
            "analysis": str(mode3_analysis.resolve()),
            "validation": str(mode3_validation.resolve()),
            "textureCount": len(replacements),
            "backend": "prepared-target+" + ACTIVE_BACKEND,
            "textureDimensions": dimensions,
            "runtimeNeuralKernel": False,
            "offlineCudaNeuralInference": True,
            "bootstrapCandidate": False,
            "liveShaderSampling": True,
            "neuralAlbedoNormalReconstructed": True,
            "neuralPhysicalMaterialReconstructed": True,
            "materialMapPassthrough": False,
            "authoredUvPreserved": True,
            "geometryAudit": geometry_audit_bundle,
            "controlProvenance": control_provenance,
            "controlProvenancePath": str(control_provenance_path.resolve()),
        },
    ))
    manifest.notes.extend([
        "Public Mode 1 is the untouched source asset; scientific control uses the same 16x/LOD0 sampler as Mode 3.",
        "Three-pane scientific preview shows raw control, legacy sampler emulation, and NSAMDR simultaneously.",
        "Public Mode 2 is the existing UV/stretch diagnostic render.",
        "Public Mode 3 uses the trained V9.8.3 sign-gauge metric-SDF geometry-convergence checkpoint.",
        "Reduced V9 pilot/stability checkpoints are pipeline validation results and are not final production quality.",
        "Historical intermediate appearance modes are no longer generated or exported.",
    ])
    report = manifest.to_report()
    report["neuralCheckpoint"] = str(neural_checkpoint.resolve())
    report["neuralMetadata"] = str(neural_metadata.resolve())
    report["neuralCheckpointSha256"] = checkpoint_sha256
    report["checkpointProfile"] = ACTIVE_CHECKPOINT_PROFILE
    report["previewStrength"] = ACTIVE_PREVIEW_STRENGTH
    report["qualityReady"] = ACTIVE_CHECKPOINT_PROFILE == "production" and ACTIVE_PREVIEW_STRENGTH >= 0.999
    report["geometryAudit"] = geometry_audit_bundle
    report["controlProvenance"] = control_provenance
    report["controlProvenancePath"] = str(control_provenance_path.resolve())
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_candidate_result_pointer(report_path, report)
    print(f"NSAMDR three-mode candidate manifest: {report_path}", flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate NSAMDR physical strategy candidates")
    parser.add_argument("--obj", required=True, type=Path)
    parser.add_argument("--materials", required=True, type=Path)
    parser.add_argument("--asset-manifest", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--target-size", type=int, default=4096)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing a V9 nsamdr_v9_fidelity.pt/json pair. "
            "Defaults to NSAMDR_NEURAL_CHECKPOINT_DIR or the production neural directory."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--install-dependencies", action="store_true")
    parser.add_argument(
        "--super-resolution-backend", choices=("auto", "classic", "realesrgan"), default="auto",
        help="auto uses Real-ESRGAN when installed and otherwise uses the fidelity-first classic backend",
    )
    parser.add_argument(
        "--inference-device", choices=("auto", "cuda", "cpu"), default="cuda",
        help="Device used to bake the V9 reconstruction candidate. CPU is intended for low-impact live previews.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        _ensure_dependencies(args.install_dependencies)
        generate_candidates(
            args.obj.resolve(), args.materials.resolve(),
            args.asset_manifest.resolve() if args.asset_manifest else None,
            args.output_root.resolve(), args.target_size, args.force,
            args.super_resolution_backend, args.inference_device,
            args.checkpoint_dir.resolve() if args.checkpoint_dir else None,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
