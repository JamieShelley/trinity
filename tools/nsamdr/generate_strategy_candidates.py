#!/usr/bin/env python3
"""Bake one renderer candidate from one immutable production NSAMDR checkpoint."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


NEURAL_DIR = Path(__file__).resolve().parent / "neural"
if str(NEURAL_DIR) not in sys.path:
    sys.path.insert(0, str(NEURAL_DIR))

from v9.inference import infer_tiled, load_trained_model, resolve_device  # type: ignore
from v9.model import MODEL_SCHEMA, UPSCALE_FACTOR, build_model_input  # type: ignore


REPORT_SCHEMA = "NSAMDR_PRODUCTION_CANDIDATE_V1"
SEMANTICS = (
    "albedo",
    "normal",
    "material",
    "glow",
    "dirt",
    "ao",
    "paint_mask",
    "roughness_map",
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


@dataclass(frozen=True)
class Usage:
    semantic: str
    x_channel: int | None = None
    y_channel: int | None = None


@dataclass
class AlbedoContext:
    normal: Path | None = None
    normal_x_channel: int = 0
    normal_y_channel: int = 1
    material: Path | None = None
    material_channel: int = 0
    glow: Path | None = None
    glow_channel: int = 0
    roughness: Path | None = None
    roughness_channel: int = 0


class StrategyCandidateGenerator:
    # Purpose: Implement sha256 for StrategyCandidateGenerator.
    # Called by: _final_binding, _provenance, _source_snapshot, generate_candidate
    # Calls: No same-class helper methods.
    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    # Purpose: Implement read json for StrategyCandidateGenerator.
    # Called by: _final_binding
    # Calls: No same-class helper methods.
    def _read_json(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"expected a JSON object: {path}")
        return payload

    # Purpose: Implement path within for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: No same-class helper methods.
    def _path_within(self, path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    # Purpose: Implement full sha for StrategyCandidateGenerator.
    # Called by: _final_binding
    # Calls: No same-class helper methods.
    def _full_sha(self, value: object, *, label: str) -> str:
        result = str(value or "").strip().casefold()
        if not re.fullmatch(r"[0-9a-f]{64}", result):
            raise RuntimeError(f"{label} is not a full SHA-256: {result!r}")
        return result

    # Purpose: Implement read tsv for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: No same-class helper methods.
    def _read_tsv(self, path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
        comments: list[str] = []
        data_lines: list[str] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip() or line.startswith("#"):
                comments.append(line)
            else:
                data_lines.append(line)
        if not data_lines:
            raise RuntimeError(f"material manifest has no rows: {path}")
        reader = csv.DictReader(data_lines, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"material manifest has no TSV header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader], comments

    # Purpose: Implement resolve path for StrategyCandidateGenerator.
    # Called by: _collect_contexts, _collect_usages, _write_material_manifest
    # Calls: No same-class helper methods.
    def _resolve_path(self, value: str, base: Path) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        return path if path.is_file() else None

    # Purpose: Implement parse channel for StrategyCandidateGenerator.
    # Called by: _collect_contexts, _collect_usages
    # Calls: No same-class helper methods.
    def _parse_channel(self, row: Mapping[str, str], name: str | None, fallback: int = 0) -> int | None:
        if name is None:
            return None
        try:
            return max(0, min(3, int(float(row.get(name, str(fallback)) or fallback))))
        except (TypeError, ValueError):
            return fallback

    # Purpose: Implement rgba to bgra for StrategyCandidateGenerator.
    # Called by: _apply_direct_maps, _resized_channel
    # Calls: No same-class helper methods.
    def _rgba_to_bgra(self, channel: int) -> int:
        return (2, 1, 0, 3)[max(0, min(3, int(channel)))]

    # Purpose: Implement collect usages for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: _parse_channel, _resolve_path
    def _collect_usages(self, rows: list[dict[str, str]], base: Path) -> dict[Path, list[Usage]]:
        usages: dict[Path, list[Usage]] = {}
        for row in rows:
            for semantic in SEMANTICS:
                source = self._resolve_path(row.get(semantic, ""), base)
                if source is None:
                    continue
                x_name, y_name = CHANNEL_COLUMNS.get(semantic, (None, None))
                usage = Usage(semantic, self._parse_channel(row, x_name), self._parse_channel(row, y_name))
                bucket = usages.setdefault(source, [])
                if usage not in bucket:
                    bucket.append(usage)
        return usages

    # Purpose: Implement collect contexts for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: _parse_channel, _resolve_path
    def _collect_contexts(self, rows: list[dict[str, str]], base: Path) -> dict[Path, AlbedoContext]:
        contexts: dict[Path, AlbedoContext] = {}
        for row in rows:
            albedo = self._resolve_path(row.get("albedo", ""), base)
            if albedo is None:
                continue
            context = contexts.setdefault(albedo, AlbedoContext())
            normal = self._resolve_path(row.get("normal", ""), base)
            material = self._resolve_path(row.get("material", ""), base)
            glow = self._resolve_path(row.get("glow", ""), base) or material
            roughness = self._resolve_path(row.get("roughness_map", ""), base) or material
            if context.normal is None and normal is not None:
                context.normal = normal
                context.normal_x_channel = self._parse_channel(row, "normal_x_channel", 0) or 0
                context.normal_y_channel = self._parse_channel(row, "normal_y_channel", 1) or 1
            if context.material is None and material is not None:
                context.material = material
                context.material_channel = self._parse_channel(row, "material_channel", 0) or 0
            if context.glow is None and glow is not None:
                context.glow = glow
                context.glow_channel = self._parse_channel(row, "glow_channel", 0) or 0
            if context.roughness is None and roughness is not None:
                context.roughness = roughness
                context.roughness_channel = self._parse_channel(row, "roughness_channel", 0) or 0
        return contexts

    # Purpose: Implement require dependencies for StrategyCandidateGenerator.
    # Called by: _apply_direct_maps, _ensure_canvas, _model_input, _read_bgra, _resized_channel, _write_png
    # Calls: No same-class helper methods.
    def _require_dependencies(self) -> tuple[Any, Any]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                r"candidate dependencies are missing; run scripts\build\nsamdr.bat setup cuda"
            ) from exc
        return cv2, np

    # Purpose: Implement read bgra for StrategyCandidateGenerator.
    # Called by: _dimensions, _ensure_canvas, _model_input, _resized_channel, generate_candidate
    # Calls: _require_dependencies
    def _read_bgra(self, path: Path):
        cv2, _np = self._require_dependencies()
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"could not read authored texture: {path}")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        elif image.shape[2] != 4:
            raise RuntimeError(f"unsupported texture channel count {image.shape[2]}: {path}")
        return image

    # Purpose: Implement dimensions for StrategyCandidateGenerator.
    # Called by: _provenance, _source_snapshot
    # Calls: _read_bgra
    def _dimensions(self, path: Path) -> list[int]:
        image = self._read_bgra(path)
        return [int(image.shape[1]), int(image.shape[0])]

    # Purpose: Implement output size for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: No same-class helper methods.
    def _output_size(self, image, target_size: int) -> tuple[int, int]:
        height, width = image.shape[:2]
        if max(width, height) <= 0:
            raise RuntimeError("authored texture has invalid dimensions")
        scale = float(target_size) / float(max(width, height))
        return max(1, round(width * scale)), max(1, round(height * scale))

    # Purpose: Implement resized channel for StrategyCandidateGenerator.
    # Called by: _model_input
    # Calls: _read_bgra, _require_dependencies, _rgba_to_bgra
    def _resized_channel(
        self,
        path: Path | None,
        channel: int,
        width: int,
        height: int,
        default: float,
    ):
        cv2, np = self._require_dependencies()
        if path is None:
            return np.full((height, width), float(default), dtype=np.float32)
        image = self._read_bgra(path)
        value = image[:, :, self._rgba_to_bgra(channel)].astype(np.float32) / 255.0
        return cv2.resize(value, (width, height), interpolation=cv2.INTER_AREA)

    # Purpose: Implement model input for StrategyCandidateGenerator.
    # Called by: _direct_maps
    # Calls: _read_bgra, _require_dependencies, _resized_channel
    def _model_input(self, albedo: Path, context: AlbedoContext, out_width: int, out_height: int):
        cv2, np = self._require_dependencies()
        lr_width = max(1, math.ceil(out_width / UPSCALE_FACTOR))
        lr_height = max(1, math.ceil(out_height / UPSCALE_FACTOR))
        source = self._read_bgra(albedo)
        rgb = cv2.cvtColor(source[:, :, :3], cv2.COLOR_BGR2RGB)
        albedo_lr = cv2.resize(rgb, (lr_width, lr_height), interpolation=cv2.INTER_AREA)

        normal_lr = None
        if context.normal is not None:
            nx = self._resized_channel(
                context.normal,
                context.normal_x_channel,
                lr_width,
                lr_height,
                0.5,
            ) * 2.0 - 1.0
            ny = self._resized_channel(
                context.normal,
                context.normal_y_channel,
                lr_width,
                lr_height,
                0.5,
            ) * 2.0 - 1.0
            normal_lr = np.stack((nx, ny), axis=-1)

        material_lr = None
        if any((context.material, context.glow, context.roughness)):
            material_lr = np.stack(
                (
                    self._resized_channel(
                        context.material,
                        context.material_channel,
                        lr_width,
                        lr_height,
                        0.0,
                    ),
                    self._resized_channel(
                        context.glow,
                        context.glow_channel,
                        lr_width,
                        lr_height,
                        0.0,
                    ),
                    self._resized_channel(
                        context.roughness,
                        context.roughness_channel,
                        lr_width,
                        lr_height,
                        0.5,
                    ),
                ),
                axis=-1,
            )
        return build_model_input(
            albedo_lr,
            normal_xy=normal_lr,
            material_rgb=material_lr,
            degradation_level=1.0,
        )

    # Purpose: Implement direct maps for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: _model_input
    def _direct_maps(
        self,
        *,
        albedo: Path,
        context: AlbedoContext,
        model,
        config,
        device,
        out_width: int,
        out_height: int,
        output_variant: str = "final",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model_input = self._model_input(albedo, context, out_width, out_height)
        maps, diagnostics = infer_tiled(
            model,
            model_input,
            device,
            tile_size=config.inference_tile_size,
            overlap=config.inference_overlap,
            return_diagnostics=True,
            return_all_maps=True,
            output_variant=output_variant,
        )
        result: dict[str, Any] = {}
        for name in ("albedo", "normal_xy", "material", "roughness", "emissive"):
            value = maps[name]
            if value.shape[0] < out_height or value.shape[1] < out_width:
                raise RuntimeError(
                    f"production output {name} is smaller than requested candidate "
                    f"{value.shape[:2]} < {(out_height, out_width)}"
                )
            result[name] = value[:out_height, :out_width].copy()
        diagnostics = dict(diagnostics)
        diagnostics.update(
            {
                "candidateAuthority": f"direct-production-forward:{output_variant}",
                "postModelBlend": False,
                "postModelBoundaryRender": False,
                "postModelReplacement": False,
                "preSuperResolutionModel": False,
                "outputCropOnly": [
                    int(maps["albedo"].shape[1] - out_width),
                    int(maps["albedo"].shape[0] - out_height),
                ],
            }
        )
        return result, diagnostics

    # Purpose: Implement ensure canvas for StrategyCandidateGenerator.
    # Called by: _apply_direct_maps
    # Calls: _read_bgra, _require_dependencies
    def _ensure_canvas(
        self,
        canvases: dict[Path, Any],
        source: Path,
        width: int,
        height: int,
    ):
        cv2, _np = self._require_dependencies()
        existing = canvases.get(source)
        if existing is not None:
            if existing.shape[:2] != (height, width):
                raise RuntimeError(
                    f"one packed texture was requested at incompatible sizes: {source}"
                )
            return existing
        original = self._read_bgra(source)
        canvas = cv2.resize(original, (width, height), interpolation=cv2.INTER_LANCZOS4)
        canvases[source] = canvas
        return canvas

    # Purpose: Implement apply direct maps for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: _ensure_canvas, _require_dependencies, _rgba_to_bgra
    def _apply_direct_maps(
        self,
        *,
        albedo: Path,
        context: AlbedoContext,
        maps: Mapping[str, Any],
        width: int,
        height: int,
        canvases: dict[Path, Any],
        semantics: dict[Path, set[str]],
    ) -> None:
        cv2, np = self._require_dependencies()
        albedo_canvas = self._ensure_canvas(canvases, albedo, width, height)
        albedo_rgb = np.uint8(np.rint(np.clip(maps["albedo"], 0.0, 1.0) * 255.0))
        albedo_canvas[:, :, :3] = cv2.cvtColor(albedo_rgb, cv2.COLOR_RGB2BGR)
        semantics.setdefault(albedo, set()).add("albedo")

        if context.normal is not None:
            normal_canvas = self._ensure_canvas(canvases, context.normal, width, height)
            normal = np.asarray(maps["normal_xy"], dtype=np.float32)
            normal_canvas[:, :, self._rgba_to_bgra(context.normal_x_channel)] = np.uint8(
                np.rint(np.clip((normal[:, :, 0] + 1.0) * 127.5, 0.0, 255.0))
            )
            normal_canvas[:, :, self._rgba_to_bgra(context.normal_y_channel)] = np.uint8(
                np.rint(np.clip((normal[:, :, 1] + 1.0) * 127.5, 0.0, 255.0))
            )
            semantics.setdefault(context.normal, set()).add("normal")

        physical = (
            (context.material, context.material_channel, maps["material"][:, :, 0], "material"),
            (context.glow, context.glow_channel, maps["emissive"][:, :, 0], "emissive"),
            (context.roughness, context.roughness_channel, maps["roughness"][:, :, 0], "roughness"),
        )
        for source, channel, value, semantic in physical:
            if source is None:
                continue
            canvas = self._ensure_canvas(canvases, source, width, height)
            canvas[:, :, self._rgba_to_bgra(channel)] = np.uint8(
                np.rint(np.clip(value, 0.0, 1.0) * 255.0)
            )
            semantics.setdefault(source, set()).add(semantic)

    # Purpose: Implement destination for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: No same-class helper methods.
    def _destination(self, output_dir: Path, source: Path) -> Path:
        token = hashlib.sha1(str(source).casefold().encode("utf-8")).hexdigest()[:10]
        return output_dir / f"{source.stem}_{token}_nsamdr_final.png"

    # Purpose: Implement write png for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: _require_dependencies
    def _write_png(self, path: Path, image: Any) -> None:
        cv2, _np = self._require_dependencies()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"could not write candidate texture: {path}")

    # Purpose: Implement write material manifest for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: _resolve_path
    def _write_material_manifest(
        self,
        output: Path,
        fields: list[str],
        rows: list[dict[str, str]],
        comments: list[str],
        replacements: Mapping[Path, Path],
        source_dir: Path,
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            handle.write("# NSAMDR_MATERIALS_V7\n")
            handle.write("# PHYSICAL_CANDIDATE NSAMDR_PRODUCTION_FINAL\n")
            for comment in comments:
                if comment.startswith("#") and "NSAMDR_MATERIALS" not in comment:
                    handle.write(comment + "\n")
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                adjusted = dict(row)
                for semantic in SEMANTICS:
                    source = self._resolve_path(row.get(semantic, ""), source_dir)
                    if source in replacements:
                        adjusted[semantic] = str(replacements[source].resolve())
                writer.writerow(adjusted)

    # Purpose: Implement final binding for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: _full_sha, _read_json, _sha256
    def _final_binding(
        self,
        final_manifest_path: Path,
        checkpoint_path: Path,
        expected_checkpoint_sha: str,
    ) -> tuple[dict[str, Any], Path, str]:
        final_manifest_path = final_manifest_path.resolve()
        if not final_manifest_path.is_file():
            raise FileNotFoundError(f"missing final manifest: {final_manifest_path}")
        final = self._read_json(final_manifest_path)
        if str(final.get("status", "")).casefold() != "completed" or final.get("qualified") is not True:
            raise RuntimeError(
                f"candidate requires a completed qualified final, got status={final.get('status')!r}"
            )
        if str(final.get("modelSchema", "")) != MODEL_SCHEMA:
            raise RuntimeError(
                f"final manifest schema {final.get('modelSchema')!r} does not match {MODEL_SCHEMA!r}"
            )
        if str(final.get("selectionKind", "")) != "production-final":
            raise RuntimeError("final manifest selectionKind is not production-final")
        binding = final.get("checkpoint")
        if not isinstance(binding, Mapping):
            raise RuntimeError("final manifest has no checkpoint binding")
        manifest_path = Path(str(binding.get("path", "")))
        if not manifest_path.is_absolute():
            manifest_path = final_manifest_path.parent / manifest_path
        manifest_path = manifest_path.resolve()
        checkpoint_path = checkpoint_path.resolve()
        if manifest_path != checkpoint_path:
            raise RuntimeError(
                f"candidate checkpoint differs from final manifest: {checkpoint_path} != {manifest_path}"
            )
        manifest_sha = self._full_sha(binding.get("sha256"), label="final manifest checkpoint SHA")
        expected_sha = self._full_sha(expected_checkpoint_sha, label="requested checkpoint SHA")
        if manifest_sha != expected_sha:
            raise RuntimeError(
                f"requested checkpoint SHA differs from final manifest: {expected_sha} != {manifest_sha}"
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"immutable checkpoint is missing: {checkpoint_path}")
        actual_sha = self._sha256(checkpoint_path)
        if actual_sha != manifest_sha:
            raise RuntimeError(
                f"immutable checkpoint SHA mismatch: expected={manifest_sha} actual={actual_sha}"
            )
        if binding.get("immutable") is not True:
            raise RuntimeError("final manifest does not mark the checkpoint immutable")
        if checkpoint_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeError(f"immutable checkpoint is writable: {checkpoint_path}")
        architecture_path = final_manifest_path.parent / "architecture_participation.json"
        architecture = self._read_json(architecture_path)
        if architecture.get("pass") is not True:
            raise RuntimeError("architecture participation report is missing or failed")
        return final, checkpoint_path, actual_sha

    # Purpose: Implement source snapshot for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: _dimensions, _sha256
    def _source_snapshot(self, paths: Iterable[Path]) -> dict[Path, dict[str, Any]]:
        return {
            path: {
                "path": str(path),
                "sha256": self._sha256(path),
                "dimensions": self._dimensions(path),
                "sizeBytes": int(path.stat().st_size),
            }
            for path in paths
        }

    # Purpose: Implement provenance for StrategyCandidateGenerator.
    # Called by: generate_candidate
    # Calls: _dimensions, _sha256
    def _provenance(
        self,
        *,
        source_before: Mapping[Path, Mapping[str, Any]],
        replacements: Mapping[Path, Path],
        usages: Mapping[Path, list[Usage]],
        material_manifest: Path,
        asset_manifest: Path,
    ) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        primary: dict[str, Any] | None = None
        for source, candidate in sorted(replacements.items(), key=lambda item: str(item[0]).casefold()):
            before = source_before[source]
            source_sha = self._sha256(source)
            candidate_sha = self._sha256(candidate)
            semantic_names = sorted({usage.semantic for usage in usages.get(source, [])})
            record = {
                "semantics": semantic_names,
                "sourcePath": str(source),
                "sourceSha256": source_sha,
                "sourceSha256Before": str(before["sha256"]),
                "sourceSha256After": source_sha,
                "sourceDimensions": self._dimensions(source),
                "candidatePath": str(candidate),
                "candidateSha256": candidate_sha,
                "candidateDimensions": self._dimensions(candidate),
                "sourceUnchanged": (
                    source_sha == before["sha256"]
                    and int(source.stat().st_size) == int(before["sizeBytes"])
                    and self._dimensions(source) == list(before["dimensions"])
                ),
            }
            if not record["sourceUnchanged"]:
                raise RuntimeError(f"authored source changed during candidate generation: {source}")
            records.append(record)
            if primary is None and "albedo" in semantic_names:
                primary = dict(record)
        if primary is None:
            raise RuntimeError("candidate provenance contains no albedo source/output pair")
        return {
            "schema": "NSAMDR_SOURCE_CANDIDATE_PROVENANCE_V1",
            "verified": True,
            "sourceMaterialManifest": str(material_manifest),
            "sourceMaterialManifestSha256": self._sha256(material_manifest),
            "assetManifest": str(asset_manifest),
            "assetManifestSha256": self._sha256(asset_manifest),
            "primaryAlbedo": primary,
            "records": records,
        }

    # Purpose: Implement generate candidate for StrategyCandidateGenerator.
    # Called by: main
    # Calls: _apply_direct_maps, _collect_contexts, _collect_usages, _destination, _direct_maps, _final_binding, _output_size, _path_within, _provenance, _read_bgra, _read_tsv, _sha256, _source_snapshot, _write_material_manifest, _write_png
    def generate_candidate(
        self,
        *,
        obj: Path,
        materials: Path,
        asset_manifest: Path,
        output_root: Path,
        target_size: int,
        checkpoint: Path,
        checkpoint_sha256: str,
        final_manifest: Path,
        inference_device: str,
    ) -> dict[str, Any]:
        obj = obj.resolve()
        materials = materials.resolve()
        asset_manifest = asset_manifest.resolve()
        output_root = output_root.resolve()
        final_manifest = final_manifest.resolve()
        experiment_dir = final_manifest.parent
        if not self._path_within(output_root, experiment_dir / "previews"):
            raise RuntimeError(
                f"candidate output must stay inside {experiment_dir / 'previews'}: {output_root}"
            )
        for label, path in (
            ("OBJ", obj),
            ("material manifest", materials),
            ("asset manifest", asset_manifest),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} is missing: {path}")

        final, checkpoint, checkpoint_sha = self._final_binding(
            final_manifest,
            checkpoint,
            checkpoint_sha256,
        )
        device_probe = type("_DeviceConfig", (), {"device": inference_device, "cuda_device_index": 0, "matmul_precision": "high"})()
        device = resolve_device(device_probe, inference_device)
        if str(device) == "cpu":
            import torch

            torch.set_num_threads(max(1, min(16, int(os.environ.get("NSAMDR_CPU_THREADS", "2")))))
        model, config, checkpoint_payload = load_trained_model(checkpoint, device)
        if str(checkpoint_payload.get("selection_kind", "")) != "production-final":
            raise RuntimeError("immutable checkpoint selection_kind is not production-final")
        qualification = checkpoint_payload.get("final_qualification")
        if not isinstance(qualification, Mapping) or qualification.get("passed") is not True:
            raise RuntimeError("immutable checkpoint has no passing uncached final qualification")
        if self._sha256(checkpoint) != checkpoint_sha:
            raise RuntimeError("immutable checkpoint changed while loading")

        fields, rows, comments = self._read_tsv(materials)
        usages = self._collect_usages(rows, materials.parent)
        contexts = self._collect_contexts(rows, materials.parent)
        if not contexts:
            raise RuntimeError("material manifest contains no readable albedo context")
        source_before = self._source_snapshot(usages)
        canvases: dict[Path, Any] = {}
        generated_semantics: dict[Path, set[str]] = {}
        inference_records: list[dict[str, Any]] = []

        print(
            f"[candidate] Direct production inference: checkpoint={checkpoint} sha256={checkpoint_sha}",
            flush=True,
        )
        for index, (albedo, context) in enumerate(
            sorted(contexts.items(), key=lambda item: str(item[0]).casefold()),
            1,
        ):
            source = self._read_bgra(albedo)
            width, height = self._output_size(source, target_size)
            maps, diagnostics = self._direct_maps(
                albedo=albedo,
                context=context,
                model=model,
                config=config,
                device=device,
                out_width=width,
                out_height=height,
            )
            self._apply_direct_maps(
                albedo=albedo,
                context=context,
                maps=maps,
                width=width,
                height=height,
                canvases=canvases,
                semantics=generated_semantics,
            )
            inference_records.append(
                {
                    "sourceAlbedo": str(albedo),
                    "outputSize": [width, height],
                    "diagnostics": diagnostics,
                }
            )
            print(
                f"[candidate] [{index}/{len(contexts)}] {albedo.name} -> {width}x{height} "
                f"via FidelityResidualNetV9.forward",
                flush=True,
            )

        texture_dir = output_root / "final_nsamdr"
        replacements: dict[Path, Path] = {}
        for source, canvas in sorted(canvases.items(), key=lambda item: str(item[0]).casefold()):
            destination = self._destination(texture_dir, source)
            self._write_png(destination, canvas)
            replacements[source] = destination.resolve()

        candidate_materials = output_root / "final.materials.tsv"
        self._write_material_manifest(
            candidate_materials,
            fields,
            rows,
            comments,
            replacements,
            materials.parent,
        )
        candidate_obj = output_root / obj.name
        candidate_obj.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(obj, candidate_obj)

        provenance = self._provenance(
            source_before=source_before,
            replacements=replacements,
            usages=usages,
            material_manifest=materials,
            asset_manifest=asset_manifest,
        )
        provenance_path = output_root / "preview_control_provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        analysis_path = output_root / "candidate_analysis.json"
        analysis = {
            "schema": "NSAMDR_PRODUCTION_CANDIDATE_ANALYSIS_V1",
            "checkpoint": str(checkpoint),
            "checkpointSha256": checkpoint_sha,
            "modelSchema": MODEL_SCHEMA,
            "selectionKind": "production-final",
            "productionModelClass": type(model).__name__,
            "productionForward": "FidelityResidualNetV9.forward(inputs)",
            "directCheckpointOutput": True,
            "preSuperResolutionModel": False,
            "postModelBlend": False,
            "postModelBoundaryRender": False,
            "postModelReplacement": False,
            "inference": inference_records,
            "generatedSemantics": {
                str(path): sorted(values) for path, values in generated_semantics.items()
            },
        }
        analysis_path.write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        validation_path = output_root / "candidate_validation.json"
        validation = {
            "schema": "NSAMDR_PRODUCTION_CANDIDATE_VALIDATION_V1",
            "passed": True,
            "checkpointSha256": checkpoint_sha,
            "selectionKind": "production-final",
            "checks": {
                "finalManifestCompleted": final.get("status") == "completed",
                "finalManifestQualified": final.get("qualified") is True,
                "architectureParticipationPassed": True,
                "checkpointStrictLoaded": True,
                "checkpointUnchangedAfterLoad": self._sha256(checkpoint) == checkpoint_sha,
                "directProductionForward": True,
                "sourceProvenanceVerified": provenance.get("verified") is True,
                "allCandidateFilesExist": all(path.is_file() for path in replacements.values()),
                "noPostModelReplacement": True,
            },
        }
        if not all(validation["checks"].values()):
            validation["passed"] = False
            raise RuntimeError(f"candidate validation failed: {validation['checks']}")
        validation_path.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        report = {
            "schema": REPORT_SCHEMA,
            "status": "verified",
            "checkpoint": str(checkpoint),
            "checkpointSha256": checkpoint_sha,
            "selectionKind": "production-final",
            "candidateObj": str(candidate_obj.resolve()),
            "candidateMaterials": str(candidate_materials.resolve()),
            "candidateAnalysis": str(analysis_path.resolve()),
            "candidateValidation": str(validation_path.resolve()),
            "controlProvenance": provenance,
            "controlProvenancePath": str(provenance_path.resolve()),
            "candidateFiles": [
                {
                    "path": str(path),
                    "sha256": self._sha256(path),
                    "semantics": sorted(generated_semantics[source]),
                }
                for source, path in sorted(
                    replacements.items(), key=lambda item: str(item[0]).casefold()
                )
            ],
            "sourceFiles": [
                {
                    "path": str(path),
                    "sha256": str(record["sha256"]),
                }
                for path, record in sorted(
                    source_before.items(), key=lambda item: str(item[0]).casefold()
                )
            ],
            "targetSize": int(target_size),
            "inferenceDevice": str(device),
        }
        report_path = output_root / "candidate_manifest.json"
        report["reportPath"] = str(report_path.resolve())
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[candidate] VERIFIED: {report_path}", flush=True)
        return report

    # Purpose: Implement build parser for StrategyCandidateGenerator.
    # Called by: main
    # Calls: No same-class helper methods.
    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Bake a direct NSAMDR final from one immutable production checkpoint"
        )
        parser.add_argument("--obj", required=True, type=Path)
        parser.add_argument("--materials", required=True, type=Path)
        parser.add_argument("--asset-manifest", required=True, type=Path)
        parser.add_argument("--output-root", required=True, type=Path)
        parser.add_argument("--target-size", type=int, default=4096)
        parser.add_argument("--checkpoint", required=True, type=Path)
        parser.add_argument("--checkpoint-sha256", required=True)
        parser.add_argument("--final-manifest", required=True, type=Path)
        parser.add_argument(
            "--inference-device",
            choices=("auto", "cuda", "cpu"),
            default="cuda",
        )
        return parser

    # Purpose: Implement main for StrategyCandidateGenerator.
    # Called by: External callers and the owning workflow.
    # Calls: build_parser, generate_candidate
    def main(self, argv: list[str] | None = None) -> int:
        args = self.build_parser().parse_args(argv)
        if not 512 <= int(args.target_size) <= 4096:
            raise SystemExit("--target-size must be from 512 to 4096")
        try:
            self.generate_candidate(
                obj=args.obj,
                materials=args.materials,
                asset_manifest=args.asset_manifest,
                output_root=args.output_root,
                target_size=int(args.target_size),
                checkpoint=args.checkpoint,
                checkpoint_sha256=args.checkpoint_sha256,
                final_manifest=args.final_manifest,
                inference_device=args.inference_device,
            )
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1

_strategy_candidate_generator = StrategyCandidateGenerator()
_sha256 = _strategy_candidate_generator._sha256
_read_json = _strategy_candidate_generator._read_json
_path_within = _strategy_candidate_generator._path_within
_full_sha = _strategy_candidate_generator._full_sha
_read_tsv = _strategy_candidate_generator._read_tsv
_resolve_path = _strategy_candidate_generator._resolve_path
_parse_channel = _strategy_candidate_generator._parse_channel
_rgba_to_bgra = _strategy_candidate_generator._rgba_to_bgra
_collect_usages = _strategy_candidate_generator._collect_usages
_collect_contexts = _strategy_candidate_generator._collect_contexts
_require_dependencies = _strategy_candidate_generator._require_dependencies
_read_bgra = _strategy_candidate_generator._read_bgra
_dimensions = _strategy_candidate_generator._dimensions
_output_size = _strategy_candidate_generator._output_size
_resized_channel = _strategy_candidate_generator._resized_channel
_model_input = _strategy_candidate_generator._model_input
_direct_maps = _strategy_candidate_generator._direct_maps
_ensure_canvas = _strategy_candidate_generator._ensure_canvas
_apply_direct_maps = _strategy_candidate_generator._apply_direct_maps
_destination = _strategy_candidate_generator._destination
_write_png = _strategy_candidate_generator._write_png
_write_material_manifest = _strategy_candidate_generator._write_material_manifest
_final_binding = _strategy_candidate_generator._final_binding
_source_snapshot = _strategy_candidate_generator._source_snapshot
_provenance = _strategy_candidate_generator._provenance
generate_candidate = _strategy_candidate_generator.generate_candidate
build_parser = _strategy_candidate_generator.build_parser
main = _strategy_candidate_generator.main


if __name__ == "__main__":
    raise SystemExit(main())
