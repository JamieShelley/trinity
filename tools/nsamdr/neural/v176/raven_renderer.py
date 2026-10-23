"""EVE Raven native research-preview material exporter for active V17.6.

Model loading is deliberately external; a verified 4x model must be supplied.
No historical experiment resolver or legacy model loader is accepted.
"""
from __future__ import annotations
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
import cv2
import numpy as np
import torch

LIVE_POINTER_SCHEMA = "NSAMDR_LIVE_CANDIDATE_POINTER_V1"
DEFAULT_RAVEN = "res:/dx9/model/ship/caldari/battleship/cb1/cb1_t1.gr2"
PATH_COLUMNS = (
    "albedo", "normal", "material", "glow", "dirt",
    "ao", "paint_mask", "roughness_map",
)
@dataclass
class TextureContext:
    normal: Path | None = None
    normal_x_channel: int = 0
    normal_y_channel: int = 1
    material: Path | None = None
    material_channel: int = 0
    glow: Path | None = None
    glow_channel: int = 1
    roughness: Path | None = None
    roughness_channel: int = 2

@dataclass(frozen=True)
class PreviewSource:
    label: str
    ordinal: int
    phase: str
    checkpoint: Path

def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for render preview but is unavailable")
    return torch.device(name)


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]], list[str]]:
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


def _resolve_path(value: str, base: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    return path if path.is_file() else None


def _channel(row: Mapping[str, str], name: str, fallback: int) -> int:
    try:
        return max(0, min(3, int(float(row.get(name, str(fallback)) or fallback))))
    except (TypeError, ValueError):
        return fallback


def _collect_contexts(
    rows: list[dict[str, str]],
    base: Path,
) -> dict[Path, TextureContext]:
    contexts: dict[Path, TextureContext] = {}
    for row in rows:
        albedo = _resolve_path(row.get("albedo", ""), base)
        if albedo is None:
            continue
        context = contexts.setdefault(albedo, TextureContext())
        normal = _resolve_path(row.get("normal", ""), base)
        material = _resolve_path(row.get("material", ""), base)
        glow = _resolve_path(row.get("glow", ""), base) or material
        roughness = _resolve_path(row.get("roughness_map", ""), base) or material
        if context.normal is None and normal is not None:
            context.normal = normal
            context.normal_x_channel = _channel(row, "normal_x_channel", 0)
            context.normal_y_channel = _channel(row, "normal_y_channel", 1)
        if context.material is None and material is not None:
            context.material = material
            context.material_channel = _channel(row, "material_channel", 0)
        if context.glow is None and glow is not None:
            context.glow = glow
            context.glow_channel = _channel(row, "glow_channel", 1)
        if context.roughness is None and roughness is not None:
            context.roughness = roughness
            context.roughness_channel = _channel(row, "roughness_channel", 2)
    return contexts


def _read_bgra(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read EVE texture: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif image.shape[2] != 4:
        raise RuntimeError(f"unsupported texture channels={image.shape[2]}: {path}")
    return image


def _bgra_channel(channel: int) -> int:
    return (2, 1, 0, 3)[max(0, min(3, int(channel)))]


def _resized_channel(
    path: Path | None,
    channel: int,
    width: int,
    height: int,
    default: float,
) -> np.ndarray:
    if path is None:
        return np.full((height, width), float(default), dtype=np.float32)
    image = _read_bgra(path)
    value = image[:, :, _bgra_channel(channel)].astype(np.float32) / 255.0
    return cv2.resize(value, (width, height), interpolation=cv2.INTER_AREA)


def _normalise_xy(value: np.ndarray) -> np.ndarray:
    length = np.sqrt(np.maximum(np.sum(value * value, axis=-1, keepdims=True), 1.0e-8))
    return (value / np.maximum(1.0, length / 0.999)).astype(np.float32)


def _output_size(image: np.ndarray, scale: int) -> tuple[int, int]:
    """Native full-resolution evidence always produces exactly scale× output.

    A UI preview size may control the viewer viewport, never the model's input
    pixels. In particular, do NOT globally shrink original EVE textures.
    """
    if image.ndim != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("native EVE albedo image must have nonempty H×W×C shape")
    if scale != 4:
        raise ValueError("V17.6 supports exactly 4× native source reconstruction")
    height, width = image.shape[:2]
    out_width, out_height = width * scale, height * scale
    # DX11 D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION and renderer limit.
    if max(out_width, out_height) > 16384:
        raise RuntimeError(
            f"native source {width}x{height} requires {out_width}x{out_height} "
            "output beyond the DX11 16384-pixel texture limit; refusing to "
            "downsample source evidence or silently reduce output scale"
        )
    return out_width, out_height


def _model_inputs(
    albedo: Path,
    context: TextureContext,
    *,
    out_width: int,
    out_height: int,
    scale: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lr_width = max(1, int(math.ceil(out_width / scale)))
    lr_height = max(1, int(math.ceil(out_height / scale)))

    source = _read_bgra(albedo)
    if (lr_width, lr_height) != (source.shape[1], source.shape[0]):
        raise RuntimeError(
            f"V17.6 Raven inference would resize original source from "
            f"{source.shape[1]}x{source.shape[0]} to {lr_width}x{lr_height}; "
            "native evidence must not be discarded"
        )
    # Read native pixels verbatim; no global training-style degradation, and
    # no INTER_AREA reduction as in the old preview path.
    albedo_lr = cv2.cvtColor(
        source[:, :, :3], cv2.COLOR_BGR2RGB,
    ).astype(np.float32) / 255.0

    nx = _resized_channel(
        context.normal,
        context.normal_x_channel,
        lr_width,
        lr_height,
        0.5,
    ) * 2.0 - 1.0
    ny = _resized_channel(
        context.normal,
        context.normal_y_channel,
        lr_width,
        lr_height,
        0.5,
    ) * 2.0 - 1.0
    normal_lr = _normalise_xy(np.stack((nx, ny), axis=-1))

    material_lr = np.stack(
        (
            _resized_channel(
                context.material,
                context.material_channel,
                lr_width,
                lr_height,
                0.0,
            ),
            _resized_channel(
                context.glow,
                context.glow_channel,
                lr_width,
                lr_height,
                0.0,
            ),
            _resized_channel(
                context.roughness,
                context.roughness_channel,
                lr_width,
                lr_height,
                0.5,
            ),
        ),
        axis=-1,
    ).astype(np.float32)

    def tensor(value: np.ndarray) -> torch.Tensor:
        return (
            torch.from_numpy(np.ascontiguousarray(value))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
        )

    return tensor(albedo_lr), tensor(normal_lr), tensor(material_lr)


def _positions(size: int, tile: int, overlap: int) -> list[int]:
    if size <= tile:
        return [0]
    step = max(1, tile - overlap)
    values = list(range(0, max(1, size - tile + 1), step))
    last = size - tile
    if values[-1] != last:
        values.append(last)
    return values


def _window(height: int, width: int, device: torch.device) -> torch.Tensor:
    wy = torch.hann_window(height, periodic=False, device=device).clamp_min(0.05)
    wx = torch.hann_window(width, periodic=False, device=device).clamp_min(0.05)
    return (wy[:, None] * wx[None, :]).view(1, 1, height, width)


@torch.no_grad()
def _tiled_maps(
    model: torch.nn.Module,
    config: Any,
    lr_albedo: torch.Tensor,
    lr_normal: torch.Tensor,
    lr_material: torch.Tensor,
    *,
    device: torch.device,
    scratch: Path,
) -> dict[str, np.memmap]:
    """Tile V17.6 inference with full native input and disk-backed HR stitching.

    The old implementation allocated all HR accumulation maps on GPU/host RAM.
    A native 4096 source requires 16384 HR output; disk-mapped float32 planes
    preserve the exact Hann blend without huge monolithic allocations.
    """
    h, w = lr_albedo.shape[-2:]
    tile = int(config.production_tile_lr)
    overlap = int(config.production_overlap_lr)
    if tile < 1 or overlap < 0 or overlap * 2 >= tile:
        raise ValueError("invalid native V17.6 tile/overlap contract")
    scale = int(config.scale)
    if scale != 4:
        raise ValueError("native V17.6 reconstruction requires 4x")
    out_h, out_w = h * scale, w * scale
    channels = {
        "baseline_albedo": 3, "baseline_normal": 2,
        "baseline_material": 3, "candidate_albedo": 3,
        "candidate_normal": 2, "candidate_material": 3,
    }
    scratch.mkdir(parents=True, exist_ok=True)
    accum: dict[str, np.memmap] = {}
    for key, count in channels.items():
        plane = np.memmap(scratch / f"{key}.f32", dtype="float32", mode="w+",
                          shape=(out_h, out_w, count))
        plane[:] = 0.0
        accum[key] = plane
    weight = np.memmap(scratch / "weights.f32", dtype="float32", mode="w+",
                       shape=(out_h, out_w))
    weight[:] = 0.0
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    ys = _positions(h, tile, overlap)
    xs = _positions(w, tile, overlap)
    for y in ys:
        for x in xs:
            a = lr_albedo[..., y : y + tile, x : x + tile]
            n = lr_normal[..., y : y + tile, x : x + tile]
            m = lr_material[..., y : y + tile, x : x + tile]
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(a, n, m)
            ph, pw = outputs["candidate_albedo"].shape[-2:]
            oy, ox = y * scale, x * scale
            win = _window(ph, pw, torch.device("cpu"))[0, 0].numpy()
            for key in channels:
                tile_data = outputs[key][0].detach().float().cpu().permute(1, 2, 0).numpy()
                if tile_data.shape != (ph, pw, channels[key]):
                    raise RuntimeError(f"V17.6 map shape changed: {key}, {tile_data.shape}")
                region = accum[key][oy:oy + ph, ox:ox + pw]
                region += tile_data * win[:, :, None]
            weight[oy:oy + ph, ox:ox + pw] += win
            del outputs
    # Normalize in bounded scanline blocks to avoid materializing entire HR maps.
    for start in range(0, out_h, 256):
        stop = min(out_h, start + 256)
        divisor = np.maximum(np.asarray(weight[start:stop]), 1.0e-6)
        for key in channels:
            region = accum[key][start:stop]
            region /= divisor[:, :, None]
            if key in ("baseline_normal", "candidate_normal"):
                lengths = np.sqrt(np.maximum(np.sum(region * region, axis=2,
                                                      keepdims=True), 1.0e-8))
                region /= np.maximum(1.0, lengths / 0.999)
    weight.flush()
    weight._mmap.close()
    del weight
    for value in accum.values():
        value.flush()
    return accum

def _tensor_image(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.ndim != 3:
            raise ValueError("native preview tile map must be H×W×C")
        return value
    return (
        value.detach()
        .float()
        .cpu()[0]
        .permute(1, 2, 0)
        .numpy()
        .astype(np.float32)
    )

def _ensure_canvas(
    canvases: dict[Path, np.ndarray],
    source: Path,
    width: int,
    height: int,
) -> np.ndarray:
    existing = canvases.get(source)
    if existing is not None:
        if existing.shape[:2] != (height, width):
            raise RuntimeError(f"packed texture requested at incompatible sizes: {source}")
        return existing
    original = _read_bgra(source)
    canvas = cv2.resize(original, (width, height), interpolation=cv2.INTER_LANCZOS4)
    canvases[source] = canvas
    return canvas


def _apply_maps(
    *,
    albedo: Path,
    context: TextureContext,
    albedo_map: np.ndarray,
    normal_map: np.ndarray,
    material_map: np.ndarray,
    width: int,
    height: int,
    canvases: dict[Path, np.ndarray],
) -> None:
    albedo_canvas = _ensure_canvas(canvases, albedo, width, height)
    rgb = np.uint8(np.rint(np.clip(albedo_map, 0.0, 1.0) * 255.0))
    albedo_canvas[:, :, :3] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if context.normal is not None:
        canvas = _ensure_canvas(canvases, context.normal, width, height)
        canvas[:, :, _bgra_channel(context.normal_x_channel)] = np.uint8(
            np.rint(np.clip((normal_map[:, :, 0] + 1.0) * 127.5, 0.0, 255.0))
        )
        canvas[:, :, _bgra_channel(context.normal_y_channel)] = np.uint8(
            np.rint(np.clip((normal_map[:, :, 1] + 1.0) * 127.5, 0.0, 255.0))
        )

    for source, channel, plane in (
        (context.material, context.material_channel, material_map[:, :, 0]),
        (context.glow, context.glow_channel, material_map[:, :, 1]),
        (context.roughness, context.roughness_channel, material_map[:, :, 2]),
    ):
        if source is None:
            continue
        canvas = _ensure_canvas(canvases, source, width, height)
        canvas[:, :, _bgra_channel(channel)] = np.uint8(
            np.rint(np.clip(plane, 0.0, 1.0) * 255.0)
        )


def _write_canvases(
    output_root: Path,
    folder: str,
    suffix: str,
    canvases: Mapping[Path, np.ndarray],
) -> dict[Path, Path]:
    root = output_root / folder
    root.mkdir(parents=True, exist_ok=True)
    replacements: dict[Path, Path] = {}
    for source, canvas in sorted(canvases.items(), key=lambda item: str(item[0]).casefold()):
        token = hashlib.sha1(str(source).casefold().encode("utf-8")).hexdigest()[:10]
        destination = root / f"{source.stem}_{token}_{suffix}.png"
        if not cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"could not write render-preview texture: {destination}")
        replacements[source] = destination.resolve()
    return replacements


def _write_material_manifest(
    output: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    comments: list[str],
    replacements: Mapping[Path, Path],
    source_dir: Path,
    label: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# NSAMDR_MATERIALS_V7\n")
        handle.write(f"# PHYSICAL_CANDIDATE {label}\n")
        for comment in comments:
            if comment.startswith("#") and "NSAMDR_MATERIALS" not in comment and "PHYSICAL_CANDIDATE" not in comment:
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
            for semantic in PATH_COLUMNS:
                source = _resolve_path(row.get(semantic, ""), source_dir)
                if source in replacements:
                    adjusted[semantic] = str(replacements[source].resolve())
            writer.writerow(adjusted)


def _generate_candidate(
    *,
    repo_root: Path,
    source: PreviewSource,
    device: torch.device,
    obj_path: Path,
    materials: Path,
    model_override: tuple[torch.nn.Module, Any, dict[str, Any]] | None = None,
    preview_schema: str = "NSAMDR_V176_RAVEN_RESEARCH_PREVIEW_V1",
) -> dict[str, Any]:
    checkpoint_sha = _sha256(source.checkpoint)
    if model_override is None:
        raise RuntimeError("V17.6 preview requires an authenticated model")
    model, config, payload = model_override
    model.eval()

    fields, rows, comments = _read_tsv(materials)
    contexts = _collect_contexts(rows, materials.parent)
    if not contexts:
        raise RuntimeError("prepared EVE ship has no aligned albedo contexts")

    live_root = repo_root / "artifacts/nsamdr/render_preview" / source.label
    output_root = live_root / "candidates" / f"step_{source.ordinal:06d}_{checkpoint_sha[:12]}"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_canvases: dict[Path, np.ndarray] = {}
    candidate_canvases: dict[Path, np.ndarray] = {}
    source_dimensions: list[dict[str, int | str]] = []
    for index, (albedo, context) in enumerate(
        sorted(contexts.items(), key=lambda item: str(item[0]).casefold()),
        start=1,
    ):
        original = _read_bgra(albedo)
        width, height = _output_size(original, int(config.scale))
        source_dimensions.append({
            "source": str(albedo.resolve()),
            "nativeWidth": int(original.shape[1]),
            "nativeHeight": int(original.shape[0]),
            "outputWidth": int(width),
            "outputHeight": int(height),
        })
        print(
            f"[v176-raven] native source={original.shape[1]}x{original.shape[0]} "
            f"evidence={original.shape[1]}x{original.shape[0]} "
            f"output={width}x{height} (4x; no preview downsample)",
            flush=True,
        )
        lr_a, lr_n, lr_m = _model_inputs(
            albedo,
            context,
            out_width=width,
            out_height=height,
            scale=int(config.scale),
            device=device,
        )
        maps = _tiled_maps(
            model,
            config,
            lr_a,
            lr_n,
            lr_m,
            device=device,
        )
        _apply_maps(
            albedo=albedo,
            context=context,
            albedo_map=_tensor_image(maps["baseline_albedo"]),
            normal_map=_tensor_image(maps["baseline_normal"]),
            material_map=_tensor_image(maps["baseline_material"]),
            width=width,
            height=height,
            canvases=baseline_canvases,
        )
        _apply_maps(
            albedo=albedo,
            context=context,
            albedo_map=_tensor_image(maps["candidate_albedo"]),
            normal_map=_tensor_image(maps["candidate_normal"]),
            material_map=_tensor_image(maps["candidate_material"]),
            width=width,
            height=height,
            canvases=candidate_canvases,
        )
        print(
            f"[render-preview] {source.label} {index}/{len(contexts)} "
            f"{albedo.name}: LR {lr_a.shape[-1]} -> HR {width}x{height}",
            flush=True,
        )
        del maps, lr_a, lr_n, lr_m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_replacements = _write_canvases(
        output_root,
        "baseline_4x",
        "baseline_4x",
        baseline_canvases,
    )
    candidate_replacements = _write_canvases(
        output_root,
        "candidate_c",
        "candidate_c",
        candidate_canvases,
    )

    baseline_materials = output_root / "baseline.materials.tsv"
    candidate_materials = output_root / "candidate.materials.tsv"
    _write_material_manifest(
        baseline_materials,
        fields,
        rows,
        comments,
        baseline_replacements,
        materials.parent,
        "NSAMDR_DETERMINISTIC_4X_BASELINE",
    )
    _write_material_manifest(
        candidate_materials,
        fields,
        rows,
        comments,
        candidate_replacements,
        materials.parent,
        "NSAMDR_TRAINING_INTERMEDIATE_UNQUALIFIED_C",
    )
    candidate_obj = output_root / obj_path.name
    shutil.copy2(obj_path, candidate_obj)

    report = {
        "schema": preview_schema,
        "authority": "training-intermediate",
        "qualified": False,
        "source": source.label,
        "ordinal": int(source.ordinal),
        "phase": source.phase,
        "checkpoint": str(source.checkpoint.resolve()),
        "checkpointSha256": checkpoint_sha,
        "checkpointSchema": str(payload.get("schema") or ""),
        "baselineObj": str(candidate_obj.resolve()),
        "baselineMaterials": str(baseline_materials.resolve()),
        "candidateObj": str(candidate_obj.resolve()),
        "candidateMaterials": str(candidate_materials.resolve()),
        "resolutionPolicy": "native_source_exact_4x",
        "nativeSourceDimensions": source_dimensions,
    }
    report_path = output_root / "candidate_manifest.json"
    report["reportPath"] = str(report_path.resolve())
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if _sha256(source.checkpoint) != checkpoint_sha:
        raise RuntimeError("render-preview checkpoint changed during candidate generation")
    return report


def _pointer_text(report: Mapping[str, Any]) -> str:
    token = (
        f"{report['source']}-{int(report['ordinal']):06d}-"
        f"{str(report['checkpointSha256'])[:16]}"
    )
    return "\n".join(
        (
            LIVE_POINTER_SCHEMA,
            f"token={token}",
            f"epoch={report['ordinal']}",
            f"phase={report['phase']}",
            f"stageVariant=C-candidate",
            f"checkpointSha256={report['checkpointSha256']}",
            f"baselineObj={report['baselineObj']}",
            f"baselineMaterials={report['baselineMaterials']}",
            f"candidateObj={report['candidateObj']}",
            f"candidateMaterials={report['candidateMaterials']}",
            f"candidateManifest={report['reportPath']}",
            "authority=training-intermediate",
            "qualified=false",
            "",
        )
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        suffix=".tmp",
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as temporary:
        temporary.write(content)
        temporary.flush()
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


