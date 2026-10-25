"""NSAMDR G0 real-EVE golden-patch evidence and independently curated contours.

No input texture is downsampled. No apparent clean target is fabricated from
the source, and no prior V17.6/V17.7 training/proof artifacts are modified.
All coordinates in trace annotations are relative to the native crop.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SCHEMA = "NSAMDR_V178_GOLDEN_PATCH_V1"
TRACE_SCHEMA = "NSAMDR_V178_INDEPENDENT_CONTOUR_TRACE_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_png(path: Path, data: np.ndarray) -> None:
    if not cv2.imwrite(str(path), data, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError(f"could not save golden-patch image {path}")


def _read_image(path: Path) -> np.ndarray:
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is None:
        raise RuntimeError(f"could not read golden-patch image {path}")
    return data


def create(
    manifest: Path, output_root: Path, *,
    crop_x: int = 320, crop_y: int = 448, size: int = 256,
    source_name: str = "",
) -> dict[str, Any]:
    from tools.nsamdr.neural.v176 import raven_renderer as eve
    if not 64 <= size <= 512:
        raise ValueError("golden patch size must be 64..512 native pixels")
    fields, rows, _ = eve._read_tsv(manifest)
    contexts = eve._collect_contexts(rows, manifest.parent)
    matches = [
        (source, context)
        for source, context in sorted(contexts.items(), key=lambda item: str(item[0]))
        if not source_name or source.name == source_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one Raven albedo, got {len(matches)}; "
            f"select --source-name from {[p.name for p in contexts]}"
        )
    albedo, context = matches[0]
    source = eve._read_bgra(albedo)
    h, w = source.shape[:2]
    if min(crop_x, crop_y) < 0 or crop_x + size > w or crop_y + size > h:
        raise ValueError(f"native ROI ({crop_x},{crop_y},{size}) outside {w}x{h}")
    directory = (
        output_root / f"{albedo.stem}_x{crop_x}_y{crop_y}_s{size}"
    )
    report_path = directory / "evidence.json"
    input_hash = sha256(albedo)
    identity = {
        "source": str(albedo.resolve()),
        "sourceSha256": input_hash,
        "sourceDimensions": [w, h],
        "nativeRoi": [crop_x, crop_y, size, size],
        "outputScale": 4,
        "materialPolicy": context.material_policy,
        "channelLayout": {
            "normalX": context.normal_x_channel,
            "normalY": context.normal_y_channel,
            "material": context.material_channel,
            "glow": context.glow_channel,
            "roughness": context.roughness_channel,
        },
    }
    if report_path.is_file():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise RuntimeError("existing golden patch has different source identity")
        for name, stored in existing.get("assets", {}).items():
            path = directory / name
            if not path.is_file() or sha256(path) != stored.get("sha256"):
                raise RuntimeError(f"golden patch source evidence changed: {name}")
        return existing
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("nonempty golden-patch directory lacks verified evidence")
    directory.mkdir(parents=True, exist_ok=True)

    # RGB is preserved in the authored native pixels. Zoom and baseline
    # are visualizations, not clean training labels.
    albedo_native = source[crop_y:crop_y+size, crop_x:crop_x+size].copy()
    native_path = directory / "native_albedo.png"
    _write_png(native_path, albedo_native)
    zoom_path = directory / "native_nearest4x.png"
    _write_png(
        zoom_path, cv2.resize(albedo_native, (size*4, size*4),
                              interpolation=cv2.INTER_NEAREST)
    )
    baseline_path = directory / "baseline_bicubic4x.png"
    _write_png(
        baseline_path, cv2.resize(albedo_native, (size*4, size*4),
                                  interpolation=cv2.INTER_CUBIC)
    )
    assets: dict[str, dict[str, Any]] = {}
    for item, role in (
        (albedo, "albedo"), (context.normal, "normal"),
        (context.material, "material"), (context.glow, "glow"),
        (context.roughness, "roughness"),
    ):
        if item is None:
            continue
        image = source if item == albedo else eve._read_bgra(item)
        ih, iw = image.shape[:2]
        # Use the SAME UV rectangle in each map's own native pixel grid.
        x0 = int(round(crop_x / w * iw))
        y0 = int(round(crop_y / h * ih))
        x1 = int(round((crop_x+size) / w * iw))
        y1 = int(round((crop_y+size) / h * ih))
        x1 = max(x0+1, min(iw, x1))
        y1 = max(y0+1, min(ih, y1))
        name = f"native_{role}.png"
        if role != "albedo":
            _write_png(directory / name, image[y0:y1, x0:x1])
        assets[name] = {
            "sha256": sha256(directory / name),
            "source": str(item.resolve()),
            "sourceSha256": sha256(item),
            "sourceDimensions": [iw, ih],
            "nativeSourceRoi": [x0, y0, x1-x0, y1-y0],
            "pixelResize": False,
        }
    for name in (zoom_path.name, baseline_path.name):
        assets[name] = {
            "sha256": sha256(directory / name),
            "visualizationOnly": True,
            "cleanTarget": False,
        }
    trace_template = {
        "schema": TRACE_SCHEMA,
        "sourceSha256": input_hash,
        "author": "",
        "method": "independent-human-contour-trace",
        "paths": [],
        "notes": "Trace authentic line centrelines. No target is inferred from pixel edges.",
    }
    (directory / "trace.template.json").write_text(
        json.dumps(trace_template, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema": SCHEMA, "decision": "AWAITING_CURATED_CONTOUR_REFERENCE",
        "identity": identity, "assets": assets,
        "referenceStatus": "pending-human-trace",
        "referenceIsSynthetic": False, "trainingAccessAllowed": False,
        "sealedHoldoutAccess": False, "productionPromotionAllowed": False,
        "referenceTemplate": str((directory/"trace.template.json").resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n",
                           encoding="utf-8")
    return report


def validate_trace(folder: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = json.loads((folder/"evidence.json").read_text(encoding="utf-8"))
    if evidence.get("schema") != SCHEMA:
        raise ValueError("unknown golden-patch evidence schema")
    if not (folder/"trace.json").is_file():
        raise RuntimeError("human-curated trace.json is required; template is not a target")
    trace = json.loads((folder/"trace.json").read_text(encoding="utf-8"))
    if trace.get("schema") != TRACE_SCHEMA:
        raise ValueError("unknown contour annotation schema")
    if trace.get("sourceSha256") != evidence["identity"]["sourceSha256"]:
        raise ValueError("trace source identity mismatch")
    if not str(trace.get("author", "")).strip():
        raise ValueError("independently traced reference needs an author")
    if trace.get("method") != "independent-human-contour-trace":
        raise ValueError("unsupported contour provenance")
    paths = trace.get("paths", [])
    if not isinstance(paths, list) or not paths:
        raise RuntimeError("no independently traced contours; qualification blocked")
    size = int(evidence["identity"]["nativeRoi"][2])
    if len(paths) > 128:
        raise ValueError("trace contains too many shapes")
    for entry in paths:
        points = entry.get("points", [])
        if not (2 <= len(points) <= 256 and
                0.10 <= float(entry.get("widthNative", 0)) <= 8):
            raise ValueError("invalid traced contour length or width")
        for point in points:
            if (not isinstance(point, list) or len(point) != 2
                    or not all(isinstance(v, (int, float)) and
                               np.isfinite(v) and 0 <= v < size for v in point)):
                raise ValueError("trace point outside native patch")
    for name, spec in evidence["assets"].items():
        if sha256(folder/name) != spec["sha256"]:
            raise RuntimeError(f"native evidence changed since tracing: {name}")
    return evidence, trace


def render_contour_trace(folder: Path) -> Path:
    """Render explicitly traced geometry at 4×; NEVER infer contour from LR."""
    evidence, trace = validate_trace(folder)
    size = int(evidence["identity"]["nativeRoi"][2])
    scale, ss = 4, 2
    output = np.zeros((size*scale*ss, size*scale*ss), dtype=np.uint8)
    for path in trace["paths"]:
        points = np.array(path["points"], dtype=np.float64)
        # Use interpolating Catmull–Rom for 3+ independently authored control
        # points; preserve straightness for two-point traces. Never estimate
        # control points from the jagged authored raster.
        interp = []
        for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
            length = float(np.linalg.norm(end-start))
            steps = max(2, int(np.ceil(length * scale * ss * 2)))
            t = np.linspace(0, 1, steps, endpoint=False)[:, None]
            if len(points) <= 2:
                segment = start[None,:]*(1-t)+end[None,:]*t
            else:
                p0 = points[max(0,index-1)]
                p1, p2 = start, end
                p3 = points[min(len(points)-1,index+2)]
                segment = .5*((2*p1)[None,:] + (-p0+p2)[None,:]*t
                              + (2*p0-5*p1+4*p2-p3)[None,:]*(t*t)
                              + (-p0+3*p1-3*p2+p3)[None,:]*(t*t*t))
            interp.append(segment)
        interp.append(points[-1:])
        vertices = np.rint(
            np.clip(np.concatenate(interp), 0, size-1) * scale * ss
        ).astype(np.int32)
        width = max(1, int(round(float(path["widthNative"])*scale*ss)))
        cv2.polylines(output,[vertices.reshape(-1,1,2)],False,255,width,
                      lineType=cv2.LINE_AA)
    mask = cv2.resize(output, (size*4,size*4), interpolation=cv2.INTER_AREA)
    destination = folder/"curated_contours_4x.png"
    _write_png(destination,mask)
    # Protect all source features sufficiently far from the independently
    # traced centreline. The network must eventually preserve these regions.
    radius = max(3,int(round(2*scale)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)
    )
    protected = 255 - cv2.dilate(mask,kernel)
    protected_path = folder/"protected_regions_4x.png"
    _write_png(protected_path, protected)
    provenance = {
        "schema": TRACE_SCHEMA, "author": trace["author"],
        "sourceSha256": evidence["identity"]["sourceSha256"],
        "traceSha256": sha256(folder/"trace.json"),
        "contourSha256": sha256(destination),
        "protectedRegions": str(protected_path.resolve()),
        "protectedRegionsSha256": sha256(protected_path),
        "cleanReferenceType": "independently-traced-contours",
        "referenceIsNetworkGenerated": False,
        "qualifiedForRealEveTraining": False,
        "productionPromotionAllowed": False,
    }
    (folder/"curated_contours.provenance.json").write_text(
        json.dumps(provenance,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    return destination
