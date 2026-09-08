#!/usr/bin/env python3
"""Fast non-promotable local-capacity proof on one authored Raven patch."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any
import zipfile

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v9.application.backend import TrainingBackend
from v9.application.configuration import CANONICAL_SEMANTIC_OVERRIDES, DATASET_SCOPE_FIELDS
from v9.config import V9Config
from v9.contours import contour_targets
from v9.dataset import (
    _edge_energy_map,
    _integral_image,
    _integral_window_mean,
    _pack_sample,
    _renormalize_normal,
    load_dataset_manifest,
)
from v9.inference import resolve_device
from v9.model import FidelityResidualNetV9, MODEL_SCHEMA, parameter_count


PRODUCTION_CONFIG = "tools/nsamdr/neural/configs/v9_fidelity_full.json"
RAVEN_CONFIG = "tools/nsamdr/neural/configs/v9_preview_raven.json"
REPORT_SCHEMA = "NSAMDR_RAVEN_MICRO_CAPACITY_V1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overfit the exact production NSAMDR architecture on one tiny fixed authored "
            "Raven region. Diagnostic only; never promotable."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--required-recovery", type=float, default=0.50)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--rebuild-dataset", action="store_true")
    parser.add_argument("--open-result", action="store_true")
    return parser


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "trackedChanges": [
            line for line in run("status", "--porcelain", "--untracked-files=no").splitlines() if line
        ],
    }


def _ensure_raven_dataset(
    root: Path,
    raven_config_path: Path,
    shared_cache: str,
    rebuild: bool,
) -> None:
    raven = V9Config.load(raven_config_path)
    manifest_path = (root / raven.dataset_manifest).resolve()
    if manifest_path.is_file() and not rebuild:
        return

    command = [
        sys.executable,
        "-u",
        str(root / "tools/nsamdr/neural/prepare_nsamdr_v9_raven_preview_dataset.py"),
        "--repo-root",
        str(root),
        "--config",
        str(raven_config_path),
        "--shared-cache",
        shared_cache,
        "--train-crops",
        "16",
        "--validation-crops",
        "4",
    ]
    if rebuild:
        command.append("--rebuild")
    print("[micro] preparing fixed Raven dataset: " + subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(command, cwd=root, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Raven dataset preparation failed with exit code {completed.returncode}")


def _micro_config(root: Path, args: argparse.Namespace) -> tuple[V9Config, V9Config]:
    production = V9Config.load(root / PRODUCTION_CONFIG)
    raven = V9Config.load(root / RAVEN_CONFIG)
    for field in DATASET_SCOPE_FIELDS:
        setattr(production, field, getattr(raven, field))
    for field, value in CANONICAL_SEMANTIC_OVERRIDES.items():
        setattr(production, field, value)

    production.tile_size = int(args.tile_size)
    production.inference_tile_size = int(args.tile_size)
    production.inference_overlap = min(
        int(production.inference_overlap), max(4, int(args.tile_size) // 4)
    )
    production.tiles_per_epoch = max(4, int(args.steps_per_epoch))
    production.validation_tiles = 2
    production.batch_size = 1
    production.synthetic_geometry_probability = 0.0
    production.data_loader_workers = 0
    production.data_loader_persistent_workers = False
    production.cuda_prefetch = False
    production.amp_dtype = str(args.amp_precision)
    production.validate()
    if production.tile_size != int(args.tile_size):
        raise RuntimeError(
            f"micro tile size {args.tile_size} is invalid; resolved to {production.tile_size}"
        )
    return production, raven


def _codec_formats(bundle: Any) -> dict[str, str]:
    values: dict[str, str] = {}
    if "metadata" not in bundle:
        return values
    try:
        metadata = json.loads(str(np.asarray(bundle["metadata"]).reshape(-1)[0]))
        sources = metadata.get("sources", {}) if isinstance(metadata, dict) else {}
        for role in ("albedo", "normal", "material"):
            source = sources.get(role) if isinstance(sources, dict) else None
            if isinstance(source, dict) and source.get("format"):
                values[role] = str(source["format"])
    except (ValueError, TypeError, json.JSONDecodeError):
        pass
    return values


def _record_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, str]]:
    with np.load(record["path"], allow_pickle=False) as bundle:
        albedo_u8 = np.asarray(bundle["albedo"], dtype=np.uint8)
        normal_u8 = np.asarray(bundle["normal"], dtype=np.uint8)
        material_u8 = np.asarray(bundle["material"], dtype=np.uint8)
        material_valid = float(
            np.asarray(bundle.get("material_valid", np.asarray([0.0], dtype=np.float32)))
            .reshape(-1)[0]
        )
        formats = _codec_formats(bundle)

    albedo = albedo_u8[..., :3].astype(np.float32) / 255.0
    normal = normal_u8[..., :2].astype(np.float32) / 127.5 - 1.0
    normal = _renormalize_normal(normal)
    material = material_u8[..., :3].astype(np.float32) / 255.0
    return albedo, normal, material, material_valid, formats


def _select_hard_patch(
    manifest: dict[str, Any],
    config: V9Config,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Choose one deterministic high-information authored Raven region and freeze it."""
    target = int(config.tile_size) * int(config.target_scale)
    best: tuple[float, int, int, dict[str, Any], tuple[np.ndarray, ...], float, dict[str, str]] | None = None
    records = [record for record in manifest["crops"] if str(record.get("split")) == "train"]
    if not records:
        raise RuntimeError("fixed Raven manifest has no training crops")

    stride = max(8, target // 4)
    for record in records:
        albedo, normal, material, valid, formats = _record_arrays(record)
        if min(albedo.shape[:2]) < target:
            continue
        energy = _edge_energy_map(albedo, normal, material, valid)
        integral = _integral_image(energy)
        max_x = albedo.shape[1] - target
        max_y = albedo.shape[0] - target
        xs = list(range(0, max_x + 1, stride))
        ys = list(range(0, max_y + 1, stride))
        if xs[-1] != max_x:
            xs.append(max_x)
        if ys[-1] != max_y:
            ys.append(max_y)
        for y in ys:
            for x in xs:
                score = float(_integral_window_mean(integral, x, y, target))
                if best is None or score > best[0]:
                    best = (
                        score,
                        x,
                        y,
                        record,
                        (albedo, normal, material),
                        valid,
                        formats,
                    )

    if best is None:
        raise RuntimeError(
            f"no authored Raven crop is large enough for {target}x{target} HR micro patch"
        )

    score, x, y, record, arrays, valid, formats = best
    albedo, normal, material = arrays
    albedo_hr = np.ascontiguousarray(albedo[y : y + target, x : x + target])
    normal_hr = np.ascontiguousarray(normal[y : y + target, x : x + target])
    material_hr = np.ascontiguousarray(material[y : y + target, x : x + target])
    sdf, orientation, edge = contour_targets(albedo_hr, normal_hr, material_hr, valid)

    # One fixed RNG means the exact same authored A and degraded LR evidence B are
    # presented for every optimizer step. That is intentional: this is an overfit
    # capacity proof, not a generalisation or production qualification run.
    sample = _pack_sample(
        albedo_hr,
        normal_hr,
        material_hr,
        valid,
        sdf,
        orientation,
        edge,
        config,
        random.Random(int(config.seed) + 404_911),
        geometry_exact=0.0,
        codec_formats=formats,
    )
    metadata = {
        "recordPath": str(record.get("path", "")),
        "recordFamily": record.get("family"),
        "recordSplit": record.get("split"),
        "xHr": int(x),
        "yHr": int(y),
        "widthHr": target,
        "heightHr": target,
        "widthLr": int(config.tile_size),
        "heightLr": int(config.tile_size),
        "edgeEnergyScore": score,
        "selection": "deterministic highest edge-energy authored Raven window",
        "augmentation": False,
        "syntheticGeometry": False,
        "degradationSeed": int(config.seed) + 404_911,
    }
    return sample, metadata


def _batch(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.unsqueeze(0) for key, value in sample.items()}


def _candidate(outputs: dict[str, torch.Tensor], phase: str) -> torch.Tensor:
    if phase in {"sdf-bootstrap", "sdf-proof", "gate-proof"}:
        return outputs.get("boundary_reconstructed_albedo", outputs["albedo"])
    return outputs["albedo"]


def _metrics(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    phase: str,
) -> dict[str, float]:
    target = batch["target_albedo"].detach().float()
    baseline = outputs["baseline_albedo"].detach().float()
    candidate = _candidate(outputs, phase).detach().float()
    edge = batch["target_edge"].detach().float()
    diff_b = (baseline - target).abs().mean(dim=1, keepdim=True)
    diff_c = (candidate - target).abs().mean(dim=1, keepdim=True)
    mask = edge >= 0.08
    if bool(mask.any().item()):
        baseline_edge = float(diff_b[mask].mean().item())
        candidate_edge = float(diff_c[mask].mean().item())
    else:
        baseline_edge = float(diff_b.mean().item())
        candidate_edge = float(diff_c.mean().item())
    baseline_global = float(diff_b.mean().item())
    candidate_global = float(diff_c.mean().item())
    return {
        "baselineMae": baseline_global,
        "candidateMae": candidate_global,
        "globalRecovery": (baseline_global - candidate_global) / max(baseline_global, 1.0e-8),
        "baselineEdgeMae": baseline_edge,
        "candidateEdgeMae": candidate_edge,
        "edgeRecovery": (baseline_edge - candidate_edge) / max(baseline_edge, 1.0e-8),
    }


def _evaluate(
    service: Any,
    model: FidelityResidualNetV9,
    batch: dict[str, torch.Tensor],
    config: V9Config,
    phase: str,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, float]]:
    model.eval()
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=use_amp
    ):
        outputs = service._forward_for_phase(model, batch, phase, config)
    return outputs, _metrics(outputs, batch, phase)


def _rgb(tensor: torch.Tensor) -> np.ndarray:
    value = tensor[0].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.uint8(np.round(value * 255.0))


def _write_abc(
    path: Path,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    phase: str,
    metrics: dict[str, float],
    title: str,
) -> None:
    from PIL import Image, ImageDraw

    target = _rgb(batch["target_albedo"])
    baseline = _rgb(outputs["baseline_albedo"])
    candidate = _rgb(_candidate(outputs, phase))
    panels = (target, baseline, candidate)
    labels = (
        "A AUTHORED MICRO TARGET",
        f"B 4X BASELINE  edgeMAE={metrics['baselineEdgeMae']:.5f}",
        f"C NSAMDR MICRO  edgeMAE={metrics['candidateEdgeMae']:.5f}",
    )
    h, w = target.shape[:2]
    header = 54
    canvas = Image.new("RGB", (w * 3, h + header), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = index * w
        canvas.paste(Image.fromarray(panel, mode="RGB"), (x, header))
        draw.text((x + 6, 7), label, fill=(245, 245, 245))
    draw.text(
        (6, 28),
        f"{title} | edge recovery={metrics['edgeRecovery']:+.1%} | global recovery={metrics['globalRecovery']:+.1%}",
        fill=(220, 220, 220),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _zip_diagnostics(directory: Path) -> Path:
    target = directory.parent / f"{directory.name}_DIAGNOSTICS.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() not in {".pt", ".pth"}:
                archive.write(path, path.relative_to(directory))
    return target


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    if int(args.tile_size) < 32 or int(args.tile_size) % 16 != 0:
        raise SystemExit("--tile-size must be >=32 and divisible by 16")
    if int(args.steps_per_epoch) < 4:
        raise SystemExit("--steps-per-epoch must be >=4")
    if not 0.0 < float(args.required_recovery) < 1.0:
        raise SystemExit("--required-recovery must be between 0 and 1")

    _ensure_raven_dataset(
        root, root / RAVEN_CONFIG, args.shared_cache, bool(args.rebuild_dataset)
    )
    config, _raven = _micro_config(root, args)
    manifest = load_dataset_manifest(root, config)
    sample, patch_metadata = _select_hard_patch(manifest, config)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "artifacts/nsamdr/micro_diagnostics" / f"MICRO_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "selected_patch.json").write_text(
        json.dumps(patch_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    resolved = config.to_dict()
    resolved["diagnosticMode"] = "raven-micro-overfit"
    resolved["promotable"] = False
    resolved["stepsPerEpoch"] = int(args.steps_per_epoch)
    resolved["requiredEdgeRecovery"] = float(args.required_recovery)
    (output_dir / "resolved_micro_config.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("=" * 78, flush=True)
    print("NSAMDR RAVEN MICRO CAPACITY PROOF", flush=True)
    print("Authority                : diagnostic only / NON-PROMOTABLE", flush=True)
    print(f"Patch                    : {config.tile_size} LR -> {config.tile_size * 4} HR", flush=True)
    print(f"Selection                : {patch_metadata['selection']}", flush=True)
    print(f"Full production epochs   : {config.total_epochs}", flush=True)
    print(f"Repeated steps / epoch   : {int(args.steps_per_epoch)}", flush=True)
    print(f"Required edge recovery   : {float(args.required_recovery):.0%}", flush=True)
    print(f"Artifacts                : {output_dir}", flush=True)
    print("=" * 78, flush=True)

    backend = TrainingBackend()
    import v9.training as training

    service = training._training_service
    device = resolve_device(config, args.device)
    service._configure_cuda(config, device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model = FidelityResidualNetV9(config).to(device)
    service._validate_v992_architecture_contract(model.architecture_contract())
    if device.type == "cuda" and config.channels_last:
        model = service._convert_model_channels_last(model)
    optimizer, optimizer_mode = service._build_optimizer(model, config, device)
    amp_dtype = service._resolve_amp_dtype(config, device)
    use_amp = device.type == "cuda"
    use_scaler = use_amp and amp_dtype == torch.float16
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_scaler, init_scale=config.amp_initial_scale
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_scaler, init_scale=config.amp_initial_scale
        )

    moved = service._move_batch(
        _batch(sample),
        device,
        channels_last=bool(config.channels_last and use_amp),
    )
    model.set_phase("sdf-bootstrap")
    initial_outputs, initial_metrics = _evaluate(
        service, model, moved, config, "sdf-bootstrap", device, amp_dtype
    )
    _write_abc(
        output_dir / "micro_abc_initial.png",
        initial_outputs,
        moved,
        "sdf-bootstrap",
        initial_metrics,
        "INITIAL",
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    previous_phase = ""
    for epoch in range(1, config.total_epochs + 1):
        phase = service._phase_for_epoch(epoch, config)
        model.set_phase(phase)
        if phase == "sdf-proof" and hasattr(model, "set_parametric_substage"):
            model.set_parametric_substage("integration")

        learning_rate = service._phase_lr(phase, config, epoch)
        canonical_steps = int(args.steps_per_epoch)
        if phase == "seam-proof":
            # Canonical trainer gives the seam proof 12 authored samples x 8 passes.
            learning_rate = 3.0e-3
            canonical_steps = max(canonical_steps, 96)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * float(group.get("lr_scale", 1.0))

        if phase != previous_phase:
            print(f"[micro] phase -> {phase}", flush=True)
            previous_phase = phase
        print(
            f"Epoch {epoch:03d}/{config.total_epochs:03d} phase={phase} lr={learning_rate:.7f}",
            flush=True,
        )
        model.train()
        epoch_started = time.perf_counter()
        running = 0.0
        for step in range(1, canonical_steps + 1):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                outputs = service._forward_for_phase(model, moved, phase, config)
            with torch.autocast(device_type=device.type, enabled=False):
                losses = training.compute_losses(outputs, moved, config, phase)
            total = losses["total"].float()
            if not bool(torch.isfinite(total).item()):
                raise RuntimeError(
                    f"micro capacity proof produced non-finite loss at epoch {epoch} step {step}"
                )
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip_norm
            )
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError(
                    f"micro capacity proof produced non-finite gradient at epoch {epoch} step {step}"
                )
            scaler.step(optimizer)
            scaler.update()
            if phase == "sdf-proof":
                structure = getattr(model.geometry_net, "production_structure", None)
                restore = getattr(structure, "restore_locked_topology_parameters", None)
                if callable(restore):
                    restore()
            running += float(total.detach().cpu().item())

            if step == 1 or step % max(1, canonical_steps // 8) == 0 or step == canonical_steps:
                elapsed = max(time.perf_counter() - epoch_started, 1.0e-6)
                rate = step / elapsed
                step_ms = (time.perf_counter() - step_started) * 1000.0
                print(
                    f"  {step:5d}/{canonical_steps:5d} total={running / step:.5f} "
                    f"step={step_ms:.1f}ms rate={rate:.2f}tile/s",
                    flush=True,
                )

        TrainingBackend._prepare_production_runtime(model)
        epoch_outputs, epoch_metrics = _evaluate(
            service, model, moved, config, phase, device, amp_dtype
        )
        row = {
            "epoch": epoch,
            "phase": phase,
            "steps": canonical_steps,
            "loss": running / max(canonical_steps, 1),
            **epoch_metrics,
            "elapsedSeconds": time.perf_counter() - epoch_started,
        }
        rows.append(row)
        print(
            f"[micro] epoch {epoch:03d} {phase}: "
            f"B edgeMAE={epoch_metrics['baselineEdgeMae']:.6f} "
            f"C edgeMAE={epoch_metrics['candidateEdgeMae']:.6f} "
            f"recovery={epoch_metrics['edgeRecovery']:+.1%}",
            flush=True,
        )
        _write_abc(
            output_dir / "micro_abc_current.png",
            epoch_outputs,
            moved,
            phase,
            epoch_metrics,
            f"EPOCH {epoch:03d} {phase}",
        )

    TrainingBackend._prepare_production_runtime(model)
    model.set_phase("physical-finetune")
    final_outputs, final_metrics = _evaluate(
        service, model, moved, config, "physical-finetune", device, amp_dtype
    )
    final_sheet = output_dir / "micro_abc_final.png"
    _write_abc(
        final_sheet,
        final_outputs,
        moved,
        "physical-finetune",
        final_metrics,
        "FINAL LOCAL CAPACITY",
    )
    _write_csv(output_dir / "metrics.csv", rows)

    required = float(args.required_recovery)
    passed = bool(
        math.isfinite(final_metrics["edgeRecovery"])
        and final_metrics["edgeRecovery"] >= required
        and final_metrics["candidateEdgeMae"] < final_metrics["baselineEdgeMae"]
        and final_metrics["candidateMae"] < final_metrics["baselineMae"]
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "passed" if passed else "failed-capacity-proof",
        "selectionKind": "diagnostic-non-promotable",
        "promotable": False,
        "qualifiedProductionCheckpoint": False,
        "startedUtc": _utc_now(),
        "modelSchema": MODEL_SCHEMA,
        "parameterCount": parameter_count(model),
        "device": str(device),
        "optimizer": optimizer_mode,
        "ampDtype": str(amp_dtype).replace("torch.", ""),
        "sourceRevision": _git_state(root),
        "patch": patch_metadata,
        "schedule": {
            "identityEpochs": config.identity_epochs,
            "residualEpochs": config.residual_epochs,
            "seamProofEpochs": config.seam_proof_epochs,
            "seamAuthorityEpochs": config.seam_authority_epochs,
            "boundaryEpochs": config.boundary_epochs,
            "detailEpochs": config.detail_epochs,
            "physicalFinetuneEpochs": config.physical_finetune_epochs,
            "totalEpochs": config.total_epochs,
            "repeatedStepsPerEpoch": int(args.steps_per_epoch),
            "seamProofMinimumSteps": 96,
        },
        "initial": initial_metrics,
        "final": final_metrics,
        "requiredEdgeRecovery": required,
        "capacityProofPass": passed,
        "interpretation": (
            "local production architecture/loss capacity demonstrated on one fixed authored Raven region"
            if passed
            else "local capacity not demonstrated; do not infer that more whole-Raven training will fix quality"
        ),
        "elapsedSeconds": time.perf_counter() - started,
        "finalAbc": str(final_sheet),
    }
    report_path = output_dir / "micro_capacity_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    diagnostics = _zip_diagnostics(output_dir)
    result_pointer = root / "artifacts/nsamdr/gui/last_nsamdr_micro_result.json"
    result_pointer.parent.mkdir(parents=True, exist_ok=True)
    result_pointer.write_text(
        json.dumps(
            {
                "status": report["status"],
                "capacityProofPass": passed,
                "directory": str(output_dir),
                "report": str(report_path),
                "diagnostics": str(diagnostics),
                "finalAbc": str(final_sheet),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 78, flush=True)
    print(f"MICRO CAPACITY PROOF     : {'PASS' if passed else 'FAIL'}", flush=True)
    print(
        f"B edge MAE              : {final_metrics['baselineEdgeMae']:.6f}", flush=True
    )
    print(
        f"C edge MAE              : {final_metrics['candidateEdgeMae']:.6f}", flush=True
    )
    print(f"Edge recovery           : {final_metrics['edgeRecovery']:+.1%}", flush=True)
    print(f"Required recovery       : {required:.1%}", flush=True)
    print(f"A/B/C                   : {final_sheet}", flush=True)
    print(f"Diagnostics             : {diagnostics}", flush=True)
    print("Production promotion    : FORBIDDEN (diagnostic checkpoint only)", flush=True)
    print("=" * 78, flush=True)

    if args.open_result and os.name == "nt":
        try:
            os.startfile(str(final_sheet))  # type: ignore[attr-defined]
        except OSError as error:
            print(f"[micro] could not open A/B/C sheet: {error}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
