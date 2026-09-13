from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterator

import cv2
import numpy as np
import torch

from .config import V14Config
from .dataset import RavenSRDataset
from .model import NSAMDRV14
from .qualification import sample_metrics


DIAGNOSTIC_SCHEMA = "NSAMDR_V14_MINI_DIAGNOSTIC_V1"
DIAGNOSTIC_REVISION = "V14.2"


def device_from_name(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V14.2 diagnostic requested CUDA but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision in {"auto", "bf16"} else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def prepare_raven_dataset(args: Any, repo_root: Path) -> None:
    script = repo_root / "tools/nsamdr/neural/prepare_nsamdr_v9_raven_preview_dataset.py"
    command = [
        sys.executable,
        "-u",
        str(script),
        "--repo-root",
        str(repo_root),
        "--shared-cache",
        args.shared_cache,
        "--train-crops",
        str(args.prepare_train_regions),
        "--validation-crops",
        str(args.prepare_validation_regions),
    ]
    if args.rebuild_dataset:
        command.append("--rebuild")
    print(
        "[v14.2-diagnostic] prepare authored Raven dataset: "
        + subprocess.list2cmdline(command),
        flush=True,
    )
    result = subprocess.run(command, cwd=repo_root, check=False)
    if result.returncode:
        raise RuntimeError(
            f"Raven dataset preparation failed with exit code {result.returncode}"
        )


def make_run_directory(repo_root: Path, mode: str) -> Path:
    root = repo_root / "artifacts/nsamdr/diagnostics/v14_mini"
    root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    path = root / f"{mode}_{stamp}"
    suffix = 1
    while path.exists():
        path = root / f"{mode}_{stamp}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def archive_run(run_dir: Path) -> Path:
    return Path(
        shutil.make_archive(
            str(run_dir) + "_DIAGNOSTICS",
            "zip",
            root_dir=run_dir,
        )
    )


def to_device_batch(
    sample: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move tensor fields to the device and preserve record_index for held-out identity."""
    result: dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.unsqueeze(0) if value.ndim == 3 else value
        result[key] = tensor.to(device, non_blocking=True)
    return result


def record_key(record: dict[str, Any]) -> str:
    return str(
        record.get("cropId")
        or record.get("crop_id")
        or record.get("path")
        or "unknown"
    )


def detail_score(record: dict[str, Any]) -> float:
    try:
        return float(record.get("detailScore", record.get("detail_score", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def pseudo_manifest(
    records: list[dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    copied: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["split"] = split
        copied.append(item)
    return {"crops": copied}


def dataset_sample(
    record: dict[str, Any],
    config: V14Config,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    dataset = RavenSRDataset(
        pseudo_manifest([record], split="validation"),
        config,
        "validation",
        1,
        seed=config.seed,
        degradation="clean",
    )
    return to_device_batch(dataset[0], device)


def iter_train_batches(
    manifest: dict[str, Any],
    config: V14Config,
    count: int,
    *,
    seed: int,
    degradation: str,
    device: torch.device,
) -> Iterator[dict[str, torch.Tensor]]:
    dataset = RavenSRDataset(
        manifest,
        config,
        "train",
        count,
        seed=seed,
        degradation=degradation,
    )
    for index in range(len(dataset)):
        yield to_device_batch(dataset[index], device)


def validation_metrics(
    model: NSAMDRV14,
    manifest: dict[str, Any],
    config: V14Config,
    device: torch.device,
    precision: str,
    *,
    final: bool,
) -> list[dict[str, float]]:
    validation_count = sum(
        1
        for record in manifest["crops"]
        if record.get("split") == "validation"
    )
    dataset = RavenSRDataset(
        manifest,
        config,
        "validation",
        validation_count,
        seed=config.seed + 701,
        degradation="clean",
    )
    metrics: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = to_device_batch(dataset[index], device)
            with autocast_context(device, precision):
                outputs = model(
                    batch["lr_albedo"],
                    batch["lr_normal"],
                    batch["lr_material"],
                )
            metrics.append(sample_metrics(outputs, batch, final=final))
    return metrics


def save_probe(
    run_dir: Path,
    batch: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    *,
    include_final: bool,
) -> Path:
    def u8(value: torch.Tensor) -> np.ndarray:
        image = (
            value.detach()
            .float()
            .clamp(0.0, 1.0)[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return np.round(image * 255.0).astype(np.uint8)

    panels: list[tuple[str, np.ndarray]] = [
        ("A AUTHORED", u8(batch["target_albedo"])),
        ("B BASELINE", u8(outputs["baseline_albedo"])),
        ("C V14.2 SR", u8(outputs["candidate_albedo"])),
    ]
    if include_final:
        panels.append(("F SELECTED", u8(outputs["albedo"])))

    rendered: list[np.ndarray] = []
    for label, image in panels:
        panel = image.copy()
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(
            panel,
            label,
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rendered.append(panel)

    path = run_dir / "ABCF_probe.png"
    contact = np.concatenate(rendered, axis=1)
    cv2.imwrite(str(path), contact[:, :, ::-1])
    return path


def write_report(run_dir: Path, report: dict[str, Any]) -> Path:
    path = run_dir / "report.json"
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
