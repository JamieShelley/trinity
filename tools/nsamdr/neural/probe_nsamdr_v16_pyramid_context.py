#!/usr/bin/env python3
"""Bounded V16 wider-LR-context proof from the latest Main checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from probe_nsamdr_v16_full_broad import (
    CHECKPOINT_SCHEMA,
    _autocast,
    _device,
    _evaluate,
    _loss_values,
    _proof_loss_terms,
    _to_device,
)
from v14.config import V16Config
from v16.broad_prior import AuthorityBalancedSRDataset
from v16.pyramid_context import PyramidContextV16Candidate


SCHEMA = "NSAMDR_V16_PYRAMID_CONTEXT_PROBE_V1"
LATEST_POINTER = "artifacts/nsamdr/main_training/latest.json"


def _latest_source(repo_root: Path) -> tuple[Path, dict]:
    pointer_path = repo_root / LATEST_POINTER
    if not pointer_path.is_file():
        raise RuntimeError(f"Main V16 latest pointer is missing: {pointer_path}")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    raw = str(pointer.get("resumeCheckpoint") or "").strip()
    if not raw:
        raise RuntimeError("Main V16 latest pointer has no resume checkpoint")
    checkpoint = Path(raw)
    if not checkpoint.is_absolute():
        checkpoint = repo_root / checkpoint
    if not checkpoint.is_file():
        raise RuntimeError(f"Main V16 checkpoint is missing: {checkpoint}")
    return checkpoint.resolve(), pointer


def _load_candidate(
    checkpoint: Path,
    device: torch.device,
) -> tuple[PyramidContextV16Candidate, V16Config, dict]:
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(f"full-broad checkpoint schema mismatch: {checkpoint}")

    raw_config = dict(payload.get("config") or {})
    fields = V16Config.__dataclass_fields__
    config = V16Config(**{k: v for k, v in raw_config.items() if k in fields})
    config.validate()
    model = PyramidContextV16Candidate(
        config,
        structure_channels=48,
        structure_blocks=4,
    ).to(device)
    incompatible = model.load_state_dict(payload["modelState"], strict=False)
    missing = set(incompatible.missing_keys)
    allowed = {name for name in model.state_dict() if name.startswith("pyramid_fusion.")}
    if missing != allowed or incompatible.unexpected_keys:
        raise RuntimeError(
            "pyramid-context checkpoint compatibility mismatch: "
            f"missing={sorted(missing)} unexpected={list(incompatible.unexpected_keys)}"
        )
    model.set_pyramid_probe_training()
    return model, config, payload


def _optimizer(model: PyramidContextV16Candidate, config: V16Config) -> torch.optim.Optimizer:
    parameters = model.pyramid_parameters()
    if not parameters:
        raise RuntimeError("pyramid-context probe has no trainable parameters")
    return torch.optim.Adam(
        parameters,
        lr=config.sr_learning_rate,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        foreach=False,
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Test whether wider phase-neutral LR context improves current Main V16"
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--checkpoint", default="")
    value.add_argument("--steps", type=int, default=596)
    value.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    value.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    value.add_argument("--preview-samples", type=int, default=4)
    return value


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = repo_root / checkpoint
        pointer = {}
    else:
        checkpoint, pointer = _latest_source(repo_root)

    device = _device(str(args.device))
    model, config, payload = _load_candidate(checkpoint, device)
    manifest_path = Path(str(payload.get("manifest") or ""))
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    start_step = int(payload.get("step") or 0)
    steps = max(1, int(args.steps))
    end_step = start_step + steps
    seed = int(payload.get("seed") or 16201)

    dataset = AuthorityBalancedSRDataset(
        manifest,
        config,
        "train",
        end_step,
        seed=seed,
        degradation="clean",
        augmentation_policy="d4-cyclic",
        spatial_policy="balanced-detail",
        detail_fraction=0.5,
        d4_samples_per_variant=2,
    )
    loader = DataLoader(
        Subset(dataset, range(start_step, end_step)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = _optimizer(model, config)
    model.train()
    started = time.monotonic()

    print("NSAMDR V16 PYRAMID-CONTEXT PROBE", flush=True)
    print(f"Checkpoint       : {checkpoint}", flush=True)
    print(f"Start/end step   : {start_step} -> {end_step}", flush=True)
    print("Trainable scope  : pyramid_fusion only", flush=True)
    print("Context scales   : LR /2, /4, /8; phase-neutral fusion", flush=True)

    for offset, batch in enumerate(loader, start=1):
        global_step = start_step + offset
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, str(args.amp_precision)):
            outputs = model(
                batch["lr_albedo"],
                batch["lr_normal"],
                batch["lr_material"],
            )
            terms = _proof_loss_terms(outputs, batch, config)
            loss = terms["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.pyramid_parameters(), 1.0)
        optimizer.step()
        if offset == 1 or global_step % 32 == 0 or global_step == end_step:
            elapsed = max(time.monotonic() - started, 1.0e-6)
            rate = offset / elapsed
            eta = (steps - offset) / max(rate, 1.0e-6)
            values = _loss_values(terms)
            print(
                f"[pyramid-context] step {global_step}/{end_step} "
                f"loss={values['total']:.6f} grad={values['gradient']:.6f} "
                f"elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m",
                flush=True,
            )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = repo_root / "artifacts/nsamdr/diagnostics/v16_pyramid_context" / f"probe_{stamp}"
    preview_root = run_dir / "previews"
    validation = _evaluate(
        model,
        manifest,
        config,
        split="validation",
        samples=0 if not manifest.get("counts") else int((manifest.get("counts") or {}).get("validationFamilies") or 38),
        device=device,
        seed=seed + 7001,
        precision=str(args.amp_precision),
        preview_root=preview_root,
        preview_samples=int(args.preview_samples),
        augmentation_policy="d4-cyclic",
        spatial_policy="balanced-detail",
        detail_fraction=0.5,
        d4_samples_per_variant=2,
    )
    train_validation = _evaluate(
        model,
        manifest,
        config,
        split="train",
        samples=int((manifest.get("counts") or {}).get("validationFamilies") or 38),
        device=device,
        seed=seed + 8001,
        precision=str(args.amp_precision),
        augmentation_policy="d4-cyclic",
        spatial_policy="balanced-detail",
        detail_fraction=0.5,
        d4_samples_per_variant=2,
    )

    before = dict(pointer.get("heldout") or {})
    report = {
        "schema": SCHEMA,
        "sourceCheckpoint": str(checkpoint),
        "startStep": start_step,
        "endStep": end_step,
        "trainableScope": "pyramid-fusion-only",
        "contextFactors": list(PyramidContextV16Candidate.PYRAMID_FACTORS),
        "beforeHeldout": before,
        "validation": validation,
        "trainValidation": train_validation,
        "elapsedSeconds": time.monotonic() - started,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("=" * 78, flush=True)
    print("PYRAMID-CONTEXT RESULT", flush=True)
    print(f"Held global   : {validation['median_global_recovery']*100:+.2f}%", flush=True)
    print(f"Held edge     : {validation['median_edge_recovery']*100:+.2f}%", flush=True)
    print(f"Held gradient : {validation['median_gradient_recovery']*100:+.2f}%", flush=True)
    print(f"Held lattice  : {validation['median_lattice_cell_excess']*100:+.2f}%", flush=True)
    print(f"Seen global   : {train_validation['median_global_recovery']*100:+.2f}%", flush=True)
    print(f"Seen edge     : {train_validation['median_edge_recovery']*100:+.2f}%", flush=True)
    print(f"Report        : {report_path}", flush=True)
    return 0, report_path


def main(argv: list[str] | None = None) -> int:
    code, _ = run(parser().parse_args(argv))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
