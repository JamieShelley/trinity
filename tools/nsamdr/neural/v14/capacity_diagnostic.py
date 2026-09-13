#!/usr/bin/env python3
"""V14.1 single-region HR residual capacity proof.

This diagnostic is deliberately non-promotable. It overfits one deterministic Raven
region using the exact production candidate path and answers one question: can C
materially beat deterministic baseline B without imprinting the 4x LR lattice?
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import subprocess
import sys

import torch

if __package__ in {None, ""}:
    NEURAL_ROOT = Path(__file__).resolve().parent.parent
    if str(NEURAL_ROOT) not in sys.path:
        sys.path.insert(0, str(NEURAL_ROOT))
    from v14.checkpoint import save_checkpoint
    from v14.config import V14Config
    from v14.dataset import load_manifest
    from v14.losses import candidate_loss
    from v14.mini_diagnostics import (
        DIAGNOSTIC_SCHEMA,
        _archive,
        _autocast,
        _dataset_sample,
        _detail_score,
        _device,
        _prepare_dataset,
        _record_key,
        _run_directory,
        _save_probe,
        _write_report,
    )
    from v14.model import MODEL_SCHEMA, NSAMDRV14
    from v14.qualification import sample_metrics
else:
    from .checkpoint import save_checkpoint
    from .config import V14Config
    from .dataset import load_manifest
    from .losses import candidate_loss
    from .mini_diagnostics import (
        DIAGNOSTIC_SCHEMA,
        _archive,
        _autocast,
        _dataset_sample,
        _detail_score,
        _device,
        _prepare_dataset,
        _record_key,
        _run_directory,
        _save_probe,
        _write_report,
    )
    from .model import MODEL_SCHEMA, NSAMDRV14
    from .qualification import sample_metrics


def _passed(metrics: dict[str, float], config: V14Config, args: argparse.Namespace) -> bool:
    return bool(
        metrics.get("edge_recovery", -1.0) >= float(args.required_edge_recovery)
        and metrics.get("global_recovery", -1.0) >= float(args.required_global_recovery)
        and metrics.get("gradient_recovery", -1.0) >= float(args.required_gradient_recovery)
        and metrics.get("lattice_cell_excess", 1.0) <= config.candidate_lattice_cell_excess_max
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSAMDR V14.1 phase-neutral Raven capacity diagnostic")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    p.add_argument("--rebuild-dataset", action="store_true")
    p.add_argument("--prepare-train-regions", type=int, default=16)
    p.add_argument("--prepare-validation-regions", type=int, default=4)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    p.add_argument("--steps", type=int, default=3072)
    p.add_argument("--report-every", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1.0e-3)
    p.add_argument("--required-edge-recovery", type=float, default=0.60)
    p.add_argument("--required-global-recovery", type=float, default=0.45)
    p.add_argument("--required-gradient-recovery", type=float, default=0.35)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    device = _device(args.device)

    _prepare_dataset(args, repo_root)
    config = V14Config(
        minimum_heldout_samples=1,
        candidate_edge_recovery_required=float(args.required_edge_recovery),
        candidate_global_recovery_required=float(args.required_global_recovery),
        candidate_gradient_recovery_required=float(args.required_gradient_recovery),
    )
    config.validate()
    manifest = load_manifest(repo_root, config)
    records = [record for record in manifest["crops"] if record.get("split") == "train"]
    if not records:
        raise RuntimeError("V14.1 capacity diagnostic found no Raven training regions")

    record = max(records, key=_detail_score)
    batch = _dataset_sample(record, config, device)
    model = NSAMDRV14(config).to(device)
    model.set_candidate_training()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(args.learning_rate), weight_decay=config.weight_decay)
    run_dir = _run_directory(repo_root, "capacity")

    print("=" * 76, flush=True)
    print("V14.1 HR RESIDUAL CAPACITY — PHASE-NEUTRAL DIAGNOSTIC ONLY", flush=True)
    print(f"Model schema  : {MODEL_SCHEMA}", flush=True)
    print(f"Region        : {_record_key(record)}", flush=True)
    print(f"Geometry      : {config.train_lr_size} -> {config.train_hr_size}", flush=True)
    print(f"Maximum steps : {args.steps}", flush=True)
    print(f"Learning rate : {args.learning_rate}", flush=True)
    print("Pass rule     : global/edge/gradient recovery + <=15% excess LR-lattice projection", flush=True)
    print("=" * 76, flush=True)

    history: list[dict[str, float | int | bool]] = []
    last_metrics: dict[str, float] = {}
    passed = False
    stop_step = int(args.steps)

    for step in range(1, int(args.steps) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, args.amp_precision):
            outputs = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
            losses = candidate_loss(outputs, batch, config)
        losses["total"].backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().item())
        optimizer.step()

        should_report = step == 1 or step % max(1, int(args.report_every)) == 0 or step == int(args.steps)
        if not should_report:
            continue

        model.eval()
        with torch.no_grad(), _autocast(device, args.amp_precision):
            evaluated = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
        last_metrics = sample_metrics(evaluated, batch, final=False)
        residual_magnitude = float(evaluated["candidate_residual_albedo"].float().abs().mean().item())
        passed = _passed(last_metrics, config, args)
        history.append({
            "step": step,
            "loss": float(losses["total"].detach().item()),
            "gradientNormPreClip": grad_norm,
            "residualMagnitude": residual_magnitude,
            "globalRecovery": float(last_metrics["global_recovery"]),
            "edgeRecovery": float(last_metrics["edge_recovery"]),
            "gradientRecovery": float(last_metrics["gradient_recovery"]),
            "latticeCellExcess": float(last_metrics["lattice_cell_excess"]),
            "passed": passed,
        })
        print(
            f"  step {step:4d}/{int(args.steps):4d} "
            f"loss={float(losses['total'].detach().item()):.6f} "
            f"global={last_metrics['global_recovery']*100:+.2f}% "
            f"edge={last_metrics['edge_recovery']*100:+.2f}% "
            f"grad={last_metrics['gradient_recovery']*100:+.2f}% "
            f"lattice={last_metrics['lattice_cell_excess']*100:+.1f}% "
            f"residual={residual_magnitude:.5f} gradNorm={grad_norm:.3f}",
            flush=True,
        )
        if passed:
            stop_step = step
            print(f"[v14.1-capacity] PASS at step {step}; stopping early.", flush=True)
            break

    checkpoint = run_dir / "candidate_checkpoint.pt"
    save_checkpoint(checkpoint, model, config, epoch=0, phase="v14.1-mini-capacity", metrics=last_metrics)
    model.eval()
    with torch.no_grad(), _autocast(device, args.amp_precision):
        outputs = model(batch["lr_albedo"], batch["lr_normal"], batch["lr_material"])
    probe = _save_probe(run_dir, batch, outputs, include_final=False)
    curve_path = run_dir / "capacity_curve.json"
    curve_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema": DIAGNOSTIC_SCHEMA,
        "mode": "capacity",
        "revision": "V14.1",
        "passed": passed,
        "promotable": False,
        "modelSchema": MODEL_SCHEMA,
        "architecture": model.architecture_contract(),
        "record": _record_key(record),
        "metrics": last_metrics,
        "stopStep": stop_step,
        "maximumSteps": int(args.steps),
        "learningRate": float(args.learning_rate),
        "thresholds": {
            "edgeRecovery": float(args.required_edge_recovery),
            "globalRecovery": float(args.required_global_recovery),
            "gradientRecovery": float(args.required_gradient_recovery),
            "maxLatticeCellExcess": config.candidate_lattice_cell_excess_max,
        },
        "candidateCheckpoint": str(checkpoint.resolve()),
        "recoveryCurve": str(curve_path.resolve()),
        "probe": str(probe.resolve()),
        "datasetFingerprint": manifest.get("fingerprint"),
    }
    _write_report(run_dir, report)
    archive = _archive(run_dir)
    print(f"[v14.1-capacity] report      : {run_dir / 'report.json'}", flush=True)
    print(f"[v14.1-capacity] curve       : {curve_path}", flush=True)
    print(f"[v14.1-capacity] diagnostics : {archive}", flush=True)
    print(f"[v14.1-capacity] result      : {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
