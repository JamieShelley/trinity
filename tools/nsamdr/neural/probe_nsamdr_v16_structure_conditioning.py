#!/usr/bin/env python3
"""Staged control-vs-structure V16 proof on the broad authored prior corpus.

The proof is deliberately non-promotable. It compares the same reduced V16 Swin
body with and without LR structure-conditioning on complete held-out authorities.
Training is authority-balanced so every authored authority is visited before any
one authority repeats. The default 512/1024/2048 ladder evaluates both models at
each checkpoint and stops before the next stage if conditioning still has no
held-out benefit after the second checkpoint.

Material recovery is telemetry only because the broad census does not resolve full
SOF material semantics for every authority.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v14.config import V16Config
from v14.model import NSAMDRV16
from v14.qualification import sample_metrics
from v16.broad_prior import AuthorityBalancedSRDataset
from v16.conditioning import StructureConditionedV16Candidate


SCHEMA = "NSAMDR_V16_STRUCTURE_CONDITIONING_PROBE_V2"
DEFAULT_MANIFEST = "artifacts/nsamdr/training_v16_authored_prior/dataset_manifest.json"
DEFAULT_STAGES = (512, 1024, 2048)
METRIC_KEYS = (
    "global_recovery",
    "edge_recovery",
    "gradient_recovery",
    "normal_recovery",
    "material_recovery",
    "lattice_cell_excess",
)
NO_BENEFIT = "structure-conditioning-no-heldout-benefit"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def _parse_stages(raw: str, single_steps: int) -> list[int]:
    if int(single_steps) > 0:
        return [int(single_steps)]
    values: list[int] = []
    for token in str(raw).replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 1:
            raise ValueError("all training stages must be positive")
        values.append(value)
    stages = sorted(set(values))
    if not stages:
        raise ValueError("at least one training stage is required")
    return stages


def _gradient(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gray = value.float().mean(dim=1, keepdim=True)
    dx = F.pad(gray[..., :, 1:] - gray[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(gray[..., 1:, :] - gray[..., :-1, :], (0, 0, 0, 1))
    return dx, dy


def _proof_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: V16Config,
) -> torch.Tensor:
    """Albedo+normal proof loss; material is intentionally unsupervised here."""

    ca = outputs["candidate_albedo"].float()
    cn = outputs["candidate_normal"].float()
    ba = outputs["baseline_albedo"].detach().float()
    bn = outputs["baseline_normal"].detach().float()
    ta = batch["target_albedo"].float()
    tn = batch["target_normal"].float()

    reconstruction = F.l1_loss(ca, ta)
    cgx, cgy = _gradient(ca)
    tgx, tgy = _gradient(ta)
    gradient = (cgx - tgx).abs().mean() + (cgy - tgy).abs().mean()
    normal = F.l1_loss(cn, tn)
    target_a = (ta - ba).clamp(
        -config.albedo_residual_cap,
        config.albedo_residual_cap,
    )
    target_n = (tn - bn).clamp(
        -config.normal_residual_cap,
        config.normal_residual_cap,
    )
    residual = F.l1_loss(outputs["predicted_residual_albedo"].float(), target_a)
    residual = residual + 0.25 * F.l1_loss(
        outputs["predicted_residual_normal"].float(),
        target_n,
    )
    return reconstruction + 0.50 * gradient + 0.35 * normal + residual


def _to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in batch.items()
        if isinstance(value, torch.Tensor)
    }


def _candidate_optimizer(
    model: torch.nn.Module,
    config: V16Config,
) -> torch.optim.Optimizer:
    if hasattr(model, "set_candidate_training"):
        model.set_candidate_training()  # type: ignore[attr-defined]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("candidate proof has no trainable parameters")
    return torch.optim.Adam(parameters, lr=config.sr_learning_rate, foreach=False)


def _train_segment(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    manifest: dict[str, Any],
    config: V16Config,
    *,
    start_step: int,
    end_step: int,
    device: torch.device,
    seed: int,
    label: str,
) -> dict[str, float | int]:
    if end_step <= start_step:
        raise ValueError("end_step must be greater than start_step")

    dataset = AuthorityBalancedSRDataset(
        manifest,
        config,
        "train",
        end_step,
        seed=seed,
        degradation="clean",
    )
    loader = DataLoader(
        Subset(dataset, range(start_step, end_step)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    losses: list[float] = []
    started = time.monotonic()
    segment_steps = end_step - start_step

    for offset, batch in enumerate(loader, start=1):
        global_step = start_step + offset
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["lr_albedo"],
            batch["lr_normal"],
            batch["lr_material"],
        )
        loss = _proof_loss(outputs, batch, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

        if offset == 1 or global_step % 64 == 0 or global_step == end_step:
            elapsed = max(time.monotonic() - started, 1.0e-6)
            rate = offset / elapsed
            eta = (segment_steps - offset) / max(rate, 1.0e-6)
            print(
                f"[{label}] step {global_step:4d}/{end_step} "
                f"loss={losses[-1]:.6f} elapsed={elapsed/60.0:.1f}m "
                f"segment-eta={eta/60.0:.1f}m",
                flush=True,
            )

    return {
        "fromStep": int(start_step),
        "toStep": int(end_step),
        "segmentSteps": int(segment_steps),
        "firstLoss": losses[0],
        "finalLoss": losses[-1],
        "medianTailLoss": float(statistics.median(losses[-min(32, len(losses)) :])),
        "elapsedSeconds": time.monotonic() - started,
        "trainAuthorityCount": int(dataset.authority_count),
    }


def _evaluate(
    model: torch.nn.Module,
    manifest: dict[str, Any],
    config: V16Config,
    *,
    samples: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    dataset = AuthorityBalancedSRDataset(
        manifest,
        config,
        "validation",
        samples,
        seed=seed,
        degradation="clean",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    metrics: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            outputs = model(
                batch["lr_albedo"],
                batch["lr_normal"],
                batch["lr_material"],
            )
            metrics.append(sample_metrics(outputs, batch, final=False))

    summary: dict[str, Any] = {
        "sampleCount": len(metrics),
        "heldOutAuthorityCount": dataset.selected_authority_count(),
        "availableHeldOutAuthorityCount": dataset.authority_count,
        "materialMetricQualified": False,
    }
    for key in METRIC_KEYS:
        summary[f"median_{key}"] = float(
            statistics.median(float(item[key]) for item in metrics)
        )
    return summary


def _config(manifest: str, hr_size: int) -> V16Config:
    if hr_size % 32:
        raise ValueError("hr-size must be divisible by 32 for the reduced Swin proof")
    config = V16Config(
        dataset_manifest=manifest,
        train_hr_size=hr_size,
        train_lr_size=hr_size // 4,
        validation_hr_size=hr_size,
        validation_lr_size=hr_size // 4,
        lr_context_channels=16,
        lr_blocks=2,
        hr_channels=48,
        swin_groups=2,
        swin_blocks_per_group=2,
        swin_num_heads=4,
        swin_window_size=8,
        map_tail_blocks=1,
        selector_channels=8,
        use_gradient_checkpointing=False,
        tiles_per_epoch=1,
        validation_tiles=8,
        minimum_heldout_samples=4,
        clean_epochs=1,
        robust_epochs=1,
        selector_epochs=1,
    )
    config.validate()
    return config


def _decision(
    control: dict[str, Any],
    conditioned: dict[str, Any],
) -> tuple[str, dict[str, float]]:
    deltas = {
        "global": float(conditioned["median_global_recovery"])
        - float(control["median_global_recovery"]),
        "edge": float(conditioned["median_edge_recovery"])
        - float(control["median_edge_recovery"]),
        "gradient": float(conditioned["median_gradient_recovery"])
        - float(control["median_gradient_recovery"]),
        "normal": float(conditioned["median_normal_recovery"])
        - float(control["median_normal_recovery"]),
        "lattice": float(conditioned["median_lattice_cell_excess"])
        - float(control["median_lattice_cell_excess"]),
    }
    if (
        deltas["global"] >= 0.05
        and deltas["edge"] >= 0.05
        and deltas["lattice"] <= 0.05
    ):
        return "structure-conditioning-signal-positive", deltas
    if deltas["global"] > 0.0 and deltas["edge"] > 0.0:
        return "structure-conditioning-signal-marginal", deltas
    return NO_BENEFIT, deltas


def _continue_after_checkpoint(
    checkpoint_index: int,
    checkpoint_count: int,
    decision: str,
) -> tuple[bool, str]:
    if checkpoint_index + 1 >= checkpoint_count:
        return False, "requested-ladder-complete"
    # Always allow the second checkpoint. A 512-step result is still a very early
    # broad-prior measurement for hundreds of independent authorities.
    if checkpoint_index == 0:
        return True, "minimum-two-checkpoints"
    if decision == NO_BENEFIT:
        return False, "no-heldout-benefit-after-second-checkpoint"
    return True, "heldout-conditioning-signal-remains-positive-or-marginal"


def _checkpoint_summary(
    stage: int,
    control: dict[str, Any],
    conditioned: dict[str, Any],
    deltas: dict[str, float],
    decision: str,
) -> list[str]:
    return [
        f"Checkpoint {stage}",
        f"  control global/edge    {control['median_global_recovery']*100:+.2f}% / "
        f"{control['median_edge_recovery']*100:+.2f}%",
        f"  conditioned global/edge {conditioned['median_global_recovery']*100:+.2f}% / "
        f"{conditioned['median_edge_recovery']*100:+.2f}%",
        f"  delta global/edge      {deltas['global']*100:+.2f}pp / "
        f"{deltas['edge']*100:+.2f}pp",
        f"  delta gradient/normal  {deltas['gradient']*100:+.2f}pp / "
        f"{deltas['normal']*100:+.2f}pp",
        f"  delta lattice excess   {deltas['lattice']*100:+.2f}pp",
        f"  decision               {decision}",
    ]


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root / manifest_path).resolve()
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Broad prior manifest is missing: {manifest_path}\n"
            "Run: scripts\\build\\nsamdr.bat authored-prior-corpus"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = dict(manifest.get("counts") or {})
    validation_families = int(counts.get("validationFamilies") or 0)
    if validation_families < 4:
        raise RuntimeError(
            "Structure-conditioning proof requires at least four held-out authorities"
        )

    stages = _parse_stages(args.stages, int(args.steps))
    validation_samples = (
        int(args.validation_samples)
        if int(args.validation_samples) > 0
        else validation_families
    )
    device = _device(args.device)
    config = _config(str(manifest_path.relative_to(repo_root)), int(args.hr_size))
    seed = int(args.seed)

    torch.manual_seed(seed)
    control = NSAMDRV16(config).to(device)
    torch.manual_seed(seed)
    conditioned = StructureConditionedV16Candidate(
        config,
        structure_channels=24,
        structure_blocks=2,
    ).to(device)
    control_optimizer = _candidate_optimizer(control, config)
    conditioned_optimizer = _candidate_optimizer(conditioned, config)

    print("NSAMDR V16 STAGED STRUCTURE-CONDITIONING PROOF", flush=True)
    print(f"Manifest          : {manifest_path}", flush=True)
    print(f"Device            : {device}", flush=True)
    print(f"HR/LR             : {config.train_hr_size}/{config.train_lr_size}", flush=True)
    print(f"Stages            : {stages}", flush=True)
    print(f"Train authorities : {counts.get('trainFamilies', '?')}", flush=True)
    print(f"Held-out auth.    : {validation_families}", flush=True)
    print(f"Validation samples: {validation_samples} (authority-balanced)", flush=True)

    curve: list[dict[str, Any]] = []
    start_step = 0
    stop_reason = "requested-ladder-complete"
    for checkpoint_index, end_step in enumerate(stages):
        control_train = _train_segment(
            control,
            control_optimizer,
            manifest,
            config,
            start_step=start_step,
            end_step=end_step,
            device=device,
            seed=seed,
            label="control",
        )
        control_eval = _evaluate(
            control,
            manifest,
            config,
            samples=validation_samples,
            device=device,
            seed=seed + 7001,
        )

        conditioned_train = _train_segment(
            conditioned,
            conditioned_optimizer,
            manifest,
            config,
            start_step=start_step,
            end_step=end_step,
            device=device,
            seed=seed,
            label="conditioned",
        )
        conditioned_eval = _evaluate(
            conditioned,
            manifest,
            config,
            samples=validation_samples,
            device=device,
            seed=seed + 7001,
        )
        decision, deltas = _decision(control_eval, conditioned_eval)
        checkpoint = {
            "step": int(end_step),
            "control": {"training": control_train, "validation": control_eval},
            "conditioned": {
                "training": conditioned_train,
                "validation": conditioned_eval,
            },
            "deltaConditionedMinusControl": deltas,
            "decision": decision,
        }
        curve.append(checkpoint)
        print("\n".join(_checkpoint_summary(
            end_step,
            control_eval,
            conditioned_eval,
            deltas,
            decision,
        )), flush=True)

        should_continue, stop_reason = _continue_after_checkpoint(
            checkpoint_index,
            len(stages),
            decision,
        )
        if not should_continue:
            break
        start_step = end_step

    final = curve[-1]
    final_control = dict(final["control"]["validation"])
    final_conditioned = dict(final["conditioned"]["validation"])
    final_deltas = dict(final["deltaConditionedMinusControl"])
    final_decision = str(final["decision"])

    run_dir = (
        repo_root
        / "artifacts/nsamdr/diagnostics/v16_structure_conditioning"
        / f"probe_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": SCHEMA,
        "promotable": False,
        "manifest": str(manifest_path),
        "authoritySplit": manifest.get("authoritySplit"),
        "samplingPolicy": "authority-balanced-complete-cycle-before-repeat",
        "device": str(device),
        "requestedStages": stages,
        "completedStages": [int(item["step"]) for item in curve],
        "stopReason": stop_reason,
        "hrSize": int(args.hr_size),
        "validationSamples": validation_samples,
        "materialQualification": "excluded-until-SOF-semantic-authority-is-resolved",
        "curve": curve,
        # Final aliases keep report consumers simple and retain V1-style access.
        "control": final["control"],
        "conditioned": final["conditioned"],
        "deltaConditionedMinusControl": final_deltas,
        "decision": final_decision,
    }
    report_path = run_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = [
        "NSAMDR V16 STAGED STRUCTURE-CONDITIONING PROOF",
        "=" * 86,
        f"Sampling              : authority-balanced",
        f"Requested checkpoints : {stages}",
        f"Completed checkpoints : {report['completedStages']}",
        f"Stop reason           : {stop_reason}",
        f"Held-out authorities  : {final_control['heldOutAuthorityCount']} / "
        f"{final_control['availableHeldOutAuthorityCount']}",
        "",
    ]
    for item in curve:
        summary.extend(
            _checkpoint_summary(
                int(item["step"]),
                dict(item["control"]["validation"]),
                dict(item["conditioned"]["validation"]),
                dict(item["deltaConditionedMinusControl"]),
                str(item["decision"]),
            )
        )
        summary.append("")
    summary.extend(
        (
            f"FINAL DECISION        : {final_decision}",
            "Material is telemetry only; full SOF material semantics are not claimed.",
            f"JSON                  : {report_path}",
        )
    )
    summary_path = run_dir / "summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary), flush=True)
    return 0, report_path


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Compare reduced V16 control vs structure conditioning with "
            "authority-balanced staged broad-prior training"
        )
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--manifest", default=DEFAULT_MANIFEST)
    value.add_argument(
        "--stages",
        default=",".join(str(value) for value in DEFAULT_STAGES),
        help="cumulative evaluation checkpoints, default 512,1024,2048",
    )
    value.add_argument(
        "--steps",
        type=int,
        default=0,
        help="compatibility override: run one checkpoint at exactly N updates",
    )
    value.add_argument("--hr-size", type=int, default=128)
    value.add_argument(
        "--validation-samples",
        type=int,
        default=0,
        help="0 evaluates one authority-balanced sample per held-out authority",
    )
    value.add_argument("--seed", type=int, default=16201)
    value.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    code, _ = run(parser().parse_args(argv))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
