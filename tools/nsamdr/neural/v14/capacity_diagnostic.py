#!/usr/bin/env python3
"""V16.0 single-region HR residual capacity proof.

This diagnostic is non-promotable. It overfits one deterministic Raven region with the
exact production candidate path. It tests whether C can beat B without LR-grid imprint.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    NEURAL_ROOT = Path(__file__).resolve().parent.parent
    if str(NEURAL_ROOT) not in sys.path:
        sys.path.insert(0, str(NEURAL_ROOT))
    from v14.capacity_artifacts import CapacityArtifactWriter
    from v14.checkpoint import save_checkpoint
    from v14.config import V16Config
    from v14.dataset import load_manifest
    from v14.diagnostic_support import (
        DIAGNOSTIC_REVISION,
        DIAGNOSTIC_SCHEMA,
        archive_run,
        autocast_context,
        dataset_sample,
        detail_score,
        device_from_name,
        make_run_directory,
        prepare_raven_dataset,
        record_key,
        save_probe,
        write_report,
    )
    from v14.losses import candidate_loss
    from v14.model import MODEL_SCHEMA, NSAMDRV16
    from v14.qualification import sample_metrics
    from v14.training_stability import DivergenceMonitor
else:
    from .capacity_artifacts import CapacityArtifactWriter
    from .checkpoint import save_checkpoint
    from .config import V16Config
    from .dataset import load_manifest
    from .diagnostic_support import (
        DIAGNOSTIC_REVISION,
        DIAGNOSTIC_SCHEMA,
        archive_run,
        autocast_context,
        dataset_sample,
        detail_score,
        device_from_name,
        make_run_directory,
        prepare_raven_dataset,
        record_key,
        save_probe,
        write_report,
    )
    from .losses import candidate_loss
    from .model import MODEL_SCHEMA, NSAMDRV16
    from .qualification import sample_metrics
    from .training_stability import DivergenceMonitor


@dataclass(frozen=True)
class CapacityThresholds:
    edge_recovery: float
    global_recovery: float
    gradient_recovery: float


class CapacityDiagnostic:
    """Own one complete V16.0 Capacity run."""

    def __init__(
        self,
        args: argparse.Namespace,
        repo_root: Path,
        device: torch.device,
    ) -> None:
        self.args = args
        self.repo_root = repo_root
        self.device = device
        self.thresholds = CapacityThresholds(
            edge_recovery=float(args.required_edge_recovery),
            global_recovery=float(args.required_global_recovery),
            gradient_recovery=float(args.required_gradient_recovery),
        )
        self.config = V16Config(
            minimum_heldout_samples=1,
            candidate_edge_recovery_required=self.thresholds.edge_recovery,
            candidate_global_recovery_required=self.thresholds.global_recovery,
            candidate_gradient_recovery_required=self.thresholds.gradient_recovery,
        )
        self.config.validate()

    def _passes(self, metrics: dict[str, float]) -> bool:
        return bool(
            metrics.get("edge_recovery", -1.0) >= self.thresholds.edge_recovery
            and metrics.get("global_recovery", -1.0) >= self.thresholds.global_recovery
            and metrics.get("gradient_recovery", -1.0)
            >= self.thresholds.gradient_recovery
            and metrics.get("lattice_cell_excess", 1.0)
            <= self.config.candidate_lattice_cell_excess_max
        )

    @staticmethod
    def _score(metrics: dict[str, float]) -> float:
        return float(
            metrics.get("global_recovery", -1.0)
            + metrics.get("edge_recovery", -1.0)
            + metrics.get("gradient_recovery", -1.0)
        )

    def _vram(self) -> tuple[float, float]:
        if self.device.type != "cuda":
            return 0.0, 0.0
        gib = float(1024**3)
        return (
            float(torch.cuda.max_memory_allocated(self.device) / gib),
            float(torch.cuda.max_memory_reserved(self.device) / gib),
        )

    def _entry(
        self,
        *,
        step: int,
        loss_value: float,
        grad_norm: float,
        outputs: dict[str, torch.Tensor],
        metrics: dict[str, float],
        saturation: dict[str, float],
        passed: bool,
        diverged: bool,
        divergence_reason: str | None,
    ) -> dict[str, object]:
        allocated, reserved = self._vram()
        raw = CapacityArtifactWriter.raw_residual_magnitude(outputs)
        return {
            "step": step,
            "loss": loss_value,
            "gradientNormPreClip": grad_norm,
            "residualMagnitude": float(
                outputs["predicted_residual_albedo"].float().abs().mean().item()
            ),
            "rawResidualMagnitudeAlbedo": raw["albedo"],
            "rawResidualMagnitudeNormal": raw["normal"],
            "rawResidualMagnitudeMaterial": raw["material"],
            "globalRecovery": float(metrics["global_recovery"]),
            "edgeRecovery": float(metrics["edge_recovery"]),
            "gradientRecovery": float(metrics["gradient_recovery"]),
            "detailRecovery1px": float(metrics["detail_recovery_1px"]),
            "detailRecovery2px": float(metrics["detail_recovery_2px"]),
            "detailRecovery4px": float(metrics["detail_recovery_4px"]),
            "albedoRecovery": float(metrics["albedo_recovery"]),
            "normalRecovery": float(metrics["normal_recovery"]),
            "materialRecovery": float(metrics["material_recovery"]),
            "latticeCellExcess": float(metrics["lattice_cell_excess"]),
            "residualCapSaturationAlbedo": saturation["albedo"],
            "residualCapSaturationNormal": saturation["normal"],
            "residualCapSaturationMaterial": saturation["material"],
            "peakAllocatedVRAMGiB": allocated,
            "peakReservedVRAMGiB": reserved,
            "passed": passed,
            "diverged": diverged,
            "divergenceReason": divergence_reason,
        }

    def _write_divergence_report(
        self,
        run_dir: Path,
        *,
        step: int,
        reason: str,
        monitor: DivergenceMonitor,
        history: list[dict[str, object]],
        best_checkpoint: Path | None,
        last_stable_checkpoint: Path | None,
    ) -> Path:
        path = run_dir / "divergence_report.json"
        payload = {
            "schema": "NSAMDR_V16_CAPACITY_DIVERGENCE_V1",
            "revision": DIAGNOSTIC_REVISION,
            "modelSchema": MODEL_SCHEMA,
            "step": int(step),
            "reason": reason,
            "monitor": monitor.state(),
            "bestCheckpoint": (
                str(best_checkpoint.resolve()) if best_checkpoint else None
            ),
            "lastStableCheckpoint": (
                str(last_stable_checkpoint.resolve()) if last_stable_checkpoint else None
            ),
            "lastHistoryEntry": history[-1] if history else None,
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def run(self) -> int:
        prepare_raven_dataset(self.args, self.repo_root)
        manifest = load_manifest(self.repo_root, self.config)
        records = [
            record for record in manifest["crops"] if record.get("split") == "train"
        ]
        if not records:
            raise RuntimeError("V16.0 Capacity found no Raven training regions")

        record = max(records, key=detail_score)
        batch = dataset_sample(record, self.config, self.device)
        model = NSAMDRV16(self.config).to(self.device)
        model.set_candidate_training()
        parameters = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(
            parameters,
            lr=float(self.args.learning_rate),
            betas=(0.9, 0.999),
            eps=1.0e-8,
        )
        run_dir = make_run_directory(self.repo_root, "capacity")
        writer = CapacityArtifactWriter(run_dir)
        monitor = DivergenceMonitor()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        print("=" * 78, flush=True)
        print("V16.0 HR RESIDUAL CAPACITY - SWINIR-STYLE HR DIAGNOSTIC", flush=True)
        print(f"Model schema  : {MODEL_SCHEMA}", flush=True)
        print(f"Region        : {record_key(record)}", flush=True)
        print(
            f"Geometry      : {self.config.train_lr_size} -> {self.config.train_hr_size}",
            flush=True,
        )
        print(
            "Backbone      : "
            f"{self.config.swin_groups} RSTB x {self.config.swin_blocks_per_group} layers, "
            f"{self.config.hr_channels} channels, window {self.config.swin_window_size}, "
            f"{self.config.swin_num_heads} heads",
            flush=True,
        )
        print(f"Maximum steps : {self.args.steps}", flush=True)
        print(f"Learning rate : {self.args.learning_rate}", flush=True)
        print("Optimizer     : Adam", flush=True)
        print(
            "Pass rule     : global/edge/gradient recovery + <=15% LR-lattice excess",
            flush=True,
        )
        print("=" * 78, flush=True)

        history: list[dict[str, object]] = []
        last_metrics: dict[str, float] = {}
        last_evaluated: dict[str, torch.Tensor] | None = None
        passed = False
        diverged = False
        divergence_reason: str | None = None
        stop_step = int(self.args.steps)
        best_score = -math.inf
        best_checkpoint = run_dir / "best_checkpoint.pt"
        last_stable_checkpoint = run_dir / "last_stable_checkpoint.pt"
        best_written = False
        stable_written = False

        for step in range(1, int(self.args.steps) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device, self.args.amp_precision):
                outputs = model(
                    batch["lr_albedo"],
                    batch["lr_normal"],
                    batch["lr_material"],
                )
                losses = candidate_loss(outputs, batch, self.config)

            loss_value = float(losses["total"].detach().float().item())
            decision = monitor.check_scalar_integrity(loss=loss_value)
            if decision.diverged:
                diverged = True
                divergence_reason = decision.reason
                stop_step = step
                break

            losses["total"].backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().float().item()
            )
            decision = monitor.check_scalar_integrity(
                loss=loss_value,
                gradient_norm=grad_norm,
            )
            if decision.diverged:
                diverged = True
                divergence_reason = decision.reason
                stop_step = step
                break
            optimizer.step()

            should_report = (
                step == 1
                or step % max(1, int(self.args.report_every)) == 0
                or step == int(self.args.steps)
            )
            if not should_report:
                continue

            model.eval()
            with torch.no_grad(), autocast_context(self.device, self.args.amp_precision):
                evaluated = model(
                    batch["lr_albedo"],
                    batch["lr_normal"],
                    batch["lr_material"],
                )
            last_evaluated = evaluated
            last_metrics = sample_metrics(evaluated, batch, final=False)
            saturation = CapacityArtifactWriter.residual_cap_saturation(
                evaluated,
                self.config,
            )
            stability = monitor.observe_report(
                loss=loss_value,
                gradient_norm=grad_norm,
                saturation=saturation,
            )
            passed = self._passes(last_metrics) and not stability.diverged
            entry = self._entry(
                step=step,
                loss_value=loss_value,
                grad_norm=grad_norm,
                outputs=evaluated,
                metrics=last_metrics,
                saturation=saturation,
                passed=passed,
                diverged=stability.diverged,
                divergence_reason=stability.reason,
            )
            history.append(entry)

            raw_max = max(
                float(entry["rawResidualMagnitudeAlbedo"]),
                float(entry["rawResidualMagnitudeNormal"]),
                float(entry["rawResidualMagnitudeMaterial"]),
            )
            print(
                f"  step {step:4d}/{int(self.args.steps):4d} "
                f"loss={float(entry['loss']):.6f} "
                f"global={float(entry['globalRecovery'])*100:+.2f}% "
                f"edge={float(entry['edgeRecovery'])*100:+.2f}% "
                f"grad={float(entry['gradientRecovery'])*100:+.2f}% "
                f"lattice={float(entry['latticeCellExcess'])*100:+.1f}% "
                f"sat={max(saturation.values())*100:.1f}% "
                f"raw={raw_max:.4f} "
                f"VRAM={float(entry['peakAllocatedVRAMGiB']):.2f}GiB",
                flush=True,
            )

            if stability.diverged:
                diverged = True
                divergence_reason = stability.reason
                stop_step = step
                print(
                    f"[v16.0-capacity] DIVERGED at step {step}: {divergence_reason}",
                    flush=True,
                )
                break

            save_checkpoint(
                last_stable_checkpoint,
                model,
                self.config,
                epoch=0,
                phase="v16.0-mini-capacity-last-stable",
                metrics=last_metrics,
            )
            stable_written = True

            score = self._score(last_metrics)
            if score > best_score:
                best_score = score
                save_checkpoint(
                    best_checkpoint,
                    model,
                    self.config,
                    epoch=0,
                    phase="v16.0-mini-capacity-best",
                    metrics=last_metrics,
                )
                best_written = True

            if passed:
                stop_step = step
                print(
                    f"[v16.0-capacity] PASS at step {step}; stopping early.",
                    flush=True,
                )
                break

        if last_evaluated is None:
            model.eval()
            with torch.no_grad(), autocast_context(self.device, self.args.amp_precision):
                last_evaluated = model(
                    batch["lr_albedo"],
                    batch["lr_normal"],
                    batch["lr_material"],
                )
            last_metrics = sample_metrics(last_evaluated, batch, final=False)

        final_checkpoint: Path | None = None
        if not diverged:
            final_checkpoint = run_dir / "candidate_checkpoint.pt"
            save_checkpoint(
                final_checkpoint,
                model,
                self.config,
                epoch=0,
                phase="v16.0-mini-capacity",
                metrics=last_metrics,
            )

        probe_path = save_probe(run_dir, batch, last_evaluated, include_final=False)
        diagnostic_images = writer.write_all(batch, last_evaluated)
        curve_path = run_dir / "capacity_curve.json"
        curve_path.write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

        divergence_report: Path | None = None
        if diverged:
            divergence_report = self._write_divergence_report(
                run_dir,
                step=stop_step,
                reason=divergence_reason or "unknown",
                monitor=monitor,
                history=history,
                best_checkpoint=(best_checkpoint if best_written else None),
                last_stable_checkpoint=(last_stable_checkpoint if stable_written else None),
            )

        allocated, reserved = self._vram()
        status = "DIVERGED" if diverged else ("PASS" if passed else "FAIL")
        report = {
            "schema": DIAGNOSTIC_SCHEMA,
            "mode": "capacity",
            "revision": DIAGNOSTIC_REVISION,
            "status": status.lower(),
            "passed": bool(passed and not diverged),
            "diverged": diverged,
            "divergenceReason": divergence_reason,
            "promotable": False,
            "modelSchema": MODEL_SCHEMA,
            "architecture": model.architecture_contract(),
            "optimizer": "Adam",
            "record": record_key(record),
            "metrics": last_metrics,
            "stopStep": stop_step,
            "maximumSteps": int(self.args.steps),
            "learningRate": float(self.args.learning_rate),
            "peakAllocatedVRAMGiB": allocated,
            "peakReservedVRAMGiB": reserved,
            "stabilityMonitor": monitor.state(),
            "thresholds": {
                "edgeRecovery": self.thresholds.edge_recovery,
                "globalRecovery": self.thresholds.global_recovery,
                "gradientRecovery": self.thresholds.gradient_recovery,
                "maxLatticeCellExcess": self.config.candidate_lattice_cell_excess_max,
            },
            "candidateCheckpoint": (
                str(final_checkpoint.resolve()) if final_checkpoint else None
            ),
            "bestCheckpoint": (
                str(best_checkpoint.resolve()) if best_written else None
            ),
            "lastStableCheckpoint": (
                str(last_stable_checkpoint.resolve()) if stable_written else None
            ),
            "divergenceReport": (
                str(divergence_report.resolve()) if divergence_report else None
            ),
            "recoveryCurve": str(curve_path.resolve()),
            "probe": str(probe_path.resolve()),
            "diagnosticImages": diagnostic_images,
            "datasetFingerprint": manifest.get("fingerprint"),
        }
        write_report(run_dir, report)
        archive = archive_run(run_dir)

        print(f"[v16.0-capacity] report      : {run_dir / 'report.json'}", flush=True)
        print(f"[v16.0-capacity] curve       : {curve_path}", flush=True)
        print(f"[v16.0-capacity] diagnostics : {archive}", flush=True)
        print(f"[v16.0-capacity] result      : {status}", flush=True)
        if diverged:
            return 3
        return 0 if passed else 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NSAMDR V16.0 SwinIR-style Raven capacity diagnostic"
    )
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    p.add_argument("--rebuild-dataset", action="store_true")
    p.add_argument("--prepare-train-regions", type=int, default=16)
    p.add_argument("--prepare-validation-regions", type=int, default=4)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    p.add_argument("--steps", type=int, default=3072)
    p.add_argument("--report-every", type=int, default=32)
    p.add_argument(
        "--learning-rate",
        type=float,
        default=V16Config().sr_learning_rate,
    )
    p.add_argument("--required-edge-recovery", type=float, default=0.60)
    p.add_argument("--required-global-recovery", type=float, default=0.45)
    p.add_argument("--required-gradient-recovery", type=float, default=0.35)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    return CapacityDiagnostic(
        args,
        args.repo_root.resolve(),
        device_from_name(args.device),
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
