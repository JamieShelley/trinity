#!/usr/bin/env python3
"""V14.2 single-region HR residual capacity proof.

This diagnostic is non-promotable. It overfits one deterministic Raven region with the
exact production candidate path. It tests whether C can beat B without LR-grid imprint.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    NEURAL_ROOT = Path(__file__).resolve().parent.parent
    if str(NEURAL_ROOT) not in sys.path:
        sys.path.insert(0, str(NEURAL_ROOT))
    from v14.capacity_artifacts import CapacityArtifactWriter
    from v14.checkpoint import save_checkpoint
    from v14.config import V14Config
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
    from v14.model import MODEL_SCHEMA, NSAMDRV14
    from v14.qualification import sample_metrics
else:
    from .capacity_artifacts import CapacityArtifactWriter
    from .checkpoint import save_checkpoint
    from .config import V14Config
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
    from .model import MODEL_SCHEMA, NSAMDRV14
    from .qualification import sample_metrics


@dataclass(frozen=True)
class CapacityThresholds:
    edge_recovery: float
    global_recovery: float
    gradient_recovery: float


class CapacityDiagnostic:
    """Own one complete V14.2 capacity run."""

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
        self.config = V14Config(
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
            and metrics.get("gradient_recovery", -1.0) >= self.thresholds.gradient_recovery
            and metrics.get("lattice_cell_excess", 1.0)
            <= self.config.candidate_lattice_cell_excess_max
        )

    def _vram_telemetry(self) -> tuple[float, float]:
        if self.device.type != "cuda":
            return 0.0, 0.0
        gib = float(1024 ** 3)
        allocated = torch.cuda.max_memory_allocated(self.device) / gib
        reserved = torch.cuda.max_memory_reserved(self.device) / gib
        return float(allocated), float(reserved)

    def _history_entry(
        self,
        *,
        step: int,
        loss: torch.Tensor,
        grad_norm: float,
        outputs: dict[str, torch.Tensor],
        metrics: dict[str, float],
        passed: bool,
    ) -> dict[str, float | int | bool]:
        saturation = CapacityArtifactWriter.residual_cap_saturation(outputs, self.config)
        allocated, reserved = self._vram_telemetry()
        return {
            "step": step,
            "loss": float(loss.detach().item()),
            "gradientNormPreClip": grad_norm,
            "residualMagnitude": float(
                outputs["candidate_residual_albedo"].float().abs().mean().item()
            ),
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
        }

    def run(self) -> int:
        prepare_raven_dataset(self.args, self.repo_root)
        manifest = load_manifest(self.repo_root, self.config)
        records = [
            record
            for record in manifest["crops"]
            if record.get("split") == "train"
        ]
        if not records:
            raise RuntimeError("V14.2 capacity diagnostic found no Raven training regions")

        record = max(records, key=detail_score)
        batch = dataset_sample(record, self.config, self.device)
        model = NSAMDRV14(self.config).to(self.device)
        model.set_candidate_training()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(self.args.learning_rate),
            weight_decay=self.config.weight_decay,
        )
        run_dir = make_run_directory(self.repo_root, "capacity")
        writer = CapacityArtifactWriter(run_dir)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        print("=" * 78, flush=True)
        print(
            "V14.2 HR RESIDUAL CAPACITY - MULTI-SCALE RCAN DIAGNOSTIC ONLY",
            flush=True,
        )
        print(f"Model schema  : {MODEL_SCHEMA}", flush=True)
        print(f"Region        : {record_key(record)}", flush=True)
        print(
            f"Geometry      : {self.config.train_lr_size} -> {self.config.train_hr_size}",
            flush=True,
        )
        print(f"Maximum steps : {self.args.steps}", flush=True)
        print(f"Learning rate : {self.args.learning_rate}", flush=True)
        print(
            "Pass rule     : global/edge/gradient recovery + <=15% excess LR-lattice projection",
            flush=True,
        )
        print("=" * 78, flush=True)

        history: list[dict[str, float | int | bool]] = []
        last_metrics: dict[str, float] = {}
        passed = False
        stop_step = int(self.args.steps)

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
            losses["total"].backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().item()
            )
            optimizer.step()

            should_report = (
                step == 1
                or step % max(1, int(self.args.report_every)) == 0
                or step == int(self.args.steps)
            )
            if not should_report:
                continue

            model.eval()
            with torch.no_grad(), autocast_context(
                self.device,
                self.args.amp_precision,
            ):
                evaluated = model(
                    batch["lr_albedo"],
                    batch["lr_normal"],
                    batch["lr_material"],
                )
            last_metrics = sample_metrics(evaluated, batch, final=False)
            passed = self._passes(last_metrics)
            entry = self._history_entry(
                step=step,
                loss=losses["total"],
                grad_norm=grad_norm,
                outputs=evaluated,
                metrics=last_metrics,
                passed=passed,
            )
            history.append(entry)

            print(
                f"  step {step:4d}/{int(self.args.steps):4d} "
                f"loss={entry['loss']:.6f} "
                f"global={entry['globalRecovery']*100:+.2f}% "
                f"edge={entry['edgeRecovery']*100:+.2f}% "
                f"grad={entry['gradientRecovery']*100:+.2f}% "
                f"d1={entry['detailRecovery1px']*100:+.1f}% "
                f"d2={entry['detailRecovery2px']*100:+.1f}% "
                f"d4={entry['detailRecovery4px']*100:+.1f}% "
                f"lattice={entry['latticeCellExcess']*100:+.1f}% "
                f"VRAM={entry['peakAllocatedVRAMGiB']:.2f}GiB",
                flush=True,
            )
            if passed:
                stop_step = step
                print(
                    f"[v14.2-capacity] PASS at step {step}; stopping early.",
                    flush=True,
                )
                break

        checkpoint_path = run_dir / "candidate_checkpoint.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            self.config,
            epoch=0,
            phase="v14.2-mini-capacity",
            metrics=last_metrics,
        )

        model.eval()
        with torch.no_grad(), autocast_context(self.device, self.args.amp_precision):
            outputs = model(
                batch["lr_albedo"],
                batch["lr_normal"],
                batch["lr_material"],
            )

        probe_path = save_probe(
            run_dir,
            batch,
            outputs,
            include_final=False,
        )
        diagnostic_images = writer.write_all(batch, outputs)
        curve_path = run_dir / "capacity_curve.json"
        curve_path.write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )

        allocated, reserved = self._vram_telemetry()
        report = {
            "schema": DIAGNOSTIC_SCHEMA,
            "mode": "capacity",
            "revision": DIAGNOSTIC_REVISION,
            "passed": passed,
            "promotable": False,
            "modelSchema": MODEL_SCHEMA,
            "architecture": model.architecture_contract(),
            "record": record_key(record),
            "metrics": last_metrics,
            "stopStep": stop_step,
            "maximumSteps": int(self.args.steps),
            "learningRate": float(self.args.learning_rate),
            "peakAllocatedVRAMGiB": allocated,
            "peakReservedVRAMGiB": reserved,
            "thresholds": {
                "edgeRecovery": self.thresholds.edge_recovery,
                "globalRecovery": self.thresholds.global_recovery,
                "gradientRecovery": self.thresholds.gradient_recovery,
                "maxLatticeCellExcess": self.config.candidate_lattice_cell_excess_max,
            },
            "candidateCheckpoint": str(checkpoint_path.resolve()),
            "recoveryCurve": str(curve_path.resolve()),
            "probe": str(probe_path.resolve()),
            "diagnosticImages": diagnostic_images,
            "datasetFingerprint": manifest.get("fingerprint"),
        }
        write_report(run_dir, report)
        archive = archive_run(run_dir)

        print(f"[v14.2-capacity] report      : {run_dir / 'report.json'}", flush=True)
        print(f"[v14.2-capacity] curve       : {curve_path}", flush=True)
        print(f"[v14.2-capacity] diagnostics : {archive}", flush=True)
        print(
            f"[v14.2-capacity] result      : {'PASS' if passed else 'FAIL'}",
            flush=True,
        )
        return 0 if passed else 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NSAMDR V14.2 multi-scale RCAN Raven capacity diagnostic"
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
    p.add_argument("--learning-rate", type=float, default=1.0e-3)
    p.add_argument("--required-edge-recovery", type=float, default=0.60)
    p.add_argument("--required-global-recovery", type=float, default=0.45)
    p.add_argument("--required-gradient-recovery", type=float, default=0.35)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    return CapacityDiagnostic(
        args,
        repo_root,
        device_from_name(args.device),
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
