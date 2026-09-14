#!/usr/bin/env python3
"""V16.0 multi-region candidate generalisation diagnostic."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    NEURAL_ROOT = Path(__file__).resolve().parent.parent
    if str(NEURAL_ROOT) not in sys.path:
        sys.path.insert(0, str(NEURAL_ROOT))
    from v14.checkpoint import load_checkpoint, save_checkpoint
    from v14.config import V16Config
    from v14.dataset import RavenSRDataset, load_manifest
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
        pseudo_manifest,
        save_probe,
        to_device_batch,
        validation_metrics,
        write_report,
    )
    from v14.losses import candidate_loss
    from v14.model import MODEL_SCHEMA, NSAMDRV16
    from v14.qualification import aggregate_candidate
else:
    from .checkpoint import load_checkpoint, save_checkpoint
    from .config import V16Config
    from .dataset import RavenSRDataset, load_manifest
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
        pseudo_manifest,
        save_probe,
        to_device_batch,
        validation_metrics,
        write_report,
    )
    from .losses import candidate_loss
    from .model import MODEL_SCHEMA, NSAMDRV16
    from .qualification import aggregate_candidate


MIN_TRAIN_REGIONS = 4
MIN_VALIDATION_REGIONS = 4
DEFAULT_MAX_STEPS = 2560
DEFAULT_VALIDATE_EVERY = 256


def balanced_region_index(step: int, region_count: int) -> int:
    """Map a 1-based optimizer step to a deterministic round-robin region index."""

    if step < 1:
        raise ValueError("step must be >= 1")
    if region_count < 1:
        raise ValueError("region_count must be >= 1")
    return (int(step) - 1) % int(region_count)


class MultiRegionDiagnostic:
    """Test whether V16.0 candidate gains survive held-out Raven spatial domains."""

    def __init__(
        self,
        args: argparse.Namespace,
        repo_root: Path,
        device: torch.device,
    ) -> None:
        self.args = args
        self.repo_root = repo_root
        self.device = device

    def _config(self) -> V16Config:
        config = V16Config(
            minimum_heldout_samples=MIN_VALIDATION_REGIONS,
            candidate_edge_recovery_required=float(self.args.required_edge_recovery),
            candidate_global_recovery_required=float(self.args.required_global_recovery),
            candidate_gradient_recovery_required=float(self.args.required_gradient_recovery),
        )
        # Multi-Region is step-based. These values remain valid configuration fields,
        # but they do not control the diagnostic training budget.
        config.tiles_per_epoch = max(1, int(getattr(self.args, "tiles_per_epoch", 64)))
        config.clean_epochs = 1
        config.robust_epochs = 0
        config.validate()
        return config

    def _records(
        self,
        manifest: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        train = sorted(
            [record for record in manifest["crops"] if record.get("split") == "train"],
            key=detail_score,
            reverse=True,
        )[: int(self.args.train_regions)]
        validation = sorted(
            [record for record in manifest["crops"] if record.get("split") == "validation"],
            key=detail_score,
            reverse=True,
        )[: int(self.args.validation_regions)]
        return train, validation

    def _balanced_train_datasets(
        self,
        records: list[dict[str, Any]],
        config: V16Config,
        max_steps: int,
    ) -> list[RavenSRDataset]:
        visits = max(1, int(math.ceil(max_steps / max(1, len(records)))))
        datasets: list[RavenSRDataset] = []
        for region_index, record in enumerate(records):
            datasets.append(
                RavenSRDataset(
                    pseudo_manifest([record], split="train"),
                    config,
                    "train",
                    visits,
                    seed=config.seed + 10_007 * (region_index + 1),
                    degradation="clean",
                )
            )
        return datasets

    def _evaluate_records(
        self,
        model: NSAMDRV16,
        records: list[dict[str, Any]],
        config: V16Config,
    ) -> dict[str, object]:
        metrics = validation_metrics(
            model,
            pseudo_manifest(records, split="validation"),
            config,
            self.device,
            self.args.amp_precision,
            final=False,
        )
        return aggregate_candidate(metrics, config)

    @staticmethod
    def _score(report: dict[str, object]) -> float:
        return (
            float(report["medianGlobalRecovery"])
            + float(report["medianEdgeRecovery"])
            + float(report["medianGradientRecovery"])
        )

    @staticmethod
    def _diagnosis(
        train_report: dict[str, object],
        validation_report: dict[str, object],
    ) -> str:
        train_pass = bool(train_report.get("passed"))
        validation_pass = bool(validation_report.get("passed"))
        if validation_pass and train_pass:
            return "passed"
        if validation_pass and not train_pass:
            return "validation-pass-train-anomaly"
        if train_pass:
            return "generalisation-failure"
        return "training-capacity-or-optimization-failure"

    def _write_curve(self, run_dir: Path, curve: list[dict[str, object]]) -> Path:
        path = run_dir / "generalisation_curve.json"
        path.write_text(
            json.dumps(curve, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def run(self) -> tuple[int, Path]:
        self.args.prepare_train_regions = max(
            int(self.args.prepare_train_regions),
            int(self.args.train_regions),
            MIN_TRAIN_REGIONS,
        )
        self.args.prepare_validation_regions = max(
            int(self.args.prepare_validation_regions),
            int(self.args.validation_regions),
            MIN_VALIDATION_REGIONS,
        )
        prepare_raven_dataset(self.args, self.repo_root)
        config = self._config()
        manifest = load_manifest(self.repo_root, config)
        train_records, validation_records = self._records(manifest)
        run_dir = make_run_directory(self.repo_root, "multiregion")

        if (
            len(train_records) < MIN_TRAIN_REGIONS
            or len(validation_records) < MIN_VALIDATION_REGIONS
        ):
            write_report(
                run_dir,
                {
                    "schema": DIAGNOSTIC_SCHEMA,
                    "revision": DIAGNOSTIC_REVISION,
                    "mode": "multiregion",
                    "modelSchema": MODEL_SCHEMA,
                    "passed": False,
                    "promotable": False,
                    "reason": "insufficient-spatial-domain-regions",
                    "trainRegions": len(train_records),
                    "validationRegions": len(validation_records),
                    "requiredTrainRegions": MIN_TRAIN_REGIONS,
                    "requiredValidationRegions": MIN_VALIDATION_REGIONS,
                    "datasetFingerprint": manifest.get("fingerprint"),
                    "datasetSplitPolicy": manifest.get("splitPolicy"),
                },
            )
            return 2, run_dir

        max_steps = max(1, int(self.args.max_steps))
        validate_every = max(1, int(self.args.validate_every))
        model = NSAMDRV16(config).to(self.device)
        model.set_candidate_training()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(self.args.learning_rate),
            betas=(0.9, 0.999),
            eps=1.0e-8,
        )
        train_datasets = self._balanced_train_datasets(
            train_records,
            config,
            max_steps,
        )
        region_visits = [0 for _ in train_records]
        curve: list[dict[str, object]] = []
        best_key = (-1, -1.0e9)
        best_path = run_dir / "best_checkpoint.pt"
        best_train_report: dict[str, object] | None = None
        best_validation_report: dict[str, object] | None = None
        best_step = 0
        interval_loss = 0.0
        interval_count = 0
        completed_steps = 0
        early_pass = False

        print("=" * 76, flush=True)
        print("V16.0 MULTI-REGION SR MINI - DIAGNOSTIC ONLY", flush=True)
        print(
            f"Train/held-out : {len(train_records)} / {len(validation_records)} windows",
            flush=True,
        )
        print(
            "Split rule     : train and validation pixel domains are disjoint; "
            "windows may overlap only inside one split",
            flush=True,
        )
        print("Sampling       : deterministic balanced round-robin", flush=True)
        print(f"Maximum steps  : {max_steps}", flush=True)
        print(f"Validate every : {validate_every} steps", flush=True)
        print(f"Learning rate  : {float(self.args.learning_rate):.7f}", flush=True)
        print("=" * 76, flush=True)

        for step in range(1, max_steps + 1):
            model.train()
            region_index = balanced_region_index(step, len(train_records))
            visit_index = region_visits[region_index]
            region_visits[region_index] += 1
            batch = to_device_batch(
                train_datasets[region_index][visit_index],
                self.device,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device, self.args.amp_precision):
                outputs = model(
                    batch["lr_albedo"],
                    batch["lr_normal"],
                    batch["lr_material"],
                )
                losses = candidate_loss(outputs, batch, config)
            loss = losses["total"]
            if not bool(torch.isfinite(loss).item()):
                self._write_curve(run_dir, curve)
                write_report(
                    run_dir,
                    {
                        "schema": DIAGNOSTIC_SCHEMA,
                        "revision": DIAGNOSTIC_REVISION,
                        "mode": "multiregion",
                        "modelSchema": MODEL_SCHEMA,
                        "passed": False,
                        "promotable": False,
                        "reason": "diverged-non-finite-loss",
                        "step": step,
                        "trainRegionVisits": region_visits,
                        "datasetFingerprint": manifest.get("fingerprint"),
                    },
                )
                return 3, run_dir

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not bool(torch.isfinite(grad_norm).item()):
                self._write_curve(run_dir, curve)
                write_report(
                    run_dir,
                    {
                        "schema": DIAGNOSTIC_SCHEMA,
                        "revision": DIAGNOSTIC_REVISION,
                        "mode": "multiregion",
                        "modelSchema": MODEL_SCHEMA,
                        "passed": False,
                        "promotable": False,
                        "reason": "diverged-non-finite-gradient",
                        "step": step,
                        "trainRegionVisits": region_visits,
                        "datasetFingerprint": manifest.get("fingerprint"),
                    },
                )
                return 3, run_dir

            optimizer.step()
            loss_value = float(loss.detach().item())
            interval_loss += loss_value
            interval_count += 1
            completed_steps = step

            if step == 1 or step % 64 == 0:
                print(
                    f"  step {step:4d}/{max_steps} "
                    f"region={region_index + 1}/{len(train_records)} "
                    f"loss={loss_value:.6f} grad={float(grad_norm):.4f}",
                    flush=True,
                )

            if step % validate_every != 0 and step != max_steps:
                continue

            train_report = self._evaluate_records(
                model,
                train_records,
                config,
            )
            validation_report = self._evaluate_records(
                model,
                validation_records,
                config,
            )
            diagnosis = self._diagnosis(train_report, validation_report)
            generalisation_pass = diagnosis == "passed"
            curve_item: dict[str, object] = {
                "step": step,
                "meanTrainingLossSincePreviousEvaluation": (
                    interval_loss / max(1, interval_count)
                ),
                "trainRegionVisits": list(region_visits),
                "train": train_report,
                "validation": validation_report,
                "diagnosis": diagnosis,
            }
            curve.append(curve_item)
            self._write_curve(run_dir, curve)
            interval_loss = 0.0
            interval_count = 0

            print(
                "  train    "
                f"global={float(train_report['medianGlobalRecovery'])*100:+.2f}% "
                f"edge={float(train_report['medianEdgeRecovery'])*100:+.2f}% "
                f"grad={float(train_report['medianGradientRecovery'])*100:+.2f}% "
                f"lattice={float(train_report['maxLatticeCellExcess'])*100:+.2f}% "
                f"qualified={'YES' if train_report['passed'] else 'NO'}",
                flush=True,
            )
            print(
                "  held-out "
                f"global={float(validation_report['medianGlobalRecovery'])*100:+.2f}% "
                f"edge={float(validation_report['medianEdgeRecovery'])*100:+.2f}% "
                f"grad={float(validation_report['medianGradientRecovery'])*100:+.2f}% "
                f"lattice={float(validation_report['maxLatticeCellExcess'])*100:+.2f}% "
                f"qualified={'YES' if validation_report['passed'] else 'NO'} "
                f"diagnosis={diagnosis}",
                flush=True,
            )

            score = self._score(validation_report)
            key = (1 if generalisation_pass else 0, score)
            if best_validation_report is None or key > best_key:
                best_key = key
                best_step = step
                best_train_report = train_report
                best_validation_report = validation_report
                save_checkpoint(
                    best_path,
                    model,
                    config,
                    epoch=step,
                    phase="v16.0-mini-multiregion-step",
                    metrics={
                        "step": step,
                        "train": train_report,
                        "validation": validation_report,
                        "diagnosis": diagnosis,
                    },
                )

            if generalisation_pass:
                early_pass = True
                print(
                    f"[v16.0-multiregion] early PASS at step {step}; "
                    "training and held-out qualification gates are satisfied",
                    flush=True,
                )
                break

        if best_validation_report is None or best_train_report is None or best_step < 1:
            raise RuntimeError("V16.0 multi-region diagnostic produced no evaluated checkpoint")

        selected_model, _payload = load_checkpoint(best_path, self.device)
        selected_train_report = self._evaluate_records(
            selected_model,
            train_records,
            config,
        )
        selected_validation_report = self._evaluate_records(
            selected_model,
            validation_records,
            config,
        )
        selected_train_report["selectedCheckpoint"] = str(best_path.resolve())
        selected_train_report["selectedStep"] = int(best_step)
        selected_validation_report["selectedCheckpoint"] = str(best_path.resolve())
        selected_validation_report["selectedStep"] = int(best_step)
        selected_diagnosis = self._diagnosis(
            selected_train_report,
            selected_validation_report,
        )
        selected_pass = selected_diagnosis == "passed"

        preview_batch = dataset_sample(validation_records[0], config, self.device)
        selected_model.eval()
        with torch.no_grad(), autocast_context(self.device, self.args.amp_precision):
            preview_outputs = selected_model(
                preview_batch["lr_albedo"],
                preview_batch["lr_normal"],
                preview_batch["lr_material"],
            )
        probe_path = save_probe(
            run_dir,
            preview_batch,
            preview_outputs,
            include_final=False,
        )
        curve_path = self._write_curve(run_dir, curve)
        report = {
            "schema": DIAGNOSTIC_SCHEMA,
            "revision": DIAGNOSTIC_REVISION,
            "mode": "multiregion",
            "passed": selected_pass,
            "promotable": False,
            "modelSchema": MODEL_SCHEMA,
            "diagnosis": selected_diagnosis,
            "candidate": selected_validation_report,
            "trainCandidate": selected_train_report,
            "candidateCheckpoint": str(best_path.resolve()),
            "selectedStep": int(best_step),
            "completedSteps": int(completed_steps),
            "earlyPass": bool(early_pass),
            "trainingBudget": {
                "maximumSteps": int(max_steps),
                "validationInterval": int(validate_every),
                "learningRate": float(self.args.learning_rate),
                "sampling": "balanced-round-robin",
                "trainRegionVisits": list(region_visits),
            },
            "trainRecords": [str(record["path"]) for record in train_records],
            "validationRecords": [str(record["path"]) for record in validation_records],
            "trainSourceBoxes": [record.get("source_box") for record in train_records],
            "validationSourceBoxes": [record.get("source_box") for record in validation_records],
            "datasetSplitPolicy": manifest.get("splitPolicy"),
            "generalisationCurve": str(curve_path.resolve()),
            "probe": str(probe_path.resolve()),
            "datasetFingerprint": manifest.get("fingerprint"),
        }
        write_report(run_dir, report)
        return (0 if selected_pass else 2), run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSAMDR V16.0 multi-region SR diagnostic")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    p.add_argument("--rebuild-dataset", action="store_true")
    p.add_argument("--prepare-train-regions", type=int, default=16)
    p.add_argument("--prepare-validation-regions", type=int, default=4)
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    p.add_argument("--required-edge-recovery", type=float, default=0.60)
    p.add_argument("--required-global-recovery", type=float, default=0.45)
    p.add_argument("--required-gradient-recovery", type=float, default=0.35)
    p.add_argument("--train-regions", type=int, default=4)
    p.add_argument("--validation-regions", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--validate-every", type=int, default=DEFAULT_VALIDATE_EVERY)
    p.add_argument("--learning-rate", type=float, default=2.0e-4)
    # Kept for old direct invocations. V16 Multi-Region no longer uses epoch budget.
    p.add_argument("--epochs", type=int, default=3, help=argparse.SUPPRESS)
    p.add_argument("--tiles-per-epoch", type=int, default=64, help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    diagnostic = MultiRegionDiagnostic(args, repo_root, device_from_name(args.device))
    code, run_dir = diagnostic.run()
    archive = archive_run(run_dir)
    print(f"[v16.0-multiregion] report      : {run_dir / 'report.json'}", flush=True)
    print(f"[v16.0-multiregion] diagnostics : {archive}", flush=True)
    print(f"[v16.0-multiregion] result      : {'PASS' if code == 0 else 'FAIL'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
