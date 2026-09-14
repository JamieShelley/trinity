#!/usr/bin/env python3
"""V14.4 multi-region candidate generalisation diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    NEURAL_ROOT = Path(__file__).resolve().parent.parent
    if str(NEURAL_ROOT) not in sys.path:
        sys.path.insert(0, str(NEURAL_ROOT))
    from v14.checkpoint import load_checkpoint, save_checkpoint
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
        iter_train_batches,
        make_run_directory,
        prepare_raven_dataset,
        save_probe,
        validation_metrics,
        write_report,
    )
    from v14.losses import candidate_loss
    from v14.model import MODEL_SCHEMA, NSAMDRV14
    from v14.qualification import aggregate_candidate
else:
    from .checkpoint import load_checkpoint, save_checkpoint
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
        iter_train_batches,
        make_run_directory,
        prepare_raven_dataset,
        save_probe,
        validation_metrics,
        write_report,
    )
    from .losses import candidate_loss
    from .model import MODEL_SCHEMA, NSAMDRV14
    from .qualification import aggregate_candidate


class MultiRegionDiagnostic:
    """Test whether V14.4 candidate gains survive disjoint Raven regions."""

    def __init__(
        self,
        args: argparse.Namespace,
        repo_root: Path,
        device: torch.device,
    ) -> None:
        self.args = args
        self.repo_root = repo_root
        self.device = device

    def _config(self) -> V14Config:
        config = V14Config(
            minimum_heldout_samples=2,
            candidate_edge_recovery_required=float(self.args.required_edge_recovery),
            candidate_global_recovery_required=float(self.args.required_global_recovery),
            candidate_gradient_recovery_required=float(self.args.required_gradient_recovery),
        )
        config.tiles_per_epoch = int(self.args.tiles_per_epoch)
        config.clean_epochs = int(self.args.epochs)
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

    def run(self) -> tuple[int, Path]:
        self.args.prepare_train_regions = max(
            int(self.args.prepare_train_regions),
            int(self.args.train_regions),
        )
        self.args.prepare_validation_regions = max(
            int(self.args.prepare_validation_regions),
            int(self.args.validation_regions),
        )
        prepare_raven_dataset(self.args, self.repo_root)
        config = self._config()
        manifest = load_manifest(self.repo_root, config)
        train_records, validation_records = self._records(manifest)
        run_dir = make_run_directory(self.repo_root, "multiregion")

        if len(train_records) < 2 or len(validation_records) < 2:
            write_report(
                run_dir,
                {
                    "schema": DIAGNOSTIC_SCHEMA,
                    "revision": DIAGNOSTIC_REVISION,
                    "mode": "multiregion",
                    "modelSchema": MODEL_SCHEMA,
                    "passed": False,
                    "promotable": False,
                    "reason": "insufficient-disjoint-regions",
                    "trainRegions": len(train_records),
                    "validationRegions": len(validation_records),
                    "requiredTrainRegions": 2,
                    "requiredValidationRegions": 2,
                    "datasetFingerprint": manifest.get("fingerprint"),
                },
            )
            return 2, run_dir

        subset_manifest = {"crops": [*train_records, *validation_records]}
        model = NSAMDRV14(config).to(self.device)
        model.set_candidate_training()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(self.args.learning_rate),
            weight_decay=config.weight_decay,
        )
        best_key = (-1, -1.0e9)
        best_path: Path | None = None
        best_report: dict[str, object] | None = None

        print("=" * 76, flush=True)
        print("V14.4 MULTI-REGION SR MINI - DIAGNOSTIC ONLY", flush=True)
        print(
            f"Train/held-out : {len(train_records)} / {len(validation_records)} disjoint regions",
            flush=True,
        )
        print(f"Epochs         : {self.args.epochs} clean SR", flush=True)
        print(f"Tiles/epoch    : {self.args.tiles_per_epoch}", flush=True)
        print("=" * 76, flush=True)

        for epoch in range(1, int(self.args.epochs) + 1):
            model.train()
            running = 0.0
            for index, batch in enumerate(
                iter_train_batches(
                    subset_manifest,
                    config,
                    int(self.args.tiles_per_epoch),
                    seed=config.seed + epoch * 31,
                    degradation="clean",
                    device=self.device,
                ),
                start=1,
            ):
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(self.device, self.args.amp_precision):
                    outputs = model(
                        batch["lr_albedo"],
                        batch["lr_normal"],
                        batch["lr_material"],
                    )
                    losses = candidate_loss(outputs, batch, config)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                running += float(losses["total"].detach().item())
                if index == 1 or index % 16 == 0 or index == int(self.args.tiles_per_epoch):
                    print(
                        f"  epoch {epoch}/{self.args.epochs} tile {index:3d}/{self.args.tiles_per_epoch} "
                        f"loss={running/index:.6f}",
                        flush=True,
                    )

            metrics = validation_metrics(
                model,
                subset_manifest,
                config,
                self.device,
                self.args.amp_precision,
                final=False,
            )
            report = aggregate_candidate(metrics, config)
            report["epoch"] = epoch
            score = (
                float(report["medianGlobalRecovery"])
                + float(report["medianEdgeRecovery"])
                + float(report["medianGradientRecovery"])
            )
            checkpoint_path = run_dir / "checkpoints" / f"candidate_epoch_{epoch:02d}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                config,
                epoch=epoch,
                phase="v14.4-mini-multiregion",
                metrics=report,
            )
            print(
                "  held-out "
                f"global={float(report['medianGlobalRecovery'])*100:+.2f}% "
                f"edge={float(report['medianEdgeRecovery'])*100:+.2f}% "
                f"grad={float(report['medianGradientRecovery'])*100:+.2f}% "
                f"samples={int(report['sampleCount'])} "
                f"qualified={'YES' if report['passed'] else 'NO'}",
                flush=True,
            )
            key = (1 if bool(report["passed"]) else 0, score)
            if best_report is None or key > best_key:
                best_key = key
                best_path = checkpoint_path
                best_report = report

        if best_path is None or best_report is None:
            raise RuntimeError("V14.4 multi-region diagnostic produced no checkpoint")

        selected_model, _payload = load_checkpoint(best_path, self.device)
        selected_metrics = validation_metrics(
            selected_model,
            subset_manifest,
            config,
            self.device,
            self.args.amp_precision,
            final=False,
        )
        selected_report = aggregate_candidate(selected_metrics, config)
        selected_report["selectedCheckpoint"] = str(best_path.resolve())
        selected_report["selectedEpoch"] = int(best_report["epoch"])

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
        report = {
            "schema": DIAGNOSTIC_SCHEMA,
            "revision": DIAGNOSTIC_REVISION,
            "mode": "multiregion",
            "passed": bool(selected_report["passed"]),
            "promotable": False,
            "modelSchema": MODEL_SCHEMA,
            "candidate": selected_report,
            "candidateCheckpoint": str(best_path.resolve()),
            "trainRecords": [str(record["path"]) for record in train_records],
            "validationRecords": [str(record["path"]) for record in validation_records],
            "probe": str(probe_path.resolve()),
            "datasetFingerprint": manifest.get("fingerprint"),
        }
        write_report(run_dir, report)
        return (0 if bool(selected_report["passed"]) else 2), run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSAMDR V14.4 multi-region SR diagnostic")
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
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--tiles-per-epoch", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=2.0e-4)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    diagnostic = MultiRegionDiagnostic(args, repo_root, device_from_name(args.device))
    code, run_dir = diagnostic.run()
    archive = archive_run(run_dir)
    print(f"[v14.4-multiregion] report      : {run_dir / 'report.json'}", flush=True)
    print(f"[v14.4-multiregion] diagnostics : {archive}", flush=True)
    print(f"[v14.4-multiregion] result      : {'PASS' if code == 0 else 'FAIL'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
