#!/usr/bin/env python3
"""V14.3 BenefitSelector retention diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    NEURAL_ROOT = Path(__file__).resolve().parent.parent
    if str(NEURAL_ROOT) not in sys.path:
        sys.path.insert(0, str(NEURAL_ROOT))
    from v14.checkpoint import load_checkpoint, save_checkpoint
    from v14.dataset import load_manifest
    from v14.diagnostic_support import (
        DIAGNOSTIC_REVISION,
        DIAGNOSTIC_SCHEMA,
        archive_run,
        autocast_context,
        dataset_sample,
        device_from_name,
        iter_train_batches,
        make_run_directory,
        save_probe,
        validation_metrics,
        write_report,
    )
    from v14.losses import selector_loss
    from v14.model import MODEL_SCHEMA
    from v14.qualification import aggregate_final
else:
    from .checkpoint import load_checkpoint, save_checkpoint
    from .dataset import load_manifest
    from .diagnostic_support import (
        DIAGNOSTIC_REVISION,
        DIAGNOSTIC_SCHEMA,
        archive_run,
        autocast_context,
        dataset_sample,
        device_from_name,
        iter_train_batches,
        make_run_directory,
        save_probe,
        validation_metrics,
        write_report,
    )
    from .losses import selector_loss
    from .model import MODEL_SCHEMA
    from .qualification import aggregate_final


def _latest_passing_multiregion(
    repo_root: Path,
) -> tuple[Path, dict[str, Any]]:
    root = repo_root / "artifacts/nsamdr/diagnostics/v14_mini"
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    if root.is_dir():
        for path in root.glob("multiregion_*/report.json"):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
                checkpoint_path = Path(str(report.get("candidateCheckpoint") or ""))
            except (OSError, ValueError, TypeError):
                continue
            if (
                report.get("schema") == DIAGNOSTIC_SCHEMA
                and report.get("revision") == DIAGNOSTIC_REVISION
                and report.get("mode") == "multiregion"
                and report.get("modelSchema") == MODEL_SCHEMA
                and report.get("passed") is True
                and checkpoint_path.is_file()
            ):
                candidates.append((path.stat().st_mtime, path, report))
    if not candidates:
        raise RuntimeError(
            "Selector Retention Mini requires a passing V14.3 Multi-Region SR Mini first"
        )
    _mtime, path, report = max(candidates, key=lambda item: item[0])
    return path, report


class SelectorRetentionDiagnostic:
    """Freeze a passing V14.3 candidate and train only BenefitSelector."""

    def __init__(
        self,
        args: argparse.Namespace,
        repo_root: Path,
        device: torch.device,
    ) -> None:
        self.args = args
        self.repo_root = repo_root
        self.device = device

    def _source_records(
        self,
        manifest: dict[str, Any],
        source_report: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        train_paths = set(map(str, source_report.get("trainRecords") or []))
        validation_paths = set(map(str, source_report.get("validationRecords") or []))
        train = [
            record
            for record in manifest["crops"]
            if str(record.get("path")) in train_paths
        ]
        validation = [
            record
            for record in manifest["crops"]
            if str(record.get("path")) in validation_paths
        ]
        return train, validation

    def run(self) -> tuple[int, Path]:
        source_report_path, source_report = _latest_passing_multiregion(self.repo_root)
        source_checkpoint = Path(str(source_report["candidateCheckpoint"]))
        model, _payload = load_checkpoint(source_checkpoint, self.device)
        config = model.config
        config.selector_edge_retention_required = float(self.args.required_retention)
        config.selector_global_retention_required = float(self.args.required_retention)
        config.protected_preservation_required = float(self.args.protected_preservation)
        config.validate()

        manifest = load_manifest(self.repo_root, config)
        if source_report.get("datasetFingerprint") != manifest.get("fingerprint"):
            raise RuntimeError(
                "Raven dataset changed since the passing V14.3 multi-region mini; rerun it first"
            )

        train_records, validation_records = self._source_records(manifest, source_report)
        if len(train_records) < 2 or len(validation_records) < 2:
            raise RuntimeError(
                "Passing V14.3 multi-region mini no longer resolves its original Raven regions"
            )
        source_candidate = dict(source_report.get("candidate") or {})
        if source_candidate.get("passed") is not True:
            raise RuntimeError(
                "Passing V14.3 multi-region report has no qualified candidate block"
            )
        subset_manifest = {"crops": [*train_records, *validation_records]}

        model.set_selector_training()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(self.args.selector_learning_rate),
            weight_decay=config.weight_decay,
        )
        run_dir = make_run_directory(self.repo_root, "selector")
        best_key = (-1, -1.0e9)
        best_path: Path | None = None
        best_report: dict[str, object] | None = None

        print("=" * 76, flush=True)
        print("V14.3 SELECTOR RETENTION MINI - DIAGNOSTIC ONLY", flush=True)
        print(f"Candidate source : {source_report_path.parent}", flush=True)
        print(f"Selector epochs  : {self.args.selector_epochs}", flush=True)
        print(f"Tiles/epoch      : {self.args.tiles_per_epoch}", flush=True)
        print("=" * 76, flush=True)

        for epoch in range(1, int(self.args.selector_epochs) + 1):
            model.train()
            for index, batch in enumerate(
                iter_train_batches(
                    subset_manifest,
                    config,
                    int(self.args.tiles_per_epoch),
                    seed=config.seed + 900 + epoch,
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
                    losses = selector_loss(outputs, batch)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                if index == 1 or index % 16 == 0 or index == int(self.args.tiles_per_epoch):
                    print(
                        f"  epoch {epoch}/{self.args.selector_epochs} tile {index:3d}/{self.args.tiles_per_epoch} "
                        f"loss={float(losses['total'].detach().item()):.6f}",
                        flush=True,
                    )

            final_metrics = validation_metrics(
                model,
                subset_manifest,
                config,
                self.device,
                self.args.amp_precision,
                final=True,
            )
            report = aggregate_final(source_candidate, final_metrics, config)
            checkpoint_path = run_dir / "checkpoints" / f"selector_epoch_{epoch:02d}.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                config,
                epoch=epoch,
                phase="v14.3-mini-selector",
                metrics=report,
            )
            score = float(report["medianGlobalRecovery"]) + float(report["medianEdgeRecovery"])
            key = (1 if bool(report["passed"]) else 0, score)
            if best_report is None or key > best_key:
                best_key = key
                best_path = checkpoint_path
                best_report = report
            print(
                "  final "
                f"global={float(report['medianGlobalRecovery'])*100:+.2f}% "
                f"edge={float(report['medianEdgeRecovery'])*100:+.2f}% "
                f"edgeRetention={float(report['selectorEdgeRetention'])*100:.1f}% "
                f"globalRetention={float(report['selectorGlobalRetention'])*100:.1f}% "
                f"protected={float(report['medianProtectedPreservation'])*100:.2f}% "
                f"qualified={'YES' if report['passed'] else 'NO'}",
                flush=True,
            )

        if best_path is None or best_report is None:
            raise RuntimeError("V14.3 selector diagnostic produced no checkpoint")

        selected_model, _payload = load_checkpoint(best_path, self.device)
        final_metrics = validation_metrics(
            selected_model,
            subset_manifest,
            config,
            self.device,
            self.args.amp_precision,
            final=True,
        )
        selected_report = aggregate_final(source_candidate, final_metrics, config)
        selected_report["selectedCheckpoint"] = str(best_path.resolve())

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
            include_final=True,
        )
        report = {
            "schema": DIAGNOSTIC_SCHEMA,
            "revision": DIAGNOSTIC_REVISION,
            "mode": "selector",
            "passed": bool(selected_report["passed"]),
            "promotable": False,
            "modelSchema": MODEL_SCHEMA,
            "sourceMultiregionReport": str(source_report_path.resolve()),
            "candidate": source_candidate,
            "final": selected_report,
            "selectorCheckpoint": str(best_path.resolve()),
            "probe": str(probe_path.resolve()),
            "datasetFingerprint": manifest.get("fingerprint"),
        }
        write_report(run_dir, report)
        return (0 if bool(selected_report["passed"]) else 2), run_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NSAMDR V14.3 selector retention diagnostic")
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    p.add_argument("--selector-epochs", type=int, default=2)
    p.add_argument("--tiles-per-epoch", type=int, default=64)
    p.add_argument("--selector-learning-rate", type=float, default=2.0e-4)
    p.add_argument("--required-retention", type=float, default=0.90)
    p.add_argument("--protected-preservation", type=float, default=0.99)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    diagnostic = SelectorRetentionDiagnostic(args, repo_root, device_from_name(args.device))
    code, run_dir = diagnostic.run()
    archive = archive_run(run_dir)
    print(f"[v14.3-selector] report      : {run_dir / 'report.json'}", flush=True)
    print(f"[v14.3-selector] diagnostics : {archive}", flush=True)
    print(f"[v14.3-selector] result      : {'PASS' if code == 0 else 'FAIL'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
