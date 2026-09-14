from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import promote_final, save_checkpoint
from .config import V16Config
from .dataset import RavenSRDataset, load_manifest, native_family_samples
from .losses import candidate_loss, selector_loss
from .model import MODEL_SCHEMA, NSAMDRV16
from .qualification import aggregate_candidate, aggregate_final, sample_metrics


class V16Trainer:
    """Own V16 candidate training, qualification, selector training, and promotion."""

    def __init__(
        self,
        repo_root: Path,
        experiment_dir: Path,
        config: V16Config,
        *,
        device: torch.device,
        workers: int,
        prefetch_factor: int,
        amp_precision: str,
        live_preview_target_size: int,
    ) -> None:
        self.repo_root = repo_root
        self.experiment_dir = experiment_dir
        self.config = config
        self.device = device
        self.workers = max(0, int(workers))
        self.prefetch_factor = max(1, int(prefetch_factor))
        self.amp_precision = amp_precision
        self.live_preview_target_size = int(live_preview_target_size)
        self.manifest = load_manifest(repo_root, config)
        self.native_samples = native_family_samples(self.manifest, config)
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        self.model = NSAMDRV16(config).to(device)

    def _autocast(self):
        if self.device.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)
        dtype = (
            torch.bfloat16
            if self.amp_precision in {"auto", "bf16"}
            else torch.float16
        )
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _to_device(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.to(self.device, non_blocking=True)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }

    def _loader(
        self,
        split: str,
        length: int,
        degradation: str,
        seed: int,
    ) -> DataLoader:
        dataset = RavenSRDataset(
            self.manifest,
            self.config,
            split,
            length,
            seed=seed,
            degradation=degradation,
        )
        kwargs: dict[str, Any] = {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": self.workers if split == "train" else min(1, self.workers),
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": self.workers > 0 and split == "train",
        }
        if kwargs["num_workers"]:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(dataset, **kwargs)

    def _validate(
        self,
        *,
        final: bool = False,
    ) -> tuple[list[dict[str, float]], dict[str, object]]:
        loader = self._loader(
            "validation",
            self.config.validation_tiles,
            "clean",
            self.config.seed + 700,
        )
        metrics: list[dict[str, float]] = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                with self._autocast():
                    outputs = self.model(
                        batch["lr_albedo"],
                        batch["lr_normal"],
                        batch["lr_material"],
                    )
                metrics.append(sample_metrics(outputs, batch, final=final))

            # Native 256->1024 samples are scale telemetry only. Their dataset samples
            # do not carry held-out record identity, so aggregate qualification cannot
            # use them to satisfy held-out coverage.
            for sample in self.native_samples:
                batch = {
                    key: value.to(self.device)
                    for key, value in sample.items()
                    if isinstance(value, torch.Tensor)
                }
                with self._autocast():
                    outputs = self.model(
                        batch["lr_albedo"],
                        batch["lr_normal"],
                        batch["lr_material"],
                    )
                metrics.append(sample_metrics(outputs, batch, final=final))

        report = aggregate_candidate(metrics, self.config)
        report["nativeValidationSamples"] = len(self.native_samples)
        return metrics, report

    @staticmethod
    def _u8_rgb(value: torch.Tensor) -> np.ndarray:
        image = (
            value.detach()
            .float()
            .clamp(0.0, 1.0)[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return np.round(image * 255.0).astype(np.uint8)

    def _save_preview(self, epoch: int, phase: str) -> None:
        if not self.native_samples:
            return
        sample = self.native_samples[0]
        batch = {
            key: value.to(self.device)
            for key, value in sample.items()
            if isinstance(value, torch.Tensor)
        }
        self.model.eval()
        with torch.no_grad(), self._autocast():
            outputs = self.model(
                batch["lr_albedo"],
                batch["lr_normal"],
                batch["lr_material"],
            )
        target = batch["target_albedo"]
        a = self._u8_rgb(target)
        b = self._u8_rgb(outputs["baseline_albedo"])
        c = self._u8_rgb(outputs["candidate_albedo"])
        f = self._u8_rgb(outputs["albedo"])
        if (
            self.live_preview_target_size > 0
            and a.shape[0] != self.live_preview_target_size
        ):
            size = (self.live_preview_target_size, self.live_preview_target_size)
            a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
            c = cv2.resize(c, size, interpolation=cv2.INTER_AREA)
            f = cv2.resize(f, size, interpolation=cv2.INTER_AREA)

        folder = (
            self.experiment_dir
            / "previews"
            / "live"
            / "candidates"
            / f"epoch_{epoch:04d}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        for name, image in (
            ("A_authored.png", a),
            ("B_baseline.png", b),
            ("C_candidate.png", c),
            ("F_final.png", f),
        ):
            cv2.imwrite(str(folder / name), image[:, :, ::-1])

        panels = []
        for label, image in (
            ("A AUTHORED", a),
            ("B BASELINE", b),
            ("C V16 SR", c),
            ("F SELECTED", f),
        ):
            panel = image.copy()
            cv2.rectangle(
                panel,
                (0, 0),
                (panel.shape[1], 42),
                (0, 0, 0),
                -1,
            )
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
            panels.append(panel)
        contact = np.concatenate(panels, axis=1)
        live_root = self.experiment_dir / "previews" / "live"
        cv2.imwrite(str(folder / "ABCF.png"), contact[:, :, ::-1])
        cv2.imwrite(str(live_root / "latest_ABCF.png"), contact[:, :, ::-1])
        pointer = {
            "schema": "NSAMDR_V16_LIVE_PREVIEW_V1",
            "epoch": epoch,
            "phase": phase,
            "contactSheet": str((folder / "ABCF.png").resolve()),
            "nativeTarget": int(target.shape[-1]),
            "lrInput": int(batch["lr_albedo"].shape[-1]),
        }
        (live_root / "checkpoint_ready.json").write_text(
            json.dumps(pointer, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[live-preview] epoch {epoch} {phase}: A {target.shape[-1]} / "
            f"B,C {outputs['candidate_albedo'].shape[-1]} from LR "
            f"{batch['lr_albedo'].shape[-1]} -> {folder / 'ABCF.png'}",
            flush=True,
        )

    def _train_candidate(self) -> tuple[Path, dict[str, object]]:
        self.model.set_candidate_training()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.config.sr_learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-8,
        )
        best_key = (-1, -1.0e9)
        best_path: Path | None = None
        best_report: dict[str, object] | None = None

        for epoch in range(1, self.config.sr_epochs + 1):
            degradation = (
                "clean" if epoch <= self.config.clean_epochs else "robust"
            )
            loader = self._loader(
                "train",
                self.config.tiles_per_epoch,
                degradation,
                self.config.seed + epoch * 17,
            )
            self.model.train()
            start = time.perf_counter()
            running = 0.0
            print(
                f"Epoch {epoch:03d}/{self.config.sr_epochs + self.config.selector_epochs:03d} "
                f"phase=sr-{degradation} lr={self.config.sr_learning_rate:.7f}",
                flush=True,
            )
            for index, batch in enumerate(loader, start=1):
                batch = self._to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    outputs = self.model(
                        batch["lr_albedo"],
                        batch["lr_normal"],
                        batch["lr_material"],
                    )
                    losses = candidate_loss(outputs, batch, self.config)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                running += float(losses["total"].detach().item())
                if index == 1 or index % 24 == 0 or index == len(loader):
                    elapsed = max(time.perf_counter() - start, 1.0e-6)
                    print(
                        f"  {index:4d}/{len(loader):4d} total={running/index:.6f} "
                        f"rate={index/elapsed:.2f}tile/s",
                        flush=True,
                    )

            _metrics, report = self._validate(final=False)
            report["epoch"] = epoch
            report["degradation"] = degradation
            score = (
                float(report["medianGlobalRecovery"])
                + float(report["medianEdgeRecovery"])
                + float(report["medianGradientRecovery"])
            )
            checkpoint_path = (
                self.experiment_dir
                / "checkpoints"
                / "candidate"
                / f"epoch_{epoch:04d}.pt"
            )
            save_checkpoint(
                checkpoint_path,
                self.model,
                self.config,
                epoch=epoch,
                phase=f"sr-{degradation}",
                metrics=report,
            )
            print(
                "  valid "
                f"global={float(report['medianGlobalRecovery'])*100:+.2f}% "
                f"edge={float(report['medianEdgeRecovery'])*100:+.2f}% "
                f"grad={float(report['medianGradientRecovery'])*100:+.2f}% "
                f"wins={float(report['positiveGlobalFraction'])*100:.1f}% "
                f"latticeExcess={float(report['maxLatticeCellExcess'])*100:+.1f}% "
                f"qualified={'YES' if report['passed'] else 'NO'}",
                flush=True,
            )
            self._save_preview(epoch, f"sr-{degradation}")
            selection_key = (1 if bool(report["passed"]) else 0, score)
            if selection_key > best_key:
                best_key = selection_key
                best_path = checkpoint_path
                best_report = report

        if best_path is None or best_report is None:
            raise RuntimeError("V16 candidate training produced no checkpoint")

        model, _ = self._reload(best_path)
        self.model = model
        _metrics, selected_report = self._validate(final=False)
        selected_report["selectedCheckpoint"] = str(best_path.resolve())
        selected_report["selectedEpoch"] = int(best_report["epoch"])
        return best_path, selected_report

    def _reload(self, path: Path) -> tuple[NSAMDRV16, dict[str, Any]]:
        from .checkpoint import load_checkpoint

        return load_checkpoint(path, self.device)

    def _train_selector(
        self,
        candidate_path: Path,
        candidate_report: dict[str, object],
    ) -> tuple[Path, dict[str, object]]:
        self.model, _ = self._reload(candidate_path)
        self.model.set_selector_training()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.config.selector_learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-8,
        )
        best_path: Path | None = None
        best_report: dict[str, object] | None = None
        best_key = (-1, -1.0e9)

        for step in range(1, self.config.selector_epochs + 1):
            epoch = self.config.sr_epochs + step
            loader = self._loader(
                "train",
                self.config.tiles_per_epoch,
                "robust",
                self.config.seed + 900 + step,
            )
            self.model.train()
            start = time.perf_counter()
            running = 0.0
            print(
                f"Epoch {epoch:03d}/{self.config.sr_epochs + self.config.selector_epochs:03d} "
                f"phase=selector lr={self.config.selector_learning_rate:.7f}",
                flush=True,
            )
            for index, batch in enumerate(loader, start=1):
                batch = self._to_device(batch)
                optimizer.zero_grad(set_to_none=True)
                with self._autocast():
                    outputs = self.model(
                        batch["lr_albedo"],
                        batch["lr_normal"],
                        batch["lr_material"],
                    )
                    losses = selector_loss(outputs, batch)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                running += float(losses["total"].detach().item())
                if index == 1 or index % 24 == 0 or index == len(loader):
                    elapsed = max(time.perf_counter() - start, 1.0e-6)
                    print(
                        f"  {index:4d}/{len(loader):4d} total={running/index:.6f} "
                        f"rate={index/elapsed:.2f}tile/s",
                        flush=True,
                    )

            final_metrics, _ = self._validate(final=True)
            report = aggregate_final(candidate_report, final_metrics, self.config)
            report["epoch"] = epoch
            score = float(report["medianGlobalRecovery"]) + float(
                report["medianEdgeRecovery"]
            )
            checkpoint_path = (
                self.experiment_dir
                / "checkpoints"
                / "selector"
                / f"epoch_{epoch:04d}.pt"
            )
            save_checkpoint(
                checkpoint_path,
                self.model,
                self.config,
                epoch=epoch,
                phase="selector",
                metrics=report,
            )
            print(
                "  valid-final "
                f"global={float(report['medianGlobalRecovery'])*100:+.2f}% "
                f"edge={float(report['medianEdgeRecovery'])*100:+.2f}% "
                f"protected={float(report['medianProtectedPreservation'])*100:.2f}% "
                f"qualified={'YES' if report['passed'] else 'NO'}",
                flush=True,
            )
            self._save_preview(epoch, "selector")
            selection_key = (1 if bool(report["passed"]) else 0, score)
            if selection_key > best_key:
                best_key = selection_key
                best_path = checkpoint_path
                best_report = report

        if best_path is None or best_report is None:
            raise RuntimeError("V16 selector training produced no checkpoint")

        self.model, _ = self._reload(best_path)
        final_metrics, _ = self._validate(final=True)
        selected_report = aggregate_final(
            candidate_report,
            final_metrics,
            self.config,
        )
        selected_report["selectedCheckpoint"] = str(best_path.resolve())
        selected_report["selectedEpoch"] = int(best_report["epoch"])
        return best_path, selected_report

    def run(self) -> dict[str, object]:
        architecture = self.model.architecture_contract()
        (self.experiment_dir / "architecture.json").write_text(
            json.dumps(architecture, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("=" * 72, flush=True)
        print("NSAMDR V16.0 HR-FIRST SWINIR SR", flush=True)
        print(f"Model schema             : {MODEL_SCHEMA}", flush=True)
        print(
            "Production authority     : B -> SwinIR HR refinement C -> BenefitSelector F",
            flush=True,
        )
        print(
            f"Training geometry        : {self.config.train_lr_size} LR -> "
            f"{self.config.train_hr_size} HR",
            flush=True,
        )
        print(
            f"Swin backbone            : {self.config.swin_groups} groups x "
            f"{self.config.swin_blocks_per_group} layers, window "
            f"{self.config.swin_window_size}",
            flush=True,
        )
        print(
            f"Native validation samples: {len(self.native_samples)}",
            flush=True,
        )
        print(
            f"Parameters               : {sum(p.numel() for p in self.model.parameters()):,}",
            flush=True,
        )
        print("Legacy geometry/SDF/seam : NOT CONSTRUCTED", flush=True)
        print("=" * 72, flush=True)

        candidate_path, candidate_report = self._train_candidate()
        (self.experiment_dir / "candidate_qualification.json").write_text(
            json.dumps(candidate_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not bool(candidate_report.get("passed")):
            return {
                "status": "training-rejected",
                "qualified": False,
                "failedStage": "sr-candidate",
                "candidate": candidate_report,
                "checkpoint": str(candidate_path),
            }

        selector_path, final_report = self._train_selector(
            candidate_path,
            candidate_report,
        )
        (self.experiment_dir / "final_qualification.json").write_text(
            json.dumps(final_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not bool(final_report.get("passed")):
            return {
                "status": "training-rejected",
                "qualified": False,
                "failedStage": "benefit-selector",
                "candidate": candidate_report,
                "final": final_report,
                "checkpoint": str(selector_path),
            }

        final_path = promote_final(
            selector_path,
            self.experiment_dir,
            final_report,
        )
        return {
            "status": "completed",
            "qualified": True,
            "candidate": candidate_report,
            "final": final_report,
            "checkpoint": str(final_path),
        }


# Compatibility alias for existing workflow imports.
V14Trainer = V16Trainer
