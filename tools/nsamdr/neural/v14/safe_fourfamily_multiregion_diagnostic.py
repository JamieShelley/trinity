#!/usr/bin/env python3
"""Crash-resilient V16 Stage 2 runtime for the four-family diagnostic.

This keeps the V16 model, losses, optimizer, data split, qualification gates, and
balanced family selection unchanged.  It adds only execution safety:

- durable JSONL progress and heartbeat output,
- periodic atomic resume checkpoints including optimizer state,
- compatible-run resume after a process/PC interruption,
- optional CUDA pacing and thermal pauses,
- elapsed/ETA/VRAM/GPU telemetry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
NEURAL_ROOT = HERE.parent
if str(NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURAL_ROOT))

# Importing this module installs the existing multi-family and four-family record
# selection/evaluation hooks on the base diagnostic class.
from v14 import fourfamily_multiregion_diagnostic as four

base = four.multi.base

RESUME_SCHEMA = "NSAMDR_V16_STAGE2_RESUME_V1"
SAFE_PROFILE = {
    "checkpointEvery": 128,
    "progressEvery": 32,
    "cooldownEvery": 16,
    "cooldownSeconds": 0.5,
    "maxGpuTemperatureC": 80.0,
    "temperatureResumeMarginC": 5.0,
    "thermalPollSeconds": 5.0,
    "validationCooldownSeconds": 2.0,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Stage 2 resume checkpoint is not a dictionary: {path}")
    return payload


def _record_paths(records: list[dict[str, Any]]) -> list[str]:
    return [str(record.get("path") or "") for record in records]


class SafeFourFamilyDiagnostic(base.MultiRegionDiagnostic):
    def _diagnostics_root(self) -> Path:
        return self.repo_root / "artifacts/nsamdr/diagnostics/v16_mini"

    def _resume_pointer(self) -> Path:
        return self._diagnostics_root() / "stage2_resume_pointer.json"

    def _safe_values(self) -> dict[str, Any]:
        safe = bool(getattr(self.args, "safe_mode", False))

        def numeric(name: str, safe_default: float) -> float:
            value = float(getattr(self.args, name, 0) or 0)
            if value > 0:
                return value
            return float(safe_default if safe else 0.0)

        return {
            "safeMode": safe,
            "resume": bool(getattr(self.args, "resume_stage2", False) or safe),
            "checkpointEvery": int(numeric("checkpoint_every", SAFE_PROFILE["checkpointEvery"])),
            "progressEvery": int(numeric("progress_every", SAFE_PROFILE["progressEvery"])),
            "cooldownEvery": int(numeric("cooldown_every", SAFE_PROFILE["cooldownEvery"])),
            "cooldownSeconds": numeric("cooldown_seconds", SAFE_PROFILE["cooldownSeconds"]),
            "maxGpuTemperatureC": numeric(
                "max_gpu_temperature", SAFE_PROFILE["maxGpuTemperatureC"]
            ),
            "temperatureResumeMarginC": numeric(
                "temperature_resume_margin",
                SAFE_PROFILE["temperatureResumeMarginC"],
            ),
            "thermalPollSeconds": numeric(
                "thermal_poll_seconds", SAFE_PROFILE["thermalPollSeconds"]
            ),
            "validationCooldownSeconds": numeric(
                "validation_cooldown_seconds",
                SAFE_PROFILE["validationCooldownSeconds"],
            ),
        }

    def _gpu_telemetry(self) -> dict[str, Any]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return {}
        try:
            device_index = self.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            command = [
                "nvidia-smi",
                "-i",
                str(device_index),
                "--query-gpu=temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=3,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return {}
            parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
            if len(parts) < 6:
                return {}

            def number(value: str) -> float | None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None

            allocated = torch.cuda.memory_allocated(device_index) / (1024.0 * 1024.0)
            reserved = torch.cuda.memory_reserved(device_index) / (1024.0 * 1024.0)
            peak = torch.cuda.max_memory_allocated(device_index) / (1024.0 * 1024.0)
            return {
                "temperatureC": number(parts[0]),
                "powerW": number(parts[1]),
                "powerLimitW": number(parts[2]),
                "utilizationPercent": number(parts[3]),
                "vramUsedMiB": number(parts[4]),
                "vramTotalMiB": number(parts[5]),
                "torchAllocatedMiB": allocated,
                "torchReservedMiB": reserved,
                "torchPeakAllocatedMiB": peak,
            }
        except (OSError, subprocess.SubprocessError, RuntimeError):
            return {}

    @staticmethod
    def _gpu_text(telemetry: dict[str, Any]) -> str:
        if not telemetry:
            return "gpu-telemetry=unavailable"
        pieces: list[str] = []
        temperature = telemetry.get("temperatureC")
        if temperature is not None:
            pieces.append(f"temp={float(temperature):.0f}C")
        power = telemetry.get("powerW")
        power_limit = telemetry.get("powerLimitW")
        if power is not None:
            if power_limit is not None:
                pieces.append(f"power={float(power):.0f}/{float(power_limit):.0f}W")
            else:
                pieces.append(f"power={float(power):.0f}W")
        utilization = telemetry.get("utilizationPercent")
        if utilization is not None:
            pieces.append(f"gpu={float(utilization):.0f}%")
        used = telemetry.get("vramUsedMiB")
        total = telemetry.get("vramTotalMiB")
        if used is not None and total is not None:
            pieces.append(f"vram={float(used)/1024.0:.1f}/{float(total)/1024.0:.1f}GiB")
        return " ".join(pieces) if pieces else "gpu-telemetry=unavailable"

    def _thermal_guard(
        self,
        safe: dict[str, Any],
        run_dir: Path,
        step: int,
        max_steps: int,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        telemetry = self._gpu_telemetry()
        threshold = float(safe["maxGpuTemperatureC"])
        if threshold <= 0 or not telemetry:
            return telemetry
        temperature = telemetry.get("temperatureC")
        if temperature is None or float(temperature) < threshold:
            return telemetry

        resume_below = max(
            0.0,
            threshold - float(safe["temperatureResumeMarginC"]),
        )
        print(
            f"[v16.0-stage2-safe] thermal pause at step {step}: "
            f"GPU {float(temperature):.0f}C >= {threshold:.0f}C; "
            f"resume below {resume_below:.0f}C",
            flush=True,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()

        while True:
            _atomic_json(
                run_dir / "stage2_heartbeat.json",
                {
                    "schema": "NSAMDR_V16_STAGE2_HEARTBEAT_V1",
                    "status": "thermal-pause",
                    "step": int(step),
                    "maximumSteps": int(max_steps),
                    "elapsedSeconds": float(elapsed_seconds),
                    "gpu": telemetry,
                    "updatedUnix": time.time(),
                },
            )
            time.sleep(max(1.0, float(safe["thermalPollSeconds"])))
            telemetry = self._gpu_telemetry()
            temperature = telemetry.get("temperatureC") if telemetry else None
            if temperature is None or float(temperature) <= resume_below:
                print(
                    f"[v16.0-stage2-safe] thermal pause ended at step {step}: "
                    + self._gpu_text(telemetry),
                    flush=True,
                )
                return telemetry

    def _write_resume_checkpoint(
        self,
        path: Path,
        *,
        model: Any,
        optimizer: torch.optim.Optimizer,
        config: Any,
        run_dir: Path,
        manifest: dict[str, Any],
        train_records: list[dict[str, Any]],
        validation_records: list[dict[str, Any]],
        step: int,
        max_steps: int,
        region_visits: list[int],
        curve: list[dict[str, object]],
        best_key: tuple[int, float],
        best_step: int,
        best_train_report: dict[str, object] | None,
        best_validation_report: dict[str, object] | None,
        interval_loss: float,
        interval_count: int,
        elapsed_seconds: float,
        safe: dict[str, Any],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload: dict[str, Any] = {
            "schema": RESUME_SCHEMA,
            "modelSchema": base.MODEL_SCHEMA,
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "trainingState": {
                "runDir": str(run_dir.resolve()),
                "step": int(step),
                "maximumSteps": int(max_steps),
                "learningRate": float(self.args.learning_rate),
                "ampPrecision": str(self.args.amp_precision),
                "datasetFingerprint": manifest.get("fingerprint"),
                "trainRecords": _record_paths(train_records),
                "validationRecords": _record_paths(validation_records),
                "trainRegionVisits": list(region_visits),
                "curve": list(curve),
                "bestKey": [int(best_key[0]), float(best_key[1])],
                "bestStep": int(best_step),
                "bestTrainReport": best_train_report,
                "bestValidationReport": best_validation_report,
                "intervalLoss": float(interval_loss),
                "intervalCount": int(interval_count),
                "elapsedSeconds": float(elapsed_seconds),
                "safeRuntime": safe,
            },
            "torchRngState": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                payload["cudaRngStateAll"] = torch.cuda.get_rng_state_all()
            except RuntimeError:
                pass
        torch.save(payload, temporary)
        os.replace(temporary, path)
        _atomic_json(
            self._resume_pointer(),
            {
                "schema": "NSAMDR_V16_STAGE2_RESUME_POINTER_V1",
                "checkpoint": str(path.resolve()),
                "runDir": str(run_dir.resolve()),
                "step": int(step),
                "maximumSteps": int(max_steps),
                "datasetFingerprint": manifest.get("fingerprint"),
                "updatedUnix": time.time(),
            },
        )

    def _compatible_resume(
        self,
        *,
        manifest: dict[str, Any],
        train_records: list[dict[str, Any]],
        validation_records: list[dict[str, Any]],
        max_steps: int,
    ) -> tuple[Path, dict[str, Any]] | None:
        pointer = self._resume_pointer()
        if not bool(self._safe_values()["resume"]) or not pointer.is_file():
            return None
        try:
            pointer_data = json.loads(pointer.read_text(encoding="utf-8"))
            checkpoint = Path(str(pointer_data.get("checkpoint") or ""))
            if not checkpoint.is_file():
                return None
            payload = _load_torch_payload(checkpoint)
            state = dict(payload.get("trainingState") or {})
            compatible = (
                payload.get("schema") == RESUME_SCHEMA
                and payload.get("modelSchema") == base.MODEL_SCHEMA
                and state.get("datasetFingerprint") == manifest.get("fingerprint")
                and list(state.get("trainRecords") or []) == _record_paths(train_records)
                and list(state.get("validationRecords") or []) == _record_paths(validation_records)
                and abs(float(state.get("learningRate") or 0.0) - float(self.args.learning_rate)) < 1.0e-12
                and str(state.get("ampPrecision") or "") == str(self.args.amp_precision)
                and 0 < int(state.get("step") or 0) < int(max_steps)
            )
            if not compatible:
                print(
                    "[v16.0-stage2-safe] existing resume checkpoint is incompatible; "
                    "starting a fresh controlled run",
                    flush=True,
                )
                return None
            return checkpoint, payload
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            print(
                f"[v16.0-stage2-safe] resume checkpoint ignored: {exc}",
                flush=True,
            )
            return None

    def _clear_resume_pointer(self, checkpoint: Path | None = None) -> None:
        try:
            self._resume_pointer().unlink(missing_ok=True)
        except OSError:
            pass
        if checkpoint is not None:
            try:
                checkpoint.unlink(missing_ok=True)
            except OSError:
                pass

    def run(self) -> tuple[int, Path]:
        self.args.prepare_train_regions = max(
            int(self.args.prepare_train_regions),
            int(self.args.train_regions),
            base.MIN_TRAIN_REGIONS,
        )
        self.args.prepare_validation_regions = max(
            int(self.args.prepare_validation_regions),
            int(self.args.validation_regions),
            base.MIN_VALIDATION_REGIONS,
        )
        base.prepare_raven_dataset(self.args, self.repo_root)
        config = self._config()
        manifest = base.load_manifest(self.repo_root, config)
        train_records, validation_records = self._records(manifest)

        if (
            len(train_records) < base.MIN_TRAIN_REGIONS
            or len(validation_records) < base.MIN_VALIDATION_REGIONS
        ):
            run_dir = base.make_run_directory(self.repo_root, "multiregion")
            base.write_report(
                run_dir,
                {
                    "schema": base.DIAGNOSTIC_SCHEMA,
                    "revision": base.DIAGNOSTIC_REVISION,
                    "mode": "multiregion",
                    "modelSchema": base.MODEL_SCHEMA,
                    "passed": False,
                    "promotable": False,
                    "reason": "insufficient-spatial-domain-regions",
                    "trainRegions": len(train_records),
                    "validationRegions": len(validation_records),
                    "requiredTrainRegions": base.MIN_TRAIN_REGIONS,
                    "requiredValidationRegions": base.MIN_VALIDATION_REGIONS,
                    "datasetFingerprint": manifest.get("fingerprint"),
                    "datasetSplitPolicy": manifest.get("splitPolicy"),
                },
            )
            return 2, run_dir

        max_steps = max(1, int(self.args.max_steps))
        validate_every = max(1, int(self.args.validate_every))
        safe = self._safe_values()
        resume = self._compatible_resume(
            manifest=manifest,
            train_records=train_records,
            validation_records=validation_records,
            max_steps=max_steps,
        )

        if resume is not None:
            resume_path, resume_payload = resume
            resume_state = dict(resume_payload.get("trainingState") or {})
            run_dir = Path(str(resume_state["runDir"]))
            run_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[v16.0-stage2-safe] RESUME step {int(resume_state['step'])}/{max_steps} "
                f"from {resume_path}",
                flush=True,
            )
        else:
            resume_path = None
            resume_payload = None
            resume_state = {}
            run_dir = base.make_run_directory(self.repo_root, "multiregion")

        progress_path = run_dir / "stage2_progress.jsonl"
        heartbeat_path = run_dir / "stage2_heartbeat.json"
        session_path = run_dir / "stage2_session.json"
        resume_checkpoint = run_dir / "resume_checkpoint.pt"

        if resume_payload is None:
            _atomic_json(
                session_path,
                {
                    "schema": "NSAMDR_V16_STAGE2_SESSION_V1",
                    "runDir": str(run_dir.resolve()),
                    "datasetFingerprint": manifest.get("fingerprint"),
                    "trainRecords": _record_paths(train_records),
                    "validationRecords": _record_paths(validation_records),
                    "maximumSteps": int(max_steps),
                    "validationInterval": int(validate_every),
                    "learningRate": float(self.args.learning_rate),
                    "ampPrecision": str(self.args.amp_precision),
                    "safeRuntime": safe,
                    "createdUnix": time.time(),
                },
            )

        model = base.NSAMDRV16(config).to(self.device)
        model.set_candidate_training()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(
            parameters,
            lr=float(self.args.learning_rate),
            betas=(0.9, 0.999),
            eps=1.0e-8,
        )
        train_datasets = self._balanced_train_datasets(train_records, config, max_steps)

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
        elapsed_before = 0.0
        start_step = 1
        early_pass = False

        if resume_payload is not None:
            model.load_state_dict(resume_payload["state_dict"], strict=True)
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            region_visits = [int(value) for value in resume_state.get("trainRegionVisits", region_visits)]
            curve = list(resume_state.get("curve") or [])
            raw_best_key = list(resume_state.get("bestKey") or [-1, -1.0e9])
            best_key = (int(raw_best_key[0]), float(raw_best_key[1]))
            best_step = int(resume_state.get("bestStep") or 0)
            best_train_report = resume_state.get("bestTrainReport")
            best_validation_report = resume_state.get("bestValidationReport")
            interval_loss = float(resume_state.get("intervalLoss") or 0.0)
            interval_count = int(resume_state.get("intervalCount") or 0)
            completed_steps = int(resume_state.get("step") or 0)
            elapsed_before = float(resume_state.get("elapsedSeconds") or 0.0)
            start_step = completed_steps + 1
            rng = resume_payload.get("torchRngState")
            if isinstance(rng, torch.Tensor):
                torch.set_rng_state(rng)
            cuda_rng = resume_payload.get("cudaRngStateAll")
            if cuda_rng is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(cuda_rng)
                except RuntimeError:
                    pass
            model.set_candidate_training()

        print("=" * 76, flush=True)
        print("V16.0 MULTI-REGION SR MINI - SAFE FOUR-FAMILY RUNTIME", flush=True)
        print(f"Train/held-out : {len(train_records)} / {len(validation_records)} windows", flush=True)
        print(
            "Split rule     : train and validation pixel domains are disjoint; "
            "windows may overlap only inside one split",
            flush=True,
        )
        print("Sampling       : deterministic balanced round-robin", flush=True)
        print(f"Maximum steps  : {max_steps}", flush=True)
        print(f"Validate every : {validate_every} steps", flush=True)
        print(f"Learning rate  : {float(self.args.learning_rate):.7f}", flush=True)
        print(f"Safe mode      : {'ON' if safe['safeMode'] else 'OFF'}", flush=True)
        print(
            f"Crash recovery : {'ON' if safe['resume'] else 'OFF'}; "
            f"checkpoint every {safe['checkpointEvery'] or 0} steps",
            flush=True,
        )
        if safe["safeMode"]:
            print(
                "GPU pacing     : "
                f"{safe['cooldownSeconds']:.2f}s every {safe['cooldownEvery']} steps; "
                f"thermal pause >= {safe['maxGpuTemperatureC']:.0f}C",
                flush=True,
            )
        print(f"Progress file  : {progress_path}", flush=True)
        print(f"Heartbeat      : {heartbeat_path}", flush=True)
        print("=" * 76, flush=True)

        session_started = time.monotonic()

        def elapsed() -> float:
            return elapsed_before + (time.monotonic() - session_started)

        def eta(step: int) -> float | None:
            if step <= 0:
                return None
            seconds_per_step = elapsed() / float(step)
            return max(0.0, seconds_per_step * float(max_steps - step))

        def durable_progress(status: str, step: int, loss_value: float | None, grad_value: float | None, gpu: dict[str, Any]) -> None:
            payload = {
                "schema": "NSAMDR_V16_STAGE2_PROGRESS_V1",
                "status": status,
                "step": int(step),
                "maximumSteps": int(max_steps),
                "loss": loss_value,
                "gradientNorm": grad_value,
                "elapsedSeconds": elapsed(),
                "etaSeconds": eta(step),
                "trainRegionVisits": list(region_visits),
                "gpu": gpu,
                "updatedUnix": time.time(),
            }
            _append_jsonl(progress_path, payload)
            _atomic_json(heartbeat_path, payload)

        durable_progress("running", completed_steps, None, None, self._gpu_telemetry())

        try:
            for step in range(start_step, max_steps + 1):
                model.train()
                region_index = base.balanced_region_index(step, len(train_records))
                visit_index = region_visits[region_index]
                region_visits[region_index] += 1
                batch = base.to_device_batch(
                    train_datasets[region_index][visit_index],
                    self.device,
                )

                optimizer.zero_grad(set_to_none=True)
                with base.autocast_context(self.device, self.args.amp_precision):
                    outputs = model(
                        batch["lr_albedo"],
                        batch["lr_normal"],
                        batch["lr_material"],
                    )
                    losses = base.candidate_loss(outputs, batch, config)
                loss = losses["total"]
                if not bool(torch.isfinite(loss).item()):
                    self._write_curve(run_dir, curve)
                    durable_progress("non-finite-loss", step, None, None, self._gpu_telemetry())
                    return 3, run_dir

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not bool(torch.isfinite(grad_norm).item()):
                    self._write_curve(run_dir, curve)
                    durable_progress("non-finite-gradient", step, None, None, self._gpu_telemetry())
                    return 3, run_dir

                optimizer.step()
                loss_value = float(loss.detach().item())
                grad_value = float(grad_norm)
                interval_loss += loss_value
                interval_count += 1
                completed_steps = step

                checkpoint_every = int(safe["checkpointEvery"] or 0)
                if bool(safe["resume"]) and checkpoint_every > 0 and step % checkpoint_every == 0:
                    self._write_resume_checkpoint(
                        resume_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        run_dir=run_dir,
                        manifest=manifest,
                        train_records=train_records,
                        validation_records=validation_records,
                        step=step,
                        max_steps=max_steps,
                        region_visits=region_visits,
                        curve=curve,
                        best_key=best_key,
                        best_step=best_step,
                        best_train_report=best_train_report,
                        best_validation_report=best_validation_report,
                        interval_loss=interval_loss,
                        interval_count=interval_count,
                        elapsed_seconds=elapsed(),
                        safe=safe,
                    )

                progress_every = int(safe["progressEvery"] or 0)
                if step == 1 or step % max(1, progress_every or 64) == 0:
                    gpu = self._thermal_guard(safe, run_dir, step, max_steps, elapsed())
                    durable_progress("running", step, loss_value, grad_value, gpu)
                    eta_seconds = eta(step)
                    eta_text = "?" if eta_seconds is None else f"{eta_seconds/60.0:.1f}m"
                    print(
                        f"  step {step:4d}/{max_steps} "
                        f"region={region_index + 1}/{len(train_records)} "
                        f"loss={loss_value:.6f} grad={grad_value:.4f} "
                        f"elapsed={elapsed()/60.0:.1f}m eta={eta_text} "
                        + self._gpu_text(gpu),
                        flush=True,
                    )

                cooldown_every = int(safe["cooldownEvery"] or 0)
                cooldown_seconds = float(safe["cooldownSeconds"] or 0.0)
                if (
                    self.device.type == "cuda"
                    and cooldown_every > 0
                    and cooldown_seconds > 0
                    and step % cooldown_every == 0
                ):
                    torch.cuda.synchronize(self.device)
                    time.sleep(cooldown_seconds)

                if step % validate_every != 0 and step != max_steps:
                    continue

                # Save immediately before the evaluation burst.  A driver/PC failure
                # during validation can therefore resume from this exact optimizer step.
                if bool(safe["resume"]):
                    self._write_resume_checkpoint(
                        resume_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        run_dir=run_dir,
                        manifest=manifest,
                        train_records=train_records,
                        validation_records=validation_records,
                        step=step,
                        max_steps=max_steps,
                        region_visits=region_visits,
                        curve=curve,
                        best_key=best_key,
                        best_step=best_step,
                        best_train_report=best_train_report,
                        best_validation_report=best_validation_report,
                        interval_loss=interval_loss,
                        interval_count=interval_count,
                        elapsed_seconds=elapsed(),
                        safe=safe,
                    )

                train_report = self._evaluate_records(model, train_records, config)
                validation_report = self._evaluate_records(model, validation_records, config)
                diagnosis = self._diagnosis(train_report, validation_report)
                generalisation_pass = diagnosis == "passed"
                curve_item: dict[str, object] = {
                    "step": step,
                    "meanTrainingLossSincePreviousEvaluation": interval_loss / max(1, interval_count),
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
                    base.save_checkpoint(
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

                durable_progress("validated", step, loss_value, grad_value, self._gpu_telemetry())

                if bool(safe["resume"]):
                    self._write_resume_checkpoint(
                        resume_checkpoint,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        run_dir=run_dir,
                        manifest=manifest,
                        train_records=train_records,
                        validation_records=validation_records,
                        step=step,
                        max_steps=max_steps,
                        region_visits=region_visits,
                        curve=curve,
                        best_key=best_key,
                        best_step=best_step,
                        best_train_report=best_train_report,
                        best_validation_report=best_validation_report,
                        interval_loss=interval_loss,
                        interval_count=interval_count,
                        elapsed_seconds=elapsed(),
                        safe=safe,
                    )

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                validation_cooldown = float(safe["validationCooldownSeconds"] or 0.0)
                if validation_cooldown > 0:
                    time.sleep(validation_cooldown)

                if generalisation_pass:
                    early_pass = True
                    print(
                        f"[v16.0-multiregion] early PASS at step {step}; "
                        "training and held-out qualification gates are satisfied",
                        flush=True,
                    )
                    break

        except KeyboardInterrupt:
            if completed_steps > 0 and bool(safe["resume"]):
                self._write_resume_checkpoint(
                    resume_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    run_dir=run_dir,
                    manifest=manifest,
                    train_records=train_records,
                    validation_records=validation_records,
                    step=completed_steps,
                    max_steps=max_steps,
                    region_visits=region_visits,
                    curve=curve,
                    best_key=best_key,
                    best_step=best_step,
                    best_train_report=best_train_report,
                    best_validation_report=best_validation_report,
                    interval_loss=interval_loss,
                    interval_count=interval_count,
                    elapsed_seconds=elapsed(),
                    safe=safe,
                )
            durable_progress("interrupted", completed_steps, None, None, self._gpu_telemetry())
            print(
                f"[v16.0-stage2-safe] interrupted at step {completed_steps}; "
                "resume checkpoint retained",
                flush=True,
            )
            return 130, run_dir

        if best_validation_report is None or best_train_report is None or best_step < 1:
            raise RuntimeError("V16.0 multi-region diagnostic produced no evaluated checkpoint")

        selected_model, _payload = base.load_checkpoint(best_path, self.device)
        selected_train_report = self._evaluate_records(selected_model, train_records, config)
        selected_validation_report = self._evaluate_records(selected_model, validation_records, config)
        selected_train_report["selectedCheckpoint"] = str(best_path.resolve())
        selected_train_report["selectedStep"] = int(best_step)
        selected_validation_report["selectedCheckpoint"] = str(best_path.resolve())
        selected_validation_report["selectedStep"] = int(best_step)
        selected_diagnosis = self._diagnosis(selected_train_report, selected_validation_report)
        selected_pass = selected_diagnosis == "passed"

        preview_batch = base.dataset_sample(validation_records[0], config, self.device)
        selected_model.eval()
        with torch.no_grad(), base.autocast_context(self.device, self.args.amp_precision):
            preview_outputs = selected_model(
                preview_batch["lr_albedo"],
                preview_batch["lr_normal"],
                preview_batch["lr_material"],
            )
        probe_path = base.save_probe(run_dir, preview_batch, preview_outputs, include_final=False)
        curve_path = self._write_curve(run_dir, curve)
        report = {
            "schema": base.DIAGNOSTIC_SCHEMA,
            "revision": base.DIAGNOSTIC_REVISION,
            "mode": "multiregion",
            "passed": selected_pass,
            "promotable": False,
            "modelSchema": base.MODEL_SCHEMA,
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
            "safeRuntime": {
                **safe,
                "elapsedSeconds": elapsed(),
                "progress": str(progress_path.resolve()),
                "heartbeat": str(heartbeat_path.resolve()),
                "resumed": bool(resume_payload is not None),
            },
            "trainRecords": _record_paths(train_records),
            "validationRecords": _record_paths(validation_records),
            "trainSourceBoxes": [record.get("source_box") for record in train_records],
            "validationSourceBoxes": [record.get("source_box") for record in validation_records],
            "datasetSplitPolicy": manifest.get("splitPolicy"),
            "generalisationCurve": str(curve_path.resolve()),
            "probe": str(probe_path.resolve()),
            "datasetFingerprint": manifest.get("fingerprint"),
        }
        base.write_report(run_dir, report)
        durable_progress("completed", completed_steps, None, None, self._gpu_telemetry())
        self._clear_resume_pointer(resume_checkpoint)
        return (0 if selected_pass else 2), run_dir


def parser() -> argparse.ArgumentParser:
    p = base.parser()
    p.description = "NSAMDR V16.0 safe four-family Stage 2 diagnostic"
    p.add_argument(
        "--safe-mode",
        action="store_true",
        help="Enable conservative CUDA pacing, thermal pauses, durable progress, and crash recovery.",
    )
    p.add_argument(
        "--resume-stage2",
        action="store_true",
        help="Resume a compatible interrupted Stage 2 run from its atomic checkpoint.",
    )
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=0)
    p.add_argument("--cooldown-every", type=int, default=0)
    p.add_argument("--cooldown-seconds", type=float, default=0.0)
    p.add_argument("--max-gpu-temperature", type=float, default=0.0)
    p.add_argument("--temperature-resume-margin", type=float, default=0.0)
    p.add_argument("--thermal-poll-seconds", type=float, default=0.0)
    p.add_argument("--validation-cooldown-seconds", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    diagnostic = SafeFourFamilyDiagnostic(
        args,
        repo_root,
        base.device_from_name(args.device),
    )
    code, run_dir = diagnostic.run()
    archive = base.archive_run(run_dir)
    print(f"[v16.0-multiregion] report      : {run_dir / 'report.json'}", flush=True)
    print(f"[v16.0-multiregion] diagnostics : {archive}", flush=True)
    if code == 130:
        result = "PAUSED/INTERRUPTED - RESUMABLE"
    else:
        result = "PASS" if code == 0 else "FAIL"
    print(f"[v16.0-multiregion] result      : {result}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
