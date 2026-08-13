"""Resumable high-throughput mixed-precision training for NSAMDR V9."""
from __future__ import annotations

from contextlib import nullcontext
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from datetime import datetime
from typing import Any, Iterator

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import V9Config
from .dataset import (
    PhysicalTileDatasetV9, SyntheticGeometryValidationDataset,
    dataset_fingerprint, load_dataset_manifest,
)
from .inference import resolve_device
from .losses import compute_losses
from .model import MODEL_SCHEMA, FidelityResidualNetV9, architecture_summary, model_hash, parameter_count


class NonFiniteTrainingError(RuntimeError):
    pass


def _status(message: str = "") -> None:
    """Emit one training status line immediately, including under GUI pipes."""
    print(message, flush=True)


# These fields only affect throughput. They are intentionally omitted from the
# resume hash so a checkpoint created before the performance patch can continue.
_RUNTIME_ONLY_CONFIG_FIELDS = {
    "performance_profile",
    "data_loader_workers",
    "data_loader_prefetch_factor",
    "data_loader_persistent_workers",
    "cuda_prefetch",
    "reactive_vram_enabled",
    "reactive_vram_target_free_fraction",
    "reactive_vram_pause_free_fraction",
    "reactive_vram_resume_free_fraction",
    "reactive_vram_expand_hysteresis_fraction",
    "reactive_vram_expand_stable_steps",
    "reactive_vram_poll_seconds",
    "reactive_vram_oom_retries",
    "reactive_vram_release_cache",
    "reactive_vram_burst_reserve_fraction",
    "reactive_vram_stability_samples",
    "reactive_vram_stability_interval_seconds",
    "reactive_vram_dynamic_allocator_ceiling",
    "reactive_vram_start_in_offload",
    "reactive_host_pause_free_fraction",
    "reactive_host_resume_free_fraction",
    "channels_last",
    "amp_dtype",
    "fused_optimizer",
    "cudnn_benchmark",
    "allow_tf32",
    "loss_precision",
    "torch_compile_mode",
}


def _config_hash(config: V9Config) -> str:
    payload = config.to_dict()
    for field_name in _RUNTIME_ONLY_CONFIG_FIELDS:
        payload.pop(field_name, None)
    # Preserve resume compatibility with V9.2.x states created before the
    # optimizer/scheduler selectors existed. Default values are semantically
    # identical to that legacy behavior and are omitted from the hash.
    legacy_optimizer_defaults = {
        "optimizer_name": "adamw",
        "optimizer_beta1": 0.90,
        "optimizer_beta2": 0.99,
        "scheduler_name": "phase",
        "scheduler_min_lr_ratio": 0.25,
    }
    if all(payload.get(key) == value for key, value in legacy_optimizer_defaults.items()):
        for key in legacy_optimizer_defaults:
            payload.pop(key, None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _phase_for_epoch(epoch: int, config: V9Config) -> str:
    boundary = config.identity_epochs
    if epoch <= boundary:
        return "sdf-bootstrap"
    boundary += config.residual_epochs
    if epoch <= boundary:
        return "sdf-proof"
    boundary += config.boundary_epochs
    if epoch <= boundary:
        return "gate-proof"
    boundary += config.detail_epochs
    if epoch <= boundary:
        return "boundary-hardening"
    return "physical-finetune"


def _phase_lr(phase: str, config: V9Config, epoch: int | None = None) -> float:
    base = {
        "sdf-bootstrap": config.identity_learning_rate,
        "sdf-proof": config.learning_rate,
        "gate-proof": config.boundary_learning_rate,
        "boundary-hardening": config.detail_learning_rate,
        "physical-finetune": config.finetune_learning_rate,
    }[phase]
    if config.scheduler_name != "cosine-phase" or epoch is None:
        return base

    lengths = {
        "sdf-bootstrap": config.identity_epochs,
        "sdf-proof": config.residual_epochs,
        "gate-proof": config.boundary_epochs,
        "boundary-hardening": config.detail_epochs,
        "physical-finetune": config.physical_finetune_epochs,
    }
    starts = {
        "sdf-bootstrap": 1,
        "sdf-proof": 1 + config.identity_epochs,
        "gate-proof": 1 + config.identity_epochs + config.residual_epochs,
        "boundary-hardening": 1 + config.identity_epochs + config.residual_epochs + config.boundary_epochs,
        "physical-finetune": 1 + config.identity_epochs + config.residual_epochs + config.boundary_epochs + config.detail_epochs,
    }
    length = max(1, lengths[phase])
    if length <= 1:
        return base
    position = min(max(epoch - starts[phase], 0), length - 1) / float(length - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * position))
    ratio = config.scheduler_min_lr_ratio + (1.0 - config.scheduler_min_lr_ratio) * cosine
    return base * ratio


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", mode="w", encoding="utf-8", delete=False
    ) as temporary:
        json.dump(payload, temporary, indent=2)
        temporary.write("\n")
        temp_path = Path(temporary.name)
    os.replace(temp_path, path)


def _move_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    *,
    channels_last: bool,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        tensor = value.to(device, non_blocking=device.type == "cuda")
        if channels_last and tensor.ndim == 4 and tensor.is_floating_point():
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        moved[key] = tensor
    return moved


def _finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item())


def _bad_gradient_names(model: torch.nn.Module, limit: int = 24) -> list[str]:
    # Detailed per-parameter scans are diagnostic-only. The hot path uses one
    # aggregate gradient norm instead of synchronizing once per parameter.
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not _finite_tensor(parameter.grad):
            names.append(name)
            if len(names) >= limit:
                break
    return names


def _bad_parameter_names(model: torch.nn.Module, limit: int = 24) -> list[str]:
    names: list[str] = []
    for name, parameter in model.named_parameters():
        if not _finite_tensor(parameter):
            names.append(name)
            if len(names) >= limit:
                break
    return names


def _parameters_are_finite(model: torch.nn.Module) -> bool:
    tensors = [parameter.detach() for parameter in model.parameters()]
    if not tensors:
        return True
    try:
        norms = torch._foreach_norm(tensors)
        finite = torch.isfinite(torch.stack([value.float() for value in norms])).all()
    except (AttributeError, RuntimeError):
        finite = torch.stack([
            torch.isfinite(value).all() for value in tensors
        ]).all()
    return bool(finite.item())


def _abort_nonfinite(
    output_dir: Path,
    *,
    epoch: int,
    phase: str,
    batch_index: int,
    stage: str,
    details: dict[str, Any],
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor] | None = None,
) -> None:
    diagnostics = output_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    stem = f"nonfinite_epoch_{epoch:03d}_{phase}_batch_{batch_index:05d}_{stage}"
    report_path = diagnostics / f"{stem}.json"
    tensor_path = diagnostics / f"{stem}.pt"
    report = {
        "schema": "NSAMDR_V9_NONFINITE_DIAGNOSTIC_V1",
        "epoch": epoch,
        "phase": phase,
        "batch": batch_index,
        "stage": stage,
        "details": details,
        "badGradients": _bad_gradient_names(model),
        "badParameters": _bad_parameter_names(model),
    }
    _atomic_json(report, report_path)
    payload: dict[str, Any] = {"report": report, "state_dict": model.state_dict()}
    if batch is not None:
        payload["batch"] = {key: value.detach().cpu() for key, value in batch.items()}
    _atomic_torch_save(payload, tensor_path)
    raise NonFiniteTrainingError(
        f"Non-finite V9 training state at epoch {epoch}, phase {phase}, batch {batch_index}, "
        f"stage {stage}. Diagnostic report: {report_path}"
    )


def _capture_rng() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _rng_byte_tensor(value: Any) -> torch.Tensor:
    """Return a contiguous CPU ByteTensor accepted by PyTorch RNG APIs.

    Resume states are loaded with map_location=device so an RNG state saved on
    CPU can be remapped to CUDA with the rest of the checkpoint. PyTorch RNG
    restoration APIs require contiguous CPU uint8 tensors.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.array(value, dtype=np.uint8, copy=True)).contiguous()
    return torch.tensor(value, dtype=torch.uint8, device="cpu").contiguous()


def _restore_rng(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])

    numpy_state = payload["numpy"]
    if isinstance(numpy_state, list):
        numpy_state = tuple(numpy_state)
    if (
        isinstance(numpy_state, tuple)
        and len(numpy_state) >= 2
        and not isinstance(numpy_state[1], np.ndarray)
    ):
        numpy_state = (
            numpy_state[0],
            np.asarray(numpy_state[1], dtype=np.uint32),
            *numpy_state[2:],
        )
    np.random.set_state(numpy_state)

    torch.set_rng_state(_rng_byte_tensor(payload["torch"]))

    if torch.cuda.is_available() and "cuda" in payload:
        raw_cuda_states = payload["cuda"]
        if isinstance(raw_cuda_states, torch.Tensor):
            raw_cuda_states = [raw_cuda_states]
        cuda_states = [_rng_byte_tensor(state) for state in raw_cuda_states]
        device_count = torch.cuda.device_count()
        if len(cuda_states) == device_count:
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            for device_index, state in enumerate(cuda_states[:device_count]):
                torch.cuda.set_rng_state(state, device=device_index)


def _save_contact_sheet(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    path: Path,
) -> None:
    from PIL import Image, ImageDraw

    def rgb(tensor: torch.Tensor) -> np.ndarray:
        value = tensor[0].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
        return np.uint8(np.round(value * 255.0))

    target = rgb(batch["target_albedo"])
    baseline = rgb(outputs["baseline_albedo"])
    reconstructed = rgb(outputs["albedo"])
    edge = outputs["edge_logits"][0, 0].detach().float().cpu().sigmoid().numpy()
    confidence = outputs["confidence"][0, 0].detach().float().cpu().numpy()
    sdf = outputs["sdf"][0, 0].detach().float().cpu().numpy()
    diagnostics = [
        np.repeat(np.uint8(np.clip(edge, 0.0, 1.0)[..., None] * 255.0), 3, axis=2),
        np.repeat(np.uint8(np.clip(confidence, 0.0, 1.0)[..., None] * 255.0), 3, axis=2),
        np.stack((
            np.uint8(np.clip(sdf * 0.5 + 0.5, 0.0, 1.0) * 255.0),
            np.uint8(np.clip(-sdf * 0.5 + 0.5, 0.0, 1.0) * 255.0),
            np.zeros_like(np.uint8(sdf)),
        ), axis=2),
    ]
    images = [baseline, reconstructed, target, *diagnostics]
    labels = ["bicubic baseline", "V9 fidelity", "authored target", "edge", "confidence", "contour SDF"]
    height, width = images[0].shape[:2]
    canvas = Image.new("RGB", (width * 3, height * 2 + 28 * 2))
    draw = ImageDraw.Draw(canvas)
    for index, (array, label) in enumerate(zip(images, labels)):
        row, column = divmod(index, 3)
        x, y = column * width, row * (height + 28)
        canvas.paste(Image.fromarray(array, mode="RGB"), (x, y + 28))
        draw.text((x + 6, y + 6), label, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


class _MetricAccumulator:
    """Accumulate scalar losses on-device and synchronize only when reported."""

    def __init__(self) -> None:
        self.names: tuple[str, ...] = ()
        self.values: torch.Tensor | None = None
        self.count = 0

    def add(self, losses: dict[str, torch.Tensor]) -> None:
        if not self.names:
            self.names = tuple(losses.keys())
            self.values = torch.zeros(
                len(self.names), device=losses[self.names[0]].device, dtype=torch.float32
            )
        assert self.values is not None
        self.values.add_(torch.stack([losses[name].detach().float() for name in self.names]))
        self.count += 1

    def average_value(self, name: str) -> float:
        if self.values is None or self.count <= 0:
            return 0.0
        index = self.names.index(name)
        return float((self.values[index] / self.count).item())

    def averages(self) -> dict[str, float]:
        if self.values is None or self.count <= 0:
            return {}
        values = (self.values / self.count).detach().cpu().tolist()
        return dict(zip(self.names, (float(value) for value in values)))


def _data_worker_init(worker_id: int) -> None:
    # OpenCV otherwise creates its own thread team inside every DataLoader
    # process, oversubscribing the CPU and starving the CUDA submission thread.
    del worker_id
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    np.random.seed(seed)


def _build_loader(
    dataset: PhysicalTileDatasetV9,
    *,
    batch_size: int,
    device: torch.device,
    workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if workers > 0:
        kwargs.update({
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "worker_init_fn": _data_worker_init,
            # Never let a dead Windows worker make the GUI look frozen forever.
            # Normal V9 batches arrive far faster than this; timeout converts a
            # genuine worker startup/decode deadlock into an actionable error.
            "timeout": 120.0,
        })
    return DataLoader(**kwargs)


class _CudaPrefetcher(Iterator[dict[str, torch.Tensor]]):
    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        *,
        channels_last: bool,
    ) -> None:
        self.iterator = iter(loader)
        self.device = device
        self.channels_last = channels_last
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch: dict[str, torch.Tensor] | None = None
        self._preload()

    def _preload(self) -> None:
        try:
            raw_batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = _move_batch(
                raw_batch, self.device, channels_last=self.channels_last
            )

    def __iter__(self) -> "_CudaPrefetcher":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        if self.next_batch is None:
            raise StopIteration
        current = torch.cuda.current_stream(self.device)
        current.wait_stream(self.stream)
        batch = self.next_batch
        for tensor in batch.values():
            tensor.record_stream(current)
        self._preload()
        return batch


def _iter_batches(
    loader: DataLoader,
    device: torch.device,
    *,
    channels_last: bool,
    cuda_prefetch: bool,
) -> Iterator[dict[str, torch.Tensor]]:
    if device.type == "cuda" and cuda_prefetch:
        return _CudaPrefetcher(loader, device, channels_last=channels_last)
    return iter(
        _move_batch(raw_batch, device, channels_last=channels_last)
        for raw_batch in loader
    )



def _host_memory_info() -> tuple[int, int] | None:
    """Return (available, total) physical RAM without third-party packages."""
    try:
        if os.name == "nt":
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys), int(status.ullTotalPhys)
            return None

        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        total_pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * available_pages), int(page_size * total_pages)
    except (AttributeError, OSError, ValueError):
        return None


class _ReactiveCudaMemoryGovernor:
    """Reactively share CUDA memory with unrelated foreground applications.

    The learned model and loss remain unchanged. The governor only changes
    where autograd's saved backward tensors live:
      gpu      -> normal GPU-resident saved activations (fast)
      offload  -> saved activations live in system RAM until backward needs them
      yield    -> no batch starts until global free VRAM recovers

    This deliberately operates between complete optimizer steps, so one batch
    is always trained with one coherent memory policy.
    """

    def __init__(self, device: torch.device, config: V9Config) -> None:
        self.device = device
        self.config = config
        self.enabled = bool(device.type == "cuda" and config.reactive_vram_enabled)
        self.mode = (
            "offload"
            if self.enabled and config.reactive_vram_start_in_offload
            else "gpu"
        )
        self.stable_recovery_steps = 0
        self._last_reported_mode = ""
        self.total_bytes = 0
        self.target_free_bytes = 0
        self.pause_free_bytes = 0
        self.resume_free_bytes = 0
        self.expand_hysteresis_bytes = 0
        self.host_total_bytes = 0
        self.host_pause_free_bytes = 0
        self.host_resume_free_bytes = 0
        self.burst_reserve_bytes = 0
        self.dynamic_allocator_fraction = 0.0

        # Conservative initial estimate of how much additional transient VRAM a
        # normal GPU-resident step may need beyond its idle allocations. The
        # estimate is replaced by measurements from successful steps.
        self.estimated_gpu_step_extra = 0
        self.estimated_offload_step_extra = 0

        if self.enabled:
            self.total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
            self.target_free_bytes = int(
                self.total_bytes * float(config.reactive_vram_target_free_fraction)
            )
            self.pause_free_bytes = int(
                self.total_bytes * float(config.reactive_vram_pause_free_fraction)
            )
            self.resume_free_bytes = int(
                self.total_bytes * float(config.reactive_vram_resume_free_fraction)
            )
            self.expand_hysteresis_bytes = int(
                self.total_bytes * float(config.reactive_vram_expand_hysteresis_fraction)
            )
            self.burst_reserve_bytes = int(
                self.total_bytes * float(config.reactive_vram_burst_reserve_fraction)
            )
            host_memory = _host_memory_info()
            if host_memory is not None:
                _host_free, self.host_total_bytes = host_memory
                self.host_pause_free_bytes = int(
                    self.host_total_bytes * float(config.reactive_host_pause_free_fraction)
                )
                self.host_resume_free_bytes = int(
                    self.host_total_bytes * float(config.reactive_host_resume_free_fraction)
                )
            self.estimated_gpu_step_extra = int(self.total_bytes * 0.34)
            self.estimated_offload_step_extra = int(self.total_bytes * 0.18)

    @staticmethod
    def _gib(value: int | float) -> float:
        return float(value) / (1024.0 ** 3)

    def _free_bytes(self) -> int:
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        return int(free_bytes)

    def _our_reserved_bytes(self) -> int:
        return int(torch.cuda.memory_reserved(self.device))

    def _our_allocated_bytes(self) -> int:
        return int(torch.cuda.memory_allocated(self.device))

    def _required_free_bytes(self, mode: str) -> int:
        transient = (
            self.estimated_gpu_step_extra
            if mode == "gpu"
            else self.estimated_offload_step_extra
        )
        # Reserve is for an unrelated foreground application that allocates
        # during our batch after the pre-step check.
        return int(transient + self.burst_reserve_bytes)

    def _apply_dynamic_allocator_ceiling(self, free_bytes: int) -> float:
        """Set a hard PyTorch allocator ceiling from current device pressure.

        Approximate non-V9 usage as total - global_free - our_reserved. The V9
        allocator budget is whatever remains after preserving the foreground
        burst reserve. Lowering the ceiling cannot free live tensors, so callers
        also ensure current allocated bytes already fit the resulting budget.
        """
        if not self.config.reactive_vram_dynamic_allocator_ceiling:
            return 1.0

        our_reserved = self._our_reserved_bytes()
        other_usage = max(0, self.total_bytes - free_bytes - our_reserved)
        safe_budget = max(
            0,
            self.total_bytes - other_usage - self.burst_reserve_bytes,
        )

        # Never set a ceiling below already-live V9 allocations. If live state
        # itself no longer fits the safety envelope the batch gate yields.
        live = self._our_allocated_bytes()
        floor_budget = live + int(self.total_bytes * 0.02)
        safe_budget = max(safe_budget, floor_budget)

        fraction = min(0.95, max(0.05, safe_budget / max(float(self.total_bytes), 1.0)))
        torch.cuda.set_per_process_memory_fraction(fraction, device=self.device)
        self.dynamic_allocator_fraction = fraction
        return fraction

    def _stable_resource_snapshot(self) -> int:
        """Require several safe device-memory samples before launching a batch."""
        samples = int(self.config.reactive_vram_stability_samples)
        interval = float(self.config.reactive_vram_stability_interval_seconds)
        minimum = None
        for index in range(samples):
            current = self._free_bytes()
            minimum = current if minimum is None else min(minimum, current)
            if index + 1 < samples:
                time.sleep(interval)
        return int(minimum if minimum is not None else self._free_bytes())

    def _release_cache(self, *, reason: str, report: bool = False) -> int:
        before = self._free_bytes()
        if self.config.reactive_vram_release_cache:
            torch.cuda.empty_cache()
        after = self._free_bytes()
        if report and after > before:
            _status(
                f"  [VRAM] released CUDA cache ({reason}): "
                f"{self._gib(before):.2f} -> {self._gib(after):.2f} GiB free"
            )
        return after

    def _set_mode(self, mode: str, *, free_bytes: int, reason: str) -> None:
        if mode == self.mode and self._last_reported_mode == mode:
            return
        previous = self.mode
        self.mode = mode
        self._last_reported_mode = mode
        _status(
            f"  [VRAM] mode {previous} -> {mode}; "
            f"free={self._gib(free_bytes):.2f} GiB; {reason}"
        )

    def _host_free_bytes(self) -> int | None:
        info = _host_memory_info()
        if info is None:
            return None
        return int(info[0])

    def _host_offload_safe(self, *, resume: bool = False) -> bool:
        if self.host_total_bytes <= 0:
            return True
        free_bytes = self._host_free_bytes()
        if free_bytes is None:
            return True
        floor = self.host_resume_free_bytes if resume else self.host_pause_free_bytes
        return free_bytes >= floor

    def _resource_text(self, gpu_free_bytes: int) -> str:
        host_free = self._host_free_bytes()
        if host_free is None or self.host_total_bytes <= 0:
            return f"GPU free={self._gib(gpu_free_bytes):.2f} GiB"
        return (
            f"GPU free={self._gib(gpu_free_bytes):.2f} GiB, "
            f"host RAM free={self._gib(host_free):.1f} GiB"
        )

    def _wait_for_recovery(
        self,
        free_bytes: int,
        *,
        offload_required: bool = False,
        full_required: int | None = None,
    ) -> int:
        gpu_critical = free_bytes < self.pause_free_bytes
        host_critical = offload_required and not self._host_offload_safe()
        if not gpu_critical and not host_critical:
            return free_bytes

        reason_parts = []
        if gpu_critical:
            reason_parts.append(
                f"GPU free below {self._gib(self.pause_free_bytes):.2f} GiB pause floor"
            )
        if host_critical:
            reason_parts.append(
                "host RAM too pressured for safe activation offload"
            )
        self._set_mode(
            "yield",
            free_bytes=free_bytes,
            reason="; ".join(reason_parts),
        )
        _status(
            "  [VRAM] yielding between batches; "
            "waiting for either enough GPU VRAM for normal mode or enough "
            "GPU + host RAM for activation-offload mode"
        )

        while True:
            if self.config.reactive_vram_release_cache:
                torch.cuda.empty_cache()
            time.sleep(float(self.config.reactive_vram_poll_seconds))
            free_bytes = self._free_bytes()

            enough_for_gpu = (
                full_required is not None and free_bytes >= full_required
            )
            enough_for_offload = (
                free_bytes >= self.resume_free_bytes
                and self._host_offload_safe(resume=True)
            )
            if enough_for_gpu or enough_for_offload:
                break

        self.stable_recovery_steps = 0
        self._last_reported_mode = ""
        if full_required is not None and free_bytes >= full_required:
            self.mode = "gpu"
            self._set_mode(
                "gpu",
                free_bytes=free_bytes,
                reason="resources recovered enough for GPU-resident activations",
            )
        else:
            self.mode = "offload"
            self._set_mode(
                "offload",
                free_bytes=free_bytes,
                reason=(
                    "resources recovered enough for low-VRAM activation-offload mode; "
                    + self._resource_text(free_bytes)
                ),
            )
        return free_bytes

    def before_step(self) -> tuple[str, int, int]:
        if not self.enabled:
            allocated = (
                int(torch.cuda.memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            )
            return "gpu", allocated, 0

        # Return allocator cache first, then require multiple device-wide
        # snapshots. This is deliberately conservative for a workstation where
        # another GPU application may be changing its working set.
        free_bytes = self._release_cache(
            reason="pre-step cooperative safety gate",
            report=False,
        )
        free_bytes = min(free_bytes, self._stable_resource_snapshot())

        while True:
            gpu_required = self._required_free_bytes("gpu")
            offload_required = self._required_free_bytes("offload")

            gpu_safe = free_bytes >= gpu_required
            offload_safe = (
                free_bytes >= offload_required
                and self._host_offload_safe(resume=False)
            )

            # Keep a dynamically shrinking/expanding hard ceiling beneath
            # PyTorch's allocator. It is a backstop, not the primary policy.
            self._apply_dynamic_allocator_ceiling(free_bytes)

            # If current live V9 allocations alone have consumed the foreground
            # reserve, do not attempt another batch.
            live = self._our_allocated_bytes()
            other_usage = max(
                0,
                self.total_bytes - free_bytes - self._our_reserved_bytes(),
            )
            live_budget = max(
                0,
                self.total_bytes - other_usage - self.burst_reserve_bytes,
            )
            if live > live_budget:
                gpu_safe = False
                offload_safe = False

            if self.mode == "gpu":
                if not gpu_safe:
                    self.stable_recovery_steps = 0
                    if offload_safe:
                        self._set_mode(
                            "offload",
                            free_bytes=free_bytes,
                            reason=(
                                "strict safety envelope: GPU-resident predicted "
                                f"step + foreground reserve requires "
                                f"{self._gib(gpu_required):.2f} GiB free"
                            ),
                        )
                    else:
                        self.mode = "yield"
            elif self.mode == "offload":
                if not offload_safe:
                    self.mode = "yield"
                elif gpu_safe:
                    self.stable_recovery_steps += 1
                    if self.stable_recovery_steps >= int(
                        self.config.reactive_vram_expand_stable_steps
                    ):
                        self._set_mode(
                            "gpu",
                            free_bytes=free_bytes,
                            reason=(
                                "strict safety envelope remained satisfied for "
                                f"{self.config.reactive_vram_expand_stable_steps} "
                                "steps; expanding to GPU-resident activations"
                            ),
                        )
                        self.stable_recovery_steps = 0
                else:
                    self.stable_recovery_steps = 0
            else:
                # yield
                if offload_safe:
                    self._set_mode(
                        "offload",
                        free_bytes=free_bytes,
                        reason="strict safety envelope recovered for offload mode",
                    )
                elif gpu_safe:
                    self._set_mode(
                        "gpu",
                        free_bytes=free_bytes,
                        reason="strict safety envelope recovered for GPU mode",
                    )

            if self.mode != "yield":
                break

            self._last_reported_mode = ""
            self._set_mode(
                "yield",
                free_bytes=free_bytes,
                reason=(
                    "strict safety envelope not satisfied; refusing to launch "
                    "another batch"
                ),
            )
            _status(
                f"  [VRAM] safety wait: GPU mode needs "
                f"{self._gib(gpu_required):.2f} GiB free; offload mode needs "
                f"{self._gib(offload_required):.2f} GiB free plus host-RAM "
                "headroom. Training is yielding, not paging."
            )

            if self.config.reactive_vram_release_cache:
                torch.cuda.empty_cache()
            time.sleep(float(self.config.reactive_vram_poll_seconds))
            free_bytes = self._stable_resource_snapshot()

        # Recalculate the allocator ceiling immediately before the forward.
        self._apply_dynamic_allocator_ceiling(free_bytes)
        allocated_before = int(torch.cuda.memory_allocated(self.device))
        torch.cuda.reset_peak_memory_stats(self.device)
        return self.mode, allocated_before, free_bytes

    def autograd_context(self, mode: str):
        if (
            self.enabled
            and mode == "offload"
            and hasattr(torch.autograd, "graph")
            and hasattr(torch.autograd.graph, "save_on_cpu")
        ):
            # Do not pin: this is intentionally conservative toward overall
            # system memory while unrelated programs are active.
            return torch.autograd.graph.save_on_cpu(pin_memory=False)
        return nullcontext()

    def after_step(self, mode: str, allocated_before: int) -> tuple[int, int]:
        if not self.enabled:
            peak = (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda"
                else 0
            )
            return peak, 0

        peak = int(torch.cuda.max_memory_allocated(self.device))
        extra = max(0, peak - int(allocated_before))

        if mode == "gpu":
            # Safety high-watermark: once a larger transient requirement has
            # been observed, never predict less during this process lifetime.
            self.estimated_gpu_step_extra = max(
                self.estimated_gpu_step_extra,
                int(extra * 1.18),
            )
        elif mode == "offload":
            self.estimated_offload_step_extra = max(
                self.estimated_offload_step_extra,
                int(extra * 1.18),
            )

        free_bytes = self._free_bytes()

        # If the just-completed step left less than the sharing target, release
        # allocator cache immediately and make the next step lower-memory.
        if free_bytes < self.target_free_bytes:
            free_bytes = self._release_cache(
                reason="post-step sharing target",
                report=False,
            )
            if mode == "gpu":
                self.stable_recovery_steps = 0
                self._set_mode(
                    "offload",
                    free_bytes=free_bytes,
                    reason="foreground GPU usage increased during the previous batch",
                )
        elif mode == "offload" and self.config.reactive_vram_release_cache:
            # Offload mode is explicitly the cooperative mode: do not hoard
            # cached CUDA blocks between batches.
            torch.cuda.empty_cache()
            free_bytes = self._free_bytes()

        return peak, free_bytes

    def handle_oom(self, *, batch_index: int, attempt: int) -> None:
        if not self.enabled:
            return
        torch.cuda.empty_cache()
        free_bytes = self._free_bytes()
        self.mode = "offload"
        self.stable_recovery_steps = 0
        self._last_reported_mode = ""
        _status(
            f"  [VRAM] CUDA OOM intercepted at batch {batch_index}; "
            f"retry={attempt}; free after cache release={self._gib(free_bytes):.2f} GiB"
        )
        self._set_mode(
            "yield",
            free_bytes=free_bytes,
            reason=(
                "OOM recovery: forcing the next retry through the strict "
                "multi-sample safety gate"
            ),
        )

    def status_text(self) -> str:
        if not self.enabled:
            return "fixed"
        free_bytes = self._free_bytes()
        cap = (
            f" cap={self.dynamic_allocator_fraction * 100.0:.0f}%"
            if self.dynamic_allocator_fraction > 0.0
            else ""
        )
        return (
            f"{self.mode} free={self._gib(free_bytes):.2f}GiB"
            f" reserve={self._gib(self.burst_reserve_bytes):.2f}GiB{cap}"
        )


def _is_cuda_oom(error: BaseException) -> bool:
    if isinstance(error, getattr(torch, "OutOfMemoryError", RuntimeError)):
        return "out of memory" in str(error).lower() or error.__class__.__name__ == "OutOfMemoryError"
    return "out of memory" in str(error).lower() and "cuda" in str(error).lower()


def _resolve_amp_dtype(config: V9Config, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if config.amp_dtype == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("ampDtype=bf16 requested but this CUDA device does not support BF16")
        return torch.bfloat16
    if config.amp_dtype == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _configure_cuda(config: V9Config, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
    except AttributeError:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)
    except AttributeError:
        pass


def _convert_model_channels_last(model: torch.nn.Module) -> torch.nn.Module:
    converter = getattr(torch.nn.utils, "convert_conv2d_weight_memory_format", None)
    if converter is not None:
        return converter(model, torch.channels_last)
    return model.to(memory_format=torch.channels_last)


def _build_optimizer(
    model: torch.nn.Module,
    config: V9Config,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, str]:
    common = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
        "betas": (config.optimizer_beta1, config.optimizer_beta2),
    }
    optimizer_class = torch.optim.AdamW if config.optimizer_name == "adamw" else torch.optim.Adam
    display = "AdamW" if config.optimizer_name == "adamw" else "Adam"
    if device.type == "cuda" and config.fused_optimizer:
        try:
            return optimizer_class(model.parameters(), fused=True, **common), f"fused {display}"
        except (TypeError, RuntimeError):
            pass
    if device.type == "cuda":
        try:
            return optimizer_class(model.parameters(), foreach=True, **common), f"foreach {display}"
        except TypeError:
            pass
    return optimizer_class(model.parameters(), **common), f"standard {display}"


def _restore_optimizer_implementation(
    optimizer: torch.optim.Optimizer,
    optimizer_mode: str,
) -> None:
    """Keep the selected kernel after loading an older optimizer state.

    Optimizer.load_state_dict restores old param-group flags as well as moment
    tensors. A state produced by standard AdamW contains fused=None, which
    would silently disable the new fused kernel unless it is reapplied.
    """
    if optimizer_mode.startswith("fused "):
        optimizer.defaults["fused"] = True
        optimizer.defaults["foreach"] = None
        for group in optimizer.param_groups:
            group["fused"] = True
            group["foreach"] = None
    elif optimizer_mode.startswith("foreach "):
        optimizer.defaults["fused"] = None
        optimizer.defaults["foreach"] = True
        for group in optimizer.param_groups:
            group["fused"] = None
            group["foreach"] = True


def _parameter_memory_format(parameter: torch.Tensor) -> torch.memory_format:
    if parameter.ndim == 4 and parameter.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    if parameter.ndim == 5 and parameter.is_contiguous(memory_format=torch.channels_last_3d):
        return torch.channels_last_3d
    return torch.contiguous_format


def _align_optimizer_state_to_parameters(
    optimizer: torch.optim.Optimizer,
) -> int:
    """Match Adam state tensors to the selected optimizer implementation.

    Parameter-shaped moment buffers must share each parameter's device, dtype
    and memory layout. In addition, fused/capturable AdamW requires its scalar
    ``step`` tensors on the same device as the corresponding parameters.

    This matters when a checkpoint is loaded on CPU for benchmarking: the
    legacy optimizer state is restored before the fused flag is reapplied, so
    PyTorch leaves ``step`` on CPU unless it is migrated explicitly.
    """
    migrated = 0
    moment_names = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")

    for group in optimizer.param_groups:
        step_on_parameter_device = bool(group.get("fused")) or bool(
            group.get("capturable")
        )
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state:
                continue

            if step_on_parameter_device and "step" in state:
                step = state["step"]
                if isinstance(step, torch.Tensor):
                    aligned_step = step.detach().to(
                        device=parameter.device,
                        dtype=torch.float32,
                        non_blocking=False,
                    ).contiguous()
                else:
                    aligned_step = torch.tensor(
                        float(step),
                        device=parameter.device,
                        dtype=torch.float32,
                    )
                if (
                    not isinstance(step, torch.Tensor)
                    or step.device != aligned_step.device
                    or step.dtype != aligned_step.dtype
                    or not step.is_contiguous()
                ):
                    state["step"] = aligned_step
                    migrated += 1
                elif state["step"] is not aligned_step:
                    state["step"] = aligned_step

            memory_format = _parameter_memory_format(parameter)
            for name in moment_names:
                value = state.get(name)
                if not isinstance(value, torch.Tensor):
                    continue

                aligned = value.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                    non_blocking=False,
                )
                if aligned.shape == parameter.shape:
                    aligned = aligned.contiguous(memory_format=memory_format)
                else:
                    aligned = aligned.contiguous()

                layout_matches = (
                    aligned.stride() == parameter.stride()
                    if aligned.shape == parameter.shape
                    else True
                )
                if (
                    value.device != aligned.device
                    or value.dtype != aligned.dtype
                    or value.stride() != aligned.stride()
                    or not layout_matches
                ):
                    state[name] = aligned
                    migrated += 1
                elif state[name] is not aligned:
                    state[name] = aligned

    return migrated


def _is_fused_layout_error(error: RuntimeError) -> bool:
    text = str(error).lower()
    return (
        "params, grads, exp_avgs, and exp_avg_sqs" in text
        and "same dtype, device, and layout" in text
    )


def _compile_training_model(
    model: FidelityResidualNetV9,
    mode: str,
) -> tuple[torch.nn.Module, str]:
    mode = str(mode).strip().lower()
    if mode == "off":
        return model, "off"
    compiler = getattr(torch, "compile", None)
    if compiler is None:
        print("WARNING: torch.compile is unavailable; using eager execution.")
        return model, "unavailable"
    try:
        compile_mode = None if mode == "default" else mode
        if compile_mode is None:
            return compiler(model, fullgraph=False, dynamic=False), "default"
        return compiler(
            model,
            mode=compile_mode,
            fullgraph=False,
            dynamic=False,
        ), mode
    except Exception as error:
        print(f"WARNING: torch.compile setup failed ({error!r}); using eager execution.")
        return model, "failed->off"


def _teacher_boundary_gate(
    batch: dict[str, torch.Tensor],
    config: V9Config,
) -> torch.Tensor:
    """Hard replacement authority across the known contaminated contour band.

    Stage A/B ask whether the renderer and predicted SDF can reconstruct a
    boundary when placement authority is known. A soft edge-derived gate kept a
    fraction of the degraded transition in EXP_0003, so the teacher is now 1.0
    across the reconstruction band with only a short outer falloff.
    """
    target_sdf_pixels = (
        batch["target_sdf"].float()
        * float(config.contour_sdf_max_distance_pixels)
    ).abs()
    radius = max(
        7.0,
        min(float(config.boundary_renderer_band_pixels) + 5.0, 9.0),
    )
    falloff = 0.25
    outside = torch.relu(target_sdf_pixels - radius)
    return torch.where(
        target_sdf_pixels <= radius,
        torch.ones_like(target_sdf_pixels),
        torch.exp(-outside / falloff),
    ).clamp(0.0, 1.0)


def _forward_for_phase(
    model: FidelityResidualNetV9,
    batch: dict[str, torch.Tensor],
    phase: str,
    config: V9Config,
) -> dict[str, torch.Tensor]:
    if phase == "sdf-proof":
        return model(
            batch["input"],
            gate_override=_teacher_boundary_gate(batch, config),
            renderer_enabled_override=True,
        )
    return model(batch["input"])


def _validate(
    model: FidelityResidualNetV9,
    loader: DataLoader,
    config: V9Config,
    device: torch.device,
    phase: str,
    *,
    amp_dtype: torch.dtype,
) -> tuple[dict[str, float], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    model.eval()
    totals = _MetricAccumulator()
    last_outputs: dict[str, torch.Tensor] | None = None
    last_batch: dict[str, torch.Tensor] | None = None
    use_amp = device.type == "cuda"
    batches = _iter_batches(
        loader,
        device,
        channels_last=config.channels_last and use_amp,
        cuda_prefetch=config.cuda_prefetch and use_amp,
    )
    with torch.inference_mode():
        for batch_index, batch in enumerate(batches, 1):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = _forward_for_phase(model, batch, phase, config)
            with torch.autocast(device_type=device.type, enabled=False):
                losses = compute_losses(outputs, batch, config, phase)
            if not _finite_tensor(losses["total"]):
                _abort_nonfinite(
                    Path(config.output_dir), epoch=0, phase=phase, batch_index=batch_index,
                    stage="validation-loss", details={"total": float(losses["total"].detach().cpu())},
                    model=model, batch=batch)
            totals.add(losses)
            last_outputs, last_batch = outputs, batch
    assert last_outputs is not None and last_batch is not None
    return totals.averages(), last_outputs, last_batch


def train_v9(
    config: V9Config,
    repo_root: Path,
    requested_device: str | None = None,
    *,
    resume: bool = False,
    restart: bool = False,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
    stop_after_phase: str | None = None,
) -> dict[str, Any]:
    config.validate()
    _status("[startup] Trainer entered; configuration validated.")
    output_dir = (repo_root / config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / config.checkpoint_name
    metadata_path = output_dir / config.metadata_name
    state_path = output_dir / config.training_state_name
    best_path = output_dir / "nsamdr_v9_best_final_phase.pt"
    if restart:
        _status(f"[startup] Restart requested; preparing prior-state backup in {output_dir}...")
        cleanup_started = time.perf_counter()

        restart_sources = [
            path for path in (checkpoint_path, metadata_path, state_path, best_path)
            if path.is_file()
        ]
        if restart_sources:
            backup_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = output_dir / "restart_backups" / backup_stamp
            backup_dir.mkdir(parents=True, exist_ok=False)
            for path in restart_sources:
                shutil.copy2(path, backup_dir / path.name)
            _status(
                f"[startup] Restart backup created: {backup_dir} "
                f"({len(restart_sources)} file(s))"
            )

        _status(f"[startup] Restart confirmed; clearing active prior state in {output_dir}...")
        for path in (checkpoint_path, metadata_path, state_path, best_path):
            path.unlink(missing_ok=True)
        # A semantic/model restart must not leave validation sheets or forensic
        # dumps from an incompatible earlier schema mixed into the new run.
        shutil.rmtree(output_dir / "samples", ignore_errors=True)
        shutil.rmtree(output_dir / config.diagnostics_dir_name, ignore_errors=True)
        _status(
            f"[startup] Restart cleanup complete in "
            f"{time.perf_counter() - cleanup_started:.1f}s."
        )

    _status("[startup] Resolving CUDA device and runtime settings...")
    device = resolve_device(config, requested_device)
    _configure_cuda(config, device)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    _status("[startup] Loading V9 dataset manifest...")
    manifest = load_dataset_manifest(repo_root, config)
    fingerprint = dataset_fingerprint(manifest, config)
    _status("[startup] Dataset manifest loaded; constructing training/validation datasets...")
    train_dataset = PhysicalTileDatasetV9(
        manifest, config, "train", config.tiles_per_epoch, seed=config.seed)
    validation_dataset = PhysicalTileDatasetV9(
        manifest, config, "validation", config.validation_tiles, seed=config.seed + 77)
    synthetic_validation_dataset = SyntheticGeometryValidationDataset(
        config, config.sdf_synthetic_validation_tiles, seed=config.seed + 9_911
    )

    workers = config.data_loader_workers if device.type == "cuda" else 0
    validation_workers = min(workers, 1)
    _status(
        f"[startup] Constructing DataLoaders: train workers={workers}, "
        f"validation workers={validation_workers}, prefetch={config.data_loader_prefetch_factor}..."
    )
    train_loader = _build_loader(
        train_dataset,
        batch_size=config.batch_size,
        device=device,
        workers=workers,
        prefetch_factor=config.data_loader_prefetch_factor,
        persistent_workers=config.data_loader_persistent_workers,
    )
    validation_loader = _build_loader(
        validation_dataset,
        batch_size=1,
        device=device,
        workers=validation_workers,
        prefetch_factor=config.data_loader_prefetch_factor,
        persistent_workers=config.data_loader_persistent_workers,
    )
    synthetic_validation_loader = _build_loader(
        synthetic_validation_dataset,
        batch_size=1,
        device=device,
        workers=0,
        prefetch_factor=config.data_loader_prefetch_factor,
        persistent_workers=False,
    )

    _status(
        f"[startup] DataLoaders constructed; fixed synthetic SDF validation="
        f"{config.sdf_synthetic_validation_tiles} tiles."
    )

    _status("[startup] Allocating V9.8 geometry-convergence BoundaryRenderer model on training device...")
    model_started = time.perf_counter()
    model = FidelityResidualNetV9(config).to(device)
    contract = model.architecture_contract()
    if (
        contract.get("schema") != MODEL_SCHEMA
        or contract.get("geometryModel") != "GeometryNet"
        or contract.get("renderer") != "BoundaryRenderer"
        or bool(contract.get("geometryCanPaintRgb"))
        or tuple(contract.get("geometryOutputs", ())) != (
            "sdf", "edge", "orientation", "hardness", "boundary_gate"
        )
        or not bool(contract.get("sharedAcrossPhysicalMaps"))
    ):
        raise RuntimeError(f"V9.8 architecture contract failed: {contract!r}")
    _status(
        f"[startup] Model allocated in {time.perf_counter() - model_started:.1f}s; "
        f"{parameter_count(model):,} parameters."
    )
    if device.type == "cuda" and config.channels_last:
        model = _convert_model_channels_last(model)
    _status("[startup] Building optimizer and AMP state...")
    optimizer_started = time.perf_counter()
    optimizer, optimizer_mode = _build_optimizer(model, config, device)
    amp_dtype = _resolve_amp_dtype(config, device)
    use_amp = device.type == "cuda"
    use_scaler = use_amp and amp_dtype == torch.float16
    memory_governor = _ReactiveCudaMemoryGovernor(device, config)
    effective_cuda_prefetch = bool(
        config.cuda_prefetch and use_amp and not memory_governor.enabled
    )
    try:
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_scaler, init_scale=config.amp_initial_scale
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_scaler, init_scale=config.amp_initial_scale
        )

    _status(
        f"[startup] Optimizer/AMP state ready in "
        f"{time.perf_counter() - optimizer_started:.1f}s."
    )
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    best_sdf_score = float("inf")
    optimizer_state_migrations = 0
    if resume:
        if not state_path.is_file():
            raise RuntimeError(f"V9 resume requested but training state is missing: {state_path}")
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("schema") != MODEL_SCHEMA:
            raise RuntimeError("V9 resume state schema mismatch")
        if state.get("config_hash") != _config_hash(config) or state.get("dataset_fingerprint") != fingerprint:
            raise RuntimeError("V9 resume state does not match the current semantic config or dataset")
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        _restore_optimizer_implementation(optimizer, optimizer_mode)
        optimizer_state_migrations = _align_optimizer_state_to_parameters(optimizer)
        scaler.load_state_dict(state.get("scaler", {}))
        _restore_rng(state["rng"])
        history = list(state.get("history", []))
        best_validation = float(state.get("best_validation", float("inf")))
        best_sdf_score = float(state.get("best_sdf_score", float("inf")))
        start_epoch = int(state["completed_epoch"]) + 1

    _status("[startup] Preparing execution model...")
    execution_model, compile_status = _compile_training_model(
        model, config.torch_compile_mode if device.type == "cuda" else "off"
    )
    _status("[startup] Initialization complete; entering epoch loop.")

    precision_name = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }[amp_dtype]
    _status("=" * 68)
    _status("NSAMDR V9 PHYSICAL RECONSTRUCTION")
    _status("=" * 68)
    _status(f"Device                   : {device}")
    _status(f"Architecture schema      : {MODEL_SCHEMA}")
    _status("Model                    : implicit-SDF GeometryNet -> staged forced-gate proof -> shared BoundaryRenderer")
    _status(f"Tile geometry            : {config.tile_size} LR -> {config.tile_size * 4} HR")
    _status(f"Widths                   : {list(config.widths)}")
    _status(f"Parameters               : {parameter_count(model):,}")
    _status("Geometry outputs         : coordinate-conditioned SDF + edge + orientation + hardness")
    _status("Geometry RGB authority   : NONE")
    _status(f"Appearance enabled       : {bool(config.appearance_enabled)}")
    _status("Boundary primitive       : staged oracle/predicted SDF + two-sided sub-pixel coverage")
    _status("Boundary sampling        : best-of-%d at %.0f%% probability" % (
        config.boundary_candidate_count, config.boundary_sampling_probability * 100.0))
    _status(f"Exact geometry training   : {config.synthetic_geometry_probability * 100.0:.0f}% of training tiles")
    _status("Contour objective         : multi-scale tangent + curvature coherence")
    if config.loss_precision == "mixed" and use_amp:
        _status("Loss numerics            : reduced-precision kernels, FP32 scalar reductions")
    else:
        _status("Loss numerics            : complete objective in FP32")
    _status(f"Training state           : {state_path}")
    _status(f"Performance profile      : {config.performance_profile}")
    _status(f"AMP compute precision    : {precision_name}")
    _status(f"Convolution layout       : {'channels-last' if config.channels_last and use_amp else 'contiguous'}")
    _status(f"Optimizer implementation : {optimizer_mode}")
    _status(f"Learning-rate scheduler  : {config.scheduler_name}")
    _status(f"torch.compile            : {compile_status}")
    if resume:
        _status(f"Optimizer state migration: {optimizer_state_migrations} moment tensor(s) aligned")
    _status(f"Data pipeline            : workers={workers} prefetch={config.data_loader_prefetch_factor} persistent={workers > 0 and config.data_loader_persistent_workers}")
    _status(f"CUDA batch prefetch      : {effective_cuda_prefetch}")
    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(device).total_memory / (1024.0 ** 3)
        if memory_governor.enabled:
            _status("CUDA memory policy      : elastic/reactive sharing")
            _status(
                f"VRAM target / pause      : "
                f"{config.reactive_vram_target_free_fraction * 100.0:.0f}% / "
                f"{config.reactive_vram_pause_free_fraction * 100.0:.0f}% free"
            )
            _status(
                "VRAM pressure response    : GPU activations -> CPU-saved "
                "activations -> yield between batches"
            )
            _status(
                f"Foreground burst reserve : "
                f"{config.reactive_vram_burst_reserve_fraction * 100.0:.0f}% "
                f"({memory_governor.burst_reserve_bytes / (1024.0 ** 3):.2f} GiB)"
            )
            _status(
                f"Pre-batch stability gate : "
                f"{config.reactive_vram_stability_samples} samples x "
                f"{config.reactive_vram_stability_interval_seconds:.2f}s"
            )
            _status(
                "Allocator ceiling         : reactive per-batch physical-VRAM envelope"
            )
            _status(
                f"VRAM expansion policy    : restore GPU activations after "
                f"{config.reactive_vram_expand_stable_steps} stable recovery steps"
            )
            host_memory = _host_memory_info()
            if host_memory is not None:
                host_free, host_total = host_memory
                _status(
                    f"Host RAM offload guard   : "
                    f"pause below {config.reactive_host_pause_free_fraction * 100.0:.0f}% free; "
                    f"resume above {config.reactive_host_resume_free_fraction * 100.0:.0f}% "
                    f"(now {host_free / (1024.0 ** 3):.1f}/{host_total / (1024.0 ** 3):.1f} GiB free)"
                )
        else:
            _status(f"CUDA memory policy      : non-reactive ({total_vram:.1f} GiB device)")
    _status(f"cuDNN benchmark / TF32   : {config.cudnn_benchmark} / {config.allow_tf32}")
    if resume:
        _status(f"Resume epoch             : {start_epoch:03d}/{config.total_epochs:03d}")
    _status("=" * 68)

    started = time.perf_counter()
    global_step = sum(int(item.get("batches", 0)) for item in history)
    previous_finetune_validation = [
        float(item.get("validation", {}).get("total"))
        for item in history
        if item.get("phase") == "physical-finetune"
        and item.get("validation", {}).get("total") is not None
    ]
    early_stop_best = min(previous_finetune_validation, default=math.inf)
    early_stop_stale = 0
    early_stopped_epoch: int | None = None
    valid_stop_phases = {
        None,
        "sdf-bootstrap",
        "sdf-proof",
        "gate-proof",
        "boundary-hardening",
        "physical-finetune",
    }
    if stop_after_phase not in valid_stop_phases:
        raise ValueError(f"unsupported stop_after_phase={stop_after_phase!r}")
    staged_stop_epoch = None
    if stop_after_phase is not None:
        phase_ends = {
            "sdf-bootstrap": config.identity_epochs,
            "sdf-proof": config.identity_epochs + config.residual_epochs,
            "gate-proof": config.identity_epochs + config.residual_epochs + config.boundary_epochs,
            "boundary-hardening": config.identity_epochs + config.residual_epochs + config.boundary_epochs + config.detail_epochs,
            "physical-finetune": config.total_epochs,
        }
        staged_stop_epoch = int(phase_ends[stop_after_phase])
        _status(
            f"Staged training stop     : after {stop_after_phase} "
            f"(epoch {staged_stop_epoch:03d})"
        )
    if early_stop_patience is not None and early_stop_patience > 0:
        _status(
            f"Tuning early stop        : physical-finetune only; patience={early_stop_patience} "
            f"minDelta={early_stop_min_delta:.6f}"
        )
    for epoch in range(start_epoch, config.total_epochs + 1):
        phase = _phase_for_epoch(epoch, config)
        model.set_phase(phase)
        model.train()
        learning_rate = _phase_lr(phase, config, epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        totals = _MetricAccumulator()
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _status(f"Epoch {epoch:03d}/{config.total_epochs:03d} phase={phase} lr={learning_rate:.7f}")
        _status(
            f"  Starting DataLoader: {workers} worker(s), "
            f"{len(train_loader)} training batch(es). Waiting for first batch..."
        )
        data_start = time.perf_counter()
        train_batches = _iter_batches(
            train_loader,
            device,
            channels_last=config.channels_last and use_amp,
            cuda_prefetch=effective_cuda_prefetch,
        )
        _status(
            f"  First batch prefetched in {time.perf_counter() - data_start:.1f}s; "
            "CUDA training loop active."
        )
        epoch_peak_bytes = 0
        for batch_index, batch in enumerate(train_batches, 1):
            step_completed = False
            losses = None
            step_peak_bytes = 0
            step_free_bytes = 0
            memory_attempt = 0

            while not step_completed:
                memory_mode, allocated_before, free_before = memory_governor.before_step()

                try:
                    for retry in range(config.amp_overflow_retries + 1):
                        optimizer.zero_grad(set_to_none=True)

                        # Saved tensors from the forward graph are either kept on
                        # GPU or moved to system RAM according to current global
                        # VRAM pressure. The numerical model/loss is unchanged.
                        with memory_governor.autograd_context(memory_mode):
                            with torch.autocast(
                                device_type=device.type,
                                dtype=amp_dtype,
                                enabled=use_amp,
                            ):
                                if phase == "sdf-proof":
                                    outputs = execution_model(
                                        batch["input"],
                                        gate_override=_teacher_boundary_gate(batch, config),
                                        renderer_enabled_override=True,
                                    )
                                else:
                                    outputs = execution_model(batch["input"])
                            with torch.autocast(
                                device_type=device.type,
                                enabled=False,
                            ):
                                losses = compute_losses(outputs, batch, config, phase)

                            if not _finite_tensor(losses["total"]):
                                _abort_nonfinite(
                                    output_dir,
                                    epoch=epoch,
                                    phase=phase,
                                    batch_index=batch_index,
                                    stage="loss",
                                    details={
                                        "total": float(
                                            losses["total"].detach().cpu()
                                        )
                                    },
                                    model=model,
                                    batch=batch,
                                )

                            scale_before = (
                                float(scaler.get_scale()) if use_scaler else 1.0
                            )
                            scaler.scale(losses["total"]).backward()

                        scaler.unscale_(optimizer)
                        try:
                            gradient_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                config.gradient_clip_norm,
                                foreach=True,
                            )
                        except TypeError:
                            gradient_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                config.gradient_clip_norm,
                            )

                        gradient_finite = bool(
                            torch.isfinite(gradient_norm.detach()).item()
                        )
                        if not gradient_finite:
                            terminal_overflow = (
                                not use_scaler
                                or retry >= config.amp_overflow_retries
                                or scale_before <= config.amp_minimum_scale
                            )
                            bad_gradient_names = (
                                _bad_gradient_names(model)
                                if terminal_overflow
                                else []
                            )
                            optimizer.zero_grad(set_to_none=True)
                            if terminal_overflow:
                                _abort_nonfinite(
                                    output_dir,
                                    epoch=epoch,
                                    phase=phase,
                                    batch_index=batch_index,
                                    stage="gradient",
                                    details={
                                        "badGradients": bad_gradient_names,
                                        "gradientNorm": float(
                                            gradient_norm.detach().cpu()
                                        ),
                                        "scale": scale_before,
                                        "retry": retry,
                                    },
                                    model=model,
                                    batch=batch,
                                )
                            next_scale = max(
                                config.amp_minimum_scale,
                                scale_before * 0.5,
                            )
                            scaler.update(new_scale=next_scale)
                            _status(
                                f"  AMP overflow retry batch={batch_index} "
                                f"retry={retry + 1} "
                                f"scale={scale_before:.1f}->{next_scale:.1f}"
                            )
                            continue

                        try:
                            scaler.step(optimizer)
                        except RuntimeError as error:
                            if (
                                not optimizer_mode.startswith("fused ")
                                or not _is_fused_layout_error(error)
                            ):
                                raise
                            migrated = _align_optimizer_state_to_parameters(
                                optimizer
                            )
                            if migrated <= 0:
                                raise
                            _status(
                                "  Fused AdamW state-layout recovery "
                                f"batch={batch_index} aligned={migrated}"
                            )
                            scaler.step(optimizer)

                        scaler.update()
                        step_completed = True
                        break

                    if not step_completed:
                        raise RuntimeError(
                            "V9 optimizer step exhausted AMP retries"
                        )

                except RuntimeError as error:
                    if (
                        device.type != "cuda"
                        or not memory_governor.enabled
                        or not _is_cuda_oom(error)
                        or memory_attempt >= config.reactive_vram_oom_retries
                    ):
                        raise

                    optimizer.zero_grad(set_to_none=True)
                    # Break references to the failed graph before releasing the
                    # allocator cache. The current DataLoader batch remains valid
                    # and is retried unchanged.
                    try:
                        del outputs
                    except UnboundLocalError:
                        pass
                    losses = None
                    memory_attempt += 1
                    memory_governor.handle_oom(
                        batch_index=batch_index,
                        attempt=memory_attempt,
                    )
                    continue

            if losses is None:
                raise RuntimeError(
                    "V9 training step completed without a loss payload"
                )

            step_peak_bytes, step_free_bytes = memory_governor.after_step(
                memory_mode,
                allocated_before,
            )
            epoch_peak_bytes = max(epoch_peak_bytes, step_peak_bytes)

            global_step += 1

            if device.type == "cuda" and batch_index == 1:
                total_bytes = torch.cuda.get_device_properties(device).total_memory
                _status(
                    f"  First-step CUDA peak: "
                    f"{step_peak_bytes / (1024.0 ** 3):.2f} GiB "
                    f"({step_peak_bytes / max(float(total_bytes), 1.0) * 100.0:.1f}% "
                    f"of physical VRAM); mode={memory_mode}; "
                    f"free-after={step_free_bytes / (1024.0 ** 3):.2f} GiB"
                )

            if global_step % config.parameter_finite_check_interval == 0:
                if not _parameters_are_finite(model):
                    _abort_nonfinite(
                        output_dir, epoch=epoch, phase=phase, batch_index=batch_index,
                        stage="parameters", details={"badParameters": _bad_parameter_names(model)},
                        model=model, batch=batch)
            totals.add(losses)
            if (
                batch_index == 1
                or batch_index % max(1, len(train_loader) // 8) == 0
                or batch_index == len(train_loader)
            ):
                elapsed = max(1.0e-6, time.perf_counter() - epoch_started)
                tiles_done = min(config.tiles_per_epoch, batch_index * config.batch_size)
                tiles_per_second = tiles_done / elapsed
                step_ms = elapsed * 1000.0 / batch_index
                memory_text = ""
                if device.type == "cuda":
                    peak_gib = epoch_peak_bytes / (1024.0 ** 3)
                    memory_text = (
                        f" peak={peak_gib:.2f}GiB "
                        f"vram={memory_governor.status_text()}"
                    )
                _status(
                    f"  {batch_index:5d}/{len(train_loader):5d} "
                    f"total={totals.average_value('total'):.5f} "
                    f"step={step_ms:.1f}ms rate={tiles_per_second:.2f}tile/s{memory_text}"
                )

        train_metrics = totals.averages()
        if memory_governor.enabled and config.reactive_vram_release_cache:
            torch.cuda.empty_cache()
        validation_metrics, validation_outputs, validation_batch = _validate(
            model, validation_loader, config, device, phase, amp_dtype=amp_dtype)
        synthetic_sdf_metrics: dict[str, float] | None = None
        if phase in {"sdf-bootstrap", "sdf-proof"}:
            synthetic_sdf_metrics, _synthetic_outputs, _synthetic_batch = _validate(
                model, synthetic_validation_loader, config, device, phase, amp_dtype=amp_dtype
            )
        if memory_governor.enabled and config.reactive_vram_release_cache:
            torch.cuda.empty_cache()
        seconds = time.perf_counter() - epoch_started
        tiles_per_second = config.tiles_per_epoch / max(seconds, 1.0e-6)
        print(
            f"  train total={train_metrics['total']:.6f} "
            f"sdf={train_metrics['sdf']:.6f} "
            f"profile={train_metrics.get('boundary_profile', 0.0):.6f} "
            f"sdfSurf={train_metrics.get('sdf_surface', 0.0):.4f} "
            f"coarse={train_metrics.get('coarse_sdf_surface', 0.0):.4f} "
            f"eik={train_metrics.get('sdf_eikonal', 0.0):.4f} "
            f"metricGrad={train_metrics.get('sdf_metric_gradient', 0.0):.4f} "
            f"fuzz={train_metrics.get('boundary_fuzz', 0.0):.4f} halo={train_metrics.get('boundary_halo', 0.0):.4f} "
            f"zero={train_metrics.get('boundary_sdf_zero', 0.0):.4f} "
            f"gateRaw={train_metrics.get('boundary_gate_probability_edge_mean', 0.0):.3f} "
            f"gateEdge={train_metrics.get('boundary_gate_edge_mean', 0.0):.3f} "
            f"gateApplied={train_metrics.get('boundary_gate_applied_edge_mean', 0.0):.3f} "
            f"pixRegret={train_metrics.get('boundary_pixel_regret', 0.0):.5f} "
            f"width={train_metrics.get('boundary_transition_width_mean', 0.0):.3f}px"
        )
        _status(
            f"  valid total={validation_metrics['total']:.6f} "
            f"albedo={validation_metrics['albedo']:.6f} "
            f"regret={validation_metrics['regret']:.6f} "
            f"tangent={validation_metrics['tangent_coherence']:.6f} "
            f"curvature={validation_metrics['curvature_coherence']:.6f} "
            f"profile={validation_metrics.get('boundary_profile', 0.0):.6f} "
            f"sdfSurf={validation_metrics.get('sdf_surface', 0.0):.4f} "
            f"coarse={validation_metrics.get('coarse_sdf_surface', 0.0):.4f} "
            f"eik={validation_metrics.get('sdf_eikonal', 0.0):.4f} "
            f"metricGrad={validation_metrics.get('sdf_metric_gradient', 0.0):.4f} "
            f"fuzz={validation_metrics.get('boundary_fuzz', 0.0):.4f} halo={validation_metrics.get('boundary_halo', 0.0):.4f} "
            f"zero={validation_metrics.get('boundary_sdf_zero', 0.0):.4f} "
            f"gateRaw={validation_metrics.get('boundary_gate_probability_edge_mean', 0.0):.3f} "
            f"gateEdge={validation_metrics.get('boundary_gate_edge_mean', 0.0):.3f} "
            f"gateFlat={validation_metrics.get('boundary_gate_flat_mean', 0.0):.3f} "
            f"gateApplied={validation_metrics.get('boundary_gate_applied_edge_mean', 0.0):.3f} "
            f"pixRegret={validation_metrics.get('boundary_pixel_regret', 0.0):.5f} "
            f"width={validation_metrics.get('boundary_transition_width_mean', 0.0):.3f}px "
            f"delta={validation_metrics.get('boundary_delta_rms', 0.0):.5f} "
            f"proxy={validation_metrics.get('geometry_proxy_improvement', 0.0):+.6f} "
            f"regression={validation_metrics['regression_fraction']*100.0:.2f}% "
            f"improved={validation_metrics['improvement_fraction']*100.0:.2f}%"
        )
        if synthetic_sdf_metrics is not None:
            sdf_score = (
                float(synthetic_sdf_metrics.get("sdf_zero_rms_pixels", 999.0))
                + 4.0 * float(synthetic_sdf_metrics.get("sdf_eikonal", 999.0))
                + 0.50 * float(synthetic_sdf_metrics.get("sdf_metric_gradient", 999.0))
            )
            _status(
                f"  synthetic-sdf zeroRMS={synthetic_sdf_metrics.get('sdf_zero_rms_pixels', 0.0):.3f}px "
                f"eik={synthetic_sdf_metrics.get('sdf_eikonal', 0.0):.3f} "
                f"gradMean={synthetic_sdf_metrics.get('sdf_grad_norm_mean', 0.0):.3f} "
                f"metricGrad={synthetic_sdf_metrics.get('sdf_metric_gradient', 0.0):.3f} "
                f"sign=+{synthetic_sdf_metrics.get('sdf_positive_fraction', 0.0)*100.0:.1f}%/"
                f"-{synthetic_sdf_metrics.get('sdf_negative_fraction', 0.0)*100.0:.1f}% "
                f"polarity+={synthetic_sdf_metrics.get('sdf_polarity_positive_fraction', 0.0)*100.0:.1f}% "
                f"score={sdf_score:.3f}"
            )
            if float(synthetic_sdf_metrics.get("sdf_zero_rms_pixels", 0.0)) > 4.0:
                _status("  [SDF-COLLAPSE-WATCH] zero-set remains far from exact synthetic contours.")
        print(f"  seconds={seconds:.1f} throughput={tiles_per_second:.2f}tile/s")
        sample_path = output_dir / "samples" / f"epoch_{epoch:03d}_{phase}" / "validation_contact_sheet.png"
        _save_contact_sheet(validation_outputs, validation_batch, sample_path)

        record = {
            "epoch": epoch,
            "phase": phase,
            "seconds": seconds,
            "batches": len(train_loader),
            "tiles": config.tiles_per_epoch,
            "tiles_per_second": tiles_per_second,
            "train": train_metrics,
            "validation": validation_metrics,
            "syntheticSdfValidation": synthetic_sdf_metrics,
            "performance": {
                "profile": config.performance_profile,
                "precision": precision_name,
                "optimizer": optimizer_mode,
                "workers": workers,
                "prefetchFactor": config.data_loader_prefetch_factor,
                "channelsLast": config.channels_last and use_amp,
                "cudaPrefetch": config.cuda_prefetch and use_amp,
            },
        }
        history.append(record)
        if synthetic_sdf_metrics is not None:
            sdf_score = (
                float(synthetic_sdf_metrics.get("sdf_zero_rms_pixels", 999.0))
                + 4.0 * float(synthetic_sdf_metrics.get("sdf_eikonal", 999.0))
                + 0.50 * float(synthetic_sdf_metrics.get("sdf_metric_gradient", 999.0))
            )
            if sdf_score < best_sdf_score:
                best_sdf_score = sdf_score
                _atomic_torch_save({
                    "epoch": epoch,
                    "validation_total": sdf_score,
                    "selection_kind": "synthetic-sign-gauge-metric-sdf",
                    "synthetic_sdf_validation": synthetic_sdf_metrics,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                }, best_path)
                _status(f"  synthetic SDF checkpoint selected: epoch={epoch:03d} score={sdf_score:.3f}")
        if phase == "physical-finetune" and validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            _atomic_torch_save({
                "epoch": epoch,
                "validation_total": best_validation,
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            }, best_path)

        _atomic_torch_save({
            "schema": MODEL_SCHEMA,
            "config_hash": _config_hash(config),
            "dataset_fingerprint": fingerprint,
            "completed_epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": _capture_rng(),
            "history": history,
            "best_validation": best_validation,
            "best_sdf_score": best_sdf_score,
        }, state_path)

        if phase == "physical-finetune" and early_stop_patience is not None and early_stop_patience > 0:
            current_validation = float(validation_metrics["total"])
            if current_validation < early_stop_best - max(0.0, float(early_stop_min_delta)):
                early_stop_best = current_validation
                early_stop_stale = 0
            else:
                early_stop_stale += 1
            _status(
                f"  convergence gate: best={early_stop_best:.6f} "
                f"stale={early_stop_stale}/{early_stop_patience}"
            )
            if early_stop_stale >= early_stop_patience:
                early_stopped_epoch = epoch
                _status(
                    f"  Early convergence stop at epoch {epoch:03d}; all V9 phases were entered "
                    "and held-out validation stopped improving materially."
                )
                break

        if staged_stop_epoch is not None and epoch >= staged_stop_epoch:
            _status(
                f"  Staged checkpoint stop reached after {phase}; "
                "preserving optimizer/RNG state for gated resume."
            )
            break

    if not best_path.is_file():
        # Stability profiles may omit the fine-tune phase. Use the final finite
        # epoch explicitly, never a NaN comparison fallback.
        best_validation = float(history[-1]["validation"]["total"])
        _atomic_torch_save({
            "epoch": int(history[-1]["epoch"]),
            "validation_total": best_validation,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        }, best_path)
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["state_dict"], strict=True)
    model.to(device).eval()
    checkpoint = {
        "schema": MODEL_SCHEMA,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": config.to_dict(),
        "parameter_count": parameter_count(model),
        "model_sha256": model_hash(model),
        "best_epoch": int(best["epoch"]),
        "best_validation_total": float(best["validation_total"]),
        "selection_kind": str(best.get("selection_kind", "real-validation-total")),
        "best_synthetic_sdf_validation": best.get("synthetic_sdf_validation"),
        "acceptance_regression_fraction": float(history[int(best["epoch"]) - 1]["validation"].get("regression_fraction", 1.0)),
        "training_safety_pass": float(history[int(best["epoch"]) - 1]["validation"].get("regression_fraction", 1.0)) <= config.maximum_validation_regression_fraction,
        # Compatibility alias only. Overall reconstruction acceptance is decided
        # later by the staged analytic proof and real-asset audit.
        "acceptance_pass": float(history[int(best["epoch"]) - 1]["validation"].get("regression_fraction", 1.0)) <= config.maximum_validation_regression_fraction,
        "reconstruction_acceptance_pass": False,
        "dataset_fingerprint": fingerprint,
        "history": history,
        "planned_epochs": config.total_epochs,
        "epochs_completed": len(history),
        "early_stopped": early_stopped_epoch is not None,
        "early_stop_epoch": early_stopped_epoch,
        "staged_stop_phase": stop_after_phase,
        "staged_stop_reached": bool(
            stop_after_phase is not None and len(history) < config.total_epochs
        ),
    }
    _atomic_torch_save(checkpoint, checkpoint_path)
    elapsed_minutes = (time.perf_counter() - started) / 60.0
    metadata = {
        "schema": MODEL_SCHEMA,
        "checkpoint": str(checkpoint_path),
        "checkpointSha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "modelSha256": checkpoint["model_sha256"],
        "bestEpoch": checkpoint["best_epoch"],
        "bestValidationTotal": checkpoint["best_validation_total"],
        "selectionKind": checkpoint["selection_kind"],
        "bestSyntheticSdfValidation": checkpoint.get("best_synthetic_sdf_validation"),
        "acceptanceRegressionFraction": checkpoint["acceptance_regression_fraction"],
        "trainingSafetyPass": checkpoint["training_safety_pass"],
        "acceptancePass": checkpoint["acceptance_pass"],
        "reconstructionAcceptancePass": False,
        "trainingMinutes": elapsed_minutes,
        "plannedEpochs": config.total_epochs,
        "epochsCompleted": len(history),
        "earlyStopped": early_stopped_epoch is not None,
        "earlyStopEpoch": early_stopped_epoch,
        "stagedStopPhase": stop_after_phase,
        "stagedStopReached": bool(
            stop_after_phase is not None and len(history) < config.total_epochs
        ),
        "datasetFingerprint": fingerprint,
        "architecture": architecture_summary(model),
        "performance": history[-1].get("performance", {}),
        "qualityContract": {
            "physicalReconstruction": True,
            "illustratedPresentation": False,
            "fullMapProposals": True,
            "contourSdf": True,
            "upscaleFactor": 4,
        },
    }
    _atomic_json(metadata, metadata_path)
    print("=" * 68)
    print("NSAMDR V9.8.3 GEOMETRY CHECKPOINT READY")
    print(f"Checkpoint               : {checkpoint_path}")
    print(f"Metadata                 : {metadata_path}")
    print(f"Best validation total    : {checkpoint['best_validation_total']:.6f} (epoch {checkpoint['best_epoch']:03d})")
    print(f"Validation regressions   : {checkpoint['acceptance_regression_fraction']*100.0:.2f}%")
    print(f"Training safety gate     : {'PASS' if checkpoint['training_safety_pass'] else 'REJECT'}")
    print("Reconstruction acceptance: PENDING staged geometry proof")
    print(f"Training time            : {elapsed_minutes:.1f} minutes")
    print("=" * 68)
    return metadata
