from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import torch

from .config import V14Config
from .model import MODEL_SCHEMA, NSAMDRV14


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    path: Path,
    model: NSAMDRV14,
    config: V14Config,
    *,
    epoch: int,
    phase: str,
    metrics: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": MODEL_SCHEMA,
        "epoch": int(epoch),
        "phase": str(phase),
        "config": config.to_dict(),
        "state_dict": model.state_dict(),
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[NSAMDRV14, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or payload.get("schema") != MODEL_SCHEMA:
        raise RuntimeError(f"V14 checkpoint schema mismatch: {path}")
    config_payload = dict(payload.get("config") or {})
    fields = V14Config.__dataclass_fields__
    config = V14Config(**{k: v for k, v in config_payload.items() if k in fields})
    config.validate()
    model = NSAMDRV14(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def promote_final(source: Path, experiment_dir: Path, qualification: dict[str, object]) -> Path:
    final_dir = experiment_dir / "checkpoints" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    destination = final_dir / "nsamdr_v14.pt"
    shutil.copy2(source, destination)

    # Promotion is not trusted until the exact copied bytes strict-load into a fresh
    # production model. This verifies the same artifact later used by preview/baking.
    reloaded, payload = load_checkpoint(destination, torch.device("cpu"))
    contract = reloaded.architecture_contract()
    if contract.get("schema") != MODEL_SCHEMA or tuple(contract.get("retiredComponents", ())) != ():
        raise RuntimeError("V14 promoted checkpoint failed clean production architecture verification")

    digest = sha256_file(destination)
    manifest = {
        "schema": "NSAMDR_V14_FINAL_MANIFEST_V1",
        "status": "completed",
        "qualified": True,
        "selectionKind": "production-final",
        "checkpoint": {
            "path": str(destination.relative_to(experiment_dir)),
            "sha256": digest,
            "schema": MODEL_SCHEMA,
            "immutable": True,
            "selectionKind": "production-final",
            "strictReloadVerified": True,
            "epoch": int(payload.get("epoch", 0)),
            "phase": str(payload.get("phase", "")),
        },
        "productionRuntimeIntegrity": {
            "passed": True,
            "strictReload": True,
            "modelSchema": MODEL_SCHEMA,
            "architecture": contract,
        },
        "qualification": qualification,
    }
    (experiment_dir / "final_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        destination.chmod(0o444)
    except OSError:
        pass
    return destination
