#!/usr/bin/env python3
"""Validate the trained NSAMDR V4 tile-context checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_nsamdr_kernel as trainer


def validate(repo_root: Path, metadata_path: Path | None = None) -> None:
    if metadata_path is None:
        metadata_path = repo_root / "artifacts/nsamdr/neural/nsamdr_tile_context.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema"] == trainer.MODEL_SCHEMA
    checkpoint_path = Path(metadata["checkpointPath"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path
    assert checkpoint_path.is_file(), checkpoint_path

    model, config, checkpoint = trainer.load_trained_model(checkpoint_path, "cpu")
    count = trainer.parameter_count(model)
    assert count == int(metadata["parameterCount"])
    assert 100_000 <= count <= 500_000, count
    assert config.residual_blocks == 8
    assert metadata["receptiveFieldPixels"] >= 125
    assert metadata["runtime"] == "offline-overlapping-tile-inference"
    assert metadata["inputChannels"] == [
        "albedoR", "albedoG", "albedoB", "normalX", "normalY", "material", "paint", "roughness"
    ]
    assert metadata["outputs"] == [
        "offsetX", "offsetY", "residualR", "residualG", "residualB", "confidence"
    ]
    assert checkpoint["model_sha256"] == trainer.model_hash(model)

    rng = np.random.default_rng(1337)
    sample = torch.from_numpy(rng.random((2, trainer.INPUT_CHANNELS, 64, 64), dtype=np.float32))
    with torch.no_grad():
        output = model(sample)
    assert output["corrected"].shape == (2, 3, 64, 64)
    assert output["flow"].shape == (2, 2, 64, 64)
    assert output["residual"].shape == (2, 3, 64, 64)
    assert output["confidence"].shape == (2, 1, 64, 64)
    for name, value in output.items():
        assert torch.isfinite(value).all(), name
    assert float(output["flow"].abs().max()) <= config.max_offset_pixels + 1.0e-4
    assert float(output["residual"].abs().max()) <= config.max_residual + 1.0e-4
    assert float(output["confidence"].min()) >= 0.0
    assert float(output["confidence"].max()) <= 1.0
    assert float(output["corrected"].min()) >= 0.0
    assert float(output["corrected"].max()) <= 1.0

    metrics = metadata.get("metrics", {})
    assert float(metrics.get("validationL1", 999.0)) < 0.25
    assert float(metrics.get("validationEdge", 999.0)) < 0.25
    assert float(metrics.get("flatDeltaMean", 999.0)) < 0.10

    print("NSAMDR V4 tile-context checkpoint validation passed")
    print(
        f"  parameters={count:,} receptive_field={metadata['receptiveFieldPixels']}px "
        f"validation_l1={float(metrics.get('validationL1', 0.0)):.6f}"
    )
    print(f"  checkpoint={checkpoint_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args(argv)
    validate(args.repo_root.resolve(), args.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
