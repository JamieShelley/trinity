#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v9.config import V9Config
from v9.dataset import _pack_sample, _synthetic_geometry_sample
from v9.losses import compute_losses
from v9.model import FidelityResidualNetV9, parameter_count


class NSAMDRSmokeTestApplication:
    # Purpose: Implement main for NSAMDRSmokeTestApplication.
    # Called by: External callers and the owning workflow.
    # Calls: No same-class helper methods.
    def main(self) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        args = parser.parse_args()
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable")
        device = torch.device(args.device)

        cfg = V9Config(
            tile_size=32,
            widths=(32, 32, 48, 64),
            blocks_per_level=(1, 1, 1, 1),
            decoder_blocks=(1, 1, 1),
            attention_heads=4,
            batch_size=1,
            synthetic_geometry_probability=1.0,
        )
        cfg.validate()
        model = FidelityResidualNetV9(cfg).to(device)

        # Identity contract remains exact even after adding the geometry branch.
        x = torch.rand(1, 17, 32, 32, device=device)
        x[:, 3:5] = x[:, 3:5] * 2.0 - 1.0
        model.set_phase("sdf-bootstrap")
        with torch.no_grad():
            identity = model(x)
        max_delta = float((identity["albedo"] - identity["baseline_albedo"]).abs().max().cpu())
        if max_delta > 1e-7:
            raise SystemExit(f"V9 identity initialization failed: max delta={max_delta}")

        # Exercise the real exact-geometry generation/degradation path and every
        # boundary-coherence loss in forward/backward.
        rng = random.Random(20260807)
        target_size = cfg.tile_size * cfg.target_scale
        albedo, normal, material, sdf, orientation, edge = _synthetic_geometry_sample(
            target_size, cfg, rng
        )
        sample = _pack_sample(
            albedo, normal, material, 1.0, sdf, orientation, edge,
            cfg, rng, geometry_exact=1.0,
        )
        batch: dict[str, torch.Tensor] = {}
        for key, value in sample.items():
            batch[key] = value.unsqueeze(0).to(device)

        model.set_phase("boundary-hardening")
        outputs = model(batch["input"])
        losses = compute_losses(outputs, batch, cfg, "boundary-hardening")
        losses["total"].backward()
        if not torch.isfinite(losses["total"]):
            raise SystemExit("non-finite V9 geometric smoke loss")
        for name in (
            "geometric_alignment", "tangent_coherence", "curvature_coherence", "sdf_curvature"
        ):
            if name not in losses or not torch.isfinite(losses[name]):
                raise SystemExit(f"missing/non-finite V9 geometric loss: {name}")

        expected = {
            "albedo": (1, 3, target_size, target_size),
            "normal_xy": (1, 2, target_size, target_size),
            "confidence": (1, 1, target_size, target_size),
            "albedo_delta_fine": (1, 3, target_size, target_size),
        }
        for key, shape in expected.items():
            if tuple(outputs[key].shape) != shape:
                raise SystemExit(f"bad {key} shape: {tuple(outputs[key].shape)}")

        print(
            f"NSAMDR V9 geometric smoke passed device={device} "
            f"parameters={parameter_count(model):,} identityDelta={max_delta:.9f} "
            f"total={float(losses['total'].detach().cpu()):.6f}"
        )
        return 0

_n_s_a_m_d_r_smoke_test_application = NSAMDRSmokeTestApplication()
main = _n_s_a_m_d_r_smoke_test_application.main


if __name__ == "__main__":
    raise SystemExit(main())
