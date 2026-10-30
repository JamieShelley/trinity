#!/usr/bin/env python3
"""Experimental V178 structural-continuity training; does not promote V5.

Separate CLI keeps previously authenticated V5 checkpoint/source identity intact.
No real EVE data or analytic stress cases enter training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.nsamdr.neural.train_nsamdr_v178_contour import evaluate
from tools.nsamdr.neural.v178.composed_geometry import mixed_batch
from tools.nsamdr.neural.v178.contour_model import (
    StructuralContourHead, contour_supervised_loss,
)

MODEL_SCHEMA = "NSAMDR_V178_CONTOUR_HEAD_V6_COMPOSED_GEOMETRY_EXPERIMENT"
REPORT_SCHEMA = "NSAMDR_V178_G2_CONTOUR_REPORT_V4_COMPOSED_GEOMETRY_EXPERIMENT"
GEOMETRY_POLICY = "MIXED_V5_AND_V178_CONTINUOUS_COMPOSITIONS_P50_V1"
RGB_POLICY = "SEEDED_PER_SAMPLE_RGB_CONTRAST_INVERSION_P50_V1"
AUX_POLICY = "INDEPENDENT_COARSE_FIELDS_WITH_MISSING_MAP_DEFAULTS_V1"
FILES = (
    "tools/nsamdr/neural/v177/synthetic_geometry.py",
    "tools/nsamdr/neural/v178/contour_model.py",
    "tools/nsamdr/neural/train_nsamdr_v178_contour.py",
    "tools/nsamdr/neural/v178/composed_geometry.py",
    "tools/nsamdr/neural/train_nsamdr_v178_continuity.py",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_hash(repo: Path) -> str:
    h = hashlib.sha256()
    for name in FILES:
        h.update(name.encode())
        h.update((repo / name).read_bytes())
    return h.hexdigest()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=17852)
    p.add_argument("--learning-rate", type=float, default=.0007)
    p.add_argument("--resume", choices=("fresh", "auto"), default="fresh")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not (1 <= args.steps <= 12000 and 1 <= args.batch_size <= 16
            and 16 <= args.patch_size <= 128 and 0 < args.learning_rate <= .01):
        raise ValueError("out-of-bounds G2 synthetic training parameters")
    repo = args.repo_root.resolve()
    identity = source_hash(repo)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    if args.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = StructuralContourHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    config = {"seed": args.seed, "patchSize": args.patch_size,
              "batchSize": args.batch_size, "learningRate": args.learning_rate,
              "sourceHash": identity, "auxiliaryInputPolicy": AUX_POLICY,
              "rgbInputPolicy": RGB_POLICY, "geometryPolicy": GEOMETRY_POLICY}
    folder = repo / "artifacts/nsamdr/diagnostics/v178_golden_contour" / (
        f"seed_{args.seed}_patch_{args.patch_size}_batch_{args.batch_size}")
    # Fail closed on an occupied seed. V4/V5 are never overwritten by V6.
    if folder.exists() and any((folder / f).exists()
                               for f in ("candidate.pt", "report.json")):
        raise RuntimeError("existing candidate/report must not be overwritten")
    folder.mkdir(parents=True, exist_ok=True)
    resume = folder / "resume.pt"
    start = 0
    if args.resume == "fresh" and resume.exists():
        raise RuntimeError("a resume exists; use --resume auto or a new seed")
    if args.resume == "auto" and resume.exists():
        previous = torch.load(resume, map_location="cpu", weights_only=True)
        if previous.get("schema") != MODEL_SCHEMA or previous.get("config") != config:
            raise RuntimeError("V6 resume source/configuration mismatch")
        model.load_state_dict(previous["model"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        for state in optimizer.state.values():
            for name, value in state.items():
                if torch.is_tensor(value):
                    state[name] = value.to(device)
        start = int(previous["step"])
        if start > args.steps:
            raise ValueError("training horizon precedes resumed step")
    before = evaluate(model, device, patch=args.patch_size, seed=args.seed)
    before_inverted = evaluate(model, device, patch=args.patch_size,
                               seed=args.seed, invert_rgb=True)
    began = time.monotonic()
    for i in range(start, args.steps):
        model.train()
        a, n, m, target = mixed_batch(args.batch_size, args.patch_size,
                                       args.seed + i * 8191, device,
                                       augment_rgb=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(a, n, m)
        loss, _ = contour_supervised_loss(prediction["candidate_contour"], target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite G2 contour loss at {i+1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        step = i + 1
        if step == 1 or step % 25 == 0 or step == args.steps:
            print(f"[v178-continuity] update={step}/{args.steps} "
                  f"contour_loss={float(loss):.6f}", flush=True)
        if step % 50 == 0 or step == args.steps:
            temp = folder / "resume.tmp"
            torch.save({"schema": MODEL_SCHEMA, "config": config, "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, temp)
            os.replace(temp, resume)
    after = evaluate(model, device, patch=args.patch_size, seed=args.seed)
    after_inverted = evaluate(model, device, patch=args.patch_size,
                              seed=args.seed, invert_rgb=True)
    # This is the historical synthetic gate, not real-domain qualification.
    qualified = (
        after["relativeContourRecovery"] >= .20
        and after["candidateContourLoss"] < before["candidateContourLoss"]
        and after["candidateContourLoss"] < after["baselineContourLoss"]
        and after["candidateForeground"]["contourRecall"] >= .45
        and after["candidateForeground"]["contourPrecision"] >= .20
        and after["candidateForeground"]["contourIoU"] >= .20
        and after["candidateForeground"]["predictedActiveFraction"] >= .005
    )
    checkpoint = folder / "candidate.pt"
    temp = folder / "candidate.tmp"
    torch.save({"schema": MODEL_SCHEMA, "config": config,
                "model": model.state_dict(), "syntheticPass": qualified,
                "steps": args.steps, "productionPromotionAllowed": False,
                "realEveReferenceQualified": False}, temp)
    os.replace(temp, checkpoint)
    report = {"schema": REPORT_SCHEMA, "modelSchema": MODEL_SCHEMA,
              "device": str(device), "torchVersion": str(torch.__version__),
              "decision": "SYNTHETIC_CONTOUR_PASS_ONLY" if qualified else "G2_SYNTHETIC_FAIL",
              "sourceHash": identity, "config": config,
              "checkpoint": str(checkpoint.resolve()),
              "checkpointSha256": digest(checkpoint), "before": before,
              "after": after, "beforeContrastInverted": before_inverted,
              "afterContrastInverted": after_inverted,
              "requiredRelativeContourRecovery": .20,
              "requiredContourRecall": .45, "requiredContourPrecision": .20,
              "requiredContourIoU": .20, "requiredPredictedActiveFraction": .005,
              "trainingUpdates": args.steps - start,
              "elapsedSeconds": time.monotonic() - began,
              "realRavenReferenceAvailable": False,
              "realEveQualityVerified": False,
              "productionPromotionAllowed": False, "sealedTestAccess": False}
    output = folder / "report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"[v178-continuity] {report['decision']} report={output}", flush=True)
    return 0 if qualified else 2


if __name__ == "__main__":
    raise SystemExit(main())
