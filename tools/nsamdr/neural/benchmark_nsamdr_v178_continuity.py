#!/usr/bin/env python3
"""Non-gating V178 paired-sampling continuity/false-structure probe.

The analytic cases here are fixed holdouts: never use these seeds for training.
Scores describe clean synthetic geometry; no real EVE ground truth is inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.nsamdr.diagnostics.v178_contrast_transfer.probe_analytic_stress import (
    authenticated_models, longest_gap,
)
from tools.nsamdr.neural.train_nsamdr_v178_continuity import (
    MODEL_SCHEMA, REPORT_SCHEMA, source_hash, digest,
)
from tools.nsamdr.neural.v178.composed_geometry import KINDS, make_case
from tools.nsamdr.neural.v178.contour_model import (
    StructuralContourHead, foreground_metrics,
)

BENCHMARK_SCHEMA = "NSAMDR_V178_COMPOSED_CONTINUITY_DIAGNOSTIC_V1"
SEED_START = 50000000


def check_candidate(path: Path, repo: Path, device: torch.device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    report = json.loads((path.parent / "report.json").read_text())
    identity = source_hash(repo)
    if not (state.get("schema") == MODEL_SCHEMA
            and report.get("schema") == REPORT_SCHEMA
            and report.get("modelSchema") == MODEL_SCHEMA
            and report.get("config") == state.get("config")
            and state.get("config", {}).get("sourceHash") == identity
            and report.get("sourceHash") == identity
            and report.get("checkpointSha256") == digest(path)
            and state.get("steps", 0) > 0):
        raise RuntimeError("V6 candidate authentication failed")
    model = StructuralContourHead().to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, {"checkpointSha256": digest(path),
                   "modelSchema": MODEL_SCHEMA, "sourceHash": identity,
                   "trainingSeed": state["config"]["seed"],
                   "trainingSteps": state["steps"],
                   "trainingDecision": report["decision"]}


def band_survival(prediction: torch.Tensor, segment, *, threshold=.30):
    p, q = segment
    length = math.dist(p, q)
    count = max(2, math.ceil(length / .25) + 1)
    t = torch.linspace(0, 1, count)
    center = torch.stack((p[0] + t * (q[0] - p[0]),
                          p[1] + t * (q[1] - p[1])), dim=1)
    direction = torch.tensor([-(q[1] - p[1]) / length,
                               (q[0] - p[0]) / length])
    offsets = torch.linspace(-.5, .5, 9)
    positions = center[:, None] + offsets[None, :, None] * direction
    # Pixel-center convention: input positions are native continuous units.
    grid = (positions * 4. / prediction.shape[-1]) * 2 - 1
    band = F.grid_sample(prediction.cpu(), grid[None], align_corners=False,
                         padding_mode="zeros")[0, 0].amax(1)
    survived = band >= threshold
    return {"survivalFraction": float(survived.float().mean()),
            "longestGapNative": longest_gap(~survived, False) * length / (count - 1),
            "meanBandCoverage": float(band.mean())}


def point_value(field: torch.Tensor, point: tuple[float, float]) -> float:
    x, y = point
    grid = torch.tensor([[[[(x * 4 / field.shape[-1]) * 2 - 1,
                             (y * 4 / field.shape[-2]) * 2 - 1]]]])
    return float(F.grid_sample(field.cpu(), grid, align_corners=False)[0, 0, 0, 0])


def metrics(prediction, target, details):
    stats = foreground_metrics(prediction, target)
    outside = target < .01
    stats["falseStructureFractionOutsideTarget"] = float(
        ((prediction >= .30) & outside).sum() / outside.sum())
    scores = [band_survival(prediction, seg) for seg in details["segments"]]
    stats["paths"] = scores
    if scores:
        stats["meanPathSurvival"] = sum(v["survivalFraction"] for v in scores) / len(scores)
        stats["longestPathGapNative"] = max(v["longestGapNative"] for v in scores)
    if details["junction"] is not None:
        # A junction counts only if each component arm is present near its
        # genuine crossing. This is a diagnostic, not graph-connectivity proof.
        cx, cy = details["junction"]
        stats["junctionCenterSurvived"] = point_value(prediction, (cx, cy)) >= .30
        stats["junctionArmsSurvived"] = all(
            band_survival(prediction, s)["survivalFraction"] >= .75
            for s in details["segments"])
    if details["kind"] == "parallel":
        one, two = details["segments"]
        merged = 0
        supported = 0
        for t in (.2, .3, .4, .5, .6, .7, .8):
            a = (one[0][0] + (one[1][0] - one[0][0]) * t,
                 one[0][1] + (one[1][1] - one[0][1]) * t)
            b = (two[0][0] + (two[1][0] - two[0][0]) * t,
                 two[0][1] + (two[1][1] - two[0][1]) * t)
            mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            if min(point_value(prediction, a), point_value(prediction, b)) >= .30:
                supported += 1
                merged += point_value(prediction, mid) >= .30
        stats["parallelBothSurvivedStations"] = supported
        stats["parallelMergedFractionWhenBothSurvive"] = (
            merged / supported if supported else None)
    return stats


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    models, provenance = authenticated_models(device)
    if args.candidate:
        models["v6"], provenance["v6"] = check_candidate(
            args.candidate.resolve(), args.repo_root.resolve(), device)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for kind_index, kind in enumerate(KINDS):
        for index in range(4):
            seed = SEED_START + 10000 * kind_index + 1723 * index
            for sampling in ("area", "point"):
                a, n, m, t, details = make_case(32, seed, kind=kind,
                                                 sampling=sampling, phase=(2, 1))
                a, n, m, t = (v[None].to(device) for v in (a, n, m, t))
                row = {"kind": kind, "index": index, "seed": seed,
                       "sampling": sampling, "targetSha256": hashlib.sha256(
                           t.cpu().numpy().tobytes()).hexdigest(),
                       "targetCoverage": float(t.mean()),
                       "nativeEdgeMean": float(m[:, :1].mean()),
                       "scores": {}}
                for polarity in ("original", "inverted"):
                    rgb = a if polarity == "original" else 1 - a
                    row["scores"][polarity] = {}
                    for key, model in models.items():
                        pred = model(rgb, n, m)["candidate_contour"]
                        row["scores"][polarity][key] = metrics(pred, t, details)
                rows.append(row)
    # Explicit paired intervention ensures exactly the same analytic target.
    pairs = []
    for i in range(0, len(rows), 2):
        if rows[i]["targetSha256"] != rows[i + 1]["targetSha256"]:
            raise RuntimeError(f"paired sampling changed analytic geometry: {rows[i]['kind']} {rows[i]['seed']} {rows[i]['targetSha256']} {rows[i+1]['targetSha256']}")
        pairs.append({"kind": rows[i]["kind"], "seed": rows[i]["seed"],
                      "areaEdgeMean": rows[i]["nativeEdgeMean"],
                      "pointEdgeMean": rows[i + 1]["nativeEdgeMean"]})
    samples = {key: {polarity: {} for polarity in ("original", "inverted")}
               for key in KINDS}
    for kind in KINDS:
        for polarity in ("original", "inverted"):
            for model in models:
                values = [r["scores"][polarity][model] for r in rows
                          if r["kind"] == kind]
                numeric = ("contourPrecision", "contourRecall", "contourIoU",
                           "falseStructureFractionOutsideTarget")
                summary = {key: sum(v[key] for v in values) / len(values)
                           for key in numeric}
                scored = [v for v in values if "meanPathSurvival" in v]
                if scored:
                    summary["meanPathSurvival"] = sum(
                        v["meanPathSurvival"] for v in scored) / len(scored)
                    summary["worstGapNative"] = max(
                        v["longestPathGapNative"] for v in scored)
                samples[kind][polarity][model] = summary
    report = {"schema": BENCHMARK_SCHEMA, "diagnosticOnly": True,
              "acceptanceGateChanged": False, "realEveAccuracyProven": False,
              "cases": len(rows), "pairedSamplingCases": len(pairs),
              "trainingSeedDisjoint": True, "provenance": provenance,
              "limitations": ["Analytic geometry only; not EVE qualification",
                              "Point/area edge means are descriptive, not an observability proof",
                              "Junction survival is not exact topology equivalence"],
              "summary": samples, "pairedSampling": pairs, "rows": rows}
    for model, info in provenance.items():
        if model not in models or model not in ("v4", "v5", "v6"):
            continue
        if model == "v6":
            train = {info["trainingSeed"] + 8191 * i
                     for i in range(info["trainingSteps"])}
            seeds = {r["seed"] for r in rows}
            if train & seeds:
                raise RuntimeError("training and continuity holdout seeds overlap")
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"continuity cases={len(rows)} models={','.join(models)} "
          f"report={args.output / 'report.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
