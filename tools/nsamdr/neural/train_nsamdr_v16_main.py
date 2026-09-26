#!/usr/bin/env python3
"""Canonical V16.2 main research training entrypoint.

This promotes the evidence-backed D4 + authority-balanced broad-authority recipe
into the operator workflow without weakening production qualification gates.
The run remains research/non-promotable until candidate C, material semantics,
and BenefitSelector qualification are complete.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from probe_nsamdr_v16_full_broad import DEFAULT_MANIFEST, run as run_full_broad


POINTER_SCHEMA = "NSAMDR_V16_MAIN_RESEARCH_TRAINING_V1"
LATEST_ROOT = "artifacts/nsamdr/main_training"


def _manifest_path(repo_root: Path, raw: str) -> Path:
    value = Path(raw)
    if not value.is_absolute():
        value = (repo_root / value).resolve()
    if not value.is_file():
        raise RuntimeError(
            f"main V16 manifest is missing: {value}\n"
            "Run: scripts\\build\\nsamdr.bat authored-prior-corpus"
        )
    return value


def _training_geometry(manifest: dict[str, Any]) -> tuple[int, int, int]:
    records = [
        record
        for record in manifest.get("crops", [])
        if str(record.get("split") or "") == "train"
    ]
    if not records:
        raise RuntimeError("main V16 manifest has no training crops")

    counts: Counter[str] = Counter()
    for record in records:
        family = str(record.get("family_id") or record.get("familyId") or "").strip()
        if not family:
            raise RuntimeError("main V16 training crop has no authority id")
        counts[family] += 1

    crop_counts = sorted(set(counts.values()))
    if len(crop_counts) != 1:
        raise RuntimeError(
            "main V16 D4 schedule requires a uniform crop count per authority; "
            f"observed counts={crop_counts}"
        )
    crops_per_authority = crop_counts[0]
    if crops_per_authority < 1:
        raise RuntimeError("main V16 training has no authored crops per authority")

    return len(counts), crops_per_authority, len(records)


def _stage_schedule(train_crop_count: int, d4_passes: int) -> list[int]:
    passes = max(1, int(d4_passes))
    base = int(train_crop_count)
    if base < 1:
        raise ValueError("train crop count must be positive")

    stages = {base, base * 2, base * 4, base * 8}
    full_d4 = base * 8
    for index in range(2, passes + 1):
        stages.add(full_d4 * index)
    return sorted(stages)


def _write_latest_pointer(
    repo_root: Path,
    *,
    report_path: Path,
    manifest_path: Path,
    train_authorities: int,
    crops_per_authority: int,
    train_crop_count: int,
    d4_passes: int,
    stages: list[int],
) -> Path:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    final = dict(report.get("final") or {})
    validation = dict(final.get("validation") or {})
    final_step = int(final.get("step") or 0)
    preview_root = report_path.parent / "previews" / f"step_{final_step:06d}"
    comparisons = sorted(preview_root.glob("sample_*/albedo_comparison.png"))

    root = repo_root / LATEST_ROOT
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": POINTER_SCHEMA,
        "role": "main-v16-research-training",
        "qualificationEligible": False,
        "productionQualified": False,
        "qualificationReason": (
            "research training preview only; candidate C numerical gates, "
            "material semantics, and BenefitSelector qualification remain required"
        ),
        "trainingRecipe": {
            "sampling": "authority-balanced-complete-cycle-before-repeat",
            "augmentation": "d4-cyclic",
            "degradation": "clean",
            "trainAuthorityCount": int(train_authorities),
            "cropsPerAuthority": int(crops_per_authority),
            "trainCropCount": int(train_crop_count),
            "d4PassesRequested": int(d4_passes),
            "stages": [int(value) for value in stages],
        },
        "manifest": str(manifest_path.resolve()),
        "report": str(report_path.resolve()),
        "runDirectory": str(report_path.parent.resolve()),
        "resumeCheckpoint": str((report_path.parent / "resume_checkpoint.pt").resolve()),
        "finalStep": final_step,
        "previewRoot": str(preview_root.resolve()),
        "previewAlbedoComparison": (
            str(comparisons[0].resolve()) if comparisons else ""
        ),
        "candidateGateStatus": report.get("candidateGateStatus"),
        "heldout": {
            "globalRecovery": validation.get("median_global_recovery"),
            "edgeRecovery": validation.get("median_edge_recovery"),
            "gradientRecovery": validation.get("median_gradient_recovery"),
            "normalRecovery": validation.get("median_normal_recovery"),
            "latticeCellExcess": validation.get("median_lattice_cell_excess"),
            "heldOutAuthorityCount": validation.get("heldOutAuthorityCount"),
        },
    }
    pointer = root / "latest.json"
    pointer.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "LATEST.txt").write_text(str(report_path.parent.resolve()) + "\n", encoding="utf-8")
    return pointer


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run the canonical V16.2 main research training recipe: "
            "authority-balanced broad-authority sampling plus deterministic D4"
        )
    )
    value.add_argument("--repo-root", type=Path, default=Path.cwd())
    value.add_argument("--manifest", default=DEFAULT_MANIFEST)
    value.add_argument("--d4-passes", type=int, default=1)
    value.add_argument("--hr-size", type=int, default=512)
    value.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )
    value.add_argument(
        "--amp-precision",
        choices=("auto", "bf16", "fp16"),
        default="auto",
    )
    value.add_argument("--preview-samples", type=int, default=4)
    value.add_argument(
        "--resume",
        default="",
        help="resume_checkpoint.pt from an earlier main V16 research-training run",
    )
    return value


def run(args: argparse.Namespace) -> tuple[int, Path]:
    repo_root = args.repo_root.resolve()
    manifest_path = _manifest_path(repo_root, str(args.manifest))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_authorities, crops_per_authority, train_crop_count = _training_geometry(manifest)
    stages = _stage_schedule(train_crop_count, int(args.d4_passes))

    print("NSAMDR V16.2 MAIN RESEARCH TRAINING", flush=True)
    print(f"Manifest            : {manifest_path}", flush=True)
    print(f"Train authorities   : {train_authorities}", flush=True)
    print(f"Crops / authority   : {crops_per_authority}", flush=True)
    print(f"Train crops          : {train_crop_count}", flush=True)
    print("Sampling            : authority-balanced", flush=True)
    print("Augmentation        : deterministic D4 cyclic", flush=True)
    print(f"D4 passes requested : {int(args.d4_passes)}", flush=True)
    print(f"Checkpoint stages   : {stages}", flush=True)
    print("Qualification       : research preview only; production gates unchanged", flush=True)

    forwarded = SimpleNamespace(
        repo_root=repo_root,
        manifest=str(manifest_path),
        stages=",".join(str(value) for value in stages),
        hr_size=int(args.hr_size),
        validation_samples=0,
        train_validation_samples=0,
        seed=16201,
        device=str(args.device),
        amp_precision=str(args.amp_precision),
        augmentation_policy="d4-cyclic",
        resume=str(args.resume or ""),
        preview_samples=int(args.preview_samples),
        preview_only=False,
    )
    code, report_path = run_full_broad(forwarded)
    if code:
        return int(code), report_path

    pointer = _write_latest_pointer(
        repo_root,
        report_path=report_path,
        manifest_path=manifest_path,
        train_authorities=train_authorities,
        crops_per_authority=crops_per_authority,
        train_crop_count=train_crop_count,
        d4_passes=int(args.d4_passes),
        stages=stages,
    )
    latest = json.loads(pointer.read_text(encoding="utf-8"))
    heldout = dict(latest.get("heldout") or {})

    def percent(key: str) -> float:
        value = heldout.get(key)
        return float(value) * 100.0 if value is not None else float("nan")

    print("=" * 86, flush=True)
    print("MAIN RESEARCH TRAINING COMPLETE", flush=True)
    print(f"Final step          : {latest.get('finalStep')}", flush=True)
    print(f"Held global         : {percent('globalRecovery'):+.2f}%", flush=True)
    print(f"Held edge           : {percent('edgeRecovery'):+.2f}%", flush=True)
    print(f"Held gradient       : {percent('gradientRecovery'):+.2f}%", flush=True)
    print(f"Held lattice        : {percent('latticeCellExcess'):+.2f}%", flush=True)
    print(f"Research preview    : {latest.get('previewAlbedoComparison') or latest.get('previewRoot')}", flush=True)
    print(f"Latest pointer      : {pointer}", flush=True)
    print("Production promotion: BLOCKED until all existing gates pass", flush=True)
    return 0, pointer


def main(argv: list[str] | None = None) -> int:
    code, _ = run(parser().parse_args(argv))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
