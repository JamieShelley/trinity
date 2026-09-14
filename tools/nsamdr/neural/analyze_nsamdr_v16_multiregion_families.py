#!/usr/bin/env python3
"""Evaluate an existing V16 Stage 2 checkpoint by authored texture family.

This is an evaluation-only diagnostic. It does not train, rebuild data, or alter a
checkpoint. It verifies that the current dataset matches the source run before it
evaluates the selected checkpoint on the exact train and held-out records recorded
in that run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
NEURAL_ROOT = HERE
if str(NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURAL_ROOT))

from v14 import multifamily_multiregion_diagnostic as multifamily
from v14 import multiregion_diagnostic as base
from v14.config import V16Config
from v14.dataset import load_manifest
from v14.diagnostic_support import device_from_name


SCHEMA = "NSAMDR_V16_MULTIREGION_FAMILY_TELEMETRY_V1"


def _normal_path(value: str | Path) -> str:
    return str(Path(value).resolve()).replace("\\", "/").casefold()


def _latest_run(repo_root: Path) -> Path:
    root = repo_root / "artifacts/nsamdr/diagnostics/v16_mini"
    candidates = sorted(
        (
            path
            for path in root.glob("multiregion_*")
            if path.is_dir() and (path / "report.json").is_file()
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        raise RuntimeError(f"No V16 multi-region diagnostic runs found under {root}")
    return candidates[-1]


def _resolve_run(repo_root: Path, raw: str | None) -> tuple[Path, Path]:
    if not raw:
        run_dir = _latest_run(repo_root)
        return run_dir, run_dir / "report.json"
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if path.is_dir():
        report_path = path / "report.json"
        run_dir = path
    else:
        report_path = path
        run_dir = path.parent
    if not report_path.is_file():
        raise RuntimeError(f"Stage 2 report not found: {report_path}")
    return run_dir, report_path


def _record_lookup(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_path: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for record in manifest.get("crops", []):
        if not isinstance(record, dict) or not record.get("path"):
            continue
        by_path[_normal_path(str(record["path"]))] = record
        by_name[Path(str(record["path"])).name.casefold()] = record
    return by_path, by_name


def _resolve_records(
    paths: list[str],
    manifest: dict[str, Any],
    *,
    split: str,
) -> list[dict[str, Any]]:
    by_path, by_name = _record_lookup(manifest)
    result: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw in paths:
        record = by_path.get(_normal_path(raw))
        if record is None:
            record = by_name.get(Path(raw).name.casefold())
        if record is None or str(record.get("split") or "") != split:
            missing.append(raw)
            continue
        result.append(record)
    if missing:
        raise RuntimeError(
            f"Current dataset cannot resolve {split} records from the source run: {missing}"
        )
    return result


def _checkpoint(run_dir: Path, report: dict[str, Any]) -> Path:
    raw = str(report.get("candidateCheckpoint") or "").strip()
    candidate = Path(raw) if raw else run_dir / "best_checkpoint.pt"
    if candidate.is_file():
        return candidate.resolve()
    fallback = run_dir / "best_checkpoint.pt"
    if fallback.is_file():
        return fallback.resolve()
    raise RuntimeError(f"Selected Stage 2 checkpoint is missing: {candidate}")


def _compact(report: dict[str, object]) -> dict[str, object]:
    keys = (
        "medianGlobalRecovery",
        "medianEdgeRecovery",
        "medianGradientRecovery",
        "medianDetailRecovery1px",
        "medianDetailRecovery2px",
        "medianDetailRecovery4px",
        "medianNormalRecovery",
        "medianMaterialRecovery",
        "positiveGlobalFraction",
        "positiveEdgeFraction",
        "worstGlobalRecovery",
        "maxLatticeCellExcess",
        "sampleCount",
        "perFamily",
    )
    return {key: report.get(key) for key in keys if key in report}


def run(args: argparse.Namespace) -> Path:
    repo_root = args.repo_root.resolve()
    run_dir, report_path = _resolve_run(repo_root, args.run)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("mode") != "multiregion":
        raise RuntimeError(f"Not a V16 multi-region report: {report_path}")

    config = V16Config(
        minimum_heldout_samples=4,
        candidate_edge_recovery_required=float(args.required_edge_recovery),
        candidate_global_recovery_required=float(args.required_global_recovery),
        candidate_gradient_recovery_required=float(args.required_gradient_recovery),
    )
    config.validate()
    manifest = load_manifest(repo_root, config)
    expected_fingerprint = str(report.get("datasetFingerprint") or "")
    actual_fingerprint = str(manifest.get("fingerprint") or "")
    if expected_fingerprint and actual_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Current V16 dataset does not match the source Stage 2 run: "
            f"report={expected_fingerprint} current={actual_fingerprint}"
        )

    train_records = _resolve_records(
        [str(value) for value in report.get("trainRecords", [])],
        manifest,
        split="train",
    )
    validation_records = _resolve_records(
        [str(value) for value in report.get("validationRecords", [])],
        manifest,
        split="validation",
    )
    if not train_records or not validation_records:
        raise RuntimeError("Source Stage 2 run does not contain train and held-out records")

    device = device_from_name(args.device)
    model, _checkpoint_payload = base.load_checkpoint(
        _checkpoint(run_dir, report),
        device,
    )
    parser_args = base.parser().parse_args([])
    parser_args.device = args.device
    parser_args.amp_precision = args.amp_precision
    parser_args.required_edge_recovery = float(args.required_edge_recovery)
    parser_args.required_global_recovery = float(args.required_global_recovery)
    parser_args.required_gradient_recovery = float(args.required_gradient_recovery)
    diagnostic = base.MultiRegionDiagnostic(parser_args, repo_root, device)

    print(
        f"[v16-family-analysis] Source run: {report_path}",
        flush=True,
    )
    print(
        f"[v16-family-analysis] Selected step: {report.get('selectedStep')} "
        f"checkpoint={_checkpoint(run_dir, report)}",
        flush=True,
    )
    train_report = diagnostic._evaluate_records(model, train_records, config)
    validation_report = diagnostic._evaluate_records(model, validation_records, config)

    payload = {
        "schema": SCHEMA,
        "sourceReport": str(report_path.resolve()),
        "selectedStep": report.get("selectedStep"),
        "checkpoint": str(_checkpoint(run_dir, report)),
        "datasetFingerprint": actual_fingerprint,
        "train": _compact(train_report),
        "heldOut": _compact(validation_report),
    }
    output_path = run_dir / "family_metrics.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[v16-family-analysis] Report: {output_path}", flush=True)
    return output_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate an existing V16 Stage 2 checkpoint by authored family"
    )
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument(
        "--run",
        default=None,
        help="Stage 2 run directory or report.json; default is the latest run",
    )
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--amp-precision", choices=("auto", "bf16", "fp16"), default="auto")
    p.add_argument("--required-edge-recovery", type=float, default=0.60)
    p.add_argument("--required-global-recovery", type=float, default=0.45)
    p.add_argument("--required-gradient-recovery", type=float, default=0.35)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
