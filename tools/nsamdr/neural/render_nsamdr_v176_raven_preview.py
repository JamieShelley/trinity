#!/usr/bin/env python3
"""Research-only V17.6 candidate preview on the real EVE Raven (DX11).

Uses authenticated existing comparison-training checkpoints. This is a visual
diagnostic, not an independent validation, qualification, or quadtree inference.
The target HR texture never enters the model inference path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for import_path in (ROOT, HERE, HERE.parent):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import eve_asset_test as eve  # type: ignore
from tools.nsamdr.neural.v176 import raven_renderer as native
from probe_nsamdr_v17_comprehensive import _batch_config, _origin
from probe_nsamdr_v176_capacity import (
    ROOT as COMPARISON_ROOT, STAGES as COMPARISON_STAGES,
    _arm_name, _make_backbone, _tensor_sha, _verify_comparison,
)
from tools.nsamdr.neural.v17.diversity_corpus import sha256_file
from tools.nsamdr.neural.v176.detail_capacity import create_matched_variants

SCHEMA = "NSAMDR_V176_RAVEN_RESEARCH_PREVIEW_V1"
SOURCE_NAME = "V176_RAVEN"
STAGE = COMPARISON_STAGES[-1]


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError(f"invalid JSON record: {path}")
    return result


def _resolve_comparison(repo: Path, authority_index: int, mode: str, explicit: str = "") -> tuple[Path, dict[str, Any], dict[str, Any], Path]:
    """Choose the newest *authenticated* completed training-only comparison."""
    root = repo / COMPARISON_ROOT
    found_errors: list[str] = []
    candidates = ([Path(explicit).resolve()] if explicit else
                  sorted(root.glob("proof_*/report.json"),
                         key=lambda p: p.stat().st_mtime, reverse=True))
    for report_path in candidates:
        try:
            contract = _json(report_path.parent / "run_manifest.json")
            _report, audit = _verify_comparison(report_path, contract)
            provenance = _json(report_path.parent / "provenance.json")
            backbone_path = Path(str(provenance["stage9A12Checkpoint"]))
            if not backbone_path.is_file():
                raise RuntimeError("Stage9 frozen checkpoint missing")
            if sha256_file(backbone_path) != contract["parentStage9CheckpointSha256"]:
                raise RuntimeError("Stage9 frozen checkpoint SHA mismatch")
            if int(contract["trainingAuthorities"][str(authority_index)]["authorityId"], 16) == 0:
                raise RuntimeError("training authority identity missing")
            arm = _arm_name(authority_index, mode)
            checkpoint = report_path.parent / arm / f"stage_{STAGE:06d}" / "checkpoint.pt"
            if sha256_file(checkpoint) != audit["comparisonCheckpointSha256"][arm]:
                raise RuntimeError("selected specialist checkpoint SHA mismatch")
            return report_path, contract, provenance, checkpoint
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            found_errors.append(f"{report_path}: {exc}")
    raise RuntimeError(
        "No authenticated V17.6 compare-features training checkpoint available. "
        "Complete v176-capacity-proof --phase compare-features first. "
        + " | ".join(found_errors[:3])
    )


def _load_preview_model(
    repo: Path, report_path: Path, contract: dict[str, Any],
    provenance: dict[str, Any], checkpoint: Path, device: torch.device,
    authority_index: int, mode: str,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    source = _origin(repo, "")
    if sha256_file(Path(source["manifest"])) != contract["sourceManifestSha256"]:
        raise RuntimeError("original source manifest differs from the frozen comparison")
    config = _batch_config(source)
    backbone = _make_backbone(config, Path(provenance["stage9A12Checkpoint"]), device)
    if _tensor_sha(backbone) != contract["parentBackboneTensorSha256"]:
        raise RuntimeError("frozen backbone tensor SHA changed")
    plain, rich = create_matched_variants(backbone, seed=int(contract["seed"]) + authority_index)
    model = (plain if mode == "output10" else rich).to(device).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (state.get("contract") != contract
        or state.get("arm") != _arm_name(authority_index, mode)
        or state.get("update") != STAGE
        or state.get("architecture") != model.architecture_contract()):
        raise RuntimeError("V17.6 checkpoint architecture or contract mismatch")
    model.specialist.load_state_dict(state["detailState"], strict=True)
    if _tensor_sha(model.backbone) != contract["parentBackboneTensorSha256"]:
        raise RuntimeError("V17.6 model backbone differs from frozen parent")
    return model, config, {
        "schema": state["schema"],
        "sourceComparisonReport": str(report_path.resolve()),
        "sourceComparisonSha256": sha256_file(report_path),
        "sourceMode": mode,
        "authorityIndex": authority_index,
        "researchOnly": True,
        "qualified": False,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    p.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    p.add_argument("--ship-query", default=native.DEFAULT_RAVEN)
    p.add_argument("--authority-index", type=int, choices=(1,10), default=10)
    p.add_argument("--mode", choices=("output10","hr_features58"), default="output10")
    p.add_argument("--comparison-report", default="")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = args.repo_root.resolve()
    device = native._device(args.device)
    report_path, contract, provenance, checkpoint = _resolve_comparison(
        repo, args.authority_index, args.mode, args.comparison_report)
    model, config, payload = _load_preview_model(
        repo, report_path, contract, provenance, checkpoint, device,
        args.authority_index, args.mode,
    )
    if int(config.scale) != 4:
        raise RuntimeError("V17.6 Raven preview requires exact 4x scale")
    print(f"[v176-raven] model={args.mode} authority={args.authority_index} "
          f"checkpoint={checkpoint} (UNQUALIFIED; NATIVE 4X INFERENCE)", flush=True)

    (obj_path, albedo, normal, pgs, environment, environments,
     materials, asset_manifest, catalog, cache_root) = eve.prepare_asset(
        repo, args.shared_cache, args.ship_query, "",
    )
    if materials is None or not materials.is_file():
        raise RuntimeError("Raven physical material manifest unavailable")
    label = f"{SOURCE_NAME}_TRAIN{args.authority_index:02d}_{args.mode.upper()}"
    source = native.PreviewSource(
        label=label, ordinal=STAGE, phase="v176-trained-comparison-research",
        checkpoint=checkpoint,
    )
    report = native._generate_candidate(
        repo_root=repo, source=source,
        device=device, obj_path=obj_path, materials=materials,
        model_override=(model, config, payload), preview_schema=SCHEMA,
    )
    if (report["checkpointSchema"] != payload["schema"]
        or report["checkpointSha256"] != sha256_file(checkpoint)):
        raise RuntimeError("V17.6 preview checkpoint identity changed")
    report.update({
        "researchOnly": True, "productionPromotionAllowed": False,
        "frozenBackboneSha256": contract["parentBackboneTensorSha256"],
        "comparisonReport": str(report_path.resolve()),
        "modelMode": args.mode, "trainingAuthorityIndex": args.authority_index,
        "quadtreeRouting": False,
    })
    Path(report["reportPath"]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    live_dir = repo / "artifacts/nsamdr/render_preview" / label
    pointer_path = live_dir / "current_candidate.txt"
    native._atomic_text(pointer_path, native._pointer_text(report))
    print(f"[v176-raven] PUBLISHED {report['reportPath']}", flush=True)

    os.environ.update({
        "NSAMDR_PREVIEW_EXPERIMENT": label,
        "NSAMDR_PREVIEW_AUTHORITY": "training-intermediate",
        "NSAMDR_LIVE_CANDIDATE_POINTER": str(pointer_path.resolve()),
        "NSAMDR_PREVIEW_REUSE_EXISTING_VIEWER": "1",
    })
    asset = native._read_json(asset_manifest) or {}
    actual_query = str((asset.get("model") or {}).get("logical") or args.ship_query)
    return int(eve.launch_preview(
        repo, repo / "scripts/build/run_nsamdr_obj_preview_dx11.bat",
        obj_path, albedo, normal, pgs, environment, environments,
        materials, asset_manifest, catalog, cache_root, actual_query, None,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
