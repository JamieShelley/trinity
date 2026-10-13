"""Stage 9 provenance and matched A12/B64 continuation decisions.

No pixel decoding here. The authoritative source is the *trained* Stage 8
frozen_selection.json, never a newly discovered or reranked authority roster.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .diversity_corpus import sha256_file, sha256_json, verify_frozen_selection

SCHEMA = "NSAMDR_V17_5_MATCHED_CONTINUATION_PROVENANCE_V1"
STAGE8_SCHEMA = "NSAMDR_V17_5_NATIVE_AUTHORITY_DIVERSITY_PROOF_V1"
UPDATES = (36864, 49152)
ARMS = ("A12", "B64")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_stage8_report(repo: Path, specified: str) -> Path:
    if specified:
        path = Path(specified).resolve()
        if not path.is_file():
            raise RuntimeError(f"specified Stage 8 report missing: {path}")
        return path
    root = repo / "artifacts/nsamdr/diagnostics/v17_5_diversity"
    completed = []
    for path in root.glob("proof_*/report.json"):
        try:
            item = _read(path)
        except (OSError, ValueError):
            continue
        if (item.get("schema") == STAGE8_SCHEMA
            and item.get("selectionOnly") is not True
            and set(item.get("armResults") or {}) == set(ARMS)
            and item.get("trainingUpdates") == 49152):
            completed.append(path)
    if not completed:
        raise RuntimeError("no completed A12/B64 Stage 8 report; supply --stage8-report")
    return max(completed, key=lambda path: path.stat().st_mtime).resolve()


def _identity(selection: dict) -> dict:
    return {
        "controlIds": selection["controlTrainingCropIds"],
        "expandedIds": selection["expandedTrainingCropIds"],
        "developmentIds": selection["developmentCropIds"],
        "sealedIds": selection["sealedFinalCropIds"],
        "sourceHashes": [{
            "crop": x["cropId"],
            "albedo": x["albedo"]["sha256"],
            "normal": x["normal"]["sha256"],
            "cropBundle": x["cropBundleSha256"],
        } for x in selection["selectedAuthoritySources"]],
    }


def compare_selection_preview(
    selection: dict, stage8_report: Path, *, specified: str = ""
) -> dict[str, Any]:
    """Report dry-run drift; training roster remains anchored to Stage 8.

    A mismatch in an earlier selection-only run does not invalidate the
    internal trained A12/B64 pair. It must nevertheless be disclosed.
    """
    if specified:
        preview_path = Path(specified).resolve()
        if not preview_path.is_file():
            raise RuntimeError(f"specified selection-only report missing: {preview_path}")
    else:
        preview_path = None
        candidates = []
        for path in stage8_report.parent.parent.glob("proof_*/report.json"):
            if path.resolve() == stage8_report.resolve():
                continue
            try:
                data = _read(path)
                if (data.get("selectionOnly") is True and
                    data.get("schema") == STAGE8_SCHEMA and
                    data.get("runContract",{}).get("sourceManifestSha256")
                    == selection["manifestSha256"]):
                    candidates.append(path)
            except (OSError, KeyError, ValueError):
                continue
        prior = [p for p in candidates if p.stat().st_mtime <= stage8_report.stat().st_mtime]
        if prior:
            preview_path = max(prior, key=lambda p:p.stat().st_mtime).resolve()
    if preview_path is None:
        return {"status":"NOT_AVAILABLE","trainedRosterAuthenticated":True,
                "interpretation":"No prior selection-only proof available for comparison"}
    preview = _read(preview_path)
    if preview.get("schema") != STAGE8_SCHEMA or preview.get("selectionOnly") is not True:
        raise RuntimeError("explicit preview was not a Stage 8 selection-only proof")
    roster_path = preview_path.parent / "frozen_selection.json"
    if not roster_path.is_file():
        return {"status":"PREVIEW_ROSTER_MISSING","previewReport":str(preview_path),
                "trainedRosterAuthenticated":True}
    frozen = _read(roster_path)
    left = _identity(selection)
    right = _identity(frozen)
    drift = [field for field in left if left[field] != right[field]]
    return {
        "status":("MATCH" if not drift else "MISMATCH_ROSTER_OR_SOURCE"),
        "previewReport":str(preview_path),
        "previewSelectionSha256":sha256_json(frozen),
        "trainedSelectionSha256":sha256_json(selection),
        "differingIdentityFields":drift,
        "metadataOnlyDifference":not drift and sha256_json(frozen)!=sha256_json(selection),
        "trainedRosterAuthenticated":True,
        "interpretation":"Only the Stage 8 training roster is continued; preview drift never silently changes it",
    }


def verify_stage8(
    report_path: Path, source: dict, manifest: dict, parts: dict,
    *, preview_path: str = ""
) -> tuple[dict, dict, dict]:
    """Verify source lineage, the A12/B64 arm reports, and both start checkpoints."""
    report = _read(report_path)
    if (report.get("schema") != STAGE8_SCHEMA or
        report.get("selectionOnly") is True or
        report.get("finalTestEvaluated") is not False or
        report.get("finalTestEvaluationCount") != 0 or
        report.get("productionPromotionAllowed") is not False or
        report.get("trainingUpdates") != 49152 or
        set(report.get("armResults") or {}) != set(ARMS)):
        raise RuntimeError("Stage 8 must be completed research-only with sealed final")
    root = report_path.parent
    selection_path = root / "frozen_selection.json"
    contract_path = root / "run_manifest.json"
    selection = _read(selection_path)
    stage8_contract = _read(contract_path)
    verify_frozen_selection(selection, source, manifest, parts, verify_bytes=True)
    digest = sha256_json(selection)
    if (report.get("frozenSelectionSha256") != digest or
        report.get("runContract") != stage8_contract or
        stage8_contract.get("selectionSha256") != digest or
        stage8_contract.get("trainingSeed") != 17301 or
        stage8_contract.get("stages") != [6144, 12288, 24576] or
        stage8_contract.get("device") != "cuda" or
        stage8_contract.get("sealedFinalEnabled") is not False):
        raise RuntimeError("Stage 8 run contract or training selection fingerprint mismatch")
    checkpoints = {}
    sha_map = {}
    for arm in ARMS:
        row = report["armResults"][arm]
        path = root / arm / "stage_024576" / "checkpoint.pt"
        if (row.get("arm") != arm or row.get("optimizerUpdates") != 24576 or
            row.get("runContractSha256") != sha256_json(stage8_contract) or
            [x.get("update") for x in row.get("history") or []] !=
            [6144, 12288, 24576] or
            row.get("trainingAuthorities") != [
                x["familyId"] for x in selection["selectedAuthoritySources"][
                    :12 if arm == "A12" else 64
                ]
            ] or not path.is_file()):
            raise RuntimeError(f"Stage 8 {arm} staged proof and roster invalid")
        checkpoints[arm] = str(path.resolve())
        sha_map[arm] = sha256_file(path)
    audit = {
        "schema":SCHEMA,
        "stage8Report":str(report_path.resolve()),
        "stage8ReportSha256":sha256_file(report_path),
        "stage8RunManifestSha256":sha256_file(contract_path),
        "trainedSelectionSha256":digest,
        "trainedSelectionFileSha256":sha256_file(selection_path),
        "stage8CheckpointPaths":checkpoints,
        "stage8CheckpointSha256":sha_map,
        "previewComparison":compare_selection_preview(
            selection, report_path, specified=preview_path),
        "trainingAuthorities":[
            x["familyId"] for x in selection["selectedAuthoritySources"]],
        "developmentCropIds":selection["developmentCropIds"],
        "sealedFinalCropIds":selection["sealedFinalCropIds"],
        "sourceDataReverified":True,
        "finalTestPixelsOpened":0,
    }
    return selection, stage8_contract, audit


def _values(evaluation: dict, metric: str) -> list[float]:
    return [float(row["metrics"][metric]) for row in evaluation["perCrop"]]


def probe_groups(evaluation: dict, arm: str) -> dict:
    count = 12 if arm == "A12" else 64
    records = evaluation["perCrop"]
    if len(records) != count:
        raise RuntimeError(f"{arm} fixed training probe count {len(records)} not {count}")
    def describe(rows: list[dict]) -> dict:
        keys = ("edge_recovery","global_recovery","gradient_recovery",
                "detail_recovery_1px","lattice_cell_excess")
        return {"count":len(rows),
                "mean":{k:sum(float(r["metrics"][k]) for r in rows)/len(rows) for k in keys},
                "allGatesPassedCount":sum(all(r.get("gateChecks",{}).values()) for r in rows)}
    groups = {"all":describe(records),"commonA12":describe(records[:12])}
    if arm == "B64":
        groups["additional52"] = describe(records[12:])
    return groups


def diagnostic_summary(evaluation: dict) -> dict:
    metrics = ("residual_cosine_similarity","target_weighted_sign_agreement",
               "candidate_phase_energy_spread","target_phase_energy_spread",
               "candidate_to_target_residual_ratio")
    result = []
    for i,row in enumerate(evaluation["perCrop"]):
        evidence = row.get("residualDiagnostics") or {}
        score = {"index":i}
        for metric in metrics:
            score[metric] = float(evidence[metric]) if metric in evidence else None
        score["edgeRecovery"] = float(row["metrics"]["edge_recovery"])
        score["latticeExcess"] = float(row["metrics"]["lattice_cell_excess"])
        score["gateChecks"] = row.get("gateChecks")
        result.append(score)
    return {"perCrop":result}


def decide(matched: list[dict], final_results: dict) -> str:
    current = matched[-1]
    if any(result["atFinalUpdate"]["trained"]["allPass"] and
           result["atFinalUpdate"]["validation"]["allPass"]
           for result in final_results.values()):
        return "VALIDATION_QUALIFIED_RESEARCH_ONLY"
    if (current["B64WorstEdge"] - current["A12WorstEdge"] >= .03 and
        current["B64MeanEdge"] - current["A12MeanEdge"] >= .02 and
        current["B64MaxLattice"] <= .15 and
        all(value >= -.01 for value in current["globalDeltaByCrop"])):
        return "EXPOSURE_BENEFIT_SIGNAL"
    if (current["B64MeanEdge"] > matched[0]["B64MeanEdge"] and
        current["B64MaxLattice"] > .15):
        return "RECOVERY_LATTICE_CONFLICT"
    common = final_results["B64"]["atFinalUpdate"]["trainProbeGroups"]["commonA12"]["mean"]["edge_recovery"]
    control = final_results["A12"]["atFinalUpdate"]["trainProbeGroups"]["commonA12"]["mean"]["edge_recovery"]
    if control-common > .05:
        return "PERSISTENT_COMMON_AUTHORITY_UNDERFIT"
    return "NO_COMPELLING_CONTINUATION_BENEFIT"
