#!/usr/bin/env python3
"""Launch the real Raven Navy Issue Mode 1/2/3 preview for one experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v9.experiments import experiment_dir, load_experiment_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview a completed NSAMDR V9 tuning experiment")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
    parser.add_argument("--target-size", type=int, default=4096)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--force-candidate", action="store_true")
    parser.add_argument("--geometry-audit", dest="geometry_audit", action="store_true", default=True)
    parser.add_argument("--no-geometry-audit", dest="geometry_audit", action="store_false")
    parser.add_argument("--geometry-critic", choices=("off", "auto", "required"), default="auto")
    parser.add_argument("--geometry-audit-policy", choices=("report", "strict"), default="report")
    parser.add_argument("--geometry-evidence-regions", type=int, default=12)
    parser.add_argument("--critic-checkpoint", type=Path)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    experiment_id = args.experiment.strip().upper()
    directory = experiment_dir(root, experiment_id)
    checkpoint = directory / "nsamdr_v9_fidelity.pt"
    capabilities_path = directory / "capabilities.json"
    if not checkpoint.is_file() or not capabilities_path.is_file():
        raise RuntimeError(f"experiment is not completed/previewable: {experiment_id}")
    capabilities = json.loads(capabilities_path.read_text(encoding="utf-8"))
    experiment_manifest = load_experiment_manifest(root, experiment_id)
    training_mode = str(experiment_manifest.get("trainingMode") or "full").lower()
    training_safety_pass = bool(experiment_manifest.get("trainingSafetyPass", experiment_manifest.get("acceptancePass")))
    acceptance_pass = bool(experiment_manifest.get("reconstructionAcceptancePass", False))
    acceptance_regression = experiment_manifest.get("acceptanceRegressionFraction")
    metadata_path = directory / "nsamdr_v9_fidelity.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        training_safety_pass = bool(metadata.get("trainingSafetyPass", metadata.get("acceptancePass")))
        acceptance_pass = bool(metadata.get("reconstructionAcceptancePass", False))
        acceptance_regression = metadata.get("acceptanceRegressionFraction")
    assets = capabilities.get("supportedAssets", [])
    if not assets:
        raise RuntimeError(f"experiment has no supported preview asset: {experiment_id}")
    asset = assets[0]
    query = str(asset.get("query") or "")
    selection_key = str(asset.get("selectionKey") or "")

    previews = directory / "previews"
    previews.mkdir(parents=True, exist_ok=True)
    candidate_pointer = previews / "last_candidate_result.json"
    candidate_pointer.unlink(missing_ok=True)
    critic_checkpoint = (args.critic_checkpoint or (root / "artifacts/nsamdr/geometry_critic/geometry_pair_critic.pt")).resolve()

    env = os.environ.copy()
    env.update(
        {
            "NSAMDR_V9_PREVIEW_STRENGTH": "1.0",
            "NSAMDR_NEURAL_ARCHITECTURE": "V9",
            "NSAMDR_NEURAL_CHECKPOINT_DIR": str(directory),
            "NSAMDR_INFERENCE_DEVICE": args.device,
            "NSAMDR_MODE3_TARGET_SIZE": str(args.target_size),
            "NSAMDR_MODE3_CANDIDATE_TAG": experiment_id.lower(),
            "NSAMDR_GEOMETRY_AUDIT": "1" if args.geometry_audit else "0",
            "NSAMDR_GEOMETRY_CRITIC": args.geometry_critic,
            "NSAMDR_GEOMETRY_AUDIT_POLICY": args.geometry_audit_policy,
            "NSAMDR_GEOMETRY_AUDIT_EVIDENCE": str(max(0, args.geometry_evidence_regions)),
            "NSAMDR_GEOMETRY_CRITIC_CHECKPOINT": str(critic_checkpoint),
            "NSAMDR_GEOMETRY_CRITIC_DEVICE": args.device,
            "NSAMDR_CANDIDATE_RESULT_FILE": str(candidate_pointer),
        }
    )
    if args.force_candidate:
        env["NSAMDR_FORCE_CANDIDATE"] = "1"

    preview_record = {
        "schema": "NSAMDR_V9_EXPERIMENT_PREVIEW_V1",
        "experiment": experiment_id,
        "createdUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "asset": asset,
        "targetSize": args.target_size,
        "device": args.device,
        "candidateTag": experiment_id.lower(),
        "evaluation": {
            "policy": "native-source-4x",
            "targetSize": args.target_size,
            "expectedModelInputSize": args.target_size // 4,
        },
        "qualityGate": {
            "trainingMode": training_mode,
            "trainingSafetyPass": training_safety_pass,
            "reconstructionAcceptancePass": acceptance_pass,
            "acceptanceRegressionFraction": acceptance_regression,
        },
        "geometryAuditConfiguration": {
            "enabled": args.geometry_audit,
            "critic": args.geometry_critic,
            "policy": args.geometry_audit_policy,
            "evidenceRegions": args.geometry_evidence_regions,
            "criticCheckpoint": str(critic_checkpoint),
        },
    }

    helper = root / "tools/nsamdr/eve_asset_test.py"
    launcher = root / "scripts/build/run_nsamdr_obj_preview_dx11.bat"
    command = [
        sys.executable,
        str(helper),
        "prepare-run",
        "--repo-root",
        str(root),
        "--shared-cache",
        args.shared_cache,
        "--query",
        query,
        "--selection-key",
        selection_key,
        "--neural-checkpoint-dir",
        str(directory),
        "--launcher",
        str(launcher),
    ]
    print(f"[preview] Experiment: {experiment_id}", flush=True)
    print(f"[preview] Asset: {asset.get('displayName')} ({selection_key})", flush=True)
    print(f"[preview] Checkpoint: {checkpoint}", flush=True)
    return_code = subprocess.run(command, cwd=root, env=env, check=False).returncode
    if return_code == 0:
        preview_record["status"] = "completed"
        preview_record["completedUtc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        real_audit: dict[str, object] = {}
        if candidate_pointer.is_file():
            candidate_result = json.loads(candidate_pointer.read_text(encoding="utf-8"))
            real_audit = candidate_result.get("geometryAudit", {}) if isinstance(candidate_result, dict) else {}
            if isinstance(real_audit, dict) and real_audit.get("jsonPath"):
                source_audit_dir = Path(str(real_audit["jsonPath"])).resolve().parent
                local_audit_dir = previews / "real_geometry_audit"
                if local_audit_dir.exists():
                    shutil.rmtree(local_audit_dir)
                if source_audit_dir.is_dir():
                    shutil.copytree(source_audit_dir, local_audit_dir)
                    local_json = local_audit_dir / "geometry_audit.json"
                    if local_json.is_file():
                        real_audit = json.loads(local_json.read_text(encoding="utf-8"))
                        real_audit["localPath"] = str(local_json)
            preview_record["candidateResult"] = candidate_result
            provenance = candidate_result.get("controlProvenance", {}) if isinstance(candidate_result, dict) else {}
            provenance_path = Path(str(candidate_result.get("controlProvenancePath", ""))) if isinstance(candidate_result, dict) else Path()
            if provenance_path.is_file():
                local_provenance = previews / "preview_control_provenance.json"
                shutil.copy2(provenance_path, local_provenance)
                preview_record["controlProvenancePath"] = str(local_provenance)
                preview_record["controlProvenance"] = provenance
        preview_record["realGeometryAudit"] = real_audit

        synthetic_path = previews / "synthetic_geometry_audit" / "synthetic_geometry_audit.json"
        synthetic_audit: dict[str, object] = {}
        if synthetic_path.is_file():
            synthetic_audit = json.loads(synthetic_path.read_text(encoding="utf-8"))
        preview_record["syntheticGeometryAudit"] = synthetic_audit
        audit_verdicts = [
            str(payload.get("verdict")) for payload in (synthetic_audit, real_audit)
            if isinstance(payload, dict) and payload.get("verdict")
        ]
        staged_failures = ("RENDERER_FAIL", "SDF_FAIL", "GATE_FAIL", "TOPOLOGY_FAIL", "FUZZ_FAIL", "HALO_FAIL", "DOUBLE_EDGE_FAIL", "PBR_ALIGNMENT_FAIL")
        combined_verdict = (
            "FAIL" if "FAIL" in audit_verdicts else
            next((v for v in staged_failures if v in audit_verdicts), None)
            or ("PASS" if audit_verdicts and all(v == "PASS" for v in audit_verdicts) else
                "NEUTRAL_BOUNDARY_INACTIVE"
                if any(v == "NEUTRAL_BOUNDARY_INACTIVE" for v in audit_verdicts) else
                "NEUTRAL_NO_NET_GAIN")
        )
        preview_record["combinedGeometryAuditVerdict"] = combined_verdict
        reconstruction_pass = combined_verdict == "PASS"
        preview_record["qualityGate"]["reconstructionAcceptancePass"] = reconstruction_pass
        manifest_file = directory / "experiment.json"
        if manifest_file.is_file():
            manifest_payload = json.loads(manifest_file.read_text(encoding="utf-8"))
            manifest_payload["trainingSafetyPass"] = training_safety_pass
            manifest_payload["reconstructionAcceptancePass"] = reconstruction_pass
            manifest_payload["combinedGeometryAuditVerdict"] = combined_verdict
            manifest_file.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        feedback_bundle = previews / f"{experiment_id}_geometry_feedback.zip"
        preview_record["feedbackBundle"] = str(feedback_bundle)
        manifest_path = previews / "preview_manifest.json"
        manifest_path.write_text(
            json.dumps(preview_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with zipfile.ZipFile(feedback_bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for source in (
                manifest_path, directory / "experiment.json", directory / "resolved_config.json",
                directory / "metrics.json", directory / "training_log.csv", candidate_pointer,
                previews / "preview_control_provenance.json",
            ):
                if source.is_file():
                    archive.write(source, arcname=source.name)
            for folder_name in ("synthetic_geometry_audit", "real_geometry_audit"):
                folder = previews / folder_name
                if folder.is_dir():
                    for source in sorted(folder.rglob("*")):
                        if source.is_file():
                            archive.write(source, arcname=str(Path(folder_name) / source.relative_to(folder)))
        provenance_summary = preview_record.get("controlProvenance", {})
        print(
            f"[preview] Control provenance: "
            f"{'VERIFIED' if isinstance(provenance_summary, dict) and provenance_summary.get('verified') else 'MISSING/FAILED'}",
            flush=True,
        )
        print(f"[preview] Geometry audit verdict: {combined_verdict}", flush=True)
        if synthetic_audit:
            renderer = synthetic_audit.get("rendererProof", {}) if isinstance(synthetic_audit, dict) else {}
            sdf_proof = synthetic_audit.get("sdfProof", {}) if isinstance(synthetic_audit, dict) else {}
            gate_proof = synthetic_audit.get("gateProof", {}) if isinstance(synthetic_audit, dict) else {}
            print(
                f"[preview] Staged analytic proof: {synthetic_audit.get('verdict')} | "
                f"renderer={float(renderer.get('chamferImprovementMean', 0.0)):+.2%} "
                f"sdf={float(sdf_proof.get('chamferImprovementMean', 0.0)):+.2%} "
                f"final={float(gate_proof.get('chamferImprovementMean', 0.0)):+.2%}",
                flush=True,
            )
        if real_audit:
            print(f"[preview] Real Raven texture audit: {real_audit.get('verdict')} | proxy={float(real_audit.get('proxyGeometryImprovementMean', 0.0)):+.4f}", flush=True)
        print(f"[preview] Feedback bundle: {feedback_bundle}", flush=True)

        if training_mode == "full" and reconstruction_pass:
            print(f"[preview] Full proof preview + reconstruction acceptance PASS: {manifest_path}", flush=True)
        elif training_mode == "full":
            print("[preview] Renderer completed, but reconstruction acceptance FAILED; promotion remains locked.", flush=True)
        else:
            print("[preview] Quick renderer preview completed; promotion remains locked by design.", flush=True)
    else:
        print(f"[preview] Preview failed with exit code {return_code}; promotion remains locked.", flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
