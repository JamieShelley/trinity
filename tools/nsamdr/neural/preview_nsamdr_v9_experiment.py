#!/usr/bin/env python3
"""Generate, verify, then render one qualified immutable NSAMDR final."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
NSAMDR_ROOT = HERE.parent
for import_root in (HERE, NSAMDR_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import eve_asset_test as eve  # type: ignore
from v9.experiments import (
    experiment_dir,
    load_experiment_manifest,
    load_final_manifest,
    sha256_file,
    write_final_manifest,
)


class ExperimentPreviewApplication:
    # Purpose: Implement utc now for ExperimentPreviewApplication.
    # Called by: main
    # Calls: No same-class helper methods.
    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Purpose: Implement path within for ExperimentPreviewApplication.
    # Called by: _verify_candidate
    # Calls: No same-class helper methods.
    def _path_within(self, path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
            return True
        except ValueError:
            return False

    # Purpose: Implement verified file for ExperimentPreviewApplication.
    # Called by: _verify_candidate
    # Calls: No same-class helper methods.
    def _verified_file(self, path_value: object, sha_value: object, *, label: str) -> tuple[Path, str]:
        path = Path(str(path_value or "")).expanduser().resolve()
        expected = str(sha_value or "").strip().lower()
        if not path.is_file():
            raise RuntimeError(f"{label} is missing: {path}")
        actual = sha256_file(path)
        if len(expected) != 64 or actual != expected:
            raise RuntimeError(
                f"{label} SHA-256 mismatch: expected={expected or '<missing>'} actual={actual} path={path}"
            )
        return path, actual

    # Purpose: Implement parser for ExperimentPreviewApplication.
    # Called by: main
    # Calls: No same-class helper methods.
    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Preview a qualified NSAMDR EXP_#### final")
        parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        parser.add_argument("--experiment", required=True)
        parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
        parser.add_argument("--target-size", type=int, default=4096)
        parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
        return parser

    # Purpose: Implement generate candidate for ExperimentPreviewApplication.
    # Called by: main
    # Calls: No same-class helper methods.
    def _generate_candidate(
        self,
        *,
        root: Path,
        directory: Path,
        checkpoint: Path,
        checkpoint_sha: str,
        args: argparse.Namespace,
        obj_path: Path,
        material_manifest: Path,
        asset_manifest: Path,
    ) -> tuple[dict[str, Any], Path]:
        candidate_root = directory / "previews" / "candidate"
        candidate_root.mkdir(parents=True, exist_ok=True)
        report_path = candidate_root / "candidate_manifest.json"
        final_manifest = directory / "final_manifest.json"
        generator = root / "tools/nsamdr/generate_strategy_candidates.py"
        command = [
            sys.executable,
            "-u",
            str(generator),
            "--obj",
            str(obj_path),
            "--materials",
            str(material_manifest),
            "--asset-manifest",
            str(asset_manifest),
            "--output-root",
            str(candidate_root),
            "--target-size",
            str(args.target_size),
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-sha256",
            checkpoint_sha,
            "--final-manifest",
            str(final_manifest),
            "--inference-device",
            args.device,
        ]
        print("[preview] Generating candidate from the immutable final checkpoint...", flush=True)
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode != 0 or not report_path.is_file():
            raise RuntimeError(f"candidate generation failed with exit code {result.returncode}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise RuntimeError(f"candidate generator wrote an invalid report: {report_path}")
        report["reportPath"] = str(report_path)
        return report, report_path

    # Purpose: Implement verify candidate for ExperimentPreviewApplication.
    # Called by: main
    # Calls: _path_within, _verified_file
    def _verify_candidate(
        self,
        report: dict[str, Any],
        *,
        directory: Path,
        checkpoint: Path,
        checkpoint_sha: str,
    ) -> dict[str, Any]:
        reported_checkpoint = Path(str(report.get("checkpoint") or "")).expanduser().resolve()
        if reported_checkpoint != checkpoint.resolve():
            raise RuntimeError(
                f"candidate used a different checkpoint: expected={checkpoint} actual={reported_checkpoint}"
            )
        reported_sha = str(report.get("checkpointSha256") or "").strip().lower()
        if reported_sha != checkpoint_sha:
            raise RuntimeError(
                f"candidate checkpoint SHA-256 mismatch: expected={checkpoint_sha} actual={reported_sha}"
            )
        candidate_selection = str(report.get("selectionKind") or "").strip()
        if candidate_selection != "production-final":
            raise RuntimeError(
                "candidate checkpoint authority mismatch: expected='production-final' "
                f"actual={candidate_selection!r}"
            )

        provenance = report.get("controlProvenance")
        if not isinstance(provenance, dict) or provenance.get("verified") is not True:
            raise RuntimeError("candidate raw-source provenance is missing or failed")
        primary = provenance.get("primaryAlbedo")
        if not isinstance(primary, dict):
            raise RuntimeError("candidate provenance has no primary albedo binding")
        source_path, source_sha = self._verified_file(
            primary.get("sourcePath"),
            primary.get("sourceSha256After"),
            label="raw source",
        )
        candidate_path, candidate_sha = self._verified_file(
            primary.get("candidatePath"),
            primary.get("candidateSha256"),
            label="NSAMDR candidate",
        )
        if not self._path_within(candidate_path, directory / "previews"):
            raise RuntimeError(f"candidate output escaped the experiment preview directory: {candidate_path}")
        required = (
            "candidateObj",
            "candidateMaterials",
            "candidateAnalysis",
            "candidateValidation",
        )
        missing = [name for name in required if not Path(str(report.get(name) or "")).is_file()]
        if missing:
            raise RuntimeError("candidate report is incomplete: " + ", ".join(missing))

        candidate_root = (directory / "previews" / "candidate").resolve()
        for name in required:
            output = Path(str(report[name])).resolve()
            if not self._path_within(output, candidate_root):
                raise RuntimeError(f"{name} escaped the canonical candidate directory: {output}")

        for record_kind, records, expected_parent in (
            ("candidate", report.get("candidateFiles"), candidate_root),
            ("source", report.get("sourceFiles"), None),
        ):
            if not isinstance(records, list) or not records:
                raise RuntimeError(f"candidate report has no {record_kind} file records")
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise RuntimeError(f"invalid {record_kind} file record at index {index}")
                record_path, _record_sha = self._verified_file(
                    record.get("path"),
                    record.get("sha256"),
                    label=f"{record_kind} file {index}",
                )
                if expected_parent is not None and not self._path_within(record_path, expected_parent):
                    raise RuntimeError(
                        f"candidate file escaped the canonical candidate directory: {record_path}"
                    )
        return {
            "candidatePath": str(candidate_path),
            "candidateSha256": candidate_sha,
            "source": {"sourcePath": str(source_path), "sourceSha256": source_sha},
            "provenanceVerified": True,
            "provenancePath": str(report.get("controlProvenancePath") or ""),
            "reportPath": str(report.get("reportPath") or ""),
        }

    # Purpose: Implement main for ExperimentPreviewApplication.
    # Called by: External callers and the owning workflow.
    # Calls: _generate_candidate, _parser, _utc_now, _verify_candidate
    def main(self, argv: list[str] | None = None) -> int:
        args = self._parser().parse_args(argv)
        root = args.repo_root.resolve()
        experiment_id = args.experiment.strip().upper()
        directory = experiment_dir(root, experiment_id)
        experiment = load_experiment_manifest(root, experiment_id)
        if str(experiment.get("status") or "").lower() != "completed" or experiment.get("qualified") is not True:
            raise RuntimeError(
                f"preview requires a completed qualified experiment, got status={experiment.get('status')!r}"
            )
        final = load_final_manifest(root, experiment_id, require_qualified=True)
        checkpoint = Path(final["_checkpointPath"])
        checkpoint_sha = str(final["checkpoint"]["sha256"]).lower()
        source_selection_kind = str(
            final["checkpoint"].get("sourceSelectionKind")
            or final["checkpoint"].get("selectionKind")
            or ""
        )
        # Re-read immediately before generation so a post-qualification mutation
        # cannot be hidden by the earlier manifest verification.
        if sha256_file(checkpoint) != checkpoint_sha:
            raise RuntimeError("immutable final checkpoint changed before candidate generation")

        asset = experiment.get("asset")
        if not isinstance(asset, dict) or not str(asset.get("query") or "").strip():
            raise RuntimeError(f"experiment has no canonical asset identity: {experiment_id}")
        query = str(asset["query"])
        selection_key = str(asset.get("selectionKey") or "")
        (
            obj_path,
            albedo,
            normal,
            pgs,
            environment,
            environments,
            material_manifest,
            asset_manifest,
            catalog,
            cache_root,
        ) = eve.prepare_asset(root, args.shared_cache, query, selection_key)
        if material_manifest is None or not material_manifest.is_file():
            raise RuntimeError("prepared Raven asset has no physical-map material manifest")

        report, report_path = self._generate_candidate(
            root=root,
            directory=directory,
            checkpoint=checkpoint,
            checkpoint_sha=checkpoint_sha,
            args=args,
            obj_path=obj_path,
            material_manifest=material_manifest,
            asset_manifest=asset_manifest,
        )
        candidate = self._verify_candidate(
            report,
            directory=directory,
            checkpoint=checkpoint,
            checkpoint_sha=checkpoint_sha,
        )
        if sha256_file(checkpoint) != checkpoint_sha:
            raise RuntimeError("immutable final checkpoint changed during candidate generation")

        previews = directory / "previews"
        preview_manifest_path = previews / "preview_manifest.json"
        preview_record: dict[str, Any] = {
            "schema": "NSAMDR_PRODUCTION_PREVIEW_V1",
            "experiment": experiment_id,
            "status": "candidate-verified",
            "createdUtc": self._utc_now(),
            "checkpoint": str(checkpoint),
            "checkpointSha256": checkpoint_sha,
            "selectionKind": "production-final",
            "sourceSelectionKind": source_selection_kind,
            "candidate": candidate,
            "source": candidate["source"],
            "controlProvenance": report["controlProvenance"],
            "candidateFiles": report["candidateFiles"],
            "sourceFiles": report["sourceFiles"],
            "asset": asset,
            "targetSize": args.target_size,
            "device": args.device,
            "candidateReport": str(report_path),
            "provenanceVerifiedBeforeRenderer": True,
        }
        preview_manifest_path.write_text(
            json.dumps(preview_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        final.pop("_checkpointPath", None)
        final["candidate"] = candidate
        final["renderer"] = {"status": "candidate-verified", "verifiedUtc": self._utc_now()}
        write_final_manifest(directory / "final_manifest.json", final)

        # The contract consumes the generated manifest but runs before the native
        # process. A mismatch therefore cannot reach the A/B renderer.
        preview_contract = directory / "evidence" / "preview_preflight.json"
        contract_command = [
            sys.executable,
            "-u",
            str(root / "tools/nsamdr/neural/raven_architecture_contract.py"),
            "preview",
            "--repo-root",
            str(root),
            "--experiment-dir",
            str(directory),
            "--output",
            str(preview_contract),
        ]
        contract_result = subprocess.run(contract_command, cwd=root, check=False)
        if contract_result.returncode != 0:
            raise RuntimeError(
                f"preview provenance contract failed with exit code {contract_result.returncode}"
            )

        os.environ.update(
            {
                "NSAMDR_PREVIEW_EXPERIMENT": experiment_id,
                "NSAMDR_PREVIEW_CHECKPOINT": str(checkpoint),
                "NSAMDR_PREVIEW_CHECKPOINT_SHA256": checkpoint_sha,
                "NSAMDR_PREVIEW_AUTHORITY": "production-final",
                "NSAMDR_FINAL_MANIFEST": str(directory / "final_manifest.json"),
            }
        )
        selected_query = str(
            json.loads(asset_manifest.read_text(encoding="utf-8")).get("model", {}).get("logical")
            or query
        )
        selected_catalog_key = selection_key or selected_query
        launcher = root / "scripts/build/run_nsamdr_obj_preview_dx11.bat"
        return_code = eve.launch_preview(
            root,
            launcher,
            obj_path,
            albedo,
            normal,
            pgs,
            environment,
            environments,
            material_manifest,
            asset_manifest,
            catalog,
            cache_root,
            selected_catalog_key,
            report,
        )
        preview_record["status"] = "launched" if return_code == 0 else "renderer-failed"
        preview_record["rendererReturnCode"] = return_code
        preview_record["completedUtc"] = self._utc_now()
        preview_manifest_path.write_text(
            json.dumps(preview_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        final["renderer"] = {
            "status": "launched" if return_code == 0 else "failed",
            "returnCode": return_code,
            "completedUtc": self._utc_now(),
        }
        write_final_manifest(directory / "final_manifest.json", final)
        if return_code != 0:
            raise RuntimeError(f"native renderer failed with exit code {return_code}")
        print(f"[preview] Renderer launched from immutable checkpoint SHA-256 {checkpoint_sha}", flush=True)
        return 0

_experiment_preview_application = ExperimentPreviewApplication()
_utc_now = _experiment_preview_application._utc_now
_path_within = _experiment_preview_application._path_within
_verified_file = _experiment_preview_application._verified_file
_parser = _experiment_preview_application._parser
_generate_candidate = _experiment_preview_application._generate_candidate
_verify_candidate = _experiment_preview_application._verify_candidate
main = _experiment_preview_application.main


if __name__ == "__main__":
    raise SystemExit(main())
