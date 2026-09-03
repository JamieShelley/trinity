#!/usr/bin/env python3
"""Live EVE A/B/C preview of deterministic baseline and completed NSAMDR stages."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
NSAMDR_ROOT = HERE.parent
for import_root in (HERE, NSAMDR_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch
import eve_asset_test as eve  # type: ignore
from generate_strategy_candidates import SEMANTICS, StrategyCandidateGenerator  # type: ignore
from v9.experiments import experiment_dir, load_experiment_manifest, load_resolved_config
from v9.inference import resolve_device
from v9.model import MODEL_SCHEMA, FidelityResidualNetV9


class LiveTrainingPreviewApplication:
    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Hot-reload the real EVE renderer from completed NSAMDR training epochs")
        parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        parser.add_argument("--experiment", required=True)
        parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
        parser.add_argument("--target-size", type=int, default=1024)
        parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
        parser.add_argument("--poll-seconds", type=float, default=0.75)
        return parser

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, ValueError, TypeError):
            return None

    def _atomic_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", mode="w", encoding="utf-8", delete=False
        ) as temporary:
            temporary.write(text)
            temporary.flush()
            temp_path = Path(temporary.name)
        os.replace(temp_path, path)

    def _load_epoch_model(self, checkpoint: Path, config, device: torch.device, epoch: int):
        try:
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location=device)
        if not isinstance(payload, dict) or str(payload.get("schema") or "") != MODEL_SCHEMA:
            raise RuntimeError(f"live checkpoint schema mismatch: {checkpoint}")
        completed_epoch = int(payload.get("completed_epoch", 0))
        if completed_epoch != int(epoch):
            raise RuntimeError(
                f"live checkpoint epoch mismatch: pointer={epoch} payload={completed_epoch}"
            )
        model = FidelityResidualNetV9(config).to(device)
        if device.type == "cuda" and config.channels_last:
            converter = getattr(torch.nn.utils, "convert_conv2d_weight_memory_format", None)
            model = converter(model, torch.channels_last) if converter is not None else model.to(memory_format=torch.channels_last)
        model.load_state_dict(payload["state_dict"], strict=True)
        model._checkpoint_selection_kind = f"training-intermediate-epoch-{epoch}"
        model.eval()
        return model

    def _write_material_manifest(
        self,
        helper: StrategyCandidateGenerator,
        output: Path,
        fields: list[str],
        rows: list[dict[str, str]],
        comments: list[str],
        replacements: Mapping[Path, Path],
        source_dir: Path,
        physical_candidate: str = "NSAMDR_TRAINING_INTERMEDIATE_UNQUALIFIED",
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            handle.write("# NSAMDR_MATERIALS_V7\n")
            handle.write(f"# PHYSICAL_CANDIDATE {physical_candidate}\n")
            for comment in comments:
                if comment.startswith("#") and "NSAMDR_MATERIALS" not in comment and "PHYSICAL_CANDIDATE" not in comment:
                    handle.write(comment + "\n")
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            for row in rows:
                adjusted = dict(row)
                for semantic in SEMANTICS:
                    source = helper._resolve_path(row.get(semantic, ""), source_dir)
                    if source in replacements:
                        adjusted[semantic] = str(replacements[source].resolve())
                writer.writerow(adjusted)

    def _stage_output_variant(self, phase: str) -> str:
        phase = str(phase).strip().casefold()
        if phase in {"sdf-bootstrap", "sdf-proof"}:
            return "structural"
        if phase in {"seam-proof", "seam-authority", "gate-proof"}:
            return "seam"
        if phase == "detail-reconstruction":
            return "detail"
        return "final"

    def _generate_candidate(
        self,
        *,
        root: Path,
        directory: Path,
        checkpoint: Path,
        epoch: int,
        phase: str,
        target_size: int,
        requested_device: str,
        obj_path: Path,
        materials: Path,
        asset_manifest: Path,
    ) -> dict[str, Any]:
        checkpoint = checkpoint.resolve()
        checkpoint_sha = self._sha256(checkpoint)
        config = load_resolved_config(root, directory.name)
        device = resolve_device(config, requested_device)
        model = self._load_epoch_model(checkpoint, config, device, epoch)
        helper = StrategyCandidateGenerator()
        fields, rows, comments = helper._read_tsv(materials)
        usages = helper._collect_usages(rows, materials.parent)
        contexts = helper._collect_contexts(rows, materials.parent)
        if not contexts:
            raise RuntimeError("live EVE material manifest contains no albedo contexts")
        source_before = helper._source_snapshot(usages.keys())
        stage_variant = self._stage_output_variant(phase)

        output_root = directory / "previews" / "live" / "candidates" / f"epoch_{epoch:04d}"
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        stage_canvases: dict[Path, Any] = {}
        baseline_canvases: dict[Path, Any] = {}
        stage_semantics: dict[Path, set[str]] = {}
        baseline_semantics: dict[Path, set[str]] = {}
        inference_records: list[dict[str, Any]] = []
        for index, (albedo, context) in enumerate(sorted(contexts.items(), key=lambda item: str(item[0]).casefold()), start=1):
            source = helper._read_bgra(albedo)
            width, height = helper._output_size(source, target_size)
            baseline_maps, baseline_diagnostics = helper._direct_maps(
                albedo=albedo,
                context=context,
                model=model,
                config=config,
                device=device,
                out_width=width,
                out_height=height,
                output_variant="baseline",
            )
            stage_maps, stage_diagnostics = helper._direct_maps(
                albedo=albedo,
                context=context,
                model=model,
                config=config,
                device=device,
                out_width=width,
                out_height=height,
                output_variant=stage_variant,
            )
            helper._apply_direct_maps(
                albedo=albedo,
                context=context,
                maps=baseline_maps,
                width=width,
                height=height,
                canvases=baseline_canvases,
                semantics=baseline_semantics,
            )
            helper._apply_direct_maps(
                albedo=albedo,
                context=context,
                maps=stage_maps,
                width=width,
                height=height,
                canvases=stage_canvases,
                semantics=stage_semantics,
            )
            inference_records.append({
                "source": str(albedo),
                "outputSize": [int(width), int(height)],
                "baselineDiagnostics": baseline_diagnostics,
                "stageVariant": stage_variant,
                "stageDiagnostics": stage_diagnostics,
            })
            print(
                f"[live-preview] epoch {epoch} [{index}/{len(contexts)}] {albedo.name} "
                f"-> {width}x{height} baseline + {stage_variant}",
                flush=True,
            )

        def write_canvases(canvases, folder: str, suffix: str) -> dict[Path, Path]:
            texture_dir = output_root / folder
            replacements: dict[Path, Path] = {}
            for source, canvas in sorted(canvases.items(), key=lambda item: str(item[0]).casefold()):
                token = hashlib.sha1(str(source).casefold().encode("utf-8")).hexdigest()[:10]
                destination = texture_dir / f"{source.stem}_{token}_{suffix}.png"
                helper._write_png(destination, canvas)
                replacements[source] = destination.resolve()
            return replacements

        baseline_replacements = write_canvases(
            baseline_canvases, "baseline_4x", "baseline_4x"
        )
        stage_replacements = write_canvases(
            stage_canvases, "live_nsamdr", f"nsamdr_{stage_variant}"
        )

        baseline_materials = output_root / "baseline.materials.tsv"
        self._write_material_manifest(
            helper, baseline_materials, fields, rows, comments,
            baseline_replacements, materials.parent,
            physical_candidate="NSAMDR_DETERMINISTIC_4X_BASELINE",
        )
        candidate_materials = output_root / "live.materials.tsv"
        self._write_material_manifest(
            helper, candidate_materials, fields, rows, comments,
            stage_replacements, materials.parent,
        )
        candidate_obj = output_root / obj_path.name
        shutil.copy2(obj_path, candidate_obj)
        provenance = helper._provenance(
            source_before=source_before,
            replacements=stage_replacements,
            usages=usages,
            material_manifest=materials,
            asset_manifest=asset_manifest,
        )
        provenance_path = output_root / "live_control_provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        analysis_path = output_root / "live_candidate_analysis.json"
        analysis_path.write_text(
            json.dumps({
                "schema": "NSAMDR_LIVE_TRAINING_CANDIDATE_ANALYSIS_V2",
                "authority": "training-intermediate",
                "qualified": False,
                "epoch": int(epoch),
                "phase": phase,
                "stageVariant": stage_variant,
                "baselineVariant": "baseline",
                "checkpoint": str(checkpoint),
                "checkpointSha256": checkpoint_sha,
                "modelSchema": MODEL_SCHEMA,
                "productionForward": "FidelityResidualNetV9.forward(inputs)",
                "baselineForward": "deterministic production 4x baseline; no model.forward",
                "inference": inference_records,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = {
            "schema": "NSAMDR_LIVE_TRAINING_CANDIDATE_V2",
            "status": "live-preview-ready",
            "authority": "training-intermediate",
            "qualified": False,
            "epoch": int(epoch),
            "phase": phase,
            "stageVariant": stage_variant,
            "checkpoint": str(checkpoint),
            "checkpointSha256": checkpoint_sha,
            "candidateObj": str(candidate_obj.resolve()),
            "baselineObj": str(candidate_obj.resolve()),
            "baselineMaterials": str(baseline_materials.resolve()),
            "candidateMaterials": str(candidate_materials.resolve()),
            "candidateAnalysis": str(analysis_path.resolve()),
            "controlProvenance": provenance,
            "controlProvenancePath": str(provenance_path.resolve()),
            "targetSize": int(target_size),
            "inferenceDevice": str(device),
        }
        report_path = output_root / "candidate_manifest.json"
        report["reportPath"] = str(report_path.resolve())
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if self._sha256(checkpoint) != checkpoint_sha:
            raise RuntimeError("completed live checkpoint changed during candidate generation")
        return report

    def _candidate_pointer_text(self, report: Mapping[str, Any]) -> str:
        epoch = int(report["epoch"])
        phase = str(report["phase"])
        checkpoint_sha = str(report["checkpointSha256"])
        token = f"epoch-{epoch:04d}-{checkpoint_sha[:16]}"
        return "\n".join((
            "NSAMDR_LIVE_CANDIDATE_POINTER_V1",
            f"token={token}",
            f"epoch={epoch}",
            f"phase={phase}",
            f"checkpointSha256={checkpoint_sha}",
            f"stageVariant={report['stageVariant']}",
            f"baselineObj={report['baselineObj']}",
            f"baselineMaterials={report['baselineMaterials']}",
            f"candidateObj={report['candidateObj']}",
            f"candidateMaterials={report['candidateMaterials']}",
            f"candidateManifest={report['reportPath']}",
            "authority=training-intermediate",
            "qualified=false",
            "",
        ))

    def _stage_terminal_epoch(self, config, phase: str) -> int | None:
        """Return the final epoch belonging to one user-visible training stage."""
        boundary = int(config.identity_epochs)
        terminal = {"sdf-bootstrap": boundary}
        boundary += int(config.residual_epochs)
        terminal["sdf-proof"] = boundary
        boundary += int(config.seam_proof_epochs)
        terminal["seam-proof"] = boundary
        boundary += int(config.seam_authority_epochs)
        terminal["seam-authority"] = boundary
        boundary += int(config.boundary_epochs)
        terminal["gate-proof"] = boundary
        boundary += int(config.detail_epochs)
        terminal["detail-reconstruction"] = boundary
        boundary += int(config.physical_finetune_epochs)
        terminal["boundary-hardening"] = boundary
        terminal["physical-finetune"] = boundary
        return terminal.get(str(phase))

    def _stage_complete(self, config, epoch: int, phase: str) -> bool:
        """Tell a closed renderer when it may reopen without nagging every epoch."""
        terminal = self._stage_terminal_epoch(config, phase)
        return terminal is not None and int(epoch) >= int(terminal)

    def main(self, argv: list[str] | None = None) -> int:
        args = self._parser().parse_args(argv)
        if not 512 <= int(args.target_size) <= 2048:
            raise SystemExit("--target-size must be from 512 to 2048 for live preview")
        root = args.repo_root.resolve()
        experiment = args.experiment.strip().upper()
        directory = experiment_dir(root, experiment)
        manifest = load_experiment_manifest(root, experiment)
        asset = manifest.get("asset")
        if not isinstance(asset, dict):
            raise RuntimeError(f"{experiment} has no EVE asset identity")
        query = str(asset.get("query") or "").strip()
        selection_key = str(asset.get("selectionKey") or "")
        if not query:
            raise RuntimeError(f"{experiment} has no EVE asset query")
        resolved_config = load_resolved_config(root, experiment)

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
            raise RuntimeError("prepared live EVE asset has no material manifest")

        live_root = directory / "previews" / "live"
        checkpoint_pointer = live_root / "checkpoint_ready.json"
        candidate_pointer = live_root / "current_candidate.txt"
        launcher = root / "scripts/build/run_nsamdr_obj_preview_dx11.bat"
        renderer_thread: threading.Thread | None = None
        renderer_result: list[int] = []
        renderer_has_opened = False
        reopen_after_stage = False
        last_published_epoch = 0
        last_deferred_epoch = 0
        retry_after = 0.0

        def launch_renderer() -> None:
            os.environ.update({
                "NSAMDR_PREVIEW_EXPERIMENT": experiment,
                "NSAMDR_PREVIEW_AUTHORITY": "training-intermediate",
                "NSAMDR_LIVE_CANDIDATE_POINTER": str(candidate_pointer.resolve()),
            })
            selected_query = str(
                json.loads(asset_manifest.read_text(encoding="utf-8")).get("model", {}).get("logical")
                or query
            )
            selected_catalog_key = selection_key or selected_query
            renderer_result.append(eve.launch_preview(
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
                None,
            ))

        print(
            f"[live-preview] waiting for completed epoch snapshots from {checkpoint_pointer}",
            flush=True,
        )
        while True:
            current_manifest = self._read_json(directory / "experiment.json") or {}
            terminal = str(current_manifest.get("status") or "").casefold() in {
                "completed", "training-rejected", "interrupted-or-failed", "stage-paused"
            }
            if renderer_thread is not None and not renderer_thread.is_alive():
                exit_code = int(renderer_result[-1] if renderer_result else 0)
                print(
                    f"[live-preview] renderer closed (exit={exit_code}); watcher remains active and will reopen after the next completed stage",
                    flush=True,
                )
                renderer_thread = None
                renderer_result.clear()
                if terminal:
                    return 0
                if renderer_has_opened:
                    reopen_after_stage = True
            pointer = self._read_json(checkpoint_pointer)
            if pointer is not None:
                epoch = int(pointer.get("epoch", 0))
                phase = str(pointer.get("phase") or "unknown")
                checkpoint = Path(str(pointer.get("checkpoint") or ""))
                stage_complete = self._stage_complete(resolved_config, epoch, phase)
                can_publish = not reopen_after_stage or stage_complete
                if (
                    reopen_after_stage
                    and epoch > last_published_epoch
                    and not stage_complete
                    and epoch != last_deferred_epoch
                ):
                    print(
                        f"[live-preview] renderer is closed; epoch {epoch} ({phase}) is retained while waiting for the current stage to complete before reopening",
                        flush=True,
                    )
                    last_deferred_epoch = epoch
                if (
                    epoch > last_published_epoch
                    and checkpoint.is_file()
                    and time.monotonic() >= retry_after
                    and can_publish
                ):
                    try:
                        report = self._generate_candidate(
                            root=root,
                            directory=directory,
                            checkpoint=checkpoint,
                            epoch=epoch,
                            phase=phase,
                            target_size=int(args.target_size),
                            requested_device=args.device,
                            obj_path=obj_path,
                            materials=material_manifest,
                            asset_manifest=asset_manifest,
                        )
                        latest = self._read_json(checkpoint_pointer) or {}
                        latest_epoch = int(latest.get("epoch", epoch))
                        if latest_epoch > epoch:
                            print(
                                f"[live-preview] epoch {epoch} finished after epoch {latest_epoch} became available; skipping stale publish",
                                flush=True,
                            )
                            last_published_epoch = epoch
                            continue
                        self._atomic_text(candidate_pointer, self._candidate_pointer_text(report))
                        last_published_epoch = epoch
                        retry_after = 0.0
                        print(
                            f"[live-preview] PUBLISHED epoch {epoch} ({phase}) to the running EVE A/B renderer",
                            flush=True,
                        )
                        if renderer_thread is None:
                            renderer_result.clear()
                            renderer_thread = threading.Thread(target=launch_renderer, daemon=True)
                            renderer_thread.start()
                            renderer_has_opened = True
                            reopen_after_stage = False
                            last_deferred_epoch = 0
                    except Exception as exc:  # noqa: BLE001 - preview must never own training success
                        retry_after = time.monotonic() + 5.0
                        print(
                            f"[live-preview] epoch {epoch} preview failed ({type(exc).__name__}: {exc}); training continues and preview will retry",
                            file=sys.stderr,
                            flush=True,
                        )
            if terminal and renderer_thread is None:
                return 0
            time.sleep(max(0.25, float(args.poll_seconds)))


_app = LiveTrainingPreviewApplication()
main = _app.main

if __name__ == "__main__":
    raise SystemExit(main())
