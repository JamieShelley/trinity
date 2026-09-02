from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


# -----------------------------------------------------------------------------
# training.py: publish a lean, cryptographically bound live checkpoint only
# when the user has explicitly started the live EVE preview.  Synchronize the
# expensive live bake between epochs so it cannot compete with training CUDA.
# -----------------------------------------------------------------------------
path = Path("tools/nsamdr/neural/v9/training.py")
text = path.read_text(encoding="utf-8")
anchor = '''    # Purpose: Implement atomic torch save for TrainingService.\n'''
method = '''    # Purpose: Publish one lean checkpoint for the opt-in live EVE renderer.\n    # Called by: train_v9\n    # Calls: _atomic_json, _atomic_torch_save, _status\n    def _publish_live_preview_checkpoint(\n        self,\n        *,\n        model: torch.nn.Module,\n        config: V9Config,\n        epoch: int,\n        phase: str,\n        state_path: Path,\n    ) -> None:\n        live_dir = state_path.parent / "previews" / "live"\n        request_path = live_dir / "request.json"\n        if not request_path.is_file():\n            return\n        try:\n            request = json.loads(request_path.read_text(encoding="utf-8"))\n        except (OSError, ValueError, TypeError):\n            return\n        if not isinstance(request, dict) or request.get("enabled") is not True:\n            return\n\n        checkpoint_path = live_dir / "live_checkpoint.pt"\n        manifest_path = live_dir / "live_checkpoint.json"\n        self._atomic_torch_save(\n            {\n                "schema": MODEL_SCHEMA,\n                "config": config.to_dict(),\n                "selection_kind": "training-live",\n                "epoch": int(epoch),\n                "phase": str(phase),\n                "state_dict": model.state_dict(),\n            },\n            checkpoint_path,\n        )\n        digest = hashlib.sha256()\n        with checkpoint_path.open("rb") as handle:\n            for block in iter(lambda: handle.read(1024 * 1024), b""):\n                digest.update(block)\n        checkpoint_sha = digest.hexdigest()\n        self._atomic_json(\n            {\n                "schema": "NSAMDR_TRAINING_LIVE_CHECKPOINT_V1",\n                "authority": "training-live",\n                "modelSchema": MODEL_SCHEMA,\n                "epoch": int(epoch),\n                "phase": str(phase),\n                "checkpoint": {\n                    "path": str(checkpoint_path.resolve()),\n                    "sha256": checkpoint_sha,\n                },\n            },\n            manifest_path,\n        )\n        self._status(\n            f"[live-preview] published epoch={epoch:03d} phase={phase} sha256={checkpoint_sha[:12]}"\n        )\n\n        # Live candidate inference is deliberately serialized between epochs.\n        # Otherwise a second CUDA process can steal VRAM from the training step\n        # and reproduce the WDDM/offload failures this pipeline already guards.\n        if request.get("synchronize", True) is True:\n            consumed_path = live_dir / "consumed.json"\n            started = time.monotonic()\n            last_status = started\n            while request_path.is_file() and time.monotonic() - started < 900.0:\n                try:\n                    consumed = json.loads(consumed_path.read_text(encoding="utf-8"))\n                except (OSError, ValueError, TypeError):\n                    consumed = {}\n                if isinstance(consumed, dict) and consumed.get("checkpointSha256") == checkpoint_sha:\n                    self._status(f"[live-preview] epoch={epoch:03d} EVE candidate consumed; training resumes")\n                    break\n                now = time.monotonic()\n                if now - last_status >= 10.0:\n                    self._status(\n                        f"[live-preview] epoch={epoch:03d} paused between epochs while EVE candidate renders"\n                    )\n                    last_status = now\n                time.sleep(0.25)\n\n'''
text = replace_once(text, anchor, method + anchor, "training live publisher")
state_save_end = '''                "cache_equivalence": cache_equivalence,\n            }, state_path)\n\n            if phase == "physical-finetune" and early_stop_patience is not None and early_stop_patience > 0:\n'''
state_save_new = '''                "cache_equivalence": cache_equivalence,\n            }, state_path)\n            self._publish_live_preview_checkpoint(\n                model=model,\n                config=config,\n                epoch=epoch,\n                phase=phase,\n                state_path=state_path,\n            )\n\n            if phase == "physical-finetune" and early_stop_patience is not None and early_stop_patience > 0:\n'''
text = replace_once(text, state_save_end, state_save_new, "training epoch publish call")
path.write_text(text, encoding="utf-8")


# -----------------------------------------------------------------------------
# generate_strategy_candidates.py: preserve the production-final path exactly,
# while adding a distinct training-live binding accepted only with an explicit
# live manifest.
# -----------------------------------------------------------------------------
path = Path("tools/nsamdr/generate_strategy_candidates.py")
text = path.read_text(encoding="utf-8")
anchor = '''    # Purpose: Implement source snapshot for StrategyCandidateGenerator.\n'''
live_binding = '''    # Purpose: Bind an explicitly unqualified live-training checkpoint.\n    # Called by: generate_candidate\n    # Calls: _full_sha, _read_json, _sha256\n    def _live_binding(\n        self,\n        live_manifest_path: Path,\n        checkpoint_path: Path,\n        expected_checkpoint_sha: str,\n    ) -> tuple[dict[str, Any], Path, str]:\n        live_manifest_path = live_manifest_path.resolve()\n        if not live_manifest_path.is_file():\n            raise FileNotFoundError(f"missing training-live manifest: {live_manifest_path}")\n        live = self._read_json(live_manifest_path)\n        if str(live.get("authority", "")) != "training-live":\n            raise RuntimeError("live manifest authority is not training-live")\n        if str(live.get("modelSchema", "")) != MODEL_SCHEMA:\n            raise RuntimeError(\n                f"live manifest schema {live.get('modelSchema')!r} does not match {MODEL_SCHEMA!r}"\n            )\n        binding = live.get("checkpoint")\n        if not isinstance(binding, Mapping):\n            raise RuntimeError("live manifest has no checkpoint binding")\n        manifest_path = Path(str(binding.get("path", ""))).expanduser()\n        if not manifest_path.is_absolute():\n            manifest_path = live_manifest_path.parent / manifest_path\n        manifest_path = manifest_path.resolve()\n        checkpoint_path = checkpoint_path.resolve()\n        if manifest_path != checkpoint_path:\n            raise RuntimeError(\n                f"live candidate checkpoint differs from live manifest: {checkpoint_path} != {manifest_path}"\n            )\n        manifest_sha = self._full_sha(binding.get("sha256"), label="live manifest checkpoint SHA")\n        expected_sha = self._full_sha(expected_checkpoint_sha, label="requested live checkpoint SHA")\n        if manifest_sha != expected_sha:\n            raise RuntimeError(\n                f"requested live checkpoint SHA differs from manifest: {expected_sha} != {manifest_sha}"\n            )\n        if not checkpoint_path.is_file():\n            raise FileNotFoundError(f"training-live checkpoint is missing: {checkpoint_path}")\n        actual_sha = self._sha256(checkpoint_path)\n        if actual_sha != manifest_sha:\n            raise RuntimeError(\n                f"training-live checkpoint SHA mismatch: expected={manifest_sha} actual={actual_sha}"\n            )\n        return live, checkpoint_path, actual_sha\n\n'''
text = replace_once(text, anchor, live_binding + anchor, "candidate live binding")
old_sig = '''        checkpoint_sha256: str,\n        final_manifest: Path,\n        inference_device: str,\n    ) -> dict[str, Any]:\n'''
new_sig = '''        checkpoint_sha256: str,\n        final_manifest: Path | None = None,\n        live_manifest: Path | None = None,\n        inference_device: str = "cuda",\n    ) -> dict[str, Any]:\n'''
text = replace_once(text, old_sig, new_sig, "candidate signature")
old_top = '''        output_root = output_root.resolve()\n        final_manifest = final_manifest.resolve()\n        experiment_dir = final_manifest.parent\n        if not self._path_within(output_root, experiment_dir / "previews"):\n            raise RuntimeError(\n                f"candidate output must stay inside {experiment_dir / 'previews'}: {output_root}"\n            )\n'''
new_top = '''        output_root = output_root.resolve()\n        if (final_manifest is None) == (live_manifest is None):\n            raise RuntimeError("candidate requires exactly one of final_manifest or live_manifest")\n        if live_manifest is not None:\n            binding_manifest = live_manifest.resolve()\n            if len(binding_manifest.parents) < 3:\n                raise RuntimeError(f"invalid live manifest location: {binding_manifest}")\n            experiment_dir = binding_manifest.parents[2]\n            authority = "training-live"\n        else:\n            assert final_manifest is not None\n            binding_manifest = final_manifest.resolve()\n            experiment_dir = binding_manifest.parent\n            authority = "production-final"\n        if not self._path_within(output_root, experiment_dir / "previews"):\n            raise RuntimeError(\n                f"candidate output must stay inside {experiment_dir / 'previews'}: {output_root}"\n            )\n'''
text = replace_once(text, old_top, new_top, "candidate binding mode")
old_bind = '''        final, checkpoint, checkpoint_sha = self._final_binding(\n            final_manifest,\n            checkpoint,\n            checkpoint_sha256,\n        )\n'''
new_bind = '''        if authority == "training-live":\n            binding_record, checkpoint, checkpoint_sha = self._live_binding(\n                binding_manifest, checkpoint, checkpoint_sha256\n            )\n        else:\n            binding_record, checkpoint, checkpoint_sha = self._final_binding(\n                binding_manifest, checkpoint, checkpoint_sha256\n            )\n'''
text = replace_once(text, old_bind, new_bind, "candidate authority binding")
old_qualification = '''        model, config, checkpoint_payload = load_trained_model(checkpoint, device)\n        if str(checkpoint_payload.get("selection_kind", "")) != "production-final":\n            raise RuntimeError("immutable checkpoint selection_kind is not production-final")\n        qualification = checkpoint_payload.get("final_qualification")\n        if not isinstance(qualification, Mapping) or qualification.get("passed") is not True:\n            raise RuntimeError("immutable checkpoint has no passing uncached final qualification")\n        if self._sha256(checkpoint) != checkpoint_sha:\n            raise RuntimeError("immutable checkpoint changed while loading")\n'''
new_qualification = '''        model, config, checkpoint_payload = load_trained_model(checkpoint, device)\n        checkpoint_selection = str(checkpoint_payload.get("selection_kind", ""))\n        if authority == "production-final":\n            if checkpoint_selection != "production-final":\n                raise RuntimeError("immutable checkpoint selection_kind is not production-final")\n            qualification = checkpoint_payload.get("final_qualification")\n            if not isinstance(qualification, Mapping) or qualification.get("passed") is not True:\n                raise RuntimeError("immutable checkpoint has no passing uncached final qualification")\n        else:\n            if checkpoint_selection != "training-live":\n                raise RuntimeError("live checkpoint selection_kind is not training-live")\n            if int(checkpoint_payload.get("epoch", -1)) != int(binding_record.get("epoch", -2)):\n                raise RuntimeError("live checkpoint epoch differs from its live manifest")\n        if self._sha256(checkpoint) != checkpoint_sha:\n            raise RuntimeError(f"{authority} checkpoint changed while loading")\n'''
text = replace_once(text, old_qualification, new_qualification, "candidate checkpoint qualification")
text = text.replace('            "selectionKind": "production-final",\n            "productionModelClass": type(model).__name__,', '            "selectionKind": authority,\n            "productionModelClass": type(model).__name__,', 1)
old_validation = '''        validation_path = output_root / "candidate_validation.json"\n        validation = {\n            "schema": "NSAMDR_PRODUCTION_CANDIDATE_VALIDATION_V1",\n            "passed": True,\n            "checkpointSha256": checkpoint_sha,\n            "selectionKind": "production-final",\n            "checks": {\n                "finalManifestCompleted": final.get("status") == "completed",\n                "finalManifestQualified": final.get("qualified") is True,\n                "architectureParticipationPassed": True,\n                "checkpointStrictLoaded": True,\n                "checkpointUnchangedAfterLoad": self._sha256(checkpoint) == checkpoint_sha,\n                "directProductionForward": True,\n                "sourceProvenanceVerified": provenance.get("verified") is True,\n                "allCandidateFilesExist": all(path.is_file() for path in replacements.values()),\n                "noPostModelReplacement": True,\n            },\n        }\n'''
new_validation = '''        validation_path = output_root / "candidate_validation.json"\n        if authority == "production-final":\n            checks = {\n                "finalManifestCompleted": binding_record.get("status") == "completed",\n                "finalManifestQualified": binding_record.get("qualified") is True,\n                "architectureParticipationPassed": True,\n                "checkpointStrictLoaded": True,\n                "checkpointUnchangedAfterLoad": self._sha256(checkpoint) == checkpoint_sha,\n                "directProductionForward": True,\n                "sourceProvenanceVerified": provenance.get("verified") is True,\n                "allCandidateFilesExist": all(path.is_file() for path in replacements.values()),\n                "noPostModelReplacement": True,\n            }\n        else:\n            checks = {\n                "trainingLiveAuthority": binding_record.get("authority") == "training-live",\n                "liveEpochMatches": int(checkpoint_payload.get("epoch", -1)) == int(binding_record.get("epoch", -2)),\n                "checkpointStrictLoaded": True,\n                "checkpointUnchangedAfterLoad": self._sha256(checkpoint) == checkpoint_sha,\n                "directProductionForward": True,\n                "sourceProvenanceVerified": provenance.get("verified") is True,\n                "allCandidateFilesExist": all(path.is_file() for path in replacements.values()),\n                "noPostModelReplacement": True,\n            }\n        validation = {\n            "schema": "NSAMDR_PRODUCTION_CANDIDATE_VALIDATION_V1",\n            "passed": True,\n            "checkpointSha256": checkpoint_sha,\n            "selectionKind": authority,\n            "checks": checks,\n        }\n'''
text = replace_once(text, old_validation, new_validation, "candidate validation split")
# Replace the report's selection kind (the analysis occurrence was replaced above).
report_anchor = '''            "status": "verified",\n            "checkpoint": str(checkpoint),\n            "checkpointSha256": checkpoint_sha,\n            "selectionKind": "production-final",\n'''
report_new = '''            "status": "verified",\n            "checkpoint": str(checkpoint),\n            "checkpointSha256": checkpoint_sha,\n            "selectionKind": authority,\n            "authority": authority,\n'''
text = replace_once(text, report_anchor, report_new, "candidate report authority")
parser_old = '''        parser.add_argument("--final-manifest", required=True, type=Path)\n'''
parser_new = '''        binding = parser.add_mutually_exclusive_group(required=True)\n        binding.add_argument("--final-manifest", type=Path)\n        binding.add_argument("--live-manifest", type=Path)\n'''
text = replace_once(text, parser_old, parser_new, "candidate parser binding")
main_old = '''                final_manifest=args.final_manifest,\n                inference_device=args.inference_device,\n'''
main_new = '''                final_manifest=args.final_manifest,\n                live_manifest=args.live_manifest,\n                inference_device=args.inference_device,\n'''
text = replace_once(text, main_old, main_new, "candidate main binding")
export_anchor = '''_final_binding = _strategy_candidate_generator._final_binding\n'''
text = replace_once(text, export_anchor, export_anchor + '_live_binding = _strategy_candidate_generator._live_binding\n', "candidate live binding export")
path.write_text(text, encoding="utf-8")


# -----------------------------------------------------------------------------
# New live preview worker: request checkpoint publication, bake one real EVE
# candidate per epoch at a small default size, launch the native renderer once,
# and update a generation marker only after each candidate is complete.
# -----------------------------------------------------------------------------
Path("tools/nsamdr/neural/live_preview_nsamdr_v9_experiment.py").write_text(r'''#!/usr/bin/env python3
"""Continuously preview the current training epoch in the real EVE renderer."""
from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
NSAMDR_ROOT = HERE.parent
for import_root in (HERE, NSAMDR_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import eve_asset_test as eve  # type: ignore
from v9.experiments import experiment_dir, load_experiment_manifest


class LiveExperimentPreviewApplication:
    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Preview an unqualified training epoch in the real EVE renderer")
        parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        parser.add_argument("--experiment", required=True)
        parser.add_argument("--shared-cache", default=r"C:\CCP\EVE")
        parser.add_argument("--target-size", type=int, default=512)
        parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
        return parser

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _atomic_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", suffix=".tmp", delete=False) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        os.replace(temporary, path)

    def _atomic_text(self, path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", suffix=".tmp", delete=False) as handle:
            handle.write(value)
            temporary = Path(handle.name)
        os.replace(temporary, path)

    def _read_manifest(self, path: Path) -> dict | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, ValueError, TypeError):
            return None

    def _stable_checkpoint(self, published: dict, live_dir: Path) -> tuple[Path, Path, str, int, str] | None:
        binding = published.get("checkpoint")
        if not isinstance(binding, dict):
            return None
        source = Path(str(binding.get("path") or "")).resolve()
        expected = str(binding.get("sha256") or "").lower()
        if not source.is_file() or len(expected) != 64:
            return None
        working = live_dir / "working_checkpoint.pt"
        temporary = live_dir / "working_checkpoint.pt.tmp"
        try:
            shutil.copy2(source, temporary)
            actual = self._sha256(temporary)
            if actual != expected:
                temporary.unlink(missing_ok=True)
                return None
            os.replace(temporary, working)
        finally:
            temporary.unlink(missing_ok=True)
        epoch = int(published.get("epoch", -1))
        phase = str(published.get("phase") or "unknown")
        binding_manifest = live_dir / "working_manifest.json"
        self._atomic_json(
            binding_manifest,
            {
                "schema": "NSAMDR_TRAINING_LIVE_CHECKPOINT_V1",
                "authority": "training-live",
                "modelSchema": str(published.get("modelSchema") or ""),
                "epoch": epoch,
                "phase": phase,
                "checkpoint": {"path": str(working.resolve()), "sha256": actual},
            },
        )
        return working, binding_manifest, actual, epoch, phase

    def main(self, argv: list[str] | None = None) -> int:
        args = self._parser().parse_args(argv)
        if not 512 <= int(args.target_size) <= 4096:
            raise SystemExit("--target-size must be from 512 to 4096")
        root = args.repo_root.resolve()
        experiment = args.experiment.strip().upper()
        directory = experiment_dir(root, experiment)
        manifest = load_experiment_manifest(root, experiment)
        asset = manifest.get("asset")
        if not isinstance(asset, dict) or not str(asset.get("query") or "").strip():
            raise RuntimeError(f"{experiment} has no canonical EVE asset identity")
        query = str(asset["query"])
        selection_key = str(asset.get("selectionKey") or "")

        live_dir = directory / "previews" / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        request_path = live_dir / "request.json"
        consumed_path = live_dir / "consumed.json"
        published_path = live_dir / "live_checkpoint.json"
        generation_path = live_dir / "generation.txt"
        self._atomic_json(
            request_path,
            {
                "schema": "NSAMDR_TRAINING_LIVE_PREVIEW_REQUEST_V1",
                "enabled": True,
                "synchronize": True,
                "targetSize": int(args.target_size),
                "device": args.device,
            },
        )
        print(f"[live-preview] requested {experiment}; waiting for the next completed epoch checkpoint", flush=True)

        renderer_thread: threading.Thread | None = None
        renderer_result: list[int] = []
        last_sha = ""
        try:
            (
                obj_path, albedo, normal, pgs, environment, environments,
                material_manifest, asset_manifest, catalog, cache_root,
            ) = eve.prepare_asset(root, args.shared_cache, query, selection_key)
            if material_manifest is None or not material_manifest.is_file():
                raise RuntimeError("prepared EVE asset has no physical-map material manifest")
            asset_payload = json.loads(asset_manifest.read_text(encoding="utf-8"))
            selected_query = str(asset_payload.get("model", {}).get("logical") or query)
            selected_catalog_key = selection_key or selected_query
            launcher = root / "scripts/build/run_nsamdr_obj_preview_dx11.bat"
            report_path = live_dir / "candidate" / "candidate_manifest.json"

            while True:
                if renderer_thread is not None and not renderer_thread.is_alive():
                    return renderer_result[0] if renderer_result else 0
                published = self._read_manifest(published_path)
                if not published or published.get("authority") != "training-live":
                    time.sleep(0.5)
                    continue
                binding = published.get("checkpoint")
                published_sha = str(binding.get("sha256") or "") if isinstance(binding, dict) else ""
                if not published_sha or published_sha == last_sha:
                    time.sleep(0.5)
                    continue
                stable = self._stable_checkpoint(published, live_dir)
                if stable is None:
                    time.sleep(0.25)
                    continue
                checkpoint, live_manifest, checkpoint_sha, epoch, phase = stable
                command = [
                    sys.executable, "-u", str(root / "tools/nsamdr/generate_strategy_candidates.py"),
                    "--obj", str(obj_path),
                    "--materials", str(material_manifest),
                    "--asset-manifest", str(asset_manifest),
                    "--output-root", str(live_dir / "candidate"),
                    "--target-size", str(args.target_size),
                    "--checkpoint", str(checkpoint),
                    "--checkpoint-sha256", checkpoint_sha,
                    "--live-manifest", str(live_manifest),
                    "--inference-device", args.device,
                ]
                print(f"[live-preview] baking real EVE candidate epoch={epoch:03d} phase={phase} target={args.target_size}", flush=True)
                completed = subprocess.run(command, cwd=root, check=False)
                if completed.returncode != 0 or not report_path.is_file():
                    print(f"[live-preview] candidate bake failed for epoch={epoch:03d}; keeping previous visible generation", flush=True)
                    time.sleep(0.5)
                    continue
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("selectionKind") != "training-live":
                    raise RuntimeError("live candidate generator returned non-live authority")
                token = f"epoch={epoch}\nphase={phase}\ncheckpointSha256={checkpoint_sha}\n"
                self._atomic_text(generation_path, token)
                self._atomic_json(
                    consumed_path,
                    {"epoch": epoch, "phase": phase, "checkpointSha256": checkpoint_sha},
                )
                last_sha = checkpoint_sha
                print(f"[live-preview] epoch={epoch:03d} ready in native EVE renderer", flush=True)

                if renderer_thread is None:
                    os.environ.update(
                        {
                            "NSAMDR_PREVIEW_EXPERIMENT": experiment,
                            "NSAMDR_PREVIEW_CHECKPOINT": str(checkpoint),
                            "NSAMDR_PREVIEW_CHECKPOINT_SHA256": checkpoint_sha,
                            "NSAMDR_PREVIEW_AUTHORITY": "training-live",
                            "NSAMDR_LIVE_PREVIEW_GENERATION_FILE": str(generation_path.resolve()),
                        }
                    )

                    def run_renderer() -> None:
                        renderer_result.append(
                            eve.launch_preview(
                                root, launcher, obj_path, albedo, normal, pgs,
                                environment, environments, material_manifest,
                                asset_manifest, catalog, cache_root,
                                selected_catalog_key, report,
                            )
                        )

                    renderer_thread = threading.Thread(target=run_renderer, daemon=True)
                    renderer_thread.start()
                    print("[live-preview] native Raven/EVE preview window launched; future epochs hot-reload in place", flush=True)
        finally:
            request_path.unlink(missing_ok=True)


_app = LiveExperimentPreviewApplication()
main = _app.main

if __name__ == "__main__":
    raise SystemExit(main())
''', encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI: expose the opt-in live preview as a distinct command from final preview.
# -----------------------------------------------------------------------------
path = Path("tools/nsamdr/nsamdr_cli.py")
text = path.read_text(encoding="utf-8")
anchor = '''    # Purpose: Implement validate layout for NSAMDRCommandLineApplication.\n'''
method = '''    # Purpose: Implement command live preview for NSAMDRCommandLineApplication.\n    # Called by: External callers and the owning workflow.\n    # Calls: _python_script, _repo_args\n    def _command_live_preview(self, args: argparse.Namespace) -> int:\n        forwarded = ["--experiment", args.subject]\n        for option, attribute in (\n            ("--shared-cache", "shared_cache"),\n            ("--target-size", "target_size"),\n            ("--device", "device"),\n        ):\n            value = getattr(args, attribute, None)\n            if value is not None:\n                forwarded += [option, str(value)]\n        return self._python_script(\n            "tools/nsamdr/neural/live_preview_nsamdr_v9_experiment.py",\n            self._repo_args(forwarded),\n        )\n\n'''
text = replace_once(text, anchor, method + anchor, "CLI live command method")
parser_anchor = '''        validate = commands.add_parser("validate", help="validate layout, contract and architecture")\n'''
parser_new = '''        live_preview = commands.add_parser("live-preview", help="preview the current unqualified training epoch in the real EVE renderer")\n        live_preview.add_argument("subject", help="active EXP_####")\n        live_preview.add_argument("--shared-cache")\n        live_preview.add_argument("--target-size", type=int, default=512)\n        live_preview.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")\n        self._set_handler(live_preview, _command_live_preview)\n\n        validate = commands.add_parser("validate", help="validate layout, contract and architecture")\n'''
text = replace_once(text, parser_anchor, parser_new, "CLI live parser")
export_anchor = '''_command_preview = _n_s_a_m_d_r_command_line_application._command_preview\n'''
text = replace_once(text, export_anchor, export_anchor + '_command_live_preview = _n_s_a_m_d_r_command_line_application._command_live_preview\n', "CLI live export")
path.write_text(text, encoding="utf-8")


# -----------------------------------------------------------------------------
# Workflow GUI: explicit live EVE button + cheap default target.  It uses the
# existing preview worker/log window and is available while Quick/Full runs.
# -----------------------------------------------------------------------------
path = Path("tools/nsamdr/gui/nsamdr_v9_workflow_gui.py")
text = path.read_text(encoding="utf-8")
quick_anchor = '''            self._row("Preview target", "target", "4096", ("1024", "2048", "4096"))\n            self._row("Preview device", "device", "cuda", ("cuda", "cpu", "auto"))\n            self._training_performance_rows(default_workers="4")\n'''
quick_new = '''            self._row("Preview target", "target", "4096", ("1024", "2048", "4096"))\n            self._row("Preview device", "device", "cuda", ("cuda", "cpu", "auto"))\n            self._row("Live EVE preview target", "live_target", "512", ("512", "1024", "2048"))\n            self._row("Live EVE preview device", "live_device", "cuda", ("cuda", "cpu", "auto"))\n            self._training_performance_rows(default_workers="4")\n'''
text = replace_once(text, quick_anchor, quick_new, "GUI quick live fields")
full_anchor = '''            self._row("Preview target", "target", "4096", ("1024", "2048", "4096"))\n            self._row("Preview device", "device", "cuda", ("cuda", "cpu", "auto"))\n            self._training_performance_rows(default_workers="8")\n'''
full_new = '''            self._row("Preview target", "target", "4096", ("1024", "2048", "4096"))\n            self._row("Preview device", "device", "cuda", ("cuda", "cpu", "auto"))\n            self._row("Live EVE preview target", "live_target", "512", ("512", "1024", "2048"))\n            self._row("Live EVE preview device", "live_device", "cuda", ("cuda", "cpu", "auto"))\n            self._training_performance_rows(default_workers="8")\n'''
text = replace_once(text, full_anchor, full_new, "GUI full live fields")
footer_anchor = '''        ttk.Button(footer, text="Stop current process", command=self.stop).pack(side="left", padx=5)\n        ttk.Label(footer, text="Preview:").pack(side="left", padx=(6, 2))\n'''
footer_new = '''        ttk.Button(footer, text="Stop current process", command=self.stop).pack(side="left", padx=5)\n        ttk.Button(footer, text="Live EVE training preview", command=self.live_preview_active_training).pack(side="left", padx=(0, 5))\n        ttk.Label(footer, text="Preview:").pack(side="left", padx=(6, 2))\n'''
text = replace_once(text, footer_anchor, footer_new, "GUI live footer")
phase_anchor = '''        if "direct production inference" in lower:\n            return "Running direct production-model inference from the immutable final"\n'''
phase_new = '''        if lower.startswith("[live-preview]"):\n            if "waiting" in lower:\n                return "Waiting for the next completed training epoch"\n            if "baking" in lower:\n                return "Baking the current epoch into the real EVE asset preview"\n            if "hot-reload" in lower or "ready" in lower:\n                return "Live EVE renderer updated from the current training epoch"\n        if "direct production inference" in lower:\n            return "Running direct production-model inference from the immutable final"\n'''
text = replace_once(text, phase_anchor, phase_new, "GUI preview phase")
method_anchor = '''    # Purpose: Implement preview available model for App.\n'''
live_method = '''    # Purpose: Launch the current training experiment in the real EVE renderer.\n    # Called by: footer button\n    # Calls: _dispatcher_argv, _launch_preview_with_detailed_log, _value\n    def live_preview_active_training(self) -> None:\n        if self.process is None or self.active_stage not in {"quick", "train"}:\n            messagebox.showinfo(APP_TITLE, "Live EVE preview requires an active Quick or Full training run.")\n            return\n        result_path = self.repo / "artifacts/nsamdr/gui/last_nsamdr_workflow_result.json"\n        try:\n            payload = json.loads(result_path.read_text(encoding="utf-8"))\n            experiment = str(payload.get("experiment") or "").strip().upper()\n        except (OSError, ValueError, TypeError):\n            experiment = ""\n        if not EXPERIMENT_RE.fullmatch(experiment):\n            messagebox.showinfo(\n                APP_TITLE,\n                "The active experiment has not been allocated yet. Start the live preview after EXP_#### appears.",\n            )\n            return\n        args = [\n            experiment,\n            "--shared-cache", self._value("cache", r"C:\\CCP\\EVE"),\n            "--target-size", self._value("live_target", "512"),\n            "--device", self._value("live_device", "cuda"),\n        ]\n        command = self._dispatcher_argv(("live-preview",), args)\n        self._launch_preview_with_detailed_log(experiment, command)\n        self.progress_text.set(\n            f"Live EVE preview requested for {experiment}; training will pause between epochs while each 512px-class candidate is baked"\n        )\n\n'''
text = replace_once(text, method_anchor, live_method + method_anchor, "GUI live method")
path.write_text(text, encoding="utf-8")


# -----------------------------------------------------------------------------
# Native renderer: keep one ship/background/camera window alive and hot-reload
# only candidate GPU resources when the worker atomically changes generation.txt.
# -----------------------------------------------------------------------------
path = Path("trinityal/tests/nsamdr/NSAMDRPreviewApplication.cpp")
text = path.read_text(encoding="utf-8")
ns_anchor = '''namespace nsamdr\n{\n'''
helper = '''namespace nsamdr\n{\nnamespace\n{\nstd::string ReadLiveGenerationToken(const std::string& path)\n{\n    if (path.empty()) return {};\n    std::ifstream input(path, std::ios::binary);\n    if (!input) return {};\n    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());\n}\n}\n'''
text = replace_once(text, ns_anchor, helper, "native generation helper")
load_anchor = '''    ASSERT_TRUE(m_processing.LoadCandidates(\n        m_host.device,\n        m_host.context,\n        resources,\n        albedoPath,\n        candidates));\n\n    IMGUI_CHECKVERSION();\n'''
load_new = '''    ASSERT_TRUE(m_processing.LoadCandidates(\n        m_host.device,\n        m_host.context,\n        resources,\n        albedoPath,\n        candidates));\n    const std::string liveGenerationPath = GetEnvironmentString("NSAMDR_LIVE_PREVIEW_GENERATION_FILE");\n    std::string liveGenerationToken = ReadLiveGenerationToken(liveGenerationPath);\n    auto nextLiveReloadCheck = std::chrono::steady_clock::now();\n\n    IMGUI_CHECKVERSION();\n'''
text = replace_once(text, load_anchor, load_new, "native initial live token")
frame_anchor = '''        previousFrame = now;\n\n        uint32_t resizeWidth = 0U;\n'''
frame_new = '''        previousFrame = now;\n\n        if (!liveGenerationPath.empty() && now >= nextLiveReloadCheck)\n        {\n            nextLiveReloadCheck = now + std::chrono::milliseconds(500);\n            const std::string token = ReadLiveGenerationToken(liveGenerationPath);\n            if (!token.empty() && token != liveGenerationToken)\n            {\n                FinalCandidateSet replacement;\n                if (m_processing.LoadCandidates(\n                        m_host.device, m_host.context, resources, albedoPath, replacement) &&\n                    replacement.candidate.available)\n                {\n                    candidates = std::move(replacement);\n                    liveGenerationToken = token;\n                    std::printf("NSAMDR training-live candidate hot-reloaded: %s\\n", token.c_str());\n                }\n            }\n        }\n\n        uint32_t resizeWidth = 0U;\n'''
text = replace_once(text, frame_anchor, frame_new, "native hot reload loop")
path.write_text(text, encoding="utf-8")


# -----------------------------------------------------------------------------
# Native provenance: final remains fail-closed; training-live has its own exact
# authority gate and never masquerades as production-final.
# -----------------------------------------------------------------------------
path = Path("trinityal/tests/nsamdr/NSAMDRPreviewProcessing.cpp")
text = path.read_text(encoding="utf-8")
validator_anchor = '''std::string ValidateFinalCandidateProvenance(\n'''
live_validator = '''std::string ValidateLiveCandidateProvenance(\n    const PreviewResources& resources,\n    const std::string& rawAlbedoPath,\n    const CandidateAssetGpu& candidate)\n{\n    if (ToLowerAscii(GetEnvironmentString("NSAMDR_PREVIEW_AUTHORITY")) != "training-live")\n        return "checkpoint authority is not exactly training-live";\n    if (GetEnvironmentString("NSAMDR_PROVENANCE_STATUS") != "VERIFIED")\n        return "NSAMDR_PROVENANCE_STATUS is not VERIFIED";\n\n    const std::string sourcePath = GetEnvironmentString("NSAMDR_PROVENANCE_SOURCE");\n    const std::string candidatePath = GetEnvironmentString("NSAMDR_PROVENANCE_CANDIDATE");\n    const std::string provenanceFile = GetEnvironmentString("NSAMDR_PROVENANCE_FILE");\n    const std::string analysisFile = GetEnvironmentString("NSAMDR_FINAL_ANALYSIS");\n    const std::string validationFile = GetEnvironmentString("NSAMDR_FINAL_VALIDATION");\n    const std::string generationFile = GetEnvironmentString("NSAMDR_LIVE_PREVIEW_GENERATION_FILE");\n    if (!IsReadableFile(generationFile)) return "live generation marker is missing or unreadable";\n    if (!IsReadableFile(provenanceFile)) return "live provenance evidence is missing or unreadable";\n    if (!IsReadableFile(analysisFile)) return "live analysis is missing or unreadable";\n    if (!ValidationPassed(validationFile)) return "live validation is missing or did not pass";\n    if (!BaselineContainsAlbedo(resources, rawAlbedoPath, sourcePath))\n        return "raw pane albedo does not match the proven source path";\n    if (!CandidateContainsAlbedo(candidate, candidatePath))\n        return "live pane albedo does not match the proven candidate path";\n    if (!CandidateUsesSourceDrawRanges(resources, candidate))\n        return "live material groups do not map to the raw source mesh draw ranges";\n    return {};\n}\n\nstd::string ValidateFinalCandidateProvenance(\n'''
text = replace_once(text, validator_anchor, live_validator, "native live validator")
load_old = '''    CandidateAssetGpu& candidate = candidates.candidate;\n    if (!m_assetProcessor.LoadCandidateAsset(\n            device,\n            context,\n            "NSAMDR FINAL",\n            GetEnvironmentString("NSAMDR_FINAL_OBJ"),\n            GetEnvironmentString("NSAMDR_FINAL_MATERIALS"),\n            candidate))\n    {\n        return false;\n    }\n    if (!candidate.available) return true;\n\n    const std::string provenanceFailure = ValidateFinalCandidateProvenance(\n        resources,\n        rawAlbedoPath,\n        candidate);\n    if (!provenanceFailure.empty())\n    {\n        candidate.available = false;\n        candidate.status = "provenance gate blocked NSAMDR FINAL: " + provenanceFailure;\n    }\n    return true;\n'''
load_new = '''    CandidateAssetGpu& candidate = candidates.candidate;\n    const bool live = ToLowerAscii(GetEnvironmentString("NSAMDR_PREVIEW_AUTHORITY")) == "training-live";\n    if (!m_assetProcessor.LoadCandidateAsset(\n            device,\n            context,\n            live ? "NSAMDR TRAINING LIVE" : "NSAMDR FINAL",\n            GetEnvironmentString("NSAMDR_FINAL_OBJ"),\n            GetEnvironmentString("NSAMDR_FINAL_MATERIALS"),\n            candidate))\n    {\n        return false;\n    }\n    if (!candidate.available) return true;\n\n    const std::string provenanceFailure = live\n        ? ValidateLiveCandidateProvenance(resources, rawAlbedoPath, candidate)\n        : ValidateFinalCandidateProvenance(resources, rawAlbedoPath, candidate);\n    if (!provenanceFailure.empty())\n    {\n        candidate.available = false;\n        candidate.status = std::string("provenance gate blocked ") +\n            (live ? "NSAMDR TRAINING LIVE: " : "NSAMDR FINAL: ") + provenanceFailure;\n    }\n    else if (live)\n    {\n        candidate.status = "UNQUALIFIED TRAINING LIVE - visual evidence only";\n    }\n    return true;\n'''
text = replace_once(text, load_old, load_new, "native authority split")
path.write_text(text, encoding="utf-8")


# -----------------------------------------------------------------------------
# Native panel: make the non-final nature impossible to miss while preserving
# the normal immutable-final wording for qualified preview.
# -----------------------------------------------------------------------------
path = Path("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
text = path.read_text(encoding="utf-8")
intro_old = '''    ImGui::TextUnformatted("NSAMDR — Neural Stretch-Aware Material Detail Reconstruction");\n    ImGui::TextWrapped("Fixed production comparison: A RAW SOURCE and B NSAMDR FINAL. Both panes use the source mesh, one camera, one material shader and the same 16x anisotropic sampler at zero LOD bias. The renderer does not sharpen, denoise or otherwise alter the final candidate.");\n\n    const std::string previewExperiment = ReadEnvironmentVariable("NSAMDR_PREVIEW_EXPERIMENT");\n'''
intro_new = '''    ImGui::TextUnformatted("NSAMDR — Neural Stretch-Aware Material Detail Reconstruction");\n    const std::string previewAuthority = ReadEnvironmentVariable("NSAMDR_PREVIEW_AUTHORITY");\n    const bool trainingLive = previewAuthority == "training-live";\n    if (trainingLive)\n    {\n        ImGui::TextColored(ImVec4(1.0f, 0.76f, 0.18f, 1.0f), "UNQUALIFIED TRAINING LIVE - VISUAL EVIDENCE ONLY");\n        ImGui::TextWrapped("A RAW SOURCE and B NSAMDR TRAINING LIVE use the real EVE ship, environment, camera and material renderer. B hot-reloads after each completed epoch and is not a production-final qualification.");\n    }\n    else\n    {\n        ImGui::TextWrapped("Fixed production comparison: A RAW SOURCE and B NSAMDR FINAL. Both panes use the source mesh, one camera, one material shader and the same 16x anisotropic sampler at zero LOD bias. The renderer does not sharpen, denoise or otherwise alter the final candidate.");\n    }\n\n    const std::string previewExperiment = ReadEnvironmentVariable("NSAMDR_PREVIEW_EXPERIMENT");\n'''
text = replace_once(text, intro_old, intro_new, "panel live intro")
dupe = '''    const std::string previewAuthority = ReadEnvironmentVariable("NSAMDR_PREVIEW_AUTHORITY");\n'''
# One declaration remains later in the provenance block; remove exactly that next occurrence.
idx = text.find(dupe, text.find('const bool trainingLive'))
if idx < 0:
    raise SystemExit("panel duplicate authority declaration not found")
text = text[:idx] + text[idx + len(dupe):]
header_old = '''    ImGui::Separator();\n    ImGui::TextUnformatted("Immutable final provenance");\n'''
header_new = '''    ImGui::Separator();\n    ImGui::TextUnformatted(trainingLive ? "Training-live provenance" : "Immutable final provenance");\n'''
text = replace_once(text, header_old, header_new, "panel provenance header")
candidate_old = '''    ImGui::Separator();\n    ImGui::TextUnformatted("NSAMDR FINAL candidate");\n'''
candidate_new = '''    ImGui::Separator();\n    ImGui::TextUnformatted(trainingLive ? "NSAMDR TRAINING LIVE candidate" : "NSAMDR FINAL candidate");\n'''
text = replace_once(text, candidate_old, candidate_new, "panel candidate header")
comparison_old = '''    ImGui::TextWrapped("A RAW SOURCE and B NSAMDR FINAL are always visible. Both panes use the same source vertex/index buffers, camera, transform, lighting, environment, material shader, gradient sampling and 16x anisotropic sampler at zero LOD bias.");\n'''
comparison_new = '''    ImGui::TextWrapped(trainingLive\n        ? "A RAW SOURCE and B NSAMDR TRAINING LIVE remain on the same EVE mesh, camera, lighting and environment while B hot-reloads between epochs."\n        : "A RAW SOURCE and B NSAMDR FINAL are always visible. Both panes use the same source vertex/index buffers, camera, transform, lighting, environment, material shader, gradient sampling and 16x anisotropic sampler at zero LOD bias.");\n'''
text = replace_once(text, comparison_old, comparison_new, "panel comparison wording")
path.write_text(text, encoding="utf-8")


# -----------------------------------------------------------------------------
# Tests: CLI inventory/routing and live-preview separation/hot-reload contract.
# -----------------------------------------------------------------------------
path = Path("tools/nsamdr/tests/test_nsamdr_cli.py")
text = path.read_text(encoding="utf-8")
text = replace_once(text, '    ("preview",),\n    ("validate",),', '    ("preview",),\n    ("live-preview",),\n    ("validate",),', "CLI test public inventory")
text = replace_once(text, '    ("preview",): ["preview", "EXP_0001"],\n    ("validate",):', '    ("preview",): ["preview", "EXP_0001"],\n    ("live-preview",): ["live-preview", "EXP_0001"],\n    ("validate",):', "CLI test leaf inventory")
routing_anchor = '''    # Purpose: Implement test contract route uses canonical contract script for DispatcherRoutingTests.\n'''
routing_test = '''    def test_live_preview_routes_to_unqualified_epoch_previewer(self) -> None:\n        with mock.patch.object(CLI, "_python_script", return_value=31) as backend:\n            result = CLI.main(["live-preview", "EXP_0042", "--target-size", "512", "--device", "cpu"])\n        self.assertEqual(31, result)\n        backend.assert_called_once_with(\n            "tools/nsamdr/neural/live_preview_nsamdr_v9_experiment.py",\n            [\n                "--repo-root", os.fspath(CLI.REPO_ROOT),\n                "--experiment", "EXP_0042",\n                "--target-size", "512",\n                "--device", "cpu",\n            ],\n        )\n\n'''
text = replace_once(text, routing_anchor, routing_test + routing_anchor, "CLI live routing test")
path.write_text(text, encoding="utf-8")

Path("tools/nsamdr/tests/test_live_eve_training_preview_contract.py").write_text(r'''from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_training_publishes_live_checkpoint_only_on_explicit_request() -> None:
    text = source("tools/nsamdr/neural/v9/training.py")
    assert 'request_path = live_dir / "request.json"' in text
    assert 'if not request_path.is_file()' in text
    assert '"selection_kind": "training-live"' in text
    assert '"config": config.to_dict()' in text
    assert 'self._publish_live_preview_checkpoint(' in text
    assert 'paused between epochs while EVE candidate renders' in text


def test_candidate_generator_keeps_final_and_training_live_authorities_separate() -> None:
    text = source("tools/nsamdr/generate_strategy_candidates.py")
    assert 'def _live_binding(' in text
    assert 'if str(live.get("authority", "")) != "training-live"' in text
    assert 'if str(final.get("selectionKind", "")) != "production-final"' in text
    assert 'if checkpoint_selection != "production-final"' in text
    assert 'if checkpoint_selection != "training-live"' in text
    assert 'binding.add_argument("--live-manifest", type=Path)' in text


def test_live_worker_uses_real_eve_asset_pipeline_and_one_hot_reload_marker() -> None:
    text = source("tools/nsamdr/neural/live_preview_nsamdr_v9_experiment.py")
    assert 'eve.prepare_asset(' in text
    assert 'tools/nsamdr/generate_strategy_candidates.py' in text
    assert 'eve.launch_preview(' in text
    assert '"NSAMDR_PREVIEW_AUTHORITY": "training-live"' in text
    assert '"NSAMDR_LIVE_PREVIEW_GENERATION_FILE"' in text
    assert 'generation.txt' in text
    assert 'target-size", type=int, default=512' in text


def test_native_renderer_hot_reloads_candidate_without_restarting_scene() -> None:
    app = source("trinityal/tests/nsamdr/NSAMDRPreviewApplication.cpp")
    assert 'NSAMDR_LIVE_PREVIEW_GENERATION_FILE' in app
    assert 'ReadLiveGenerationToken' in app
    assert 'FinalCandidateSet replacement;' in app
    assert 'candidates = std::move(replacement);' in app
    assert 'm_processing.LoadCandidates(' in app


def test_native_final_gate_is_preserved_and_live_gate_is_explicit() -> None:
    processing = source("trinityal/tests/nsamdr/NSAMDRPreviewProcessing.cpp")
    assert 'ValidateFinalCandidateProvenance' in processing
    assert 'ValidateLiveCandidateProvenance' in processing
    assert 'authority.find("final") == std::string::npos' in processing
    assert 'ToLowerAscii(GetEnvironmentString("NSAMDR_PREVIEW_AUTHORITY")) != "training-live"' in processing
    assert 'live ? ValidateLiveCandidateProvenance' in processing
    panel = source("trinityal/tests/nsamdr/NSAMDRPreviewPanel.cpp")
    assert 'UNQUALIFIED TRAINING LIVE - VISUAL EVIDENCE ONLY' in panel
    assert 'NSAMDR TRAINING LIVE candidate' in panel


def test_workflow_gui_exposes_opt_in_cheap_real_eve_preview() -> None:
    gui = source("tools/nsamdr/gui/nsamdr_v9_workflow_gui.py")
    assert 'text="Live EVE training preview"' in gui
    assert '"Live EVE preview target", "live_target", "512"' in gui
    assert 'def live_preview_active_training(' in gui
    assert 'self._dispatcher_argv(("live-preview",), args)' in gui
''', encoding="utf-8")
