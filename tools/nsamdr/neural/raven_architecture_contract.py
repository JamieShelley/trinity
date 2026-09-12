#!/usr/bin/env python3
"""Fail-closed audit for the V13.3 production SR graph.

V13.3 has exactly one active learned reconstruction path:

    deterministic B -> DetailNet SR candidate C -> BenefitSelector F

Historical geometry/profile/seam modules remain in the state dict for strict-load
compatibility but must be BYPASSED by production model(input).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


_ACTIVE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "DetailNet": ("detail_net",),
    "AlbedoHead": ("detail_net.albedo_head",),
    "NormalHead": ("detail_net.normal_head",),
    "MaterialHead": ("detail_net.material_head",),
    "Confidence/Regret": (
        "detail_net.confidence_head",
        "detail_net.regret_head",
    ),
    "BenefitSelector": ("benefit_selector",),
}

_RETIRED_COMPONENTS: dict[str, tuple[str, ...]] = {
    "GeometryNet": ("geometry_net",),
    "Spline/SDF": (
        "geometry_net.production_structure",
        "geometry_net.parametric_primitive_field",
    ),
    "ExplicitRefiner": ("geometry_net.production_structure.geometry_refiner",),
    "BoundaryRenderer": ("boundary_renderer",),
    "BoundaryProfile": ("boundary_specialist",),
    "PhaseAwareSeamSR": ("seam_restorer.phase_sr",),
    "SeamAuthority": ("seam_restorer.authority",),
}

_COMPONENT_PATHS = {**_RETIRED_COMPONENTS, **_ACTIVE_COMPONENTS}
_REQUIRED_OUTPUTS = ("albedo", "normal_xy", "material", "roughness", "emissive")


class RavenArchitectureContract:
    def _normalise(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _read_json(self, path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"expected JSON object: {path}")
        return payload

    def _write_report(self, output: Path, payload: dict[str, Any]) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        lines = [
            "NSAMDR V13.3 ARCHITECTURE PARTICIPATION",
            f"PASS: {str(bool(payload.get('pass'))).upper()}",
            f"Schema: {payload.get('schema', '')}",
            f"Model: {payload.get('modelClass', '')}",
            "",
        ]
        components = payload.get("components", {})
        if isinstance(components, Mapping):
            for label in _COMPONENT_PATHS:
                row = components.get(label, {})
                if not isinstance(row, Mapping):
                    row = {}
                calls = int(row.get("forwardCalls", 0) or 0)
                expected = str(row.get("expectedState", ""))
                state = "ACTIVE" if calls else "BYPASSED"
                lines.append(
                    f"{label:24s} {state:9s} expected={expected:16s} "
                    f"calls={calls:3d} params={int(row.get('parameterCount', 0) or 0):9d}"
                )
        failures = payload.get("failures", [])
        if failures:
            lines.extend(("", "Failures:", *(f"- {item}" for item in failures)))
        output.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _install_import_path(self, repo: Path) -> None:
        neural = repo / "tools" / "nsamdr" / "neural"
        if str(neural) not in sys.path:
            sys.path.insert(0, str(neural))

    def _load_model_api(self, repo: Path, config_path: Path):
        self._install_import_path(repo)
        import torch  # noqa: WPS433
        from v9.config import V9Config  # type: ignore
        from v9.model import FidelityResidualNetV9, INPUT_CHANNELS, MODEL_SCHEMA, UPSCALE_FACTOR  # type: ignore

        config = V9Config.load(config_path)
        return torch, config, FidelityResidualNetV9, str(MODEL_SCHEMA), int(INPUT_CHANNELS), int(UPSCALE_FACTOR)

    def _source_fingerprints(self, repo: Path) -> dict[str, str]:
        relatives = (
            "tools/nsamdr/neural/v9/__init__.py",
            "tools/nsamdr/neural/v9/model.py",
            "tools/nsamdr/neural/v9/sr_first_contract.py",
            "tools/nsamdr/neural/v9/sr_first_runtime_contract.py",
            "tools/nsamdr/neural/v9/sr_first_generalization_contract.py",
            "tools/nsamdr/neural/v9/inference.py",
            "tools/nsamdr/neural/v9/training.py",
            "tools/nsamdr/neural/v9/application/backend.py",
            "tools/nsamdr/neural/v9/application/pipeline.py",
            "tools/nsamdr/neural/raven_architecture_contract.py",
        )
        return {
            relative: self._sha256(repo / relative)
            for relative in relatives
            if (repo / relative).is_file()
        }

    def _module_at(self, model: Any, path: str):
        value = model
        for part in path.split("."):
            if not hasattr(value, part):
                return None
            value = getattr(value, part)
        return value

    def _tensor_summary(self, torch: Any, value: Any) -> dict[str, Any]:
        tensors: list[Any] = []

        def collect(item: Any) -> None:
            if torch.is_tensor(item):
                tensors.append(item)
            elif isinstance(item, Mapping):
                for nested in item.values():
                    collect(nested)
            elif isinstance(item, (tuple, list)):
                for nested in item:
                    collect(nested)

        collect(value)
        return {
            "tensorCount": len(tensors),
            "finite": all(
                not tensor.is_floating_point() or bool(torch.isfinite(tensor).all().item())
                for tensor in tensors
            ),
            "shapes": [[int(part) for part in tensor.shape] for tensor in tensors[:12]],
        }

    def _sample_input(self, torch: Any, channels: int, size: int):
        sample = torch.rand((1, channels, size, size), dtype=torch.float32)
        if channels >= 5:
            sample[:, 3:5] = sample[:, 3:5] * 2.0 - 1.0
        return sample

    def _observe_production_forward(
        self,
        torch: Any,
        model: Any,
        *,
        input_channels: int,
        upscale: int,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
        rows: dict[str, dict[str, Any]] = {}
        handles: list[Any] = []
        failures: list[str] = []

        for label, candidates in _COMPONENT_PATHS.items():
            modules: list[tuple[str, Any]] = []
            for path in candidates:
                module = self._module_at(model, path)
                if module is not None:
                    modules.append((path, module))
            expected = "ACTIVE" if label in _ACTIVE_COMPONENTS else "RETIRED-BYPASSED"
            row: dict[str, Any] = {
                "modulePaths": [path for path, _ in modules],
                "classes": [type(module).__name__ for _, module in modules],
                "parameterCount": int(
                    sum(sum(parameter.numel() for parameter in module.parameters()) for _, module in modules)
                ),
                "forwardCalls": 0,
                "outputs": [],
                "expectedState": expected,
            }
            rows[label] = row
            if not modules:
                failures.append(f"{label}: checkpoint-compatible module path is absent")
                continue
            for path, module in modules:
                def hook(_module, _args, output, *, label=label, path=path):
                    rows[label]["forwardCalls"] += 1
                    summary = self._tensor_summary(torch, output)
                    summary["modulePath"] = path
                    rows[label]["outputs"].append(summary)
                handles.append(module.register_forward_hook(hook))

        outputs: Any = {}
        size = 32
        try:
            if hasattr(model, "set_inference_mode"):
                model.set_inference_mode()
            model.eval()
            with torch.inference_mode():
                outputs = model(self._sample_input(torch, input_channels, size))
        except Exception as exc:  # pragma: no cover - diagnostic surface
            failures.append(f"direct production model(input) failed: {type(exc).__name__}: {exc}")
        finally:
            for handle in handles:
                handle.remove()

        if not isinstance(outputs, Mapping):
            failures.append("production model returned a non-mapping output")
            outputs = {}

        output_report: dict[str, Any] = {}
        expected_height = size * upscale
        for key in _REQUIRED_OUTPUTS:
            value = outputs.get(key)
            if value is None or not torch.is_tensor(value):
                failures.append(f"production output is missing tensor {key!r}")
                continue
            shape = [int(part) for part in value.shape]
            finite = bool(torch.isfinite(value).all().item())
            shape_ok = len(shape) == 4 and shape[0] == 1 and shape[-2:] == [expected_height, expected_height]
            output_report[key] = {"shape": shape, "finite": finite, "shapeMatches4x": shape_ok}
            if not finite:
                failures.append(f"production output {key!r} contains non-finite values")
            if not shape_ok:
                failures.append(
                    f"production output {key!r} has shape {shape}, expected 1xCx{expected_height}x{expected_height}"
                )

        for label in _ACTIVE_COMPONENTS:
            row = rows[label]
            if int(row["forwardCalls"]) == 0:
                failures.append(f"{label}: active V13.3 module did not participate in model(input)")
            if any(not bool(item.get("finite")) for item in row["outputs"]):
                failures.append(f"{label}: active module produced a non-finite tensor")
        for label in _RETIRED_COMPONENTS:
            row = rows[label]
            if int(row["forwardCalls"]) != 0:
                failures.append(
                    f"{label}: retired module executed {row['forwardCalls']} time(s) in V13.3 production forward"
                )

        forward = {
            "call": "model(input)",
            "training": False,
            "cacheUsed": False,
            "overridesUsed": False,
            "inputShape": [1, input_channels, size, size],
            "outputs": output_report,
            "activeGraph": "B -> DetailNet C -> BenefitSelector F",
            "retiredComponentsBypassed": all(rows[label]["forwardCalls"] == 0 for label in _RETIRED_COMPONENTS),
        }
        return rows, forward, failures

    def _git_provenance(self, repo: Path) -> dict[str, Any]:
        def run(*arguments: str) -> tuple[int, str]:
            try:
                completed = subprocess.run(
                    ["git", "-C", str(repo), *arguments],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                )
            except (OSError, subprocess.TimeoutExpired):
                return 127, ""
            return int(completed.returncode), completed.stdout.strip()

        head_code, head = run("rev-parse", "HEAD")
        branch_code, branch = run("rev-parse", "--abbrev-ref", "HEAD")
        status_code, status = run("status", "--porcelain=v1", "--untracked-files=no")
        changes = [line for line in status.splitlines() if line.strip()]
        return {
            "available": head_code == 0,
            "head": head if head_code == 0 else None,
            "branch": branch if branch_code == 0 else None,
            "trackedDirty": bool(changes) if status_code == 0 else None,
            "trackedChanges": changes,
        }

    def _validate_trainer_contract(self, repo: Path, contract: Mapping[str, Any]) -> None:
        self._install_import_path(repo)
        import v9.training as training  # type: ignore  # noqa: WPS433
        from v9.application.backend import TrainingBackend  # type: ignore  # noqa: WPS433

        TrainingBackend()
        training._validate_v992_architecture_contract(dict(contract))
        service = getattr(training, "_training_service", None)
        if service is None:
            raise RuntimeError("trainer has no TrainingService singleton")
        service._validate_v992_architecture_contract(dict(contract))

    def _preflight(self, repo: Path, config_path: Path, output: Path) -> int:
        torch, config, model_cls, schema, channels, upscale = self._load_model_api(repo, config_path)
        torch.manual_seed(int(getattr(config, "seed", 1337)))
        model = model_cls(config)
        failures: list[str] = []
        try:
            self._validate_trainer_contract(repo, model.architecture_contract())
        except Exception as exc:
            failures.append(f"trainer architecture contract failed: {type(exc).__name__}: {exc}")
        rows, forward, forward_failures = self._observe_production_forward(
            torch, model, input_channels=channels, upscale=upscale
        )
        failures.extend(forward_failures)
        source_revision = self._git_provenance(repo)
        if source_revision.get("trackedDirty") is True:
            failures.append("tracked source differs from Git HEAD; commit/stash before training")
        report = {
            "kind": "nsamdr-v13.3-production-architecture-preflight",
            "pass": not failures,
            "schema": schema,
            "modelClass": type(model).__name__,
            "parameterCount": int(sum(parameter.numel() for parameter in model.parameters())),
            "components": rows,
            "productionForward": forward,
            "failures": failures,
            "config": str(config_path),
            "sourceSha256": self._source_fingerprints(repo),
            "sourceRevision": source_revision,
            "trainerContractValidated": not any("trainer architecture" in item for item in failures),
            "invariant": "V13.3 production executes only B -> DetailNet C -> BenefitSelector F",
        }
        self._write_report(output, report)
        print(
            f"[architecture] source HEAD={source_revision.get('head') or '<unavailable>'} "
            f"branch={source_revision.get('branch') or '<unavailable>'} "
            f"trackedDirty={source_revision.get('trackedDirty')}",
            flush=True,
        )
        for label, row in rows.items():
            state = "ACTIVE" if row["forwardCalls"] else "BYPASSED"
            print(
                f"[architecture] {label:24s} {state:9s} expected={row['expectedState']} "
                f"calls={row['forwardCalls']} params={row['parameterCount']}",
                flush=True,
            )
        if failures:
            for failure in failures:
                print(f"[architecture] FAIL: {failure}", flush=True)
            return 2
        print("[architecture] PREFLIGHT=PASS", flush=True)
        return 0

    def _config_path(self, experiment_dir: Path) -> Path:
        path = experiment_dir / "resolved_config.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing resolved config: {path}")
        return path

    def _final_binding(self, experiment_dir: Path) -> tuple[dict[str, Any], Path, str]:
        manifest_path = experiment_dir / "final_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing final manifest: {manifest_path}")
        manifest = self._read_json(manifest_path)
        checkpoint = manifest.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("final_manifest.json has no checkpoint object")
        raw_path = str(checkpoint.get("path", "")).strip()
        expected_sha = str(checkpoint.get("sha256", "")).strip().casefold()
        if not raw_path or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            raise RuntimeError("final manifest checkpoint binding is incomplete")
        path = Path(raw_path)
        if not path.is_absolute():
            path = experiment_dir / path
        path = path.resolve()
        try:
            path.relative_to(experiment_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"final checkpoint escapes experiment directory: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"immutable final checkpoint missing: {path}")
        return manifest, path, expected_sha

    def _checkpoint_is_read_only(self, path: Path) -> bool:
        mode = path.stat().st_mode
        return not bool(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    def _postflight(self, repo: Path, experiment_dir: Path, output: Path) -> int:
        manifest, checkpoint_path, expected_sha = self._final_binding(experiment_dir)
        actual_sha = self._sha256(checkpoint_path)
        config_path = self._config_path(experiment_dir)
        torch, config, model_cls, schema, channels, upscale = self._load_model_api(repo, config_path)
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
            raise RuntimeError(f"final checkpoint contains no complete state_dict: {checkpoint_path}")

        failures: list[str] = []
        checkpoint_schema = str(payload.get("schema", ""))
        manifest_schema = str(manifest.get("modelSchema", ""))
        selection_kind = str(payload.get("selection_kind", manifest.get("selectionKind", "")))
        if actual_sha != expected_sha:
            failures.append("immutable checkpoint SHA differs from final_manifest.json")
        if checkpoint_schema != schema or manifest_schema != schema:
            failures.append(
                f"schema mismatch: production={schema!r}, checkpoint={checkpoint_schema!r}, manifest={manifest_schema!r}"
            )
        if selection_kind != "production-final":
            failures.append(f"selection kind is not production-final: {selection_kind!r}")
        if not bool(manifest.get("checkpoint", {}).get("immutable")):
            failures.append("final manifest does not mark the checkpoint immutable")
        if not self._checkpoint_is_read_only(checkpoint_path):
            failures.append("canonical final checkpoint remains writable")

        torch.manual_seed(int(getattr(config, "seed", 1337)))
        model = model_cls(config)
        strict_error = None
        try:
            model.load_state_dict(payload["state_dict"], strict=True)
        except Exception as exc:  # pragma: no cover
            strict_error = str(exc)
            failures.append(f"strict state_dict load failed: {exc}")

        rows: dict[str, dict[str, Any]] = {}
        forward: dict[str, Any] = {}
        if strict_error is None:
            rows, forward, forward_failures = self._observe_production_forward(
                torch, model, input_channels=channels, upscale=upscale
            )
            failures.extend(forward_failures)

        cache_equivalence = payload.get("cache_equivalence")
        if not isinstance(cache_equivalence, Mapping) or not bool(cache_equivalence.get("passed")):
            failures.append("checkpoint has no passing cached-versus-uncached equivalence evidence")
        runtime_integrity = payload.get("final_qualification")
        if not isinstance(runtime_integrity, Mapping) or not bool(runtime_integrity.get("passed")):
            failures.append("checkpoint has no passing production runtime-integrity evidence")

        report = {
            "kind": "nsamdr-v13.3-production-architecture-participation",
            "pass": not failures,
            "schema": schema,
            "modelClass": type(model).__name__,
            "parameterCount": int(sum(parameter.numel() for parameter in model.parameters())),
            "checkpoint": str(checkpoint_path),
            "checkpointSha256": actual_sha,
            "manifestCheckpointSha256": expected_sha,
            "checkpointReadOnly": self._checkpoint_is_read_only(checkpoint_path),
            "checkpointSchema": checkpoint_schema,
            "manifestSchema": manifest_schema,
            "selectionKind": selection_kind,
            "strictStateDictLoad": strict_error is None,
            "strictLoadError": strict_error,
            "components": rows,
            "productionForward": forward,
            "uncachedProductionForwardPass": not any("production" in item or "retired module" in item for item in failures),
            "cacheEquivalence": cache_equivalence,
            "productionRuntimeIntegrity": runtime_integrity,
            # Compatibility alias consumed by FINAL_MANIFEST_V1. New diagnostics use
            # productionRuntimeIntegrity exclusively.
            "trainerFinalQualification": runtime_integrity,
            "sourceSha256": self._source_fingerprints(repo),
            "sourceRevision": self._git_provenance(repo),
            "failures": failures,
            "invariant": "strict full state + direct V13.3 SR forward + immutable checkpoint provenance",
        }
        self._write_report(output, report)
        if failures:
            for failure in failures:
                print(f"[architecture] FAIL: {failure}", flush=True)
            print("[architecture] POSTFLIGHT=FAIL; preview blocked", flush=True)
            return 3
        print(f"[architecture] checkpoint_sha256={actual_sha}", flush=True)
        print("[architecture] POSTFLIGHT=PASS", flush=True)
        return 0

    def _exact_checkpoint_hashes(self, value: Any) -> list[str]:
        hashes: list[str] = []
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if self._normalise(str(key)) in {
                    "checkpointsha256", "neuralcheckpointsha256", "finalcheckpointsha256"
                } and isinstance(nested, str):
                    hashes.append(nested.casefold())
                hashes.extend(self._exact_checkpoint_hashes(nested))
        elif isinstance(value, list):
            for nested in value:
                hashes.extend(self._exact_checkpoint_hashes(nested))
        return hashes

    def _file_records(self, value: Any) -> Iterable[tuple[str, Path, str]]:
        if isinstance(value, Mapping):
            for kind, path_key, hash_key in (
                ("candidate", "candidatePath", "candidateSha256"),
                ("candidate", "path", "sha256"),
                ("source", "sourcePath", "sourceSha256After"),
                ("source", "sourcePath", "sourceSha256"),
            ):
                raw_path = value.get(path_key)
                raw_hash = value.get(hash_key)
                if isinstance(raw_path, str) and raw_path and isinstance(raw_hash, str) and raw_hash:
                    yield kind, Path(raw_path), raw_hash.casefold()
            for nested in value.values():
                yield from self._file_records(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from self._file_records(nested)

    def _previewflight(self, experiment_dir: Path, output: Path) -> int:
        manifest, checkpoint_path, expected_sha = self._final_binding(experiment_dir)
        failures: list[str] = []
        if str(manifest.get("status", "")).casefold() != "completed":
            failures.append("final manifest status is not completed")
        if not bool(manifest.get("qualified")):
            failures.append("final manifest is not qualified")
        actual_checkpoint_sha = self._sha256(checkpoint_path)
        if actual_checkpoint_sha != expected_sha:
            failures.append("checkpoint changed after qualification")

        preview_path = experiment_dir / "previews" / "preview_manifest.json"
        preview = self._read_json(preview_path) if preview_path.is_file() else {}
        if not preview:
            failures.append(f"missing preview manifest: {preview_path}")

        recorded_hashes = self._exact_checkpoint_hashes(preview)
        if expected_sha not in recorded_hashes:
            failures.append("preview manifest does not bind the exact final checkpoint SHA-256")
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in recorded_hashes):
            failures.append("preview manifest contains a malformed checkpoint hash")
        if any(value != expected_sha for value in recorded_hashes):
            failures.append("preview manifest references a different checkpoint SHA-256")

        verified_records = []
        record_kinds: set[str] = set()
        seen: set[tuple[str, str, str]] = set()
        for kind, path, expected in self._file_records(preview):
            key = (kind, str(path), expected)
            if key in seen:
                continue
            seen.add(key)
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                failures.append(f"{kind} record has malformed SHA-256: {path}")
                continue
            if not path.is_absolute():
                path = experiment_dir / path
            path = path.resolve()
            if not path.is_file():
                failures.append(f"{kind} provenance file missing: {path}")
                continue
            actual = self._sha256(path)
            if actual != expected:
                failures.append(f"{kind} provenance SHA mismatch: {path}")
                continue
            record_kinds.add(kind)
            verified_records.append({"kind": kind, "path": str(path), "sha256": actual})
        if "candidate" not in record_kinds:
            failures.append("preview manifest has no verified candidate file record")
        if "source" not in record_kinds:
            failures.append("preview manifest has no verified raw-source file record")

        report = {
            "kind": "nsamdr-production-preview-provenance",
            "pass": not failures,
            "schema": str(manifest.get("modelSchema", "")),
            "checkpoint": str(checkpoint_path),
            "checkpointSha256": actual_checkpoint_sha,
            "previewManifest": str(preview_path),
            "recordedCheckpointSha256": recorded_hashes,
            "verifiedFiles": verified_records,
            "components": {},
            "failures": failures,
        }
        self._write_report(output, report)
        if failures:
            for failure in failures:
                print(f"[provenance] FAIL: {failure}", flush=True)
            print("[provenance] PREVIEW=FAIL; native renderer blocked", flush=True)
            return 4
        print("[provenance] PREVIEW=PASS", flush=True)
        return 0

    def main(self, argv: list[str] | None = None) -> int:
        parser = argparse.ArgumentParser(
            description="Audit V13.3 active SR architecture and immutable artifacts"
        )
        parser.add_argument("mode", choices=("pre", "post", "preview"))
        parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        parser.add_argument("--config", type=Path)
        parser.add_argument("--experiment-dir", type=Path)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(argv)

        repo = args.repo_root.resolve()
        output = args.output if args.output.is_absolute() else (repo / args.output).resolve()
        if args.mode == "pre":
            if args.config is None:
                parser.error("--config is required for pre")
            config = args.config if args.config.is_absolute() else repo / args.config
            return self._preflight(repo, config.resolve(), output)

        if args.experiment_dir is None:
            parser.error("--experiment-dir is required for post/preview")
        experiment = (
            args.experiment_dir
            if args.experiment_dir.is_absolute()
            else repo / args.experiment_dir
        ).resolve()
        if args.mode == "post":
            return self._postflight(repo, experiment, output)
        return self._previewflight(experiment, output)


_raven_architecture_contract = RavenArchitectureContract()
main = _raven_architecture_contract.main


if __name__ == "__main__":
    raise SystemExit(main())
