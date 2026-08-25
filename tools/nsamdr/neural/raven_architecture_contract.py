#!/usr/bin/env python3
"""Fail-closed audit for the single production NSAMDR forward graph.

This module defines no model, loss, trainer, cache, or candidate path. It
instruments and strict-loads the implementation in :mod:`v9.model`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


_COMPONENT_PATHS: dict[str, tuple[str, ...]] = {
    "GeometryNet": ("geometry_net",),
    "Spline/SDF": (
        "geometry_net.production_structure",
        "geometry_net.parametric_primitive_field",
    ),
    "BoundaryRenderer": ("boundary_renderer",),
    "BoundaryProfile": ("boundary_specialist",),
    "PhaseAwareSeamSR": ("seam_restorer.phase_sr",),
    "SeamAuthority": ("seam_restorer.authority",),
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

_REQUIRED_OUTPUTS = (
    "albedo",
    "normal_xy",
    "material",
    "roughness",
    "emissive",
)


def _normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return payload


def _write_report(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    components = payload.get("components", {})
    lines = [
        "NSAMDR ARCHITECTURE PARTICIPATION",
        f"PASS: {str(bool(payload.get('pass'))).upper()}",
        f"Schema: {payload.get('schema', '')}",
        f"Model: {payload.get('modelClass', '')}",
        "",
    ]
    if isinstance(components, Mapping):
        for label in _COMPONENT_PATHS:
            row = components.get(label, {})
            if not isinstance(row, Mapping):
                row = {}
            active = "ACTIVE" if int(row.get("forwardCalls", 0) or 0) > 0 else "BYPASSED"
            training = str(row.get("trainingState", "UNVERIFIED")).upper()
            lines.append(
                f"{label:24s} {active:9s} {training:10s} "
                f"params={int(row.get('parameterCount', 0) or 0):9d} "
                f"grad={row.get('gradientNorm', 'n/a')} "
                f"delta={row.get('weightDelta', 'n/a')} "
                f"loss={row.get('lossContribution', 'n/a')}"
            )
    failures = payload.get("failures", [])
    if failures:
        lines.extend(("", "Failures:", *(f"- {item}" for item in failures)))
    human_path = output.with_suffix(".txt")
    human_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _install_import_path(repo: Path) -> None:
    neural = repo / "tools" / "nsamdr" / "neural"
    if str(neural) not in sys.path:
        sys.path.insert(0, str(neural))


def _load_model_api(repo: Path, config_path: Path):
    _install_import_path(repo)
    import torch  # noqa: WPS433
    from v9.config import V9Config  # type: ignore
    from v9.model import (  # type: ignore
        FidelityResidualNetV9,
        INPUT_CHANNELS,
        MODEL_SCHEMA,
        UPSCALE_FACTOR,
    )

    config = V9Config.load(config_path)
    return (
        torch,
        config,
        FidelityResidualNetV9,
        str(MODEL_SCHEMA),
        int(INPUT_CHANNELS),
        int(UPSCALE_FACTOR),
    )


def _source_fingerprints(repo: Path) -> dict[str, str]:
    relatives = (
        "tools/nsamdr/neural/v9/model.py",
        "tools/nsamdr/neural/v9/inference.py",
        "tools/nsamdr/neural/v9/training.py",
        "tools/nsamdr/neural/v9/losses.py",
        "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py",
        "tools/nsamdr/neural/run_nsamdr_v9_raven_tune_preview.py",
        "tools/nsamdr/neural/preview_nsamdr_v9_experiment.py",
        "tools/nsamdr/generate_strategy_candidates.py",
    )
    return {
        relative: _sha256(repo / relative)
        for relative in relatives
        if (repo / relative).is_file()
    }


def _module_at(model, path: str):
    current = model
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _declared_component_paths(model) -> dict[str, tuple[str, ...]]:
    result = dict(_COMPONENT_PATHS)
    if not hasattr(model, "architecture_contract"):
        return result
    declared = model.architecture_contract()
    if not isinstance(declared, Mapping):
        return result
    mapping = declared.get("productionComponents")
    if not isinstance(mapping, Mapping):
        return result
    aliases = {
        "GeometryNet": ("GeometryNet", "geometry"),
        "Spline/SDF": ("Spline/SDF", "structural representation"),
        "BoundaryRenderer": ("BoundaryRenderer", "boundary renderer"),
        "BoundaryProfile": ("BoundaryProfile", "boundary/profile"),
        "PhaseAwareSeamSR": ("PhaseAwareSeamSR",),
        "SeamAuthority": ("SeamAuthority", "seam authority"),
        "DetailNet": ("DetailNet", "conditioned detail"),
        "AlbedoHead": ("AlbedoHead", "albedo physical head"),
        "NormalHead": ("NormalHead", "normal physical head"),
        "MaterialHead": ("MaterialHead", "material physical head"),
        "BenefitSelector": ("BenefitSelector",),
    }
    normalised_mapping = {_normalise(str(key)): value for key, value in mapping.items()}
    for label in result:
        if label == "Confidence/Regret":
            confidence = normalised_mapping.get(_normalise("confidence"))
            regret = normalised_mapping.get(_normalise("regret"))
            values = [value for value in (confidence, regret) if isinstance(value, str) and value]
            if len(values) == 2:
                result[label] = tuple(values)
            continue
        value = None
        for alias in aliases.get(label, (label,)):
            value = normalised_mapping.get(_normalise(alias))
            if value is not None:
                break
        if isinstance(value, str) and value:
            result[label] = (value,)
        elif isinstance(value, (list, tuple)) and value and all(
            isinstance(item, str) and item for item in value
        ):
            result[label] = tuple(value)
    return result


def _tensor_summary(torch, value: Any) -> dict[str, Any]:
    tensors = []

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
    finite = True
    shapes: list[list[int]] = []
    for tensor in tensors[:12]:
        shapes.append([int(part) for part in tensor.shape])
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            finite = False
    return {"tensorCount": len(tensors), "finite": finite, "shapes": shapes}


def _sample_input(torch, channels: int, size: int):
    sample = torch.rand((1, channels, size, size), dtype=torch.float32)
    if channels >= 5:
        sample[:, 3:5] = sample[:, 3:5] * 2.0 - 1.0
    return sample


def _observe_production_forward(
    torch,
    model,
    *,
    input_channels: int,
    upscale: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    paths = _declared_component_paths(model)
    rows: dict[str, dict[str, Any]] = {}
    handles = []
    failures: list[str] = []

    for label, candidates in paths.items():
        modules: list[tuple[str, Any]] = []
        for path in candidates:
            module = _module_at(model, path)
            if module is not None:
                modules.append((path, module))
        row: dict[str, Any] = {
            "modulePaths": [path for path, _ in modules],
            "classes": [type(module).__name__ for _, module in modules],
            "parameterCount": int(
                sum(sum(parameter.numel() for parameter in module.parameters()) for _, module in modules)
            ),
            "forwardCalls": 0,
            "outputs": [],
            "trainingState": "UNVERIFIED",
            "gradientNorm": None,
            "weightDelta": None,
            "lossContribution": None,
            "validationMetric": None,
        }
        rows[label] = row
        if not modules:
            failures.append(f"{label}: production module path is absent")
            continue

        for path, module in modules:
            def hook(_module, _args, output, *, label=label, path=path):
                rows[label]["forwardCalls"] += 1
                summary = _tensor_summary(torch, output)
                summary["modulePath"] = path
                rows[label]["outputs"].append(summary)

            handles.append(module.register_forward_hook(hook))

    outputs: Any = {}
    size = 32
    try:
        if hasattr(model, "set_inference_mode"):
            model.set_inference_mode()
        model.eval()
        sample = _sample_input(torch, input_channels, size)
        with torch.inference_mode():
            outputs = model(sample)
    except Exception as exc:  # pragma: no cover - diagnostic detail
        failures.append(f"direct production model(input) failed: {type(exc).__name__}: {exc}")
    finally:
        for handle in handles:
            handle.remove()

    if not isinstance(outputs, Mapping):
        failures.append("production model returned a non-mapping output")
        outputs = {}

    output_report: dict[str, Any] = {}
    for key in _REQUIRED_OUTPUTS:
        value = outputs.get(key)
        if value is None or not torch.is_tensor(value):
            failures.append(f"production output is missing tensor {key!r}")
            continue
        expected_height = size * upscale
        shape = [int(part) for part in value.shape]
        finite = bool(torch.isfinite(value).all().item())
        shape_ok = len(shape) == 4 and shape[0] == 1 and shape[-2:] == [
            expected_height,
            expected_height,
        ]
        output_report[key] = {"shape": shape, "finite": finite, "shapeMatches4x": shape_ok}
        if not finite:
            failures.append(f"production output {key!r} contains non-finite values")
        if not shape_ok:
            failures.append(
                f"production output {key!r} has shape {shape}, expected 1xCx{expected_height}x{expected_height}"
            )

    for label, row in rows.items():
        if int(row["forwardCalls"]) == 0:
            failures.append(f"{label}: module exists but did not participate in model(input)")
        if any(not bool(item.get("finite")) for item in row["outputs"]):
            failures.append(f"{label}: module produced a non-finite tensor")

    forward = {
        "call": "model(input)",
        "kwargs": [],
        "training": False,
        "cacheUsed": False,
        "overridesUsed": False,
        "inputShape": [1, input_channels, size, size],
        "outputs": output_report,
    }
    return rows, forward, failures


def _training_components(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    evidence = payload.get("architecture_participation", {})
    if not isinstance(evidence, Mapping):
        return {}
    components = evidence.get("components", evidence)
    if not isinstance(components, Mapping):
        return {}
    return {
        _normalise(str(label)): row
        for label, row in components.items()
        if isinstance(row, Mapping)
    }


def _merge_training_evidence(
    rows: dict[str, dict[str, Any]],
    checkpoint: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    evidence = _training_components(checkpoint)
    if not evidence:
        return ["checkpoint contains no architecture_participation component evidence"]

    evidence_aliases = {
        "GeometryNet": ("GeometryNet", "geometry"),
        "Spline/SDF": ("Spline/SDF", "structural representation"),
        "BoundaryRenderer": ("BoundaryRenderer", "boundary renderer"),
        "BoundaryProfile": ("BoundaryProfile", "boundary/profile"),
        "PhaseAwareSeamSR": ("PhaseAwareSeamSR",),
        "SeamAuthority": ("SeamAuthority", "seam authority"),
        "DetailNet": ("DetailNet", "conditioned detail"),
        "AlbedoHead": ("AlbedoHead", "albedo physical head"),
        "NormalHead": ("NormalHead", "normal physical head"),
        "MaterialHead": ("MaterialHead", "material physical head"),
        "BenefitSelector": ("BenefitSelector",),
    }
    for label, row in rows.items():
        if label == "Confidence/Regret":
            sources = [
                evidence.get(_normalise("confidence")),
                evidence.get(_normalise("regret")),
            ]
            sources = [source for source in sources if isinstance(source, Mapping)]
            if len(sources) == 2:
                source = {
                    "trained": all(bool(item.get("trained")) for item in sources),
                    "frozen": all(bool(item.get("frozen")) for item in sources),
                    "maxGradientNorm": max(float(item.get("maxGradientNorm", 0.0)) for item in sources),
                    "stageStartWeightDeltaL2": max(float(item.get("stageStartWeightDeltaL2", 0.0)) for item in sources),
                    "lossContribution": {
                        name: value
                        for item in sources
                        for name, value in dict(item.get("lossContribution", {})).items()
                    },
                    "heads": sources,
                }
            else:
                source = None
        else:
            source = None
            for alias in evidence_aliases.get(label, (label,)):
                candidate = evidence.get(_normalise(alias))
                if candidate is not None:
                    source = candidate
                    break
        if source is None:
            failures.append(f"{label}: no training participation evidence")
            continue
        state = str(source.get("trainingState", source.get("status", ""))).upper()
        if not state and bool(source.get("trained")):
            state = "TRAINED"
        elif not state and bool(source.get("frozen")):
            state = "FROZEN"
        elif not state and source.get("frozenPhases"):
            state = "FROZEN"
        gradient = source.get(
            "gradientNorm",
            source.get("maxGradientNorm", source.get("lastGradientNorm")),
        )
        delta = source.get(
            "weightDelta",
            source.get("maxWeightDelta", source.get("stageStartWeightDeltaL2")),
        )
        loss = source.get("lossContribution")
        metric = source.get("validationMetric")
        row.update(
            {
                "trainingState": state or "UNVERIFIED",
                "gradientNorm": gradient,
                "weightDelta": delta,
                "lossContribution": loss,
                "validationMetric": metric,
                "stageEvidence": source,
            }
        )
        parameter_count = int(row.get("parameterCount", 0) or 0)
        if parameter_count == 0:
            continue
        if state not in {"TRAINED", "FROZEN"}:
            failures.append(f"{label}: invalid training state {state or 'missing'}")
            continue
        if state == "TRAINED":
            try:
                gradient_ok = float(gradient) > 0.0
                delta_ok = float(delta) > 0.0
            except (TypeError, ValueError):
                gradient_ok = delta_ok = False
            if not gradient_ok:
                failures.append(f"{label}: trained component has no non-zero gradient norm")
            if not delta_ok:
                failures.append(f"{label}: trained component has no non-zero stage weight delta")
            if loss is None:
                failures.append(f"{label}: trained component has no recorded loss contribution")
    return failures


def _config_path(experiment_dir: Path) -> Path:
    path = experiment_dir / "resolved_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing resolved config: {path}")
    return path


def _final_binding(experiment_dir: Path) -> tuple[dict[str, Any], Path, str]:
    manifest_path = experiment_dir / "final_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing final manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("final_manifest.json has no checkpoint object")
    raw_path = str(checkpoint.get("path", "")).strip()
    expected_sha = str(checkpoint.get("sha256", "")).strip().casefold()
    if not raw_path:
        raise RuntimeError("final_manifest checkpoint.path is empty")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
        raise RuntimeError("final_manifest checkpoint.sha256 is not a full SHA-256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = experiment_dir / path
    path = path.resolve()
    try:
        path.relative_to(experiment_dir.resolve())
    except ValueError as exc:
        raise RuntimeError(f"final checkpoint escapes experiment directory: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"immutable final checkpoint is missing: {path}")
    return manifest, path, expected_sha


def _checkpoint_is_read_only(path: Path) -> bool:
    mode = path.stat().st_mode
    return not bool(mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _preflight(repo: Path, config_path: Path, output: Path) -> int:
    torch, config, model_cls, schema, channels, upscale = _load_model_api(repo, config_path)
    torch.manual_seed(int(getattr(config, "seed", 1337)))
    model = model_cls(config)
    rows, forward, failures = _observe_production_forward(
        torch,
        model,
        input_channels=channels,
        upscale=upscale,
    )
    payload = {
        "kind": "nsamdr-production-architecture-preflight",
        "pass": not failures,
        "schema": schema,
        "modelClass": type(model).__name__,
        "parameterCount": int(sum(parameter.numel() for parameter in model.parameters())),
        "components": rows,
        "productionForward": forward,
        "failures": failures,
        "config": str(config_path),
        "sourceSha256": _source_fingerprints(repo),
        "invariant": "Raven changes dataset/work budget only; model and direct forward are production-identical",
    }
    _write_report(output, payload)
    for label, row in rows.items():
        print(
            f"[architecture] {label:24s} "
            f"{'ACTIVE' if row['forwardCalls'] else 'BYPASSED'} "
            f"calls={row['forwardCalls']} params={row['parameterCount']}",
            flush=True,
        )
    if failures:
        for failure in failures:
            print(f"[architecture] FAIL: {failure}", flush=True)
        return 2
    print("[architecture] PREFLIGHT=PASS", flush=True)
    return 0


def _postflight(repo: Path, experiment_dir: Path, output: Path) -> int:
    manifest, checkpoint_path, expected_sha = _final_binding(experiment_dir)
    actual_sha = _sha256(checkpoint_path)
    config_path = _config_path(experiment_dir)
    torch, config, model_cls, schema, channels, upscale = _load_model_api(repo, config_path)
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
    if not _checkpoint_is_read_only(checkpoint_path):
        failures.append("canonical final checkpoint remains writable")

    torch.manual_seed(int(getattr(config, "seed", 1337)))
    model = model_cls(config)
    strict_error = None
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:  # pragma: no cover - diagnostic detail
        strict_error = str(exc)
        failures.append(f"strict state_dict load failed: {exc}")

    rows: dict[str, dict[str, Any]] = {}
    forward: dict[str, Any] = {}
    if strict_error is None:
        observed_rows, forward, forward_failures = _observe_production_forward(
            torch,
            model,
            input_channels=channels,
            upscale=upscale,
        )
        rows = observed_rows
        failures.extend(forward_failures)
        failures.extend(_merge_training_evidence(rows, payload))

    cache_equivalence = payload.get("cache_equivalence")
    if not isinstance(cache_equivalence, Mapping) or not bool(cache_equivalence.get("passed")):
        failures.append("checkpoint has no passing cached-versus-uncached equivalence evidence")
    trainer_qualification = payload.get("final_qualification")
    if not isinstance(trainer_qualification, Mapping) or not bool(trainer_qualification.get("passed")):
        failures.append("checkpoint has no passing trainer uncached-final qualification evidence")

    report = {
        "kind": "nsamdr-production-architecture-participation",
        "pass": not failures,
        "schema": schema,
        "modelClass": type(model).__name__,
        "parameterCount": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint": str(checkpoint_path),
        "checkpointSha256": actual_sha,
        "manifestCheckpointSha256": expected_sha,
        "checkpointReadOnly": _checkpoint_is_read_only(checkpoint_path),
        "checkpointSchema": checkpoint_schema,
        "manifestSchema": manifest_schema,
        "selectionKind": selection_kind,
        "strictStateDictLoad": strict_error is None,
        "strictLoadError": strict_error,
        "components": rows,
        "productionForward": forward,
        "cacheEquivalence": cache_equivalence,
        "trainerFinalQualification": trainer_qualification,
        "sourceSha256": _source_fingerprints(repo),
        "failures": failures,
        "invariant": "strict full state + direct uncached model(input) + exact immutable checkpoint provenance",
    }
    _write_report(output, report)
    if failures:
        for failure in failures:
            print(f"[architecture] FAIL: {failure}", flush=True)
        print("[architecture] POSTFLIGHT=FAIL; preview blocked", flush=True)
        return 3
    print(f"[architecture] checkpoint_sha256={actual_sha}", flush=True)
    print("[architecture] POSTFLIGHT=PASS", flush=True)
    return 0


def _exact_checkpoint_hashes(value: Any) -> list[str]:
    hashes: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _normalise(str(key)) in {
                "checkpointsha256",
                "neuralcheckpointsha256",
                "finalcheckpointsha256",
            } and isinstance(nested, str):
                hashes.append(nested.casefold())
            hashes.extend(_exact_checkpoint_hashes(nested))
    elif isinstance(value, list):
        for nested in value:
            hashes.extend(_exact_checkpoint_hashes(nested))
    return hashes


def _file_records(value: Any) -> Iterable[tuple[str, Path, str]]:
    if isinstance(value, Mapping):
        pairs = (
            ("candidate", "candidatePath", "candidateSha256"),
            ("candidate", "path", "sha256"),
            ("source", "sourcePath", "sourceSha256After"),
            ("source", "sourcePath", "sourceSha256"),
        )
        for kind, path_key, hash_key in pairs:
            raw_path = value.get(path_key)
            raw_hash = value.get(hash_key)
            if isinstance(raw_path, str) and raw_path and isinstance(raw_hash, str) and raw_hash:
                yield kind, Path(raw_path), raw_hash.casefold()
        for nested in value.values():
            yield from _file_records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _file_records(nested)


def _previewflight(experiment_dir: Path, output: Path) -> int:
    manifest, checkpoint_path, expected_sha = _final_binding(experiment_dir)
    failures: list[str] = []
    if str(manifest.get("status", "")).casefold() != "completed":
        failures.append("final manifest status is not completed")
    if not bool(manifest.get("qualified")):
        failures.append("final manifest is not qualified")
    actual_checkpoint_sha = _sha256(checkpoint_path)
    if actual_checkpoint_sha != expected_sha:
        failures.append("checkpoint changed after qualification")

    preview_path = experiment_dir / "previews" / "preview_manifest.json"
    if not preview_path.is_file():
        failures.append(f"missing preview manifest: {preview_path}")
        preview: dict[str, Any] = {}
    else:
        preview = _read_json(preview_path)

    recorded_hashes = _exact_checkpoint_hashes(preview)
    if expected_sha not in recorded_hashes:
        failures.append("preview manifest does not bind the exact final checkpoint SHA-256")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in recorded_hashes):
        failures.append("preview manifest contains a partial or malformed checkpoint hash")
    if any(value != expected_sha for value in recorded_hashes):
        failures.append("preview manifest references a different checkpoint SHA-256")

    verified_records = []
    record_kinds: set[str] = set()
    seen: set[tuple[str, str, str]] = set()
    for kind, path, expected in _file_records(preview):
        key = (kind, str(path), expected)
        if key in seen:
            continue
        seen.add(key)
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            failures.append(f"{kind} record has a partial or malformed SHA-256: {path}")
            continue
        if not path.is_absolute():
            path = experiment_dir / path
        path = path.resolve()
        if not path.is_file():
            failures.append(f"{kind} provenance file is missing: {path}")
            continue
        actual = _sha256(path)
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
    _write_report(output, report)
    if failures:
        for failure in failures:
            print(f"[provenance] FAIL: {failure}", flush=True)
        print("[provenance] PREVIEW=FAIL; native renderer blocked", flush=True)
        return 4
    print("[provenance] PREVIEW=PASS", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the one production NSAMDR architecture and immutable artifacts"
    )
    parser.add_argument("mode", choices=("pre", "post", "preview"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    repo = args.repo_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = (repo / output).resolve()
    if args.mode == "pre":
        if args.config is None:
            parser.error("--config is required for pre")
        config = args.config
        if not config.is_absolute():
            config = repo / config
        return _preflight(repo, config.resolve(), output)

    if args.experiment_dir is None:
        parser.error("--experiment-dir is required for post/preview")
    experiment = args.experiment_dir
    if not experiment.is_absolute():
        experiment = repo / experiment
    experiment = experiment.resolve()
    if args.mode == "post":
        return _postflight(repo, experiment, output)
    return _previewflight(experiment, output)


if __name__ == "__main__":
    raise SystemExit(main())
