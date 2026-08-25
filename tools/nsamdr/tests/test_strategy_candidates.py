from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
GENERATOR_PATH = ROOT / "tools" / "nsamdr" / "generate_strategy_candidates.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("nsamdr_candidate_generator_test", GENERATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _final_tree(tmp_path: Path):
    generator = _load_generator()
    experiment = tmp_path / "EXP_0001"
    checkpoint = experiment / "checkpoints" / "final" / "nsamdr_v9_fidelity.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.chmod(stat.S_IREAD)
    sha = _sha(checkpoint)
    manifest = experiment / "final_manifest.json"
    _write_json(
        manifest,
        {
            "status": "completed",
            "qualified": True,
            "modelSchema": generator.MODEL_SCHEMA,
            "selectionKind": "production-final",
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": sha,
                "immutable": True,
            },
        },
    )
    _write_json(experiment / "architecture_participation.json", {"pass": True})
    return generator, experiment, checkpoint, manifest, sha


def test_parser_requires_exact_checkpoint_and_manifest() -> None:
    generator = _load_generator()
    parser = generator.build_parser()
    actions = {option for action in parser._actions for option in action.option_strings}
    assert {"--checkpoint", "--checkpoint-sha256", "--final-manifest"} <= actions
    assert "--checkpoint-dir" not in actions
    assert "--super-resolution-backend" not in actions
    assert "--force-candidate" not in actions


def test_generator_contains_no_alternate_sr_or_post_model_authority() -> None:
    source = GENERATOR_PATH.read_text(encoding="utf-8").casefold()
    forbidden = (
        "realesrgan",
        "candidate_safety_fallback",
        "representative_final_preview",
        "boundary_render(",
        "active_preview_strength",
        "fit_parametric_primitives_lr",
        "sdf_override=",
        "gate_override=",
    )
    assert [token for token in forbidden if token in source] == []
    assert '"postmodelblend": false' in source
    assert '"postmodelreplacement": false' in source


def test_final_binding_requires_exact_immutable_sha(tmp_path: Path) -> None:
    generator, _experiment, checkpoint, manifest, sha = _final_tree(tmp_path)
    final, bound_path, bound_sha = generator._final_binding(manifest, checkpoint, sha)
    assert final["qualified"] is True
    assert bound_path == checkpoint.resolve()
    assert bound_sha == sha
    with pytest.raises(RuntimeError, match="requested checkpoint SHA differs"):
        generator._final_binding(manifest, checkpoint, "f" * 64)


def test_final_binding_rejects_intermediate_or_unqualified_manifest(tmp_path: Path) -> None:
    generator, _experiment, checkpoint, manifest, sha = _final_tree(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["status"] = "frozen-pending-qualification"
    payload["qualified"] = False
    _write_json(manifest, payload)
    with pytest.raises(RuntimeError, match="completed qualified"):
        generator._final_binding(manifest, checkpoint, sha)


def test_direct_maps_calls_production_infer_without_override(monkeypatch) -> None:
    generator = _load_generator()
    calls = []
    monkeypatch.setattr(
        generator,
        "_model_input",
        lambda *_args, **_kwargs: np.zeros((17, 2, 2), dtype=np.float32),
    )

    def fake_infer(model, model_input, device, **kwargs):
        calls.append((model, model_input.shape, device, kwargs))
        maps = {
            "albedo": np.full((8, 8, 3), 0.5, dtype=np.float32),
            "normal_xy": np.zeros((8, 8, 2), dtype=np.float32),
            "material": np.full((8, 8, 3), 0.25, dtype=np.float32),
            "roughness": np.full((8, 8, 1), 0.75, dtype=np.float32),
            "emissive": np.zeros((8, 8, 1), dtype=np.float32),
        }
        return maps, {"productionForward": "FidelityResidualNetV9.forward(inputs)"}

    monkeypatch.setattr(generator, "infer_tiled", fake_infer)
    config = type("Config", (), {"inference_tile_size": 64, "inference_overlap": 8})()
    context = generator.AlbedoContext()
    maps, diagnostics = generator._direct_maps(
        albedo=Path("unused.png"),
        context=context,
        model=object(),
        config=config,
        device="cpu",
        out_width=8,
        out_height=8,
    )
    assert len(calls) == 1
    assert calls[0][3] == {
        "tile_size": 64,
        "overlap": 8,
        "return_diagnostics": True,
        "return_all_maps": True,
    }
    assert maps["albedo"].shape == (8, 8, 3)
    assert diagnostics["candidateAuthority"] == "direct-production-forward"
    assert diagnostics["postModelReplacement"] is False


def test_provenance_rehashes_source_and_candidate(tmp_path: Path) -> None:
    generator = _load_generator()
    source = tmp_path / "source.png"
    candidate = tmp_path / "candidate.png"
    materials = tmp_path / "materials.tsv"
    asset = tmp_path / "asset.json"
    source.write_bytes(b"source")
    candidate.write_bytes(b"candidate")
    materials.write_text("materials", encoding="utf-8")
    asset.write_text("{}", encoding="utf-8")
    before = {
        source: {
            "sha256": _sha(source),
            "dimensions": [1, 1],
            "sizeBytes": source.stat().st_size,
        }
    }
    monkey = pytest.MonkeyPatch()
    monkey.setattr(generator, "_dimensions", lambda _path: [1, 1])
    try:
        provenance = generator._provenance(
            source_before=before,
            replacements={source: candidate},
            usages={source: [generator.Usage("albedo")]},
            material_manifest=materials,
            asset_manifest=asset,
        )
    finally:
        monkey.undo()
    assert provenance["verified"] is True
    assert provenance["primaryAlbedo"]["sourceSha256"] == _sha(source)
    assert provenance["primaryAlbedo"]["candidateSha256"] == _sha(candidate)


def test_eve_launcher_uses_final_candidate_environment_names() -> None:
    source = (ROOT / "tools" / "nsamdr" / "eve_asset_test.py").read_text(encoding="utf-8")
    assert "NSAMDR_FINAL_OBJ" in source
    assert "NSAMDR_FINAL_MATERIALS" in source
    assert "NSAMDR_FINAL_ANALYSIS" in source
    assert "NSAMDR_FINAL_VALIDATION" in source
    assert "NSAMDR_MODE3_OBJ" not in source
