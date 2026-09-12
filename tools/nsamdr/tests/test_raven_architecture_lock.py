from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parents[3]
NEURAL = ROOT / "tools" / "nsamdr" / "neural"


class TestRavenArchitectureLock:
    def _load_contract(self):
        path = NEURAL / "raven_architecture_contract.py"
        spec = importlib.util.spec_from_file_location("nsamdr_architecture_contract_test", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _sha(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def _qualified_preview_tree(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        experiment = tmp_path / "EXP_0001"
        checkpoint = experiment / "checkpoints" / "final" / "nsamdr_v9_fidelity.pt"
        source = tmp_path / "authored.png"
        candidate = experiment / "previews" / "candidate" / "final.png"
        checkpoint.parent.mkdir(parents=True)
        candidate.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"immutable checkpoint")
        source.write_bytes(b"authored source")
        candidate.write_bytes(b"direct candidate")
        checkpoint.chmod(stat.S_IREAD)
        checkpoint_sha = self._sha(checkpoint)
        self._write_json(
            experiment / "final_manifest.json",
            {
                "status": "completed",
                "qualified": True,
                "modelSchema": "schema-for-preview-test",
                "selectionKind": "production-final",
                "checkpoint": {
                    "path": str(checkpoint),
                    "sha256": checkpoint_sha,
                    "immutable": True,
                },
            },
        )
        self._write_json(
            experiment / "previews" / "preview_manifest.json",
            {
                "checkpointSha256": checkpoint_sha,
                "candidate": {
                    "candidatePath": str(candidate),
                    "candidateSha256": self._sha(candidate),
                },
                "source": {
                    "sourcePath": str(source),
                    "sourceSha256": self._sha(source),
                },
            },
        )
        return experiment, source, candidate

    def test_required_component_inventory_separates_active_and_retired_graphs(self) -> None:
        contract = self._load_contract()
        assert tuple(contract._COMPONENT_PATHS) == (
            "GeometryNet",
            "Spline/SDF",
            "ExplicitRefiner",
            "BoundaryRenderer",
            "BoundaryProfile",
            "PhaseAwareSeamSR",
            "SeamAuthority",
            "DetailNet",
            "AlbedoHead",
            "NormalHead",
            "MaterialHead",
            "Confidence/Regret",
            "BenefitSelector",
        )
        assert tuple(contract._ACTIVE_COMPONENTS) == (
            "DetailNet",
            "AlbedoHead",
            "NormalHead",
            "MaterialHead",
            "Confidence/Regret",
            "BenefitSelector",
        )
        assert tuple(contract._RETIRED_COMPONENTS) == (
            "GeometryNet",
            "Spline/SDF",
            "ExplicitRefiner",
            "BoundaryRenderer",
            "BoundaryProfile",
            "PhaseAwareSeamSR",
            "SeamAuthority",
        )

    def test_runner_uses_production_trainer_contract_and_preview_only(self) -> None:
        source = (NEURAL / "run_nsamdr_v9_raven_tune_preview.py").read_text(encoding="utf-8")
        assert "train_nsamdr_v9_preview_experiment.py" in source
        assert "raven_architecture_contract.py" in source
        assert "preview_nsamdr_v9_experiment.py" in source
        assert "capability_first" not in source
        assert "capability_generalization" not in source

    def test_previewflight_accepts_only_exact_file_and_checkpoint_hashes(self, tmp_path: Path) -> None:
        contract = self._load_contract()
        experiment, _source, _candidate = self._qualified_preview_tree(tmp_path)
        output = experiment / "evidence" / "preview_contract.json"
        assert contract._previewflight(experiment, output) == 0
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["pass"] is True
        assert {row["kind"] for row in report["verifiedFiles"]} == {"source", "candidate"}

    def test_previewflight_rejects_candidate_mutation_before_renderer(self, tmp_path: Path) -> None:
        contract = self._load_contract()
        experiment, _source, candidate = self._qualified_preview_tree(tmp_path)
        candidate.write_bytes(b"mutated after provenance")
        output = experiment / "evidence" / "preview_contract.json"
        assert contract._previewflight(experiment, output) == 4
        report = json.loads(output.read_text(encoding="utf-8"))
        assert any("candidate provenance SHA mismatch" in item for item in report["failures"])

    def test_previewflight_rejects_partial_checkpoint_hash(self, tmp_path: Path) -> None:
        contract = self._load_contract()
        experiment, _source, _candidate = self._qualified_preview_tree(tmp_path)
        preview_path = experiment / "previews" / "preview_manifest.json"
        preview = json.loads(preview_path.read_text(encoding="utf-8"))
        preview["checkpointSha256"] = preview["checkpointSha256"][:12]
        self._write_json(preview_path, preview)
        output = experiment / "evidence" / "preview_contract.json"
        assert contract._previewflight(experiment, output) == 4
        report = json.loads(output.read_text(encoding="utf-8"))
        assert any("malformed checkpoint hash" in item for item in report["failures"])

    def test_previewflight_rejects_intermediate_or_unqualified_final(self, tmp_path: Path) -> None:
        contract = self._load_contract()
        experiment, _source, _candidate = self._qualified_preview_tree(tmp_path)
        manifest_path = experiment / "final_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "frozen-pending-qualification"
        manifest["qualified"] = False
        self._write_json(manifest_path, manifest)
        output = experiment / "evidence" / "preview_contract.json"
        assert contract._previewflight(experiment, output) == 4
        report = json.loads(output.read_text(encoding="utf-8"))
        assert "final manifest status is not completed" in report["failures"]
        assert "final manifest is not qualified" in report["failures"]


_test_raven_architecture_lock = TestRavenArchitectureLock()
_load_contract = _test_raven_architecture_lock._load_contract
_sha = _test_raven_architecture_lock._sha
_write_json = _test_raven_architecture_lock._write_json
_qualified_preview_tree = _test_raven_architecture_lock._qualified_preview_tree
