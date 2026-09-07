from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"anchor not found in {path}: {old[:100]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1. Make architecture preflight exercise the same trainer contract that production
# training uses, expose the explicit refiner as its own production component, and
# record source/Git provenance for every experiment.
contract = ROOT / "tools/nsamdr/neural/raven_architecture_contract.py"
replace_once(contract, "import stat\nimport sys\n", "import stat\nimport subprocess\nimport sys\n")
replace_once(
    contract,
    '''    "Spline/SDF": (\n        "geometry_net.production_structure",\n        "geometry_net.parametric_primitive_field",\n    ),\n    "BoundaryRenderer": ("boundary_renderer",),\n''',
    '''    "Spline/SDF": (\n        "geometry_net.production_structure",\n        "geometry_net.parametric_primitive_field",\n    ),\n    "ExplicitRefiner": ("geometry_net.production_structure.geometry_refiner",),\n    "BoundaryRenderer": ("boundary_renderer",),\n''',
)
replace_once(
    contract,
    '''            "Spline/SDF": ("Spline/SDF", "structural representation"),\n            "BoundaryRenderer": ("BoundaryRenderer", "boundary renderer"),\n''',
    '''            "Spline/SDF": ("Spline/SDF", "structural representation"),\n            "ExplicitRefiner": ("ExplicitRefiner", "explicit geometry refiner"),\n            "BoundaryRenderer": ("BoundaryRenderer", "boundary renderer"),\n''',
)
replace_once(
    contract,
    '''        relatives = (\n            "tools/nsamdr/neural/v9/model.py",\n            "tools/nsamdr/neural/v9/inference.py",\n            "tools/nsamdr/neural/v9/training.py",\n            "tools/nsamdr/neural/v9/losses.py",\n            "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py",\n            "tools/nsamdr/neural/run_nsamdr_v9_raven_tune_preview.py",\n            "tools/nsamdr/neural/preview_nsamdr_v9_experiment.py",\n            "tools/nsamdr/generate_strategy_candidates.py",\n        )\n''',
    '''        relatives = (\n            "tools/nsamdr/neural/v9/__init__.py",\n            "tools/nsamdr/neural/v9/model.py",\n            "tools/nsamdr/neural/v9/inference.py",\n            "tools/nsamdr/neural/v9/training.py",\n            "tools/nsamdr/neural/v9/losses.py",\n            "tools/nsamdr/neural/v9/local_boundary_production_contract.py",\n            "tools/nsamdr/neural/v9/explicit_spline_refiner.py",\n            "tools/nsamdr/neural/v9/edge_constrained_spline_graph.py",\n            "tools/nsamdr/neural/v9/spline_graph.py",\n            "tools/nsamdr/neural/v9/application/backend.py",\n            "tools/nsamdr/neural/v9/application/pipeline.py",\n            "tools/nsamdr/neural/v9/application/runner.py",\n            "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py",\n            "tools/nsamdr/neural/run_nsamdr_v9_raven_tune_preview.py",\n            "tools/nsamdr/neural/preview_nsamdr_v9_experiment.py",\n            "tools/nsamdr/generate_strategy_candidates.py",\n        )\n''',
)
preflight_anchor = '''    # Purpose: Implement preflight for RavenArchitectureContract.\n    # Called by: main\n    # Calls: _load_model_api, _observe_production_forward, _source_fingerprints, _write_report\n    def _preflight(self, repo: Path, config_path: Path, output: Path) -> int:\n        torch, config, model_cls, schema, channels, upscale = self._load_model_api(repo, config_path)\n        torch.manual_seed(int(getattr(config, "seed", 1337)))\n        model = model_cls(config)\n        rows, forward, failures = self._observe_production_forward(\n            torch,\n            model,\n            input_channels=channels,\n            upscale=upscale,\n        )\n        payload = {\n'''
preflight_new = '''    # Purpose: Capture the exact committed source revision used by this process.\n    # Called by: _preflight, _postflight.\n    # Calls: subprocess.run().\n    def _git_provenance(self, repo: Path) -> dict[str, Any]:\n        def run(*arguments: str) -> tuple[int, str]:\n            try:\n                completed = subprocess.run(\n                    ["git", "-C", str(repo), *arguments],\n                    check=False, capture_output=True, text=True, timeout=10.0,\n                )\n            except (OSError, subprocess.TimeoutExpired):\n                return 127, ""\n            return int(completed.returncode), completed.stdout.strip()\n\n        head_code, head = run("rev-parse", "HEAD")\n        branch_code, branch = run("rev-parse", "--abbrev-ref", "HEAD")\n        status_code, status = run("status", "--porcelain=v1", "--untracked-files=no")\n        tracked_changes = [line for line in status.splitlines() if line.strip()]\n        return {\n            "available": head_code == 0,\n            "head": head if head_code == 0 else None,\n            "branch": branch if branch_code == 0 else None,\n            "trackedDirty": bool(tracked_changes) if status_code == 0 else None,\n            "trackedChanges": tracked_changes,\n        }\n\n    # Purpose: Exercise the same installed trainer architecture validator used by train_v9.\n    # Called by: _preflight.\n    # Calls: TrainingBackend(), trainer architecture validators.\n    def _validate_trainer_contract(self, repo: Path, contract: Mapping[str, Any]) -> None:\n        self._install_import_path(repo)\n        import v9.training as training  # type: ignore  # noqa: WPS433\n        from v9.application.backend import TrainingBackend  # type: ignore  # noqa: WPS433\n\n        # TrainingBackend owns the production install/synchronisation order. Constructing\n        # it here deliberately reproduces the exact contract state train_v9 will see.\n        TrainingBackend()\n        training._validate_v992_architecture_contract(dict(contract))\n        service = getattr(training, "_training_service", None)\n        if service is None:\n            raise RuntimeError("trainer has no TrainingService singleton")\n        service._validate_v992_architecture_contract(dict(contract))\n\n    # Purpose: Implement preflight for RavenArchitectureContract.\n    # Called by: main\n    # Calls: _git_provenance, _load_model_api, _observe_production_forward, _source_fingerprints, _validate_trainer_contract, _write_report\n    def _preflight(self, repo: Path, config_path: Path, output: Path) -> int:\n        torch, config, model_cls, schema, channels, upscale = self._load_model_api(repo, config_path)\n        torch.manual_seed(int(getattr(config, "seed", 1337)))\n        model = model_cls(config)\n        trainer_failures: list[str] = []\n        try:\n            self._validate_trainer_contract(repo, model.architecture_contract())\n        except Exception as exc:\n            trainer_failures.append(\n                f"trainer architecture contract failed before training: {type(exc).__name__}: {exc}"\n            )\n        rows, forward, failures = self._observe_production_forward(\n            torch,\n            model,\n            input_channels=channels,\n            upscale=upscale,\n        )\n        failures = trainer_failures + failures\n        source_revision = self._git_provenance(repo)\n        if source_revision.get("trackedDirty") is True:\n            failures.append(\n                "tracked source files differ from Git HEAD; commit/stash source changes before training"\n            )\n        payload = {\n'''
replace_once(contract, preflight_anchor, preflight_new)
replace_once(
    contract,
    '''            "sourceSha256": self._source_fingerprints(repo),\n            "invariant": "Raven changes dataset/work budget only; model and direct forward are production-identical",\n''',
    '''            "sourceSha256": self._source_fingerprints(repo),\n            "sourceRevision": source_revision,\n            "trainerContractValidated": not trainer_failures,\n            "invariant": "Raven changes dataset/work budget only; model and direct forward are production-identical",\n''',
)
replace_once(
    contract,
    '''        for label, row in rows.items():\n            print(\n''',
    '''        print(\n            f"[architecture] source HEAD={source_revision.get('head') or '<unavailable>'} "\n            f"branch={source_revision.get('branch') or '<unavailable>'} "\n            f"trackedDirty={source_revision.get('trackedDirty')}",\n            flush=True,\n        )\n        for label, row in rows.items():\n            print(\n''',
)
# Postflight records the source state too, allowing diagnostics to show if code changed
# between preflight/training/final qualification.
replace_once(
    contract,
    '''            "sourceSha256": self._source_fingerprints(repo),\n            "failures": failures,\n            "invariant": "strict full state + direct uncached model(input) + exact immutable checkpoint provenance",\n''',
    '''            "sourceSha256": self._source_fingerprints(repo),\n            "sourceRevision": self._git_provenance(repo),\n            "failures": failures,\n            "invariant": "strict full state + direct uncached model(input) + exact immutable checkpoint provenance",\n''',
)

# 2. Retire V11.1-V11.3 source-shape tests that assert architectures that no longer
# exist. Their live responsibilities are covered by the current pass-driven/V12 tests.
for relative in (
    "tools/nsamdr/tests/test_b1b_reserved_substage_budget_contract.py",
    "tools/nsamdr/tests/test_local_boundary_v111_contract.py",
    "tools/nsamdr/tests/test_local_boundary_v112_contract.py",
    "tools/nsamdr/tests/test_local_boundary_v113_contract.py",
    "tools/nsamdr/tests/test_raven_evolution_workflow_v114.py",
):
    path = ROOT / relative
    if path.exists():
        path.unlink()

# Keep the small historical-retirement checks, but point them at current owners.
b1a = ROOT / "tools/nsamdr/tests/test_b1a_parametric_bootstrap_contract.py"
replace_once(
    b1a,
    '''        assert 'losses["parametric_anchor"]' in source\n        assert 'losses["sdf_topology_sign"]' in source\n''',
    '''        assert 'losses["spline_graph_topology_control"]' in source\n        assert 'losses["spline_graph_topology_sign"]' in source\n        assert 'B1a trains only the differentiable topology/control field' in source\n''',
)
b1b = ROOT / "tools/nsamdr/tests/test_b1b_staged_contract.py"
text = b1b.read_text(encoding="utf-8")
text = text.replace(
    'ENTRY = ROOT / "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py"\n',
    'BACKEND = ROOT / "tools/nsamdr/neural/v9/application/backend.py"\n',
)
text = text.replace(
    '''        source = ENTRY.read_text(encoding="utf-8")\n        assert "install_b1b_staged_contract" not in source\n        assert "install_local_boundary_training_contract" in source\n''',
    '''        source = BACKEND.read_text(encoding="utf-8")\n        assert "install_b1b_staged_contract" not in source\n        assert "install_local_boundary_training_contract" in source\n        assert "_synchronize_training_service_contract" in source\n''',
)
b1b.write_text(text, encoding="utf-8")

# 3. Version-patch test filenames are replaced by semantic names. This preserves
# their coverage while restoring the repository-wide no-test_v*.py rule.
renames = {
    "test_v116_edge_constrained_spline_contract.py": "test_edge_constrained_spline_contract.py",
    "test_v117_baseline_relative_contract.py": "test_baseline_relative_structural_contract.py",
    "test_v117_quick_baseline_smoke.py": "test_quick_baseline_smoke.py",
    "test_v120_explicit_geometry_refiner_contract.py": "test_explicit_geometry_refiner_contract.py",
    "test_v121_proposal_baseline_regret.py": "test_proposal_baseline_regret.py",
    "test_v121_refiner_telemetry_contract.py": "test_refiner_telemetry_contract.py",
    "test_v122_evolution_proposal_contract.py": "test_evolution_proposal_contract.py",
    "test_v123_refiner_inference_context.py": "test_refiner_inference_context.py",
    "test_v124_trainer_architecture_contract.py": "test_trainer_architecture_contract.py",
}
tests = ROOT / "tools/nsamdr/tests"
for old_name, new_name in renames.items():
    old = tests / old_name
    new = tests / new_name
    if old.exists():
        if new.exists():
            raise SystemExit(f"rename target already exists: {new}")
        old.rename(new)

# 4. OOP dispatcher tests must patch the singleton methods actually invoked by bound
# handlers, not stale module aliases.
cli_test = ROOT / "tools/nsamdr/tests/test_nsamdr_cli.py"
text = cli_test.read_text(encoding="utf-8")
text = text.replace(
    '_SPEC.loader.exec_module(CLI)\n\n\nPUBLIC_COMMAND_PATHS',
    '_SPEC.loader.exec_module(CLI)\nAPP = CLI._n_s_a_m_d_r_command_line_application\n\n\nPUBLIC_COMMAND_PATHS',
)
text = text.replace('mock.patch.object(CLI, "_command_workflow"', 'mock.patch.object(APP, "_command_workflow"')
text = text.replace('mock.patch.object(CLI, "_python_script"', 'mock.patch.object(APP, "_python_script"')
text = text.replace('mock.patch.object(CLI, "validate_layout"', 'mock.patch.object(APP, "validate_layout"')
text = text.replace('mock.patch.object(CLI, "_command_test"', 'mock.patch.object(APP, "_command_test"')
cli_test.write_text(text, encoding="utf-8")

# 5. Strategy-candidate tests likewise patch the object that owns the bound methods.
strategy_test = ROOT / "tools/nsamdr/tests/test_strategy_candidates.py"
text = strategy_test.read_text(encoding="utf-8")
text = text.replace(
    '''        monkeypatch.setattr(\n            generator,\n            "_model_input",\n''',
    '''        monkeypatch.setattr(\n            generator._strategy_candidate_generator,\n            "_model_input",\n''',
)
text = text.replace(
    'monkey.setattr(generator, "_dimensions", lambda _path: [1, 1])',
    'monkey.setattr(generator._strategy_candidate_generator, "_dimensions", lambda _path: [1, 1])',
)
strategy_test.write_text(text, encoding="utf-8")

# 6. Keep the architecture diagrams discoverable from the README.
readme = ROOT / "tools/nsamdr/README.md"
text = readme.read_text(encoding="utf-8")
if "NSAMDR_EVOLUTIONARY_RECOVERY_ARCHITECTURE.png" not in text:
    text += (\n        "\n## Architecture diagrams\n\n"
        "- `NSAMDR_FULL_SYSTEM_ARCHITECTURE.png` — complete production pipeline.\n"
        "- `NSAMDR_EVOLUTIONARY_RECOVERY_ARCHITECTURE.png` — bounded training-only evolutionary recovery.\n"
    )
readme.write_text(text, encoding="utf-8")

# 7. Add durable tests for the new hardening contract.
provenance_test = tests / "test_source_revision_preflight_contract.py"
provenance_test.write_text('''from __future__ import annotations\n\nfrom pathlib import Path\nimport sys\n\nROOT = Path(__file__).resolve().parents[3]\nNEURAL = ROOT / "tools/nsamdr/neural"\nif str(NEURAL) not in sys.path:\n    sys.path.insert(0, str(NEURAL))\n\n\ndef test_preflight_validates_exact_trainer_contract_and_explicit_refiner():\n    from raven_architecture_contract import RavenArchitectureContract, _COMPONENT_PATHS\n    from v9 import FidelityResidualNetV9, V9Config\n\n    audit = RavenArchitectureContract()\n    config = V9Config()\n    model = FidelityResidualNetV9(config)\n    audit._validate_trainer_contract(ROOT, model.architecture_contract())\n    assert _COMPONENT_PATHS["ExplicitRefiner"] == (\n        "geometry_net.production_structure.geometry_refiner",\n    )\n\n\ndef test_preflight_fingerprints_all_structural_v12_owners():\n    from raven_architecture_contract import RavenArchitectureContract\n\n    hashes = RavenArchitectureContract()._source_fingerprints(ROOT)\n    for relative in (\n        "tools/nsamdr/neural/v9/local_boundary_production_contract.py",\n        "tools/nsamdr/neural/v9/explicit_spline_refiner.py",\n        "tools/nsamdr/neural/v9/edge_constrained_spline_graph.py",\n        "tools/nsamdr/neural/v9/spline_graph.py",\n        "tools/nsamdr/neural/v9/application/backend.py",\n    ):\n        assert relative in hashes\n        assert len(hashes[relative]) == 64\n\n\ndef test_git_provenance_records_exact_head_and_clean_tracked_state():\n    from raven_architecture_contract import RavenArchitectureContract\n\n    provenance = RavenArchitectureContract()._git_provenance(ROOT)\n    assert provenance["available"] is True\n    assert isinstance(provenance["head"], str) and len(provenance["head"]) == 40\n    assert provenance["trackedDirty"] is False\n''', encoding="utf-8")
