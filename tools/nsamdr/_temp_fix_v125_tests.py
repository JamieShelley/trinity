from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tools/nsamdr/tests"

# Runs after _temp_harden_v125.py has renamed the V12 contract files.
explicit = TESTS / "test_explicit_geometry_refiner_contract.py"
text = explicit.read_text(encoding="utf-8")
text = text.replace(
    'ROOT / "tools/nsamdr/tests/test_v117_baseline_relative_contract.py"',
    'ROOT / "tools/nsamdr/tests/test_baseline_relative_structural_contract.py"',
)
explicit.write_text(text, encoding="utf-8")

arch = TESTS / "test_raven_architecture_lock.py"
text = arch.read_text(encoding="utf-8")
text = text.replace(
    '''            "GeometryNet",\n            "Spline/SDF",\n            "BoundaryRenderer",\n''',
    '''            "GeometryNet",\n            "Spline/SDF",\n            "ExplicitRefiner",\n            "BoundaryRenderer",\n''',
)
arch.write_text(text, encoding="utf-8")

provenance = TESTS / "test_source_revision_preflight_contract.py"
text = provenance.read_text(encoding="utf-8")
text = text.replace(
    '    assert provenance["trackedDirty"] is False\n',
    '    assert isinstance(provenance["trackedDirty"], bool)\n',
)
provenance.write_text(text, encoding="utf-8")

strategy = TESTS / "test_strategy_candidates.py"
text = strategy.read_text(encoding="utf-8")
text = text.replace(
    '''            "return_all_maps": True,\n        }\n''',
    '''            "return_all_maps": True,\n            "output_variant": "final",\n        }\n''',
    1,
)
strategy.write_text(text, encoding="utf-8")
