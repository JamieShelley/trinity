from pathlib import Path

path = Path("tools/nsamdr/tests/test_live_eve_training_preview_contract.py")
text = path.read_text(encoding="utf-8")
old = "    assert 'live ? ValidateLiveCandidateProvenance' in processing\n"
new = "    assert 'const std::string provenanceFailure = live' in processing\n    assert '? ValidateLiveCandidateProvenance(resources, rawAlbedoPath, candidate)' in processing\n"
if text.count(old) != 1:
    raise SystemExit(f"expected one formatting-specific assertion, found {text.count(old)}")
path.write_text(text.replace(old, new), encoding="utf-8")
