from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "tools/nsamdr/neural/v9/evolution/candidate.py"
FITNESS = ROOT / "tools/nsamdr/neural/v9/evolution/fitness.py"

text = CANDIDATE.read_text(encoding="utf-8")
old = '''        model.train()\n        parameters = [\n'''
new = '''        model.train()\n        # The disposable capacity candidate must use the same trainability contract\n        # as production B1b: topology fixed, neural continuous proposal enabled.\n        model.set_phase("sdf-proof")\n        model.set_parametric_substage("integration")\n        parameters = [\n'''
if text.count(old) != 1:
    raise RuntimeError(f"candidate phase block occurrence count={text.count(old)}")
CANDIDATE.write_text(text.replace(old, new, 1), encoding="utf-8")

text = FITNESS.read_text(encoding="utf-8")
old = "        topology_regression_fraction: float,\n"
new = "        topology_regression_fraction: float = 0.0,\n"
if text.count(old) != 1:
    raise RuntimeError(f"fitness compatibility argument occurrence count={text.count(old)}")
FITNESS.write_text(text.replace(old, new, 1), encoding="utf-8")

print("applied V12.2 B1b candidate phase alignment")
