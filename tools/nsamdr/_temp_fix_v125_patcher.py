from pathlib import Path

path = Path("tools/nsamdr/_temp_harden_v125.py")
text = path.read_text(encoding="utf-8")
old = 'text += (\\n        "\\n## Architecture diagrams\\n\\n"'
new = 'text += (\n        "\\n## Architecture diagrams\\n\\n"'
if old not in text:
    raise SystemExit("bad README newline anchor not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
