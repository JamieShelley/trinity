#!/usr/bin/env python3
"""Generate a deterministic linked-zoom comparison page for V9 experiments."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from v9.experiments import experiment_dir, latest_validation_contact_sheet

KEYS = (
    "learning_rate",
    "weight_decay",
    "batch_size",
    "regret_weight",
    "edge_weight",
    "detail_laplacian_weight",
    "geometric_alignment_weight",
    "tangent_coherence_weight",
    "curvature_coherence_weight",
    "synthetic_geometry_probability",
    "boundary_sampling_probability",
    "seed",
)


class ExperimentComparisonApplication:
    # Purpose: Implement file uri for ExperimentComparisonApplication.
    # Called by: _card
    # Calls: No same-class helper methods.
    def _file_uri(self, path: Path) -> str:
        return path.resolve().as_uri()

    # Purpose: Implement card for ExperimentComparisonApplication.
    # Called by: main
    # Calls: _file_uri
    def _card(self, repo_root: Path, experiment_id: str) -> tuple[str, dict]:
        directory = experiment_dir(repo_root, experiment_id)
        metrics_path = directory / "metrics.json"
        config_path = directory / "resolved_config.json"
        if not metrics_path.is_file() or not config_path.is_file():
            raise RuntimeError(f"experiment is not complete: {experiment_id}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        image = latest_validation_contact_sheet(repo_root, experiment_id)
        rows = "".join(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(config.get(key)))}</td></tr>"
            for key in KEYS
        )
        body = f"""
        <section class=\"card\">
          <h2>{html.escape(experiment_id)}</h2>
          <div class=\"metrics\">Best epoch: {html.escape(str(metrics.get('bestEpoch')))} &nbsp; | &nbsp;
          Validation: {html.escape(str(metrics.get('bestValidationTotal')))} &nbsp; | &nbsp;
          Acceptance: {html.escape(str(metrics.get('acceptancePass')))}</div>
          <div class=\"viewport\"><img class=\"sync-image\" src=\"{html.escape(self._file_uri(image))}\"></div>
          <details><summary>Resolved configuration</summary><table>{rows}</table></details>
        </section>
        """
        return body, {"id": experiment_id, "image": str(image), "metrics": metrics, "config": config}

    # Purpose: Implement main for ExperimentComparisonApplication.
    # Called by: External callers and the owning workflow.
    # Calls: _card
    def main(self) -> int:
        parser = argparse.ArgumentParser(description="Compare completed NSAMDR V9 tuning experiments")
        parser.add_argument("--repo-root", type=Path, default=Path.cwd())
        parser.add_argument("--experiments", nargs="+", required=True)
        parser.add_argument("--no-open", action="store_true")
        args = parser.parse_args()
        root = args.repo_root.resolve()
        ids = []
        for value in args.experiments:
            value = value.strip().upper()
            if value and value not in ids:
                ids.append(value)
        if len(ids) < 2:
            raise RuntimeError("compare requires at least two distinct completed experiments")
        if len(ids) > 3:
            raise RuntimeError("compare currently supports up to three experiments")

        cards = []
        records = []
        for experiment_id in ids:
            card, record = self._card(root, experiment_id)
            cards.append(card)
            records.append(record)

        output_dir = root / "artifacts/nsamdr/experiments/compare"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / ("compare_" + "_".join(ids) + ".html")
        content = f"""<!doctype html>
    <html><head><meta charset=\"utf-8\"><title>NSAMDR V9 Experiment Compare</title>
    <style>
    html,body{{background:#050505;color:#eee;font-family:Segoe UI,Arial,sans-serif;margin:0;padding:0}}
    header{{position:sticky;top:0;background:#111;padding:10px 16px;border-bottom:1px solid #333;z-index:5}}
    button{{margin-right:6px;padding:6px 12px}}
    .grid{{display:grid;grid-template-columns:repeat({len(cards)},minmax(0,1fr));gap:10px;padding:10px}}
    .card{{background:#111;border:1px solid #333;padding:8px;min-width:0}}
    .card h2{{margin:2px 0 6px}}
    .metrics{{font-size:13px;color:#bbb;margin-bottom:6px}}
    .viewport{{height:620px;overflow:hidden;background:#000;border:1px solid #333;cursor:grab;position:relative}}
    .viewport:active{{cursor:grabbing}}
    .sync-image{{transform-origin:0 0;user-select:none;pointer-events:none;max-width:none}}
    table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border:1px solid #333;padding:4px;text-align:left}}
    summary{{cursor:pointer;margin-top:8px}}
    </style></head>
    <body><header><b>NSAMDR V9 deterministic Raven comparison</b> &nbsp;
    <span>Same fixed held-out set, linked pan/zoom.</span><br>
    <button onclick=\"setScale(1)\">100%</button><button onclick=\"setScale(2)\">200%</button>
    <button onclick=\"setScale(4)\">400%</button><button onclick=\"resetView()\">Reset</button>
    </header><main class=\"grid\">{''.join(cards)}</main>
    <script>
    let scale=1, tx=0, ty=0, dragging=false, sx=0, sy=0;
    const imgs=[...document.querySelectorAll('.sync-image')];
    function apply(){{imgs.forEach(i=>i.style.transform=`translate(${{tx}}px,${{ty}}px) scale(${{scale}})`);}}
    function setScale(v){{scale=v;apply();}}
    function resetView(){{scale=1;tx=0;ty=0;apply();}}
    document.querySelectorAll('.viewport').forEach(v=>{{
     v.addEventListener('wheel',e=>{{e.preventDefault();scale=Math.max(.5,Math.min(8,scale*(e.deltaY<0?1.15:.87)));apply();}},{{passive:false}});
     v.addEventListener('mousedown',e=>{{dragging=true;sx=e.clientX-tx;sy=e.clientY-ty;}});
    }});
    window.addEventListener('mousemove',e=>{{if(dragging){{tx=e.clientX-sx;ty=e.clientY-sy;apply();}}}});
    window.addEventListener('mouseup',()=>dragging=false); apply();
    </script></body></html>"""
        output_path.write_text(content, encoding="utf-8")
        manifest = {
            "schema": "NSAMDR_V9_EXPERIMENT_COMPARE_V1",
            "experiments": ids,
            "output": str(output_path),
            "records": records,
            "comparisonGuarantees": [
                "same fixed Raven dataset manifest",
                "same held-out crop identities",
                "same deterministic validation seed",
                "linked pan and zoom",
            ],
        }
        output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Comparison: {output_path}")
        if not args.no_open:
            if os.name == "nt":
                os.startfile(output_path)  # type: ignore[attr-defined]
            else:
                import webbrowser
                webbrowser.open(output_path.as_uri())
        return 0

_experiment_comparison_application = ExperimentComparisonApplication()
_file_uri = _experiment_comparison_application._file_uri
_card = _experiment_comparison_application._card
main = _experiment_comparison_application.main


if __name__ == "__main__":
    raise SystemExit(main())
