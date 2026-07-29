from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "validate_baseline.py"
spec = importlib.util.spec_from_file_location("validate_baseline", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class BaselineValidationTests(unittest.TestCase):
    def test_complete_report_passes(self) -> None:
        complete, blockers = module.validate({
            "complete": True,
            "areas": [{
                "group": 0,
                "areaName": "primary",
                "shaderFamily": "v5_packed",
                "semanticComplete": True,
                "parameterComplete": True,
                "baselineComplete": True,
                "missingSemantics": [],
            }],
        })
        self.assertTrue(complete)
        self.assertEqual(blockers, [])

    def test_unknown_shader_and_missing_semantics_block(self) -> None:
        complete, blockers = module.validate({
            "complete": False,
            "areas": [{
                "group": 4,
                "areaName": "tint-only",
                "shaderFamily": "unknown",
                "semanticComplete": False,
                "parameterComplete": False,
                "baselineComplete": False,
                "missingSemantics": ["sof_visual_manifest"],
            }],
        })
        self.assertFalse(complete)
        joined = "\n".join(blockers)
        self.assertIn("unknown shader family", joined)
        self.assertIn("sof_visual_manifest", joined)
        self.assertIn("material-slot parameters incomplete", joined)


if __name__ == "__main__":
    unittest.main()
