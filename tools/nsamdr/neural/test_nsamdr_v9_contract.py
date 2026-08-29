#!/usr/bin/env python3
"""Fail-fast semantic contract for the cleaned NSAMDR production workflow.

This is intentionally lightweight. It verifies that Raven Quick and Full route
through the same production model/config semantics and that the removed
capability/generalisation paths cannot become runnable product entry points.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_nsamdr_v9_preview_experiment import (  # noqa: E402
    CANONICAL_SEMANTIC_OVERRIDES,
    DATASET_SCOPE_FIELDS,
    FULL_MINIMUM_WORK_BUDGET,
    QUICK_WORK_BUDGET,
    _canonical_overrides,
    _set_values,
)
from v9.config import V9Config  # noqa: E402
from v9.model import (  # noqa: E402
    INPUT_CHANNELS,
    MODEL_SCHEMA,
    UPSCALE_FACTOR,
    FidelityResidualNetV9,
)


FULL_CONFIG = ROOT / "tools/nsamdr/neural/configs/v9_fidelity_full.json"
RAVEN_CONFIG = ROOT / "tools/nsamdr/neural/configs/v9_preview_raven.json"
RUNNER = ROOT / "tools/nsamdr/neural/run_nsamdr_v9_raven_tune_preview.py"
TRAINER = ROOT / "tools/nsamdr/neural/train_nsamdr_v9_preview_experiment.py"
CLI = ROOT / "tools/nsamdr/nsamdr_cli.py"
README = ROOT / "tools/nsamdr/README.md"

REMOVED_ROUTES = (
    "scripts/build/run_nsamdr_raven_architecture_locked.bat",
    "scripts/build/run_nsamdr_raven_capability_first.bat",
    "scripts/build/run_nsamdr_raven_capability_full_renderer.bat",
    "scripts/build/run_nsamdr_raven_capability_generalization.bat",
    "scripts/build/run_nsamdr_raven_capability_viewer.bat",
    "tools/nsamdr/gui/raven_capability_viewer.py",
    "tools/nsamdr/neural/raven_capability_first.py",
    "tools/nsamdr/neural/raven_capability_full_asset_probe.py",
    "tools/nsamdr/neural/raven_capability_generalization.py",
    "tools/nsamdr/neural/run_raven_capability_generalization_latest.py",
)

REQUIRED_COMPONENTS = {
    "geometry": "geometry_net",
    "structural representation": "geometry_net.parametric_primitive_field",
    "boundary renderer": "boundary_renderer",
    "boundary/profile": "boundary_specialist",
    "PhaseAwareSeamSR": "seam_restorer.phase_sr",
    "seam authority": "seam_restorer.authority",
    "conditioned detail": "detail_net",
    "albedo physical head": "detail_net.albedo_head",
    "normal physical head": "detail_net.normal_head",
    "material physical head": "detail_net.material_head",
    "confidence": "detail_net.confidence_head",
    "regret": "detail_net.regret_head",
    "BenefitSelector": "benefit_selector",
}


class TestNsamdrV9Contract:
    # Purpose: Implement require for TestNsamdrV9Contract.
    # Called by: _verify_production_model_contract, _verify_quick_full_semantic_identity, _verify_routing_and_cleanup
    # Calls: No same-class helper methods.
    def _require(self, condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(message)

    # Purpose: Implement workflow args for TestNsamdrV9Contract.
    # Called by: _resolved
    # Calls: No same-class helper methods.
    def _workflow_args(self, training_mode: str) -> argparse.Namespace:
        return argparse.Namespace(
            training_mode=training_mode,
            tiles_per_epoch=None,
            validation_tiles=None,
            performance_profile="optimized",
            workers=4,
            prefetch_factor=2,
            amp_precision="auto",
        )

    # Purpose: Implement resolved for TestNsamdrV9Contract.
    # Called by: _verify_quick_full_semantic_identity
    # Calls: _workflow_args
    def _resolved(self, base: V9Config, data: V9Config, mode: str) -> V9Config:
        value = copy.deepcopy(base)
        _set_values(value, _canonical_overrides(self._workflow_args(mode), base, data))
        value.validate()
        return value

    # Purpose: Implement verify quick full semantic identity for TestNsamdrV9Contract.
    # Called by: main
    # Calls: _require, _resolved
    def _verify_quick_full_semantic_identity(self) -> None:
        base = V9Config.load(FULL_CONFIG)
        raven = V9Config.load(RAVEN_CONFIG)
        quick = self._resolved(base, raven, "quick")
        full = self._resolved(base, base, "full")

        allowed_difference = set(DATASET_SCOPE_FIELDS)
        allowed_difference.update(QUICK_WORK_BUDGET)
        allowed_difference.update(FULL_MINIMUM_WORK_BUDGET)

        quick_values = quick.to_dict()
        full_values = full.to_dict()
        illegal = {
            key: (quick_values.get(key), full_values.get(key))
            for key in sorted(set(quick_values) | set(full_values))
            if key not in allowed_difference and quick_values.get(key) != full_values.get(key)
        }
        self._require(
            not illegal,
            "Raven Quick diverges from Full in semantic fields: " + repr(illegal),
        )

        for key, expected in CANONICAL_SEMANTIC_OVERRIDES.items():
            self._require(getattr(quick, key) == expected, f"Quick semantic invariant differs: {key}")
            self._require(getattr(full, key) == expected, f"Full semantic invariant differs: {key}")

    # Purpose: Implement verify production model contract for TestNsamdrV9Contract.
    # Called by: main
    # Calls: _require
    def _verify_production_model_contract(self) -> None:
        config = V9Config.load(FULL_CONFIG)
        model = FidelityResidualNetV9(config)

        self._require(UPSCALE_FACTOR == 4, f"production upscale is {UPSCALE_FACTOR}, expected 4")
        self._require(INPUT_CHANNELS == 17, f"production input channels are {INPUT_CHANNELS}, expected 17")
        self._require("4X" in str(MODEL_SCHEMA).upper(), f"MODEL_SCHEMA is not a 4x schema: {MODEL_SCHEMA}")

        signature = inspect.signature(FidelityResidualNetV9.forward)
        self._require(
            tuple(signature.parameters) == ("self", "inputs"),
            "public production forward exposes override/cached authority: " + str(signature),
        )

        contract = model.architecture_contract()
        self._require(contract.get("schema") == MODEL_SCHEMA, "architecture contract schema mismatch")
        components = contract.get("productionComponents")
        self._require(isinstance(components, dict), "productionComponents is missing")
        for label, path in REQUIRED_COMPONENTS.items():
            self._require(components.get(label) == path, f"production component mismatch: {label}")

        self._require(
            contract.get("productionForward") == "FidelityResidualNetV9.forward(inputs) with no override authority",
            "production forward contract does not explicitly forbid override authority",
        )
        self._require(contract.get("detailReconstructionEnabled") is True, "detail reconstruction is disabled")
        self._require(contract.get("directionalSeamEnabled") is True, "phase-aware seam reconstruction is disabled")

    # Purpose: Implement verify routing and cleanup for TestNsamdrV9Contract.
    # Called by: main
    # Calls: _require
    def _verify_routing_and_cleanup(self) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        trainer = TRAINER.read_text(encoding="utf-8")
        cli = CLI.read_text(encoding="utf-8")
        readme = README.read_text(encoding="utf-8")

        self._require("train_nsamdr_v9_preview_experiment.py" in runner, "canonical runner bypasses production trainer")
        self._require("raven_architecture_contract.py" in runner, "canonical runner does not run architecture audit")
        self._require("preview_nsamdr_v9_experiment.py" in runner, "canonical runner does not use production preview")
        self._require("--training-mode" in runner, "Quick/Full mode is not a work-budget selector")
        self._require("train_v9(" in trainer, "experiment wrapper does not call production train_v9")

        forbidden_tokens = (
            "raven_capability_first",
            "raven_capability_generalization",
            "capability_generalization",
            "capability_first",
        )
        canonical_source = "\n".join((runner, trainer, cli))
        for token in forbidden_tokens:
            self._require(token not in canonical_source, f"obsolete alternate route remains reachable: {token}")

        for relative in REMOVED_ROUTES:
            self._require(not (ROOT / relative).exists(), f"obsolete alternate route still exists: {relative}")

        self._require('"raven-quick"' in cli, "canonical CLI has no raven-quick command")
        self._require('"full-train"' in cli, "canonical CLI has no full-train command")
        self._require('"preview"' in cli, "canonical CLI has no qualified experiment preview command")

        self._require(
            "NSAMDR** stands for **Neural Structure-Aware Material Detail Reconstruction" in readme,
            "README does not expand the NSAMDR name",
        )
        self._require(
            "Raven Quick uses the complete production NSAMDR model" in readme,
            "README does not state the Raven architecture invariant",
        )

    # Purpose: Implement main for TestNsamdrV9Contract.
    # Called by: External callers and the owning workflow.
    # Calls: _verify_production_model_contract, _verify_quick_full_semantic_identity, _verify_routing_and_cleanup
    def main(self) -> int:
        self._verify_quick_full_semantic_identity()
        self._verify_production_model_contract()
        self._verify_routing_and_cleanup()
        print("NSAMDR production semantic contract passed")
        print(f"  model={FidelityResidualNetV9.__name__}")
        print(f"  schema={MODEL_SCHEMA}")
        print("  Raven Quick divergence=dataset/work-budget only")
        print("  alternate capability/generalisation routes=absent")
        return 0

_test_nsamdr_v9_contract = TestNsamdrV9Contract()
_require = _test_nsamdr_v9_contract._require
_workflow_args = _test_nsamdr_v9_contract._workflow_args
_resolved = _test_nsamdr_v9_contract._resolved
_verify_quick_full_semantic_identity = _test_nsamdr_v9_contract._verify_quick_full_semantic_identity
_verify_production_model_contract = _test_nsamdr_v9_contract._verify_production_model_contract
_verify_routing_and_cleanup = _test_nsamdr_v9_contract._verify_routing_and_cleanup
main = _test_nsamdr_v9_contract.main


if __name__ == "__main__":
    raise SystemExit(main())
