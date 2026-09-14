#!/usr/bin/env python3
"""Safe/live V16 Stage 2 runtime with explicit checkpoint selection.

The existing safe four-family runtime already supports automatic resume through the
latest resume pointer. This wrapper adds an explicit --resume-checkpoint option so
the GUI can offer a Preview-style selector for any retained Stage 2 checkpoint.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

# This file is launched directly by the GUI, so Python initially places this v14
# directory on sys.path rather than tools/nsamdr/neural. Add the package root before
# importing v14, matching the other direct diagnostic entry points.
NEURAL_ROOT = Path(__file__).resolve().parent.parent
if str(NEURAL_ROOT) not in sys.path:
    sys.path.insert(0, str(NEURAL_ROOT))

from v14 import safe_live_fourfamily_multiregion_diagnostic as live

safe = live.safe
base = safe.base

_original_parser = safe.parser
_original_safe_values = safe.SafeFourFamilyDiagnostic._safe_values
_original_compatible_resume = safe.SafeFourFamilyDiagnostic._compatible_resume


def _parser() -> Any:
    parser = _original_parser()
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help=(
            "Explicit Stage 2 resume_checkpoint.pt to resume. The checkpoint must "
            "match the current dataset, records, optimizer budget, learning rate, "
            "AMP precision, and V16 model schema."
        ),
    )
    return parser


def _safe_values(self: safe.SafeFourFamilyDiagnostic) -> dict[str, Any]:
    values = _original_safe_values(self)
    if getattr(self.args, "resume_checkpoint", None):
        values["resume"] = True
    return values


def _compatible_resume(
    self: safe.SafeFourFamilyDiagnostic,
    *,
    manifest: dict[str, Any],
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    max_steps: int,
) -> tuple[Path, dict[str, Any]] | None:
    requested = getattr(self.args, "resume_checkpoint", None)
    if requested is None:
        return _original_compatible_resume(
            self,
            manifest=manifest,
            train_records=train_records,
            validation_records=validation_records,
            max_steps=max_steps,
        )

    checkpoint = Path(requested)
    if not checkpoint.is_absolute():
        checkpoint = (self.repo_root / checkpoint).resolve()
    else:
        checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise RuntimeError(
            f"Selected Stage 2 resume checkpoint does not exist: {checkpoint}"
        )

    payload = safe._load_torch_payload(checkpoint)  # noqa: SLF001 - same runtime contract
    state = dict(payload.get("trainingState") or {})
    problems: list[str] = []

    if payload.get("schema") != safe.RESUME_SCHEMA:
        problems.append(
            f"resume schema {payload.get('schema')!r} != {safe.RESUME_SCHEMA!r}"
        )
    if payload.get("modelSchema") != base.MODEL_SCHEMA:
        problems.append("model schema does not match V16")
    if state.get("datasetFingerprint") != manifest.get("fingerprint"):
        problems.append("dataset fingerprint changed")
    if list(state.get("trainRecords") or []) != safe._record_paths(train_records):  # noqa: SLF001
        problems.append("training-region selection changed")
    if list(state.get("validationRecords") or []) != safe._record_paths(validation_records):  # noqa: SLF001
        problems.append("held-out-region selection changed")
    if abs(float(state.get("learningRate") or 0.0) - float(self.args.learning_rate)) >= 1.0e-12:
        problems.append("learning rate changed")
    if str(state.get("ampPrecision") or "") != str(self.args.amp_precision):
        problems.append("AMP precision changed")

    checkpoint_max_steps = int(state.get("maximumSteps") or 0)
    if checkpoint_max_steps and checkpoint_max_steps != int(max_steps):
        problems.append(
            f"training budget changed ({checkpoint_max_steps} -> {int(max_steps)})"
        )
    step = int(state.get("step") or 0)
    if step < 1 or step >= int(max_steps):
        problems.append(f"checkpoint step {step} is not resumable for budget {max_steps}")

    run_dir_text = str(state.get("runDir") or "").strip()
    if not run_dir_text:
        problems.append("checkpoint has no run directory")
    else:
        run_dir = Path(run_dir_text).resolve()
        if run_dir != checkpoint.parent.resolve():
            problems.append("checkpoint run directory does not match its parent directory")

    if problems:
        raise RuntimeError(
            "Selected Stage 2 checkpoint is incompatible; refusing to start a fresh "
            "run implicitly: " + "; ".join(problems)
        )

    print(
        f"[v16.0-stage2-safe] selected GUI resume checkpoint: "
        f"step {step}/{max_steps} {checkpoint}",
        flush=True,
    )
    return checkpoint, payload


safe.parser = _parser
safe.SafeFourFamilyDiagnostic._safe_values = _safe_values
safe.SafeFourFamilyDiagnostic._compatible_resume = _compatible_resume


def main(argv: list[str] | None = None) -> int:
    return live.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
