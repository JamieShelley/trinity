"""Persisted training state and failed-attempt artifact ownership."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import torch

from ..config import V9Config
from .gates import QualificationGates


class TrainingStateService:
    """Own persisted stage state inspection, promotion, and structural-attempt reset."""

    def __init__(self, gates: QualificationGates) -> None:
        """Compose metric extraction needed by local-geometry promotion.

        Purpose:
            Keep persistence operations independent from pipeline control flow.
        Called by:
            TrainingApplication._build_pipeline().
        Calls:
            No project functions.
        """
        self.gates = gates

    def snapshot(self, directory: Path, config: V9Config) -> dict[str, Any]:
        """Read persisted qualification state used to resume the stage plan.

        Purpose:
            Reconstruct only the orchestration facts needed between trainer invocations.
        Called by:
            PassDrivenPipeline.run(), PassDrivenPipeline structural recovery.
        Calls:
            torch.load(), json.loads().
        """
        snapshot: dict[str, Any] = {
            "topologyBootstrapped": False,
            "geometryQualified": False,
            "renderQualified": False,
            "seamReconstructionQualified": False,
            "seamAuthorityQualified": False,
            "detailQualified": False,
        }
        state_path = directory / config.training_state_name
        if state_path.is_file():
            try:
                state = torch.load(state_path, map_location="cpu", weights_only=False)
            except Exception:
                state = {}
            if isinstance(state, dict):
                snapshot.update(
                    {
                        "topologyBootstrapped": bool(state.get("topology_bootstrapped", False)),
                        "geometryQualified": bool(state.get("structure_qualified", False)),
                        "renderQualified": bool(state.get("render_qualified", False)),
                        "seamReconstructionQualified": bool(
                            state.get("seam_reconstruction_qualified", False)
                        ),
                        "seamAuthorityQualified": bool(
                            state.get("seam_authority_qualified", False)
                        ),
                        "detailQualified": bool(state.get("detail_qualified", False)),
                    }
                )

        metadata_path = directory / config.metadata_name
        if metadata_path.is_file():
            try:
                previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                previous = {}
            if isinstance(previous, dict):
                snapshot["bestSyntheticSdfValidation"] = previous.get(
                    "bestSyntheticSdfValidation"
                )
        return snapshot

    def promote_local_geometry(
        self,
        directory: Path,
        config: V9Config,
        metadata: dict[str, Any],
    ) -> None:
        """Persist direct local structure/redraw qualification for canonical resume.

        Purpose:
            Advance only the retired B1b resume cursor after real B1/B2 success and seed downstream restore files.
        Called by:
            PassDrivenPipeline._complete_structural_stage().
        Calls:
            QualificationGates.local_geometry_metrics(), torch.load(), torch.save().
        """
        state_path = directory / config.training_state_name
        if not state_path.is_file():
            raise RuntimeError(f"local geometry stage produced no training state: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise RuntimeError("local geometry training state is not a mapping")

        state["topology_bootstrapped"] = True
        state["structure_qualified"] = True
        state["render_qualified"] = True

        learned_epoch = int(state.get("completed_epoch", 0))
        structural_end = int(config.identity_epochs)
        if learned_epoch < structural_end:
            raise RuntimeError(
                f"local geometry promotion occurred too early: epoch={learned_epoch} "
                f"expected>={structural_end}"
            )

        retired_end = int(config.identity_epochs) + int(config.residual_epochs)
        retired_skipped = max(0, retired_end - learned_epoch)
        state["completed_epoch"] = max(learned_epoch, retired_end)
        state["retired_b1b_epochs_skipped"] = retired_skipped
        state["retired_b1b_reason"] = (
            "whole-tile primitive classifier removed from V11.4 production structure"
        )
        torch.save(state, state_path)

        metrics = self.gates.local_geometry_metrics(metadata)
        payload = {
            "schema": str(state.get("schema", "")),
            "config": config.to_dict(),
            "phase": "sdf-bootstrap",
            "epoch": learned_epoch,
            "selection_kind": "v114-evolutionary-local-boundary-structure-redraw",
            "metrics": metrics,
            "qualified": True,
            "retired_b1b_epochs_skipped": retired_skipped,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in state["state_dict"].items()
            },
        }
        torch.save(payload, directory / "best_b2_redraw.pt")
        torch.save(payload, directory / "best_b1b_geometry.pt")
        print(
            f"[pipeline] retired whole-tile B1b slot skipped: {retired_skipped} epoch(s); "
            f"resume cursor={int(state['completed_epoch'])}",
            flush=True,
        )

    def archive_structural_attempt(
        self,
        directory: Path,
        config: V9Config,
        *,
        attempt: int,
    ) -> Path:
        """Archive a failed B1/B2 attempt and reset only in-progress structural state.

        Purpose:
            Preserve failed evidence while allowing deterministic retry with an evolved genome.
        Called by:
            PassDrivenPipeline._recover_structural_failure().
        Calls:
            shutil.copy2(), Path.unlink().
        """
        archive = directory / "evolution" / f"failed_structural_attempt_{attempt:02d}"
        archive.mkdir(parents=True, exist_ok=True)
        names = {
            str(getattr(config, "training_state_name", "nsamdr_v9_training_state.pt")),
            str(getattr(config, "metadata_name", "nsamdr_v9_fidelity.json")),
            str(getattr(config, "checkpoint_name", "nsamdr_v9_fidelity.pt")),
            "training_log.csv",
            "best_b1a_topology.pt",
            "best_b1b_geometry.pt",
            "best_b2_redraw.pt",
        }
        for name in sorted(names):
            source = directory / name
            if not source.is_file():
                continue
            target = archive / source.name
            shutil.copy2(source, target)
            source.unlink()
        return archive
