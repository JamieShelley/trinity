"""Real-Raven sample acquisition service for evolutionary microproofs."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class RavenSampleProvider:
    """Provide deterministic real Raven train/validation samples to evolution."""

    def __init__(self, repo_root: Path, config: Any, seed: int) -> None:
        """Capture immutable sample-provider inputs.

        Purpose:
            Separate dataset concerns from the evolutionary controller.
        Called by:
            EvolutionaryRecoveryController.__init__().
        Calls:
            Path.resolve().
        """
        self.repo_root = Path(repo_root).resolve()
        self.config = config
        self.seed = int(seed)

    def load_pair(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load one deterministic training and one validation Raven sample.

        Purpose:
            Supply a bounded capacity proof with actual deployment-domain evidence.
        Called by:
            EvolutionaryRecoveryController.search().
        Calls:
            load_dataset_manifest(), PhysicalTileDatasetV9.__getitem__().
        """
        from ..dataset import PhysicalTileDatasetV9, load_dataset_manifest

        manifest = load_dataset_manifest(self.repo_root, self.config)
        available = {str(record.get("split", "")) for record in manifest.get("crops", [])}
        train_split = "train" if "train" in available else next(iter(available), "train")
        if "validation" in available:
            validation_split = "validation"
        elif "val" in available:
            validation_split = "val"
        else:
            validation_split = train_split
        train = PhysicalTileDatasetV9(
            manifest, self.config, train_split, 1, seed=self.seed + 71001
        )[0]
        validation = PhysicalTileDatasetV9(
            manifest, self.config, validation_split, 1, seed=self.seed + 91001
        )[0]
        return train, validation
