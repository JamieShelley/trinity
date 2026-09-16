"""NSAMDR V16.2 structure-conditioning components.

Keep package initialization lightweight.  The broad-prior utilities and structure
target helpers must be importable without importing the active V14 reconstruction
runtime.  Consumers that need the conditioned model should import it explicitly
from ``v16.conditioning`` (script mode) or ``tools.nsamdr.neural.v16.conditioning``
(package mode).
"""

from .structure import (
    StructureConditioningEncoder,
    StructureTargetConfig,
    derive_structure_targets,
    structure_channels,
)

__all__ = [
    "StructureConditioningEncoder",
    "StructureTargetConfig",
    "derive_structure_targets",
    "structure_channels",
]
