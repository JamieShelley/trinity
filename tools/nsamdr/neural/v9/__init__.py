"""NSAMDR production package."""
from .config import V9Config
from .model import FidelityResidualNetV9
from .local_boundary_production_contract import install_local_boundary_model_contract

# Quick and Full instantiate the same production supernet. The training-only
# evolutionary controller may choose bounded genome values inside this fixed
# topology, but the chosen genome is checkpointed and the controller itself has
# no inference authority.
install_local_boundary_model_contract()

__all__ = ["V9Config", "FidelityResidualNetV9"]
