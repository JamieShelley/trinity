"""NSAMDR production package."""
from .config import V9Config
from .model import FidelityResidualNetV9
from . import model as _model
from . import local_boundary_production_contract as _local_boundary

# The local-boundary contract is object-owned, but several of its methods are
# installed as extension methods on GeometryNet/FidelityResidualNetV9.  Those
# callbacks must be unbound class functions so Python binds the target model
# instance as ``self``.  Training-module callbacks remain bound to the contract
# owner because they are ordinary module callables rather than descriptors.
_local_boundary._geometry_init = _local_boundary.LocalBoundaryProductionContract._geometry_init
_local_boundary._geometry_encode = _local_boundary.LocalBoundaryProductionContract._geometry_encode
_local_boundary._geometry_forward = _local_boundary.LocalBoundaryProductionContract._geometry_forward
_local_boundary._geometry_query_from_outputs = (
    _local_boundary.LocalBoundaryProductionContract._geometry_query_from_outputs
)
_local_boundary._set_phase = _local_boundary.LocalBoundaryProductionContract._set_phase
_local_boundary._set_parametric_substage = (
    _local_boundary.LocalBoundaryProductionContract._set_parametric_substage
)
_local_boundary._architecture_contract = (
    _local_boundary.LocalBoundaryProductionContract._architecture_contract
)

# Two migrated helpers intentionally receive the target GeometryNet instance as
# an explicit argument.  Keep them non-binding on GeometryNet so the existing
# extension-method bodies preserve their original call signatures.
_model.GeometryNet._require_current_v11_instance = staticmethod(
    _local_boundary._local_boundary_production_contract._require_current_v11_instance
)
_model.GeometryNet._geometry_encode = staticmethod(
    _local_boundary.LocalBoundaryProductionContract._geometry_encode
)

# Quick and Full instantiate the same production supernet. The training-only
# evolutionary controller may choose bounded genome values inside this fixed
# topology, but the chosen genome is checkpointed and the controller itself has
# no inference authority.
_local_boundary.install_local_boundary_model_contract()

__all__ = ["V9Config", "FidelityResidualNetV9"]
