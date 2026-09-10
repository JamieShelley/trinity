"""NSAMDR production package."""
from .config import V9Config
from .model import FidelityResidualNetV9
from . import model as _model
from . import edge_constrained_spline_graph as _edge_constrained_spline

# V11.6 constrains every marching-squares crossing to its owning edge, replaces
# the B1b node teacher with the target zero crossing on that same edge, and
# vectorises the existing Hermite distance query. Install it before the local
# production contract captures the spline/loss callables.
_edge_constrained_spline.install()

from . import local_boundary_production_contract as _local_boundary
_edge_constrained_spline.install_schema(_local_boundary)

# The local-boundary contract is object-owned, but several of its methods are
# installed as extension methods on GeometryNet/FidelityResidualNetV9. Those
# callbacks must be unbound class functions so Python binds the target model
# instance as ``self``. Training-module callbacks remain bound to the contract
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
# an explicit argument. Keep them non-binding on GeometryNet so the existing
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

# V12.2 authority alignment is a production-graph contract, not a trainer-only
# convenience. Install it here so architecture preflight, standalone inference,
# Micro/Quick/Full and final preview all execute identical seam/selector/refiner
# semantics. TrainingBackend separately installs only the aligned loss adapter.
from . import authority_alignment_contract as _authority_alignment
_authority_alignment.install_authority_alignment_model_contract()

# V12.2.2 makes the exact rendered B1 candidate the sole sdf-proof SGD objective.
# Legacy spline/proxy terms remain telemetry only, preventing proxy improvement
# from purchasing a worse production render.
from . import b1_production_objective_contract as _b1_production_objective
_b1_production_objective.install_b1_production_objective_contract()

# V12.3 promotes the direct-residual capacity result into production composition:
# detail is reconstructed independently over deterministic B and the final selector
# no longer depends on learned geometry/seam evidence. Geometry and seam remain
# separately auditable specialists until they independently qualify.
from . import parallel_detail_contract as _parallel_detail
_parallel_detail.install_parallel_detail_contract()

__all__ = ["V9Config", "FidelityResidualNetV9"]