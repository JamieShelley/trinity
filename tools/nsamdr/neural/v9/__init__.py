"""NSAMDR production package and executable architecture-contract installation."""
from .config import V9Config
from .model import FidelityResidualNetV9
from . import model as _model
from . import edge_constrained_spline_graph as _edge_constrained_spline

# V11.6 owns same-edge topology safety and must install before the local production
# contract captures the spline/loss callables it extends.
_edge_constrained_spline.install()

from . import local_boundary_production_contract as _local_boundary
_edge_constrained_spline.install_schema(_local_boundary)

# The local-boundary owner exposes selected unbound class functions as extension
# methods. Keep Python descriptor binding explicit: model methods receive the model
# instance, while training-module callbacks remain normal module callables.
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

# These helpers intentionally receive GeometryNet explicitly rather than binding as
# instance methods; changing that calling convention would alter the production path.
_model.GeometryNet._require_current_v11_instance = staticmethod(
    _local_boundary._local_boundary_production_contract._require_current_v11_instance
)
_model.GeometryNet._geometry_encode = staticmethod(
    _local_boundary.LocalBoundaryProductionContract._geometry_encode
)

# Quick and Full use the same checkpointed production supernet. Evolution may tune
# bounded values during training but never installs a separate inference network.
_local_boundary.install_local_boundary_model_contract()

# V12.2 aligns the explicit refiner, seam proposal and BenefitSelector with the exact
# outputs they control in deployment.
from . import authority_alignment_contract as _authority_alignment
_authority_alignment.install_authority_alignment_model_contract()

# B1b keeps an exact deployed-render objective. V12.10 below restores direct
# same-edge node/tangent supervision alongside it rather than replacing deployment
# alignment with a proxy objective.
from . import b1_production_objective_contract as _b1_production_objective
_b1_production_objective.install_b1_production_objective_contract()

# V12.3 makes appearance/detail an independent bounded residual over deterministic B.
# Upstream geometry or seam failures therefore cannot poison the proven detail path.
from . import parallel_detail_contract as _parallel_detail
_parallel_detail.install_parallel_detail_contract()

# Reproduce the optimiser regime that demonstrated direct-detail capacity while
# preserving every other production optimiser group.
from . import parallel_detail_optimization_contract as _parallel_detail_optimization
_parallel_detail_optimization.install_parallel_detail_optimization_contract()

# V12.4 makes the baseline-centred specialist policy explicit. Authored HR supervises
# a >=99% preservation objective only during training/qualification; inference still
# consumes LR evidence only and retains the unchanged checkpoint topology.
from . import baseline_relative_specialist_contract as _baseline_relative_specialists
_baseline_relative_specialists.install_baseline_relative_specialist_contract()

# V12.5 completes the production composition: structure, seam and detail remain
# independent B-relative candidates, bounded support fusion builds the complete
# candidate, and BenefitSelector alone owns the final B-versus-candidate decision.
from . import parallel_specialist_fusion_contract as _parallel_specialist_fusion
_parallel_specialist_fusion.install_parallel_specialist_fusion_contract()

# Train seam authority and BenefitSelector against the exact V12.5 B-relative seam
# and fused-U candidates while retaining V12.4 protected preservation as the outer
# loss wrapper.
from . import parallel_specialist_training_contract as _parallel_specialist_training
_parallel_specialist_training.install_parallel_specialist_training_contract()

# V12.6 restores detail support-head supervision and sharpens specialist authority
# before convex fusion so weak candidates cannot linearly dilute a stronger one.
from . import parallel_specialist_arbitration_contract as _parallel_specialist_arbitration
_parallel_specialist_arbitration.install_parallel_specialist_arbitration_contract()

# V12.7 isolates support-head gradients from the direct-detail residual backbone,
# derives support labels strictly from D-versus-B benefit, and makes B1b supervise
# the exact deployed structure candidate G including structural residual authority.
from . import parallel_specialist_training_isolation_contract as _parallel_training_isolation
_parallel_training_isolation.install_parallel_specialist_training_isolation_contract()

# V12.8 makes the independently qualified D candidate the fusion anchor. G/S only
# contribute complementary low-conflict residuals where D support is weak, while
# selector supervision explicitly respects V12.4 protected-B drift.
from . import parallel_specialist_safety_contract as _parallel_specialist_safety
_parallel_specialist_safety.install_parallel_specialist_safety_contract()

# V12.10 restores the exact same-owning-edge spline node/tangent teacher as primary
# B1 continuous-geometry authority while retaining the deployed B-relative G
# objective. This changes training gradients only; inference/checkpoint topology is
# unchanged.
from . import parallel_specialist_geometry_training_contract as _parallel_geometry_training
_parallel_geometry_training.install_parallel_specialist_geometry_training_contract()

__all__ = ["V9Config", "FidelityResidualNetV9"]