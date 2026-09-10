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

# B1b SGD is driven by the exact rendered structural candidate; historical proxy
# geometry losses remain telemetry and cannot purchase a worse production render.
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

__all__ = ["V9Config", "FidelityResidualNetV9"]
