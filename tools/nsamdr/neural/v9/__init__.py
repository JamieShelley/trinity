"""NSAMDR production package and executable architecture-contract installation."""
from .config import V9Config
from .model import FidelityResidualNetV9
from . import model as _model
from . import edge_constrained_spline_graph as _edge_constrained_spline

# Legacy V11/V12 modules are installed only to preserve strict checkpoint topology
# and state-dict compatibility. V13.3 takes final ownership of active forward/loss/LR.
_edge_constrained_spline.install()

from . import local_boundary_production_contract as _local_boundary
_edge_constrained_spline.install_schema(_local_boundary)

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

_model.GeometryNet._require_current_v11_instance = staticmethod(
    _local_boundary._local_boundary_production_contract._require_current_v11_instance
)
_model.GeometryNet._geometry_encode = staticmethod(
    _local_boundary.LocalBoundaryProductionContract._geometry_encode
)

_local_boundary.install_local_boundary_model_contract()

from . import authority_alignment_contract as _authority_alignment
_authority_alignment.install_authority_alignment_model_contract()

from . import b1_production_objective_contract as _b1_production_objective
_b1_production_objective.install_b1_production_objective_contract()

from . import parallel_detail_contract as _parallel_detail
_parallel_detail.install_parallel_detail_contract()

from . import parallel_detail_optimization_contract as _parallel_detail_optimization
_parallel_detail_optimization.install_parallel_detail_optimization_contract()

from . import baseline_relative_specialist_contract as _baseline_relative_specialists
_baseline_relative_specialists.install_baseline_relative_specialist_contract()

from . import parallel_specialist_fusion_contract as _parallel_specialist_fusion
_parallel_specialist_fusion.install_parallel_specialist_fusion_contract()

from . import parallel_specialist_training_contract as _parallel_specialist_training
_parallel_specialist_training.install_parallel_specialist_training_contract()

from . import parallel_specialist_arbitration_contract as _parallel_specialist_arbitration
_parallel_specialist_arbitration.install_parallel_specialist_arbitration_contract()

from . import parallel_specialist_training_isolation_contract as _parallel_training_isolation
_parallel_training_isolation.install_parallel_specialist_training_isolation_contract()

from . import parallel_specialist_safety_contract as _parallel_specialist_safety
_parallel_specialist_safety.install_parallel_specialist_safety_contract()

from . import parallel_specialist_geometry_training_contract as _parallel_geometry_training
_parallel_specialist_geometry_training.install_parallel_specialist_geometry_training_contract()

from . import sr_first_contract as _sr_first
_sr_first.install_sr_first_contract()

from . import sr_first_quality_contract as _sr_first_quality
_sr_first_quality.install_sr_first_quality_contract()

from . import sr_first_quick_config_contract as _sr_first_quick_config
_sr_first_quick_config.install_sr_first_quick_config_contract()

from . import sr_first_generalization_contract as _sr_first_generalization
_sr_first_generalization.install_sr_first_generalization_contract()

# Final V13.3 authority. Everything above this line is compatibility/topology only.
# The active runtime is B -> direct multi-map SR C -> BenefitSelector F.
from . import sr_first_runtime_contract as _sr_first_runtime
_sr_first_runtime.install_sr_first_runtime_contract()

# Final strict-reload audit must enforce the same active graph and reject any
# accidental execution of the checkpoint-only geometry/profile/seam modules.
from . import sr_first_runtime_integrity_contract as _sr_runtime_integrity
_sr_runtime_integrity.install_sr_first_runtime_integrity_contract()

__all__ = ["V9Config", "FidelityResidualNetV9"]
