"""Spatial failure mechanism diagnostics (P2.1) and zero-shot region mitigation (P2.2-P2.3)."""

from .mechanism import decompose_pair_matrix, analyze_frozen_spatial_runs
from .quadrant_mitigation import relational_scores_from_regions, evaluate_geometric_upper_bound

__all__ = [
    "decompose_pair_matrix",
    "analyze_frozen_spatial_runs",
    "relational_scores_from_regions",
    "evaluate_geometric_upper_bound",
]

