"""udg — unit-distance graph toolkit (counting, audit, search, structure id).

Public API re-exported from the submodules; see plan/contracts.md.
"""

from udg.audit import AuditReport, audit
from udg.bisector import (
    bisector_energy_float,
    bisector_energy_int,
    bisector_energy_tri,
)
from udg.configio import load_csv, save_csv
from udg.counting import popular, unit_count, unit_edges
from udg.hinge import (
    Family,
    FamilyClassification,
    FireCheck,
    FlexResult,
    LockResult,
    classify_families,
    fire_check,
    flex_dimension,
    follow_flex,
    internal_flex_basis,
    lock_family,
    rigidity_matrix,
)
from udg.moser import direction_families, lattice_id
from udg.search import SearchResult, multi_search, search

__all__ = [
    "AuditReport",
    "Family",
    "FamilyClassification",
    "FireCheck",
    "FlexResult",
    "LockResult",
    "SearchResult",
    "audit",
    "bisector_energy_float",
    "bisector_energy_int",
    "bisector_energy_tri",
    "classify_families",
    "direction_families",
    "fire_check",
    "flex_dimension",
    "follow_flex",
    "internal_flex_basis",
    "lattice_id",
    "load_csv",
    "lock_family",
    "multi_search",
    "popular",
    "rigidity_matrix",
    "save_csv",
    "search",
    "unit_count",
    "unit_edges",
]
