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
from udg.moser import direction_families, lattice_id
from udg.search import SearchResult, multi_search, search

__all__ = [
    "AuditReport",
    "SearchResult",
    "audit",
    "bisector_energy_float",
    "bisector_energy_int",
    "bisector_energy_tri",
    "direction_families",
    "lattice_id",
    "load_csv",
    "multi_search",
    "popular",
    "save_csv",
    "search",
    "unit_count",
    "unit_edges",
]
