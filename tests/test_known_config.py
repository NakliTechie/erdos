"""Integration re-audit of the known-good 132-edge config (n=40).

First cross-module consumer test: pushes data/udg40_132edges.csv through the
REAL package (udg.configio -> udg.counting -> udg.audit -> udg.moser) and
asserts every property documented in DATA_APPENDIX.md sections E and F:

- 132 unit edges at tol=1e-9; audit passed (min_sep 0.283 in (0.28, 0.29),
  0 K_{2,3} violations, Gauss-Newton total residual ~1.1e-27 < 1e-24,
  132/132 edges exact at 1e-12, max point move ~4.3e-11 < 1e-9).
- Degree sequence (desc): [10,9,8,8,8] + [7]*19 + [6]*12 + [5,4,4,3].
- 134 distinct distances among the 780 pairs (rounded to 1e-9).
- 12 undirected edge directions in 4 families of 3, each family internally
  60 degrees apart; family offsets ~ {0.04, 0.94, 16.87, 34.12} deg.
- Moser-lattice alignment: 6 of 12 directions match (the 0-degree and
  16.78-degree families); the 0.94 and 34.12 families are floating hinges.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from udg.audit import audit
from udg.configio import load_csv
from udg.counting import dist_matrix, unit_edges
from udg.moser import direction_families, lattice_id

CSV = Path(__file__).resolve().parents[1] / "data" / "udg40_132edges.csv"

DEGREE_SEQ = [10, 9, 8, 8, 8] + [7] * 19 + [6] * 12 + [5, 4, 4, 3]


@pytest.fixture(scope="module")
def P() -> np.ndarray:
    return load_csv(CSV)


@pytest.fixture(scope="module")
def edges(P) -> list[tuple[int, int]]:
    return unit_edges(P)


@pytest.fixture(scope="module")
def report(P):
    return audit(P)


def test_load_shape(P):
    assert P.shape == (40, 2)
    assert P.dtype == np.float64


def test_exactly_132_unit_edges(edges):
    assert len(edges) == 132
    # canonical i<j orientation, no duplicates
    assert all(i < j for i, j in edges)
    assert len(set(edges)) == 132


def test_audit_passes_all_documented_thresholds(report):
    assert report.passed is True
    assert report.n == 40
    assert report.n_edges == 132
    # DATA_APPENDIX E: min separation 0.283
    assert 0.28 < report.min_sep < 0.29
    assert report.k23_violations == 0
    # Gauss-Newton projection: total residual 1.1e-27, all 132 exact at 1e-12,
    # max point movement 4.3e-11
    assert report.gn_total_residual < 1e-24
    assert report.gn_edges_exact == 132
    assert report.gn_max_move < 1e-9
    # projection must not break separation either
    assert report.min_sep_after >= 0.2


def test_degree_sequence(P, edges):
    deg = np.zeros(len(P), dtype=int)
    for i, j in edges:
        deg[i] += 1
        deg[j] += 1
    assert sorted(deg.tolist(), reverse=True) == DEGREE_SEQ
    assert int(deg.sum()) == 2 * 132


def test_distinct_distances_134(P):
    D = dist_matrix(P)
    iu, ju = np.triu_indices(len(P), k=1)
    d = D[iu, ju]
    assert d.size == 780
    # DATA_APPENDIX F: 134 distinct distances at 1e-9 rounding
    assert len(np.unique(np.round(d, 9))) == 134


def test_direction_families_12_in_4_families_of_3(P, edges):
    fams = direction_families(P, edges, cluster_tol=0.5)
    assert len(fams) == 12
    # every edge is in exactly one family
    assert sum(c for c, _ in fams) == 132

    # group the 12 mean angles into families by their offset mod 60
    angles = sorted(a for _, a in fams)
    offsets = sorted(a % 60.0 for a in angles)
    groups: list[list[float]] = [[offsets[0]]]
    for o in offsets[1:]:
        if o - groups[-1][-1] < 0.6:
            groups[-1].append(o)
        else:
            groups.append([o])
    # circular wrap mod 60 (not triggered here, but correct in general)
    if len(groups) > 1 and (groups[0][0] + 60.0 - groups[-1][-1]) < 0.6:
        groups[0] = [o - 60.0 for o in groups.pop()] + groups[0]

    assert len(groups) == 4
    assert all(len(g) == 3 for g in groups)

    # each family's three angles are internally 60 degrees apart
    for g in groups:
        center = float(np.mean(g)) % 60.0
        members = sorted(a for a in angles if min(abs(a % 60.0 - center),
                                                  60.0 - abs(a % 60.0 - center)) < 0.6)
        assert len(members) == 3
        gaps = np.diff(members)
        assert np.all(np.abs(gaps - 60.0) < 0.6)

    # DATA_APPENDIX F: family offsets ~ {0.04, 0.94, 16.87, 34.12} deg
    family_offsets = sorted(float(np.mean(g)) % 60.0 for g in groups)
    expected = [0.04, 0.94, 16.87, 34.12]
    assert np.allclose(family_offsets, expected, atol=0.05)


def test_lattice_id_matches_6_of_12(P, edges):
    lid = lattice_id(P, edges, match_tol=0.15)
    assert lid["n_dirs"] == 12
    assert lid["n_matched"] == 6
    assert lid["matched_mask"].sum() == 6
    assert len(lid["dirs"]) == 12

    # DATA_APPENDIX F: the matched directions are the 0-degree and
    # 16.78-degree families; the 0.94 and 34.12 families float (hinges).
    matched_offsets = sorted(float(d) % 60.0 for d in lid["dirs"][lid["matched_mask"]])
    assert np.allclose(
        matched_offsets, [0.04, 0.04, 0.04, 16.87, 16.87, 16.87], atol=0.05
    )
    unmatched_offsets = [float(d) % 60.0 for d in lid["dirs"][~lid["matched_mask"]]]
    for o in unmatched_offsets:
        assert min(abs(o - 0.94), abs(o - 34.12)) < 0.05
