"""Tests for udg.mlgraph — the exact discrete Moser-lattice toolkit.

The validations that matter (assignment spec):
1. 6-wheel: 7 vertices, exactly 12 exact edges.
2. minkowski(6-wheel, 6-wheel) with the two wheels on DIFFERENT sublattice
   families = 49 vertices, 180 exact edges (Engel et al. Table 2 record at
   n=49); the same-family sum collapses to the 19-point hexagon.
3. data/mlcoords certificates reproduce their exact edge counts:
   n=40 -> 136 (both classes), n=50 -> 181, n=70 -> 277.
4. greedy_improve on a deliberately weakened 136-edge config recovers
   >= 136 edges at n=40.

Plus: int fast path == Fraction reference path, library shapes, float
export consistency (counting.unit_edges on to_float == exact_edges), CSV
round-trip, Minkowski algebra, local-move invariants, and the float
three-audit pipeline passing on a config produced from exact coords.
"""

from __future__ import annotations

import itertools
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from udg.audit import audit
from udg.configio import load_csv
from udg.counting import unit_edges
from udg.mlgraph import (
    MLConfig,
    add_best_point,
    candidate_positions,
    degrees,
    dist2_components,
    drop_worst_point,
    exact_dist2,
    exact_edge_count,
    exact_edges,
    exact_xy,
    greedy_improve,
    is_unit,
    is_unit_exact,
    minkowski,
    moser_spindle,
    save_csv,
    to_float,
    tri_patch,
    unit_edge,
    unit_rhombus,
    unit_triangle,
    wheel6,
)
from udg.moser import UNIT_COEFFS, W1, W3

MLCOORDS = Path(__file__).resolve().parents[1] / "data" / "mlcoords"
ORIGIN = (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Exact arithmetic: fast integer path == Fraction reference path
# ---------------------------------------------------------------------------

def test_all_18_unit_coeffs_are_exact_units_on_both_paths():
    for c in UNIT_COEFFS:
        assert is_unit(c, ORIGIN), c
        assert is_unit_exact(c, ORIGIN), c
        assert dist2_components(c, ORIGIN) == (1, 0, 0, 0)


def test_non_units_rejected():
    assert not is_unit(ORIGIN, ORIGIN)               # |0|^2 = 0
    assert not is_unit((2, 0, 0, 0), ORIGIN)         # |2|^2 = 4
    assert not is_unit((1, 1, 0, 0), ORIGIN)         # |1+w1|^2 = 3
    assert not is_unit((1, 0, -1, 0), ORIGIN)        # |1-w3|^2 = 1/3
    assert dist2_components((1, 0, -1, 0), ORIGIN) == (Fraction(1, 3), 0, 0, 0)


def test_int_path_matches_reference_on_random_tuples():
    rng = random.Random(20260610)
    pts = [tuple(rng.randint(-4, 4) for _ in range(4)) for _ in range(40)]
    n_units = 0
    for p, q in itertools.combinations(pts, 2):
        fast = is_unit(p, q)
        ref = is_unit_exact(p, q)
        assert fast == ref, (p, q)
        n_units += fast
        # the scaled-invariant identity reproduces the rational + s33
        # components of |diff|^2 exactly (s3/s11 components vanish)
        d2 = exact_dist2(p, q)
        da, db, dc, dd = (a - b for a, b in zip(p, q))
        A = 12 * da + 6 * db + 10 * dc + 5 * dd
        B = 6 * db + 5 * dd
        C = 2 * dc + dd
        D = -dd
        assert d2.a == Fraction(A * A + 3 * B * B + 11 * C * C + 33 * D * D, 144)
        assert d2.d == Fraction(2 * (A * D + B * C), 144)
        assert d2.b == 0 and d2.c == 0
    assert n_units > 0  # the sample actually exercises the unit branch


def test_exact_xy_matches_float_basis():
    basis = [1.0 + 0.0j, W1, W3, W1 * W3]
    rng = random.Random(7)
    for _ in range(50):
        p = tuple(rng.randint(-5, 5) for _ in range(4))
        z = sum(m * b for m, b in zip(p, basis))
        re, im = exact_xy(p)
        assert abs(re.to_float() - z.real) < 1e-12
        assert abs(im.to_float() - z.imag) < 1e-12


# ---------------------------------------------------------------------------
# MLConfig basics
# ---------------------------------------------------------------------------

def test_mlconfig_dedup_order_and_set_equality():
    cfg = MLConfig([(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0)])
    assert len(cfg) == 2
    assert cfg.points == ((0, 0, 0, 0), (1, 0, 0, 0))  # first occurrence wins
    assert (1, 0, 0, 0) in cfg and (2, 0, 0, 0) not in cfg
    assert cfg == MLConfig([(1, 0, 0, 0), (0, 0, 0, 0)])  # set equality
    assert hash(cfg) == hash(MLConfig([(1, 0, 0, 0), (0, 0, 0, 0)]))


def test_mlconfig_rejects_bad_points():
    with pytest.raises(ValueError):
        MLConfig([(1, 0, 0)])  # wrong arity
    with pytest.raises(ValueError):
        MLConfig([(0.5, 0, 0, 0)])  # non-integer


def test_translate_preserves_exact_edges():
    cfg = moser_spindle()
    moved = cfg.translate((3, -2, 1, 5))
    assert exact_edge_count(moved) == exact_edge_count(cfg) == 11


def test_from_generators_two_unit_gens_is_the_rhombus():
    cfg = MLConfig.from_generators([(1, 0, 0, 0), (0, 1, 0, 0)])
    assert cfg == unit_rhombus()
    assert exact_edge_count(cfg) == 5


def test_from_generators_three_edges_cube():
    # "Minkowski sum of 3 edges": 2^3 = 8 vertices when disjoint
    cfg = MLConfig.from_generators([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)])
    assert len(cfg) == 8
    assert exact_edge_count(cfg) >= 12  # each of the 3 gens contributes 4 parallel edges


# ---------------------------------------------------------------------------
# Validation 1 + library shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cfg, n, m",
    [
        (unit_edge(), 2, 1),
        (unit_triangle(), 3, 3),
        (unit_rhombus(), 4, 5),
        (wheel6("w1"), 7, 12),
        (wheel6("w3"), 7, 12),
        (moser_spindle(), 7, 11),
    ],
    ids=["edge", "triangle", "rhombus", "wheel6_w1", "wheel6_w3", "spindle"],
)
def test_library_shapes(cfg, n, m):
    assert len(cfg) == n
    edges = exact_edges(cfg)
    assert len(edges) == m
    assert all(i < j for i, j in edges)
    assert len(set(edges)) == m


def test_wheel6_rejects_unknown_family():
    with pytest.raises(ValueError):
        wheel6("w2")


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_tri_patch_counts(k):
    cfg = tri_patch(k)
    assert len(cfg) == (k + 1) * (k + 2) // 2
    assert exact_edge_count(cfg) == 3 * k * (k + 1) // 2


# ---------------------------------------------------------------------------
# Validation 2: the n=49 record as a Minkowski sum of two 6-wheels
# ---------------------------------------------------------------------------

def test_minkowski_two_wheels_different_families_is_the_49_180_record():
    s = minkowski(wheel6("w1"), wheel6("w3"))
    assert len(s) == 49
    assert exact_edge_count(s) == 180  # Engel et al. Table 2, n=49


def test_minkowski_two_wheels_same_family_collapses_to_hexagon():
    # w1^k + w1^(k+2) = w1^(k+1) etc.: the same-sublattice sum is NOT
    # disjoint -- it collapses to the 19-point hexagon H(2)
    s = minkowski(wheel6("w1"), wheel6("w1"))
    assert len(s) == 19
    assert exact_edge_count(s) == 42


def test_minkowski_identity_and_translation():
    spindle = moser_spindle()
    assert minkowski(spindle, MLConfig([ORIGIN])) == spindle
    shifted = minkowski(spindle, MLConfig([(2, 1, 0, -1)]))
    assert shifted == spindle.translate((2, 1, 0, -1))


# ---------------------------------------------------------------------------
# Validation 3: certified configs reproduce their exact counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fname, n, m",
    [
        ("udg40_136edges.json", 40, 136),
        ("udg40_136edges_b.json", 40, 136),
        ("udg50_181edges.json", 50, 181),
        ("udg70_277edges.json", 70, 277),
    ],
)
def test_certificates_reproduce_exact_edge_counts(fname, n, m):
    cfg = MLConfig.from_json(MLCOORDS / fname)
    assert len(cfg) == n
    assert exact_edge_count(cfg) == m


# ---------------------------------------------------------------------------
# Float export: to_float / save_csv / audit pipeline
# ---------------------------------------------------------------------------

def test_to_float_agrees_with_float_unit_edges_on_certificates():
    for fname in ["udg40_136edges.json", "udg70_277edges.json"]:
        cfg = MLConfig.from_json(MLCOORDS / fname)
        P = to_float(cfg)
        assert P.shape == (len(cfg), 2) and P.dtype == np.float64
        assert set(unit_edges(P, tol=1e-9)) == set(exact_edges(cfg))


def test_to_float_agrees_on_the_49_record():
    s = minkowski(wheel6("w1"), wheel6("w3"))
    assert set(unit_edges(to_float(s), tol=1e-9)) == set(exact_edges(s))


def test_save_csv_roundtrip(tmp_path):
    s = minkowski(wheel6("w1"), wheel6("w3"))
    path = tmp_path / "wheel49.csv"
    save_csv(s, path)
    P = load_csv(path)
    np.testing.assert_array_equal(P, to_float(s))  # configio round-trip is exact


def test_49_record_passes_the_float_three_audit_pipeline(tmp_path):
    # any config claimed from exact coords must also pass the float audits
    s = minkowski(wheel6("w1"), wheel6("w3"))
    path = tmp_path / "wheel49.csv"
    save_csv(s, path)
    report = audit(load_csv(path))
    assert report.n_edges == 180
    assert report.passed, report


# ---------------------------------------------------------------------------
# Local moves
# ---------------------------------------------------------------------------

def test_candidate_positions_exclude_existing_and_are_adjacent():
    cfg = wheel6("w1")
    cands = candidate_positions(cfg)
    assert len(cands) > 0
    assert cfg._set.isdisjoint(cands)
    for c in cands:
        assert any(is_unit(c, p) for p in cfg.points)
    # every neighbor of every point is present
    expected = {
        tuple(p[i] + u[i] for i in range(4)) for p in cfg.points for u in UNIT_COEFFS
    } - cfg._set
    assert set(cands) == expected


def test_degrees_sum_is_twice_edge_count():
    cfg = MLConfig.from_json(MLCOORDS / "udg40_136edges.json")
    deg = degrees(cfg)
    assert int(deg.sum()) == 2 * 136


def test_drop_worst_point_removes_min_degree():
    cfg = moser_spindle()
    before = exact_edge_count(cfg)
    min_deg = int(degrees(cfg).min())
    smaller = drop_worst_point(cfg)
    assert len(smaller) == len(cfg) - 1
    assert exact_edge_count(smaller) == before - min_deg


def test_add_best_point_restores_a_broken_wheel():
    wheel = wheel6("w1")
    broken = wheel.without_point((1, 0, 0, 0))  # ring point: degree 3
    assert exact_edge_count(broken) == 12 - 3
    repaired = add_best_point(broken)
    assert len(repaired) == 7
    assert exact_edge_count(repaired) >= 12  # the missing ring slot gains 3


def test_greedy_improve_never_loses_edges():
    cfg = wheel6("w1")
    out = greedy_improve(cfg, passes=2)
    assert len(out) == len(cfg)
    assert exact_edge_count(out) >= 12


# ---------------------------------------------------------------------------
# Validation 4: greedy recovery of a weakened record config
# ---------------------------------------------------------------------------

def test_greedy_improve_recovers_weakened_136_config():
    cfg = MLConfig.from_json(MLCOORDS / "udg40_136edges.json")
    assert exact_edge_count(cfg) == 136
    # drop the MAX-degree vertex (worst damage), add a useless far-away point
    deg = degrees(cfg)
    v = int(np.argmax(deg))
    lost = int(deg[v])
    bad = (25, 0, 0, 0)
    weak = cfg.without_index(v).with_point(bad)
    assert len(weak) == 40
    assert int(degrees(weak)[-1]) == 0  # the bad point really is isolated
    assert exact_edge_count(weak) == 136 - lost
    repaired = greedy_improve(weak, passes=4)
    assert len(repaired) == 40
    assert exact_edge_count(repaired) >= 136
