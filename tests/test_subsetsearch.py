"""Tests for udg.subsetsearch -- the subset-in-closure + free ILS engine
promoted from runs/chase/n30/engine.py (re-derived the n=30 record, 93)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from udg.mlgraph import exact_edge_count, minkowski, unit_mask, wheel6
from udg.subsetsearch import (
    adjacency,
    canon_t,
    climb,
    edge_count,
    hex_patch,
    ils,
    neighbor_closure,
    perturb,
    subset_ils,
    subset_search,
    to_w3,
    wheel49,
)


# ---------------------------------------------------------------------------
# ambient builders
# ---------------------------------------------------------------------------

def test_wheel49_matches_mlgraph_wheel_sum():
    w = wheel49()
    direct = minkowski(wheel6("w1"), wheel6("w3"))
    assert w == direct
    assert len(w) == 49
    assert exact_edge_count(w) == 180


def test_hex_patch_sizes_and_edges():
    # |H(r)| = 1 + 3r(r+1); H(r) is a triangular-lattice hexagon
    assert len(hex_patch(1)) == 7
    assert len(hex_patch(2)) == 19
    assert len(hex_patch(3)) == 37
    assert exact_edge_count(hex_patch(1)) == 12  # the 6-wheel


def test_to_w3_maps_to_the_w3_sublattice():
    h = hex_patch(1)
    h3 = to_w3(h)
    assert len(h3) == 7
    assert exact_edge_count(h3) == 12  # rigid motion preserves edges
    assert all(p[0] == 0 and p[1] == 0 for p in h3.points)
    with pytest.raises(ValueError):
        to_w3(h3)  # not Eisenstein-only


def test_neighbor_closure_grows_and_contains_input():
    w = wheel49().as_array().astype(np.int64)
    c1 = neighbor_closure(w, 1)
    assert set(map(tuple, w.tolist())) <= set(map(tuple, c1.tolist()))
    assert len(c1) == 409  # the radius-1 ball around the wheel sum
    # closure-0 is the identity
    assert np.array_equal(neighbor_closure(w, 0), np.array(sorted(map(tuple, w.tolist())), dtype=np.int64))


def test_adjacency_is_exact_and_symmetric():
    w = wheel49().as_array().astype(np.int64)
    A = adjacency(w)
    assert A.dtype == bool
    assert (A == A.T).all()
    assert int(A.sum()) // 2 == 180
    assert not A.diagonal().any()


# ---------------------------------------------------------------------------
# subset mode
# ---------------------------------------------------------------------------

def test_subset_search_counts_are_consistent():
    """subset_search's incremental count equals the recomputed exact count."""
    w = wheel49().as_array().astype(np.int64)
    A = adjacency(w)
    np.fill_diagonal(A, False)
    for seed in range(4):
        m, idx = subset_search(A, 7, random.Random(seed), iters=4000)
        assert m == edge_count(w[idx])
        assert len(idx) == 7


def test_subset_ils_finds_the_full_wheel_inside_wheel49():
    """Densest 7-subset of the wheel49 host is a 12-edge 6-wheel; the kick
    loop finds it where plain restarts plateau at 11."""
    w = wheel49().as_array().astype(np.int64)
    m, pts = subset_ils(w, 7, seed=0, max_iters=200, target=12)
    assert m == 12
    assert edge_count(pts) == 12


def test_subset_ils_rederives_the_n30_record_93_in_wheel49_closure1():
    """THE promotion test: the n=30 record (93 edges, RESULTS.md 2026-06-10)
    re-derived as a densest-30-subgraph of closure-1 of the wheel49 sum.
    A few seconds across the restart ladder (first hit: seed 4)."""
    amb = neighbor_closure(wheel49().as_array().astype(np.int64), 1)
    best = 0
    for seed in range(8):
        m, pts = subset_ils(amb, 30, seed=seed, max_iters=120, target=93)
        best = max(best, m)
        if best >= 93:
            break
    assert best == 93
    assert len(pts) == 30
    assert edge_count(pts) == 93
    # the subset inherits exact lattice coords: verify edge count exactly
    assert int(unit_mask(pts, pts).sum()) // 2 == 93


def test_subset_ils_respects_max_iters_determinism():
    amb = neighbor_closure(wheel49().as_array().astype(np.int64), 1)
    m1, p1 = subset_ils(amb, 12, seed=3, max_iters=20)
    m2, p2 = subset_ils(amb, 12, seed=3, max_iters=20)
    assert m1 == m2
    assert np.array_equal(p1, p2)


def test_subset_ils_should_stop_halts_immediately():
    amb = neighbor_closure(wheel49().as_array().astype(np.int64), 1)
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return True

    m, pts = subset_ils(amb, 10, seed=0, max_iters=10_000, should_stop=stop)
    assert calls["n"] == 1  # stopped at the first kick boundary
    assert len(pts) == 10


# ---------------------------------------------------------------------------
# free mode
# ---------------------------------------------------------------------------

def test_climb_never_loses_edges_and_keeps_distinctness():
    w = wheel49().as_array().astype(np.int64)
    rng = random.Random(0)
    start = w[:10].copy()
    e0 = edge_count(start)
    e1, out = climb(start, rng)
    assert e1 >= e0
    assert len({tuple(r) for r in out.tolist()}) == 10


def test_perturb_keeps_points_distinct():
    w = wheel49().as_array().astype(np.int64)[:12]
    rng = random.Random(1)
    q = perturb(w, rng, 3)
    assert len({tuple(r) for r in q.tolist()}) == 12


def test_free_ils_improves_a_weak_start():
    rng_start = wheel49().as_array().astype(np.int64)[:8]
    e0 = edge_count(rng_start)
    m, best = ils(rng_start, seed=0, minutes=0.05, target=e0 + 1)
    assert m >= e0


def test_canon_t_translation_invariance():
    w = wheel49().as_array().astype(np.int64)[:6]
    assert canon_t(w) == canon_t(w + np.array([2, -1, 3, 0], dtype=np.int64))
