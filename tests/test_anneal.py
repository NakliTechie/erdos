"""Tests for udg.anneal -- the exact-lattice simulated annealer promoted from
runs/chase/n70/ (the engine that crossed the 280-attractor and tied n=70/281).

Record values asserted here are densest-known per the local Engel DB /
arXiv:2406.15317 Table 2: u(12) = 27, u(70) = 281 (seed config 277).
"""

from __future__ import annotations

import numpy as np
import pytest

from udg.anneal import (
    MINSEP,
    anneal_task,
    beam_grow,
    candidates,
    canon_t,
    drop_worst,
    edge_count,
    greedy_add,
    min_sep,
    mink_sum_arr,
    plateau_search,
    rot_w1,
    rot_w3_eis,
    steepest_climb,
)
from udg.mlgraph import (
    MLConfig,
    canon,
    exact_edge_count,
    minkowski,
    tri_patch,
    unit_triangle,
    wheel6,
)

TRI = list(unit_triangle().points)


# ---------------------------------------------------------------------------
# exact helpers agree with udg.mlgraph
# ---------------------------------------------------------------------------

def test_edge_count_matches_mlgraph_on_wheel_sum():
    s = minkowski(wheel6("w1"), wheel6("w3"))
    arr = s.as_array().astype(np.int64)
    assert edge_count(arr) == exact_edge_count(s) == 180


def test_min_sep_of_triangle_is_one():
    assert min_sep(np.array(TRI, dtype=np.int64)) == pytest.approx(1.0)


def test_candidates_exclude_existing_points():
    arr = np.array(TRI, dtype=np.int64)
    cand = candidates(arr, two_step=False)
    existing = set(map(tuple, arr.tolist()))
    assert existing.isdisjoint(set(map(tuple, cand.tolist())))
    # every 1-step candidate is at exact unit distance from some point
    from udg.mlgraph import unit_mask

    assert unit_mask(cand, arr).any(axis=1).all()


def test_rot_w1_and_rot_w3_preserve_edge_counts():
    pts = list(tri_patch(2).points)
    e0 = edge_count(np.array(pts, dtype=np.int64))
    e1 = edge_count(np.array(rot_w1(pts), dtype=np.int64))
    e3 = edge_count(np.array(rot_w3_eis(pts), dtype=np.int64))
    assert e0 == e1 == e3 == 9
    # rotations are congruences: same canonical class
    assert canon(np.array(pts, dtype=np.int64)) == canon(
        np.array(rot_w1(pts), dtype=np.int64)
    )


def test_mink_sum_arr_matches_mlgraph_minkowski():
    A = list(wheel6("w1").points)
    B = list(wheel6("w3").points)
    S = mink_sum_arr(A, B)
    S2 = minkowski(wheel6("w1"), wheel6("w3")).as_array().astype(np.int64)
    assert set(map(tuple, S.tolist())) == set(map(tuple, S2.tolist()))


def test_canon_t_is_translation_invariant_only():
    pts = list(tri_patch(2).points)
    shifted = [(a + 3, b - 2, c + 1, d) for (a, b, c, d) in pts]
    assert canon_t(pts) == canon_t(shifted)


# ---------------------------------------------------------------------------
# local moves
# ---------------------------------------------------------------------------

def test_greedy_add_then_drop_worst_roundtrip_monotone():
    pts = greedy_add(TRI, k=3)
    e = edge_count(np.array(pts, dtype=np.int64))
    assert len(pts) == 6 and e >= 6  # greedy on a triangle gains >=1 per add
    back = drop_worst(pts, k=3)
    assert len(back) == 3
    assert min_sep(np.array(pts, dtype=np.int64)) >= MINSEP


def test_steepest_climb_never_loses_edges():
    pts = greedy_add(TRI, k=5)
    e0 = edge_count(np.array(pts, dtype=np.int64))
    out, e1 = steepest_climb(list(pts))
    assert e1 >= e0
    assert min_sep(np.array(out, dtype=np.int64)) >= MINSEP


def test_plateau_search_returns_best_at_least_start():
    import random

    pts = greedy_add(TRI, k=5)
    e0 = edge_count(np.array(pts, dtype=np.int64))
    bp, be, fp, fe = plateau_search(list(pts), random.Random(0), steps=20)
    assert be >= e0
    assert edge_count(np.array(bp, dtype=np.int64)) == be


def test_beam_grow_small_sizes_are_optimal():
    lib = beam_grow(4, beam=64, per_parent=24)
    # u(2)=1, u(3)=3, u(4)=5 (unit rhombus)
    assert lib[2][0][0] == 1
    assert lib[3][0][0] == 3
    assert lib[4][0][0] == 5


# ---------------------------------------------------------------------------
# the annealer itself
# ---------------------------------------------------------------------------

def test_anneal_task_tiny_n12_reaches_record_27():
    """Exact smoke: greedy 12-point seed (21 edges) -> annealer reaches the
    densest-known u(12) = 27 (per the local Engel DB) in well under a second."""
    pts = greedy_add(TRI, k=9)  # deterministic greedy to n=12
    assert edge_count(np.array(pts, dtype=np.int64)) == 21
    e, best = anneal_task((pts, 1, 4_000, 1.0, 0.05, 1))
    assert len(best) == 12
    assert len(set(best)) == 12
    assert e == edge_count(np.array(best, dtype=np.int64)) == 27
    assert min_sep(np.array(best, dtype=np.int64)) >= MINSEP


def test_anneal_task_is_deterministic_per_seed():
    pts = greedy_add(TRI, k=7)  # n=10
    e1, b1 = anneal_task((pts, 7, 1_500, 1.0, 0.05, 0))
    e2, b2 = anneal_task((pts, 7, 1_500, 1.0, 0.05, 0))
    assert e1 == e2 and b1 == b2


def test_anneal_from_certified_277_reaches_at_least_278():
    """The promotion smoke: seeded from the certified n=70 277-edge config,
    one anneal task must cross into the 280-attractor (>= 278; the recipe
    that ultimately tied 281). ~3 s with these parameters."""
    cfg = MLConfig.from_json("data/mlcoords/udg70_277edges.json")
    pts = list(cfg.points)
    assert edge_count(cfg.as_array().astype(np.int64)) == 277
    e, best = anneal_task((pts, 1, 20_000, 1.0, 0.05, 1))
    assert e >= 278
    arr = np.array(best, dtype=np.int64)
    assert len(best) == 70 and len(set(best)) == 70
    assert edge_count(arr) == e
    assert min_sep(arr) >= MINSEP
