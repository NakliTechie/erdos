"""Tests for udg.search (contract: plan/contracts.md, section src/udg/search.py)."""

import numpy as np

from udg.search import MIN_SEP, TOL, SearchResult, circle_intersections, multi_search, search


# --- inline helpers (module self-containment: tests don't import other udg modules) ---

def _dist_matrix(P):
    return np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))


def _min_separation(P):
    D = _dist_matrix(P)
    np.fill_diagonal(D, np.inf)
    return float(D.min())


def _unit_edges(P, tol=TOL):
    D = _dist_matrix(P)
    n = len(P)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if abs(D[i, j] - 1) < tol]


def _unit_rhombus():
    """Two unit triangles glued: 4 points, 5 unit edges."""
    s = np.sqrt(3) / 2
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.5, s], [0.5, -s]])


# --- circle_intersections ---

def test_circle_intersections_distance_one():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    pts = circle_intersections(a, b)
    assert len(pts) == 2
    for p in pts:
        assert abs(np.sqrt(((p - a) ** 2).sum()) - 1.0) < 1e-12
        assert abs(np.sqrt(((p - b) ** 2).sum()) - 1.0) < 1e-12
    # the two intersections are distinct (mirror images across the ab line)
    assert np.sqrt(((pts[0] - pts[1]) ** 2).sum()) > 1.0


def test_circle_intersections_far_apart_empty():
    a = np.array([0.0, 0.0])
    assert circle_intersections(a, np.array([2.0, 0.0])) == []
    assert circle_intersections(a, np.array([2.5, 0.0])) == []


def test_circle_intersections_coincident_centers_empty():
    a = np.array([0.7, -0.3])
    assert circle_intersections(a, a.copy()) == []


# --- search ---

def test_search_small_run():
    r = search(n=8, steps=8_000, seed=0)
    assert isinstance(r, SearchResult)
    assert r.n == 8 and r.steps == 8_000 and r.seed == 0
    assert r.P.shape == (8, 2)
    assert r.best_count >= 5
    # hard min-sep holds on the returned config
    assert _min_separation(r.P) >= MIN_SEP
    # all claimed edges within tol, and the incremental counter is exact
    edges = _unit_edges(r.P)
    assert len(edges) == r.best_count
    D = _dist_matrix(r.P)
    for i, j in edges:
        assert abs(D[i, j] - 1.0) < TOL


def test_search_deterministic_per_seed():
    r1 = search(n=8, steps=4_000, seed=3)
    r2 = search(n=8, steps=4_000, seed=3)
    assert r1.best_count == r2.best_count
    assert np.array_equal(r1.P, r2.P)


def test_search_warm_start_never_loses_seed_config():
    rhombus = _unit_rhombus()  # 5 unit edges
    extra = np.array([[3.0, 0.0], [3.0, 3.0], [0.0, 3.0], [-2.0, 2.0]])
    P0 = np.vstack([rhombus, extra])
    initial_count = len(_unit_edges(P0))
    assert initial_count == 5  # sanity on the fixture
    r = search(steps=3_000, seed=1, T0=0.05, T1=0.01, P0=P0)
    assert r.n == len(P0)
    assert r.P.shape == P0.shape
    assert r.best_count >= initial_count
    # warm start must not mutate the caller's array
    assert np.array_equal(P0[:4], rhombus)


def test_search_warm_start_deterministic():
    P0 = np.vstack([_unit_rhombus(), np.array([[3.0, 0.0], [3.0, 3.0], [0.0, 3.0], [-2.0, 2.0]])])
    r1 = search(steps=1_500, seed=5, T0=0.05, T1=0.01, P0=P0)
    r2 = search(steps=1_500, seed=5, T0=0.05, T1=0.01, P0=P0)
    assert r1.best_count == r2.best_count
    assert np.array_equal(r1.P, r2.P)


# --- multi_search ---

def test_multi_search_two_seeds():
    seeds = [0, 1]
    res = multi_search(8, seeds, 2_000)
    assert len(res) == 2
    # order of results = order of seeds
    assert [r.seed for r in res] == seeds
    for r in res:
        assert isinstance(r, SearchResult)
        assert r.n == 8 and r.steps == 2_000
        assert _min_separation(r.P) >= MIN_SEP
        assert len(_unit_edges(r.P)) == r.best_count
    # reproducible per seed: parallel result == serial search with the same seed
    for r in res:
        s = search(n=8, steps=2_000, seed=r.seed)
        assert s.best_count == r.best_count
        assert np.array_equal(s.P, r.P)
    # same seed twice -> same best_count
    res2 = multi_search(8, seeds, 2_000)
    assert [r.best_count for r in res2] == [r.best_count for r in res]
