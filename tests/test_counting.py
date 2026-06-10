"""Regression tests for udg.counting (DATA_APPENDIX §A/§B/§C exact targets)."""

import math

import numpy as np

from udg.counting import TOL, dist_matrix, popular, r2, unit_count, unit_edges


def square_grid(m: int) -> list[tuple[int, int]]:
    return [(x, y) for x in range(m) for y in range(m)]


# ---------------------------------------------------------------- §A grids

def test_square_grid_bests_appendix_A():
    # m, best d^2, U(n) from DATA_APPENDIX §A / exp1_grid.py
    targets = {5: (5, 48), 8: (5, 168), 12: (25, 456), 17: (25, 1136)}
    for m, (d2, count) in targets.items():
        n, top = popular(square_grid(m), topk=5)
        assert n == m * m
        assert top[0] == (d2, count)


# ----------------------------------------------------- §B 50x50 + r2 ratio

def test_grid50_top5_and_r2_ratio_appendix_B():
    n, top = popular(square_grid(50), topk=5)
    assert n == 2500
    assert top == [(325, 17680), (65, 16144), (425, 15640),
                   (85, 15440), (125, 14688)]
    # interior-point prediction: count <= n*r2(k)/2, ratio < 1 (boundary loss)
    expected_ratio = {325: 0.589, 65: 0.807, 425: 0.521, 85: 0.772, 125: 0.734}
    for d2, c in top:
        pred = n * r2(d2) // 2
        assert c <= pred
        assert round(2 * c / (n * r2(d2)), 3) == expected_ratio[d2]


def test_r2_small_values_exact():
    # hand-checked signed/ordered representation counts
    assert r2(0) == 1
    assert r2(1) == 4    # (+-1,0),(0,+-1)
    assert r2(2) == 4    # (+-1,+-1)
    assert r2(3) == 0
    assert r2(5) == 8
    assert r2(25) == 12
    assert r2(65) == 16
    assert r2(325) == 24
    assert r2(-1) == 0


# ------------------------------------------------ §C triangular form (1,1,1)

def test_triangular_form_top5_appendix_C():
    n, top = popular(square_grid(50), form=(1, 1, 1), topk=5)
    assert n == 2500
    assert top[0] == (91, 22120)
    assert dict(top) == {91: 22120, 133: 20720, 217: 18112,
                         49: 18107, 247: 17464}


def test_form_default_matches_squared_distance():
    rng = np.random.default_rng(3)
    P = rng.integers(-20, 21, size=(60, 2))
    n, top = popular(P, form=(1, 0, 1), topk=3)
    n2, top2 = popular(P, topk=3)
    assert n == n2 == 60
    assert top == top2


# ----------------------------------------------- unit_edges / unit_count

def unit_rhombus() -> np.ndarray:
    """Two unit triangles glued along (0,0)-(1,0): 4 points, 5 unit edges."""
    h = math.sqrt(3) / 2
    return np.array([(0.0, 0.0), (1.0, 0.0), (0.5, h), (0.5, -h)])


def test_unit_edges_rhombus_exact():
    P = unit_rhombus()
    E = unit_edges(P)
    assert set(E) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)}
    assert unit_count(P) == 5
    # long diagonal (2,3) has length sqrt(3), not unit
    assert abs(dist_matrix(P)[2, 3] - math.sqrt(3)) < 1e-12


def test_unit_edges_tolerance_behavior():
    # pair at distance 1 + 1e-6: excluded at TOL=1e-9, included at looser tol
    P = np.array([(0.0, 0.0), (1.0 + 1e-6, 0.0)])
    assert unit_edges(P, tol=TOL) == []
    assert unit_edges(P, tol=1e-5) == [(0, 1)]


def test_dist_matrix_basic():
    P = np.array([(0.0, 0.0), (3.0, 4.0)])
    D = dist_matrix(P)
    assert D.shape == (2, 2)
    assert D[0, 0] == 0.0 and D[1, 1] == 0.0
    assert abs(D[0, 1] - 5.0) < 1e-12
    assert D[0, 1] == D[1, 0]
