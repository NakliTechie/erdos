"""Regression tests for udg.bisector — exact targets from DATA_APPENDIX §D (n=1600).

The random and perturbed-grid fixtures depend on EXACT rng call order
(exp3_bisector.py): default_rng(7), the random draw FIRST, then the jitter
draw. Both are drawn once in a module-scoped fixture to preserve that order.
"""

import numpy as np
import pytest

from udg.bisector import (
    BisectorResult,
    bisector_energy_float,
    bisector_energy_int,
    bisector_energy_tri,
)

M = 40
N = M * M                      # 1600
NUM_PAIRS = N * (N - 1) // 2   # 1_279_200


def square_grid(m: int) -> np.ndarray:
    return np.array([(x, y) for x in range(m) for y in range(m)],
                    dtype=np.int64)


@pytest.fixture(scope="module")
def rng_draws():
    """Replicate exp3's rng call order exactly: random first, then jitter."""
    rng = np.random.default_rng(7)
    rand = rng.integers(0, 10 ** 7, size=(M * M, 2))
    pg = square_grid(M) * 10 + rng.integers(-1, 2, size=(M * M, 2))
    return rand, pg


def test_square_grid_40x40():
    res = bisector_energy_int(square_grid(M))
    assert isinstance(res, BisectorResult)
    assert res.num_pairs == NUM_PAIRS
    assert res.E == 66_748_264
    assert res.E_nontrivial == 66_352_916
    assert res.max_mult == 800


def test_tri_lattice_40x40():
    # exp3 tri_energy point construction: (2i+j, j) meaning (2i+j, j*sqrt(3))
    pts = np.array([(2 * i + j, j) for i in range(M) for j in range(M)],
                   dtype=np.int64)
    res = bisector_energy_tri(pts)
    assert res.num_pairs == NUM_PAIRS
    assert res.E == 67_742_200
    assert res.E_nontrivial == 67_255_450
    assert res.max_mult == 780


def test_random_points(rng_draws):
    rand, _ = rng_draws
    res = bisector_energy_int(rand)
    assert res.num_pairs == NUM_PAIRS
    assert res.E == 1_279_200          # trivial: every pair its own line
    assert res.E == res.num_pairs
    assert res.E_nontrivial == 0
    assert res.max_mult == 1


def test_perturbed_grid(rng_draws):
    _, pg = rng_draws
    res = bisector_energy_int(pg)
    assert res.num_pairs == NUM_PAIRS
    assert res.E == 2_573_622
    assert res.E_nontrivial == 1_378_169
    assert res.max_mult == 95


def test_float_unit_square():
    # Unit square: bisectors are x=1/2 (mult 2), y=1/2 (mult 2), and the two
    # diagonals (mult 1 each) -> E = 4+4+1+1, E_nontrivial = 8.
    P = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
    res = bisector_energy_float(P)
    assert res.num_pairs == 6
    assert res.E_nontrivial > 0
    assert res.E == 10
    assert res.E_nontrivial == 8
    assert res.num_lines == 4
    assert res.max_mult == 2


def test_float_matches_int_on_small_grid():
    # On a small integer grid the bucketed float energy must agree with the
    # exact integer energy (lines are well separated at decimals=6).
    G = square_grid(6)
    ri = bisector_energy_int(G)
    rf = bisector_energy_float(G.astype(np.float64))
    assert rf.E == ri.E
    assert rf.E_nontrivial == ri.E_nontrivial
    assert rf.num_lines == ri.num_lines
    assert rf.max_mult == ri.max_mult
