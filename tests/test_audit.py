"""Tests for udg.audit — the three-audit gate (load-bearing module).

Covers the four contract-mandated cases plus a vectorization-equivalence
check against a naive per-edge-loop Gauss-Newton (the exp7 original) and an
unrealizability detection case (K4).
"""

import numpy as np
import pytest

from udg.audit import (
    MIN_SEP,
    TOL,
    AuditReport,
    audit,
    gauss_newton,
    k23_violations,
    min_separation,
)


# ---------------------------------------------------------------- fixtures


def unit_rhombus() -> np.ndarray:
    """Two unit triangles glued along edge AB: 4 points, exactly 5 unit edges
    (AB, AC, BC, AD, BD; CD = sqrt(3) is not unit)."""
    s = np.sqrt(3) / 2
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.5, s], [0.5, -s]])


def tolerance_exploit_config() -> np.ndarray:
    """The canonical fake config (pitfall HANDOFF §3 — the '400 edges at n=40'
    disaster). Two centers 1e-4 apart; three points spread TANGENTIALLY by
    1e-6 around an intersection of their unit circles. Distance error to the
    centers is second-order (~1e-10 .. 1e-12), so all 6 center-cluster pairs
    pass tol = 1e-9 — yet the config grossly violates min-sep and fakes a
    K_{2,3} (the two centers share 3 common unit-neighbors)."""
    d = 1e-4
    eps = 1e-6
    c1 = np.array([0.0, 0.0])
    c2 = np.array([d, 0.0])
    theta0 = np.arctan2(np.sqrt(1 - d * d / 4), d / 2)  # circle intersection
    pts = [c1, c2]
    for k in range(3):
        th = theta0 + k * eps
        pts.append(c1 + np.array([np.cos(th), np.sin(th)]))
    return np.array(pts)


def triangular_patch(m: int = 3) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """m x m parallelogram patch of the triangular lattice + its unit edges."""
    pts = [(i + 0.5 * j, j * np.sqrt(3) / 2) for j in range(m) for i in range(m)]
    P = np.array(pts)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    n = len(P)
    edges = [
        (i, j) for i in range(n) for j in range(i + 1, n) if abs(D[i, j] - 1) < TOL
    ]
    return P, edges


# ------------------------------------------------- 1. tolerance exploit


def test_tolerance_exploit_must_fail_audit():
    """Regression test for the '400 edges at n=40' disaster: a fake config
    whose every claimed edge passes tol = 1e-9 MUST NEVER pass audit."""
    P = tolerance_exploit_config()
    report = audit(P)

    # The exploit really does fake edges at tol: all 6 center-cluster pairs.
    assert report.n == 5
    assert report.n_edges == 6

    # ...but the audit catches it, on BOTH independent grounds:
    assert report.min_sep < MIN_SEP            # cluster points ~1e-6 apart
    assert report.min_sep < 1e-3
    assert report.k23_violations >= 1          # centers share 3 unit-neighbors

    assert report.passed is False


def test_tolerance_exploit_individual_checks():
    P = tolerance_exploit_config()
    assert min_separation(P) < 1e-3
    # The fake edge set is exactly the K_{2,3}: centers 0,1 vs cluster 2,3,4.
    edges = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]
    assert k23_violations(5, edges) >= 1


# ------------------------------------------------- 2. known-good config


def test_unit_rhombus_passes_audit():
    P = unit_rhombus()
    report = audit(P)
    assert isinstance(report, AuditReport)
    assert report.n == 4
    assert report.n_edges == 5                 # exactly the 5 edges constructed
    assert report.min_sep == pytest.approx(1.0)
    assert report.k23_violations == 0
    assert report.gn_total_residual < 1e-24
    assert report.gn_edges_exact == 5
    assert report.gn_max_move < 1e-9           # already exact; snap barely moves
    assert report.min_sep_after >= MIN_SEP
    assert report.passed is True


# ---------------------------------------- 3. Gauss-Newton convergence


def test_gauss_newton_converges_on_perturbed_patch():
    """Perturb an exact triangular patch by 1e-6; snap with edges from the
    UNPERTURBED config; must reach residual < 1e-24 with all edges exact."""
    P0, edges = triangular_patch(3)
    assert len(edges) == 16
    rng = np.random.default_rng(42)
    P = P0 + 1e-6 * rng.standard_normal(P0.shape)

    Q, tot = gauss_newton(P, edges)
    assert tot < 1e-24

    D = np.sqrt(((Q[:, None, :] - Q[None, :, :]) ** 2).sum(-1))
    assert all(abs(D[i, j] - 1.0) < 1e-12 for (i, j) in edges)
    # snap moves points by about the perturbation scale, no further
    assert np.sqrt(((Q - P) ** 2).sum(1)).max() < 1e-4


def test_gauss_newton_matches_naive_edge_loop():
    """The vectorized inner loop must reproduce the exp7 per-edge Python loop
    bit-for-bit in trajectory shape (same math, same update order)."""
    P0, edges = triangular_patch(3)
    rng = np.random.default_rng(7)
    P = P0 + 1e-4 * rng.standard_normal(P0.shape)
    iters = 50

    # exp7 original (naive loop), verbatim math
    Q_ref = P.copy()
    tot_ref = 0.0
    for _ in range(iters):
        grad = np.zeros_like(Q_ref)
        tot_ref = 0.0
        for (i, j) in edges:
            v = Q_ref[i] - Q_ref[j]
            d = np.linalg.norm(v)
            rr = d - 1.0
            tot_ref += rr * rr
            g = (rr / max(d, 1e-9)) * v
            grad[i] += g
            grad[j] -= g
        Q_ref -= 0.08 * grad
        if tot_ref < 1e-28:
            break

    Q, tot = gauss_newton(P, edges, lr=0.08, iters=iters, target=1e-28)
    assert np.allclose(Q, Q_ref, rtol=0, atol=1e-13)
    assert tot == pytest.approx(tot_ref, rel=1e-9)


def test_gauss_newton_empty_edges_is_noop():
    P = unit_rhombus()
    Q, tot = gauss_newton(P, [])
    assert tot == 0.0
    assert np.array_equal(Q, P)
    assert Q is not P  # returns a copy, never aliases the input


def test_gauss_newton_detects_unrealizable_k4():
    """All 6 pairwise distances of 4 planar points cannot be unit: the snap
    must NOT reach the pass threshold (realizability audit has teeth)."""
    P = unit_rhombus()  # 4 points; claim ALL 6 edges including the two non-unit
    E4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    _, tot = gauss_newton(P, E4)
    assert tot > 1e-6  # stuck far from zero (prototype value ~0.17)


# ------------------------------------------------- 4. K_{2,3} detection


def test_k23_violation_detected_from_edge_list():
    """Hand-built K_{2,3}: centers 0,1 each adjacent to 2,3,4 (geometrically
    impossible with exact unit distances; fed directly as an edge list)."""
    edges = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]
    assert k23_violations(5, edges) >= 1
    assert k23_violations(5, edges) == 1  # exactly the pair (0, 1)


def test_k23_clean_on_legit_configs():
    P = unit_rhombus()
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4) if abs(D[i, j] - 1) < TOL]
    assert k23_violations(4, edges) == 0
    # K_{2,2} (two vertices sharing exactly 2 neighbors) is allowed
    assert k23_violations(4, [(0, 2), (0, 3), (1, 2), (1, 3)]) == 0


# ------------------------------------------------------------ helpers


def test_min_separation_basic():
    P = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0]])
    assert min_separation(P) == pytest.approx(1.0)
    assert min_separation(unit_rhombus()) == pytest.approx(1.0)


def test_audit_report_fields_complete():
    report = audit(unit_rhombus())
    for field in (
        "n",
        "n_edges",
        "min_sep",
        "k23_violations",
        "gn_total_residual",
        "gn_edges_exact",
        "gn_max_move",
        "min_sep_after",
        "passed",
    ):
        assert hasattr(report, field)
