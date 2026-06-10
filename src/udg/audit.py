"""udg.audit — the three-audit gate for claimed unit-distance configurations.

THE load-bearing module of the package (HANDOFF §3). Never report an edge
count that has not passed all three audits:

1. **Min-separation** (`min_separation`): every pair of points must be at
   least ``MIN_SEP = 0.2`` apart. Guards the canonical *tolerance exploit*:
   near-coincident clusters spread tangentially along a unit circle have
   second-order distance error (~delta^2/2), so a delta = 1e-6 separation
   passes tol = 1e-9 per edge while faking absurd counts (the "400 edges at
   n=40" disaster).
2. **K_{2,3}-freeness** (`k23_violations`): two unit circles meet in at most
   2 points, so no two vertices of a true UDG share >= 3 common
   unit-neighbors.
3. **Exact realizability** (`gauss_newton`): damped Gauss-Newton on ALL
   claimed edges, minimizing sum (|p_i - p_j| - 1)^2. Accept only if the
   total residual drops below 1e-24 and every claimed edge is within 1e-12
   of unit after projection.

Ported from ``code/exp6_clean.py`` (audit: K23 + min-sep) and
``code/exp7_audit.py`` (damped Gauss-Newton), with the Gauss-Newton inner
loop vectorized over edges (index arrays + ``np.add.at``; identical math).

Self-contained: stdlib + numpy only, no intra-package imports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOL = 1e-9        # unit-distance tolerance for float configs
MIN_SEP = 0.2     # hard minimum separation (pitfall HANDOFF §3.1)

# Realizability acceptance thresholds (HANDOFF §3, exp7).
GN_RESIDUAL_PASS = 1e-24   # total residual after Gauss-Newton must be below this
GN_EDGE_EXACT = 1e-12      # per-edge |d - 1| after projection must be below this


@dataclass
class AuditReport:
    n: int
    n_edges: int              # edges claimed at tol BEFORE projection
    min_sep: float
    k23_violations: int
    gn_total_residual: float  # sum of (|p_i-p_j|-1)^2 after Gauss-Newton on claimed edges
    gn_edges_exact: int       # edges within 1e-12 of 1 after projection
    gn_max_move: float        # max point displacement under projection
    min_sep_after: float
    passed: bool              # ALL of: min_sep >= MIN_SEP, k23_violations == 0,
                              # gn_total_residual < 1e-24, gn_edges_exact == n_edges


def _dist_matrix(P: np.ndarray) -> np.ndarray:
    """(n, n) Euclidean distance matrix, float64."""
    P = np.asarray(P, dtype=np.float64)
    return np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))


def _unit_edges(P: np.ndarray, tol: float = TOL) -> list[tuple[int, int]]:
    """All i < j pairs with | |p_i - p_j| - 1 | < tol."""
    D = _dist_matrix(P)
    n = len(P)
    iu, ju = np.triu_indices(n, 1)
    mask = np.abs(D[iu, ju] - 1.0) < tol
    return [(int(i), int(j)) for i, j in zip(iu[mask], ju[mask])]


def min_separation(P) -> float:
    """Smallest pairwise distance in the configuration."""
    D = _dist_matrix(P)
    np.fill_diagonal(D, np.inf)
    return float(D.min())


def k23_violations(n: int, edges: list[tuple[int, int]]) -> int:
    """Number of vertex pairs sharing >= 3 common unit-neighbors.

    True UDGs are K_{2,3}-free (two unit circles meet in <= 2 points), so any
    violation proves the claimed edge set is not unit-realizable with distinct
    points. Port of exp6 ``audit``.
    """
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    bad = 0
    for i in range(n):
        for j in range(i + 1, n):
            if len(adj[i] & adj[j]) >= 3:
                bad += 1
    return bad


def gauss_newton(
    P,
    edges: list[tuple[int, int]],
    lr: float = 0.08,
    iters: int = 8000,
    target: float = 1e-28,
) -> tuple[np.ndarray, float]:
    """Damped Gauss-Newton snap of ALL claimed edges to exact unit length.

    Minimizes ``sum over (i,j) in edges of (|p_i - p_j| - 1)^2``. Exact port
    of the exp7 math — per edge ``g = (r/d) * v`` with ``d`` floored at 1e-9,
    gradient accumulated +g at i and -g at j, step ``Q -= lr * grad``, early
    stop once the residual (measured BEFORE the step, as in exp7) drops below
    ``target`` — but vectorized: edges become index arrays and the gradient
    scatter uses ``np.add.at`` (no Python loop over edges inside the
    iteration loop).

    Returns ``(Q, total_residual)`` where ``total_residual`` is the last
    residual measured (exp7 semantics: the value that triggered early stop,
    or the final iteration's pre-step residual).
    """
    Q = np.asarray(P, dtype=np.float64).copy()
    if not edges:
        return Q, 0.0
    E = np.asarray(edges, dtype=np.intp)
    ei, ej = E[:, 0], E[:, 1]
    tot = 0.0
    for _ in range(iters):
        v = Q[ei] - Q[ej]                       # (m, 2)
        d = np.sqrt((v * v).sum(1))             # (m,)
        rr = d - 1.0
        tot = float((rr * rr).sum())
        g = (rr / np.maximum(d, 1e-9))[:, None] * v
        grad = np.zeros_like(Q)
        np.add.at(grad, ei, g)
        np.add.at(grad, ej, -g)
        Q -= lr * grad
        if tot < target:
            break
    return Q, tot


def audit(P, tol: float = TOL) -> AuditReport:
    """Run all three audits (HANDOFF §3) on a claimed configuration.

    ``passed`` is True only if ALL hold:
    min_sep >= MIN_SEP, k23_violations == 0, gn_total_residual < 1e-24,
    gn_edges_exact == n_edges.
    """
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    edges = _unit_edges(P, tol)
    ms = min_separation(P)
    bad = k23_violations(n, edges)

    Q, tot = gauss_newton(P, edges)
    D2 = _dist_matrix(Q)
    exact = sum(1 for (i, j) in edges if abs(D2[i, j] - 1.0) < GN_EDGE_EXACT)
    move = float(np.sqrt(((Q - P) ** 2).sum(1)).max()) if n else 0.0
    ms_after = min_separation(Q)

    passed = (
        ms >= MIN_SEP
        and bad == 0
        and tot < GN_RESIDUAL_PASS
        and exact == len(edges)
    )
    return AuditReport(
        n=n,
        n_edges=len(edges),
        min_sep=ms,
        k23_violations=bad,
        gn_total_residual=tot,
        gn_edges_exact=exact,
        gn_max_move=move,
        min_sep_after=ms_after,
        passed=passed,
    )
