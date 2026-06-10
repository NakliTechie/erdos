"""Distance counting for the unit-distance research program.

Ported from code/exp1_grid.py (chunked integer-exact multiplicity counting)
and code/exp2_structure.py (popular distances, r2, triangular-form block).

Conventions (HANDOFF §3):
- Squared distances + exact int64 arithmetic for lattice points (`popular`).
- tol = 1e-9 with float64 otherwise (`unit_edges` / `unit_count`).

Self-contained: stdlib + numpy only, no intra-package imports.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

TOL = 1e-9        # unit-distance tolerance for float configs
MIN_SEP = 0.2     # hard minimum separation (pitfall HANDOFF §3.1)

_CHUNK = 1200     # row-chunk size for pairwise counting (exp2)


def dist_matrix(P: np.ndarray) -> np.ndarray:
    """(n, n) matrix of Euclidean distances, float64."""
    P = np.asarray(P, dtype=np.float64)
    diff = P[:, None, :] - P[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def unit_edges(P: np.ndarray, tol: float = TOL) -> list[tuple[int, int]]:
    """All i<j pairs with | |p_i - p_j| - 1 | < tol."""
    D = dist_matrix(P)
    n = D.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    mask = np.abs(D[iu, ju] - 1.0) < tol
    return [(int(i), int(j)) for i, j in zip(iu[mask], ju[mask])]


def unit_count(P: np.ndarray, tol: float = TOL) -> int:
    """Number of unit-distance pairs at tolerance `tol`."""
    return len(unit_edges(P, tol))


def popular(
    points,
    form: tuple[int, int, int] = (1, 0, 1),
    topk: int = 5,
) -> tuple[int, list[tuple[int, int]]]:
    """Integer-exact pairwise quadratic-form multiplicity counts, chunked.

    form (a, b, c): value = a*dx^2 + b*dx*dy + c*dy^2 on int64 lattice points.
      (1, 0, 1) -> squared Euclidean distance (square grid, exp1/exp2);
      (1, 1, 1) -> Eisenstein/triangular norm x^2 + xy + y^2 (exp2 block 2).

    Returns (n, [(value, count), ...]) with the topk values sorted by count
    descending (exp2 ``popular`` / Counter.most_common).
    """
    P = np.asarray(points, dtype=np.int64)
    n = len(P)
    a, b, c = form
    cnt: Counter[int] = Counter()
    for i in range(0, n, _CHUNK):
        A = P[i:i + _CHUNK]
        dx = A[:, None, 0] - P[None, :, 0]
        dy = A[:, None, 1] - P[None, :, 1]
        D = a * dx * dx + b * dx * dy + c * dy * dy
        gi = np.arange(i, i + len(A))[:, None]
        gj = np.arange(n)[None, :]
        vals = D[gi < gj]
        u, uc = np.unique(vals, return_counts=True)
        for uu, cc in zip(u.tolist(), uc.tolist()):
            cnt[uu] += cc
    return n, cnt.most_common(topk)


def r2(k: int) -> int:
    """Number of representations of k as x^2 + y^2 (signed, ordered).

    exp2's "simpler exact method": walk x over [-isqrt(k), isqrt(k)] and
    count exact square complements (y == 0 contributes 1, else 2 for ±y).
    """
    if k < 0:
        return 0
    c = 0
    r = math.isqrt(k)
    for x in range(-r, r + 1):
        y2 = k - x * x
        y = math.isqrt(y2)
        if y * y == y2:
            c += 1 if y == 0 else 2
    return c
