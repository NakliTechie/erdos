"""Perpendicular-bisector energy of planar point sets.

Port of code/exp3_bisector.py (bisector_energy / tri_energy), numpy-only.

The bisector of segment (a, b) is the line 2(b-a).x = |b|^2 - |a|^2. For
integer points the triple (2*dx, 2*dy, |b|^2-|a|^2) is integer-exact; we
normalize each triple by gcd and a sign convention (first nonzero of the
(A, B) part positive), then count distinct lines with a structured-array
np.unique. The energy E = sum over distinct lines of multiplicity^2 is
computed with object-dtype (python int) arithmetic to avoid int64 overflow.

Self-contained: stdlib + numpy only.
"""

from dataclasses import dataclass

import numpy as np

TOL = 1e-9        # unit-distance tolerance for float configs
MIN_SEP = 0.2     # hard minimum separation (pitfall HANDOFF §3.1)


@dataclass
class BisectorResult:
    E: int            # sum of multiplicity^2 over distinct bisector lines
    E_nontrivial: int  # same but only lines with multiplicity > 1
    num_lines: int
    max_mult: int
    num_pairs: int


def _count_normalized_int_triples(A: np.ndarray, B: np.ndarray, C: np.ndarray,
                                  num_pairs: int) -> BisectorResult:
    """gcd+sign normalize integer line triples and count multiplicities.

    Exact port of the normalization + counting core shared by exp3's
    bisector_energy and tri_energy.
    """
    # normalize each triple by gcd
    g = np.gcd(np.gcd(np.abs(A), np.abs(B)), np.abs(C))
    g[g == 0] = 1
    A //= g
    B //= g
    C //= g
    # sign fix: first nonzero of (A,B) positive
    s = np.where(A != 0, np.sign(A), np.sign(B)).astype(np.int64)
    s[s == 0] = 1
    A *= s
    B *= s
    C *= s
    # structured numpy unique (fast, exact)
    T = np.stack([A, B, C], axis=1)
    Tv = np.ascontiguousarray(T).view(
        [('a', np.int64), ('b', np.int64), ('c', np.int64)])
    u, counts = np.unique(Tv, return_counts=True)
    E = int((counts.astype(object) ** 2).sum())
    E_nt = int((counts[counts > 1].astype(object) ** 2).sum())
    return BisectorResult(E=E, E_nontrivial=E_nt, num_lines=len(u),
                          max_mult=int(counts.max()), num_pairs=num_pairs)


def bisector_energy_int(P) -> BisectorResult:
    """Bisector energy of integer points P, shape (n, 2).

    Line triple (2*dx, 2*dy, |b|^2 - |a|^2), gcd+sign normalized,
    counted exactly. Port of exp3 bisector_energy.
    """
    P = np.asarray(P, dtype=np.int64)
    n = len(P)
    i, j = np.triu_indices(n, k=1)
    A = 2 * (P[j, 0] - P[i, 0])
    B = 2 * (P[j, 1] - P[i, 1])
    C = (P[j, 0] ** 2 + P[j, 1] ** 2) - (P[i, 0] ** 2 + P[i, 1] ** 2)
    return _count_normalized_int_triples(A, B, C, len(i))


def bisector_energy_tri(P) -> BisectorResult:
    """Bisector energy for triangular-lattice coords: integer P, shape (n, 2),
    where row (a, b) means the real point (a, b*sqrt(3)).

    |b|^2 - |a|^2 = (bx^2 - ax^2) + 3*(by^2 - ay^2) is integer; the bisector
    line is 2*dx*X + 6*dy*Yt = C with Y = Yt*sqrt(3) — all-integer triples,
    so the count is exact. Port of exp3 tri_energy.
    """
    pts = np.asarray(P, dtype=np.int64)
    n = len(pts)
    i, j = np.triu_indices(n, k=1)
    A = 2 * (pts[j, 0] - pts[i, 0])
    B = 6 * (pts[j, 1] - pts[i, 1])
    C = ((pts[j, 0] ** 2 + 3 * pts[j, 1] ** 2)
         - (pts[i, 0] ** 2 + 3 * pts[i, 1] ** 2))
    return _count_normalized_int_triples(A, B, C, len(i))


def bisector_energy_float(P, decimals: int = 6) -> BisectorResult:
    """BUCKETED bisector energy for arbitrary float configs.

    Each pair's bisector is represented as (dx, dy, (|b|^2-|a|^2)/2),
    normalized so (dx, dy) has unit norm with the sign convention that the
    first nonzero of (dx, dy) is positive, then all three components are
    rounded to `decimals` decimal places and counted.

    WARNING: this is a bucketed approximation — equal lines always collide,
    but distinct lines closer than the rounding grid collide too. Use ONLY
    for relative comparisons between float configs, never as an exact count.
    """
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    i, j = np.triu_indices(n, k=1)
    dx = P[j, 0] - P[i, 0]
    dy = P[j, 1] - P[i, 1]
    c = ((P[j, 0] ** 2 + P[j, 1] ** 2)
         - (P[i, 0] ** 2 + P[i, 1] ** 2)) / 2.0
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0  # degenerate coincident pair; triple stays (0,0,0)
    dx = dx / norm
    dy = dy / norm
    c = c / norm
    # round first so the sign decision is made on the bucketed values
    dx = np.round(dx, decimals)
    dy = np.round(dy, decimals)
    c = np.round(c, decimals)
    # sign fix: first nonzero of (dx, dy) positive
    s = np.where(dx != 0, np.sign(dx), np.sign(dy))
    s[s == 0] = 1.0
    dx = dx * s + 0.0  # "+ 0.0" canonicalizes -0.0 to +0.0
    dy = dy * s + 0.0
    c = c * s + 0.0
    T = np.stack([dx, dy, c], axis=1)
    Tv = np.ascontiguousarray(T).view(
        [('a', np.float64), ('b', np.float64), ('c', np.float64)])
    u, counts = np.unique(Tv, return_counts=True)
    E = int((counts.astype(object) ** 2).sum())
    E_nt = int((counts[counts > 1].astype(object) ** 2).sum())
    return BisectorResult(E=E, E_nontrivial=E_nt, num_lines=len(u),
                          max_mult=int(counts.max()), num_pairs=len(i))
