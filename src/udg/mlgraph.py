"""Exact discrete Moser-lattice toolkit: configs, edges, sums, local moves.

The Moser lattice ML = Z<1, w1, w3, w1*w3> (w1 = exp(i*pi/3), w3 =
exp(i*arccos(5/6))) is a rank-4 free Z-module densely embedded in C. A point
is an integer 4-tuple (a, b, c, d) meaning a + b*w1 + c*w3 + d*w1*w3. In
these coordinates everything is EXACT: a pair is a unit edge iff the squared
modulus of its coordinate difference, evaluated in Q(sqrt3, sqrt11), equals
1 exactly. No tolerances anywhere in this module.

Exact unit test, two implementations that must agree:

1. Reference path (`exact_dist2`, lifted from scripts/ml_coords.py): field
   arithmetic in Q(sqrt3, sqrt11) over the basis (1, sqrt3, sqrt11, sqrt33)
   with Fraction coordinates; unit iff all four components == (1, 0, 0, 0).

2. Scaled-integer fast path (`is_unit`, `_unit_mask_block`): for a
   coordinate difference (da, db, dc, dd) the exact planar embedding is

       12 * Re = A + D*sqrt33,   12 * Im = B*sqrt3 + C*sqrt11

   with the INTEGER invariants

       A = 12*da + 6*db + 10*dc + 5*dd
       B =          6*db         + 5*dd
       C =                 2*dc  +   dd
       D =                          -dd

   so   144 * |diff|^2 = (A^2 + 3*B^2 + 11*C^2 + 33*D^2)
                        + 2*(A*D + B*C) * sqrt33.

   (The sqrt3 / sqrt11 components of |diff|^2 vanish structurally: Re lives
   in Q + Q*sqrt33, Im in Q*sqrt3 + Q*sqrt11.) Since sqrt33 is irrational,

       |diff|^2 == 1  <=>  A^2 + 3*B^2 + 11*C^2 + 33*D^2 == 144
                           and  A*D + B*C == 0,

   pure integer arithmetic, vectorizable with numpy int64. With max
   coordinate magnitude M the worst term is bounded by 1584 * (2*M)^2, so
   int64 is exact for M <= ~3.8e7; beyond `_INT64_SAFE` the same code runs
   on dtype=object (Python ints), still exact.

Contents:
- `MLConfig`: immutable deduplicated point set; constructors from a list,
  from a data/mlcoords certificate JSON, from generators (subset/multiple
  sums -- the paper's "Minkowski sum of k edges" records).
- `exact_edges` / `exact_edge_count` / `degrees`: exact unit-pair queries.
- `to_float` / `save_csv`: float64 positions and the project CSV format
  (via udg.configio) for handoff to the float audit pipeline.
- `minkowski(A, B)`: exact coordinate-tuple Minkowski sum.
- Small library: unit_edge, unit_triangle, unit_rhombus, wheel6 (two
  sublattice families -- the disjoint 6-wheel (+) 6-wheel record at n=49
  needs one wheel per family), moser_spindle, tri_patch(k).
- Local moves: candidate_positions, greedy_improve, add_best_point,
  drop_worst_point -- exact-edge-delta hill climbing on the lattice.

Self-contained except for udg.moser.UNIT_COEFFS (the 18 ML unit vectors)
and udg.configio.save_csv.
"""

from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from udg import configio
from udg.moser import UNIT_COEFFS

__all__ = [
    "QF",
    "MLConfig",
    "exact_xy",
    "exact_dist2",
    "dist2_components",
    "is_unit",
    "is_unit_exact",
    "exact_edges",
    "exact_edge_count",
    "degrees",
    "to_float",
    "float_xy",
    "unit_mask",
    "canon",
    "save_csv",
    "minkowski",
    "unit_edge",
    "unit_triangle",
    "unit_rhombus",
    "wheel6",
    "moser_spindle",
    "tri_patch",
    "candidate_positions",
    "greedy_improve",
    "add_best_point",
    "drop_worst_point",
]

Coeffs = tuple[int, int, int, int]

_SQRT3 = math.sqrt(3.0)
_SQRT11 = math.sqrt(11.0)
_SQRT33 = math.sqrt(33.0)

# int64 exactness bound for the scaled invariants. For coordinate magnitude
# M the difference magnitude is <= 2*M, |A| <= 33*(2*M), and the largest
# intermediate is A^2 + 3B^2 + 11C^2 + 33D^2 <= 1584*(2*M)^2; demanding
# 1584*4*M^2 < 2^63 gives M < ~3.8e7. 1e7 leaves a 14x margin.
_INT64_SAFE = 10_000_000

# rows-x-cols budget per vectorized block (keeps peak memory ~100 MB)
_BLOCK_CELLS = 1 << 22


# ---------------------------------------------------------------------------
# Exact arithmetic in Q(sqrt3, sqrt11), basis (1, sqrt3, sqrt11, sqrt33)
# (reference path, adapted from scripts/ml_coords.py)
# ---------------------------------------------------------------------------

class QF:
    """Element a + b*sqrt3 + c*sqrt11 + d*sqrt33 of Q(sqrt3, sqrt11).

    Coordinates are fractions.Fraction; all operations are exact. The
    multiplication table of the basis (1, s3, s11, s33):

        s3*s3 = 3        s11*s11 = 11      s33*s33 = 33
        s3*s11 = s33     s3*s33 = 3*s11    s11*s33 = 11*s3
    """

    __slots__ = ("a", "b", "c", "d")

    def __init__(self, a=0, b=0, c=0, d=0):
        self.a = Fraction(a)
        self.b = Fraction(b)
        self.c = Fraction(c)
        self.d = Fraction(d)

    def __add__(self, o: "QF") -> "QF":
        return QF(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d)

    def __sub__(self, o: "QF") -> "QF":
        return QF(self.a - o.a, self.b - o.b, self.c - o.c, self.d - o.d)

    def __mul__(self, o: "QF") -> "QF":
        a1, b1, c1, d1 = self.a, self.b, self.c, self.d
        a2, b2, c2, d2 = o.a, o.b, o.c, o.d
        return QF(
            a1 * a2 + 3 * b1 * b2 + 11 * c1 * c2 + 33 * d1 * d2,
            a1 * b2 + b1 * a2 + 11 * (c1 * d2 + d1 * c2),
            a1 * c2 + c1 * a2 + 3 * (b1 * d2 + d1 * b2),
            a1 * d2 + d1 * a2 + b1 * c2 + c1 * b2,
        )

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, QF):
            return NotImplemented
        return self.a == o.a and self.b == o.b and self.c == o.c and self.d == o.d

    def __hash__(self) -> int:
        return hash((self.a, self.b, self.c, self.d))

    def is_zero(self) -> bool:
        return self.a == 0 and self.b == 0 and self.c == 0 and self.d == 0

    def to_float(self) -> float:
        return (
            float(self.a)
            + float(self.b) * _SQRT3
            + float(self.c) * _SQRT11
            + float(self.d) * _SQRT33
        )

    def __repr__(self) -> str:
        return f"QF({self.a}, {self.b}, {self.c}, {self.d})"


QF_ZERO = QF()
QF_ONE = QF(1)

# Exact real/imag parts of the ML basis (1, w1, w3, w1*w3) in Q(sqrt3, sqrt11):
#   w1   = 1/2 + (sqrt3/2) i
#   w3   = 5/6 + (sqrt11/6) i
#   w1w3 = 5/12 - sqrt33/12 + (sqrt11/12 + 5*sqrt3/12) i
_RE_BASIS = (
    QF(1),
    QF(Fraction(1, 2)),
    QF(Fraction(5, 6)),
    QF(Fraction(5, 12), 0, 0, Fraction(-1, 12)),
)
_IM_BASIS = (
    QF(0),
    QF(0, Fraction(1, 2)),
    QF(0, 0, Fraction(1, 6)),
    QF(0, Fraction(5, 12), Fraction(1, 12), 0),
)


def exact_xy(coeffs: Sequence[int]) -> tuple[QF, QF]:
    """Exact (Re, Im) in Q(sqrt3, sqrt11) of an integer ML coordinate 4-tuple."""
    re = QF_ZERO
    im = QF_ZERO
    for m, rb, ib in zip(coeffs, _RE_BASIS, _IM_BASIS):
        if m:
            mq = QF(m)
            re = re + mq * rb
            im = im + mq * ib
    return re, im


def exact_dist2(ca: Sequence[int], cb: Sequence[int]) -> QF:
    """Exact |p(ca) - p(cb)|^2 for two integer ML coordinate 4-tuples."""
    ra, ia = exact_xy(ca)
    rb, ib = exact_xy(cb)
    dre = ra - rb
    dim = ia - ib
    d2 = dre * dre + dim * dim
    # structural invariant: Re, Im of ML points live in Q+Q*s33 and
    # Q*s3+Q*s11 respectively, so |.|^2 has no s3/s11 component
    assert d2.b == 0 and d2.c == 0, f"impossible dist^2 {d2!r}"
    return d2


def dist2_components(
    ca: Sequence[int], cb: Sequence[int]
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """The 4 rational components of |p(ca)-p(cb)|^2 over (1, s3, s11, s33)."""
    d2 = exact_dist2(ca, cb)
    return (d2.a, d2.b, d2.c, d2.d)


def is_unit_exact(ca: Sequence[int], cb: Sequence[int]) -> bool:
    """Reference unit test: |diff|^2 components == (1, 0, 0, 0) exactly."""
    return dist2_components(ca, cb) == (Fraction(1), Fraction(0), Fraction(0), Fraction(0))


# ---------------------------------------------------------------------------
# Scaled-integer fast path
# ---------------------------------------------------------------------------

def _invariants(diff):
    """Integer invariants (A, B, C, D) of coordinate differences.

    Works on python int 4-tuples and on numpy arrays of shape (..., 4)
    (int64 or object dtype) alike. 12*Re(diff) = A + D*s33 and
    12*Im(diff) = B*s3 + C*s11.
    """
    if isinstance(diff, np.ndarray):
        da, db, dc, dd = diff[..., 0], diff[..., 1], diff[..., 2], diff[..., 3]
    else:
        da, db, dc, dd = diff
    A = 12 * da + 6 * db + 10 * dc + 5 * dd
    B = 6 * db + 5 * dd
    C = 2 * dc + dd
    D = -dd
    return A, B, C, D


def is_unit(ca: Sequence[int], cb: Sequence[int]) -> bool:
    """Exact unit test on two integer 4-tuples (python-int arithmetic)."""
    A, B, C, D = _invariants(tuple(int(a) - int(b) for a, b in zip(ca, cb)))
    return A * A + 3 * B * B + 11 * C * C + 33 * D * D == 144 and A * D + B * C == 0


def _unit_mask_block(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Boolean (len(X), len(Y)) mask: which cross pairs are exact unit edges."""
    diff = X[:, None, :] - Y[None, :, :]
    A, B, C, D = _invariants(diff)
    return np.asarray(
        (A * A + 3 * B * B + 11 * C * C + 33 * D * D == 144) & (A * D + B * C == 0),
        dtype=bool,
    )


def _unit_mask(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Chunked exact unit mask between two coordinate arrays (m,4) x (n,4)."""
    m, n = len(X), len(Y)
    if m * n <= _BLOCK_CELLS:
        return _unit_mask_block(X, Y)
    out = np.zeros((m, n), dtype=bool)
    rows = max(1, _BLOCK_CELLS // max(n, 1))
    for i0 in range(0, m, rows):
        out[i0 : i0 + rows] = _unit_mask_block(X[i0 : i0 + rows], Y)
    return out


def unit_mask(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Public exact unit mask on raw (m,4) x (n,4) integer coordinate arrays.

    Boolean (len(X), len(Y)): which cross pairs are exact unit edges. The
    chunked scaled-integer fast path -- the shared primitive of the promoted
    chase engines (udg.anneal, udg.subsetsearch, scripts/chase40lib.py).
    """
    return _unit_mask(np.asarray(X), np.asarray(Y))


def _as_coeff_array(points: Sequence[Coeffs]) -> np.ndarray:
    """(n, 4) coordinate array; int64 when exact-safe, else object (big ints)."""
    if not points:
        return np.zeros((0, 4), dtype=np.int64)
    max_abs = max(abs(x) for p in points for x in p)
    dtype = np.int64 if max_abs <= _INT64_SAFE else object
    arr = np.empty((len(points), 4), dtype=dtype)
    for i, p in enumerate(points):
        arr[i] = p
    return arr


# ---------------------------------------------------------------------------
# MLConfig
# ---------------------------------------------------------------------------

def _canon_point(p: Sequence[int]) -> Coeffs:
    if len(p) != 4:
        raise ValueError(f"ML point must have 4 integer coordinates, got {p!r}")
    out = []
    for x in p:
        xi = int(x)
        if xi != x:
            raise ValueError(f"non-integer ML coordinate {x!r} in {p!r}")
        out.append(xi)
    return (out[0], out[1], out[2], out[3])


class MLConfig:
    """An immutable, exactly-deduplicated set of Moser-lattice points.

    Points are integer 4-tuples over the basis (1, w1, w3, w1*w3); distinct
    tuples are distinct points of C (the basis is Q-linearly independent),
    so tuple dedup == exact point distinctness. Input order is preserved
    (first occurrence wins); equality and hashing are set-based.
    """

    __slots__ = ("points", "_set", "_arr")

    def __init__(self, points: Iterable[Sequence[int]] = ()):
        seen: dict[Coeffs, None] = {}
        for p in points:
            seen.setdefault(_canon_point(p), None)
        self.points: tuple[Coeffs, ...] = tuple(seen)
        self._set = frozenset(self.points)
        self._arr: np.ndarray | None = None

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_points(cls, points: Iterable[Sequence[int]]) -> "MLConfig":
        return cls(points)

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> "MLConfig":
        """Load from a data/mlcoords certificate JSON (key "coords"), or a
        bare JSON list of 4-tuples."""
        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        coords = data["coords"] if isinstance(data, dict) else data
        return cls(coords)

    @classmethod
    def from_generators(
        cls,
        gens: Sequence[Sequence[int]],
        coeffs: Sequence[int] = (0, 1),
    ) -> "MLConfig":
        """All sums sum_i c_i * g_i with each c_i drawn from ``coeffs``.

        Default coeffs (0, 1) gives the 2^k subset sums -- the paper's
        "Minkowski sum of k edges" constructions (n=64 record = 6 edges).
        """
        gen_tuples = [_canon_point(g) for g in gens]
        pts: list[Coeffs] = [(0, 0, 0, 0)]
        for g in gen_tuples:
            pts = [
                (p[0] + c * g[0], p[1] + c * g[1], p[2] + c * g[2], p[3] + c * g[3])
                for p in pts
                for c in coeffs
            ]
        return cls(pts)

    # -- container protocol --------------------------------------------------

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[Coeffs]:
        return iter(self.points)

    def __contains__(self, p: object) -> bool:
        try:
            return _canon_point(p) in self._set  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return False

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, MLConfig):
            return NotImplemented
        return self._set == o._set

    def __hash__(self) -> int:
        return hash(self._set)

    def __repr__(self) -> str:
        return f"MLConfig(n={len(self.points)})"

    # -- views ----------------------------------------------------------------

    def as_array(self) -> np.ndarray:
        """(n, 4) coordinate array (int64 when safe, object dtype otherwise)."""
        if self._arr is None:
            self._arr = _as_coeff_array(self.points)
        return self._arr

    # -- derived configs -------------------------------------------------------

    def with_point(self, p: Sequence[int]) -> "MLConfig":
        return MLConfig(self.points + (_canon_point(p),))

    def without_point(self, p: Sequence[int]) -> "MLConfig":
        q = _canon_point(p)
        return MLConfig(pt for pt in self.points if pt != q)

    def without_index(self, i: int) -> "MLConfig":
        return MLConfig(pt for k, pt in enumerate(self.points) if k != i)

    def translate(self, t: Sequence[int]) -> "MLConfig":
        tc = _canon_point(t)
        return MLConfig(
            (p[0] + tc[0], p[1] + tc[1], p[2] + tc[2], p[3] + tc[3])
            for p in self.points
        )

    # -- conveniences (delegate to module functions) ----------------------------

    def exact_edges(self) -> list[tuple[int, int]]:
        return exact_edges(self)

    def exact_edge_count(self) -> int:
        return exact_edge_count(self)

    def degrees(self) -> np.ndarray:
        return degrees(self)

    def to_float(self) -> np.ndarray:
        return to_float(self)

    def save_csv(self, path: str | os.PathLike) -> None:
        save_csv(self, path)


# ---------------------------------------------------------------------------
# Exact edge queries
# ---------------------------------------------------------------------------

def exact_edges(cfg: MLConfig) -> list[tuple[int, int]]:
    """All i<j index pairs of cfg.points with |diff|^2 == 1 EXACTLY."""
    pts = cfg.as_array()
    n = len(pts)
    if n < 2:
        return []
    edges: list[tuple[int, int]] = []
    rows = max(1, _BLOCK_CELLS // n)
    for i0 in range(0, n, rows):
        M = _unit_mask_block(pts[i0 : i0 + rows], pts)
        ii, jj = np.nonzero(M)
        ii = ii + i0
        keep = ii < jj
        edges.extend(zip(ii[keep].tolist(), jj[keep].tolist()))
    return edges


def exact_edge_count(cfg: MLConfig) -> int:
    """Number of exact unit-distance pairs in cfg."""
    return len(exact_edges(cfg))


def degrees(cfg: MLConfig) -> np.ndarray:
    """Exact unit-graph degree of each point, aligned with cfg.points."""
    pts = cfg.as_array()
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    deg = np.zeros(n, dtype=np.int64)
    rows = max(1, _BLOCK_CELLS // n)
    for i0 in range(0, n, rows):
        M = _unit_mask_block(pts[i0 : i0 + rows], pts)
        deg[i0 : i0 + rows] = M.sum(axis=1)  # self-pair has |diff|^2 == 0, never unit
    return deg


# ---------------------------------------------------------------------------
# Float export
# ---------------------------------------------------------------------------

def float_xy(arr: np.ndarray) -> np.ndarray:
    """(m, 2) float64 positions of raw integer coordinate rows (m, 4).

    x = (A + D*s33)/12, y = (B*s3 + C*s11)/12 with A, B, C, D the exact
    integer invariants of each point, so each coordinate is a single
    rounding of an exact value (error ~1e-16 * |p|).
    """
    A, B, C, D = _invariants(np.asarray(arr))
    A = A.astype(float)
    B = B.astype(float)
    C = C.astype(float)
    D = D.astype(float)
    x = (A + D * _SQRT33) / 12.0
    y = (B * _SQRT3 + C * _SQRT11) / 12.0
    return np.column_stack([x, y])


def to_float(cfg: MLConfig) -> np.ndarray:
    """(n, 2) float64 positions of cfg.points (see float_xy)."""
    return float_xy(cfg.as_array())


def save_csv(cfg: MLConfig, path: str | os.PathLike) -> None:
    """Write float64 positions in the project CSV format (udg.configio)."""
    configio.save_csv(to_float(cfg), path)


# ---------------------------------------------------------------------------
# Minkowski sums
# ---------------------------------------------------------------------------

def minkowski(a: MLConfig, b: MLConfig) -> MLConfig:
    """Exact Minkowski sum {p + q : p in a, q in b}, deduplicated exactly."""
    return MLConfig(
        (p[0] + q[0], p[1] + q[1], p[2] + q[2], p[3] + q[3])
        for p in a.points
        for q in b.points
    )


# ---------------------------------------------------------------------------
# Canonical form under the 12-element ML point group x translation
# (promoted from scripts/chase40lib.py -- THE one canonical implementation)
# ---------------------------------------------------------------------------
#
# The 12 rigid motions of ML = 6 rotations (multiplication by w1^k;
# w1^3 = -1) x optional reflection z -> w3 * conj(z) (verified module
# automorphism: w3*conj(1)=w3, w3*conj(w1)=w3-w1w3, w3*conj(w3)=1,
# w3*conj(w1w3)=1-w1), plus translations (subtract the lexicographically
# smallest row). canon-equal <=> identical point set up to ML symmetry.


def _rot_w1_arr(P: np.ndarray) -> np.ndarray:
    """Multiplication by w1: (a,b,c,d) -> (-b, a+b, -d, c+d)."""
    a, b, c, d = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    return np.stack([-b, a + b, -d, c + d], axis=1)


def _refl_w3_arr(P: np.ndarray) -> np.ndarray:
    """z -> w3*conj(z): (a,b,c,d) -> (c+d, -d, a+b, -b)."""
    a, b, c, d = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    return np.stack([c + d, -d, a + b, -b], axis=1)


def canon(P: np.ndarray) -> bytes:
    """Canonical bytes key of an (n,4) integer point set under the 12 ML
    motions + translation. Keys are only comparable to keys produced by
    this same function (within and across processes of one version)."""
    best = None
    Q = np.asarray(P, dtype=np.int64)
    for _ in range(6):
        for X in (Q, _refl_w3_arr(Q)):
            rows = sorted(map(tuple, X.tolist()))
            base = rows[0]
            key = np.array(
                [[r[k] - base[k] for k in range(4)] for r in rows], dtype=np.int64
            ).tobytes()
            if best is None or key < best:
                best = key
        Q = _rot_w1_arr(Q)
    return best


# ---------------------------------------------------------------------------
# Small library of named UDGs
# ---------------------------------------------------------------------------

# the two unit hexagonal rings of ML: powers of w1 (w1^2 = w1 - 1, w1^3 = -1)
# and the same ring rotated by w3 (w3*w1^k -- all 6 are ML unit vectors).
_W1_RING: tuple[Coeffs, ...] = (
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (-1, 1, 0, 0),
    (-1, 0, 0, 0),
    (0, -1, 0, 0),
    (1, -1, 0, 0),
)
_W3_RING: tuple[Coeffs, ...] = (
    (0, 0, 1, 0),
    (0, 0, 0, 1),
    (0, 0, -1, 1),
    (0, 0, -1, 0),
    (0, 0, 0, -1),
    (0, 0, 1, -1),
)


def unit_edge() -> MLConfig:
    """2 vertices, 1 edge: {0, 1}."""
    return MLConfig([(0, 0, 0, 0), (1, 0, 0, 0)])


def unit_triangle() -> MLConfig:
    """3 vertices, 3 edges: {0, 1, w1} (|w1 - 1| = 1)."""
    return MLConfig([(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)])


def unit_rhombus() -> MLConfig:
    """4 vertices, 5 edges: {0, 1, w1, 1+w1} -- two triangles glued; the
    short diagonal 1 -- w1 is also unit."""
    return MLConfig([(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0)])


def wheel6(family: str = "w1") -> MLConfig:
    """Hexagonal 6-wheel: center + 6 unit ring, 7 vertices, 12 exact edges.

    family="w1": ring = w1^k (the Eisenstein sublattice Z[w1]).
    family="w3": ring = w3 * w1^k (the same hexagon rotated by
    arccos(5/6); all six are ML unit vectors).

    The Engel et al. Table 2 record at n=49 (180 edges) is the DISJOINT
    Minkowski sum of two 6-wheels -- which on ML requires one wheel from
    each family: minkowski(wheel6("w1"), wheel6("w3")) has 49 vertices and
    180 exact edges, while a same-family sum collapses to the 19-point
    hexagon H(2) (42 edges) by lattice collisions (w1^k + w1^(k+2) = w1^(k+1)).
    """
    if family == "w1":
        ring = _W1_RING
    elif family == "w3":
        ring = _W3_RING
    else:
        raise ValueError(f"unknown wheel family {family!r} (use 'w1' or 'w3')")
    return MLConfig(((0, 0, 0, 0),) + ring)


def moser_spindle() -> MLConfig:
    """The Moser spindle: 7 vertices, 11 exact edges.

    Two 60-degree rhombi sharing vertex 0, the second = the first rotated
    by w3. The long diagonals have length sqrt(3) and the tip-tip distance
    is |1 + w1| * |1 - w3| = sqrt(3) * (1/sqrt(3)) = 1 -- this is exactly
    why w3 = exp(i*arccos(5/6)) is a generator of ML.
    """
    return MLConfig(
        [
            (0, 0, 0, 0),
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (1, 1, 0, 0),  # tip of rhombus 1
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (0, 0, 1, 1),  # tip of rhombus 2 = w3 * (1 + w1)
        ]
    )


def tri_patch(k: int) -> MLConfig:
    """Triangular patch T(k) of the Z[w1] sublattice: {a + b*w1 : a, b >= 0,
    a + b <= k}. (k+1)(k+2)/2 vertices, 3*k*(k+1)/2 exact edges."""
    if k < 0:
        raise ValueError("k must be >= 0")
    return MLConfig(
        (a, b, 0, 0) for a in range(k + 1) for b in range(k + 1 - a)
    )


# ---------------------------------------------------------------------------
# Local moves: exact hill climbing on the lattice
# ---------------------------------------------------------------------------

_UC_ARR = np.array(UNIT_COEFFS, dtype=np.int64)  # (18, 4)


def candidate_positions(cfg: MLConfig) -> list[Coeffs]:
    """{p + u : p in cfg, u in the 18 ML unit vectors} minus existing points.

    Every position at exact unit distance from at least one point of cfg --
    the complete relocation/addition neighborhood. Sorted (deterministic).
    """
    pts = cfg.as_array()
    if len(pts) == 0:
        return []
    cand = (pts[:, None, :] + _UC_ARR[None, :, :]).reshape(-1, 4)
    out = {tuple(int(x) for x in row) for row in cand}
    out -= cfg._set
    return sorted(out)  # type: ignore[arg-type]


def _gains(cand_arr: np.ndarray, rest_arr: np.ndarray) -> np.ndarray:
    """Exact unit-edge count from each candidate row to all of rest."""
    if len(cand_arr) == 0 or len(rest_arr) == 0:
        return np.zeros(len(cand_arr), dtype=np.int64)
    return _unit_mask(cand_arr, rest_arr).sum(axis=1)


def _best_candidate(
    rest: list[Coeffs], exclude: frozenset[Coeffs] | set[Coeffs]
) -> tuple[Coeffs | None, int]:
    """Best position adjacent to ``rest`` (max exact edges to rest), not in
    ``exclude``. Ties broken by lexicographically smallest tuple."""
    rest_arr = _as_coeff_array(rest)
    cand_set = {
        (p[0] + u[0], p[1] + u[1], p[2] + u[2], p[3] + u[3])
        for p in rest
        for u in UNIT_COEFFS
    }
    cand_set -= set(exclude)
    if not cand_set:
        return None, 0
    cand = sorted(cand_set)
    gains = _gains(_as_coeff_array(cand), rest_arr)
    k = int(np.argmax(gains))  # first max = lexicographically smallest
    return cand[k], int(gains[k])


def greedy_improve(cfg: MLConfig, passes: int = 4) -> MLConfig:
    """Hill-climb: relocate vertices (worst exact degree first) to the best
    candidate position, accepting only strict exact-edge-count gains.

    Each accepted move strictly increases the total exact edge count, so the
    result never has fewer edges than the input. Stops early on a pass with
    no accepted move. Deterministic.
    """
    pts: list[Coeffs] = list(cfg.points)
    for _ in range(max(0, passes)):
        deg = degrees(MLConfig(pts))
        order = sorted(range(len(pts)), key=lambda i: (int(deg[i]), pts[i]))
        moved = False
        for v in order:
            rest = pts[:v] + pts[v + 1 :]
            rest_arr = _as_coeff_array(rest)
            cur = int(_gains(_as_coeff_array([pts[v]]), rest_arr)[0])
            best, gain = _best_candidate(rest, set(rest) | {pts[v]})
            if best is not None and gain > cur:
                pts[v] = best
                moved = True
        if not moved:
            break
    return MLConfig(pts)


def add_best_point(cfg: MLConfig) -> MLConfig:
    """Append the candidate position gaining the most exact edges (n+1 points).

    Ties broken by lexicographically smallest tuple; deterministic.
    """
    if len(cfg) == 0:
        return MLConfig([(0, 0, 0, 0)])
    best, _gain = _best_candidate(list(cfg.points), cfg._set)
    if best is None:  # unreachable for nonempty cfg
        raise RuntimeError("no candidate positions")
    return cfg.with_point(best)


def drop_worst_point(cfg: MLConfig) -> MLConfig:
    """Remove the vertex of minimum exact degree (n-1 points).

    Ties broken by lexicographically smallest point; deterministic. The new
    edge count is exactly old_count - min_degree.
    """
    if len(cfg) == 0:
        raise ValueError("cannot drop from an empty config")
    deg = degrees(cfg)
    worst = min(range(len(cfg)), key=lambda i: (int(deg[i]), cfg.points[i]))
    return cfg.without_index(worst)
