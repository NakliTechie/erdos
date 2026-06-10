"""Tests for scripts/ml_coords.py — exact Moser-lattice certification.

(a) the 18 UNIT_COEFFS tuples are EXACTLY unit length under the Q(sqrt3,
    sqrt11) arithmetic, and exact sums/differences reproduce floats to 1e-12;
(b) a synthetic exact ML patch (~50 points from UNIT_COEFFS sums), rotated
    by 12.34 deg and translated, certifies with exact count == float count
    and the rotation recovered (mod the lattice's 60-degree symmetry);
(c) a 1e-3-jittered version of the same patch must FAIL certification.
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ml_coords  # noqa: E402
from ml_coords import (  # noqa: E402
    QF,
    QF_ONE,
    certify_config,
    exact_dist2,
    exact_xy,
    refine_rotation,
    sanity_check_arithmetic,
)
from udg.moser import UNIT_COEFFS, W1, W3, unit_vectors  # noqa: E402


# ---------------------------------------------------------------------------
# (a) exact arithmetic
# ---------------------------------------------------------------------------

def test_multiplication_table():
    s3 = QF(0, 1)
    s11 = QF(0, 0, 1)
    s33 = QF(0, 0, 0, 1)
    assert s3 * s3 == QF(3)
    assert s11 * s11 == QF(11)
    assert s33 * s33 == QF(33)
    assert s3 * s11 == s33
    assert s3 * s33 == QF(0, 0, 3)    # 3*sqrt11
    assert s11 * s33 == QF(0, 11)     # 11*sqrt3
    # a nontrivial product, checked against floats
    x = QF(Fraction(1, 2), Fraction(-2, 3), Fraction(5, 12), Fraction(1, 7))
    y = QF(Fraction(-3, 4), Fraction(1, 6), Fraction(2, 5), Fraction(-1, 2))
    assert abs((x * y).to_float() - x.to_float() * y.to_float()) < 1e-12


def test_18_unit_vectors_exactly_unit():
    origin = (0, 0, 0, 0)
    for c in UNIT_COEFFS:
        assert exact_dist2(c, origin) == QF_ONE, c


def test_exact_matches_float_basis():
    # exact Re/Im of the basis vs numpy complex
    basis = [1 + 0j, W1, W3, W1 * W3]
    for k in range(4):
        coeffs = tuple(1 if t == k else 0 for t in range(4))
        re, im = exact_xy(coeffs)
        assert abs(re.to_float() - basis[k].real) < 1e-15
        assert abs(im.to_float() - basis[k].imag) < 1e-15
    # full sums/differences sweep over the 18 unit vectors (raises on failure)
    sanity_check_arithmetic()


# ---------------------------------------------------------------------------
# synthetic exact ML patch
# ---------------------------------------------------------------------------

def _ml_patch(npts: int = 50) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """{0} u {uv_i} u {uv_i+uv_j}, dedup, truncated to npts (closest first).

    Connectivity is guaranteed: every uv_i is unit distance from 0, every
    kept sum uv_i+uv_j is unit distance from uv_i and uv_j.
    """
    pts = {(0, 0, 0, 0)}
    pts.update(UNIT_COEFFS)
    sums = set()
    for i in range(len(UNIT_COEFFS)):
        for j in range(i + 1, len(UNIT_COEFFS)):
            s = tuple(a + b for a, b in zip(UNIT_COEFFS[i], UNIT_COEFFS[j]))
            if s not in pts:
                sums.add(s)
    core = [(0, 0, 0, 0)] + sorted(UNIT_COEFFS)

    def _r(c):
        re, im = exact_xy(c)
        return (re.to_float() ** 2 + im.to_float() ** 2, c)

    extra = [c for _, c in sorted(_r(c) for c in sums)]
    coeffs = (core + extra)[:npts]
    P = np.array(
        [(exact_xy(c)[0].to_float(), exact_xy(c)[1].to_float()) for c in coeffs]
    )
    return coeffs, P


def _rotate_translate(P: np.ndarray, deg: float, t=(0.31, -0.73)) -> np.ndarray:
    th = math.radians(deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return P @ R.T + np.asarray(t)


# ---------------------------------------------------------------------------
# (b) exact patch certifies, rotation recovered
# ---------------------------------------------------------------------------

def test_exact_patch_certifies():
    coeffs, P0 = _ml_patch(50)
    P = _rotate_translate(P0, 12.34)
    r = certify_config(P, name="synthetic_ml_patch")
    assert r["certified"] is True
    assert r["n"] == 50
    assert r["float_unit_edges"] > 50  # dense little patch
    assert r["exact_unit_pairs"] == r["float_unit_edges"]
    assert r["cycle_failures"] == 0
    assert r["distinct_ok"] is True
    assert r["n_components"] == 1
    assert r["max_angular_residual_rad"] < 1e-9
    assert r["embed_max_err"] < 1e-9
    # rotation recovered modulo the lattice's own 60-deg rotational symmetry
    # (multiplication by w1 maps the 18 unit vectors to themselves)
    delta = (r["rotation_deg"] - 12.34) % 60.0
    assert min(delta, 60.0 - delta) < 1e-6


def test_exact_patch_unrotated_zero_rotation():
    _, P0 = _ml_patch(30)
    r = certify_config(P0, name="synthetic_unrotated")
    assert r["certified"] is True
    delta = r["rotation_deg"] % 60.0
    assert min(delta, 60.0 - delta) < 1e-9


# ---------------------------------------------------------------------------
# (c) jittered patch must fail
# ---------------------------------------------------------------------------

def test_jittered_patch_fails():
    _, P0 = _ml_patch(50)
    P = _rotate_translate(P0, 12.34)
    rng = np.random.default_rng(42)
    Pj = P + rng.normal(scale=1e-3, size=P.shape)
    # at tol 1e-9 the jittered config simply has no unit edges at all
    r_tight = certify_config(Pj, name="jittered_tight")
    assert r_tight["certified"] is False
    # at a loose tol the edges are back, the combinatorial coords still
    # propagate (1e-3 jitter cannot flip a nearest-of-18 match across the
    # ~8.4 deg decision boundary) -- the float-consistency gates must trip
    r = certify_config(Pj, name="jittered_loose", tol=5e-3)
    assert r["certified"] is False
    assert (
        r["max_angular_residual_rad"] >= ml_coords.ANG_RESID_GATE
        or r.get("embed_max_err", 1.0) >= ml_coords.EMBED_ERR_GATE
        or r["cycle_failures"] > 0
    )


# ---------------------------------------------------------------------------
# rotation refinement in isolation
# ---------------------------------------------------------------------------

def test_refine_rotation_recovers_angle():
    _, P0 = _ml_patch(40)
    P = _rotate_translate(P0, 7.3)
    from udg.counting import unit_edges

    edges = unit_edges(P)
    uv = unit_vectors()
    # start from a coarse estimate 0.05 deg off
    theta, assign, max_resid = refine_rotation(P, edges, math.radians(7.35))
    assert max_resid < 1e-10
    E = np.asarray(edges)
    v = (P[E[:, 1], 0] - P[E[:, 0], 0]) + 1j * (P[E[:, 1], 1] - P[E[:, 0], 1])
    # every edge vector equals e^{i theta} * its matched unit vector
    assert np.max(np.abs(v - np.exp(1j * theta) * uv[assign])) < 1e-9
