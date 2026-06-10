"""Tests for udg.moser — Moser-lattice unit vectors, directions, alignment.

Regression targets from plan/contracts.md ("src/udg/moser.py" section) and
DATA_APPENDIX section F.
"""

import numpy as np
import pytest

from udg.moser import (
    UNIT_COEFFS,
    direction_families,
    lattice_id,
    ml_directions,
    unit_vectors,
)

ML_EXPECTED = [0.0, 16.78, 33.56, 60.0, 76.78, 93.56, 120.0, 136.78, 153.56]


def _unit_edges(P: np.ndarray, tol: float = 1e-9) -> list[tuple[int, int]]:
    n = len(P)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    return [(i, j) for i in range(n) for j in range(i + 1, n) if abs(D[i, j] - 1) < tol]


def moser_patch() -> np.ndarray:
    """Small exact Moser-lattice patch: {0} u {uv_i} u {uv_i + uv_j}, deduped."""
    uv = unit_vectors()
    pts = [0j] + list(uv) + [a + b for a in uv for b in uv]
    pts = np.asarray(pts)
    keys = np.round(pts.real, 9) + 1j * np.round(pts.imag, 9)
    _, idx = np.unique(keys, return_index=True)
    pts = pts[np.sort(idx)]
    return np.column_stack([pts.real, pts.imag])


def tri_patch(m: int = 4) -> np.ndarray:
    """Perfect triangular-lattice patch (m x m rhombus), unit edge length."""
    return np.array(
        [[i + 0.5 * j, j * np.sqrt(3) / 2] for i in range(m) for j in range(m)]
    )


def _rotate(P: np.ndarray, deg: float) -> np.ndarray:
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return P @ R.T


# ---------------------------------------------------------------- unit vectors


def test_unit_vectors_are_18_and_unit_length():
    uv = unit_vectors()
    assert len(UNIT_COEFFS) == 18
    assert uv.shape == (18,)
    assert np.max(np.abs(np.abs(uv) - 1.0)) < 1e-12


def test_ml_directions_nine_values():
    dirs = ml_directions()
    assert dirs.shape == (9,)
    assert np.all(np.diff(dirs) > 0)  # sorted, distinct
    assert np.max(np.abs(dirs - np.array(ML_EXPECTED))) < 0.01


# ----------------------------------------------------------- direction families


def test_direction_families_triangular_patch():
    P = tri_patch(4)
    edges = _unit_edges(P)
    assert len(edges) == 33  # 12 + 12 + 9 in a 4x4 rhombus
    fams = direction_families(P, edges, cluster_tol=0.5)
    assert len(fams) == 3
    counts = [c for c, _ in fams]
    assert counts == sorted(counts, reverse=True) == [12, 12, 9]
    assert sum(counts) == len(edges)
    angles = sorted(a for _, a in fams)
    assert np.allclose(angles, [0.0, 60.0, 120.0], atol=1e-6)
    # 3 families exactly 60 degrees apart
    assert np.allclose(np.diff(angles), [60.0, 60.0], atol=1e-6)


def test_direction_families_moser_patch_nine_families():
    P = moser_patch()
    edges = _unit_edges(P)
    fams = direction_families(P, edges, cluster_tol=0.5)
    assert len(fams) == 9
    assert sum(c for c, _ in fams) == len(edges)
    angles = np.sort([a for _, a in fams])
    assert np.max(np.abs(angles - ml_directions())) < 0.01


def test_direction_families_empty_edges():
    P = tri_patch(2)
    assert direction_families(P, []) == []


# -------------------------------------------------------------------- lattice_id


def test_lattice_id_exact_moser_patch_full_match():
    P = moser_patch()
    edges = _unit_edges(P)
    res = lattice_id(P, edges)
    assert res["n_dirs"] == 9
    assert res["n_matched"] == res["n_dirs"] == 9
    assert res["matched_mask"].shape == (9,)
    assert bool(res["matched_mask"].all())
    # unrotated: best rotation is 0 (circularly)
    rot = res["best_rotation"]
    assert min(rot, 180.0 - rot) < 0.05
    assert np.max(np.abs(np.sort(res["dirs"]) - ml_directions())) < 0.05


def test_lattice_id_rotated_patch_full_match():
    P = _rotate(moser_patch(), 7.3)
    edges = _unit_edges(P)  # rotation preserves distances
    res = lattice_id(P, edges)
    assert res["n_dirs"] == 9
    assert res["n_matched"] == res["n_dirs"] == 9
    assert bool(res["matched_mask"].all())
    assert abs(res["best_rotation"] - 7.3) < 0.05


def test_lattice_id_empty_edges():
    res = lattice_id(tri_patch(2), [])
    assert res["n_dirs"] == 0
    assert res["n_matched"] == 0
    assert res["dirs"].size == 0
    assert res["matched_mask"].size == 0
