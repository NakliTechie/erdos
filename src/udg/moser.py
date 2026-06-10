"""Moser-lattice identification.

Port of code/exp9_lattice_id.py (with clustering as in exp4/exp5/exp8):

- The Moser lattice L_3 = Z<1, w1, w3, w1*w3> with w1 = exp(i*pi/3) and
  w3 = exp(i*arccos(5/6)) contains exactly 18 unit vectors, given by the
  paper's Fig 2 coefficient list (``UNIT_COEFFS``) in the basis
  (1, w1, w3, w1*w3).  They form 9 undirected directions (degrees mod 180):
  {0, 16.78, 33.56, 60, 76.78, 93.56, 120, 136.78, 153.56}.
- ``direction_families`` clusters the undirected edge angles of a point
  configuration (chained clustering, as in exp4/exp5/exp8).
- ``lattice_id`` scores the best global rotation aligning a configuration's
  edge-direction set with the Moser-lattice directions (exp9 alignment).

Notes on the port:
- Cluster-mean angles are noisy (~0.05-0.5 deg); use ``lattice_id`` for
  identification, not raw histogram means (HANDOFF section 3).
- One fix over exp9 verbatim: an undirected angle that rounds to exactly
  180.00 (floating-point noise just below the axis, e.g. atan2(-1e-16, 1))
  is normalized to 0.00, so the same horizontal direction is never listed
  twice.  On the 132-edge reference config this is a no-op (12 directions,
  6 matched, identical to exp9/DATA_APPENDIX section F).

Self-contained: stdlib + numpy only.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "W1",
    "W3",
    "UNIT_COEFFS",
    "unit_vectors",
    "ml_directions",
    "direction_families",
    "lattice_id",
]

# Lattice generators (exp9): w1 = exp(i*pi/3), w3 = exp(i*arccos(5/6)).
W1: complex = complex(np.exp(1j * np.pi / 3))
W3: complex = complex(np.exp(1j * np.arccos(5 / 6)))

# The 18 unit vectors of the Moser lattice, as integer coefficient tuples
# in the basis (1, w1, w3, w1*w3) — paper's Fig 2 list, ported from exp9.
UNIT_COEFFS: list[tuple[int, int, int, int]] = [
    (-2, 1, 2, -1),
    (-1, -1, 1, 1),
    (-1, 0, 0, 0),
    (-1, 1, 0, 0),
    (-1, 2, 1, -2),
    (0, -1, 0, 0),
    (0, 0, -1, 0),
    (0, 0, -1, 1),
    (0, 0, 0, -1),
    (0, 0, 0, 1),
    (0, 0, 1, -1),
    (0, 0, 1, 0),
    (0, 1, 0, 0),
    (1, -2, -1, 2),
    (1, -1, 0, 0),
    (1, 0, 0, 0),
    (1, 1, -1, -1),
    (2, -1, -2, 1),
]


def unit_vectors() -> np.ndarray:
    """The 18 unit vectors of the Moser lattice as complex numbers.

    Each is sum(c_k * basis_k) for a coefficient tuple in ``UNIT_COEFFS``
    with basis (1, W1, W3, W1*W3).  All have modulus 1 to ~1e-15.
    """
    basis = np.array([1, W1, W3, W1 * W3])
    return np.array([(np.array(c) * basis).sum() for c in UNIT_COEFFS])


def ml_directions() -> np.ndarray:
    """The 9 undirected Moser-lattice directions in degrees, sorted.

    Angles of the 18 unit vectors taken mod 180 and rounded to 3 decimals
    (exp9); the +/- vector pairs collapse to 9 distinct directions:
    approximately [0, 16.78, 33.56, 60, 76.78, 93.56, 120, 136.78, 153.56].
    """
    uv = unit_vectors()
    dirs = sorted(set(np.round(np.degrees(np.angle(uv)) % 180.0, 3)))
    return np.array(dirs, dtype=float)


def _edge_angles_deg(P: np.ndarray, edges) -> np.ndarray:
    """Undirected angle in degrees, in [0, 180), of each edge (i, j)."""
    P = np.asarray(P, dtype=float)
    E = np.asarray(list(edges), dtype=int).reshape(-1, 2)
    if E.shape[0] == 0:
        return np.empty(0, dtype=float)
    d = P[E[:, 1]] - P[E[:, 0]]
    return np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0


def direction_families(
    P: np.ndarray, edges, cluster_tol: float = 0.5
) -> list[tuple[int, float]]:
    """Cluster undirected edge angles (degrees mod 180) into families.

    Chained clustering as in exp4/exp5/exp8: sort all edge angles, start a
    new cluster whenever the gap to the previous angle is >= cluster_tol;
    additionally the first and last clusters are merged circularly across
    the 0/180 boundary when they chain.  Returns [(count, mean_angle_deg)]
    sorted descending by count (ties: larger angle first, matching the exp
    scripts' ``sorted(..., reverse=True)``).
    """
    ang = _edge_angles_deg(P, edges)
    if ang.size == 0:
        return []
    a = np.sort(ang)
    clusters: list[list[float]] = [[float(a[0])]]
    for x in a[1:]:
        if float(x) - clusters[-1][-1] < cluster_tol:
            clusters[-1].append(float(x))
        else:
            clusters.append([float(x)])
    # circular wrap: a family straddling the 0/180 boundary is one family
    if len(clusters) > 1 and (clusters[0][0] + 180.0 - clusters[-1][-1]) < cluster_tol:
        tail = clusters.pop()
        clusters[0] = [t - 180.0 for t in tail] + clusters[0]
    return sorted(
        ((len(c), float(np.mean(c)) % 180.0) for c in clusters), reverse=True
    )


def lattice_id(P: np.ndarray, edges, match_tol: float = 0.15) -> dict:
    """Best global rotation aligning edge directions with the Moser lattice.

    Port of the exp9 alignment scoring:

    1. Collect the configuration's undirected edge angles, rounded to 2
       decimals (a value rounding to exactly 180.00 is normalized to 0.00),
       de-duplicated and sorted.
    2. Merge near-duplicates: walk the sorted list and keep an angle only
       when it exceeds the last kept angle by more than 0.3 deg.
    3. For every (our direction o, ML direction m) pair, try the global
       rotation rot = (o - m) mod 180; rotate all our directions by -rot
       and count how many land within ``match_tol`` degrees (circularly)
       of some ML direction.  Keep the first rotation with the highest count.

    Returns a dict with:
      n_dirs        — number of merged undirected directions
      dirs          — those directions, degrees, sorted ascending (ndarray)
      best_rotation — winning rotation in degrees, in [0, 180)
      n_matched     — directions matching an ML direction at best_rotation
      matched_mask  — boolean ndarray aligned with ``dirs``
    """
    ang = _edge_angles_deg(P, edges)
    if ang.size == 0:
        return {
            "n_dirs": 0,
            "dirs": np.empty(0, dtype=float),
            "best_rotation": 0.0,
            "n_matched": 0,
            "matched_mask": np.zeros(0, dtype=bool),
        }
    raw: set[float] = set()
    for a in ang:
        r = round(float(a), 2)
        if r == 180.0:  # fp noise just below the axis — same direction as 0
            r = 0.0
        raw.add(r)
    ours = sorted(raw)
    merged = [ours[0]]
    for a in ours[1:]:
        if a - merged[-1] > 0.3:
            merged.append(a)
    merged_arr = np.array(merged, dtype=float)

    mld = ml_directions()
    best_score = -1
    best_rot = 0.0
    best_mask = np.zeros(len(merged), dtype=bool)
    for o in merged:
        for m in mld:
            rot = (o - m) % 180.0
            ours_rot = (merged_arr - rot) % 180.0
            dm = np.min(np.abs(ours_rot[:, None] - mld[None, :]) % 180.0, axis=1)
            dm = np.minimum(dm, 180.0 - dm)
            mask = dm < match_tol
            score = int(mask.sum())
            if score > best_score:
                best_score = score
                best_rot = float(rot)
                best_mask = mask
    return {
        "n_dirs": len(merged),
        "dirs": merged_arr,
        "best_rotation": best_rot,
        "n_matched": best_score,
        "matched_mask": best_mask,
    }
