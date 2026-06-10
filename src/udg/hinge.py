"""udg.hinge — hinge-locking toolkit for the flexible-skeleton hypothesis.

Implements plan/hinge-design.md (Batch B). Context: the 132-edge n=40 config
has 12 undirected edge directions in 4 families-of-3 (each family internally
60 deg apart). After global Moser-lattice (ML) alignment two families sit on
ML angles and two float (DATA_APPENDIX section F; raw-frame family angles
~{0.04, 0.94, 16.87, 34.12} deg, aligned-frame ~{0.00, 0.90, 16.83, 34.08}).
Hypothesis (HANDOFF section 2.4 / 5.A.1): continuous search finds a *flexible
framework skeleton*; the final edges fire only when floating families are
locked at exact algebraic angles.

Capabilities
------------
classify_families   global ML alignment via udg.moser.lattice_id, then cluster
                    edge directions into families-of-3 (mod 60); per family:
                    angle (both frames), edge count, vertex set, nearest
                    candidate target angle, offset to target, locked flag.
lock_family         strategy (a): rotate-and-project homotopy. Incremental
                    rigid rotation of the family vertex set about the hinge
                    center, Gauss-Newton projection onto ALL original edges
                    after each increment, adaptive increment halving on stall.
rigidity_matrix     |E| x 2n first-order rigidity matrix.
flex_dimension      nullity(R) - 3 (the 3 trivial planar motions).
internal_flex_basis orthonormal basis of internal flexes (trivial motions
                    projected out) — robust cross-check of flex_dimension.
follow_flex         strategy (b): predictor-corrector along the null-space
                    direction maximizing the target family's angular velocity;
                    Gauss-Newton corrector each step; stops at target angle or
                    flex death.
fire_check          unit-edge count + near-miss histogram of |d - 1| in bands
                    [1e-9, 1e-6, 1e-4, 1e-2, 5e-2].

DISCIPLINE: nothing in this module is an audit. No edge count produced here
may be claimed without udg.audit.audit(P).passed — and watch min_sep_after:
point merging during projection is the tolerance exploit reappearing through
the repair step. Both lock_family and follow_flex therefore treat a projection
landing below MIN_SEP as a stall and revert it.

Angle conventions: all angles in degrees. Undirected edge directions live mod
180; a family-of-3 (directions 60 deg apart) has a well-defined angle mod 60,
which is what `family_angle`, targets and offsets use.

Uses udg.audit (gauss_newton, min_separation) and udg.moser (lattice_id,
ml_directions); otherwise stdlib + numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from udg.audit import MIN_SEP, gauss_newton, min_separation
from udg.moser import lattice_id, ml_directions

__all__ = [
    "LOCK_TOL",
    "NEAR_MISS_EDGES",
    "STALL_RESIDUAL",
    "Family",
    "FamilyClassification",
    "FireCheck",
    "FlexResult",
    "LockResult",
    "candidate_targets",
    "classify_families",
    "edge_residual",
    "family_angle",
    "fire_check",
    "flex_dimension",
    "follow_flex",
    "internal_flex_basis",
    "lock_family",
    "rigidity_matrix",
    "rotate_points",
    "signed_delta",
]

TOL = 1e-9             # unit-distance tolerance (package convention)
LOCK_TOL = 0.2         # deg; |offset to target| below this => family "locked"
STALL_RESIDUAL = 1e-18 # residual plateau above this after projection = stall
SVD_TOL = 1e-9         # singular-value threshold for rank / null space
NEAR_MISS_EDGES = (1e-9, 1e-6, 1e-4, 1e-2, 5e-2)  # |d-1| histogram band edges


# ---------------------------------------------------------------------------
# angle / geometry helpers
# ---------------------------------------------------------------------------

def _edges_list(edges) -> list[tuple[int, int]]:
    return [(int(i), int(j)) for i, j in edges]


def _family_edges(family) -> list[tuple[int, int]]:
    """Accept a Family instance or any iterable of (i, j) pairs."""
    return _edges_list(getattr(family, "edges", family))


def _edge_angles_deg(P: np.ndarray, edges) -> np.ndarray:
    """Undirected angle in degrees, in [0, 180), of each edge (i, j)."""
    P = np.asarray(P, dtype=float)
    E = np.asarray(_edges_list(edges), dtype=int).reshape(-1, 2)
    if E.shape[0] == 0:
        return np.empty(0, dtype=float)
    d = P[E[:, 1]] - P[E[:, 0]]
    return np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180.0


def signed_delta(a: float, b: float, period: float = 60.0) -> float:
    """Signed circular difference a - b on a circle of `period` degrees.

    Result in (-period/2, period/2]. signed_delta(target, current) is the
    rotation (deg) that carries `current` to `target` the short way.
    """
    d = (float(a) - float(b)) % period
    if d > period / 2.0:
        d -= period
    return d


def circular_mean(angles_deg, period: float = 60.0) -> float:
    """Circular mean of angles on a circle of `period` degrees, in [0, period)."""
    a = np.asarray(angles_deg, dtype=float)
    z = np.exp(1j * np.radians(a * (360.0 / period)))
    m = np.degrees(np.angle(z.mean())) * (period / 360.0)
    return float(m % period)


def family_angle(P, edges, period: float = 60.0) -> float:
    """Family angle: circular mean of the edges' undirected angles mod `period`.

    For a family-of-3 (directions 60 deg apart) this is the family's offset
    from the 0-deg lattice direction — the quantity DATA_APPENDIX section F
    reports as ~{0.04, 0.94, 16.87, 34.12} for the 132-edge config.
    """
    return circular_mean(_edge_angles_deg(P, edges), period)


def rotate_points(
    P, angle_deg: float, center=None, indices=None
) -> np.ndarray:
    """Rigidly rotate points (all, or the subset `indices`) by angle_deg (CCW).

    center defaults to the centroid of the rotated subset. Returns a copy.
    """
    Q = np.asarray(P, dtype=float).copy()
    idx = np.arange(len(Q)) if indices is None else np.asarray(list(indices), dtype=int)
    if center is None:
        center = Q[idx].mean(axis=0)
    c = np.asarray(center, dtype=float)
    th = np.radians(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Q[idx] = (Q[idx] - c) @ R.T + c
    return Q


def edge_residual(P, edges) -> float:
    """sum over (i,j) in edges of (|p_i - p_j| - 1)^2 (the GN objective)."""
    P = np.asarray(P, dtype=float)
    E = np.asarray(_edges_list(edges), dtype=int).reshape(-1, 2)
    if E.shape[0] == 0:
        return 0.0
    v = P[E[:, 0]] - P[E[:, 1]]
    r = np.sqrt((v * v).sum(1)) - 1.0
    return float((r * r).sum())


def _circular_index_clusters(
    values, period: float, tol: float
) -> list[np.ndarray]:
    """Chained clustering of angles on a circle of `period` degrees.

    Sort, start a new cluster when the gap to the previous value is >= tol,
    merge the first and last clusters across the 0/period boundary when they
    chain (as udg.moser.direction_families, but returning *index* groups).
    """
    v = np.asarray(values, dtype=float) % period
    if v.size == 0:
        return []
    order = np.argsort(v, kind="stable")
    sv = v[order]
    groups: list[list[int]] = [[0]]
    for k in range(1, len(sv)):
        if sv[k] - sv[k - 1] < tol:
            groups[-1].append(k)
        else:
            groups.append([k])
    if len(groups) > 1 and (sv[groups[0][0]] + period - sv[-1]) < tol:
        groups[0] = groups.pop() + groups[0]
    return [order[np.asarray(g, dtype=int)] for g in groups]


# ---------------------------------------------------------------------------
# 1. classification
# ---------------------------------------------------------------------------

def candidate_targets() -> np.ndarray:
    """Candidate lock-target angles mod 60, sorted ascending.

    The 9 ML directions plus all mod-60 shifts of arcsin(1/sqrt(12)) =
    16.7786 deg and arccos(5/6) = 33.5573 deg — which all collapse mod 60 to
    exactly {0, arcsin(1/sqrt(12)), arccos(5/6)} in degrees.
    """
    cands = [
        0.0,
        float(np.degrees(np.arcsin(1.0 / np.sqrt(12.0)))),  # 16.77865...
        float(np.degrees(np.arccos(5.0 / 6.0))),            # 33.55730...
    ]
    for m in ml_directions():
        m60 = float(m) % 60.0
        if min(abs(signed_delta(m60, c)) for c in cands) > 0.01:
            cands.append(m60)  # pragma: no cover — ML dirs all collapse
    return np.array(sorted(cands), dtype=float)


@dataclass
class Family:
    """One direction family-of-3 (see module docstring for conventions).

    mean_angle_raw is the family angle mod 60 in the INPUT frame — this is
    what DATA_APPENDIX section F calls the family "offsets" ({0.04, 0.94,
    16.87, 34.12} for the 132-edge config). mean_angle is the same quantity
    in the ALIGNED frame (what locking compares against targets); offset is
    the design doc's delta = signed distance from mean_angle to target.
    """

    index: int                      # position after sorting by mean_angle_raw
    edges: list[tuple[int, int]]    # this family's edges (subset of input)
    n_edges: int
    vertices: list[int]             # sorted vertex set spanned by the edges
    directions: list[float]         # aligned-frame mod-180 direction means (3)
    mean_angle: float               # aligned-frame family angle mod 60
    mean_angle_raw: float           # input-frame family angle mod 60
    target: float                   # nearest candidate target angle mod 60
    offset: float                   # signed_delta(mean_angle, target), deg
    locked: bool                    # |offset| < lock_tol


@dataclass
class FamilyClassification:
    P: np.ndarray            # globally ML-aligned copy of the input config
    rotation_deg: float      # rotation applied to the input (= -best_rotation)
    families: list[Family]   # sorted by mean_angle_raw ascending
    lattice: dict            # raw udg.moser.lattice_id() result on the input


def classify_families(
    P, edges, lock_tol: float = LOCK_TOL, cluster_tol: float = 0.5
) -> FamilyClassification:
    """Globally align via lattice_id, cluster edge directions into families.

    1. Rotate P by -lattice_id(P, edges)['best_rotation'] about its centroid,
       so ML-exact families sit on ML angles (rigid: edges are preserved).
    2. Chain-cluster the aligned edge angles mod 60 (tolerance cluster_tol)
       into families; each family's 3 mod-180 directions are re-derived per
       family.
    3. Per family: nearest candidate target (candidate_targets()), signed
       offset, locked flag (|offset| < lock_tol).

    Returns the aligned config + families; family edge lists partition the
    input edge list. Locking (lock_family / follow_flex) should be run on the
    ALIGNED config `result.P` so that targets are meaningful.
    """
    P = np.asarray(P, dtype=float)
    E = _edges_list(edges)
    info = lattice_id(P, E)
    rot = float(info["best_rotation"])
    Pa = rotate_points(P, -rot, center=P.mean(axis=0))

    ang_aligned = _edge_angles_deg(Pa, E)
    ang_raw = _edge_angles_deg(P, E)
    cands = candidate_targets()

    families: list[Family] = []
    for idx in _circular_index_clusters(ang_aligned % 60.0, 60.0, cluster_tol):
        fam_edges = [E[k] for k in idx]
        sub = ang_aligned[idx]
        mean_a = circular_mean(sub, 60.0)
        mean_raw = circular_mean(ang_raw[idx], 60.0)
        dirs = sorted(
            circular_mean(sub[g], 180.0)
            for g in _circular_index_clusters(sub, 180.0, cluster_tol)
        )
        deltas = np.array([signed_delta(mean_a, float(c)) for c in cands])
        j = int(np.argmin(np.abs(deltas)))
        offset = float(deltas[j])
        families.append(
            Family(
                index=0,
                edges=fam_edges,
                n_edges=len(fam_edges),
                vertices=sorted({v for e in fam_edges for v in e}),
                directions=[float(x) for x in dirs],
                mean_angle=mean_a,
                mean_angle_raw=mean_raw,
                target=float(cands[j]),
                offset=offset,
                locked=abs(offset) < lock_tol,
            )
        )
    families.sort(key=lambda f: f.mean_angle_raw)
    for i, f in enumerate(families):
        f.index = i
    return FamilyClassification(
        P=Pa, rotation_deg=-rot, families=families, lattice=info
    )


# ---------------------------------------------------------------------------
# 2. strategy (a): rotate-and-project homotopy
# ---------------------------------------------------------------------------

@dataclass
class LockResult:
    P: np.ndarray          # final configuration (last ACCEPTED projection)
    converged: bool        # reached target angle with clean residual/min-sep
    family_angle: float    # final family angle mod 60 (deg)
    residual: float        # final sum (|d|-1)^2 over ALL given edges
    min_sep: float         # final minimum pairwise separation
    stop_reason: str       # "target" | "stalled" | "max_iter"
    diagnostics: dict = field(default_factory=dict)


def lock_family(
    P,
    edges,
    family,
    target: float | None = None,
    n_increments: int = 10,
    *,
    angle_tol: float = 1e-4,
    stall_residual: float = STALL_RESIDUAL,
    max_halvings: int = 3,
    min_sep: float = MIN_SEP,
    gn_iters: int = 8000,
    max_iter: int | None = None,
) -> LockResult:
    """Rotate-and-project homotopy lock of one direction family (design 3a).

    P should satisfy the given edges (near-)exactly; `edges` are ALL original
    edge constraints, `family` a Family (or iterable of its edges), `target`
    the goal angle mod 60 (defaults to family.target).

    Hinge center: centroid of the family vertices shared with non-family
    edges (fallback: family centroid), recomputed from current positions each
    increment. Each increment rigidly rotates the family vertex set by
    delta/n_increments (clamped to the remaining angle) about the hinge
    center, then Gauss-Newton-projects onto ALL edges. A projection with
    residual > stall_residual OR min separation < min_sep is a stall: it is
    reverted and the increment halved; after max_halvings such halvings the
    lock gives up ("stalled"). Otherwise iterate until the family angle is
    within angle_tol of target ("target") or max_iter increments
    ("max_iter").

    converged requires: stop_reason == "target", final residual <=
    stall_residual, final min-sep >= min_sep. The result is NOT an audit —
    run udg.audit.audit() before claiming any edge count.
    """
    P = np.asarray(P, dtype=float).copy()
    E = _edges_list(edges)
    fam_edges = _family_edges(family)
    if target is None:
        target = float(family.target)
    if max_iter is None:
        max_iter = max(20 * n_increments, 50)

    fam_set = {tuple(sorted(e)) for e in fam_edges}
    fam_verts = sorted({v for e in fam_edges for v in e})
    core_verts = {v for e in E if tuple(sorted(e)) not in fam_set for v in e}
    hinge_verts = sorted(set(fam_verts) & core_verts) or list(fam_verts)

    theta0 = family_angle(P, fam_edges)
    delta0 = signed_delta(target, theta0)
    residual0 = edge_residual(P, E)
    step = abs(delta0) / n_increments if delta0 != 0.0 else 0.0

    halvings = 0
    history: list[dict] = []
    reason = "max_iter"
    for _ in range(max_iter):
        cur = family_angle(P, fam_edges)
        rem = signed_delta(target, cur)
        if abs(rem) <= angle_tol:
            reason = "target"
            break
        if step == 0.0:  # entered at target but drifted past tol: re-derive
            step = abs(rem) / n_increments
        d = float(np.sign(rem) * min(step, abs(rem)))
        center = P[hinge_verts].mean(axis=0)
        P_try = rotate_points(P, d, center=center, indices=fam_verts)
        Q, _ = gauss_newton(P_try, E, iters=gn_iters)
        r = edge_residual(Q, E)
        ms = min_separation(Q)
        if r > stall_residual or ms < min_sep:
            halvings += 1
            if halvings > max_halvings:
                reason = "stalled"
                break
            step /= 2.0
            continue  # revert (P unchanged), retry with smaller increment
        P = Q
        history.append(
            {
                "step_deg": d,
                "angle": family_angle(P, fam_edges),
                "residual": r,
                "min_sep": ms,
            }
        )

    final_angle = family_angle(P, fam_edges)
    final_res = edge_residual(P, E)
    final_ms = min_separation(P)
    converged = (
        reason == "target" and final_res <= stall_residual and final_ms >= min_sep
    )
    return LockResult(
        P=P,
        converged=converged,
        family_angle=final_angle,
        residual=final_res,
        min_sep=final_ms,
        stop_reason=reason,
        diagnostics={
            "hinge_vertices": hinge_verts,
            "family_vertices": fam_verts,
            "initial_angle": theta0,
            "initial_residual": residual0,
            "target": float(target),
            "delta": delta0,
            "n_steps": len(history),
            "n_halvings": halvings,
            "history": history,
        },
    )


# ---------------------------------------------------------------------------
# 3. strategy (b): rigidity / flex following
# ---------------------------------------------------------------------------

def rigidity_matrix(P, edges) -> np.ndarray:
    """First-order rigidity matrix, shape (|E|, 2n).

    Row for edge (i, j): (p_i - p_j) at columns (2i, 2i+1) and -(p_i - p_j)
    at (2j, 2j+1). R @ v = d/dt of (squared lengths)/2 under velocity field v
    (v flattened as (x0, y0, x1, y1, ...)); infinitesimal flexes = null(R).
    """
    P = np.asarray(P, dtype=float)
    E = np.asarray(_edges_list(edges), dtype=int).reshape(-1, 2)
    m, n = E.shape[0], len(P)
    R = np.zeros((m, 2 * n))
    if m == 0:
        return R
    d = P[E[:, 0]] - P[E[:, 1]]
    rows = np.arange(m)
    R[rows, 2 * E[:, 0]] = d[:, 0]
    R[rows, 2 * E[:, 0] + 1] = d[:, 1]
    R[rows, 2 * E[:, 1]] = -d[:, 0]
    R[rows, 2 * E[:, 1] + 1] = -d[:, 1]
    return R


def flex_dimension(P, edges, tol: float = SVD_TOL) -> int:
    """Internal flex dimension = nullity(rigidity_matrix) - 3.

    The 3 subtracted dimensions are the trivial planar motions (2
    translations + 1 rotation), so this assumes >= 2 distinct points (else
    the trivial space is smaller and the value undercounts). 0 = first-order
    rigid; > 0 = flexible skeleton. Singular values are thresholded at
    tol * max(1, s_max). Cross-check: internal_flex_basis(P, edges).shape[1].
    """
    R = rigidity_matrix(P, edges)
    if min(R.shape) == 0:
        return R.shape[1] - 3
    s = np.linalg.svd(R, compute_uv=False)
    thresh = tol * max(1.0, float(s[0]))
    rank = int((s > thresh).sum())
    return R.shape[1] - rank - 3


def _trivial_motions(P: np.ndarray) -> np.ndarray:
    """Orthonormal basis (3, 2n) of the trivial planar motions of P.

    Rows span {x-translation, y-translation, rotation about the centroid}.
    """
    P = np.asarray(P, dtype=float)
    n = len(P)
    c = P.mean(axis=0)
    tx = np.tile([1.0, 0.0], n)
    ty = np.tile([0.0, 1.0], n)
    rot = np.empty(2 * n)
    rot[0::2] = -(P[:, 1] - c[1])
    rot[1::2] = P[:, 0] - c[0]
    Q, _ = np.linalg.qr(np.vstack([tx, ty, rot]).T)
    return Q.T


def internal_flex_basis(P, edges, tol: float = SVD_TOL) -> np.ndarray:
    """Orthonormal basis (2n, k) of internal flexes: null(R) with the 3
    trivial motions projected out (computed as the null space of R stacked
    on the orthonormalized trivial motions). k == flex_dimension for generic
    configs; this construction stays correct even near degeneracy.
    """
    P = np.asarray(P, dtype=float)
    A = np.vstack([rigidity_matrix(P, edges), _trivial_motions(P)])
    _, s, vh = np.linalg.svd(A)
    thresh = tol * max(1.0, float(s[0]) if s.size else 1.0)
    rank = int((s > thresh).sum())
    return vh[rank:].T


def _family_angle_gradient(P: np.ndarray, fam_edges) -> np.ndarray:
    """Gradient c (2n,) of the family mean angle w.r.t. point velocities.

    d(theta_family)/dt = c @ v in radians per unit time, where for each edge
    e = p_i - p_j: d(theta_e)/dt = (e_x ed_y - e_y ed_x)/|e|^2 with
    ed = v_i - v_j, averaged over the family's edges.
    """
    P = np.asarray(P, dtype=float)
    E = np.asarray(_edges_list(fam_edges), dtype=int).reshape(-1, 2)
    c = np.zeros(2 * len(P))
    if E.shape[0] == 0:
        return c
    e = P[E[:, 0]] - P[E[:, 1]]
    L2 = (e * e).sum(1)
    m = E.shape[0]
    wx = -e[:, 1] / L2 / m  # coefficient on (v_i - v_j)_x
    wy = e[:, 0] / L2 / m   # coefficient on (v_i - v_j)_y
    np.add.at(c, 2 * E[:, 0], wx)
    np.add.at(c, 2 * E[:, 0] + 1, wy)
    np.add.at(c, 2 * E[:, 1], -wx)
    np.add.at(c, 2 * E[:, 1] + 1, -wy)
    return c


@dataclass
class FlexResult:
    P: np.ndarray          # final configuration (last ACCEPTED corrector)
    converged: bool        # reached target angle with clean residual/min-sep
    family_angle: float    # final family angle mod 60 (deg)
    residual: float        # final sum (|d|-1)^2 over ALL given edges
    min_sep: float         # final minimum pairwise separation
    flex_dim: int          # internal flex dimension at the START (a finding)
    stop_reason: str       # "target" | "flex_death_rigid" |
                           # "flex_death_velocity" | "stalled" | "max_steps"
    diagnostics: dict = field(default_factory=dict)


def follow_flex(
    P,
    edges,
    family,
    target: float | None = None,
    *,
    step_deg: float = 0.5,
    angle_tol: float = 1e-4,
    stall_residual: float = STALL_RESIDUAL,
    max_halvings: int = 3,
    max_steps: int = 500,
    min_sep: float = MIN_SEP,
    svd_tol: float = SVD_TOL,
    gn_iters: int = 8000,
    max_point_step: float = 0.2,
) -> FlexResult:
    """Predictor-corrector flex following toward a target family angle (3b).

    Each step: compute the internal flex basis B (trivial motions removed);
    pick the unit direction v in span(B) maximizing the family's angular
    velocity (v = B B^T c / |B^T c| with c = _family_angle_gradient); take a
    predictor step of at most step_deg of family rotation (and at most
    max_point_step per-point displacement); Gauss-Newton-correct onto ALL
    edges. A corrector landing above stall_residual or below min_sep is
    reverted and step_deg halved (give up after max_halvings). Stops at the
    target angle, at flex death (internal flex dim 0, or zero angular
    velocity along every remaining flex), on stall, or at max_steps.

    flex_dim in the result is the internal flex dimension of the INPUT —
    per the design doc, that number is itself a finding. NOT an audit.
    """
    P = np.asarray(P, dtype=float).copy()
    E = _edges_list(edges)
    fam_edges = _family_edges(family)
    if target is None:
        target = float(family.target)

    flex_dim0 = internal_flex_basis(P, E, svd_tol).shape[1]
    local_step = float(step_deg)
    halvings = 0
    history: list[dict] = []
    reason = "max_steps"
    for _ in range(max_steps):
        cur = family_angle(P, fam_edges)
        rem = signed_delta(target, cur)
        if abs(rem) <= angle_tol:
            reason = "target"
            break
        B = internal_flex_basis(P, E, svd_tol)
        if B.shape[1] == 0:
            reason = "flex_death_rigid"
            break
        c = _family_angle_gradient(P, fam_edges)
        w = B.T @ c
        speed = float(np.linalg.norm(w))  # rad per unit time along best flex
        if speed < 1e-12:
            reason = "flex_death_velocity"
            break
        v = (B @ w) / speed  # |v| = 1, angular velocity along v = +speed
        want = float(np.radians(min(local_step, abs(rem))) * np.sign(rem))
        eps = want / speed
        Vp = v.reshape(-1, 2)
        rmax = float(np.sqrt((Vp * Vp).sum(1)).max())
        if abs(eps) * rmax > max_point_step:
            eps = float(np.sign(eps)) * max_point_step / rmax
        Q, _ = gauss_newton(P + eps * Vp, E, iters=gn_iters)
        r = edge_residual(Q, E)
        ms = min_separation(Q)
        if r > stall_residual or ms < min_sep:
            halvings += 1
            if halvings > max_halvings:
                reason = "stalled"
                break
            local_step /= 2.0
            continue  # revert (P unchanged), retry with smaller step
        P = Q
        history.append(
            {
                "angle": family_angle(P, fam_edges),
                "residual": r,
                "min_sep": ms,
                "flex_dim": int(B.shape[1]),
                "speed": speed,
                "eps": eps,
            }
        )

    final_angle = family_angle(P, fam_edges)
    final_res = edge_residual(P, E)
    final_ms = min_separation(P)
    converged = (
        reason == "target" and final_res <= stall_residual and final_ms >= min_sep
    )
    return FlexResult(
        P=P,
        converged=converged,
        family_angle=final_angle,
        residual=final_res,
        min_sep=final_ms,
        flex_dim=flex_dim0,
        stop_reason=reason,
        diagnostics={
            "target": float(target),
            "n_steps": len(history),
            "n_halvings": halvings,
            "history": history,
        },
    )


# ---------------------------------------------------------------------------
# 4. fire check
# ---------------------------------------------------------------------------

@dataclass
class FireCheck:
    n: int                                  # number of points
    n_unit: int                             # pairs with |d - 1| < tol
    bands: list[tuple[float, float, int]]   # (lo, hi, count): |d-1| in [lo, hi)
    closest_nonunit: float                  # min |d - 1| among pairs >= tol


def fire_check(P, tol: float = TOL) -> FireCheck:
    """Unit-edge count + near-miss histogram of |d - 1| (design step 4).

    Bands are [1e-9, 1e-6), [1e-6, 1e-4), [1e-4, 1e-2), [1e-2, 5e-2) — fixed
    (NEAR_MISS_EDGES) regardless of tol. Near-misses migrating toward the
    low bands after a lock is partial evidence for the firing hypothesis even
    when no new edge crosses tol. NOT an audit: any n_unit you want to claim
    must pass udg.audit.audit().
    """
    P = np.asarray(P, dtype=float)
    n = len(P)
    iu, ju = np.triu_indices(n, 1)
    d = np.sqrt(((P[iu] - P[ju]) ** 2).sum(1))
    x = np.abs(d - 1.0)
    n_unit = int((x < tol).sum())
    bands = [
        (lo, hi, int(((x >= lo) & (x < hi)).sum()))
        for lo, hi in zip(NEAR_MISS_EDGES[:-1], NEAR_MISS_EDGES[1:])
    ]
    nonunit = x[x >= tol]
    closest = float(nonunit.min()) if nonunit.size else float("inf")
    return FireCheck(n=n, n_unit=n_unit, bands=bands, closest_nonunit=closest)
