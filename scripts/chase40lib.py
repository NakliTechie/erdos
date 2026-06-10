"""Fast exact Moser-lattice local-search engine for the n=40 record chase.

Everything operates on numpy int64 arrays of shape (n, 4) — integer ML
coordinates over the basis (1, w1, w3, w1*w3). All edge tests are EXACT
(scaled-integer invariants, see udg.mlgraph): a pair is a unit edge iff
A^2+3B^2+11C^2+33D^2 == 144 and AD+BC == 0 for the integer invariants of
the coordinate difference. Coordinates here stay tiny (|x| <= ~50), far
inside the int64-exact bound (~3.8e7).

Symmetry group used for canonical forms (tabu keys): the 12 rigid motions
of ML = 6 rotations (multiplication by w1^k; w1^3 = -1) x optional
reflection z -> w3 * conj(z) (verified module automorphism:
w3*conj(1)=w3, w3*conj(w1)=w3-w1w3, w3*conj(w3)=1, w3*conj(w1w3)=1-w1),
plus translations (subtract lexicographically smallest row).
"""

from __future__ import annotations

import numpy as np

from udg.moser import UNIT_COEFFS

UC = np.array(UNIT_COEFFS, dtype=np.int64)  # (18, 4)

# ---------------------------------------------------------------------------
# exact edge machinery
# ---------------------------------------------------------------------------


def _inv(diff):
    da, db, dc, dd = diff[..., 0], diff[..., 1], diff[..., 2], diff[..., 3]
    A = 12 * da + 6 * db + 10 * dc + 5 * dd
    B = 6 * db + 5 * dd
    C = 2 * dc + dd
    D = -dd
    return A, B, C, D


def unit_mask(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(len(X), len(Y)) boolean: exact unit pairs."""
    diff = X[:, None, :].astype(np.int64) - Y[None, :, :].astype(np.int64)
    A, B, C, D = _inv(diff)
    return (A * A + 3 * B * B + 11 * C * C + 33 * D * D == 144) & (A * D + B * C == 0)


def edge_count(P: np.ndarray) -> int:
    return int(unit_mask(P, P).sum()) // 2


def degs(P: np.ndarray) -> np.ndarray:
    return unit_mask(P, P).sum(axis=1)


def edges_of(P: np.ndarray) -> list[tuple[int, int]]:
    M = unit_mask(P, P)
    ii, jj = np.nonzero(np.triu(M, 1))
    return list(zip(ii.tolist(), jj.tolist()))


# ---------------------------------------------------------------------------
# set utilities (rows as void views for fast membership)
# ---------------------------------------------------------------------------


def _rows_view(X: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.int64)
    return X.view([("", np.int64)] * 4).reshape(-1)


def setdiff_rows(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Rows of X not present in Y (X assumed unique)."""
    keep = ~np.isin(_rows_view(X), _rows_view(Y))
    return X[keep]


def unique_rows(X: np.ndarray) -> np.ndarray:
    return np.unique(X, axis=0)


def universe(P: np.ndarray, steps: int = 1) -> np.ndarray:
    """All lattice points within `steps` unit-vector hops of P, minus P."""
    U = P
    seen = P
    for _ in range(steps):
        U = unique_rows((U[:, None, :] + UC[None, :, :]).reshape(-1, 4))
        seen = unique_rows(np.vstack([seen, U]))
    return setdiff_rows(seen, P)


# ---------------------------------------------------------------------------
# float positions + min-separation guard (audit gate: min_sep >= 0.2)
# ---------------------------------------------------------------------------

_S3 = 3.0**0.5
_S11 = 11.0**0.5
_S33 = 33.0**0.5

MINSEP = 0.2  # audit pipeline hard floor
_MINSEP2 = (MINSEP + 1e-9) ** 2  # tiny cushion against float rounding


def floatxy(X: np.ndarray) -> np.ndarray:
    """(n, 2) float positions of integer ML coordinates."""
    X = np.asarray(X, dtype=np.int64)
    A = 12 * X[:, 0] + 6 * X[:, 1] + 10 * X[:, 2] + 5 * X[:, 3]
    B = 6 * X[:, 1] + 5 * X[:, 3]
    C = 2 * X[:, 2] + X[:, 3]
    D = -X[:, 3]
    x = (A + D * _S33) / 12.0
    y = (B * _S3 + C * _S11) / 12.0
    return np.column_stack([x, y])


def sep_viol_mask(U: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(u, n) boolean: candidate-to-point float distance^2 < minsep^2."""
    FU, FP = floatxy(U), floatxy(P)
    d2 = ((FU[:, None, :] - FP[None, :, :]) ** 2).sum(-1)
    return d2 < _MINSEP2


def min_sep(P: np.ndarray) -> float:
    F = floatxy(P)
    d2 = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return float(np.sqrt(d2.min()))


# ---------------------------------------------------------------------------
# canonical form under the 12-element point group x translation
# ---------------------------------------------------------------------------


def _rot1(P: np.ndarray) -> np.ndarray:
    """Multiplication by w1: (a,b,c,d) -> (-b, a+b, -d, c+d)."""
    a, b, c, d = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    return np.stack([-b, a + b, -d, c + d], axis=1)


def _refl(P: np.ndarray) -> np.ndarray:
    """z -> w3*conj(z): (a,b,c,d) -> (c+d, -d, a+b, -b)."""
    a, b, c, d = P[:, 0], P[:, 1], P[:, 2], P[:, 3]
    return np.stack([c + d, -d, a + b, -b], axis=1)


def canon(P: np.ndarray) -> bytes:
    """Canonical bytes key of the point set under the 12 motions + translation."""
    best = None
    Q = np.asarray(P, dtype=np.int64)
    for _ in range(6):
        for X in (Q, _refl(Q)):
            rows = sorted(map(tuple, X.tolist()))
            base = rows[0]
            key = bytes(
                np.array(
                    [[r[k] - base[k] for k in range(4)] for r in rows], dtype=np.int16
                ).tobytes()
            )
            if best is None or key < best:
                best = key
        Q = _rot1(Q)
    return best


# ---------------------------------------------------------------------------
# steepest-ascent single-vertex climber
# ---------------------------------------------------------------------------


def best_single_move(P: np.ndarray, U: np.ndarray, minsep: bool = True):
    """Best (delta, v, new_pos) relocating one vertex of P to a row of U.

    U must exclude rows of P. delta = (edges gained at new pos, excluding
    the vertex being moved) - (current degree of moved vertex). With
    minsep=True a move (c, v) is only legal when c is >= MINSEP away from
    every point of P other than v (audit-gate safety).
    """
    M = unit_mask(U, P)  # (u, n)
    g_all = M.sum(axis=1)
    D = degs(P)
    delta = (g_all[:, None] - M) - D[None, :]
    if minsep:
        V = sep_viol_mask(U, P)  # (u, n)
        nviol = V.sum(axis=1)
        illegal = (nviol[:, None] - V) > 0  # violations against P minus v
        delta = np.where(illegal, -(10**6), delta)
    c, v = np.unravel_index(int(np.argmax(delta)), delta.shape)
    return int(delta[c, v]), int(v), U[c].copy()


def climb(P: np.ndarray, steps: int = 2, max_iter: int = 10_000, minsep: bool = True):
    """Steepest-ascent single-vertex relocation to fixpoint. Returns (P, count).

    Tiered: exhausts the cheap 1-hop universe first; the s-hop universes
    (s <= steps) are only consulted when all smaller ones are dry, so the
    fixpoint is exhaustive w.r.t. the full `steps`-hop universe.
    """
    P = np.array(P, dtype=np.int64)
    count = edge_count(P)
    for _ in range(max_iter):
        moved = False
        for s in range(1, steps + 1):
            U = universe(P, s)
            d, v, c = best_single_move(P, U, minsep=minsep)
            if d > 0:
                P[v] = c
                count += d
                moved = True
                break
        if not moved:
            break
    return P, count


# ---------------------------------------------------------------------------
# exhaustive 2-vertex simultaneous relocation (with pruning)
# ---------------------------------------------------------------------------


def best_pair_move(
    P: np.ndarray, want: int | None = None, steps: int = 1, minsep: bool = True
):
    """Exhaustive search over relocating ANY two vertices simultaneously.

    For each unordered vertex pair (v1, v2): remove both; candidates = the
    `steps`-hop universe of the remaining 38 points; need
    g(c1) + g(c2) + unit(c1,c2) > deg-contribution removed. Candidate c2
    adjacent ONLY to c1 is covered by scanning c1's 18 unit neighbors.

    Returns (best_delta, v1, v2, c1, c2) or (0, None...) if no strict gain.
    `want`: stop early when a move with delta >= want is found.
    """
    n = len(P)
    Mfull = unit_mask(P, P)
    D = Mfull.sum(axis=1)
    best = (0, None, None, None, None)
    Pv = _rows_view(P)
    for v1 in range(n):
        for v2 in range(v1 + 1, n):
            removed = int(D[v1] + D[v2]) - int(Mfull[v1, v2])
            keep = [i for i in range(n) if i != v1 and i != v2]
            rest = P[keep]
            U = universe(rest, steps)
            # also forbid the old positions? no -- moving back is allowed
            # (identity move gives delta 0, filtered by strict >)
            if minsep:
                U = U[~sep_viol_mask(U, rest).any(axis=1)]
                if len(U) == 0:
                    continue
            G = unit_mask(U, rest)
            g = G.sum(axis=1)
            gmax = int(g.max()) if len(g) else 0
            need = removed + 1  # strict improvement
            thr = need - gmax - 1
            S = np.nonzero(g >= max(thr, 1))[0]
            if len(S) == 0:
                continue
            gS = g[S]
            US = U[S]
            # pairs within S
            MS = unit_mask(US, US)
            tot = gS[:, None] + gS[None, :] + MS
            if minsep and len(US) > 1:
                FS = floatxy(US)
                d2 = ((FS[:, None, :] - FS[None, :, :]) ** 2).sum(-1)
                tot = np.where(d2 < _MINSEP2, -1, tot)
            np.fill_diagonal(tot, -1)
            k = int(np.argmax(tot))
            i, j = divmod(k, len(S))
            if tot[i, j] >= need:
                delta = int(tot[i, j]) - removed
                if delta > best[0]:
                    best = (delta, v1, v2, US[i].copy(), US[j].copy())
                    if want is not None and delta >= want:
                        return best
            # c2 dangling off c1 (adjacent only to c1): c2 in c1 + UC
            top = S[np.argsort(-gS)[:24]]
            for ci in top:
                if g[ci] + 1 + gmax < need:
                    break
                nb = unique_rows(U[ci][None, :] + UC)
                nb = setdiff_rows(nb, np.vstack([rest, U[ci][None, :]]))
                if len(nb) == 0:
                    continue
                if minsep:
                    nb = nb[~sep_viol_mask(nb, np.vstack([rest, U[ci][None, :]])).any(axis=1)]
                    if len(nb) == 0:
                        continue
                gnb = unit_mask(nb, rest).sum(axis=1)
                tot2 = int(g[ci]) + 1 + gnb
                k2 = int(np.argmax(tot2))
                if tot2[k2] >= need:
                    delta = int(tot2[k2]) - removed
                    if delta > best[0]:
                        best = (delta, v1, v2, U[ci].copy(), nb[k2].copy())
                        if want is not None and delta >= want:
                            return best
    return best


# ---------------------------------------------------------------------------
# add / drop
# ---------------------------------------------------------------------------


def add_best(P: np.ndarray, steps: int = 1, minsep: bool = True):
    """Append the candidate with max exact gain. Returns (P', gain)."""
    U = universe(P, steps)
    if minsep:
        U = U[~sep_viol_mask(U, P).any(axis=1)]
    g = unit_mask(U, P).sum(axis=1)
    k = int(np.argmax(g))
    return np.vstack([P, U[k][None, :]]), int(g[k])


def add_best_choices(P: np.ndarray, steps: int = 1, top: int = 8, minsep: bool = True):
    """Top-`top` candidate additions as a list of (gain, pos)."""
    U = universe(P, steps)
    if minsep:
        U = U[~sep_viol_mask(U, P).any(axis=1)]
    g = unit_mask(U, P).sum(axis=1)
    idx = np.argsort(-g)[:top]
    return [(int(g[i]), U[i].copy()) for i in idx]


def drop_index(P: np.ndarray, i: int) -> np.ndarray:
    return np.delete(P, i, axis=0)


def drop_worst(P: np.ndarray):
    D = degs(P)
    i = int(np.argmin(D))
    return drop_index(P, i), int(D[i])


# ---------------------------------------------------------------------------
# combined polish + perturbation
# ---------------------------------------------------------------------------


def polish(
    P: np.ndarray,
    steps: int = 2,
    pair_threshold: int = 134,
    pair_steps: int = 1,
    minsep: bool = True,
):
    """Singles to fixpoint; if count >= pair_threshold also alternate with
    exhaustive pair moves until jointly dry. Returns (P, count)."""
    P, count = climb(P, steps=steps, minsep=minsep)
    while count >= pair_threshold:
        d, v1, v2, c1, c2 = best_pair_move(P, steps=pair_steps, minsep=minsep)
        if d <= 0:
            break
        P = P.copy()
        P[v1] = c1
        P[v2] = c2
        count += d
        P, count = climb(P, steps=steps, minsep=minsep)
    return P, count


def perturb(P: np.ndarray, k: int, rng: np.random.Generator, minsep: bool = True):
    """Relocate k random vertices to random legal 1-hop-universe positions."""
    P = np.array(P, dtype=np.int64)
    n = len(P)
    # bias towards low-degree vertices: sample without replacement weighted
    D = degs(P).astype(float)
    w = 1.0 / (1.0 + D - D.min())
    w /= w.sum()
    vs = rng.choice(n, size=min(k, n), replace=False, p=w)
    for v in vs:
        keep = np.delete(np.arange(n), v)
        rest = P[keep]
        U = universe(rest, 1)
        if minsep:
            U = U[~sep_viol_mask(U, rest).any(axis=1)]
        if len(U) == 0:
            continue
        P[v] = U[rng.integers(len(U))]
    return P


# ---------------------------------------------------------------------------
# IO glue to the certified-config world
# ---------------------------------------------------------------------------


def load_cert(path: str) -> np.ndarray:
    import json

    with open(path) as f:
        d = json.load(f)
    coords = d["coords"] if isinstance(d, dict) else d
    return np.array(coords, dtype=np.int64)


def to_mlconfig(P: np.ndarray):
    from udg.mlgraph import MLConfig

    return MLConfig([tuple(int(x) for x in row) for row in P])


def save_checkpoint(P: np.ndarray, stem: str, outdir: str = "runs/chase/n40"):
    """Write exact-coords JSON + float CSV for a config. Returns (json, csv)."""
    import json
    from pathlib import Path

    from udg.mlgraph import save_csv

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = to_mlconfig(P)
    jpath = out / f"{stem}.json"
    cpath = out / f"{stem}.csv"
    with open(jpath, "w") as f:
        json.dump(
            {
                "n": len(cfg),
                "exact_edges": edge_count(P),
                "coords": [list(map(int, p)) for p in cfg.points],
            },
            f,
            indent=1,
        )
        f.write("\n")
    save_csv(cfg, cpath)
    return str(jpath), str(cpath)
