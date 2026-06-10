"""Subset-in-closure + free-relocation ILS engines on the Moser lattice.

Promoted from runs/chase/n30/engine.py -- the engine that re-derived the
n=30 record (93 edges) in 0.4 s as a densest-30-subgraph of closure-1 of
the wheel49 Minkowski sum (RESULTS.md 2026-06-10). Two exact attacks, both
pure-integer (no floats anywhere in the search):

1. Subset mode: densest-k-subgraph local search inside a fixed ambient ML
   point set (adjacency precomputed with the exact int invariants);
   subset_ils adds the kick-restart loop around it.
2. Free mode: iterated local search over the whole lattice -- steepest-
   ascent single-vertex relocation to {p+u} candidates, plateau walks with
   tabu, random perturbations (k-relocate / drop+add).

Ambient builders: wheel49 (THE cross-sublattice wheel sum, built from
udg.mlgraph.wheel6 + minkowski), hex_patch / to_w3 cross-sublattice hex
sums, neighbor_closure. Exact arithmetic defers to udg.mlgraph.unit_mask;
class identity uses udg.mlgraph.canon (the single 12-motion canonical
form); the cheap translation-only tabu key here is canon_t.

NOTE on min-sep: subset configs inherit the ambient's geometry, and every
ambient built here is a set of exact lattice points >= the audit gate 0.2
apart only if its generators are -- wheel sums and closures of audited
configs satisfy this in practice, but every claimed result must still pass
the float three-audit (udg.audit) downstream.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from udg.mlgraph import (
    MLConfig,
    exact_edge_count,
    minkowski,
    save_csv,
    unit_mask,
    wheel6,
)
from udg.moser import UNIT_COEFFS

UC = np.array(UNIT_COEFFS, dtype=np.int64)  # (18, 4)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def canon_t(pts: np.ndarray) -> tuple:
    """Translation-canonical hashable form of an (n,4) int coordinate array
    (cheap tabu key; rotations NOT merged -- use udg.mlgraph.canon for class
    identity)."""
    m = pts.min(axis=0)
    q = pts - m
    return tuple(sorted(map(tuple, q.tolist())))


def edge_count(pts: np.ndarray) -> int:
    M = unit_mask(pts, pts)
    return int(M.sum()) // 2


def adjacency(pts: np.ndarray) -> np.ndarray:
    return unit_mask(pts, pts)


def save_checkpoint(pts: np.ndarray, outdir, tag: str) -> Path:
    """Write exact-coords JSON + float CSV under outdir; returns json path."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = MLConfig([tuple(int(x) for x in row) for row in pts])
    m = exact_edge_count(cfg)
    name = f"n{len(cfg)}_{m}edges_{tag}"
    jpath = outdir / f"{name}.json"
    with open(jpath, "w") as f:
        json.dump({"n": len(cfg), "exact_edges": m,
                   "coords": [list(p) for p in cfg.points]}, f)
    save_csv(cfg, outdir / f"{name}.csv")
    return jpath


# ---------------------------------------------------------------------------
# 1. subset mode: densest-k-subgraph in an ambient set
# ---------------------------------------------------------------------------

def subset_search(
    adj: np.ndarray,
    k: int,
    rng: random.Random,
    iters: int = 200_000,
    start: np.ndarray | None = None,
    plateau_cap: int = 400,
) -> tuple[int, np.ndarray]:
    """Local search for a dense k-subset of the ambient graph.

    State: index array S (k,). Moves: swap one in <-> one out, steepest with
    sideways moves allowed (bounded run), restart-free (caller restarts).
    Returns (best_edges, best_indices).
    """
    N = adj.shape[0]
    A = adj.astype(np.int32)
    if start is None:
        S = np.zeros(N, dtype=bool)
        S[rng.sample(range(N), k)] = True
    else:
        S = np.zeros(N, dtype=bool)
        S[start] = True
    degS = A[:, S].sum(axis=1)  # deg into S for ALL vertices
    cur = int(degS[S].sum()) // 2
    best, bestS = cur, S.copy()
    plateau = 0
    for _ in range(iters):
        ins = np.nonzero(S)[0]
        outs = np.nonzero(~S)[0]
        if outs.size == 0 or ins.size == 0:
            break  # k == N (or k == 0): subset is fixed, nothing to swap
        # delta[u_idx, v_idx] = degS[v] - degS[u] - adj[u, v]
        delta = degS[outs][None, :] - degS[ins][:, None] - A[np.ix_(ins, outs)]
        mx = int(delta.max())
        if mx < 0 or (mx == 0 and plateau >= plateau_cap):
            break
        cand = np.argwhere(delta == mx)
        ui, vi = cand[rng.randrange(len(cand))]
        u, v = int(ins[ui]), int(outs[vi])
        S[u] = False
        S[v] = True
        degS = degS - A[:, u] + A[:, v]
        cur += mx
        if mx == 0:
            plateau += 1
        else:
            plateau = 0
        if cur > best:
            best, bestS = cur, S.copy()
    return best, np.nonzero(bestS)[0]


def subset_ils(
    ambient: np.ndarray,
    k: int,
    seed: int = 0,
    *,
    minutes: float | None = None,
    max_iters: int | None = None,
    target: int | None = None,
    warm: np.ndarray | None = None,
    climb_iters: int = 8_000,
    plateau_cap: int = 300,
    on_new_best=None,
    should_stop=None,
    log=None,
) -> tuple[int, np.ndarray]:
    """ILS for the densest k-subset of `ambient` ((N,4) int64 lattice points).

    The kick-restart loop around subset_search (the n=30/93 recipe): climb,
    then repeatedly kick the current subset (replace r in {2,3,4,6} random
    members with random outsiders) and re-climb; accept >=, plus a 10%
    downhill restart acceptance. Stops at minutes / max_iters / target /
    should_stop() (whichever first; at least one climb always runs).

    Returns (best_edges, best_points) with best_points (k,4) int64 rows of
    `ambient`. Deterministic given (ambient, k, seed, max_iters) when no
    wall-clock budget is set.
    """
    adj = adjacency(ambient)
    np.fill_diagonal(adj, False)
    N = len(ambient)
    if k > N:
        raise ValueError(f"k={k} exceeds ambient size {N}")
    rng = random.Random(seed)
    t_end = None if minutes is None else time.time() + minutes * 60

    if warm is not None and len(warm) == k:
        S0 = np.asarray(warm)
    else:
        S0 = np.array(rng.sample(range(N), k))
    cur_m, cur_S = subset_search(adj, k, rng, iters=climb_iters,
                                 start=S0, plateau_cap=plateau_cap)
    best_m, best_S = cur_m, cur_S.copy()
    if on_new_best is not None:
        on_new_best(best_m, ambient[best_S])
    it = 0
    while True:
        if target is not None and best_m >= target:
            break
        if max_iters is not None and it >= max_iters:
            break
        if t_end is not None and time.time() >= t_end:
            break
        if should_stop is not None and should_stop():
            break
        it += 1
        # kick: replace r random members with r random outsiders
        S = cur_S.copy()
        outs = np.setdiff1d(np.arange(N), S)
        r = min(rng.choice((2, 3, 4, 6)), k, len(outs))
        if r == 0:
            break  # ambient == subset: nothing to swap
        drop = rng.sample(range(k), r)
        add = rng.sample(range(len(outs)), r)
        S[drop] = outs[add]
        m, S2 = subset_search(adj, k, rng, iters=climb_iters,
                              start=S, plateau_cap=plateau_cap)
        if m >= cur_m:
            cur_m, cur_S = m, S2
        elif rng.random() < 0.10:
            cur_m, cur_S = m, S2
        if m > best_m:
            best_m, best_S = m, S2.copy()
            if log:
                log(f"[subset_ils seed={seed}] it={it}: new best {best_m}")
            if on_new_best is not None:
                on_new_best(best_m, ambient[best_S])
    return best_m, ambient[best_S]


# ---------------------------------------------------------------------------
# 2. free mode: relocation ILS on the lattice
# ---------------------------------------------------------------------------

def climb(pts: np.ndarray, rng: random.Random, two_step_when_stuck: bool = True,
          sideways_cap: int = 60, seen: set | None = None) -> tuple[int, np.ndarray]:
    """Steepest-ascent relocation with bounded sideways plateau walking.

    pts: (n,4) int64. Returns (edges, pts). Exact integer arithmetic only.
    """
    pts = pts.copy()
    cur = edge_count(pts)
    sideways = 0
    if seen is None:
        seen = set()
    while True:
        # candidate pool: one unit step from every point
        cand = (pts[:, None, :] + UC[None, :, :]).reshape(-1, 4)
        cand = np.unique(cand, axis=0)
        # remove existing points
        ex = set(map(tuple, pts.tolist()))
        keep = np.array([tuple(row) not in ex for row in cand.tolist()])
        cand = cand[keep]
        gains = unit_mask(cand, pts).sum(axis=1).astype(np.int64)  # edges to ALL pts
        adj_cp = unit_mask(cand, pts)  # (m, n) bool
        deg = unit_mask(pts, pts).sum(axis=1).astype(np.int64)
        # delta for moving vertex v to candidate c:
        #   edges(c -> pts \ {v}) - deg(v) = gains[c] - adj_cp[c, v] - deg[v]
        delta = gains[:, None] - adj_cp.astype(np.int64) - deg[None, :]
        mx = int(delta.max())
        if mx > 0:
            ci, vi = map(int, np.argwhere(delta == mx)[rng.randrange((delta == mx).sum())])
            pts[vi] = cand[ci]
            cur += mx
            sideways = 0
            continue
        if mx == 0 and sideways < sideways_cap:
            zero = np.argwhere(delta == 0)
            rng.shuffle(zero_list := zero.tolist())
            moved = False
            for ci, vi in zero_list[:80]:
                trial = pts.copy()
                trial[vi] = cand[ci]
                c = canon_t(trial)
                if c not in seen:
                    seen.add(c)
                    pts = trial
                    sideways += 1
                    moved = True
                    break
            if moved:
                continue
        # optional: two-step candidates around the 3 worst-degree vertices
        if two_step_when_stuck:
            worst = np.argsort(deg)[:3]
            base = (pts[worst][:, None, :] + UC[None, :, :]).reshape(-1, 4)
            far = (base[:, None, :] + UC[None, :, :]).reshape(-1, 4)
            far = np.unique(far, axis=0)
            keep = np.array([tuple(row) not in ex for row in far.tolist()])
            far = far[keep]
            if len(far):
                g2 = unit_mask(far, pts).sum(axis=1).astype(np.int64)
                a2 = unit_mask(far, pts)
                d2 = g2[:, None] - a2[:, worst].astype(np.int64) - deg[worst][None, :]
                mx2 = int(d2.max())
                if mx2 > 0:
                    ci, wi = map(int, np.argwhere(d2 == mx2)[0])
                    pts[int(worst[wi])] = far[ci]
                    cur += mx2
                    sideways = 0
                    continue
        break
    return cur, pts


def perturb(pts: np.ndarray, rng: random.Random, k: int) -> np.ndarray:
    """Relocate k random vertices to random 1-step candidate positions."""
    pts = pts.copy()
    n = len(pts)
    for _ in range(k):
        v = rng.randrange(n)
        p = pts[rng.randrange(n)]
        u = UC[rng.randrange(len(UC))]
        q = p + u
        # avoid duplicate points
        if any((pts == q).all(axis=1)):
            continue
        pts[v] = q
    # dedup safety: if duplicates appeared, jitter them away by extra unit steps
    while True:
        uniq, idx = np.unique(pts, axis=0, return_index=True)
        if len(uniq) == len(pts):
            break
        dup = [i for i in range(len(pts)) if i not in set(idx.tolist())]
        for i in dup:
            pts[i] = pts[i] + UC[rng.randrange(len(UC))]
    return pts


def ils(
    start: np.ndarray,
    seed: int,
    minutes: float,
    *,
    kick=(2, 3, 4),
    target: int | None = None,
    on_new_best=None,
    log=None,
) -> tuple[int, np.ndarray]:
    """Free-mode ILS: perturb -> climb, accept >= (plateau drift) plus a 5%
    downhill restart acceptance. Returns (best_edges, best_pts)."""
    rng = random.Random(seed)
    seen: set = set()
    cur_m, cur = climb(start.astype(np.int64), rng, seen=seen)
    best_m, best = cur_m, cur.copy()
    if on_new_best is not None:
        on_new_best(best_m, best)
    t0 = time.time()
    it = 0
    while time.time() - t0 < minutes * 60:
        if target is not None and best_m >= target:
            break
        it += 1
        k = rng.choice(kick)
        trial = perturb(cur, rng, k)
        m, q = climb(trial, rng, seen=seen)
        if m >= cur_m:           # accept equal: drift across plateaus
            cur_m, cur = m, q
        elif rng.random() < 0.05:  # rare downhill restart acceptance
            cur_m, cur = m, q
        if cur_m > best_m:
            best_m, best = cur_m, cur.copy()
            if log:
                log(f"[ils seed={seed} it={it}] new best {best_m} "
                    f"({time.time()-t0:.0f}s)")
            if on_new_best is not None:
                on_new_best(best_m, best)
        if len(seen) > 2_000_000:
            seen.clear()
    return best_m, best


# ---------------------------------------------------------------------------
# ambient builders
# ---------------------------------------------------------------------------

def wheel49() -> MLConfig:
    """THE cross-sublattice wheel sum: minkowski(wheel6('w1'), wheel6('w3'))
    = 49 vertices, 180 exact edges (the Engel et al. n=49 record)."""
    return minkowski(wheel6("w1"), wheel6("w3"))


def hex_patch(r: int) -> MLConfig:
    """Hexagonal patch H(r) of Z[w1]: all a+b*w1 with |a|,|b|,|a+b| <= r."""
    pts = []
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            if abs(a + b) <= r:
                pts.append((a, b, 0, 0))
    return MLConfig(pts)


def to_w3(cfg: MLConfig) -> MLConfig:
    """Map the Z[w1]-sublattice config (a,b,0,0) -> (0,0,a,b)  (mult by w3)."""
    for p in cfg.points:
        if p[2] or p[3]:
            raise ValueError("to_w3 needs Eisenstein-only points (a,b,0,0)")
    return MLConfig([(0, 0, p[0], p[1]) for p in cfg.points])


def neighbor_closure(pts: np.ndarray, steps: int = 1) -> np.ndarray:
    """pts plus all lattice points within `steps` unit-vector steps."""
    cur = {tuple(r) for r in pts.tolist()}
    frontier = set(cur)
    for _ in range(steps):
        nxt = set()
        for p in frontier:
            for u in UNIT_COEFFS:
                q = (p[0] + u[0], p[1] + u[1], p[2] + u[2], p[3] + u[3])
                if q not in cur:
                    nxt.add(q)
        cur |= nxt
        frontier = nxt
    return np.array(sorted(cur), dtype=np.int64)
