"""Exact-lattice simulated annealing + local-search toolkit on ML.

Promoted from runs/chase/n70/{chaselib.py, anneal.py} -- the engine that
crossed the n=70 280-attractor valley and tied the 281 record (RESULTS.md
2026-06-10). Everything lattice-internal is EXACT (integer invariants via
udg.mlgraph); float arithmetic is used ONLY for the min-separation
constraint (>= 0.2, the audit gate), where float64 is far more than
accurate enough.

Components:
- anneal_task / anneal_pool: integer-coords Metropolis annealing
  (relocate-to-1-2-step-candidate proposals, geometric cooling + reheats),
  pool-reseeded across processes with elite selection.
- steepest_climb: exhaustive relocation steepest ascent (every vertex x
  every candidate = 1-step and 2-step lattice neighborhoods), min-sep-safe.
- plateau_search: zero-gain random walk with tabu + opportunistic climbs.
- kick / restart_cycle: perturbation variants for random restarts.
- greedy_add / drop_worst / repair_minsep: completion utilities.
- beam_grow: small-n dense ML UDG library (factor library for sums).
- mink_sum_arr, rot_w1, rot_w3_eis: sum machinery with rotated factors.
- first_moves / two_ply_chunk / three_ply_first: bounded exhaustive
  multi-relocation workers (local-optimality proofs).

Exact arithmetic and the canonical form defer to udg.mlgraph (unit_mask,
float_xy, canon -- the single 12-element-point-group implementation).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from udg.mlgraph import MLConfig, canon, float_xy, save_csv as ml_save_csv, unit_mask
from udg.moser import UNIT_COEFFS

UC = np.array(UNIT_COEFFS, dtype=np.int64)  # (18, 4)
# 0.2 = the audit gate. Override (diagnostics only) via CHASE_MINSEP.
MINSEP = float(os.environ.get("CHASE_MINSEP", "0.2"))
NEG = -(10**9)


# ---------------------------------------------------------------------------
# exact edge counts + float min-sep on raw (n,4) int arrays
# ---------------------------------------------------------------------------

def edge_count(arr) -> int:
    return int(unit_mask(arr, arr).sum()) // 2


def degrees_arr(arr) -> np.ndarray:
    return unit_mask(arr, arr).sum(1).astype(np.int64)


def min_sep(arr) -> float:
    P = float_xy(arr)
    D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D2, 9.0)
    return float(np.sqrt(D2.min()))


def canon_t(pts) -> frozenset:
    """Translation-canonical form (cheap tabu key; rotations NOT merged --
    use udg.mlgraph.canon for class identity)."""
    m = min(pts)
    return frozenset((p[0] - m[0], p[1] - m[1], p[2] - m[2], p[3] - m[3]) for p in pts)


def as_tuples(arr) -> list[tuple]:
    return [tuple(int(x) for x in row) for row in arr]


def _row_units(c, arr) -> np.ndarray:
    """Exact unit mask of point c (4-tuple) vs arr (n,4)."""
    return unit_mask(np.asarray(c, dtype=np.int64)[None, :], arr)[0]


def _row_d2(c, arr) -> np.ndarray:
    """Float squared distances of point c (4-tuple) vs arr (n,4)."""
    F = float_xy(np.vstack([np.asarray(c, dtype=np.int64)[None, :], arr]))
    d = F[1:] - F[0]
    return (d * d).sum(1)


# ---------------------------------------------------------------------------
# candidate generation
# ---------------------------------------------------------------------------

def candidates(arr, two_step=True) -> np.ndarray:
    """All lattice positions 1 (and optionally 2) unit-steps from cfg points,
    minus the points themselves. Sorted (deterministic)."""
    c1 = (arr[:, None, :] + UC[None, :, :]).reshape(-1, 4)
    cs = set(map(tuple, c1.tolist()))
    if two_step:
        c1a = np.array(sorted(cs), dtype=np.int64)
        c2 = (c1a[:, None, :] + UC[None, :, :]).reshape(-1, 4)
        cs |= set(map(tuple, c2.tolist()))
    cs -= set(map(tuple, arr.tolist()))
    if not cs:
        return np.zeros((0, 4), dtype=np.int64)
    return np.array(sorted(cs), dtype=np.int64)


def _gain_tables(arr, two_step=True, minsep=MINSEP):
    """Returns (cand, G) where G[c,v] = exact edge gain of relocating vertex v
    to candidate c, with min-sep-inadmissible pairs set to NEG."""
    deg = degrees_arr(arr)
    cand = candidates(arr, two_step=two_step)
    if len(cand) == 0:
        return cand, np.full((0, len(arr)), NEG, dtype=np.int64)
    U = unit_mask(cand, arr).astype(np.int64)  # (m, n)
    canddeg = U.sum(1)
    G = canddeg[:, None] - U - deg[None, :]
    Pf = float_xy(arr)
    Cf = float_xy(cand)
    D2 = ((Cf[:, None, :] - Pf[None, :, :]) ** 2).sum(-1)
    near = (D2 < minsep * minsep).astype(np.int64)  # too close (includes coincide)
    ncount = near.sum(1)
    admiss = (ncount[:, None] - near) == 0  # all violations are with the vacated vertex
    return cand, np.where(admiss, G, NEG)


# ---------------------------------------------------------------------------
# steepest ascent + plateau walk
# ---------------------------------------------------------------------------

def steepest_climb(pts, two_step=True, max_moves=2000, minsep=MINSEP, rng=None):
    """Exhaustive-relocation steepest ascent to a fixpoint.

    Returns (pts, edges). Ties broken randomly if rng given, else first."""
    pts = [tuple(int(x) for x in p) for p in pts]
    for _ in range(max_moves):
        arr = np.array(pts, dtype=np.int64)
        cand, G = _gain_tables(arr, two_step=two_step, minsep=minsep)
        if G.size == 0:
            break
        g = int(G.max())
        if g <= 0:
            break
        if rng is None:
            ci, vi = np.unravel_index(int(np.argmax(G)), G.shape)
        else:
            picks = np.argwhere(G == g)
            ci, vi = picks[rng.randrange(len(picks))]
        pts[int(vi)] = tuple(int(x) for x in cand[int(ci)])
    arr = np.array(pts, dtype=np.int64)
    return pts, edge_count(arr)


def plateau_search(pts, rng, steps=60, tabu=None, two_step=True, minsep=MINSEP,
                   max_tabu=4000):
    """Random zero-gain walk with tabu; climbs whenever a positive move shows.

    Returns (best_pts, best_edges, pts, edges) - best seen and final state."""
    if tabu is None:
        tabu = set()
    pts = [tuple(int(x) for x in p) for p in pts]
    arr = np.array(pts, dtype=np.int64)
    e = edge_count(arr)
    best_pts, best_e = list(pts), e
    tabu.add(canon_t(pts))
    for _ in range(steps):
        arr = np.array(pts, dtype=np.int64)
        cand, G = _gain_tables(arr, two_step=two_step, minsep=minsep)
        if G.size == 0:
            break
        g = int(G.max())
        if g > 0:
            picks = np.argwhere(G == g)
            ci, vi = picks[rng.randrange(len(picks))]
            pts[int(vi)] = tuple(int(x) for x in cand[int(ci)])
            e += g
            if e > best_e:
                best_pts, best_e = list(pts), e
            tabu.add(canon_t(pts))
            continue
        zeros = np.argwhere(G == 0)
        if len(zeros) == 0:
            break
        moved = False
        for _try in range(8):
            ci, vi = zeros[rng.randrange(len(zeros))]
            old = pts[int(vi)]
            pts[int(vi)] = tuple(int(x) for x in cand[int(ci)])
            ck = canon_t(pts)
            if ck in tabu:
                pts[int(vi)] = old
                continue
            tabu.add(ck)
            moved = True
            break
        if not moved:
            break
        if len(tabu) > max_tabu:
            tabu.clear()
    return best_pts, best_e, pts, e


# ---------------------------------------------------------------------------
# completion / repair / kicks
# ---------------------------------------------------------------------------

def greedy_add(pts, k=1, rng=None, two_step=True, minsep=MINSEP):
    """Add k points, each the max-exact-gain admissible candidate."""
    pts = [tuple(int(x) for x in p) for p in pts]
    for _ in range(k):
        arr = np.array(pts, dtype=np.int64)
        cand = candidates(arr, two_step=two_step)
        if len(cand) == 0:
            raise RuntimeError("no candidates")
        U = unit_mask(cand, arr)
        gains = U.sum(1).astype(np.int64)
        Pf = float_xy(arr)
        Cf = float_xy(cand)
        D2 = ((Cf[:, None, :] - Pf[None, :, :]) ** 2).sum(-1)
        ok = (D2 >= minsep * minsep).all(1)
        gains = np.where(ok, gains, NEG)
        g = int(gains.max())
        if g <= NEG // 2:
            raise RuntimeError("no admissible candidate")
        picks = np.flatnonzero(gains == g)
        ci = picks[0] if rng is None else picks[rng.randrange(len(picks))]
        pts.append(tuple(int(x) for x in cand[int(ci)]))
    return pts


def drop_worst(pts, k=1, rng=None):
    """Drop k points of minimum exact degree (random tie-break with rng)."""
    pts = [tuple(int(x) for x in p) for p in pts]
    for _ in range(k):
        arr = np.array(pts, dtype=np.int64)
        deg = degrees_arr(arr)
        m = int(deg.min())
        picks = np.flatnonzero(deg == m)
        i = picks[0] if rng is None else picks[rng.randrange(len(picks))]
        pts.pop(int(i))
    return pts


def repair_minsep(pts, minsep=MINSEP):
    """Remove the lower-degree point of each < minsep pair until clean."""
    pts = [tuple(int(x) for x in p) for p in pts]
    while True:
        arr = np.array(pts, dtype=np.int64)
        P = float_xy(arr)
        D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        np.fill_diagonal(D2, 9.0)
        ii, jj = np.nonzero(D2 < minsep * minsep)
        if len(ii) == 0:
            return pts
        deg = degrees_arr(arr)
        i, j = int(ii[0]), int(jj[0])
        drop = i if deg[i] <= deg[j] else j
        pts.pop(drop)


def kick(pts, rng, minsep=MINSEP):
    """Random perturbation: relocate / drop+readd, biased to low degree."""
    pts = [tuple(int(x) for x in p) for p in pts]
    variant = rng.randrange(3)
    n = len(pts)
    if variant == 0:
        # relocate k random low-degree-biased vertices to random candidates
        k = rng.choice([2, 2, 3, 3, 4])
        arr = np.array(pts, dtype=np.int64)
        deg = degrees_arr(arr)
        order = sorted(range(n), key=lambda i: (deg[i] + rng.random() * 4.0))
        victims = order[:k]
        for vi in victims:
            arr = np.array(pts, dtype=np.int64)
            cand = candidates(arr, two_step=True)
            Pf = float_xy(np.delete(arr, vi, axis=0))
            Cf = float_xy(cand)
            D2 = ((Cf[:, None, :] - Pf[None, :, :]) ** 2).sum(-1)
            ok = np.flatnonzero((D2 >= minsep * minsep).all(1))
            if len(ok) == 0:
                continue
            ci = ok[rng.randrange(len(ok))]
            pts[vi] = tuple(int(x) for x in cand[int(ci)])
    elif variant == 1:
        k = rng.choice([2, 3, 3, 4, 5])
        pts = drop_worst(pts, k=k, rng=rng)
        pts = greedy_add(pts, k=k, rng=rng)
    else:
        k = rng.choice([2, 3, 4])
        for _ in range(k):
            pts.pop(rng.randrange(len(pts)))
        pts = greedy_add(pts, k=k, rng=rng)
    return pts


# ---------------------------------------------------------------------------
# two-/three-ply exhaustive search: first move with bounded loss, best response
# ---------------------------------------------------------------------------

def first_moves(pts, min_gain=-2, two_step=True, minsep=MINSEP):
    """All admissible (gain, ci, vi) relocations with gain >= min_gain."""
    arr = np.array(pts, dtype=np.int64)
    cand, G = _gain_tables(arr, two_step=two_step, minsep=minsep)
    out = []
    if G.size:
        ii, jj = np.nonzero(G >= min_gain)
        for ci, vi in zip(ii.tolist(), jj.tolist()):
            out.append((int(G[ci, vi]), ci, vi))
    return cand, out


def two_ply_chunk(payload):
    """Worker: evaluate a chunk of first moves; report any total gain > 0.

    payload = (pts, moves, cand_rows, two_step, minsep) where moves is a list
    of (g1, cand_row_idx, vi) and cand_rows the corresponding candidate
    coordinate rows (list of tuples). Returns list of
    (total_gain, new_pts) for improving two-move sequences, plus the best
    (total_gain, new_pts) even if not improving.
    """
    pts0, moves, cand_rows, two_step, minsep = payload
    best_total = None
    improving = []
    for (g1, ci, vi) in moves:
        pts = list(pts0)
        pts[vi] = cand_rows[ci]
        arr = np.array(pts, dtype=np.int64)
        cand2, G2 = _gain_tables(arr, two_step=two_step, minsep=minsep)
        if G2.size == 0:
            continue
        g2 = int(G2.max())
        total = g1 + g2
        ci2, vi2 = np.unravel_index(int(np.argmax(G2)), G2.shape)
        pts[int(vi2)] = tuple(int(x) for x in cand2[int(ci2)])
        if total > 0:
            improving.append((total, list(pts)))
        if best_total is None or total > best_total[0]:
            best_total = (total, list(pts))
    return improving, best_total


def three_ply_first(payload):
    """Worker: full depth-3 search under one first move.

    payload = (pts0, (g1, ci, vi), cand_rows, g2_min, total2_min, minsep).
    Enumerates second moves with gain >= g2_min and g1+g2 >= total2_min,
    then takes the best third response. Complete at depth 3 given the
    first move (steepest final move). Returns (improving, best_total) where
    improving = [(total, new_pts)] with total > 0.
    """
    pts0, (g1, ci, vi), cand_rows, g2_min, total2_min, minsep = payload
    start_key = canon_t(pts0)
    pts1 = list(pts0)
    pts1[vi] = cand_rows[ci]
    arr1 = np.array(pts1, dtype=np.int64)
    cand2, G2 = _gain_tables(arr1, two_step=True, minsep=minsep)
    improving = []
    best_total = None
    if G2.size == 0:
        return improving, best_total
    cand2_rows = [tuple(int(x) for x in r) for r in cand2]
    ii, jj = np.nonzero(G2 >= max(g2_min, total2_min - g1))
    for c2, v2 in zip(ii.tolist(), jj.tolist()):
        g2 = int(G2[c2, v2])
        pts2 = list(pts1)
        pts2[v2] = cand2_rows[c2]
        if canon_t(pts2) == start_key:
            continue  # undid the first move
        arr2 = np.array(pts2, dtype=np.int64)
        cand3, G3 = _gain_tables(arr2, two_step=True, minsep=minsep)
        if G3.size == 0:
            continue
        g3 = int(G3.max())
        total = g1 + g2 + g3
        c3, v3 = np.unravel_index(int(np.argmax(G3)), G3.shape)
        pts3 = list(pts2)
        pts3[int(v3)] = tuple(int(x) for x in cand3[int(c3)])
        if total > 0:
            improving.append((total, pts3))
        if best_total is None or total > best_total[0]:
            best_total = (total, pts3)
    return improving, best_total


# ---------------------------------------------------------------------------
# restart worker (used by drivers via multiprocessing)
# ---------------------------------------------------------------------------

def restart_cycle(payload):
    """One worker task: several kick -> climb -> plateau cycles from pts0.

    payload = (pts0, seed, cycles, plateau_steps). Returns (best_e, best_pts).
    """
    pts0, seed, cycles, plateau_steps = payload
    rng = random.Random(seed)
    best = [tuple(int(x) for x in p) for p in pts0]
    best_e = edge_count(np.array(best, dtype=np.int64))
    cur = list(best)
    tabu: set = set()
    for _ in range(cycles):
        try:
            kicked = kick(cur, rng)
            kicked, _ = steepest_climb(kicked, rng=rng)
            bp, be, fp, fe = plateau_search(
                kicked, rng, steps=plateau_steps, tabu=tabu
            )
        except RuntimeError:
            cur = list(best)
            continue
        if be > best_e:
            best, best_e = bp, be
        # diversify: half the time continue from the walk's end state
        cur = fp if rng.random() < 0.5 else list(best)
    return best_e, best


# ---------------------------------------------------------------------------
# beam grower (small-n factor library)
# ---------------------------------------------------------------------------

def beam_grow(n_max, beam=1200, per_parent=60, minsep=MINSEP, log=None):
    """Beam search for dense small ML UDGs. Returns {size: [(edges, pts), ...]}
    sorted desc by edges, canon_t-deduplicated, min-sep-clean."""
    cur = [(0, ((0, 0, 0, 0),))]
    out = {1: list(cur)}
    for size in range(2, n_max + 1):
        children: dict = {}
        for e, pts in cur:
            arr = np.array(pts, dtype=np.int64)
            cand = candidates(arr, two_step=False)
            U = unit_mask(cand, arr)
            gains = U.sum(1).astype(np.int64)
            Pf = float_xy(arr)
            Cf = float_xy(cand)
            D2 = ((Cf[:, None, :] - Pf[None, :, :]) ** 2).sum(-1)
            ok = (D2 >= minsep * minsep).all(1)
            gains = np.where(ok, gains, NEG)
            order = np.argsort(-gains)[:per_parent]
            for ci in order:
                g = int(gains[ci])
                if g < 1:
                    break
                child = pts + (tuple(int(x) for x in cand[int(ci)]),)
                ck = canon_t(child)
                ce = e + g
                prev = children.get(ck)
                if prev is None or prev[0] < ce:
                    children[ck] = (ce, child)
        cur = sorted(children.values(), key=lambda t: -t[0])[:beam]
        out[size] = cur
        if log:
            log(f"beam size {size}: best {cur[0][0]} edges, kept {len(cur)}")
    return out


# ---------------------------------------------------------------------------
# Minkowski machinery
# ---------------------------------------------------------------------------

def rot_w1(pts):
    """Multiply a point list by w1 (ML-preserving rotation by 60 deg)."""
    return [(-b, a + b, -d, c + d) for (a, b, c, d) in pts]


def rot_w3_eis(pts):
    """Multiply Eisenstein-only points (a,b,0,0) by w3. Raises otherwise."""
    out = []
    for (a, b, c, d) in pts:
        if c or d:
            raise ValueError("rot_w3_eis needs Eisenstein-only points")
        out.append((0, 0, a, b))
    return out


def mink_sum_arr(ptsA, ptsB):
    """Exact deduplicated Minkowski sum as a sorted (m,4) int64 array."""
    A = np.array(ptsA, dtype=np.int64)
    B = np.array(ptsB, dtype=np.int64)
    S = (A[:, None, :] + B[None, :, :]).reshape(-1, 4)
    ss = set(map(tuple, S.tolist()))
    return np.array(sorted(ss), dtype=np.int64)


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(pts, outdir, name):
    """Write exact-coords JSON + float CSV; returns (json_path, csv_path)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    arr = np.array(pts, dtype=np.int64)
    e = edge_count(arr)
    jpath = outdir / f"{name}.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(
            {"n": len(pts), "exact_edges": e, "min_sep": min_sep(arr),
             "coords": [[int(x) for x in p] for p in pts]},
            f,
        )
        f.write("\n")
    cpath = outdir / f"{name}.csv"
    ml_save_csv(MLConfig(pts), cpath)
    return str(jpath), str(cpath)


# ---------------------------------------------------------------------------
# the annealer
# ---------------------------------------------------------------------------

def anneal_task(payload):
    """One simulated-annealing run (a worker task; top-level for pickling).

    payload = (pts0, seed, steps, t0, t1, reheats). State = n integer ML
    4-tuples. Proposal = relocate a random vertex to a random 1-2-step
    candidate (min-sep-admissible). Exact integer edge deltas; Metropolis
    acceptance exp(dE/T) with geometric cooling, each reheat phase restarts
    from the best at 0.7x the previous T0. The best state is polished
    (steepest_climb + plateau_search) before reporting.

    Returns (best_e, best_pts).
    """
    (pts0, seed, steps, t0, t1, reheats) = payload
    rng = random.Random(seed)
    pts = [tuple(int(x) for x in p) for p in pts0]
    n = len(pts)
    arr = np.array(pts, dtype=np.int64)
    e = edge_count(arr)
    best_e, best_pts = e, list(pts)

    for phase in range(reheats + 1):
        T = t0 * (0.7**phase)
        decay = (t1 / T) ** (1.0 / steps)
        for _it in range(steps):
            T *= decay
            v = rng.randrange(n)
            # candidate: random point +- 1 or 2 unit steps
            base = pts[rng.randrange(n)]
            u1 = UC[rng.randrange(18)]
            c = (int(base[0] + u1[0]), int(base[1] + u1[1]),
                 int(base[2] + u1[2]), int(base[3] + u1[3]))
            if rng.random() < 0.5:
                u2 = UC[rng.randrange(18)]
                c = (int(c[0] + u2[0]), int(c[1] + u2[1]),
                     int(c[2] + u2[2]), int(c[3] + u2[3]))
            if c == pts[v]:
                continue
            # distinctness + min-sep vs others
            mask = np.ones(n, dtype=bool)
            mask[v] = False
            rest = arr[mask]
            d2 = _row_d2(c, rest)
            if d2.min() < MINSEP * MINSEP:
                continue
            gain = int(_row_units(c, rest).sum()) - int(_row_units(pts[v], rest).sum())
            if gain >= 0 or rng.random() < math.exp(gain / T):
                pts[v] = c
                arr[v] = c
                e += gain
                if e > best_e:
                    best_e, best_pts = e, list(pts)
        # restart each phase from the best
        pts = list(best_pts)
        arr = np.array(pts, dtype=np.int64)
        e = best_e

    # polish the best
    cp, ce = steepest_climb(list(best_pts), rng=rng)
    bp, be, _, _ = plateau_search(cp, rng, steps=100)
    if be > best_e:
        best_e, best_pts = be, bp
    return best_e, best_pts


def anneal_pool(
    starts,
    *,
    minutes: float = 45.0,
    procs: int = 6,
    steps: int = 60_000,
    t0: float = 1.0,
    t1: float = 0.05,
    reheats: int = 2,
    seed: int = 4242,
    out: str | os.PathLike | None = None,
    name_prefix: str = "anneal",
    target: int | None = None,
    log=None,
) -> tuple[int, list[tuple], dict[int, int]]:
    """Pool-reseeded parallel annealing (the n=70 281 recipe).

    starts: iterable of point lists (each a list of integer 4-tuples). The
    pool is keyed by the canonical form (udg.mlgraph.canon, 12 ML motions +
    translation); each generation launches `procs` anneal_task's seeded from
    a random member of the top-10 elite with jittered t0. New classes at
    >= best-1 join the pool; the pool is pruned to the top 40 when it
    exceeds 60. Stops at the wall-clock budget, or as soon as best >= target
    (checked between generations).

    Determinism: per-generation task seeds are seed + gen*50021 + i, so each
    completed task is reproducible; the wall clock only decides how many
    generations run.

    Returns (best_e, best_pts, hist) where hist counts task outcomes by
    edge count. If `out` is given, every new best/class is checkpointed
    there as JSON + CSV.
    """
    _log = log if log is not None else (lambda m: print(m, flush=True))
    pool: dict[bytes, tuple[int, list[tuple]]] = {}
    for pts in starts:
        pts = [tuple(int(x) for x in p) for p in pts]
        e = edge_count(np.array(pts, dtype=np.int64))
        pool[canon(np.array(pts, dtype=np.int64))] = (e, pts)
    if not pool:
        raise ValueError("anneal_pool needs at least one start config")
    best_e, best_pts = max(pool.values(), key=lambda t: t[0])
    best_pts = list(best_pts)
    _log(f"anneal pool: {len(pool)} classes, best {best_e}")

    rng = random.Random(seed)
    t_end = time.time() + minutes * 60
    gen = 0
    n_tasks = 0
    hist: dict[int, int] = {}
    with ProcessPoolExecutor(max_workers=procs) as ex:
        while time.time() < t_end and (target is None or best_e < target):
            gen += 1
            elite = sorted(pool.values(), key=lambda t: -t[0])[:10]
            payloads = [
                (
                    elite[rng.randrange(len(elite))][1],
                    seed + gen * 50021 + i,
                    steps,
                    t0 * (0.7 + 0.6 * rng.random()),
                    t1,
                    reheats,
                )
                for i in range(procs)
            ]
            for e, pts in ex.map(anneal_task, payloads):
                n_tasks += 1
                hist[e] = hist.get(e, 0) + 1
                if e >= best_e - 1:
                    ck = canon(np.array(pts, dtype=np.int64))
                    if ck not in pool or pool[ck][0] < e:
                        pool[ck] = (e, pts)
                        if e >= best_e:
                            new = e > best_e
                            if out is not None:
                                j, _c = save_checkpoint(
                                    pts, out, f"{name_prefix}_{e}_{len(pool)}"
                                )
                                _log(
                                    f"gen {gen}: "
                                    f"{'*** NEW BEST' if new else 'new class at'} "
                                    f"{e} -> {j}"
                                )
                            elif new:
                                _log(f"gen {gen}: *** NEW BEST {e}")
                            best_e = max(best_e, e)
                            best_pts = list(pts)
            if len(pool) > 60:
                pool = dict(sorted(pool.items(), key=lambda kv: -kv[1][0])[:40])
            if gen % 5 == 0:
                _log(
                    f"gen {gen}: tasks={n_tasks} best={best_e} pool={len(pool)} "
                    f"hist_top={dict(sorted(hist.items(), reverse=True)[:4])}"
                )
    _log(f"anneal done: best {best_e} after {n_tasks} tasks")
    # the pool always holds the best class (added on arrival, never pruned
    # out of the top-40 by edge count) -- recover from it authoritatively
    best_e, best_pts = max(pool.values(), key=lambda t: t[0])
    return best_e, list(best_pts), hist
