#!/usr/bin/env python
"""Exact discrete record chase at n=50 on the Moser lattice.

Move classes (all exact, integer arithmetic):
  1-move: relocate one vertex to any position at unit distance from >= 1
          other point (complete: any useful position is p+u for p in rest).
  2-move: relocate two vertices simultaneously (exhaustive over candidate
          pairs; the candidate set is augmented with the current positions
          so a vertex can take a slot another vertex vacates).
  plateau: sideways (delta = 0) 1-moves with a tabu set of canonical forms,
          punctuated by strict-gain probes (steepest 1-move + 2-move).
  kick:   randomly relocate k vertices, then re-climb.

Checkpoints any config with > BASELINE edges to runs/chase/n50/.
Worker-parallel via multiprocessing (fork) over seeds.

Usage:
  uv run python scripts/chase50_climb.py --seeds-json A.json B.json \
      [--minutes 30] [--procs 6] [--target 184] [--out runs/chase/n50]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from udg.mlgraph import (
    MLConfig,
    _as_coeff_array,
    _unit_mask,
    exact_edge_count,
    save_csv,
)
from udg.moser import UNIT_COEFFS

UC = np.array(UNIT_COEFFS, dtype=np.int64)  # (18, 4)


# ---------------------------------------------------------------------------
# canonical forms (translation + the 6 w1-rotations that preserve ML)
# ---------------------------------------------------------------------------

def rot_w1(p):
    a, b, c, d = p
    return (-b, a + b, -d, c + d)


def canon_full(points) -> tuple:
    best = None
    cur = [tuple(p) for p in points]
    for k in range(6):
        if k:
            cur = [rot_w1(p) for p in cur]
        pts = sorted(cur)
        t = pts[0]
        form = tuple(
            sorted((p[0] - t[0], p[1] - t[1], p[2] - t[2], p[3] - t[3]) for p in pts)
        )
        if best is None or form < best:
            best = form
    return best


# ---------------------------------------------------------------------------
# fast exact state: numpy adjacency + candidate machinery
# ---------------------------------------------------------------------------

class State:
    """Mutable exact lattice config with cached adjacency."""

    def __init__(self, points):
        self.pts: list[tuple] = [tuple(int(x) for x in p) for p in points]
        self._rebuild()

    def _rebuild(self):
        self.arr = _as_coeff_array(self.pts)
        self.adj = _unit_mask(self.arr, self.arr)
        np.fill_diagonal(self.adj, False)
        self.deg = self.adj.sum(axis=1)
        self.edges = int(self.deg.sum()) // 2

    def copy(self) -> "State":
        s = State.__new__(State)
        s.pts = list(self.pts)
        s.arr = self.arr.copy()
        s.adj = self.adj.copy()
        s.deg = self.deg.copy()
        s.edges = self.edges
        return s

    def set_vertex(self, v: int, c: tuple):
        self.pts[v] = tuple(int(x) for x in c)
        self._rebuild()  # n=50: trivial cost

    def candidates(self) -> np.ndarray:
        """All lattice positions at unit distance from >=1 point, EXCLUDING
        current points; sorted, (C,4) int64."""
        cand = (self.arr[:, None, :] + UC[None, :, :]).reshape(-1, 4)
        cur = {tuple(int(x) for x in p) for p in self.pts}
        out = sorted({tuple(int(x) for x in row) for row in cand} - cur)
        return np.array(out, dtype=np.int64)

    def verify(self):
        m = exact_edge_count(MLConfig(self.pts))
        assert m == self.edges and len(set(self.pts)) == len(self.pts), (
            m,
            self.edges,
            len(set(self.pts)),
        )
        return m


def move_deltas(st: State):
    """(deltas, cand) where deltas[v, c] = exact edge change relocating v->c."""
    cand = st.candidates()
    M = _unit_mask(cand, st.arr)  # (C, N)
    s = M.sum(axis=1)  # edges from c to ALL current points
    # delta(v, c) = (s[c] - M[c, v]) - deg[v]
    deltas = (s[None, :] - M.T.astype(np.int64)) - st.deg[:, None]
    return deltas, cand


def steepest_1move(st: State) -> bool:
    """Apply the best strict-gain 1-move. True if applied."""
    deltas, cand = move_deltas(st)
    k = int(np.argmax(deltas))
    v, ci = divmod(k, deltas.shape[1])
    if deltas[v, ci] <= 0:
        return False
    expect = st.edges + int(deltas[v, ci])
    st.set_vertex(v, tuple(int(x) for x in cand[ci]))
    assert st.edges == expect, (st.edges, expect)
    return True


def best_2move(st: State):
    """Exhaustive simultaneous 2-relocation. Returns (gain, v1, v2, c1, c2)
    of the best strict-gain move, or None."""
    cand = st.candidates()
    cur = st.arr.astype(np.int64)
    allc = np.concatenate([cand, cur])  # current positions may be re-used
    C = len(allc)
    n = len(st.pts)
    M = _unit_mask(allc, st.arr)  # (C, N)
    S = M.sum(axis=1).astype(np.int64)
    U = _unit_mask(allc, allc)  # (C, C) candidate-candidate adjacency
    cur_idx = np.arange(len(cand), C)  # index of current position of vertex i
    best = None
    Mi = M.astype(np.int64)
    adj = st.adj
    deg = st.deg
    for v1 in range(n - 1):
        s1 = S - Mi[:, v1]
        for v2 in range(v1 + 1, n):
            base = int(deg[v1] + deg[v2] - (1 if adj[v1, v2] else 0))
            s12 = s1 - Mi[:, v2]  # (C,) edges from c to rest12
            D = s12[:, None] + s12[None, :] + U.astype(np.int64)
            # invalid targets: any position equal to a rest12 point
            bad = np.zeros(C, dtype=bool)
            bad[cur_idx] = True
            bad[cur_idx[v1]] = False
            bad[cur_idx[v2]] = False
            D[bad, :] = -99
            D[:, bad] = -99
            np.fill_diagonal(D, -99)  # c1 != c2
            k = int(np.argmax(D))
            i1, i2 = divmod(k, C)
            gain = int(D[i1, i2]) - base
            if gain > 0 and (best is None or gain > best[0]):
                best = (
                    gain,
                    v1,
                    v2,
                    tuple(int(x) for x in allc[i1]),
                    tuple(int(x) for x in allc[i2]),
                )
    return best


def apply_2move(st: State, mv):
    gain, v1, v2, c1, c2 = mv
    expect = st.edges + gain
    st.pts[v1] = c1
    st.pts[v2] = c2
    st._rebuild()
    assert st.edges == expect and len(set(st.pts)) == len(st.pts), (
        st.edges,
        expect,
    )


def climb(st: State, use2: bool = True) -> State:
    """Steepest 1-moves to fixpoint, then 2-move; repeat until dry."""
    while True:
        while steepest_1move(st):
            pass
        if not use2:
            return st
        mv = best_2move(st)
        if mv is None:
            return st
        apply_2move(st, mv)


# ---------------------------------------------------------------------------
# plateau + kick search loop
# ---------------------------------------------------------------------------

def sideways_moves(st: State):
    deltas, cand = move_deltas(st)
    vs, cs = np.nonzero(deltas == 0)
    return [(int(v), tuple(int(x) for x in cand[c])) for v, c in zip(vs, cs)]


def grasp_readd(pts: list, k: int, rng: random.Random, slack: int = 1) -> State:
    """Re-add k points one at a time; each step enumerates ALL candidate
    positions exactly and picks uniformly among those within `slack` of the
    max gain (GRASP diversification)."""
    cur = list(pts)
    for _ in range(k):
        arr = _as_coeff_array(cur)
        cand = (arr[:, None, :] + UC[None, :, :]).reshape(-1, 4)
        cset = sorted({tuple(int(x) for x in r) for r in cand} - set(cur))
        CA = _as_coeff_array(cset)
        gains = _unit_mask(CA, arr).sum(axis=1)
        gmax = int(gains.max())
        pool = np.nonzero(gains >= gmax - slack)[0]
        cur.append(cset[int(rng.choice(pool))])
    return State(cur)


def lns_step(base: State, rng: random.Random) -> State:
    """Drop k vertices (worst-degree-biased or random), GRASP re-add."""
    k = rng.choice((2, 3, 3, 4, 4, 5, 6))
    n = len(base.pts)
    idx = list(range(n))
    if rng.random() < 0.6:
        # bias toward low exact degree (with noise)
        noise = np.array([rng.random() for _ in range(n)])
        order = sorted(idx, key=lambda i: (base.deg[i] + 2.5 * noise[i]))
        drop = set(order[:k])
    else:
        drop = set(rng.sample(idx, k))
    rest = [p for i, p in enumerate(base.pts) if i not in drop]
    return grasp_readd(rest, k, rng, slack=rng.choice((0, 1, 1)))


def checkpoint(st: State, outdir: Path, tag: str, log):
    st.verify()
    cfg = MLConfig(st.pts)
    name = f"udg50_{st.edges}edges_{tag}"
    with open(outdir / f"{name}.json", "w") as f:
        json.dump(
            {
                "n": len(st.pts),
                "exact_edges": st.edges,
                "method": tag,
                "coords": [list(p) for p in cfg.points],
            },
            f,
        )
    save_csv(cfg, outdir / f"{name}.csv")
    log(f"CHECKPOINT {name} ({st.edges} edges)")


def worker(args):
    (seed, pts0, minutes, target, baseline, outdir_s) = args
    rng = random.Random(seed)
    outdir = Path(outdir_s)
    t_end = time.time() + minutes * 60

    def log(msg):
        print(f"[w{seed}] {msg}", flush=True)

    best = State(pts0)
    best = climb(best)
    best_edges = best.edges
    log(f"start climb -> {best_edges}")
    if best_edges > baseline:
        checkpoint(best, outdir, f"w{seed}_start", log)

    tabu: set = set()
    pool: list[State] = [best.copy()]  # elite pool of distinct best-count configs
    pool_forms = {canon_full(best.pts)}
    it = 0
    n2move = 0
    while time.time() < t_end:
        it += 1
        base = pool[rng.randrange(len(pool))]
        cur = lns_step(base, rng)
        # cheap climb first (1-moves only)
        while steepest_1move(cur):
            pass
        # spend 2-move effort only near the frontier
        if cur.edges >= best_edges - 1:
            n2move += 1
            cur = climb(cur, use2=True)
        if cur.edges > best_edges:
            best = cur.copy()
            best_edges = cur.edges
            pool = [best.copy()]
            pool_forms = {canon_full(best.pts)}
            log(f"iter {it}: new best {best_edges}")
            if best_edges > baseline:
                checkpoint(best, outdir, f"w{seed}_it{it}", log)
            if best_edges >= target:
                log(f"TARGET {target} REACHED")
                break
        elif cur.edges == best_edges and len(pool) < 24:
            cf = canon_full(cur.pts)
            if cf not in pool_forms:
                pool_forms.add(cf)
                pool.append(cur.copy())
        if it % 500 == 0:
            log(
                f"iter {it}: best {best_edges}, pool {len(pool)}, "
                f"2move calls {n2move}"
            )
    # persist distinct best-count pool members for cross-seed analysis
    for i, st in enumerate(pool[:8]):
        with open(outdir / f"pool_w{seed}_{st.edges}e_{i}.json", "w") as f:
            json.dump({"n": len(st.pts), "exact_edges": st.edges,
                       "method": f"lns_pool_w{seed}",
                       "coords": [list(p) for p in st.pts]}, f)
    return seed, best_edges, [list(p) for p in best.pts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-json", nargs="+", required=True)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--target", type=int, default=184)
    ap.add_argument("--baseline", type=int, default=183)
    ap.add_argument("--out", default="runs/chase/n50")
    ap.add_argument("--restarts-per-seed", type=int, default=1)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = []
    seed = 0
    for path in args.seeds_json:
        cfg = MLConfig.from_json(path)
        for _ in range(args.restarts_per_seed):
            jobs.append(
                (
                    seed,
                    [list(p) for p in cfg.points],
                    args.minutes,
                    args.target,
                    args.baseline,
                    str(outdir),
                )
            )
            seed += 1

    import multiprocessing as mp

    with mp.get_context("fork").Pool(args.procs) as pool:
        results = pool.map(worker, jobs)
    print("=== summary ===", flush=True)
    overall = 0
    for s, e, _pts in sorted(results):
        print(f"seed {s}: best {e}", flush=True)
        overall = max(overall, e)
    print(f"overall best: {overall}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
