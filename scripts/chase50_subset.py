#!/usr/bin/env python
"""Heaviest-50-subgraph search over engineered Moser-lattice host sets.

Pick a host set H of lattice points (radius-r candidate balls around seed
configs, large Minkowski sums, unions); search for the 50-point subset
maximizing exact induced unit edges. Moves: steepest swap (one out, one
in; full delta matrix vectorized), sideways swaps with tabu, LNS restarts.

Everything exact: host adjacency built once with the int-invariant unit
test; induced counts are integer sums over the boolean adjacency.

Usage:
  uv run python scripts/chase50_subset.py --host ball2 --minutes 30 --procs 6
  hosts: ball1 | ball2 (around the 49 wheel-sum), sum7x8, sum4x13, sum5x11,
         union183 (all saved 183s + their radius-1 candidates)
"""
from __future__ import annotations

import argparse
import json
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
    minkowski,
    save_csv,
    wheel6,
)
from udg.moser import UNIT_COEFFS

sys.path.insert(0, str(Path(__file__).parent))
from chase50_climb import canon_full  # noqa: E402

UC = np.array(UNIT_COEFFS, dtype=np.int64)


def expand(points: set, steps: int) -> set:
    """points plus all lattice positions within `steps` unit-vector steps."""
    frontier = set(points)
    out = set(points)
    for _ in range(steps):
        arr = _as_coeff_array(sorted(frontier))
        cand = (arr[:, None, :] + UC[None, :, :]).reshape(-1, 4)
        new = {tuple(int(x) for x in r) for r in cand} - out
        out |= new
        frontier = new
    return out


def build_host(name: str) -> list[tuple]:
    s = minkowski(wheel6("w1"), wheel6("w3"))
    base = set(s.points)
    if name == "ball1":
        return sorted(expand(base, 1))
    if name == "ball2":
        return sorted(expand(base, 2))
    if name == "sum7x8":
        # 7-wheel x best 8-vertex piece (grown exactly): use rhombus+sums
        from chase50_minkowski import beam_grow
        lib = beam_grow(8, width=48)
        best8 = lib[8][0]
        return sorted(set(minkowski(wheel6("w1"), best8).points))
    if name == "sum4x13":
        from chase50_minkowski import beam_grow
        lib = beam_grow(13, width=48)
        return sorted(set(minkowski(MLConfig([(0,0,0,0),(1,0,0,0),(0,1,0,0),(1,1,0,0)]), lib[13][0]).points))
    if name == "sum5x11":
        from chase50_minkowski import beam_grow
        lib = beam_grow(11, width=48)
        return sorted(set(minkowski(lib[5][0], lib[11][0]).points))
    if name.endswith(".json"):
        import json as _json
        with open(name) as f:
            return sorted(tuple(int(x) for x in c) for c in _json.load(f))
    if name == "union183":
        pts: set = set()
        for p in Path("runs/chase/n50").glob("udg50_183edges_*.json"):
            with open(p) as f:
                d = json.load(f)
            pts |= {tuple(c) for c in d["coords"]}
        return sorted(expand(pts, 1))
    raise ValueError(name)


# ---------------------------------------------------------------------------
# subset search on a fixed host adjacency
# ---------------------------------------------------------------------------

def induced_edges(A: np.ndarray, sel: np.ndarray) -> int:
    idx = np.nonzero(sel)[0]
    return int(A[np.ix_(idx, idx)].sum()) // 2


def greedy_subset(A: np.ndarray, k: int, rng: random.Random, seed_idx=None) -> np.ndarray:
    """Grow a k-subset greedily by max degree-into-subset (random ties)."""
    h = A.shape[0]
    sel = np.zeros(h, dtype=bool)
    if seed_idx is not None and len(seed_idx):
        sel[seed_idx] = True
    else:
        sel[rng.randrange(h)] = True
    degS = A[:, sel].sum(axis=1).astype(np.int64)
    while int(sel.sum()) < k:
        degS_masked = np.where(sel, -1, degS)
        m = degS_masked.max()
        pool = np.nonzero(degS_masked >= max(m - 0, 0))[0]
        v = int(rng.choice(pool))
        sel[v] = True
        degS += A[:, v].astype(np.int64)
    return sel


def steepest_swaps(A: np.ndarray, sel: np.ndarray, E: int) -> tuple[np.ndarray, int]:
    """Apply best-improvement swaps until local optimum. Returns (sel, E)."""
    h = A.shape[0]
    degS = A[:, sel].sum(axis=1).astype(np.int64)
    while True:
        ins = np.nonzero(sel)[0]
        outs = np.nonzero(~sel)[0]
        # delta(i out of S, j into S) = degS[j] - degS[i] - A[i, j]
        D = degS[outs][None, :] - degS[ins][:, None] - A[np.ix_(ins, outs)]
        k = int(np.argmax(D))
        a, b = divmod(k, len(outs))
        if D[a, b] <= 0:
            return sel, E
        i, j = int(ins[a]), int(outs[b])
        sel[i] = False
        sel[j] = True
        E += int(D[a, b])
        degS = degS - A[:, i].astype(np.int64) + A[:, j].astype(np.int64)


def worker(args):
    (wid, host_pts, minutes, target, baseline, outdir_s, host_name, kk) = args
    rng = random.Random(1000 + wid)
    outdir = Path(outdir_s)
    t_end = time.time() + minutes * 60
    H = _as_coeff_array(host_pts)
    A = _unit_mask(H, H)
    np.fill_diagonal(A, False)
    h = len(host_pts)

    def log(msg):
        print(f"[{host_name}:w{wid}] {msg}", flush=True)

    best_E = -1
    best_sel = None
    it = 0
    while time.time() < t_end:
        it += 1
        if best_sel is None or rng.random() < 0.3:
            sel = greedy_subset(A, kk, rng)
        else:
            # LNS: drop 3-8 members of best, greedily refill
            sel = best_sel.copy()
            k = rng.choice((3, 4, 5, 6, 8))
            members = np.nonzero(sel)[0]
            for v in rng.sample(list(members), k):
                sel[v] = False
            degS = A[:, sel].sum(axis=1).astype(np.int64)
            while int(sel.sum()) < kk:
                dm = np.where(sel, -1, degS)
                m = dm.max()
                pool = np.nonzero(dm >= m - (1 if rng.random() < 0.4 else 0))[0]
                v = int(rng.choice(pool))
                sel[v] = True
                degS += A[:, v].astype(np.int64)
        E = induced_edges(A, sel)
        sel, E = steepest_swaps(A, sel, E)
        if E > best_E:
            best_E, best_sel = E, sel.copy()
            log(f"iter {it}: best {best_E}")
            if E > baseline:
                pts = [host_pts[i] for i in np.nonzero(sel)[0]]
                cfg = MLConfig(pts)
                m = exact_edge_count(cfg)
                assert m == E, (m, E)
                name = f"udg{kk}_{E}edges_subset_{host_name}_w{wid}_it{it}"
                with open(outdir / f"{name}.json", "w") as f:
                    json.dump({"n": kk, "exact_edges": E, "method": name,
                               "coords": [list(p) for p in cfg.points]}, f)
                save_csv(cfg, outdir / f"{name}.csv")
                log(f"CHECKPOINT {name}")
            if E >= target:
                log("TARGET REACHED")
                break
        if it % 500 == 0:
            log(f"iter {it}: best {best_E}")
    return wid, best_E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="ball1")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--target", type=int, default=184)
    ap.add_argument("--baseline", type=int, default=183)
    ap.add_argument("--out", default="runs/chase/n50")
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    host_pts = build_host(args.host)
    print(f"host {args.host}: {len(host_pts)} points", flush=True)
    H = _as_coeff_array(host_pts)
    A = _unit_mask(H, H)
    np.fill_diagonal(A, False)
    print(f"host edges: {int(A.sum())//2}, max degree {int(A.sum(axis=1).max())}",
          flush=True)

    jobs = [(w, host_pts, args.minutes, args.target, args.baseline,
             str(outdir), Path(args.host).stem if args.host.endswith(".json") else args.host, args.k) for w in range(args.procs)]
    import multiprocessing as mp
    with mp.get_context("fork").Pool(args.procs) as pool:
        results = pool.map(worker, jobs)
    print("=== summary ===", flush=True)
    for w, e in sorted(results):
        print(f"worker {w}: best {e}", flush=True)
    print("overall best:", max(e for _, e in results), flush=True)


if __name__ == "__main__":
    main()
