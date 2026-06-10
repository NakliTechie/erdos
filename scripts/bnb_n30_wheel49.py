#!/usr/bin/env python
"""Exact branch-and-bound: the densest 30-vertex subset of the 49-point wheel
sum has EXACTLY 92 edges.

Ambient: wheel49() = minkowski(wheel6('w1'), wheel6('w3')) -- 49 ML points,
180 exact unit edges (the Engel et al. n=49 record). This script PROVES, by
exhaustive include/exclude branch-and-bound over all C(49,30) subsets, that
no 30-vertex subset of this ambient carries more than 92 edges (the Table 2
target at n=30 is 93 -- i.e. the n=30 record is NOT a subset of the wheel
sum; the 93-edge config needs a different ambient).

Method:
  * vertices ordered by descending ambient degree; adjacency as 49-bit masks;
  * warm start: deterministic subset_ils (seed 0) supplies a 92-edge witness,
    so the search runs with best = 92 from node 1;
  * branch include/exclude on the next vertex; at each node with c chosen
    vertices (e edges among them) and m = 30 - c still to pick from the
    remaining suffix R, prune on the admissible degree-based upper bound

        ub = e + sum_top_m over v in R of deg_v(chosen)
               + min( m*(m-1)/2,  E(R),  floor(sum_top_m deg_v(R) / 2) )

    where deg_v(R) and E(R) (edges inside the suffix) are precomputed per
    suffix index. Every term relaxes the true completion, so ub >= the best
    completion through that node and pruning at ub <= best is exact.

Prints: max edges, a 92-edge witness subset, node count, wall time; asserts
max == 92. Run from the repo root:

    uv run python scripts/bnb_n30_wheel49.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from udg.subsetsearch import adjacency, edge_count, subset_ils, wheel49  # noqa: E402

K = 30
TARGET = 92


def main() -> int:
    amb_cfg = wheel49()
    amb = amb_cfg.as_array()
    N = len(amb)
    n_edges_amb = edge_count(amb)
    print(f"ambient: wheel49 = wheel6(w1) (+) wheel6(w3): {N} points, "
          f"{n_edges_amb} exact edges")
    assert (N, n_edges_amb) == (49, 180)

    # ---- warm start: deterministic 92-edge witness ----------------------
    m_warm, pts_warm = subset_ils(amb, K, seed=0, max_iters=400, target=TARGET)
    print(f"warm start (subset_ils seed=0): {m_warm} edges")
    assert m_warm == TARGET, f"warm start expected {TARGET}, got {m_warm}"

    # ---- order vertices by descending ambient degree, build bitmasks ----
    A = adjacency(amb)
    np.fill_diagonal(A, False)
    deg = A.sum(axis=1)
    order = sorted(range(N), key=lambda v: (-int(deg[v]), v))
    A = A[np.ix_(order, order)]
    adj = [int.from_bytes(np.packbits(A[v], bitorder="little").tobytes(), "little")
           for v in range(N)]

    # suffix tables: DEGR[i][v] = deg_v({i..N-1}); ERR[i] = edges inside {i..N-1}
    suffix = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix[i] = suffix[i + 1] | (1 << i)
    DEGR = [[(adj[v] & suffix[i]).bit_count() for v in range(N)] for i in range(N + 1)]
    ERR = [sum(DEGR[i][v] for v in range(i, N)) // 2 for i in range(N + 1)]

    best = m_warm
    best_mask = 0
    warm_set = {tuple(int(x) for x in p) for p in pts_warm}
    for v in range(N):
        if tuple(int(x) for x in amb[order[v]]) in warm_set:
            best_mask |= 1 << v
    assert best_mask.bit_count() == K

    nodes = 0
    sys.setrecursionlimit(10_000)

    def dfs(i: int, chosen: int, c: int, e: int) -> None:
        nonlocal nodes, best, best_mask
        nodes += 1
        if c == K:
            if e > best:
                best, best_mask = e, chosen
            return
        if N - i < K - c:
            return
        m = K - c
        degr_i = DEGR[i]
        deg_c = [(adj[v] & chosen).bit_count() for v in range(i, N)]
        top_c = sorted(deg_c, reverse=True)[:m]
        top_r = sorted((degr_i[v] for v in range(i, N)), reverse=True)[:m]
        ub = e + sum(top_c) + min(m * (m - 1) // 2, ERR[i], sum(top_r) // 2)
        if ub <= best:
            return
        # include vertex i, then exclude it
        dfs(i + 1, chosen | (1 << i), c + 1, e + deg_c[0])
        dfs(i + 1, chosen, c, e)

    t0 = time.time()
    dfs(0, 0, 0, 0)
    dt = time.time() - t0

    witness_idx = [v for v in range(N) if best_mask >> v & 1]
    witness_pts = amb[[order[v] for v in witness_idx]]
    e_check = edge_count(witness_pts)

    print(f"\nEXACT MAXIMUM over all C(49,30) subsets: {best} edges")
    print(f"witness check: edge_count(witness) = {e_check}")
    print(f"witness subset (30 ML lattice points of the wheel sum):")
    for p in sorted(map(tuple, witness_pts.tolist())):
        print(f"  {p}")
    print(f"nodes explored: {nodes}")
    print(f"wall time: {dt:.2f} s")

    assert best == TARGET, f"PROOF FAILED: max = {best} != {TARGET}"
    assert e_check == TARGET
    print(f"\nPROVED: the densest 30-vertex subset of the 49-point wheel sum "
          f"has exactly {TARGET} edges (Table 2 target at n=30 is 93; the "
          f"n=30 record is not a wheel-sum subset).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
