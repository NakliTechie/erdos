#!/usr/bin/env python
"""Minkowski-sum construction campaign for the n=40 record chase.

1. Enumerate small dense ML UDG factor classes (canonical BFS, edges >=
   u(n) - slack at every level).
2. For factor pairs (A, B) and all integer-valid unit rotations of B,
   form S = A (+) rot(B); steer |S| to 40 (beam drop-worst / add-best),
   then climb (singles 2-hop) and pair-polish promising results.

Exact ML multiplication over the basis (1, w1, w3, w1*w3) with
w1^2 = w1 - 1 and w3^2 = (5/3) w3 - 1:
  (a1,b1,c1,d1) * (a2,b2,c2,d2) =
    e = a1 a2 - b1 b2 - c1 c2 + d1 d2
    f = a1 b2 + b1 a2 + b1 b2 - c1 d2 - d1 c2 - d1 d2
    g = a1 c2 + c1 a2 - b1 d2 - d1 b2 + 5/3 (c1 c2 - d1 d2)
    h = a1 d2 + d1 a2 + b1 c2 + c1 b2 + b1 d2 + d1 b2 + 5/3 (c1 d2 + d1 c2 + d1 d2)
Integer-valid iff the 5/3 terms are integers (mod-3 conditions).

Usage: uv run python scripts/chase40_minkowski.py --slack 1 --nmax 10
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from chase40lib import (  # noqa: E402
    UC,
    best_pair_move,
    canon,
    climb,
    degs,
    edge_count,
    min_sep,
    save_checkpoint,
    sep_viol_mask,
    setdiff_rows,
    unique_rows,
    unit_mask,
    universe,
)

U_OPT = {1: 0, 2: 1, 3: 3, 4: 5, 5: 7, 6: 9, 7: 12, 8: 14, 9: 18, 10: 20}


# ---------------------------------------------------------------------------
# exact ML multiplication (rational, with integrality check)
# ---------------------------------------------------------------------------


def ml_mul(p, q):
    """Exact product of two ML 4-tuples; None if not an ML point."""
    a1, b1, c1, d1 = (int(x) for x in p)
    a2, b2, c2, d2 = (int(x) for x in q)
    e = a1 * a2 - b1 * b2 - c1 * c2 + d1 * d2
    f = a1 * b2 + b1 * a2 + b1 * b2 - c1 * d2 - d1 * c2 - d1 * d2
    g3 = 3 * (a1 * c2 + c1 * a2 - b1 * d2 - d1 * b2) + 5 * (c1 * c2 - d1 * d2)
    h3 = 3 * (a1 * d2 + d1 * a2 + b1 * c2 + c1 * b2 + b1 * d2 + d1 * b2) + 5 * (
        c1 * d2 + d1 * c2 + d1 * d2
    )
    if g3 % 3 or h3 % 3:
        return None
    return (e, f, g3 // 3, h3 // 3)


def rot_config(B: np.ndarray, u) -> np.ndarray | None:
    """Rotate config B by unit u about its first point; None if non-integral."""
    base = B[0]
    out = []
    for row in B:
        d = tuple(int(x) for x in (row - base))
        r = ml_mul(d, u)
        if r is None:
            return None
        out.append(r)
    return np.array(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# factor enumeration
# ---------------------------------------------------------------------------


def enumerate_factors(nmax: int, slack: int, log=print) -> dict[int, list[np.ndarray]]:
    """Canonical classes of n-point ML UDGs with edges >= u(n)-slack, n<=nmax."""
    levels: dict[int, dict[bytes, np.ndarray]] = {}
    P1 = np.zeros((1, 4), dtype=np.int64)
    levels[1] = {canon(P1): P1}
    for j in range(2, nmax + 1):
        thr = U_OPT[j] - slack
        new: dict[bytes, np.ndarray] = {}
        for P in levels[j - 1].values():
            ec = edge_count(P)
            U = universe(P, 1)
            g = unit_mask(U, P).sum(axis=1)
            need = max(1, thr - ec)
            for c in U[g >= need]:
                child = np.vstack([P, c[None, :]])
                key = canon(child)
                if key not in new:
                    new[key] = child
        levels[j] = new
        cnts = sorted({edge_count(P) for P in new.values()}, reverse=True)
        log(f"  level {j}: {len(new)} classes (thr {thr}, counts {cnts[:5]})")
    return {j: list(v.values()) for j, v in levels.items()}


# ---------------------------------------------------------------------------
# sum -> steer to 40 -> polish
# ---------------------------------------------------------------------------


def mink_sum(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    S = (A[:, None, :] + B[None, :, :]).reshape(-1, 4)
    return unique_rows(S)


def steer_to_n(S: np.ndarray, n: int, beam: int = 6) -> list[np.ndarray]:
    """Beam of configs of size n derived from S by drop-worst / add-best."""
    front = {canon(S): S}
    while True:
        size = len(next(iter(front.values())))
        if size == n:
            break
        new: dict[bytes, np.ndarray] = {}
        for P in front.values():
            if size > n:
                D = degs(P)
                order = np.argsort(D)
                for i in order[: min(4, len(order))]:
                    child = np.delete(P, int(i), axis=0)
                    new[canon(child)] = child
            else:
                U = universe(P, 1)
                U = U[~sep_viol_mask(U, P).any(axis=1)]
                if len(U) == 0:
                    continue
                g = unit_mask(U, P).sum(axis=1)
                for i in np.argsort(-g)[: min(4, len(g))]:
                    child = np.vstack([P, U[int(i)][None, :]])
                    new[canon(child)] = child
        if not new:
            return []
        ranked = sorted(new.values(), key=edge_count, reverse=True)
        front = {canon(P): P for P in ranked[:beam]}
    return list(front.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slack", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--target", type=int, default=40)
    ap.add_argument("--size-lo", type=int, default=30)
    ap.add_argument("--size-hi", type=int, default=52)
    ap.add_argument("--prod-hi", type=int, default=81,
                    help="skip pairs with nA*nB above this (collisions can't reach the window)")
    ap.add_argument("--out", default="runs/chase/n40/minkowski")
    ap.add_argument("--save-at", type=int, default=136)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a", buffering=1)

    print("enumerating factor classes ...", flush=True)
    t0 = time.time()
    lib = enumerate_factors(args.nmax, args.slack)
    print(f"factor library done in {time.time()-t0:.1f}s", flush=True)

    # factor list with sizes; only pairs whose product can reach the window
    factors: list[np.ndarray] = []
    for j in range(2, args.nmax + 1):
        factors.extend(lib[j])
    print(f"total factors: {len(factors)}", flush=True)

    units = [tuple(int(x) for x in u) for u in UC] + [(1, 0, 0, 0)]
    seen_sums: set[bytes] = set()
    pair_swept: set[bytes] = set()
    results: list[tuple[int, str]] = []
    best_overall = 0
    n_tested = 0

    # priority order: factor-pair products near the target first, denser first
    def density(F):
        return edge_count(F) / max(1, U_OPT.get(len(F), 99))

    pairs = []
    for ia, A in enumerate(factors):
        for ib in range(ia, len(factors)):
            B = factors[ib]
            prod = len(A) * len(B)
            if prod < args.size_lo or prod > args.prod_hi:
                continue
            pairs.append((abs(prod - 42), -(density(A) + density(B)), ia, ib))
    pairs.sort()
    print(f"pair queue: {len(pairs)}", flush=True)

    t0 = time.time()
    for nq, (_, _, ia, ib) in enumerate(pairs):
        A, B = factors[ia], factors[ib]
        rots = []
        seen_rot: set[bytes] = set()
        for u in units:
            rB = rot_config(B, u)
            if rB is None:
                continue
            kr = canon(rB)
            if kr in seen_rot:
                continue
            seen_rot.add(kr)
            rots.append(rB)
        for rB in rots:
            S = mink_sum(A, rB)
            if not (args.size_lo <= len(S) <= args.size_hi):
                continue
            ks = canon(S)
            if ks in seen_sums:
                continue
            seen_sums.add(ks)
            n_tested += 1
            raw = edge_count(S)
            for P in steer_to_n(S, args.target):
                P, c = climb(P, steps=2)
                while c >= 135:
                    key = canon(P)
                    if key in pair_swept:
                        break
                    pair_swept.add(key)
                    d, v1, v2, c1, c2 = best_pair_move(P, steps=1)
                    if d <= 0:
                        break
                    P = P.copy()
                    P[v1] = c1
                    P[v2] = c2
                    P, c = climb(P, steps=2)
                if c > best_overall or c >= args.save_at:
                    stem = f"mink_a{ia}_b{ib}_e{c}"
                    ms = min_sep(P)
                    save_checkpoint(P, stem, str(out))
                    log.write(json.dumps({
                        "event": "good", "count": c, "minsep": ms,
                        "nA": len(A), "nB": len(B), "eA": edge_count(A),
                        "eB": edge_count(B), "sum_n": len(S),
                        "sum_edges": raw, "stem": stem}) + "\n")
                    print(f"GOOD {c} edges (minsep {ms:.3f}) from "
                          f"{len(A)}x{len(B)} sum (|S|={len(S)}, raw {raw})",
                          flush=True)
                    results.append((c, stem))
                best_overall = max(best_overall, c)
        if nq % 200 == 0:
            print(f"[{time.time()-t0:.0f}s] pair {nq}/{len(pairs)} "
                  f"sums_tested={n_tested} best={best_overall}", flush=True)
            log.write(json.dumps({"event": "progress", "pair": nq,
                                  "tested": n_tested, "best": best_overall}) + "\n")

    print(f"DONE tested={n_tested} best={best_overall}", flush=True)
    log.write(json.dumps({"event": "done", "tested": n_tested,
                          "best": best_overall}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
