#!/usr/bin/env python
"""Exhaustive drop-k/re-add + dimension-lifting moves on n=40 configs.

Arms (all exact, all minsep-guarded):
  A. drop-k re-add, k=1,2[,3 sampled]: remove every k-subset of vertices,
     rebuild to n=40 with a beam over best additions, climb, pair-polish.
  B. lift: add one of the top-`lift_top` candidate 41st vertices, climb in
     41-space, then try every drop back to 40 (or beam), climb, pair-polish.

Usage:
  uv run python scripts/chase40_dropk.py CFG.json [CFG2.json ...] \
      --out runs/chase/n40/dropk [--k3-sample 2000]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from chase40lib import (  # noqa: E402
    best_pair_move,
    canon,
    climb,
    degs,
    edge_count,
    load_cert,
    min_sep,
    save_checkpoint,
    sep_viol_mask,
    unit_mask,
    universe,
)


def rebuild(P: np.ndarray, n: int, beam: int = 6) -> list[np.ndarray]:
    """Beam re-add from len(P) up to n points (minsep-guarded)."""
    front = {canon(P): P}
    while len(next(iter(front.values()))) < n:
        new: dict[bytes, np.ndarray] = {}
        for Q in front.values():
            U = universe(Q, 1)
            U = U[~sep_viol_mask(U, Q).any(axis=1)]
            if len(U) == 0:
                continue
            g = unit_mask(U, Q).sum(axis=1)
            order = np.argsort(-g)
            gtop = g[order[0]]
            picks = [i for i in order[:10] if g[i] >= gtop - 1]
            for i in picks[:6]:
                child = np.vstack([Q, U[int(i)][None, :]])
                new[canon(child)] = child
        if not new:
            return []
        ranked = sorted(new.values(), key=edge_count, reverse=True)
        front = {canon(Q): Q for Q in ranked[:beam]}
    return list(front.values())


_PAIR_SWEPT: set[bytes] = set()


def full_polish(P: np.ndarray, pair_threshold: int = 135):
    """Singles to fixpoint + exhaustive pair sweeps; pair sweeps are cached
    per canonical class (they are deterministic), so re-encountering the
    same local optimum costs nothing."""
    P, c = climb(P, steps=2)
    while c >= pair_threshold:
        key = canon(P)
        if key in _PAIR_SWEPT:
            break
        _PAIR_SWEPT.add(key)
        d, v1, v2, c1, c2 = best_pair_move(P, steps=1)
        if d <= 0:
            break
        P = P.copy()
        P[v1] = c1
        P[v2] = c2
        c += d
        P, c = climb(P, steps=2)
    return P, c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--out", default="runs/chase/n40/dropk")
    ap.add_argument("--k3-sample", type=int, default=0)
    ap.add_argument("--lift-top", type=int, default=12)
    ap.add_argument("--save-at", type=int, default=136)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a", buffering=1)
    rng = np.random.default_rng(0)

    best_overall = 0
    seen_results: set[bytes] = set()
    new136: list[str] = []

    for cfgpath in args.configs:
        P0 = load_cert(cfgpath)
        e0 = edge_count(P0)
        n = len(P0)
        name = Path(cfgpath).stem
        print(f"=== {name}: n={n} edges={e0}", flush=True)

        jobs: list[tuple[str, tuple[int, ...]]] = []
        jobs += [("drop1", (i,)) for i in range(n)]
        jobs += [("drop2", t) for t in itertools.combinations(range(n), 2)]
        if args.k3_sample:
            all3 = list(itertools.combinations(range(n), 3))
            idx = rng.choice(len(all3), size=min(args.k3_sample, len(all3)),
                             replace=False)
            jobs += [("drop3", all3[i]) for i in idx]

        t0 = time.time()
        local_best = 0
        for jn, (kind, drop) in enumerate(jobs):
            Q = np.delete(P0, list(drop), axis=0)
            for R in rebuild(Q, n):
                R, c = full_polish(R)
                key = canon(R)
                if key in seen_results:
                    continue
                seen_results.add(key)
                local_best = max(local_best, c)
                if c > best_overall or c >= args.save_at:
                    ms = min_sep(R)
                    stem = f"{name}_{kind}_{jn}_e{c}"
                    save_checkpoint(R, stem, str(out))
                    log.write(json.dumps({"src": name, "kind": kind,
                                          "drop": list(drop), "count": c,
                                          "minsep": ms, "stem": stem}) + "\n")
                    if c >= args.save_at:
                        new136.append(stem)
                    if c > best_overall:
                        print(f"  NEW BEST {c} ({kind} {drop}) minsep={ms:.3f}",
                              flush=True)
                best_overall = max(best_overall, c)
            if jn % 100 == 0:
                print(f"  [{time.time()-t0:.0f}s] {jn}/{len(jobs)} "
                      f"local_best={local_best} overall={best_overall}", flush=True)

        # arm B: lift to 41 then drop back
        U = universe(P0, 1)
        U = U[~sep_viol_mask(U, P0).any(axis=1)]
        g = unit_mask(U, P0).sum(axis=1)
        for i in np.argsort(-g)[: args.lift_top]:
            L = np.vstack([P0, U[int(i)][None, :]])
            L, cl = climb(L, steps=2)
            D = degs(L)
            for j in np.argsort(D)[:8]:
                R = np.delete(L, int(j), axis=0)
                R, c = full_polish(R)
                key = canon(R)
                if key in seen_results:
                    continue
                seen_results.add(key)
                if c > best_overall or c >= args.save_at:
                    ms = min_sep(R)
                    stem = f"{name}_lift_{int(i)}_{int(j)}_e{c}"
                    save_checkpoint(R, stem, str(out))
                    log.write(json.dumps({"src": name, "kind": "lift",
                                          "count": c, "minsep": ms,
                                          "stem": stem}) + "\n")
                    if c >= args.save_at:
                        new136.append(stem)
                    if c > best_overall:
                        print(f"  NEW BEST {c} (lift) minsep={ms:.3f}", flush=True)
                best_overall = max(best_overall, c)
        print(f"=== {name} done: local_best={local_best} overall={best_overall}",
              flush=True)

    print(f"DONE best={best_overall} distinct_results={len(seen_results)} "
          f"saved>=136: {len(new136)}", flush=True)
    log.write(json.dumps({"event": "done", "best": best_overall}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
