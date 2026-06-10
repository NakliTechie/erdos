#!/usr/bin/env python
"""Basin-hopping restart campaign at n=40 over exact ML coordinates.

Loop: perturb k in {2,3,4} vertices of the current config -> steepest-ascent
singles (2-hop universe) -> exhaustive pair-move polish when promising ->
accept by Metropolis on exact counts; tabu on canonical forms. All moves
respect the min_sep >= 0.2 audit floor. Any config with count >= save_at is
checkpointed immediately (exact JSON + CSV).

Usage:
  uv run python scripts/chase40_restart.py --seed 0 --hours 2 \
      --start data/mlcoords/udg40_136edges.json --out runs/chase/n40/restart_s0
"""

from __future__ import annotations

import argparse
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
    edge_count,
    load_cert,
    min_sep,
    perturb,
    save_checkpoint,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--start", nargs="+", default=["data/mlcoords/udg40_136edges.json"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-at", type=int, default=136)
    ap.add_argument("--bh-temp", type=float, default=0.8)
    ap.add_argument("--pair-threshold", type=int, default=135)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a", buffering=1)

    starts = [load_cert(s) for s in args.start]
    cur = starts[rng.integers(len(starts))].copy()
    cur_count = edge_count(cur)
    best = cur.copy()
    best_count = cur_count

    tabu: set[bytes] = {canon(cur)}
    pair_swept: set[bytes] = set()
    classes_at: dict[int, int] = {}
    t0 = time.time()
    it = 0
    n_saved = 0
    while time.time() - t0 < args.hours * 3600:
        it += 1
        k = int(rng.integers(2, 7))  # 2..6
        Q = perturb(cur, k, rng)
        Q, q = climb(Q, steps=2)
        key = canon(Q)
        # exhaustive pair-move polish, once per canonical class
        while q >= args.pair_threshold and key not in pair_swept:
            pair_swept.add(key)
            d, v1, v2, c1, c2 = best_pair_move(Q, steps=1)
            if d <= 0:
                break
            Q = Q.copy()
            Q[v1] = c1
            Q[v2] = c2
            Q, q = climb(Q, steps=2)
            key = canon(Q)
        new_class = key not in tabu
        if new_class:
            tabu.add(key)
            classes_at[q] = classes_at.get(q, 0) + 1
        if q > best_count:
            best, best_count = Q.copy(), q
            stem = f"s{args.seed}_it{it}_e{q}"
            save_checkpoint(Q, stem, str(out))
            n_saved += 1
            log.write(json.dumps({"it": it, "event": "NEW BEST", "count": q,
                                  "minsep": min_sep(Q), "stem": stem}) + "\n")
            print(f"[seed {args.seed}] it={it} NEW BEST {q}", flush=True)
        elif q >= args.save_at and new_class and n_saved < 200:
            stem = f"s{args.seed}_it{it}_e{q}"
            save_checkpoint(Q, stem, str(out))
            n_saved += 1
            log.write(json.dumps({"it": it, "event": "class", "count": q,
                                  "minsep": min_sep(Q), "stem": stem}) + "\n")
        # basin-hopping acceptance (tabu-blocked configs never accepted)
        if new_class and (q >= cur_count or
                          rng.random() < np.exp((q - cur_count) / args.bh_temp)):
            cur, cur_count = Q, q
        elif it % 50 == 0:
            # periodic re-center on best / a fresh start to avoid drift
            if rng.random() < 0.5:
                cur, cur_count = best.copy(), best_count
            else:
                cur = starts[rng.integers(len(starts))].copy()
                cur_count = edge_count(cur)
        if it % 100 == 0:
            hist = {c: classes_at[c] for c in sorted(classes_at, reverse=True)[:6]}
            msg = {"it": it, "elapsed_s": round(time.time() - t0),
                   "best": best_count, "cur": cur_count,
                   "classes_top": hist, "tabu": len(tabu)}
            log.write(json.dumps(msg) + "\n")
            print(f"[seed {args.seed}] it={it} best={best_count} "
                  f"cur={cur_count} tabu={len(tabu)} top={hist}", flush=True)

    log.write(json.dumps({"event": "done", "iters": it, "best": best_count,
                          "tabu": len(tabu)}) + "\n")
    print(f"[seed {args.seed}] DONE iters={it} best={best_count}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
