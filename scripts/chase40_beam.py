#!/usr/bin/env python
"""Diverse beam search from scratch over the Moser lattice, target n=40.

Engel et al. found the (unique) 137-edge n=40 record with diverse beam
search over ML; this is our exact-arithmetic small-scale version. Grow
configs point by point; at each level keep a width-W beam, deduplicated by
translation-canonical hash, with a per-edge-count-bucket cap for diversity
and stochastic tie-breaking. Final configs (and the n=39 layer, mirroring
the 39 = 7+8 Minkowski record) get the full polish (singles + exhaustive
pair moves) and are checkpointed when >= save-at.

Usage:
  uv run python scripts/chase40_beam.py --seed 0 --width 1200 --runs 4 \
      --out runs/chase/n40/beam_s0
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
    min_sep,
    save_checkpoint,
    sep_viol_mask,
    unit_mask,
    universe,
)

_PAIR_SWEPT: set[bytes] = set()


def full_polish(P: np.ndarray, pair_threshold: int = 135):
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


def tkey(P: np.ndarray) -> bytes:
    """Translation-canonical hash (cheap beam dedup; rotations not merged)."""
    rows = sorted(map(tuple, P.tolist()))
    base = rows[0]
    return np.array([[r[k] - base[k] for k in range(4)] for r in rows],
                    dtype=np.int16).tobytes()


def beam_run(rng: np.random.Generator, width: int, n_target: int,
             top_greedy: int = 8, top_random: int = 4,
             bucket_frac: float = 0.34):
    """One diverse beam run; returns dict level -> list[(edges, P)] for the
    last two levels (n_target-1, n_target)."""
    P0 = np.zeros((1, 4), dtype=np.int64)
    beam: list[tuple[int, np.ndarray]] = [(0, P0)]
    out: dict[int, list[tuple[int, np.ndarray]]] = {}
    for level in range(2, n_target + 1):
        children: dict[bytes, tuple[int, np.ndarray]] = {}
        for e, P in beam:
            U = universe(P, 1)
            U = U[~sep_viol_mask(U, P).any(axis=1)]
            if len(U) == 0:
                continue
            g = unit_mask(U, P).sum(axis=1)
            order = np.argsort(-g)
            picks = list(order[:top_greedy])
            rest = order[top_greedy:]
            if len(rest) and top_random:
                picks += list(rng.choice(rest,
                                         size=min(top_random, len(rest)),
                                         replace=False))
            for i in picks:
                child = np.vstack([P, U[int(i)][None, :]])
                k = tkey(child)
                if k not in children:
                    children[k] = (e + int(g[int(i)]), child)
        if not children:
            break
        # diverse selection: bucket by edge count, cap per bucket
        buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
        for e, P in children.values():
            buckets.setdefault(e, []).append((e, P))
        cap = max(8, int(width * bucket_frac))
        newbeam: list[tuple[int, np.ndarray]] = []
        for e in sorted(buckets, reverse=True):
            lst = buckets[e]
            rng.shuffle(lst)
            newbeam.extend(lst[:cap])
            if len(newbeam) >= width:
                break
        beam = newbeam[:width]
        if level >= n_target - 1:
            out[level] = sorted(beam, key=lambda t: -t[0])[:40]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-at", type=int, default=136)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a", buffering=1)
    rng = np.random.default_rng(args.seed)

    best_overall = 0
    for run in range(args.runs):
        t0 = time.time()
        layers = beam_run(rng, args.width, args.n)
        raw_best = {lv: layers[lv][0][0] for lv in layers}
        print(f"[seed {args.seed} run {run}] beam done {time.time()-t0:.0f}s "
              f"raw best {raw_best}", flush=True)
        for lv, members in sorted(layers.items()):
            for e, P in members[:25]:
                if lv < args.n:  # 39-layer: add best vertex first
                    U = universe(P, 1)
                    U = U[~sep_viol_mask(U, P).any(axis=1)]
                    if len(U) == 0:
                        continue
                    g = unit_mask(U, P).sum(axis=1)
                    P = np.vstack([P, U[int(np.argmax(g))][None, :]])
                P, c = full_polish(P)
                if c > best_overall or c >= args.save_at:
                    ms = min_sep(P)
                    stem = f"beam_s{args.seed}_r{run}_l{lv}_e{c}"
                    save_checkpoint(P, stem, str(out))
                    log.write(json.dumps({"run": run, "level": lv, "count": c,
                                          "minsep": ms, "stem": stem}) + "\n")
                    print(f"  GOOD {c} (minsep {ms:.3f}) {stem}", flush=True)
                best_overall = max(best_overall, c)
        log.write(json.dumps({"run": run, "best": best_overall,
                              "raw": raw_best}) + "\n")
    print(f"[seed {args.seed}] DONE best={best_overall}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
