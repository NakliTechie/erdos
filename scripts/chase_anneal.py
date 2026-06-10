#!/usr/bin/env python
"""Exact-lattice simulated annealing CLI (thin wrapper over udg.anneal).

The n=70 281-record recipe: pool-reseeded parallel annealing over integer
ML coordinates, seeded from exact-coords JSON files (data/mlcoords
certificates or chase checkpoints, key "coords").

Usage (from repo root):
  uv run python scripts/chase_anneal.py data/mlcoords/udg70_277edges.json \
      --minutes 45 --procs 6 --out runs/chase/anneal
"""

from __future__ import annotations

import argparse

from udg.anneal import anneal_pool, save_checkpoint
from udg.mlgraph import MLConfig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("starts", nargs="+", help="exact-coords JSON file(s)")
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--t0", type=float, default=1.0)
    ap.add_argument("--t1", type=float, default=0.05)
    ap.add_argument("--reheats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--target", type=int, default=None,
                    help="stop as soon as best >= target")
    ap.add_argument("--out", default="runs/chase/anneal")
    args = ap.parse_args()

    starts = [list(MLConfig.from_json(p).points) for p in args.starts]
    best_e, best_pts, hist = anneal_pool(
        starts,
        minutes=args.minutes,
        procs=args.procs,
        steps=args.steps,
        t0=args.t0,
        t1=args.t1,
        reheats=args.reheats,
        seed=args.seed,
        out=args.out,
        target=args.target,
    )
    j, c = save_checkpoint(best_pts, args.out, f"anneal_best_{best_e}")
    print(f"BEST {best_e} -> {j}")
    print("hist:", dict(sorted(hist.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
