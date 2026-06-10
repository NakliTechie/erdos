#!/usr/bin/env python
"""Subset-in-closure ILS CLI (thin wrapper over udg.subsetsearch).

The n=30 93-record recipe: densest-k-subgraph ILS inside an ambient ML
point set. Ambients: the wheel49 cross-sublattice sum, its closures,
hex-patch cross sums, or any exact-coords JSON (closed `--closure` steps).

Usage (from repo root):
  uv run python scripts/chase_subset.py --k 30 --ambient wheel49 --closure 1 \
      --iters 2000 --seed 0 --out runs/chase/subset
  uv run python scripts/chase_subset.py --k 50 --ambient hex:3,2 \
      --minutes 10 --out runs/chase/subset
  uv run python scripts/chase_subset.py --k 30 --ambient some_coords.json
"""

from __future__ import annotations

import argparse

import numpy as np

from udg.mlgraph import MLConfig, minkowski
from udg.subsetsearch import (
    edge_count,
    hex_patch,
    neighbor_closure,
    save_checkpoint,
    subset_ils,
    to_w3,
    wheel49,
)


def build_ambient(spec: str, closure: int) -> np.ndarray:
    if spec == "wheel49":
        pts = wheel49().as_array().astype(np.int64)
    elif spec.startswith("hex:"):
        r1, r2 = (int(x) for x in spec[4:].split(","))
        pts = minkowski(hex_patch(r1), to_w3(hex_patch(r2))).as_array().astype(np.int64)
    elif spec.endswith(".json"):
        pts = MLConfig.from_json(spec).as_array().astype(np.int64)
    else:
        raise SystemExit(f"unknown ambient spec {spec!r} "
                         "(wheel49 | hex:r1,r2 | <coords.json>)")
    if closure > 0:
        pts = neighbor_closure(pts, closure)
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--k", type=int, required=True, help="subset size (= n)")
    ap.add_argument("--ambient", default="wheel49",
                    help="wheel49 | hex:r1,r2 | <coords.json>")
    ap.add_argument("--closure", type=int, default=1,
                    help="neighbor-closure steps applied to the ambient")
    ap.add_argument("--iters", type=int, default=None, help="max ILS kicks")
    ap.add_argument("--minutes", type=float, default=None, help="wall budget")
    ap.add_argument("--target", type=int, default=None,
                    help="stop as soon as best >= target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/chase/subset")
    args = ap.parse_args()

    if args.iters is None and args.minutes is None:
        args.iters = 2000

    amb = build_ambient(args.ambient, args.closure)
    print(f"ambient {args.ambient} closure={args.closure}: {len(amb)} points, "
          f"{edge_count(amb)} edges", flush=True)
    m, pts = subset_ils(
        amb, args.k, seed=args.seed,
        minutes=args.minutes, max_iters=args.iters, target=args.target,
        log=lambda s: print(s, flush=True),
    )
    jpath = save_checkpoint(pts, args.out, f"subset_s{args.seed}")
    print(f"BEST {m} edges at k={args.k} -> {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
