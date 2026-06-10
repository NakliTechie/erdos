#!/usr/bin/env python
"""Instrumented replica of udg.search.search for the CONTROL ARM.

Identical move/accept semantics (verified against udg/search.py), plus
tracking: accepted-move counts by delta sign, max count reached, FINAL chain
count (search() only returns bestP — the final state shows hold-vs-degrade),
and an audit of the final configuration.

Answers: at low T, does the warm-started chain actively walk a neutral
plateau, degrade, or freeze? 12 runs: T0 in {0.03,0.08,0.15} x 30k steps x
4 seeds, T1=0.01. Serial (fast; respects the 4-process cap trivially).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from udg.audit import audit
from udg.configio import load_csv
from udg.search import circle_intersections

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-9
MIN_SEP = 0.2


def search_instrumented(P0, steps, seed, T0, T1=0.01):
    rng = np.random.default_rng(seed)
    P = np.array(P0, dtype=np.float64, copy=True)
    n = len(P)

    def count_at(pt, P, k):
        d = np.sqrt(((P - pt) ** 2).sum(1))
        d[k] = np.inf
        if d.min() < MIN_SEP:
            return None
        return int((np.abs(d - 1) < TOL).sum())

    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    cur = int((np.abs(D - 1) < TOL).sum() // 2)
    best = cur
    stats = {"acc_pos": 0, "acc_zero": 0, "acc_neg": 0,
             "rej_minsep_or_none": 0, "rej_metropolis": 0, "noop": 0}
    for t in range(steps):
        T = T0 * (T1 / T0) ** (t / steps)
        k = rng.integers(n)
        a, b = rng.choice(n, 2, replace=False)
        if a == k or b == k:
            stats["noop"] += 1
            continue
        cands = circle_intersections(P[a], P[b])
        if not cands:
            stats["noop"] += 1
            continue
        cand = cands[rng.integers(len(cands))]
        cn = count_at(cand, P, k)
        if cn is None:
            stats["rej_minsep_or_none"] += 1
            continue
        dold = np.sqrt(((P - P[k]) ** 2).sum(1))
        dold[k] = np.inf
        co = int((np.abs(dold - 1) < TOL).sum())
        delta = cn - co
        if delta >= 0 or rng.random() < np.exp(delta / T):
            P[k] = cand
            cur += delta
            if delta > 0:
                stats["acc_pos"] += 1
            elif delta == 0:
                stats["acc_zero"] += 1
            else:
                stats["acc_neg"] += 1
            if cur > best:
                best = cur
        else:
            stats["rej_metropolis"] += 1
    return P, cur, best, stats


def main():
    P0 = load_csv(ROOT / "data" / "udg40_132edges.csv")
    out = []
    for T0 in [0.03, 0.08, 0.15]:
        for seed in [0, 1, 2, 3]:
            P, final_cur, best, stats = search_instrumented(P0, 30_000, seed, T0)
            rep = audit(P)
            moved = float(np.sqrt(((P - P0) ** 2).sum(1)).max())
            rec = {
                "T0": T0, "steps": 30_000, "seed": seed,
                "max_count_reached": best,
                "final_chain_count": final_cur,
                "final_audited_edges": rep.n_edges,
                "final_passed": bool(rep.passed),
                "max_point_displacement_vs_input": moved,
                **stats,
            }
            out.append(rec)
            print(json.dumps(rec), flush=True)
    accz = sum(r["acc_zero"] for r in out)
    accn = sum(r["acc_neg"] for r in out)
    accp = sum(r["acc_pos"] for r in out)
    print(f"TOTAL accepted: +{accp} / 0:{accz} / -{accn} over {len(out)} runs", flush=True)
    with open(ROOT / "runs" / "hinge" / "control_instrumented.json", "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
