#!/usr/bin/env python
"""EXPERIMENT 1 extension — warm-start search battery from the 136-edge config.

The 136 (runs/hinge/homotopy_fire_search_s0.csv) is rigid (flex_dim 0), fully
ML-locked, closest non-unit pair at 0.0818 — no hinge iteration possible.
Test whether plain low-temperature search can squeeze out a 137.
T0 in {0.08, 0.15}, T1=0.01, 30k steps, seeds 0-3, max 4 processes.
"""

from __future__ import annotations

import json
import time

import numpy as np

from udg.audit import audit
from udg.configio import load_csv, save_audit_json, save_csv
from udg.search import multi_search


def main() -> None:
    P136 = load_csv("runs/hinge/homotopy_fire_search_s0.csv")
    out = {"experiment": "search battery from the 136-edge config", "runs": []}
    best = 136
    t0 = time.time()
    for T0 in (0.08, 0.15):
        results = multi_search(40, [0, 1, 2, 3], 30_000, processes=4,
                               T0=T0, T1=0.01, P0=P136)
        for r in results:
            rep = audit(r.P)
            row = {"T0": T0, "seed": r.seed, "tol_count": r.best_count,
                   "audited": rep.n_edges, "passed": bool(rep.passed),
                   "identical": bool(np.array_equal(r.P, P136))}
            if rep.passed and rep.n_edges > 136:
                path = f"runs/hinge/homotopy_136ext_T{T0}_s{r.seed}.csv"
                save_csv(r.P, path)
                save_audit_json(rep, path.replace(".csv", "_audit.json"))
                row["saved"] = path
            if rep.passed:
                best = max(best, rep.n_edges)
            out["runs"].append(row)
            print(row)
    out["seconds"] = round(time.time() - t0, 1)
    out["best_audited"] = best
    with open("runs/hinge/homotopy_136ext_summary.json", "w") as f:
        json.dump(out, f, indent=1)
    print("best audited from 136:", best)


if __name__ == "__main__":
    main()
