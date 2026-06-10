#!/usr/bin/env python
"""Re-run the 4 distinct instrumented control chains (T0 irrelevant — seed-
matched trajectories proved bit-identical across T0), audit the FINAL plateau
configs, save those that pass with >= 132 edges, and compare their near-miss
structure to the input's."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from udg.audit import audit
from udg.configio import load_csv, save_audit_json, save_csv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from control_instrumented import search_instrumented  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "runs" / "hinge"


def near_miss_min(P):
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    iu, ju = np.triu_indices(len(P), 1)
    dev = np.abs(D[iu, ju] - 1.0)
    ne = dev[dev >= 1e-9]
    bands = [1e-9, 1e-6, 1e-4, 1e-2, 5e-2]
    hist = {f"[{lo:g},{hi:g})": int(((ne >= lo) & (ne < hi)).sum())
            for lo, hi in zip(bands[:-1], bands[1:])}
    return float(ne.min()), hist


def main():
    P0 = load_csv(ROOT / "data" / "udg40_132edges.csv")
    m0, h0 = near_miss_min(P0)
    print(f"input: min nonedge |d-1| = {m0:.6e}  bands={h0}", flush=True)
    saved = []
    meta = []
    for seed in [0, 1, 2, 3]:
        P, final_cur, best, stats = search_instrumented(P0, 30_000, seed, T0=0.08)
        rep = audit(P)
        m, h = near_miss_min(P)
        print(f"seed={seed} final_count={final_cur} audited={rep.n_edges} "
              f"passed={rep.passed} min_sep_after={rep.min_sep_after:.4f} "
              f"min_nonedge_dev={m:.6e} bands={h}", flush=True)
        rec = {"seed": seed, "final_count": final_cur,
               "audited_edges": rep.n_edges, "passed": bool(rep.passed),
               "min_nonedge_abs_d_minus_1": m, "bands": h}
        if rep.passed and rep.n_edges >= 132:
            tag = f"control_plateau_seed{seed}"
            save_csv(P, OUTDIR / f"{tag}.csv")
            save_audit_json(rep, OUTDIR / f"{tag}_audit.json")
            rec["csv"] = str(OUTDIR / f"{tag}.csv")
            saved.append(str(OUTDIR / f"{tag}.csv"))
        meta.append(rec)
    with open(OUTDIR / "control_plateau_summary.json", "w") as f:
        json.dump({"input_min_nonedge": m0, "input_bands": h0, "runs": meta,
                   "saved": saved}, f, indent=2)
        f.write("\n")
    print("saved:", saved, flush=True)


if __name__ == "__main__":
    main()
