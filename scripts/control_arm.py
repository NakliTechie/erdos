#!/usr/bin/env python
"""CONTROL ARM for the hinge-locking experiment (plan/hinge-design.md).

No hinge locking: establish what PLAIN warm-start local search achieves from
data/udg40_132edges.csv, so the locking experiments have a null hypothesis.

Grid: T0 in {0.03, 0.08, 0.15} x steps in {30_000, 100_000} x 4 seeds
(T1 = 0.01 fixed), warm start P0 = the 132-edge config. Every run's best
config goes through the three-audit gate (udg.audit.audit). Any config with
audited passed=True and n_edges >= 132 that DIFFERS from the input is saved
to runs/hinge/control_*.csv (+ audit JSON). Runs whose best config is
bit-identical to the input are recorded but not re-saved (the input itself
is data/udg40_132edges.csv).

Also: a fire-check near-miss diagnostic of the ORIGINAL config — histogram
of |d - 1| over the 648 non-edge pairs in bands
[1e-9,1e-6), [1e-6,1e-4), [1e-4,1e-2), [1e-2,5e-2), plus the 12 nearest
misses verbatim.

Output: runs/hinge/control_summary.json + stdout log. Max 4 processes.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from udg.audit import audit
from udg.configio import load_csv, save_audit_json, save_csv
from udg.search import multi_search

ROOT = Path(__file__).resolve().parent.parent
SUBJECT = ROOT / "data" / "udg40_132edges.csv"
OUTDIR = ROOT / "runs" / "hinge"

T0_GRID = [0.03, 0.08, 0.15]
STEPS_GRID = [30_000, 100_000]
SEEDS = [0, 1, 2, 3]
T1 = 0.01
PROCESSES = 4  # hard cap (concurrent campaign may be running)

BANDS = [1e-9, 1e-6, 1e-4, 1e-2, 5e-2]


def near_miss_diagnostic(P: np.ndarray) -> dict:
    """|d-1| histogram over NON-edge pairs (fire-check style)."""
    n = len(P)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    iu, ju = np.triu_indices(n, 1)
    dev = np.abs(D[iu, ju] - 1.0)
    edge_mask = dev < 1e-9
    nonedge_dev = dev[~edge_mask]
    bands = {}
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        bands[f"[{lo:g},{hi:g})"] = int(((nonedge_dev >= lo) & (nonedge_dev < hi)).sum())
    order = np.argsort(nonedge_dev)
    nonedge_pairs = np.stack([iu[~edge_mask], ju[~edge_mask]], axis=1)
    nearest = [
        {
            "i": int(nonedge_pairs[k][0]),
            "j": int(nonedge_pairs[k][1]),
            "abs_d_minus_1": float(nonedge_dev[k]),
            "d": float(D[nonedge_pairs[k][0], nonedge_pairs[k][1]]),
        }
        for k in order[:12]
    ]
    return {
        "n_pairs": int(len(dev)),
        "n_edges_tol_1e-9": int(edge_mask.sum()),
        "n_nonedge_pairs": int(len(nonedge_dev)),
        "bands_abs_d_minus_1": bands,
        "nearest_12_misses": nearest,
        "min_nonedge_abs_d_minus_1": float(nonedge_dev.min()),
    }


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    P0 = load_csv(SUBJECT)

    print("=== baseline audit of subject config ===", flush=True)
    base = audit(P0)
    print(base, flush=True)

    print("=== near-miss diagnostic (original config) ===", flush=True)
    nm = near_miss_diagnostic(P0)
    print(json.dumps(nm, indent=2), flush=True)

    runs = []
    saved = []
    for T0 in T0_GRID:
        for steps in STEPS_GRID:
            t_start = time.time()
            results = multi_search(
                len(P0), SEEDS, steps, processes=PROCESSES,
                T0=T0, T1=T1, P0=P0,
            )
            wall = time.time() - t_start
            for res, seed in zip(results, SEEDS):
                rep = audit(res.P)
                identical = bool(np.array_equal(res.P, P0))
                rec = {
                    "T0": T0,
                    "T1": T1,
                    "steps": steps,
                    "seed": seed,
                    "best_count_tol": res.best_count,
                    "audited_edges": rep.n_edges,
                    "min_sep": rep.min_sep,
                    "k23_violations": rep.k23_violations,
                    "gn_total_residual": rep.gn_total_residual,
                    "gn_edges_exact": rep.gn_edges_exact,
                    "gn_max_move": rep.gn_max_move,
                    "min_sep_after": rep.min_sep_after,
                    "passed": rep.passed,
                    "identical_to_input": identical,
                }
                print(
                    f"T0={T0} steps={steps} seed={seed} "
                    f"best_count={res.best_count} audited={rep.n_edges} "
                    f"passed={rep.passed} min_sep_after={rep.min_sep_after:.4f} "
                    f"identical_to_input={identical}",
                    flush=True,
                )
                if rep.passed and rep.n_edges >= 132 and not identical:
                    tag = f"control_T{T0:g}_s{steps}_seed{seed}"
                    csv_path = OUTDIR / f"{tag}.csv"
                    save_csv(res.P, csv_path)
                    save_audit_json(rep, OUTDIR / f"{tag}_audit.json")
                    rec["saved_csv"] = str(csv_path)
                    saved.append(str(csv_path))
                runs.append(rec)
            print(f"--- combo T0={T0} steps={steps}: {wall:.1f}s wall ---", flush=True)

    summary = {
        "experiment": "control arm — plain warm-start search, no hinge locking",
        "subject": str(SUBJECT),
        "baseline_audit": dataclasses.asdict(base),
        "near_miss_diagnostic_original": nm,
        "grid": {"T0": T0_GRID, "T1": T1, "steps": STEPS_GRID, "seeds": SEEDS},
        "runs": runs,
        "saved_configs": saved,
    }
    with open(OUTDIR / "control_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    counts = sorted(r["best_count_tol"] for r in runs)
    audited = sorted(r["audited_edges"] for r in runs if r["passed"])
    print("=== DONE ===", flush=True)
    print(f"best_count_tol distribution: {counts}", flush=True)
    print(f"audited(passed) distribution: {audited}", flush=True)
    print(f"saved {len(saved)} configs: {saved}", flush=True)
    return 0


if __name__ == "__main__":  # spawn-safe for multiprocessing on macOS
    raise SystemExit(main())
