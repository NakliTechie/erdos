#!/usr/bin/env python
"""EXPERIMENT 1 follow-up — fire the near-miss triple along the f094 flex.

Finding from scripts/homotopy_lock.py: the 132-edge config has flex_dim 1;
locking the ~0.94 deg family to 0 carries ALL families to ML targets, and the
three near-miss pairs (9,26),(5,11),(10,39) move from |d-1| = 1.368e-2
(d < 1) to 1.73e-4 (d > 1): they CROSS unit along the flex. The firing point
is at family angle theta* ~ 0.011 deg (not the ML angle).

Procedure:
 1. Bisection on theta (lock_family target) over the bracket
    [0.0 deg (f=+1.7e-4), 0.8997 deg (f=-1.37e-2)] with continuation warm
    starts, until the triple is within ~1e-8 of unit.
 2. Add the 3 pairs to the edge set (135 edges) and run long Gauss-Newton;
    converged = total residual < 1e-26 (audit gate is 1e-24).
 3. Full three-audit gate; save >=132-audited configs; low-temperature
    warm-start search battery (T0=0.08, T1=0.01, 30k steps, 4 seeds,
    max 4 processes); audit + save everything >= 132.

If GN on 135 edges plateaus, the residual level itself is the finding (the
triple's crossing thetas split at ~1e-11 => not simultaneously realizable
at audit precision).

Output: runs/hinge/homotopy_fire_summary.json + homotopy_fire*.csv.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from udg.audit import MIN_SEP, audit, gauss_newton, min_separation
from udg.configio import load_csv, save_audit_json, save_csv
from udg.counting import unit_edges
from udg.hinge import (
    classify_families,
    edge_residual,
    family_angle,
    fire_check,
    flex_dimension,
    lock_family,
    signed_delta,
)
from udg.search import multi_search

ROOT = Path(__file__).resolve().parent.parent
SUBJECT = ROOT / "data" / "udg40_132edges.csv"
OUTDIR = ROOT / "runs" / "hinge"

SEEDS = [0, 1, 2, 3]
T0, T1, STEPS = 0.08, 0.01, 30_000
PROCESSES = 4


def triple_dev(P: np.ndarray, pairs) -> tuple[float, list[float]]:
    """Mean and per-pair d-1 for the candidate firing pairs."""
    devs = [float(np.linalg.norm(P[i] - P[j]) - 1.0) for i, j in pairs]
    return float(np.mean(devs)), devs


def long_gn(P, E, max_rounds=40, iters_per_round=25_000, lr=0.08):
    """Chunked Gauss-Newton with plateau detection."""
    Q = np.asarray(P, dtype=float).copy()
    hist = []
    prev = edge_residual(Q, E)
    for r in range(max_rounds):
        Q, res = gauss_newton(Q, E, lr=lr, iters=iters_per_round, target=1e-30)
        hist.append(res)
        if res < 1e-28:
            break
        if prev > 0 and res > prev * 0.5:  # < 2x improvement per round: plateau
            break
        prev = res
    return Q, res, hist


def main() -> None:
    P0 = load_csv(SUBJECT)
    E0 = unit_edges(P0)
    fc0 = classify_families(P0, E0)
    Pa = fc0.P
    E = unit_edges(Pa)
    assert len(E) == 132
    fam = min(
        (f for f in fc0.families if not f.locked),
        key=lambda f: abs(f.mean_angle_raw - 0.94),
    )
    pairs = [(9, 26), (5, 11), (10, 39)]

    summary: dict = {
        "experiment": "EXPERIMENT 1 follow-up — fire near-miss triple along the f094 flex",
        "pairs": pairs,
        "bisection": [],
    }

    # --- bracket: theta_hi = current angle (d-1 < 0), theta_lo = 0 (d-1 > 0)
    states: dict[float, np.ndarray] = {}

    def eval_theta(theta: float) -> tuple[float, np.ndarray, dict]:
        # warm start from the nearest evaluated state (continuation)
        if states:
            src = min(states, key=lambda t: abs(t - theta))
            Pst = states[src]
        else:
            Pst = Pa
        res = lock_family(
            Pst, E, fam, target=theta, n_increments=5, angle_tol=1e-10
        )
        f, devs = triple_dev(res.P, pairs)
        row = {
            "theta_target": theta,
            "theta_achieved": float(res.family_angle),
            "stop_reason": res.stop_reason,
            "residual": float(res.residual),
            "min_sep": float(res.min_sep),
            "mean_d_minus_1": f,
            "devs": devs,
        }
        states[theta] = res.P
        return f, res.P, row

    t_start = time.time()
    lo, hi = 0.0, float(fam.mean_angle)  # f(lo) > 0 expected, f(hi) < 0
    f_lo, P_lo, row = eval_theta(lo)
    summary["bisection"].append(row)
    print(f"theta={lo:.10f} f={f_lo:+.3e}")
    f_hi, P_hi, row = eval_theta(hi)
    summary["bisection"].append(row)
    print(f"theta={hi:.10f} f={f_hi:+.3e}")
    assert f_lo * f_hi < 0, "no sign change — bracket invalid"

    best_P, best_f = (P_lo, f_lo) if abs(f_lo) < abs(f_hi) else (P_hi, f_hi)
    for it in range(60):
        # secant-leaning bisection (regula falsi with bisection safeguard)
        mid_rf = lo + (hi - lo) * (f_lo / (f_lo - f_hi))
        mid_bi = 0.5 * (lo + hi)
        mid = mid_rf if (lo < mid_rf < hi) else mid_bi
        if it % 3 == 2:
            mid = mid_bi  # safeguard against one-sided stalling
        f_m, P_m, row = eval_theta(mid)
        summary["bisection"].append(row)
        print(f"  it{it:02d} theta={mid:.12f} f={f_m:+.6e} spread="
              f"{max(row['devs']) - min(row['devs']):.3e}")
        if abs(f_m) < abs(best_f):
            best_P, best_f = P_m, f_m
        if f_m == 0.0 or abs(f_m) < 1e-9:
            break
        if f_lo * f_m < 0:
            hi, f_hi = mid, f_m
        else:
            lo, f_lo = mid, f_m
        if hi - lo < 1e-12:
            break
    summary["bisection_seconds"] = round(time.time() - t_start, 1)
    summary["best_theta_mean_dev"] = best_f
    _, devs = triple_dev(best_P, pairs)
    summary["best_theta_devs"] = devs
    print(f"best |mean d-1| = {abs(best_f):.3e}; per-pair spread = "
          f"{max(devs) - min(devs):.3e}")

    # --- GN on 135 edges
    E135 = E + pairs
    t0 = time.time()
    Q, res, hist = long_gn(best_P, E135)
    summary["gn135"] = {
        "residual": res,
        "rounds": len(hist),
        "history": hist,
        "seconds": round(time.time() - t0, 1),
        "min_sep": float(min_separation(Q)),
    }
    print(f"GN-135: residual={res:.3e} after {len(hist)} rounds, "
          f"min_sep={min_separation(Q):.4f}")

    rep = audit(Q)
    summary["audit"] = dataclasses.asdict(rep)
    print(f"AUDIT: n_edges={rep.n_edges} passed={rep.passed} "
          f"min_sep={rep.min_sep:.4f} min_sep_after={rep.min_sep_after:.4f} "
          f"gn_res={rep.gn_total_residual:.3e} k23={rep.k23_violations}")

    saved = []
    if rep.passed and rep.n_edges >= 132:
        csv = OUTDIR / f"homotopy_fire{rep.n_edges}.csv"
        save_csv(Q, csv)
        save_audit_json(rep, OUTDIR / f"homotopy_fire{rep.n_edges}_audit.json")
        saved.append(str(csv))
        print(f"saved {csv}")

    # structure of the fired config
    if rep.passed:
        Ef = unit_edges(Q)
        summary["flex_dimension_fired"] = int(flex_dimension(Q, Ef))
        fcf = classify_families(Q, Ef)
        summary["fired_families"] = [
            {
                "n_edges": f.n_edges,
                "mean_angle_aligned": f.mean_angle,
                "target": f.target,
                "offset": f.offset,
                "locked": f.locked,
            }
            for f in fcf.families
        ]
        fcheck = fire_check(Q)
        summary["fire_check"] = {
            "n_unit": fcheck.n_unit,
            "bands": [[lo_, hi_, c] for lo_, hi_, c in fcheck.bands],
            "closest_nonunit": fcheck.closest_nonunit,
        }

        # --- warm-start search battery from the fired config
        t0 = time.time()
        results = multi_search(
            len(Q), SEEDS, STEPS, processes=PROCESSES, T0=T0, T1=T1, P0=Q
        )
        runs = []
        best_aud = rep.n_edges
        for r in results:
            rrep = audit(r.P)
            row = {
                "seed": r.seed,
                "best_count_tol": r.best_count,
                "audited_n_edges": rrep.n_edges,
                "passed": rrep.passed,
                "identical_to_warmstart": bool(np.array_equal(r.P, Q)),
            }
            if rrep.passed and rrep.n_edges >= 132 and not row["identical_to_warmstart"]:
                csv = OUTDIR / f"homotopy_fire_search_s{r.seed}.csv"
                save_csv(r.P, csv)
                save_audit_json(rrep, OUTDIR / f"homotopy_fire_search_s{r.seed}_audit.json")
                row["saved"] = str(csv)
                saved.append(str(csv))
            if rrep.passed:
                best_aud = max(best_aud, rrep.n_edges)
            runs.append(row)
            print(f"  search s{r.seed}: tol={r.best_count} audited={rrep.n_edges} "
                  f"passed={rrep.passed} identical={row['identical_to_warmstart']}")
        summary["search_seconds"] = round(time.time() - t0, 1)
        summary["search_runs"] = runs
        summary["post_search_best_audited"] = best_aud

    summary["saved"] = saved

    def _clean(v):
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.ndarray):
            return [_clean(x) for x in v.tolist()]
        if isinstance(v, dict):
            return {str(k): _clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_clean(x) for x in v]
        return v

    with open(OUTDIR / "homotopy_fire_summary.json", "w") as f:
        json.dump(_clean(summary), f, indent=1)
    print(f"summary -> {OUTDIR / 'homotopy_fire_summary.json'}")


if __name__ == "__main__":
    main()
