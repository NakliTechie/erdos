#!/usr/bin/env python
"""EXPERIMENT 1 — rotate-and-project HOMOTOPY LOCKING (plan/hinge-design.md (a)).

Subject: data/udg40_132edges.csv (40 pts, 132 audited unit edges).
Floating families after ML alignment: fam1 ~0.94 deg (6 edges, 9 verts,
target 0 deg) and fam3 ~34.12 deg (52 edges, 39/40 verts, target
arccos(5/6) = 33.5573 deg).

Arms:
  f094            lock fam1 alone
  f3412           lock fam3 alone
  seq094first     fam1 then re-classify then fam3-equivalent
  seq3412first    fam3 then re-classify then fam1-equivalent

Per lock outcome: GN polish on the CURRENT unit-edge set, fire_check,
non-edge near-miss histogram (delta vs original), full three-audit gate,
re-classification (relative family offsets — catches gauge-only "locks"),
save >=132-audited configs; then low-temperature warm-start search
(P0=locked, T0=0.08, T1=0.01, 30k steps, seeds 0-3, max 4 processes),
audit every search best, save >=132-audited.

Stall protocol: n_increments=10 first, then 20. The design's alternative
hinge center (family centroid) is SKIPPED with reason: for both floating
families hinge_verts == fam_verts (every family vertex touches non-family
edges), so the two centers coincide exactly.

DISCIPLINE: no count is claimed without udg.audit.audit().passed; watch
min_sep_after (point-merge exploit through the repair step).

Output: runs/hinge/homotopy_summary.json + homotopy_*.csv/_audit.json.
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
PROCESSES = 4  # hard cap — a concurrent campaign may be running
BANDS = [1e-9, 1e-6, 1e-4, 1e-2, 5e-2]


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------

def near_miss_diagnostic(P: np.ndarray, k: int = 8) -> dict:
    """|d-1| histogram over NON-edge pairs + the k nearest misses."""
    n = len(P)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    iu, ju = np.triu_indices(n, 1)
    dev = np.abs(D[iu, ju] - 1.0)
    edge_mask = dev < 1e-9
    nonedge = dev[~edge_mask]
    bands = {
        f"[{lo:g},{hi:g})": int(((nonedge >= lo) & (nonedge < hi)).sum())
        for lo, hi in zip(BANDS[:-1], BANDS[1:])
    }
    order = np.argsort(nonedge)
    ii, jj = iu[~edge_mask][order[:k]], ju[~edge_mask][order[:k]]
    nearest = [
        {"i": int(a), "j": int(b), "abs_d_minus_1": float(np.abs(D[a, b] - 1.0))}
        for a, b in zip(ii, jj)
    ]
    return {
        "n_edges_tol_1e-9": int(edge_mask.sum()),
        "bands_nonedge": bands,
        "nearest": nearest,
        "min_nonedge_abs_d_minus_1": float(nonedge.min()),
    }


def audit_dict(rep) -> dict:
    return dataclasses.asdict(rep)


def reclassify(P: np.ndarray) -> tuple[object, list[dict]]:
    """Re-classify on the CURRENT unit-edge set; relative offsets are the
    gauge-proof measure of whether a family genuinely locked."""
    E = unit_edges(P)
    fc = classify_families(P, E)
    rows = [
        {
            "index": f.index,
            "n_edges": f.n_edges,
            "mean_angle_aligned": f.mean_angle,
            "target": f.target,
            "offset": f.offset,
            "locked": f.locked,
        }
        for f in fc.families
    ]
    return fc, rows


def find_family(fc, ref_edges) -> object:
    """Family in fc with max edge overlap with ref_edges (frozenset match)."""
    ref = {tuple(sorted(e)) for e in ref_edges}
    return max(
        fc.families,
        key=lambda f: len(ref & {tuple(sorted(e)) for e in f.edges}),
    )


def lock_summary(res, attempt_label: str) -> dict:
    h = res.diagnostics["history"]
    return {
        "attempt": attempt_label,
        "converged_absolute": bool(res.converged),
        "stop_reason": res.stop_reason,
        "final_family_angle": float(res.family_angle),
        "target": res.diagnostics["target"],
        "abs_remaining_to_target": abs(
            signed_delta(res.diagnostics["target"], res.family_angle)
        ),
        "initial_angle": res.diagnostics["initial_angle"],
        "residual": float(res.residual),
        "min_sep": float(res.min_sep),
        "n_accepted_increments": res.diagnostics["n_steps"],
        "n_halvings": res.diagnostics["n_halvings"],
        "history_tail": [
            {k: float(v) for k, v in row.items()} for row in h[-3:]
        ],
    }


# ---------------------------------------------------------------------------
# lock with stall ladder
# ---------------------------------------------------------------------------

def attempt_lock(P, E, fam, target, log_prefix: str) -> tuple[object, list[dict]]:
    """lock_family with the design's stall ladder.

    n_increments=10, then 20 on non-convergence. The alternative hinge
    center is documented as a no-op here (hinge_verts == fam_verts for both
    floating families — centers coincide), so it is skipped.
    """
    attempts = []
    best = None
    for n_inc in (10, 20):
        t0 = time.time()
        res = lock_family(P, E, fam, target=target, n_increments=n_inc)
        s = lock_summary(res, f"hinge_center,n_increments={n_inc}")
        s["seconds"] = round(time.time() - t0, 2)
        attempts.append(s)
        print(
            f"  [{log_prefix}] n_inc={n_inc}: {res.stop_reason} "
            f"angle={res.family_angle:.5f} -> target={target:.5f} "
            f"(rem {s['abs_remaining_to_target']:.5f}) res={res.residual:.3e} "
            f"min_sep={res.min_sep:.4f} steps={s['n_accepted_increments']} "
            f"halvings={s['n_halvings']} {s['seconds']}s"
        )
        if best is None or (
            s["abs_remaining_to_target"]
            < abs(signed_delta(target, best.family_angle))
        ):
            best = res
        if res.converged:
            break
    return best, attempts


# ---------------------------------------------------------------------------
# per-outcome battery: polish, fire, audit, reclassify, save, search
# ---------------------------------------------------------------------------

def battery(P_locked: np.ndarray, arm: str, baseline_nm: dict) -> dict:
    out: dict = {}

    # polish on the CURRENT unit-edge set (keeps any fired edges exact)
    E_cur = unit_edges(P_locked)
    Q, res = gauss_newton(P_locked, E_cur, iters=30_000)
    ms = min_separation(Q)
    if ms < MIN_SEP or edge_residual(Q, E_cur) > edge_residual(P_locked, E_cur):
        Q = P_locked  # polish made it worse / merged points: keep unpolished
    out["polish_residual"] = float(edge_residual(Q, E_cur))
    out["polish_min_sep"] = float(min_separation(Q))

    fc_check = fire_check(Q)
    out["fire_check"] = {
        "n_unit": fc_check.n_unit,
        "bands": [[lo, hi, c] for lo, hi, c in fc_check.bands],
        "closest_nonunit": fc_check.closest_nonunit,
    }
    nm = near_miss_diagnostic(Q)
    out["near_miss"] = nm
    out["near_miss_delta_vs_original"] = {
        band: nm["bands_nonedge"][band] - baseline_nm["bands_nonedge"][band]
        for band in nm["bands_nonedge"]
    }

    rep = audit(Q)
    out["audit"] = audit_dict(rep)
    print(
        f"  [{arm}] locked-config audit: n_edges={rep.n_edges} passed={rep.passed} "
        f"min_sep={rep.min_sep:.4f} min_sep_after={rep.min_sep_after:.4f} "
        f"gn_res={rep.gn_total_residual:.3e}"
    )

    _, rows = reclassify(Q)
    out["reclassified_families"] = rows
    out["flex_dimension_after"] = int(flex_dimension(Q, unit_edges(Q)))

    saved = []
    if rep.passed and rep.n_edges >= 132:
        csv = OUTDIR / f"homotopy_{arm}_locked.csv"
        save_csv(Q, csv)
        save_audit_json(rep, OUTDIR / f"homotopy_{arm}_locked_audit.json")
        saved.append(str(csv))
    out["locked_saved"] = saved

    # ---- low-temperature warm-start search ----
    t0 = time.time()
    results = multi_search(
        len(Q), SEEDS, STEPS, processes=PROCESSES, T0=T0, T1=T1, P0=Q
    )
    out["search_seconds"] = round(time.time() - t0, 1)
    runs = []
    best_audited = 0
    for r in results:
        rrep = audit(r.P)
        row = {
            "seed": r.seed,
            "best_count_tol": r.best_count,
            "audited_n_edges": rrep.n_edges,
            "passed": rrep.passed,
            "min_sep": rrep.min_sep,
            "min_sep_after": rrep.min_sep_after,
            "gn_total_residual": rrep.gn_total_residual,
            "identical_to_warmstart": bool(np.array_equal(r.P, Q)),
        }
        if rrep.passed and rrep.n_edges >= 132:
            csv = OUTDIR / f"homotopy_{arm}_search_s{r.seed}.csv"
            save_csv(r.P, csv)
            save_audit_json(rrep, OUTDIR / f"homotopy_{arm}_search_s{r.seed}_audit.json")
            row["saved"] = str(csv)
            saved.append(str(csv))
            best_audited = max(best_audited, rrep.n_edges)
        runs.append(row)
        print(
            f"  [{arm}] search s{r.seed}: tol_count={r.best_count} "
            f"audited={rrep.n_edges} passed={rrep.passed} "
            f"identical={row['identical_to_warmstart']}"
        )
    out["search_runs"] = runs
    out["post_search_best_audited"] = best_audited
    out["P_final"] = Q  # for sequential arms (stripped before JSON dump)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    P0 = load_csv(SUBJECT)
    E0 = unit_edges(P0)
    base_rep = audit(P0)
    assert base_rep.passed and base_rep.n_edges == 132, "subject audit failed"
    baseline_nm = near_miss_diagnostic(P0)

    fc0 = classify_families(P0, E0)
    Pa = fc0.P  # ML-aligned frame; all locking runs in this frame
    Ea = unit_edges(Pa)
    assert len(Ea) == 132, "alignment must preserve edges"
    floating = [f for f in fc0.families if not f.locked]
    assert len(floating) == 2
    f094 = min(floating, key=lambda f: abs(f.mean_angle_raw - 0.94))
    f3412 = min(floating, key=lambda f: abs(f.mean_angle_raw - 34.12))

    summary: dict = {
        "experiment": "EXPERIMENT 1 — rotate-and-project homotopy locking (strategy a)",
        "subject": str(SUBJECT),
        "baseline_audit": audit_dict(base_rep),
        "baseline_near_miss": baseline_nm,
        "flex_dimension_input": int(flex_dimension(Pa, Ea)),
        "alt_center_note": (
            "alternative hinge center (family centroid) skipped: "
            "hinge_verts == fam_verts for both floating families, centers coincide"
        ),
        "families_input": [
            {
                "index": f.index,
                "n_edges": f.n_edges,
                "n_vertices": len(f.vertices),
                "mean_angle_raw": f.mean_angle_raw,
                "mean_angle_aligned": f.mean_angle,
                "target": f.target,
                "offset": f.offset,
                "locked": f.locked,
            }
            for f in fc0.families
        ],
        "search_params": {"T0": T0, "T1": T1, "steps": STEPS, "seeds": SEEDS},
        "arms": {},
    }
    print(f"input flex dimension: {summary['flex_dimension_input']}")

    arm_finals: dict[str, np.ndarray] = {}

    # ---- arm f094: lock the ~0.94 family alone ----
    print("\n=== arm f094: lock ~0.94deg family -> 0deg ===")
    best, attempts = attempt_lock(Pa, Ea, f094, f094.target, "f094")
    arm = battery(best.P, "f094", baseline_nm)
    arm["lock_attempts"] = attempts
    arm_finals["f094"] = arm.pop("P_final")
    summary["arms"]["f094"] = arm

    # ---- arm f3412: lock the ~34.12 family alone ----
    print("\n=== arm f3412: lock ~34.12deg family -> 33.5573deg ===")
    best3, attempts3 = attempt_lock(Pa, Ea, f3412, f3412.target, "f3412")
    arm3 = battery(best3.P, "f3412", baseline_nm)
    arm3["lock_attempts"] = attempts3
    arm_finals["f3412"] = arm3.pop("P_final")
    summary["arms"]["f3412"] = arm3

    # ---- sequential arms ----
    for arm_name, first_key, second_ref in (
        ("seq094first", "f094", f3412.edges),
        ("seq3412first", "f3412", f094.edges),
    ):
        print(f"\n=== arm {arm_name}: second lock after {first_key} ===")
        Pc = arm_finals[first_key]
        Ec = unit_edges(Pc)
        fc, rows = reclassify(Pc)
        fam2 = find_family(fc, second_ref)
        print(
            f"  second-lock family: idx={fam2.index} n_edges={fam2.n_edges} "
            f"aligned={fam2.mean_angle:.5f} target={fam2.target:.5f} "
            f"offset={fam2.offset:+.5f} locked={fam2.locked}"
        )
        block: dict = {
            "reclass_before_second_lock": rows,
            "second_family": {
                "n_edges": fam2.n_edges,
                "mean_angle_aligned": fam2.mean_angle,
                "target": fam2.target,
                "offset": fam2.offset,
                "already_locked": fam2.locked,
            },
        }
        # lock in the re-aligned frame fc.P (targets meaningful there)
        Pcf = fc.P
        Ecf = unit_edges(Pcf)
        bests, attemptss = attempt_lock(Pcf, Ecf, fam2, fam2.target, arm_name)
        arm_b = battery(bests.P, arm_name, baseline_nm)
        arm_b["lock_attempts"] = attemptss
        arm_finals[arm_name] = arm_b.pop("P_final")
        block.update(arm_b)
        summary["arms"][arm_name] = block

    # ---- roll-up ----
    best_overall = 132
    for name, arm in summary["arms"].items():
        a = arm["audit"]
        if a["passed"]:
            best_overall = max(best_overall, a["n_edges"])
        best_overall = max(best_overall, arm["post_search_best_audited"])
    summary["best_audited_overall"] = best_overall

    with open(OUTDIR / "homotopy_summary.json", "w") as f:
        json.dump(_clean(summary), f, indent=1)
    print(f"\nbest audited overall: {best_overall}")
    print(f"summary -> {OUTDIR / 'homotopy_summary.json'}")


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


if __name__ == "__main__":
    main()
