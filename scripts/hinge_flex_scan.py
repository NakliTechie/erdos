"""EXPERIMENT 2b — full 1-D flex-path scan of the 132-edge n=40 config.

The skeleton has exactly ONE internal flex (hinge_flex_exp2.py). Therefore the
entire reachable configuration space (with all 132 edges held exact) is a
1-parameter curve. This script:

1. Walks the curve in both directions (predictor = flex vector with direction
   continuity, corrector = Gauss-Newton on all 132 edges) until flex death /
   stall / min-sep floor, recording at every step: arc-length s, the four
   family angles, residual, min-sep, flex dim, and ALL 648 non-edge pair
   distances.
2. Detects any non-edge pair whose distance crosses 1 along the path
   (sign change of d-1), bisects to the crossing, force-projects (GN on
   132+1 edges) and AUDITS — a passing 133 would beat the plateau.
   Near-tangencies (min |d-1| < 1e-3 without sign change) are also tried.
3. Finds the gauge-corrected simultaneous-lock point: s* minimizing
   min_g sum_i w_i (offset_i(s) + g)^2 (g = global-rotation gauge,
   w_i = family edge counts). Saves + audits that config; fire-check;
   then a low-T warm-start search (T0=0.08, T1=0.01, 30k, 4 seeds, 4 procs).

Outputs: runs/hinge/flex_scan_summary.json, runs/hinge/flex_scan_path.npz
(s, angles, distances — for later analysis), audited configs as
runs/hinge/flex_all_locked.csv etc.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from udg.audit import MIN_SEP, audit, gauss_newton, min_separation
from udg.configio import load_csv, save_audit_json, save_csv
from udg.counting import unit_edges
from udg.hinge import (
    STALL_RESIDUAL,
    _family_angle_gradient,
    classify_families,
    edge_residual,
    family_angle,
    fire_check,
    internal_flex_basis,
    signed_delta,
)
from udg.search import multi_search

ROOT = Path("/Users/chiragpatnaik/Code/erdos")
OUT = ROOT / "runs" / "hinge"

DS = 0.01          # base arc-length step
MAX_STEPS = 400    # per direction
GN_ITERS = 8000


def audit_and_maybe_save(P, stem: str) -> dict:
    rep = audit(P)
    rec = {
        "n_edges": int(rep.n_edges),
        "passed": bool(rep.passed),
        "min_sep": float(rep.min_sep),
        "min_sep_after": float(rep.min_sep_after),
        "k23_violations": int(rep.k23_violations),
        "gn_total_residual": float(rep.gn_total_residual),
        "gn_edges_exact": int(rep.gn_edges_exact),
        "saved": None,
    }
    if rep.passed and rep.n_edges >= 132:
        save_csv(P, OUT / f"{stem}.csv")
        save_audit_json(rep, OUT / f"{stem}_audit.json")
        rec["saved"] = str(OUT / f"{stem}.csv")
    return rec


def flex_step(x, v_prev, ds, E):
    """One predictor-corrector step of arc-length ds along the flex.

    Returns (y, v_used) or (None, reason). Direction continuity: project
    v_prev onto the current flex space.
    """
    B = internal_flex_basis(x, E)
    if B.shape[1] == 0:
        return None, "flex_death_rigid"
    w = B.T @ v_prev
    nw = float(np.linalg.norm(w))
    if nw < 1e-8:
        return None, "flex_death_orthogonal"
    v = (B @ w) / nw
    y, _ = gauss_newton(x + ds * v.reshape(-1, 2), E, iters=GN_ITERS)
    r = edge_residual(y, E)
    ms = min_separation(y)
    if r > STALL_RESIDUAL or ms < MIN_SEP:
        return None, f"stall(res={r:.2e},min_sep={ms:.3f})"
    return y, v


def walk(x0, v0, E, fams, ds=DS, max_steps=MAX_STEPS):
    """Walk one direction. Returns list of records (s>0 increasing)."""
    recs = []
    x, v, s = x0, v0, 0.0
    step = ds
    halv = 0
    reason = "max_steps"
    while len(recs) < max_steps:
        y, vy = flex_step(x, v, step, E)
        if y is None:
            halv += 1
            if halv > 3:
                reason = vy if isinstance(vy, str) else "stalled"
                break
            step /= 2.0
            continue
        halv = 0
        x, v = y, vy
        s += step
        recs.append(
            {
                "s": s,
                "P": x.copy(),
                "angles": [family_angle(x, f.edges) for f in fams],
                "min_sep": min_separation(x),
                "residual": edge_residual(x, E),
                "flex_dim": internal_flex_basis(x, E).shape[1],
            }
        )
        if step < ds:
            step = min(ds, step * 2.0)
    return recs, reason


def nonedge_pairs(n, E):
    eset = {tuple(sorted(e)) for e in E}
    return [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in eset]


def pair_dists(P, pairs):
    A = np.asarray([P[i] for i, j in pairs])
    B = np.asarray([P[j] for i, j in pairs])
    return np.sqrt(((A - B) ** 2).sum(1))


def gauge_offset_ss(angles, fams):
    """min over gauge g of sum_i w_i*(delta_i+g)^2, and the offsets at g*."""
    d = np.array([signed_delta(a, f.target) for a, f in zip(angles, fams)])
    w = np.array([f.n_edges for f in fams], dtype=float)
    g = -(w * d).sum() / w.sum()
    return float((w * (d + g) ** 2).sum()), (d + g).tolist(), float(g)


def refine_crossing(x_lo, v_lo, h, pair, E, n_bisect=45):
    """Bisect along the flex from x_lo over arc h for sign change of d-1."""
    def f(P):
        return float(np.linalg.norm(P[pair[0]] - P[pair[1]]) - 1.0)

    lo, vlo, flo = x_lo, v_lo, f(x_lo)
    for _ in range(n_bisect):
        h /= 2.0
        if h < 1e-14:
            break
        mid, vmid = flex_step(lo, vlo, h, E)
        if mid is None:
            break
        if np.sign(f(mid)) == np.sign(flo):
            lo, vlo, flo = mid, vmid, f(mid)
    return lo, abs(flo)


def main() -> None:
    summary: dict = {"experiment": "EXP2b 1-D flex path scan"}

    P_raw = load_csv(ROOT / "data" / "udg40_132edges.csv")
    E = unit_edges(P_raw)
    assert len(E) == 132
    cls = classify_families(P_raw, E)
    P0, fams = cls.P, cls.families
    pairs = nonedge_pairs(len(P0), E)

    # orientation: +s = fam1 (floating 0.9-deg family) angle increasing
    B0 = internal_flex_basis(P0, E)
    assert B0.shape[1] == 1
    v0 = B0[:, 0]
    c1 = _family_angle_gradient(P0, fams[1].edges)
    if c1 @ v0 < 0:
        v0 = -v0

    rec0 = {
        "s": 0.0,
        "P": P0.copy(),
        "angles": [family_angle(P0, f.edges) for f in fams],
        "min_sep": min_separation(P0),
        "residual": edge_residual(P0, E),
        "flex_dim": 1,
    }
    plus, reason_p = walk(P0, v0, E, fams)
    minus, reason_m = walk(P0, -v0, E, fams)
    for r in minus:
        r["s"] = -r["s"]
    path = list(reversed(minus)) + [rec0] + plus
    svals = np.array([r["s"] for r in path])
    angles = np.array([r["angles"] for r in path])
    D = np.vstack([pair_dists(r["P"], pairs) for r in path])  # (T, 648)

    summary["walk"] = {
        "s_range": [float(svals[0]), float(svals[-1])],
        "n_points": len(path),
        "stop_reason_plus": reason_p,
        "stop_reason_minus": reason_m,
        "min_sep_range": [
            float(min(r["min_sep"] for r in path)),
            float(max(r["min_sep"] for r in path)),
        ],
        "flex_dim_values_seen": sorted({r["flex_dim"] for r in path}),
        "dtheta_ds_at_0_deg_per_arc": [
            float(x)
            for x in (angles[len(minus) + 1] - angles[len(minus) - 1])
            / (svals[len(minus) + 1] - svals[len(minus) - 1])
        ],
    }
    print("== WALK ==")
    print(f"s in [{svals[0]:.3f}, {svals[-1]:.3f}], {len(path)} pts; "
          f"stop +: {reason_p}; stop -: {reason_m}")
    print("min_sep range", summary["walk"]["min_sep_range"],
          "flex dims seen", summary["walk"]["flex_dim_values_seen"])
    print("dtheta/ds at s=0 (deg/arc):",
          ["%.3f" % x for x in summary["walk"]["dtheta_ds_at_0_deg_per_arc"]])

    np.savez_compressed(
        OUT / "flex_scan_path.npz",
        s=svals, angles=angles, dists=D,
        pairs=np.array(pairs), min_sep=np.array([r["min_sep"] for r in path]),
    )

    # ---- crossings of d = 1 along the path ------------------------------
    X = D - 1.0
    crossings = []
    for col in range(X.shape[1]):
        sgn = np.sign(X[:, col])
        idx = np.where(sgn[:-1] * sgn[1:] < 0)[0]
        for k in idx:
            crossings.append((col, int(k)))
    # near-tangency candidates: min |d-1| < 1e-3 without a crossing
    tang = [
        (col, int(np.argmin(np.abs(X[:, col]))))
        for col in range(X.shape[1])
        if np.abs(X[:, col]).min() < 1e-3
        and not any(c == col for c, _ in crossings)
    ]
    print(f"\n== CROSSINGS == {len(crossings)} sign changes, "
          f"{len(tang)} near-tangencies (<1e-3)")
    summary["crossings"] = []
    tried = set()
    for col, k in crossings + tang:
        pair = pairs[col]
        key = (col, k // 5)
        if key in tried:
            continue
        tried.add(key)
        x_lo = path[k]["P"]
        k2 = min(k + 1, len(path) - 1)
        h = abs(float(svals[k2] - svals[k])) or DS
        # local direction hint: secant toward the next path point
        vdir = (path[k2]["P"] - x_lo).ravel()
        nv = np.linalg.norm(vdir)
        if nv < 1e-12:
            continue
        vdir = vdir / nv
        x_ref, gap = refine_crossing(x_lo, vdir, h, pair, E)
        fc = fire_check(x_ref)
        # force-project the candidate 133rd edge and audit
        Q, _ = gauss_newton(x_ref, E + [pair], iters=GN_ITERS)
        arec = audit_and_maybe_save(Q, f"flex_cross_p{pair[0]}_{pair[1]}")
        crec = {
            "pair": list(pair),
            "near_s": float(svals[k]),
            "refined_gap": float(gap),
            "fire_n_unit": fc.n_unit,
            "forced_audit": arec,
        }
        summary["crossings"].append(crec)
        print(f"pair {pair} near s={svals[k]:+.3f}: refined |d-1|={gap:.2e}, "
              f"fire n_unit={fc.n_unit}, forced audit n={arec['n_edges']} "
              f"passed={arec['passed']}")

    # ---- gauge-corrected simultaneous lock point ------------------------
    ss = np.array([gauge_offset_ss(a, fams)[0] for a in angles])
    kbest = int(np.argmin(ss))
    # refine: fine walk around kbest
    x_best, s_best = path[kbest]["P"], float(svals[kbest])
    best_ss = float(ss[kbest])
    # local tangent hint at kbest (secant toward increasing s)
    klo, khi = max(kbest - 1, 0), min(kbest + 1, len(path) - 1)
    v_hint = (path[khi]["P"] - path[klo]["P"]).ravel()
    v_hint = v_hint / np.linalg.norm(v_hint)
    for fine in (DS / 10, DS / 100):
        improved = True
        while improved:
            improved = False
            for sgn in (+1.0, -1.0):
                vdir = v_hint * sgn
                y, _ = flex_step(x_best, vdir, fine, E)
                if y is None:
                    continue
                val, _, _ = gauge_offset_ss(
                    [family_angle(y, f.edges) for f in fams], fams
                )
                if val < best_ss:
                    x_best, best_ss, s_best = y, val, s_best + sgn * fine
                    improved = True
                    break
    val, offs, g = gauge_offset_ss(
        [family_angle(x_best, f.edges) for f in fams], fams
    )
    fc = fire_check(x_best)
    arec = audit_and_maybe_save(x_best, "flex_all_locked")
    summary["all_locked"] = {
        "s_star": s_best,
        "weighted_ss_deg2": val,
        "gauge_deg": g,
        "offsets_after_gauge_deg": offs,
        "fire_check": {
            "n_unit": fc.n_unit,
            "bands": [[a, b, c] for a, b, c in fc.bands],
            "closest_nonunit": fc.closest_nonunit,
        },
        "audit": arec,
    }
    print(f"\n== ALL-LOCK POINT == s*={s_best:+.5f}, weighted SS={val:.3e} deg^2")
    print("offsets after gauge (deg):", ["%+.4f" % o for o in offs])
    print(f"fire n_unit={fc.n_unit} bands={fc.bands} closest={fc.closest_nonunit:.4e}")
    print("audit:", arec)

    # warm-start search from the all-locked config
    if arec["passed"]:
        res = multi_search(40, [301, 302, 303, 304], 30_000, processes=4,
                           P0=x_best, T0=0.08, T1=0.01)
        summary["all_locked"]["search"] = []
        for sr in res:
            a = audit_and_maybe_save(sr.P, f"flex_all_locked_search_seed{sr.seed}")
            if a["saved"] and Path(a["saved"]).read_bytes() == (
                OUT / "flex_all_locked.csv"
            ).read_bytes():
                Path(a["saved"]).unlink()
                Path(a["saved"].replace(".csv", "_audit.json")).unlink()
                a["saved"] = "identical to flex_all_locked.csv (removed)"
            summary["all_locked"]["search"].append(
                {"seed": sr.seed, "best_count_raw": int(sr.best_count), "audit": a}
            )
            print(f"seed {sr.seed}: raw {sr.best_count}, audit n={a['n_edges']} "
                  f"passed={a['passed']} saved={a['saved']}")

    with open(OUT / "flex_scan_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nsummary ->", OUT / "flex_scan_summary.json")


if __name__ == "__main__":
    main()
