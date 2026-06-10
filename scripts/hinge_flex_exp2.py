"""EXPERIMENT 2 — flex analysis + flex following (plan/hinge-design.md strategy b).

Protocol on data/udg40_132edges.csv:
1. MEASURE flex_dimension (nullity of rigidity matrix - 3) + SV spectrum near
   the threshold (is the cutoff clean?).
2. If dim >= 1: follow_flex toward each floating family's target angle,
   tracking family angle vs config-space arc-length (does the flex couple?).
   fire_check + audit at the end; then low-T warm-start search
   (T0=0.08, T1=0.01, 30k steps, 4 seeds, 4 processes) and audit.
3. Characterize the flex space: per-vertex motion in the top-3 (quasi-)null
   modes vs the floating-family vertex sets; per-family angular velocity
   coupling |B^T c_F|.

Saves >=132 AUDITED configs to runs/hinge/flex_*.csv (+ audit JSONs) and a
machine-readable summary to runs/hinge/flex_exp2_summary.json.

DISCIPLINE: no count claimed without udg.audit.audit().passed; min_sep_after
is checked (point-merging through the repair step = tolerance exploit).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from udg.audit import audit
from udg.configio import load_csv, save_audit_json, save_csv
from udg.counting import unit_edges
from udg.hinge import (
    SVD_TOL,
    _family_angle_gradient,
    _trivial_motions,
    classify_families,
    family_angle,
    fire_check,
    flex_dimension,
    follow_flex,
    internal_flex_basis,
    rigidity_matrix,
    signed_delta,
)
from udg.search import multi_search

ROOT = Path("/Users/chiragpatnaik/Code/erdos")
OUT = ROOT / "runs" / "hinge"
OUT.mkdir(parents=True, exist_ok=True)

summary: dict = {"experiment": "EXP2 flex analysis + flex following"}


def jfloat(x):
    return float(x)


def audit_and_maybe_save(P, stem: str) -> dict:
    """Audit P; save CSV + audit JSON under runs/hinge if passed and >=132."""
    rep = audit(P)
    rec = {
        "n_edges": int(rep.n_edges),
        "passed": bool(rep.passed),
        "min_sep": jfloat(rep.min_sep),
        "min_sep_after": jfloat(rep.min_sep_after),
        "k23_violations": int(rep.k23_violations),
        "gn_total_residual": jfloat(rep.gn_total_residual),
        "gn_edges_exact": int(rep.gn_edges_exact),
        "saved": None,
    }
    if rep.passed and rep.n_edges >= 132:
        csv_path = OUT / f"{stem}.csv"
        save_csv(P, csv_path)
        save_audit_json(rep, OUT / f"{stem}_audit.json")
        rec["saved"] = str(csv_path)
    return rec


def main() -> None:
    # ---------------------------------------------------------------- load + align
    P_raw = load_csv(ROOT / "data" / "udg40_132edges.csv")
    E = unit_edges(P_raw)
    assert len(E) == 132, f"expected 132 edges, got {len(E)}"

    cls = classify_families(P_raw, E)
    P = cls.P  # ML-aligned frame (rigid rotation; edges preserved)
    fams = cls.families
    summary["families"] = [
        {
            "index": f.index,
            "n_edges": f.n_edges,
            "n_vertices": len(f.vertices),
            "mean_angle_raw": jfloat(f.mean_angle_raw),
            "mean_angle_aligned": jfloat(f.mean_angle),
            "target": jfloat(f.target),
            "offset": jfloat(f.offset),
            "locked": bool(f.locked),
        }
        for f in fams
    ]

    # ------------------------------------------------------- 1. MEASURE flex dim
    R = rigidity_matrix(P, E)
    s = np.linalg.svd(R, compute_uv=False)  # 80 singular values, descending
    thresh = SVD_TOL * max(1.0, float(s[0]))
    rank = int((s > thresh).sum())
    nullity = R.shape[1] - rank
    fd = flex_dimension(P, E)
    B = internal_flex_basis(P, E)
    gap_ratio = float(s[rank - 1] / s[rank]) if rank < len(s) and s[rank] > 0 else float("inf")
    summary["measure"] = {
        "rigidity_matrix_shape": list(R.shape),
        "svd_threshold": jfloat(thresh),
        "rank": rank,
        "nullity": nullity,
        "flex_dimension": fd,
        "internal_flex_basis_dim": int(B.shape[1]),
        "smallest_12_singular_values": [jfloat(x) for x in s[-12:]],
        "largest_singular_value": jfloat(s[0]),
        "cutoff_gap_ratio_s[rank-1]/s[rank]": gap_ratio,
    }
    print("== MEASURE ==")
    print(f"R shape {R.shape}, rank {rank}, nullity {nullity}, flex_dim {fd} "
          f"(basis cross-check {B.shape[1]})")
    print("smallest 12 SVs:", " ".join(f"{x:.3e}" for x in s[-12:]))
    print(f"threshold {thresh:.3e}, gap ratio across cutoff {gap_ratio:.3e}")

    # ----------------------------------------- 3. CHARACTERIZE flex / quasi-flex
    # Order the (quasi-)null modes of the stacked matrix [R; trivial] by ascending
    # singular value; the genuinely-null ones (s < thresh) are true internal
    # flexes, the rest are "quasi-flex" (softest constrained) modes.
    A = np.vstack([R, _trivial_motions(P)])
    _, sA, vh = np.linalg.svd(A)
    rankA = int((sA > SVD_TOL * max(1.0, float(sA[0]))).sum())
    order = np.argsort(sA)  # ascending
    modes = []
    float_idx = [f.index for f in fams if not f.locked]
    for rank_pos, k in enumerate(order[:3]):
        v = vh[k]
        Vp = v.reshape(-1, 2)
        mag = np.sqrt((Vp * Vp).sum(1))
        top = np.argsort(mag)[::-1][:8]
        fam_stats = {}
        for f in fams:
            inset = np.zeros(len(P), dtype=bool)
            inset[f.vertices] = True
            c = _family_angle_gradient(P, f.edges)
            fam_stats[f.index] = {
                "mean_motion_members": jfloat(mag[inset].mean()),
                "mean_motion_nonmembers": jfloat(mag[~inset].mean()) if (~inset).any() else None,
                "angular_velocity_deg_per_arc": jfloat(np.degrees(c @ v)),
            }
        modes.append(
            {
                "mode_rank": rank_pos,
                "singular_value": jfloat(sA[k]),
                "is_true_flex": bool(sA[k] <= SVD_TOL * max(1.0, float(sA[0]))),
                "top8_vertices": [int(i) for i in top],
                "top8_motion": [jfloat(mag[i]) for i in top],
                "per_family": fam_stats,
            }
        )
    summary["characterize"] = {
        "n_true_internal_flexes": int(B.shape[1]),
        "modes": modes,
    }
    # Whole-null-space coupling per family: max |dtheta/ds| achievable = |B^T c|.
    coupling = {}
    for f in fams:
        c = _family_angle_gradient(P, f.edges)
        w = B.T @ c if B.shape[1] else np.zeros(0)
        coupling[f.index] = jfloat(np.degrees(np.linalg.norm(w))) if w.size else 0.0
    summary["characterize"]["family_max_angular_velocity_deg_per_arc"] = coupling

    print("\n== CHARACTERIZE (top-3 smallest-SV modes of [R; trivial]) ==")
    for m in modes:
        print(f"mode {m['mode_rank']}: s={m['singular_value']:.3e} "
              f"true_flex={m['is_true_flex']} top verts {m['top8_vertices']}")
        for fi, st in m["per_family"].items():
            tag = "FLOAT" if fi in float_idx else "lock "
            print(f"   fam{fi} {tag} mean|v| in/out = {st['mean_motion_members']:.4f}/"
                  f"{st['mean_motion_nonmembers']:.4f}  dtheta/ds = "
                  f"{st['angular_velocity_deg_per_arc']:+.4e} deg/arc")
    print("max |dtheta/ds| over true null space per family:", coupling)

    # --------------------------------------------------- 2. FOLLOW (if dim >= 1)
    summary["follow"] = []
    if fd >= 1:
        for f in [f for f in fams if not f.locked]:
            print(f"\n== FOLLOW fam{f.index}: {f.mean_angle:.4f} -> {f.target:.4f} deg ==")
            res = follow_flex(P, E, f, step_deg=0.1, max_steps=2000)
            hist = res.diagnostics["history"]
            arc = np.cumsum([abs(h["eps"]) for h in hist]) if hist else np.array([])
            angs = [h["angle"] for h in hist]
            fc = fire_check(res.P)
            rec = {
                "family": f.index,
                "target": jfloat(f.target),
                "stop_reason": res.stop_reason,
                "converged": bool(res.converged),
                "start_angle": jfloat(f.mean_angle),
                "final_angle": jfloat(res.family_angle),
                "residual": jfloat(res.residual),
                "min_sep": jfloat(res.min_sep),
                "n_steps": len(hist),
                "total_arc_length": jfloat(arc[-1]) if len(arc) else 0.0,
                "angle_vs_arc_first5": [
                    [jfloat(a), jfloat(t)] for a, t in zip(arc[:5], angs[:5])
                ],
                "angle_vs_arc_last5": [
                    [jfloat(a), jfloat(t)] for a, t in zip(arc[-5:], angs[-5:])
                ],
                "speeds_deg_per_arc": [
                    jfloat(np.degrees(h["speed"])) for h in hist[:: max(1, len(hist) // 10)]
                ],
                "flex_dims_along_path": sorted({int(h["flex_dim"]) for h in hist}),
                "fire_check": {
                    "n_unit": fc.n_unit,
                    "bands": [[jfloat(a), jfloat(b), int(c)] for a, b, c in fc.bands],
                    "closest_nonunit": jfloat(fc.closest_nonunit),
                },
            }
            print(f"stop={res.stop_reason} angle {f.mean_angle:.4f}->{res.family_angle:.4f} "
                  f"target {f.target:.4f}, steps={len(hist)}, arc={rec['total_arc_length']:.4f}, "
                  f"residual={res.residual:.2e}, min_sep={res.min_sep:.3f}")
            print(f"fire_check: n_unit={fc.n_unit}, bands={fc.bands}, "
                  f"closest={fc.closest_nonunit:.4e}")
            rec["audit"] = audit_and_maybe_save(res.P, f"flex_fam{f.index}_followed")
            print("audit:", rec["audit"])

            # low-T warm-start search from the followed config (4 seeds, 4 procs)
            if rec["audit"]["passed"]:
                seeds = [201, 202, 203, 204]
                results = multi_search(
                    40, seeds, 30_000, processes=4,
                    P0=res.P, T0=0.08, T1=0.01,
                )
                rec["search"] = []
                for sr in results:
                    a = audit_and_maybe_save(
                        sr.P, f"flex_fam{f.index}_search_seed{sr.seed}"
                    )
                    rec["search"].append(
                        {"seed": sr.seed, "best_count_raw": int(sr.best_count), "audit": a}
                    )
                    print(f"seed {sr.seed}: raw best {sr.best_count}, audit {a}")
            summary["follow"].append(rec)
    else:
        print("\nflex_dimension == 0: framework is first-order RIGID; "
              "skipping follow_flex (this is the finding).")
        summary["follow"] = "skipped (flex_dimension == 0)"

    with open(OUT / "flex_exp2_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\nsummary ->", OUT / "flex_exp2_summary.json")


if __name__ == "__main__":
    main()
