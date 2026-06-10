"""Coincidence forcing: greedy near-miss constraint augmentation.

The constraint-side dual of hinge locking (HANDOFF §5.A.1): instead of rotating
a floating family to an exact angle and hoping edges fire, take the near-miss
pairs themselves as new unit constraints and let damped Gauss–Newton find the
nearby realization. Degenerate near-miss clusters (several pairs at the same
|d-1| to ~1e-6) are the signature of a hinge: they fire together or not at all.

Greedy loop: try augmenting (a) each degenerate cluster, (b) top single
near-misses; accept the first augmentation whose GN projection passes the full
three-audit check with a strictly larger unit-edge count; repeat until dry.

Usage: uv run python scripts/force_coincidences.py <config.csv> --out DIR [--max-miss 0.1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from udg.audit import audit, gauss_newton
from udg.configio import load_csv, save_csv, save_audit_json
from udg.counting import unit_edges


def near_misses(P: np.ndarray, tol: float = 1e-9, max_miss: float = 0.1):
    """Non-edge pairs by |d-1| ascending, within max_miss."""
    n = len(P)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    iu, ju = np.triu_indices(n, 1)
    r = np.abs(D[iu, ju] - 1.0)
    sel = (r >= tol) & (r < max_miss)
    order = np.argsort(r[sel])
    return [
        (int(iu[sel][k]), int(ju[sel][k]), float(r[sel][k])) for k in order
    ]


def degenerate_clusters(misses, eps: float = 1e-6):
    """Group consecutive near-misses whose |d-1| agree within eps; clusters of >=2."""
    clusters, cur = [], [misses[0]] if misses else []
    for m in misses[1:]:
        if abs(m[2] - cur[-1][2]) < eps:
            cur.append(m)
        else:
            if len(cur) >= 2:
                clusters.append(cur)
            cur = [m]
    if len(cur) >= 2:
        clusters.append(cur)
    return clusters


def try_augment(P, base_edges, candidates, gn_iters):
    """GN-project P onto base_edges + candidates; return (Q, report) if the full
    audit passes with MORE edges than base, else None."""
    aug = base_edges + [(i, j) for i, j, _ in candidates]
    Q, _ = gauss_newton(P, aug, iters=gn_iters)
    rep = audit(Q)
    if rep.passed and rep.n_edges > len(base_edges):
        return Q, rep
    return None


def force(P, *, max_miss=0.1, top_singles=12, gn_iters=60000, log=print):
    rep0 = audit(P)
    if not rep0.passed:
        raise SystemExit("input config fails audit — refusing to start")
    best_P, best_edges = P, unit_edges(P)
    log(f"start: {len(best_edges)} audited edges")
    trail = []
    round_no = 0
    while True:
        round_no += 1
        misses = near_misses(best_P, max_miss=max_miss)
        if not misses:
            log("no near-misses in range — dry")
            break
        attempts = []
        for cl in degenerate_clusters(misses):
            attempts.append((f"cluster x{len(cl)} @ {cl[0][2]:.3e}", cl))
        for m in misses[:top_singles]:
            attempts.append((f"single ({m[0]},{m[1]}) @ {m[2]:.3e}", [m]))
        accepted = None
        for name, cand in attempts:
            res = try_augment(best_P, best_edges, cand, gn_iters)
            if res is not None:
                Q, rep = res
                accepted = (name, Q, rep)
                break
        if accepted is None:
            log(f"round {round_no}: no augmentation realizable — dry")
            break
        name, Q, rep = accepted
        gained = rep.n_edges - len(best_edges)
        log(
            f"round {round_no}: ACCEPTED {name} -> {rep.n_edges} edges (+{gained}), "
            f"min_sep={rep.min_sep:.4f} gn_resid={rep.gn_total_residual:.2e}"
        )
        trail.append({"round": round_no, "augmentation": name, "edges": rep.n_edges})
        best_P, best_edges = Q, unit_edges(Q)
    return best_P, best_edges, trail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-miss", type=float, default=0.1)
    ap.add_argument("--top-singles", type=int, default=12)
    ap.add_argument("--gn-iters", type=int, default=60000)
    args = ap.parse_args()

    P = load_csv(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    best_P, best_edges, trail = force(
        P, max_miss=args.max_miss, top_singles=args.top_singles, gn_iters=args.gn_iters
    )
    rep = audit(best_P)
    n = len(best_P)
    print(
        f"FINAL n={n}: {rep.n_edges} audited edges "
        f"(min_sep={rep.min_sep:.4f} k23={rep.k23_violations} "
        f"gn_resid={rep.gn_total_residual:.2e} passed={rep.passed})"
    )
    save_csv(best_P, out / f"udg{n}_{rep.n_edges}edges.csv")
    save_audit_json(rep, out / f"udg{n}_{rep.n_edges}edges_audit.json")
    (out / "trail.json").write_text(json.dumps(trail, indent=1))
    print(f"saved to {out}/")


if __name__ == "__main__":
    main()
