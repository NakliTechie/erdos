#!/usr/bin/env python
"""Production driver for the unit-distance record-chase campaigns.

Fans udg.search.multi_search out across seeds, audits EVERY seed result with
udg.audit.audit (the three-audit gate — never report an unaudited count), and
saves the best PASSED configuration into --out:

    best.csv          best passed config (header x,y, full float64 precision)
    best_audit.json   its AuditReport
    summary.json      {n, steps, per-seed results, best}

Prints one line per seed:

    seed=K count=C audited_edges=E min_sep=S k23=V gn_resid=R passed=P

and a final BEST line. Always exits 0 (campaign driver: a fruitless batch of
seeds is data, not an error).

Usage:
    uv run python scripts/run_search.py --n 40 --seeds 8 --steps 150000 \
        --out runs/<name>/ [--warm seed.csv] [--t0 1.2 --t1 0.015] [--processes 4]

--seeds is either an integer COUNT (8 -> seeds 0..7) or an explicit
comma-separated list (3,17,42 -> exactly those; a trailing comma makes a
single-seed list: "7," -> [7]).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from udg.audit import audit
from udg.configio import save_audit_json, save_csv
from udg.search import multi_search


def parse_seeds(spec: str) -> list[int]:
    """'8' -> [0..7] (count); '3,17,42' -> [3, 17, 42] (explicit list)."""
    if "," in spec:
        seeds = [int(tok) for tok in spec.split(",") if tok.strip() != ""]
        if not seeds:
            raise argparse.ArgumentTypeError(f"no seeds in list: {spec!r}")
        return seeds
    return list(range(int(spec)))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--n", type=int, default=40, help="number of points (ignored with --warm)")
    ap.add_argument(
        "--seeds",
        type=parse_seeds,
        default="8",
        help="int COUNT (8 -> 0..7) or comma list (3,17,42)",
    )
    ap.add_argument("--steps", type=int, default=150_000, help="Metropolis steps per seed")
    ap.add_argument("--out", type=Path, required=True, help="output directory (created)")
    ap.add_argument("--warm", type=Path, default=None, help="warm-start config CSV (P0)")
    ap.add_argument("--t0", type=float, default=1.2, help="start temperature T0")
    ap.add_argument("--t1", type=float, default=0.015, help="end temperature T1")
    ap.add_argument("--processes", type=int, default=None, help="process-pool workers")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds: list[int] = args.seeds if isinstance(args.seeds, list) else parse_seeds(args.seeds)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    kw: dict = {"T0": args.t0, "T1": args.t1}
    n = args.n
    if args.warm is not None:
        from udg.configio import load_csv

        P0 = load_csv(args.warm)
        kw["P0"] = P0
        n = len(P0)

    results = multi_search(n, seeds, args.steps, processes=args.processes, **kw)

    per_seed = []
    best = None  # (report, result)
    for res, seed in zip(results, seeds):
        rep = audit(res.P)
        print(
            f"seed={seed} count={res.best_count} audited_edges={rep.n_edges} "
            f"min_sep={rep.min_sep:.4f} k23={rep.k23_violations} "
            f"gn_resid={rep.gn_total_residual:.3e} passed={rep.passed}",
            flush=True,
        )
        per_seed.append(
            {
                "seed": seed,
                "count": res.best_count,
                "audited_edges": rep.n_edges,
                "min_sep": rep.min_sep,
                "k23_violations": rep.k23_violations,
                "gn_total_residual": rep.gn_total_residual,
                "gn_edges_exact": rep.gn_edges_exact,
                "gn_max_move": rep.gn_max_move,
                "min_sep_after": rep.min_sep_after,
                "passed": rep.passed,
            }
        )
        if rep.passed and (
            best is None
            or (rep.n_edges, res.best_count) > (best[0].n_edges, best[1].best_count)
        ):
            best = (rep, res)

    summary: dict = {"n": n, "steps": args.steps, "seeds": seeds, "per_seed": per_seed}
    if best is not None:
        brep, bres = best
        csv_path = out / "best.csv"
        audit_path = out / "best_audit.json"
        save_csv(bres.P, csv_path)
        save_audit_json(brep, audit_path)
        summary["best"] = {
            "seed": bres.seed,
            "count": bres.best_count,
            "audited_edges": brep.n_edges,
            "passed": True,
            "csv": str(csv_path),
            "audit_json": str(audit_path),
        }
    else:
        summary["best"] = None

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    if best is not None:
        brep, bres = best
        print(
            f"BEST seed={bres.seed} count={bres.best_count} "
            f"audited_edges={brep.n_edges} min_sep={brep.min_sep:.4f} "
            f"k23={brep.k23_violations} gn_resid={brep.gn_total_residual:.3e} "
            f"passed=True csv={out / 'best.csv'}",
            flush=True,
        )
    else:
        print("BEST none — no seed passed the audit (nothing saved)", flush=True)
    return 0  # campaign driver: always exit 0


if __name__ == "__main__":  # main-guard: spawn-safe for multiprocessing
    sys.exit(main())
