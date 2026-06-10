#!/usr/bin/env python
"""Audit a unit-distance configuration CSV (the three-audit gate + lattice ID).

Usage:
    uv run python scripts/audit_config.py <config.csv>

Loads the config (header x,y), runs udg.audit.audit (min-sep, K_{2,3},
Gauss-Newton exact realizability), prints the full AuditReport, then the
direction families of the claimed unit edges and the Moser-lattice
identification summary (udg.moser).

Exit code 0 if the audit PASSED, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys

from udg.audit import audit
from udg.configio import load_csv
from udg.counting import unit_edges
from udg.moser import direction_families, lattice_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("csv", help="config CSV with header x,y")
    args = ap.parse_args(argv)

    P = load_csv(args.csv)
    rep = audit(P)

    print(f"config: {args.csv}")
    print("AuditReport:")
    print(f"  n                 = {rep.n}")
    print(f"  n_edges           = {rep.n_edges}")
    print(f"  min_sep           = {rep.min_sep:.6f}")
    print(f"  k23_violations    = {rep.k23_violations}")
    print(f"  gn_total_residual = {rep.gn_total_residual:.3e}")
    print(f"  gn_edges_exact    = {rep.gn_edges_exact}")
    print(f"  gn_max_move       = {rep.gn_max_move:.3e}")
    print(f"  min_sep_after     = {rep.min_sep_after:.6f}")
    print(f"  passed            = {rep.passed}")

    edges = unit_edges(P)
    fams = direction_families(P, edges)
    print(f"direction_families ({len(fams)} families, cluster_tol=0.5):")
    for count, angle in fams:
        print(f"  count={count:4d} angle={angle:8.2f} deg")

    lid = lattice_id(P, edges)
    print(
        f"lattice_id: n_dirs={lid['n_dirs']} n_matched={lid['n_matched']} "
        f"best_rotation={lid['best_rotation']:.2f} deg"
    )
    if lid["n_dirs"]:
        matched = [f"{d:.2f}" for d, m in zip(lid["dirs"], lid["matched_mask"]) if m]
        print(f"  matched dirs (deg): {', '.join(matched) if matched else '(none)'}")

    print("PASSED" if rep.passed else "FAILED")
    return 0 if rep.passed else 1


if __name__ == "__main__":
    sys.exit(main())
