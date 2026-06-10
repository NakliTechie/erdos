#!/usr/bin/env python
"""n=96 record recipe (418 edges, exact-certificate track): Galois trace ball
+ densest-96-subgraph.

Ambient: {p in ML : A^2 + 3B^2 + 11C^2 + 33D^2 <= 288} with (A,B,C,D) the
exact integer invariants of p (12*Re = A + D*sqrt33, 12*Im = B*sqrt3 +
C*sqrt11). This is the exact-integer form of the Galois trace ball
|z|^2 + |sigma z|^2 <= 4 (sigma flips the sign of sqrt11). The ball has 103
points carrying 456 exact edges; the unique Engel record class at n=96 is a
subset of it. subset_ils (deterministic, seed 0) recovers the densest
96-subset at 418 exact edges.

*** AUDIT CONVENTION (read before re-running): the float three-audit FAILS
on this config BY DESIGN -- it contains four point pairs at exact distance
sqrt((23 - 4*sqrt33)/3) ~= 0.085146, below the audit's 0.2 min-separation
floor. That close pair is inherent to the unique 418-edge record class (a
min-sep-clean 418 likely does not exist: ball-exhaustive clean max is 414,
and the Engel 60M-config DB holds exactly one 418 -- this class). The config
is accepted on the EXACT-CERTIFICATE track (dual-track convention, decision
2026-06-10): the ML certificate proves all 418 edges exactly unit and all 96
points exactly distinct in Q(sqrt3, sqrt11). This script therefore asserts
audit NOT passed with min_sep 0.085146, and certificate CERTIFIED. See
docs/forensics/n96-traceball.md. ***

Reads only committed inputs (pure construction + data/frontier/n96/ for the
class check). Writes outputs under runs/recipes-out/n96/.

Run from the repo root:  uv run python scripts/recipes/build_n96_traceball.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from udg.audit import audit  # noqa: E402
from udg.configio import load_csv, save_audit_json  # noqa: E402
from udg.mlgraph import MLConfig, canon, save_csv  # noqa: E402
from udg.subsetsearch import edge_count, subset_ils  # noqa: E402

from ml_coords import certify_config  # noqa: E402

OUT = ROOT / "runs/recipes-out/n96"


def invariants(p: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Exact integer invariants (A,B,C,D) of an ML point: 12*Re = A + D*sqrt33,
    12*Im = B*sqrt3 + C*sqrt11; the trace form is A^2 + 3B^2 + 11C^2 + 33D^2."""
    a, b, c, d = p
    return (12 * a + 6 * b + 10 * c + 5 * d, 6 * b + 5 * d, 2 * c + d, -d)


def trace_ball(bound: int = 288, box: int = 8) -> np.ndarray:
    """All ML points with trace form <= bound (coefficient box search)."""
    pts = []
    for a in range(-box, box + 1):
        for b in range(-box, box + 1):
            for c in range(-box, box + 1):
                for d in range(-box, box + 1):
                    A, B, C, D = invariants((a, b, c, d))
                    if A * A + 3 * B * B + 11 * C * C + 33 * D * D <= bound:
                        pts.append((a, b, c, d))
    return np.array(sorted(pts), dtype=np.int64)


def main() -> int:
    amb = trace_ball()
    n_amb, e_amb = len(amb), edge_count(amb)
    print(f"exact trace ball T<=4: {n_amb} points, {e_amb} exact edges")
    assert (n_amb, e_amb) == (103, 456), "trace-ball enumeration changed"

    m, pts = subset_ils(amb, 96, seed=0, max_iters=400, target=418)
    print(f"subset_ils k=96 seed=0: best exact edges = {m}")
    assert m == 418, f"densest-96-subset search failed: got {m}"

    C = MLConfig(map(tuple, pts.tolist()))
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "udg96_418edges_traceball.csv"
    save_csv(C, csv)
    P = load_csv(csv)

    rep = audit(P)
    save_audit_json(rep, OUT / "udg96_418edges_traceball_audit.json")
    print(f"audit: n_edges={rep.n_edges} min_sep={rep.min_sep:.6f} passed={rep.passed}")
    assert rep.n_edges == 418
    assert not rep.passed and f"{rep.min_sep:.6f}" == "0.085146", (
        "expected the documented float-audit failure (min_sep 0.085146); "
        f"got passed={rep.passed} min_sep={rep.min_sep:.6f}"
    )

    cert = certify_config(P, name=str(csv))
    with open(OUT / "udg96_418edges_traceball_cert.json", "w") as f:
        json.dump(cert, f, indent=1)
    print(f"cert: certified={cert['certified']} exact_unit_pairs={cert['exact_unit_pairs']}")
    assert cert["certified"] and cert["exact_unit_pairs"] == 418, "certification failed"

    staged = json.load(open(ROOT / "data/frontier/n96/udg96_418edges_cert.json"))
    same = canon(C.as_array()) == canon(np.asarray(staged["coords"], dtype=np.int64))
    print(f"canon class == data/frontier/n96 staged config: {same}")
    assert same, "canon class mismatch vs staged frontier config"

    print("OK: n=96 / 418 edges rebuilt; float audit fails by design "
          "(min_sep 0.085146); exact certificate CERTIFIED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
