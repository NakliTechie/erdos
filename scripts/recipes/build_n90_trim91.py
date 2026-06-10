#!/usr/bin/env python
"""n=90 record recipes (385 edges), two independent routes from committed inputs.

Route A ("neighbor-trim", the staged frontier config): our audited n=91 /
390-edge record tie (data/frontier/n91/) has exactly one vertex of minimum
exact degree 5; drop_worst_point removes it, leaving 90 points with exactly
390 - 5 = 385 exact edges = the Engel et al. Table 2 densest-known value at
n=90. The result is canon-identical to data/frontier/n90/udg90_385edges.csv.

Route B ("grow-89"): add_best_point on our audited n=89 / 380-edge record tie
(data/frontier/n89/) appends the candidate position gaining the most exact
edges (+5), giving a second, NON-isomorphic 90-point / 385-edge config — the
two routes land in two different Engel record classes (see
docs/forensics/n90-nesting.md).

Reads only committed inputs: the exact integer ML coordinates inside the
staged certificates data/frontier/n91/udg91_390edges_cert.json and
data/frontier/n89/udg89_380edges_cert.json. Writes outputs under
runs/recipes-out/n90/. Asserts its own verification: both routes produce
90 points / 385 exact edges, float audit PASSED, exact ML certificate
CERTIFIED.

Run from the repo root:  uv run python scripts/recipes/build_n90_trim91.py
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
from udg.mlgraph import (  # noqa: E402
    MLConfig,
    add_best_point,
    canon,
    degrees,
    drop_worst_point,
    exact_edge_count,
    save_csv,
)

from ml_coords import certify_config  # noqa: E402

OUT = ROOT / "runs/recipes-out/n90"


def verify_and_save(cfg: MLConfig, stem: str) -> None:
    """Save CSV + audit + cert for a 90-point/385-edge config; assert all gates."""
    assert (len(cfg), exact_edge_count(cfg)) == (90, 385)
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / f"{stem}.csv"
    save_csv(cfg, csv)
    P = load_csv(csv)
    rep = audit(P)
    save_audit_json(rep, OUT / f"{stem}_audit.json")
    print(f"  audit: n_edges={rep.n_edges} min_sep={rep.min_sep:.6f} passed={rep.passed}")
    assert rep.passed and rep.n_edges == 385, f"{stem}: float audit failed"
    cert = certify_config(P, name=str(csv))
    with open(OUT / f"{stem}_cert.json", "w") as f:
        json.dump(cert, f, indent=1)
    print(f"  cert: certified={cert['certified']} exact_unit_pairs={cert['exact_unit_pairs']}")
    assert cert["certified"] and cert["exact_unit_pairs"] == 385, f"{stem}: certification failed"


def main() -> int:
    # ---- Route A: trim the unique min-degree vertex of the n=91 record tie
    cfg91 = MLConfig.from_json(ROOT / "data/frontier/n91/udg91_390edges_cert.json")
    assert (len(cfg91), exact_edge_count(cfg91)) == (91, 390)
    deg = degrees(cfg91)
    min_deg = int(deg.min())
    n_min = int((deg == min_deg).sum())
    print(f"route A: n=91/390 loaded; min exact degree {min_deg} at {n_min} vertex(es)")
    assert min_deg == 5 and n_min == 1, "expected a unique degree-5 vertex in the n=91 config"
    trim = drop_worst_point(cfg91)  # deterministic: removes the unique min-degree vertex
    print(f"route A: trimmed -> n={len(trim)} edges={exact_edge_count(trim)}")
    verify_and_save(trim, "udg90_385edges_trim91drop1")

    staged = json.load(open(ROOT / "data/frontier/n90/udg90_385edges_cert.json"))
    same = canon(trim.as_array()) == canon(np.asarray(staged["coords"], dtype=np.int64))
    print(f"route A: canon class == data/frontier/n90 staged config: {same}")
    assert same, "route A canon class mismatch vs staged frontier config"

    # ---- Route B: grow the n=89 record tie by its best candidate point
    cfg89 = MLConfig.from_json(ROOT / "data/frontier/n89/udg89_380edges_cert.json")
    assert (len(cfg89), exact_edge_count(cfg89)) == (89, 380)
    grown = add_best_point(cfg89)  # deterministic: max-gain candidate, lex tie-break
    print(f"route B: n=89/380 + best point -> n={len(grown)} edges={exact_edge_count(grown)}")
    verify_and_save(grown, "udg90_385edges_add89plus1")

    distinct = canon(grown.as_array()) != canon(trim.as_array())
    print(f"routes A and B land in distinct canon classes: {distinct}")

    print("OK: n=90 / 385 edges rebuilt by two routes, audited, certified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
