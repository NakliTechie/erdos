#!/usr/bin/env python
"""n=64 record recipe: the flattened 6-cube (252 edges), pure construction.

The Engel et al. Table 2 record at n=64 is the Minkowski sum of 6 unit
edges ("6-cube") flattened into the Moser lattice: the 2^6 = 64 subset sums
of the generators

    g1 = (-2,  1,  2, -1)
    g2 = (-1, -1,  1,  1)
    g3 = (-1,  0,  0,  0)
    g4 = (-1,  1,  0,  0)
    g5 = ( 0,  0, -1,  0)
    g6 = ( 0,  0,  0, -1)

are 64 distinct ML points carrying 252 exact unit edges (the 192 cube edges
plus 60 lattice coincidences).

Reads only committed inputs (pure construction + data/frontier/n64/ for the
class check). Writes outputs under runs/recipes-out/n64/. Asserts its own
verification: 64 points, 252 edges, float audit PASSED, exact ML certificate
CERTIFIED, and canon class == the staged frontier config
data/frontier/n64/udg64_252edges.csv.

Run from the repo root:  uv run python scripts/recipes/build_n64_cube.py
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
from udg.mlgraph import MLConfig, canon, exact_edge_count, save_csv  # noqa: E402

from ml_coords import certify_config  # noqa: E402

GENS = (
    (-2, 1, 2, -1),
    (-1, -1, 1, 1),
    (-1, 0, 0, 0),
    (-1, 1, 0, 0),
    (0, 0, -1, 0),
    (0, 0, 0, -1),
)

OUT = ROOT / "runs/recipes-out/n64"


def main() -> int:
    C = MLConfig.from_generators(GENS)
    n, e = len(C), exact_edge_count(C)
    print(f"flattened 6-cube: n={n} exact_edges={e}")
    assert (n, e) == (64, 252), f"recipe failed: got n={n} edges={e}"

    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "udg64_252edges_cube.csv"
    save_csv(C, csv)
    P = load_csv(csv)

    rep = audit(P)
    save_audit_json(rep, OUT / "udg64_252edges_cube_audit.json")
    print(f"audit: n_edges={rep.n_edges} min_sep={rep.min_sep:.6f} passed={rep.passed}")
    assert rep.passed and rep.n_edges == 252, "float audit failed"

    cert = certify_config(P, name=str(csv))
    with open(OUT / "udg64_252edges_cube_cert.json", "w") as f:
        json.dump(cert, f, indent=1)
    print(f"cert: certified={cert['certified']} exact_unit_pairs={cert['exact_unit_pairs']}")
    assert cert["certified"] and cert["exact_unit_pairs"] == 252, "certification failed"

    staged = json.load(open(ROOT / "data/frontier/n64/udg64_252edges_cert.json"))
    same = canon(C.as_array()) == canon(np.asarray(staged["coords"], dtype=np.int64))
    print(f"canon class == data/frontier/n64 staged config: {same}")
    assert same, "canon class mismatch vs staged frontier config"

    print("OK: n=64 / 252 edges rebuilt, audited, certified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
