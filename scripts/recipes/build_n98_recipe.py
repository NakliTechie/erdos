#!/usr/bin/env python
"""n=98 record recipe (429 edges), pure construction from generators.

C98 = F (+) K   (exact Minkowski sum; 14 lattice collisions: 7*16 - 14 = 98)

  K = MLConfig.from_generators([w1^2, -w1, w3*w1^2, -w3*w1])  -- "4 edges":
      a 16-point flattened 4-cube
    = unit_rhombus(w1^2, -w1) (+) w3 * (same rhombus)
  F = MLConfig.from_generators([1, (1+w1)(1-w3), (2-w1)(1-w3)]) minus the top
      corner = 3-cube of unit generators minus the all-ones corner: 7 vertices,
      11 edges (NOT the Moser spindle: deg seq [2,3,3,3,3,4,4], 3 triangles)

This is the literature's "4 edges ⊕ 7-vertex UDG" hint made concrete; the
result is canon-identical to the staged frontier config
data/frontier/n98/udg98_429edges.csv (itself the unique 429-edge record class).
See docs/forensics/n98-minkowski.md for the forensic derivation.

Reads only committed inputs (pure construction + data/frontier/n98/ for the
class check). Writes outputs under runs/recipes-out/n98/. Asserts its own
verification: 98 points, 429 exact edges, float audit PASSED, exact ML
certificate CERTIFIED, canon class == staged frontier config.

Run from the repo root:  uv run python scripts/recipes/build_n98_recipe.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from udg.audit import audit  # noqa: E402
from udg.configio import load_csv, save_audit_json  # noqa: E402
from udg.mlgraph import (  # noqa: E402
    MLConfig,
    canon,
    degrees,
    exact_edge_count,
    minkowski,
    save_csv,
)

from ml_coords import certify_config  # noqa: E402

GENS_K = [(-1, 1, 0, 0),   # w1^2
          (0, -1, 0, 0),   # -w1
          (0, 0, -1, 1),   # w3 * w1^2
          (0, 0, 0, -1)]   # -w3 * w1
GENS_F = [(1, 0, 0, 0),    # 1
          (1, 1, -1, -1),  # (1+w1)(1-w3)
          (2, -1, -2, 1)]  # (2-w1)(1-w3)
TOP = (4, 0, -3, 0)        # 1 + (1+w1)(1-w3) + (2-w1)(1-w3) = sum(GENS_F)

OUT = ROOT / "runs/recipes-out/n98"


def main() -> int:
    K = MLConfig.from_generators(GENS_K)
    F = MLConfig.from_generators(GENS_F).without_point(TOP)
    C = minkowski(F, K)
    n, e = len(C), exact_edge_count(C)
    print(f"K: n={len(K)} edges={exact_edge_count(K)}")
    print(f"F: n={len(F)} edges={exact_edge_count(F)}")
    print(f"C = F (+) K: n={n} edges={e}")
    assert (len(K), len(F)) == (16, 7)
    assert (n, e) == (98, 429), f"recipe failed: got n={n} edges={e}"
    print("deg hist:", sorted(Counter(degrees(C).tolist()).items()))

    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "udg98_429edges_recipe.csv"
    save_csv(C, csv)
    P = load_csv(csv)

    rep = audit(P)
    save_audit_json(rep, OUT / "udg98_429edges_recipe_audit.json")
    print(f"audit: n_edges={rep.n_edges} min_sep={rep.min_sep:.6f} passed={rep.passed}")
    assert rep.passed and rep.n_edges == 429, "float audit failed"

    cert = certify_config(P, name=str(csv))
    with open(OUT / "udg98_429edges_recipe_cert.json", "w") as f:
        json.dump(cert, f, indent=1)
    print(f"cert: certified={cert['certified']} exact_unit_pairs={cert['exact_unit_pairs']}")
    assert cert["certified"] and cert["exact_unit_pairs"] == 429, "certification failed"

    staged = json.load(open(ROOT / "data/frontier/n98/udg98_429edges_cert.json"))
    same = canon(C.as_array()) == canon(np.asarray(staged["coords"], dtype=np.int64))
    print(f"canon class == data/frontier/n98 staged config: {same}")
    assert same, "canon class mismatch vs staged frontier config"

    print("OK: n=98 / 429 edges rebuilt, audited, certified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
