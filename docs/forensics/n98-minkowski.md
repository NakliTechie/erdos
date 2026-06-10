*Produced during the 2026-06-10 closure session; see `docs/paper/`. Staged from `runs/last3/n98/FORENSICS.md`.*

# n=98 record (429 edges) — forensics + generative recipe — CLOSED 2026-06-10

Gap-1 miss from the Batch-F sweep (our ladder plateaued at an audit-clean 428;
see `docs/sweep-2026-06-10.md`). The record class is **unique** in the 60M Engel
DB (1 config in `runs/engel_db/engel_records_n98.npy`). Literature hint:
"4 edges ⊕ 7-vertex UDG" — which did NOT reproduce with wheel/spindle factors
in the mop-up (sums collapsed to 80/56/72/76/108 points). This session dissected
the DB record, recovered the true factor pair, and rebuilt it from toolkit
primitives. **Closed: audited + certified 429 at n=98, same canon class as the
DB record.**

## 1. Dissection of the DB record (integer coords, all exact)

`forensics1.py` … `forensics6.py`:

- **Counts**: n=98, exactly 429 unit pairs (`exact_edge_count`). Coefficient box
  `[0,4]^4` — a tight coset structure, generator-built smell (like the n=64 6-cube).
- **Degree histogram**: `{6:2, 7:18, 8:38, 9:14, 10:8, 11:12, 12:4, 14:1, 16:1}`.
  One deg-16 hub at (2,2,2,2) and one deg-14 at (3,2,2,2) — interior points of the sum.
- **Direction families**: 9 of the 9 ML unit directions carry edges —
  w1-ring 56/56/51, w3-ring 54/54/39, full cross-sublattice third ring 44/44/31
  (the third-ring directions are the `(1±w1)(1−w3)`-type spindle tip-tip family).
- **Nesting**: the record IS a vertex-subset of 9 of the 37 n=100 record configs
  and 4 of the 7 n=99 record configs (Engel's beam visits sub-configs — two of the
  n=100 records contain it at identity motion, zero translation). It contains
  n=97 record configs #4 and #5. It is NOT a subset of OUR tied n=99 (434) or
  n=100 (439) configs — our neighbors are different classes, and exhaustive
  drop-2 from our 439 never reaches 429 (best 428). Route (1) was therefore dead
  without DB structure.
- **Translate decomposition**: union-of-2-translate factorizations exist along
  every w1-ring direction (|C ∩ (C+t)| = 56, i.e. 56+56−14 collisions) and
  w3-ring direction (54: 54+54−10). A **direct** 14×7 or 7×14 product does NOT
  exist (`forensics3.py`: zero factorizations) — the published "4 edges ⊕
  7-vertex" sum has **collisions** (7·2⁴ = 112 → 98, i.e. 14 coincidences).
- **Factor recovery** (`forensics4.py`): exhaustive scan over all
  C(18+3,4) multisets of the 18 unit vectors for a cube K with
  F = ∩_{k∈K}(C−k), ⋃(F+k) = C. **Unique hit**:
  - K = {0,1}-sums of `{w1², −w1, w3·w1², −w3·w1}` — 16 points, 41 edges.
    Equivalently K = R ⊕ w3·R where R = unit rhombus {0, w1², −w1, −1}
    (canon-verified). This is the "4 edges" factor.
  - F = 7 points, 11 edges — same counts as the Moser spindle but **NOT the
    spindle** (canon differs; deg seq [2,3,3,3,3,4,4], 3 triangles vs the
    spindle's [3,3,3,3,3,3,4], 4 triangles). Spindle ⊕ K indeed collapses to
    64/56 points — exactly the mop-up failure.
- **F's own structure** (`forensics6.py`): translated to a basepoint, F =
  {0, 1, e2, e3, 1+e2, 1+e3, e2+e3} with e2 = (1,1,−1,−1) = (1+w1)(1−w3) and
  e3 = (2,−1,−2,1) = (2−w1)(1−w3) — i.e. F is the {0,1}-sum 3-cube on unit
  generators {1, e2, e3} **minus the all-ones corner** 1+e2+e3 = (4,0,−3,0).
  ({0, e2, e3, e2+e3} is itself a unit rhombus: e3−e2 = (1,−2,−1,2) is unit.)

So the record is a **7-generator object**: (3-cube minus a corner) ⊕ (4-cube),
7·16 − 14 collisions = 98 points, 429 edges. The subtlety that killed the naive
attempts: the F-cube generators 1, (1+w1)(1−w3), (2−w1)(1−w3) mix all three ML
direction rings, and the corner deletion happens BEFORE the sum.

### Why our 428 near-miss lost

Our anneal 428 (`runs/sweep/n98/udg98_428edges_final_coords.json`) is a
different class (box [−2..2]×[−4..0]×[−2..2]×[−2..1]), degree histogram
`{6:8, 7:28, 8:38→28, …}` — 8 degree-6 vertices vs the record's 2, a fatter
low-degree tail the sum structure avoids.

## 2. Generative recipe (toolkit-only, `build_recipe.py`)

```python
from udg.mlgraph import MLConfig, minkowski

GENS_K = [(-1,1,0,0), (0,-1,0,0), (0,0,-1,1), (0,0,0,-1)]  # w1², −w1, w3·w1², −w3·w1
GENS_F = [(1,0,0,0), (1,1,-1,-1), (2,-1,-2,1)]             # 1, (1+w1)(1−w3), (2−w1)(1−w3)
K = MLConfig.from_generators(GENS_K)                        # "4 edges": 16-pt cube
F = MLConfig.from_generators(GENS_F).without_point((4,0,-3,0))  # 3-cube minus top corner
C = minkowski(F, K)                                         # 98 points, 429 edges
```

All seven generators are ML unit vectors; every primitive
(`from_generators`, `without_point`, `minkowski`) is library code.

## 3. Verification (all green)

- `exact_edge_count(C) == 429`, `len(C) == 98` (exact integer arithmetic).
- `scripts/audit_config.py runs/last3/n98/udg98_429edges_recipe.csv` → **exit 0**
  (min_sep 0.2149, K23 0, GN residual 7.3e-30, 429/429 edges exact).
- `scripts/ml_coords.py` → **CERTIFIED ML**, exact_unit_pairs = 429 = float
  edges, max resid 4.8e-16 rad, embed err 4.6e-16.
- **`chase40lib`/mlgraph canon class == the unique Engel DB record class**
  (`canon(C) == canon(engel_records_n98[0])` — we reproduce THEIR class, not a
  new one; no full-DB diff needed).
- Degree histogram matches the record exactly.

## 4. What structural information came from the DB

The DB record config was used to (a) confirm nesting behavior (informational
only — the final recipe does not use any neighbor config), and (b) recover the
factor pair via the exhaustive cube-factor scan: i.e. **the identity of the 7
unit generators and the removed corner came from dissecting their config**.
No coordinates were copied: the deliverable config is regenerated from
`from_generators` + one corner deletion + `minkowski`, and lands in their class
by construction (canon-verified).

## Files

- `build_recipe.py` — the recipe (reproduces everything from an empty tree)
- `udg98_429edges_recipe.csv` / `_coords.json` — config (float CSV + integer coords + recipe metadata)
- `udg98_429edges_recipe_audit.json` / `_cert.json` — audit report + exact ML certificate
- `forensics1.py`–`forensics7.py` — the dissection scripts (degrees/directions,
  nesting, translate factorization, cube-factor scan, spindle test, F dissection,
  near-miss comparison)
