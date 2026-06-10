*Produced during the 2026-06-10 closure session; see `docs/paper/`. Staged from `runs/last3/n90/FORENSICS.md`.*

# n=90 / 385 edges — forensics + generative recipe (gap CLOSED, twice)

Date: 2026-06-10. Context: last-3 mop-up of the frontier sweep
(docs/sweep-2026-06-10.md). n=90 was a gap −1 miss (audited best 384); the base
12-min run lost its budget to the pre-fix early-stop bug on an audit-invalid
385 (min_sep 0.0851 close pair in the t2xt4c1 closure).

**Outcome: TWO audited + certified 385-edge configs at n=90, each from a pure
toolkit recipe, landing in two different Engel record classes (DB record
indices 14 and 29 of 42).**

## 1. Dissection of the 42 DB record configs (`runs/engel_db/engel_records_n90.npy`)

- **42 configs, 42 distinct canon classes** (udg.mlgraph.canon, 12-element ML
  point group x translation), all with exactly 385 exact edges. Two families by
  degree profile:
  - classes 0–4: hub-rich (max degree 15–18; the deg-18 vertex has its full
    18-unit-vector neighborhood present — a closure point);
  - classes 5–41: flat profile, max degree 13–15, min degree 5–6.
- **Direction families** (audit_config on our reproduction): all 9 ML direction
  families used — w1-ring (0/60/120°), w3-ring (16.78/76.78/136.78°), mixed
  w1·w3 (33.56/93.56/153.56°). Edge split of the class-14 rep: 153 w1-ring,
  130 w3-ring, 102 mixed.
- **Coset structure** (class-14 rep): 13 fibers over the w3-sublattice, most of
  size 7 = hexagonal H(1) patches (sizes 7,7,3,9,7,7,7,3,7,12,7,7,7) —
  the layered H(1) ⊕ w3·T(4) signature. Top translate overlaps |C∩(C+t)| ≈
  50–52 at the six w1-ring unit vectors.
- **NESTING (the load-bearing fact):** every n=91 record has 390 edges; dropping
  any degree-5 vertex leaves 90 points / 385 edges. Scanning all such drops over
  the 78 n=91 records: **all 42 n=90 record classes are produced** (coverage
  42/42, multiplicities 1–8). The n=90 record level is entirely the drop-1
  shadow of the n=91 record level. Conversely the add-1 route from n=89 also
  reaches it (see route B).
- **Our n=91/390 sweep config** (`runs/sweep/n91/udg91_390edges_final*`,
  subset:h1xt4c1, seed 0, 38.8 s) is **lattice-identical (canon) to DB n=91
  record index 28** — we had already independently reproduced one of their
  n=91 record classes.
- **What our 384 near-miss lacked** (`runs/sweep/n90_retry/`, subset:t3xt3c1):
  degree histogram {5:4, 6:8, 7:20, 8:30, 9:8, 10:4, 11:6, 13:4, 15:6} — four
  degree-5 vertices and a 15-heavy tail, vs the record class-14 profile
  {6:4, 7:23, 8:31, 9:9, 10:9, 11:6, 12:5, 13:2, 14:1} with min degree 6. The
  t3xt3c1 ambient plateaus 1 short; the record lives in the h1xt4c1 span.

## 2. Generative recipes (both pure toolkit)

**Route A — trim our n=91 tie (deliverable config, class 14):**
1. Ambient: `neighbor_closure(minkowski(hex_patch(1), to_w3(tri_patch(4))), 1)`
   — closure-1 of H(1) ⊕ w3·T(4), 771 points (chase_n's `h1xt4c1`).
2. Densest-91-subgraph subset search in that ambient (udg.subsetsearch via
   `chase_n.py --n 91 --seed 0`, subset stage, 38.8 s) → 91 pts / 390 edges,
   audited + certified (`runs/sweep/n91/udg91_390edges_final*`).
3. `drop_worst_point` — the config has a **unique** minimum-degree vertex,
   deg 5 at (0,-1,1,2) — → **90 pts / 385 edges**.
   Reproduce: `uv run python runs/last3/n90/build_trim91.py`

**Route B — extend our n=89 tie (class 29):**
1. Our audited n=89/380 tie (`runs/sweep/n89retry/udg89_380edges_final*`).
2. `add_best_point` (deterministic best-gain candidate over the 18-unit-vector
   neighborhood) adds (1,1,3,-1) with gain 5 → **90 pts / 385 edges**.

## 3. Verification (every claim)

| check | route A (trim91drop1) | route B (add89plus1) |
|---|---|---|
| exact_edge_count | 385 = target | 385 = target |
| scripts/audit_config.py | exit 0, PASSED, min_sep 0.2149 | exit 0, PASSED, min_sep 0.2149 |
| scripts/ml_coords.py | CERTIFIED ML, 385 exact unit pairs, max_resid 4.8e-16 | CERTIFIED ML, 385 exact unit pairs, max_resid 7.3e-16 |
| canon vs 42 DB record configs | **= DB record idx 14** | **= DB record idx 29** |

Both reproduce *their* classes (no new-class claim — no full-DB diff needed).

## 4. What structural information came from the DB

- Route A/B constructions themselves use **zero DB coordinates**: the ambient,
  subset search, and trim/extend moves are all library primitives, and the
  n=91/n=89 inputs are our own sweep artifacts.
- From the DB we took: (a) the **target count 385** (published Table-2 value);
  (b) the **nesting hint** from the task brief, confirmed by our scan (42/42
  classes are n=91 drop-1 shadows) — this told us trim/extend was the cheapest
  route to try first; (c) **post-hoc class identification** (canon match of our
  outputs against the record inventory, and of our n=91 config against theirs).
- The DB record coordinates were used only for forensics + verification, never
  copied into a config.

## 5. Parallel arm

A clean post-fix `chase_n.py --n 90 --seed 3 --budget-min 40 --processes 3`
run (runs/last3/n90/chase_seed3/) was launched as a control before forensics
began; it correctly refuses the audit-failing t2xt4c1 385 (driver fix working).
See chase_seed3/summary.json for its outcome — the closure above does not
depend on it.

## Files

- `udg90_385edges_trim91drop1.{csv,_audit.json,_cert.json,_coords.json}` + `_class.json`
- `udg90_385edges_add89plus1.{csv,_audit.json,_cert.json,_coords.json}` + `_class.json`
- `mlcert/` — standalone ml_coords certificates for both CSVs
- `build_trim91.py` — reproducible route-A builder (asserts + audit + cert + class match)
- `chase_seed3/`, `chase_seed3.log` — the parallel ladder arm
