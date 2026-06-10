*Produced during the 2026-06-10 closure session; see `docs/paper/`. Staged from `runs/last3/n96/FORENSICS.md`.*

# n=96 record forensics + rebuild (target 418) — 2026-06-10

DB inventory: `runs/engel_db/engel_records_n96.npy`, shape (1, 96, 4) —
**the 418-edge class is unique among the ~1.68M stored n=96 configs.**
Our sweep best: 417 (subset:h1xh2c1 and anneal both plateaued there twice).

**VERDICT (read this first): the unique 418 class was REPRODUCED generatively
(canon-identical, exact certificate, deterministic seed-0 recipe), but it
FAILS our float audit — it contains four point pairs at distance
sqrt((23−4·sqrt33)/3) ≈ 0.08515 < MIN_SEP 0.2 (the same inherent close pair
that poisoned the t3xt3c1/t2xt4c1 closures at n=71/72). The n=96 target is
therefore structurally unreachable under this repo's audit conventions unless
a *second*, min-sep-clean 418 class exists — which a dedicated constrained
hunt did not find (and which their 60M-config DB does not contain). The
audit-clean frontier at n=96 stays at our 417.**

## Dissection of the DB record (integer ML coords)

- **Edges 418** (verified with the exact scaled-integer test, chase40lib).
- **Degree histogram** `{6:20, 7:4, 8:23, 9:30, 10:8, 11:4, 14:1, 15:2, 16:3, 18:1}`.
  One **degree-18 hub at (2,2,2,2)** — its complete 18-vector unit
  neighborhood is present (the max possible ML degree). Three degree-16 and
  two degree-15 near-saturated hubs adjacent to it. Our 417 tops out at
  degree 16 (hist `{6:9 ,7:25, 8:29, 9:5, 10:6, 11:12, 12:5, 14:1, 15:2, 16:2}`)
  — the missing 18-hub is exactly the structural delta.
- **Coefficient box** [0,4]^4, hub dead center.
- **Ambient identification (coarse)**: writing each point as alpha + w3*beta
  (alpha = (a,b), beta = (c,d), centered on the hub), every point has
  alpha in H(2) and beta in H(2) ==> the record is a 96-point subset of
  **H(2) (+) w3·H(2) = `minkowski(hex_patch(2), to_w3(hex_patch(2)))`**,
  361 lattice points. It also embeds in h1xh2c1 (the 895-point closure
  ambient the sweep DID search) and w49c2; it does NOT fit
  h1xh2/h1xh3/h2xh1 (133–259 pts). The sweep never built h2xh2 at n=96:
  `build_ambients` keeps the four smallest cross-sums (100–150 pts at
  n=96), so 361 was filtered out.
- **Ambient identification (EXACT — the generative key)**: the record lies
  inside the **Galois trace-form ball**. For p in ML with integer invariants
  (A,B,C,D) (12·Re = A + D·sqrt33, 12·Im = B·sqrt3 + C·sqrt11), the trace
  T(p) = |z(p)|² + |σ(z(p))|² over the conjugate embedding σ: sqrt11 → −sqrt11
  is purely rational, and

      T(p) <= 4   <=>   A² + 3B² + 11C² + 33D² <= 288.

  This exact integer ball (centered on the hub) has **103 points and 456
  exact edges**, contains the record's 96 points, and is a strict subset of
  h2xh2. The 7 ball points the record drops are NOT a metric/disc rule (LP
  separation by any shifted trace ball fails; Euclidean disc trims fail) —
  they are the *combinatorial* optimum: densest-96-subgraph of the 103-point
  ball. The neighbor records nest in the same ball: densest-95 = 412
  (DB n95 hub family), densest-96 = 418 (THE record), densest-97 = 423
  (= DB n97[0] class, canon-verified); densest-98 = 428 < 429 (the n98
  record leaves the ball — the hub-family chain is 95 → 96 → 97 and stops).
- **Edge directions**: only 9 of the 18 unit-vector classes carry edges,
  near-uniformly: 48 each for 1, w1, w1² (w1-ring), w3, w3·w1, w3·w1²
  (w3-ring) and the spindle-tip direction 2−w1−2w3+w1w3; 41 each for the
  other two spindle-tip directions.
- **Translate decomposition**: |C ∩ (C+t)| = 65/96 for all six
  t = ±(w3−1)·w1^k — near-invariance under the *short* hexagonal vector
  w3−1 (|w3−1| = 1/sqrt3); 48/96 for the unit translations. Consistent
  with the dense trace-ball structure, not with a clean 2-factor Minkowski
  product (no exact factorization).
- **Point-group stabilizer about the hub**: order 2 — z → w1·w3·conj(z).
  All rotation overlaps are 89/96 (high but inexact).

## THE AUDIT FINDING (root cause of the sweep miss)

`scripts/audit_config.py` on the DB record itself: **FAILED — min_sep
0.085146** (K23 = 0, Gauss–Newton residual ~1e-29, all 418 edges exact; the
ONLY failing gate is the min-sep floor). The four offending pairs (centered
coords) are

    (-2,0,2,-1)--(2,-1,-2,2)   (-2,1,2,-2)--(2,0,-2,1)
    (-1,-1,2,0)--(0,2,-1,-1)   (-1,2,0,-2)--(2,-2,-1,2)

i.e. difference class ±(4,−1,−4,3)·w1^k with exact squared length
(23 − 4·sqrt33)/3 ≈ 0.007248 — the **0.0851 close pair** already documented
at n=71/72 (where it produced audit-REJECTED raw ties). Keeping both
endpoints of those 4 pairs is what buys the record its last edges:
exhaustive over all min-sep-clean 96-subsets of the 103-point ball
(2^6 close-pair resolutions × drop-1) gives **at most 414**.

Neighbor context (all DB record configs re-audited here):

| n | record | classes | min_sep clean | dirty (0.0851) |
|---|--------|---------|---------------|-----------------|
| 95 | 412 | 11 | 1 (= OUR tied class #10, flat family) | 10 (ball/hub family) |
| 96 | 418 | 1 | **0** | **1 (THE record)** |
| 97 | 423 | 5 (6 cfgs) | 4 classes incl. OURS (#2/#3) | 1 (n97[0], ball family) |
| 98 | 429 | 1 | 1 | 0 |

n=96 is the only n in 16..100 where EVERY known record config is
min-sep-dirty — that, not search weakness, is why the ladder missed it:
chase_n's audit gate (correctly) refuses to claim the only 418 in existence.

## Nesting (canon = 12-motion + translation canonical form, chase40lib/mlgraph)

- DB n97 inventory (6 configs, 423 edges) = **5 distinct classes**;
  n97[0] has degree hist `{5:1, 6:20, ..., 18:1}` with the same hub tower.
  **R96 = n97[0] minus its unique degree-5 vertex** (canon-verified;
  the dropped vertex is the (a,b)-coset (2,0) outlier). 418 = 423 − 5
  exactly as conjectured. Both are trace-ball densest-subsets (k=96, 97).
- **Our** tied n97 (423, subset:h1xh2c1) is canon-identical to DB n97
  records #2/#3 — a *different*, audit-clean class with **min degree 6**:
  no drop-1 path from our n97 to any 418 config.
- R96 minus any one of 20 degree-6 vertices lands in DB n95 record
  classes #0–#9 (the hub family). **Our** tied n95 (412) is class #10 —
  the *flat* family (max deg 16, no 18-hub, audit-clean). The two families
  coexist at 412; only the (dirty) hub family extends to 418.
- The n98 record (429, audit-clean) drop-2 does NOT reach R96's class.

## Rebuild routes (all pure-toolkit, structure-informed)

1. `drop-1 from our n97 423`: impossible (min degree 6). Ruled out.
2. Greedy drop-worst 361 → 96 in h2xh2: 414 (deterministic control).
3. `subset_ils` k=96 in h2xh2 (361 pts): plateaus 415–416 in 10 min × 4 seeds
   (rebuild_subset.py) — right ambient, still needle-in-haystack.
4. Hub-seeded / sigma-symmetric / hub-frozen searches in h2xh2
   (rebuild2–4.py): **418 hit** (`udg96_418edges_hubfrozen_k96_s1.*`),
   canon == DB class.
5. **THE RECIPE (cheapest, deterministic)**: exact trace ball
   {A²+3B²+11C²+33D² ≤ 288} (103 pts) + `subset_ils(ball, 96, seed=0,
   max_iters=400)` → **418 on the first seed in ~1 s**
   (`rebuild_traceball.py` → `udg96_418edges_traceball.*`), canon == DB
   class. Same recipe at k=97 reproduces n97[0]/423; k=95 gives 412.
6. Constrained CLEAN hunt (`clean_hunt.py`): conflict-aware (no pair
   < 0.2) densest-96 ILS over h2xh2 / h2xh2c1 / trace balls T ≤ 5, 6
   (4 × 25 min): best clean = see clean_hunt.log (≤ 417; no clean 418
   found). Combined with the ball exhaustive (414) and the DB's own
   uniqueness evidence, the audit-clean record at n=96 is in all
   likelihood 417 = our sweep best.

## Verification chain on the deliverable (`udg96_418edges_traceball.*`)

- `exact_edge_count` = 418 == target; n = 96; coords integer ML.
- `scripts/ml_coords.py`: **CERTIFIED ML** (418/418 exact unit pairs,
  max angular residual 2.6e-16, embed err 4.5e-16) — the config is a
  rigorous exact UDG (points pairwise distinct in exact arithmetic).
- `scripts/audit_config.py`: **exit 1 — min_sep 0.085146 < 0.2** (K23 0,
  GN exact). This failure is a property of the CLASS, not of our build.
- `udg.mlgraph.canon`: equals the DB record class (THEIR class reproduced,
  not copied — recipe is ambient + deterministic subset search).

## DB information used (declared)

- The *target* count 418 and its uniqueness (record inventory metadata).
- The *ambient/selection structure*: record ⊂ H(2) ⊕ w3·H(2), and more
  precisely ⊂ the exact trace ball T ≤ 4 (learned by coset/trace dissection
  of the DB config); the densest-k-subset rule was inferred, not copied.
- The *hub motif* (deg-18 saturated center) used to seed warm starts.
- The *nesting facts* (418 = n97[0] − deg-5 vertex; n95 hub family).
- The coordinates themselves were used ONLY as the canon sanity target
  (class comparison) and for the audit post-mortem — never copied into
  our configs.
