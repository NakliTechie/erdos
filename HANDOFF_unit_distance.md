# HANDOFF: Erdős Unit Distance Problem — Computational Research Program

**For:** Claude Code, running on local machine (Apple Silicon available; MLX/GPU optional)
**From:** Prior research session (claude.ai, June 2026)
**Owner:** Bhai (NakliTechie)
**Goal:** Continue an empirical attack on the unit distance problem: chase densest-known
unit-distance graph (UDG) records for 30 ≤ n ≤ 100, and probe the structure-vs-randomness
dichotomy via bisector energy.

---

## 1. Problem statement and verified state of the art (checked June 2026)

u(n) = max number of pairs at distance exactly 1 among n distinct points in R².

- **Upper bound:** u(n) ≤ ~1.94·n^(4/3) (Ágoston–Pálvölgyi 2022, arXiv:2006.06285).
  The 4/3 exponent is unimproved since Spencer–Szemerédi–Trotter 1984.
- **Lower bound:** Erdős 1946: n^(1+c/log log n) via integer grid + sums-of-two-squares.
  **2026 development:** a construction attributed to Sawin / an OpenAI collaboration
  reportedly BEATS the conjectured grid-type bound using projections of higher-degree
  number-field lattices (see also Ruhland arXiv:2410.16172 — lattice families with
  unboundedly many unit vectors; Radchenko 2021, DCG 66:269–272). Verify current
  status with a web search before citing; this postdates training data.
- **Exact values:** u(n) known only for n ≤ 15 (Schade 1993; confirmed by later work).
- **Records table (densest KNOWN, Engel–Hammond-Lee–Su–Varga–Zsámboki,
  arXiv:2406.15317v3, Table 2).** Selected values (n: edges):
  20:54, 25:72, 30:93, 35:114, 40:137, 45:160, 49:180, 50:183, 55:206, 60:231,
  64:252, 70:281, 75:306, 80:332, 85:360, 90:385, 95:412, 100:439.
  Full table in the paper. Their code + 60M-graph DB: https://codeberg.org/zsamboki/dbs-udg
- **Key structural fact:** all known maximally dense UDGs embed in the **Moser lattice**
  ML = Z⟨1, ω₁, ω₃, ω₁ω₃⟩ ⊂ C, where ω₁ = exp(iπ/3), ω₃ = exp(i·arccos(5/6)).
  ML has exactly 18 unit vectors (9 undirected edge directions). Max vertex degree
  on ML is therefore 18. Many records are Minkowski sums of small UDGs
  (e.g. densest-known n=49 = sum of two 6-wheels; n=64 = sum of 6 edges).

## 2. What the prior session accomplished

1. **Grid statistics:** popular-distance count on m×m grids tracks n·r₂(k)/2;
   winning d² are sums of two squares with many representations (65, 325, ...).
   Triangular lattice beats square grid by ~6/4 (Eisenstein vs Gaussian unit group).
2. **Bisector energy** E = Σ over distinct perpendicular-bisector lines of
   (#point-pairs sharing that bisector)² — equivalently Σ over reflections σ of
   matched-pair counts squared. Measured: grid E ~ n^2.5 (exponent 2.60→2.54
   converging), 1% jitter collapses it ~26× to ~n^2.0, random is trivial.
   Unit-distance richness and reflection-symmetry energy rise and fall together.
3. **Search result:** 40 points, **132 exact unit distances** (record: 137) found by
   Metropolis search over circle-intersection moves from random init, ~6 seeds ×
   120k steps, minutes of CPU. Coordinates in `udg40_132edges.csv` (same folder).
4. **Structure of our 132-config:** four triangular-lattice families; two exactly on
   ML directions (0°, 16.78° = arcsin(1/√12)), two at "floating hinge" angles
   (≈0.90°, ≈34.08° ≠ exact 33.56°). Hypothesis: continuous search finds the
   flexible framework skeleton; final edges require locking hinges at exact
   algebraic angles so additional coincidences fire.

## 3. Critical pitfalls (hard-won; do not skip)

- **Tolerance exploit:** with edge tolerance τ and no minimum separation, search
  discovers near-coincident point clusters spread TANGENTIALLY along unit circles
  (distance error is second-order, ~δ²/2, so δ=1e-6 separation passes τ=1e-9).
  This fakes K_{m,m} subgraphs and absurd counts (we saw "400 edges" at n=40).
  **Mitigations (all three, always):**
  1. Hard min-separation ≥ 0.2 on every candidate placement.
  2. K_{2,3} audit: no two vertices may share ≥ 3 common unit-neighbors
     (true UDGs are K_{2,3}-free since two unit circles meet in ≤ 2 points).
  3. Exact-realizability audit: damped Gauss–Newton on ALL claimed edges,
     minimize Σ(|p_i−p_j|−1)²; accept only if total residual < 1e-24 and every
     edge is within 1e-12 after projection. (132/132 passed for our config.)
- **Free-coordinate annealing fails.** Rich configs are measure-zero; moves must be
  geometric: place a point at an intersection of two unit circles around existing
  points (every move creates ≥2 exact edges; surplus edges arise via coincidence).
- **Cluster-mean angles are noisy** (~0.05–0.5°). For lattice identification use
  exact algebra or high-precision fitting, not histogram means.
- **Counting:** use squared distances and exact integer arithmetic wherever points
  are lattice points; use τ=1e-9 with float64 otherwise, and never report a number
  that has not passed the three audits.

## 4. Code assets (WORKING SCRIPTS INCLUDED — see code_bundle.zip)

Companion files shipped with this handoff:
- `code_bundle.zip` — exp1–exp9 scripts (tested, numpy-only) + udg40_132edges.npy
- `DATA_APPENDIX.md` — all raw measured numbers (use as regression baselines)
- `udg40_132edges.csv` — the 132-edge configuration coordinates
- `session_report.md` — narrative summary of the prior session
Prefer adapting the scripts over rebuilding; module sketch below is the map:

- `popular(points)` — pairwise squared-distance multiplicity counter (chunked).
- `bisector_energy(P)` — normalize integer line triples (2Δx, 2Δy, |b|²−|a|²) by
  gcd+sign, np.unique, Σcounts². For triangular lattice use coords (a, b√3) with
  integer (a,b) and the form x²+3y² to stay exact.
- `search(n, steps, seed)` — Metropolis over circle-intersection moves:
  pick vertex k, pick pair (a,b) with |ab|<2, move k to a random intersection of
  unit circles around a,b; reject if min-sep violated; accept by Δ(unit count)
  with geometric temperature schedule T: 1.2 → 0.015.
- `audit(P)` — the three checks from §3.
- `lattice_id(P)` — compare edge-direction multiset against ML's 9 directions
  {0, 16.78, 33.56, 60, 76.78, 93.56, 120, 136.78, 153.56}° up to global rotation.

## 5. Research directions, ranked

**A. Record chasing (most concrete).**
Current gap at n=40: ours 132 vs record 137. Ideas, in order:
1. **Hinge locking:** take a found flexible config, identify floating families,
   snap their rotation to nearby exact ML angles (or other algebraic coincidence
   angles), re-run local search from there. Tests the central hypothesis directly.
2. **Hybrid move set:** add Moser-lattice moves (the 18 unit vectors as discrete
   steps) alongside continuous circle-intersection moves. The continuous moves
   explore angles the Rényi team's lattice-locked beam search cannot — this is
   the genuine edge over their method. Any n where we match or beat Table 2
   with a NON-ML-embeddable config is a publishable finding.
3. **Minkowski-sum seeding:** initialize with sums of small dense UDGs
   (triangle, 6-wheel, Moser spindle) then perturb. Records at 39, 49, 64, 98
   are Minkowski sums; intermediate n may have undiscovered sum-based records.
4. Scale: parallel seeds (multiprocessing), n up to 100; budget hours not minutes.
   Incremental unit-count updates make each move O(n).
5. Also explore lattices L_k = Z⟨1, ω₁, ω_k, ω₁ω_k⟩ for k ≠ 3
   (ω_k = exp(i·arccos(1−1/2k))) — the Rényi team checked some, not all, and
   Ruhland showed unit-vector counts in this family are unbounded.

**B. Dichotomy probing (theory-adjacent).**
Penalized search: maximize unit count − λ·(bucketed bisector energy). Map the
Pareto frontier (unit count vs symmetry energy). If high unit counts FORCE high
symmetry energy at every λ, that is empirical support for a structure theorem:
"either E is small and unit distances ≤ n^(4/3−ε), or the set is lattice-like."
A counterexample (low-E, high-unit-count config) would be even more interesting.

**C. New-construction reconnaissance.**
Fetch and study the 2026 number-field construction (search: Sawin unit distance
number fields / OpenAI unit distance). Implement small instances: rank-4+ lattices
in C with many unit vectors, projected sections. Compare unit counts per point
against ML patches at equal n. This is where the asymptotic frontier now is.

## 6. Success criteria

- Tier 1: reproduce ≥ 95% of record at three values of n ∈ {30, 50, 70}. (Baseline
  sanity — prior session already did this at n=40.)
- Tier 2: match any Table 2 record; or produce any audited config NOT embeddable
  in ML at record density.
- Tier 3: beat any Table 2 value for 15 < n ≤ 100. If achieved: re-verify with
  exact arithmetic (sympy or interval arithmetic), then check against the
  codeberg DB before claiming novelty, and consider contacting the authors
  (corresponding: zsamboki.pal@renyi.hu).

## 7. Suggested first session plan for Claude Code

1. Rebuild §4 modules as a small package with tests (audit functions first).
2. Reproduce n=40 ≥ 132 from scratch (validates the pipeline end to end).
3. Implement hinge-locking (§5.A.1) and measure: does 132 → >132?
4. Scale seeds/steps on n ∈ {30, 50}; compare to records 93 and 183.
5. Write results to a running RESULTS.md with configs saved as CSV + audit logs.

Compute notes: everything so far was CPU numpy. For scale, vectorize across seeds
(batch Metropolis) before reaching for MLX/GPU; the per-move work is O(n) and
memory-light. Exact verification (sympy) only on final candidates.
