# Batch B — hinge locking: verified synthesis report

Date: 2026-06-10. Verifier: independent adversarial re-audit (all numbers below
re-measured from disk in fresh processes; nothing taken from the arm reports).

## Design recap

Hypothesis (HANDOFF §2.4/§5.A.1, plan/hinge-design.md): the 132-edge n=40 config is a
*flexible framework skeleton*; locking its two floating direction families (~0.94°, ~34.12°)
at exact algebraic angles should fire additional unit coincidences. Three arms: CONTROL
(plain warm-start search, no locking), HOMOTOPY (strategy a: rotate-and-project), FLEX
(strategy b: rigidity-matrix flex following), all from `data/udg40_132edges.csv` (re-audited:
132 edges, passed, residual 1.15e-27).

## Independent re-audit (the verification gate)

All **53** CSVs in `runs/hinge/` re-audited via `scripts/audit_config.py` (fresh processes,
4-way parallel): **53/53 PASSED**. K23 = 0 everywhere; GN residuals 4.0e-28–1.3e-27 (all
< 1e-24); all edges exact at 1e-12; min_sep range 0.2022–0.2918; **min_sep_after == min_sep
on every config** — the point-merging tolerance exploit did NOT reappear through the GN
repair step. Baseline `data/udg40_132edges.csv` and references `data/udg40_136edges{,_b}.csv`
also re-audited: passed.

Re-audited counts (vs claims — all match):

| count | configs | claim check |
|-------|---------|-------------|
| 136 | homotopy_fire_search_s0–3, flex_135A_search_seed401–404, flex_135B_search_seed401–404, flex_135A_reheat_search_seed411–414 (16 files) | OK |
| 135 | homotopy_fire135, flex_cross_p5_11/p9_26/p10_39 (event A), flex_cross_p5_20/p12_26/p14_19 (event B) | OK |
| 133 | flex_cross_p2_39, flex_cross_p7_14 — lattice match 3/13, genuinely NON-ML | OK |
| 132 | control_plateau_seed0–3, all homotopy locked+search arms, flex_fam1/fam3_followed, flex_all_locked, mlaligned | OK |

## Per-arm outcomes

| arm | locked? | converged? | best audited | vs control |
|-----|---------|-----------|--------------|------------|
| CONTROL (no locking) | — | — | 132 (0/24 grid runs improved; 4 distinct plateau configs verified distinct from input and each other, multiset diffs 0.16–0.24) | baseline: null |
| HOMOTOPY f094 alone | yes — genuine (lattice 9/9 after lock, verified) | 33 incr, 0 halvings | 132 locked; search seeds bit-identical to lock (md5: 5 identical files, verified) | = control |
| HOMOTOPY f3412 alone | NOMINAL ONLY — pure gauge (lattice stays 6/12, verified) | 12 incr | 132 | = control |
| HOMOTOPY sequential (both orders) | yes (9/9 verified both orders) | 0 halvings | 132 locked; searches bit-identical (md5 verified) | = control |
| HOMOTOPY + bisection FIRE at theta*=0.011350713735° | n/a (fire point ≠ lock point) | — | **135 → 136 (4/4 seeds)** | +4 |
| FLEX follow (fixed-frame, fam1/fam3) | yes but overshoots (gauge artifact) | 9/6 steps | 132; 0/8 searches improved | = control |
| FLEX full 1-D path scan + crossings | n/a | scan spans s ∈ [−0.402, +0.946] | **133/135 at crossings → 136 (12/12 seeds from the 135s)** | +4 |

## Flex-dimension finding (re-measured)

Rigidity matrix of the input 132-config: **flex_dim = 1**, exactly. Re-computed SVD tail:
smallest structural SV 5.64e-1, flex SV 1.66e-11, trivial motions ~9e-16 — a 3.4e10 clean
gap, matching the FLEX arm's claim. The fired configs consume the flex: homotopy_fire135
and all 136s re-measure at **flex_dim = 0** (rigid). The two non-ML 133s are also rigid
(flex_dim = 0, newly measured here).

## Near-miss movement (re-measured)

Input 132: 648 non-edge pairs; bands [1e-9,1e-6)=0, [1e-6,1e-4)=0, [1e-4,1e-2)=0,
[1e-2,5e-2)=3 — exactly the triple (9,26),(5,11),(10,39) at d=0.9863168633
(|d−1| = 1.368314e-2, one algebraic distance in triplicate). Under locking:

- ML-locked endpoint (homotopy_f094_locked): triple OVERSHOOTS to d=1.000173 (+1.726e-4).
- Gauge-corrected simultaneous lock (flex_all_locked): triple at d=0.999985 (1.53e-5).
- Bisected fire point theta* = 0.011350713735°: all three exactly unit → audited 135 keeping
  all 132 original edge index-pairs (verified) with new edges exactly {(5,11),(9,26),(10,39)}.

Minor discrepancy found (immaterial): the CONTROL report's narrative orders the next-nearest
misses as (2,39)@6.70e-2 then the 0.91218 orbit @8.78e-2; pair (7,20) at |d−1|=8.63e-2
(d=1.0863) actually sits between them. Band counts unaffected.

## Congruence census (re-computed, distance-multiset tol 1e-6)

- The two 135 triple-fire classes (s=−0.040 vs s=+0.570) are mutually NON-congruent
  (multiset diff 0.132). **Cross-route check: the homotopy fire135 and the flex event-A 135
  are the SAME config to 3.8e-14** — two independent methods (regula-falsi on the lock
  homotopy vs the rigidity-flex path scan) land on an identical coincidence point.
- Among all 16 saved 136s + the 2 forcing-route references: **9 distinct congruence classes**.
  homotopy_fire_search_s0 ≅ flex_135B_search_seed402 ≅ data/udg40_136edges.csv;
  homotopy_fire_search_s3 ≅ data/udg40_136edges_b.csv; the other **7 classes are NEW**
  136-edge configs. All 136s: lattice 9/9 ML-locked, rigid, closest non-unit |d−1| = 8.18e-2.

## Negative results (findings, per HANDOFF §5.B framing)

1. **Plain search cannot leave 132**: 0/24 control runs improved in ~1.56M steps; all accepted
   moves are Delta=0 plateau drift; T0 in {0.03, 0.08, 0.15} has zero effect (verified distinct
   plateau endpoints, all 132).
2. **Locking alone fires nothing.** Every locked-but-unfired 132 config (4 homotopy arms,
   2 flex follows, flex_all_locked) stays at 132, and 0/28 warm searches from them improved
   (search outputs bit-identical to warm starts, checksum-verified). The ML lock point is NOT
   the firing point — the triple crosses unit at theta* ~ 0.0114° and the ML endpoint overshoots it.
3. **Locking a family spanning 39/40 vertices is pure gauge** (f3412): relative offsets
   unchanged; verify locks via re-classification, never via the absolute family angle.
4. **The 136 basin is exhausted**: 8 homotopy extension searches + 4 flex reheats (up to
   T0=0.5, 60k) all return bit-identical 136s; no near-miss below 8.18e-2 remains. 137 likely
   needs a different skeleton (Minkowski/lattice moves), not flex following.

## Verdict on the hinge-locking hypothesis: **SUPPORTED** (with a mechanistic refinement)

132 → 135 → **136 audited** (99.3% of record 137), against a control arm pinned at 132
across 24 runs. Both locking strategies independently reach the same 135 and together produce
9 distinct 136 classes (7 new). The flexible-skeleton picture is confirmed quantitatively:
flex_dim = 1, the flex couples to all four family angles, per-vertex flex motion peaks at the
firing-pair vertices, and the fired edges consume the flex (flex_dim → 0). The refinement:
new edges fire ALONG the flex at unit-coincidence angles (theta* ~ 0.0114°, s = +0.570), *near*
but not *at* the exact ML lock angle — "lock the hinge" is operationally "walk the flex and
bisect the coincidence", and two of the crossings (the 133s) are not ML-embeddable at all
(lattice 3/13), feeding the non-ML research thread (§5.A.2).

## Artifacts

- This report: `runs/hinge/REPORT.md`
- Re-audit transcripts: `/tmp/hinge_verify/*.txt` (53 PASSED) — regenerate any time via
  `uv run python scripts/audit_config.py <csv>`
- Arm summaries: `control_summary.json`, `homotopy_summary.json`, `homotopy_fire_summary.json`,
  `flex_exp2_summary.json`, `flex_scan_summary.json`, `flex_135_search_summary.json`
- Best new artifact: `runs/hinge/homotopy_fire_search_s2.csv` (audited 136, unique new
  congruence class, min_sep 0.2918)
