# erdos — unit distance problem, empirically

Computational research program on the **Erdős unit distance problem**: chase
densest-known unit-distance-graph (UDG) records for 30 ≤ n ≤ 100, and probe the
structure-vs-randomness dichotomy via bisector energy.

u(n) = max number of point pairs at distance exactly 1 among n distinct points in R².
Upper bound ~1.94·n^(4/3) (Ágoston–Pálvölgyi 2022); densest-known values to n = 100
in Engel–Hammond-Lee–Su–Varga–Zsámboki, arXiv:2406.15317, Table 2 — all of which
embed in the Moser lattice. This repo attacks from the continuous side: Metropolis
search over circle-intersection moves, no lattice prior, with a hard audit pipeline
so every reported count is exactly realizable.

**Starting point** (prior research session, June 2026): 40 points, **132 audited
exact unit distances** (record: 137) from random initialization in minutes of CPU —
see [session_report.md](session_report.md). Central hypothesis: continuous search
finds a *flexible framework skeleton*; the last few edges require locking floating
"hinge" angle families at exact algebraic angles ([HANDOFF_unit_distance.md](HANDOFF_unit_distance.md) §5.A.1).

## Status (2026-06-10): the ENTIRE frontier 16 ≤ n ≤ 100 reproduced — 87/87

Every densest-known record (Engel et al., Table 2 / their 60M-graph DB) for
**every n from 16 to 100** has been independently re-derived and exactly
certified: 86/87 audit-clean, plus n=96 under the exact-certificate track
(close-pair flag — see audit discipline below). Zero new records: every
beyond-record hunt came up dry, consistent with Table 2 being ML-optimal on
16–100. Class identity verified against the Engel DB (downloaded locally,
1.32 GB): at every checked n our isomorphism classes are exactly theirs.

Flagship configs in `data/` (n = 30, 40, 50, 64, 70, 90, 96, 98 + originals);
the full per-n table is in [docs/sweep-2026-06-10.md](docs/sweep-2026-06-10.md);
running log in [RESULTS.md](RESULTS.md). Notable reverse-engineered
constructions: n=64 = flattened 6-cube (six unit generators); n=98 = (3-cube
minus a corner) ⊕ 4-cube, a 7-generator {0,1}-sum; n=96 = densest-96-subset of
the 103-point Galois trace ball |z|² + |σz|² ≤ 4.

The hinge hypothesis was **confirmed and refined** along the way: the 132-edge
skeleton has exactly one internal flex; new edges fire *along* the flex at
unit-coincidence angles, and "coincidence forcing" (GN-realizing degenerate
near-miss clusters, [scripts/force_coincidences.py](scripts/force_coincidences.py))
converts that into an automatic method — see
[docs/hinge-REPORT-2026-06-10.md](docs/hinge-REPORT-2026-06-10.md) and
[docs/dichotomy-probe-2026-06-10.md](docs/dichotomy-probe-2026-06-10.md).

## Layout

- `HANDOFF_unit_distance.md` — the research program: state of the art, pitfalls, ranked directions
- `DATA_APPENDIX.md` — measured baselines (regression targets for the test suite)
- `code/` — original exp1–exp9 scripts from the prior session (ground truth for ports)
- `src/udg/` — the package: counting, bisector energy, audits, Moser lattice, search
- `data/` — audited configurations (`udg40_132edges.csv` = the 132-edge config)
- `scripts/` — campaign drivers (`run_search.py`, `audit_config.py`)
- `RESULTS.md` — running log of audited results

## Audit discipline (dual-track, decision 2026-06-10)

**Float-search track** — no edge count from continuous/float search is reported
unless it passes **all three**:
1. hard minimum separation ≥ 0.2,
2. K_{2,3}-freeness (two unit circles meet in ≤ 2 points),
3. exact realizability: damped Gauss–Newton on all claimed edges, total residual
   < 1e-24 and every edge within 1e-12 after projection.

Without these, tolerance-exploiting near-coincident clusters fake absurd counts
(the prior session saw a bogus "400 edges" at n = 40).

**Exact-certificate track** — a config with integer Moser-lattice coordinates
whose edges are *exactly* unit and whose points are *exactly* distinct in
ℚ(√3,√11) (`scripts/ml_coords.py` CERTIFIED) is a rigorous UDG regardless of its
float min-sep; the tolerance exploit is impossible under the certificate. Such
configs count as records with a **close-pair flag** when min-sep < 0.2. (Needed
at n = 96, whose unique record class has four pairs at exact distance²
(23−4√33)/3 ≈ 0.0851² — the literature counts it; so do we, flagged.)

## Run

```sh
uv run pytest                                   # full test suite
uv run python scripts/run_search.py --n 40 --seeds 16 --steps 150000 --out runs/n40/
uv run python scripts/audit_config.py data/udg40_132edges.csv
```
