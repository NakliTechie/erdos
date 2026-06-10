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

## Status (2026-06-10): all four working records TIED

| n | ours | record (Engel et al.) | iso classes found / known | config |
|---|------|-----------------------|---------------------------|--------|
| 30 | **93** | 93 | 2 / 2 | `data/udg30_93edges.csv` (+`_b`) |
| 40 | **137** | 137 | 1 / 1 | `data/udg40_137edges.csv` |
| 50 | **183** | 183 | 5 / 5 | `data/udg50_183edges.csv` |
| 70 | **281** | 281 | 3 / 3 | `data/udg70_281edges.csv` (+`_b`,`_c`) |

Every config passes the float three-audit AND exact certification in ℚ(√3,√11)
(integer Moser-lattice coordinates, `data/mlcoords/*.json`). Verified byproducts:
n=42/143, n=51/188 (ties Table 2), n=71/286, n=72/291. Beyond-record hunts
(94/138/184/282) all came up dry — consistent with Table 2 being ML-optimal at
these n. Class identity vs the Engel FigShare DB is unchecked (WAF-gated download).

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

## Audit discipline

No edge count is reported unless it passes **all three**:
1. hard minimum separation ≥ 0.2,
2. K_{2,3}-freeness (two unit circles meet in ≤ 2 points),
3. exact realizability: damped Gauss–Newton on all claimed edges, total residual
   < 1e-24 and every edge within 1e-12 after projection.

Without these, tolerance-exploiting near-coincident clusters fake absurd counts
(the prior session saw a bogus "400 edges" at n = 40).

## Run

```sh
uv run pytest                                   # full test suite
uv run python scripts/run_search.py --n 40 --seeds 16 --steps 150000 --out runs/n40/
uv run python scripts/audit_config.py data/udg40_132edges.csv
```
