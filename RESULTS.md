# RESULTS — running log of audited results

Every entry must reference a config CSV + audit log. No unaudited numbers.
Records column = densest-known per arXiv:2406.15317v3 Table 2.

| date | n | edges (audited) | record | % | method | config | notes |
|------|---|-----------------|--------|---|--------|--------|-------|
| 2026-06 (prior session) | 40 | 132 | 137 | 96.4% | Metropolis circle-intersection, 8 seeds × 150k | data/udg40_132edges.csv | 4 tri families; 2 on ML angles, 2 floating |
| 2026-06-10 | 40 | 132 | 137 | 96.4% | repro campaign: udg package, 16 seeds × 150k, all 16 audited | runs/n40-repro/best.csv | seed 6 reproduces shipped config bit-for-bit (port exact); seed 12 = a DISTINCT 132-edge config (min_sep 0.2421); distribution 122–132 |
