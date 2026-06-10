# RESULTS — running log of audited results

Every entry must reference a config CSV + audit log. No unaudited numbers.
Records column = densest-known per arXiv:2406.15317v3 Table 2.

| date | n | edges (audited) | record | % | method | config | notes |
|------|---|-----------------|--------|---|--------|--------|-------|
| 2026-06 (prior session) | 40 | 132 | 137 | 96.4% | Metropolis circle-intersection, 8 seeds × 150k | data/udg40_132edges.csv | 4 tri families; 2 on ML angles, 2 floating |
| 2026-06-10 | 40 | 132 | 137 | 96.4% | repro campaign: udg package, 16 seeds × 150k, all 16 audited | runs/n40-repro/best.csv | seed 6 reproduces shipped config bit-for-bit (port exact); seed 12 = a DISTINCT 132-edge config (min_sep 0.2421); distribution 122–132 |
| 2026-06-10 | 30 | 91 | 93 | 97.8% | 16 seeds × 150k, all audited | data/udg30_91edges.csv | Tier-1 ✓ (needs ≥89); distribution 87–91 |
| 2026-06-10 | 50 | 177 | 183 | 96.7% | 16 seeds × 250k, all audited | data/udg50_177edges.csv | Tier-1 ✓ (needs ≥174); distribution 154–177 |
| 2026-06-10 | 70 | 266 | 281 | 94.7% | 12 seeds × 400k, all audited | runs/n70/best.csv | best raw-search result; closed by forcing (below). Extension 16×800k: 264; re-anneals stuck at 266 |
| 2026-06-10 | 40 | **136** | 137 | **99.3%** | coincidence forcing (3-degenerate near-miss cluster → 135) + warm exploit loop (+1) | data/udg40_136edges.csv | TWO distinct 136 configs (also data/udg40_136edges_b.csv from the seed-12 132); both fully ML-aligned (9/9 dirs); hinge hypothesis CONFIRMED: 3 misses at identical &#124;d−1&#124;=1.368e-2 fired simultaneously under GN augmentation |
| 2026-06-10 | 50 | **179** | 183 | 97.8% | coincidence forcing (2-cluster) from 177 | data/udg50_179edges.csv | fully ML-aligned after forcing |
| 2026-06-10 | 70 | **274** | 281 | **97.5%** | coincidence forcing: 8-degenerate cluster fired at once from 266 | data/udg70_274edges.csv | Tier-1 ✓ (needs ≥267); fully ML-aligned (9/9); search alone was stuck at 266 |
| 2026-06-10 | 40 | 132 | 137 | 96.4% | CONTROL ARM (hinge baseline): plain warm-start search from udg40_132edges.csv, T0∈{0.03,0.08,0.15}×steps∈{30k,100k}×4 seeds (T1=0.01), 24 runs + 12 instrumented | runs/hinge/control_plateau_seed*.csv | NULL RESULT: 0/24 exceeded 132 (never reached 133 at tol in ~1.56M steps); chains walk a neutral Δ=0 plateau (~350 accepted/30k steps, 0 up, 0 down; T0-invariant trajectories); 4 distinct 132-edge plateau configs audited+saved; near-miss floor of input AND all plateau configs identical: 3 pairs at &#124;d−1&#124;=1.368e-2 (d=0.9863168633, rigid-skeleton invariant), nothing below 1e-2 |
