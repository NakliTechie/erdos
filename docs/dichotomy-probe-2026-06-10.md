# Dichotomy probe: bisector energy across the forcing transition

HANDOFF §5.B asks whether high unit counts FORCE high reflection-symmetry
(bisector) energy. Today's coincidence-forcing runs produced matched pairs of
audited configs at equal n — same skeleton, before and after hinge locking —
which is exactly the controlled comparison the question wants.

Measurement: `udg.bisector.bisector_energy_float` (bucketed, decimals 4 and 6 —
results identical at both, so the structure is real, not a bucketing artifact).
E_nt = sum of multiplicity² over bisector lines shared by ≥2 point pairs.

| config | n | unit edges | E_nt | max line mult |
|---|---|---|---|---|
| n40 pre-force (floating) | 40 | 132 | 578 | 4 |
| n40 post-force (ML-locked) | 40 | 136 | **1760** | 13 |
| n40b pre (floating) | 40 | 132 | 543 | 3 |
| n40b post (ML-locked) | 40 | 136 | **1681** | 15 |
| n30 best (provably non-ML) | 30 | 91 | 294 | 3 |
| n50 pre | 50 | 177 | 932 | 6 |
| n50 post (ML-locked) | 50 | 181 | **3292** | 17 |
| n70 pre | 70 | 266 | 1984 | 6 |
| n70 post (ML-locked) | 70 | 277 | **8323** | 24 |
| random n=40 / n=70 | 40/70 | 0 | 0 | 1 |

## Reading

1. **Locking the hinges multiplies symmetry energy 3.0–4.2× while gaining only
   +2–4% edges.** The marginal record edges are purchased with a disproportionate
   explosion of reflection structure — sharp empirical support for the
   structure-theorem intuition ("either E is small and unit distances are few, or
   the set is lattice-like"), and a quantitative version of the prior session's
   "density's last few percent IS number theory".
2. **The pre-forcing configs are the interesting middle of the Pareto frontier:**
   96–95% of record density at 1/3–1/4 of the locked symmetry energy. Flexible
   skeletons buy density cheaply in symmetry until the very end.
3. **The n=30/91 config (certified non-ML as floated) is the extreme point:**
   97.8% of record density with the lowest symmetry energy of any dense config
   measured here. If a *record-density* config could stay low-E like this, that
   would be the counterexample HANDOFF §5.B calls "even more interesting"; the
   live prediction is instead that locking it to 93 will spike its E like the
   others. Either outcome is informative — this is the next measurement when the
   n=30 chase lands.

Caveat: E_float is a bucketed approximation for relative comparison at equal n
(documented in `udg/bisector.py`); cross-n comparisons here are qualitative only.
Reproduce: the table script is inline in the session log; all configs in `data/`.
