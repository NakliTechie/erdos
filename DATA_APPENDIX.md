# Data Appendix — Measured Results (all reproducible from scripts in code bundle)

## A. Square grid popular-distance counts (exp1_grid.py)
m, n=m², best d², U(n), U/n, U/n^(4/3):
5,25,5,48,1.920,0.6566 | 8,64,5,168,2.625,0.6563 | 12,144,25,456,3.167,0.6042
17,289,25,1136,3.931,0.5945 | 24,576,65,2832,4.917,0.5909 | 34,1156,65,6672,5.772,0.5499
48,2304,325,15864,6.885,0.5213 | 60,3600,325,28200,7.833,0.5111
Implied c in U=n^(1+c/loglog n): 0.237→0.529, still rising at n=3600.

## B. 50×50 grid: count vs n·r2(k)/2 prediction (exp2_structure.py)
d²=325: count 17680, r2=24, ratio 0.589 | d²=65: 16144, r2=16, ratio 0.807
d²=425: 15640, r2=24, 0.521 | d²=85: 15440, r2=16, 0.772 | d²=125: 14688, r2=16, 0.734
(ratio <1 = boundary loss, grows with |vector|.)

## C. Triangular lattice (Eisenstein form x²+xy+y², 2500 pts)
norm=91: 22120 pairs (8.85/n) | 133: 20720 | 217: 18112 | 49: 18107 | 247: 17464
vs square grid 50×50 best 17680 (7.07/n) → tri beats square ~1.25–1.5×.

## D. Bisector energy, n=1600 (exp3): E, E_nontrivial, max line multiplicity
square grid: 66,748,264 / 66,352,916 / 800
tri lattice: 67,742,200 / 67,255,450 / 780
random:      1,279,200 / 0 / 1   (trivial: = #pairs)
perturbed grid (1% jitter): 2,573,622 / 1,378,169 / 95
Grid scaling (exp run): local exponent 2.601 (n=196) → 2.535 (n=3136), → 2.5 (=M·n², M=√n).

## E. n=40 search history (exp4–exp6)
- Free annealing: best 13 edges (failure mode: collapse or no signal).
- Circle-intersection search, NO min-sep: "400 edges" — tolerance exploit
  (tangential near-coincident clusters; distance error second-order).
- With min-sep 0.2 + audits, 8 seeds × 150k steps: 122,126,129,129,130,130,130,132.
- Best 132: residuals — 39 edges <1e-15, 15 in [1e-15,1e-12), 78 in [1e-12,1e-10);
  Gauss–Newton projection: total residual 1.1e-27, all 132 edges exact at 1e-12,
  max point movement 4.3e-11, min separation 0.283. K_{2,3} violations: 0.

## F. Structure of the 132-edge config (exp8, exp9)
Degree sequence: 10,9,8,8,8,7×19,6×12,5,4,4,3 (avg 6.6 > tri-lattice interior 6).
Distinct distances among 780 pairs: 134. Distinct radii from centroid: 40 (no
concentric-circle structure). 12 undirected edge directions in 4 families of 3
(each family internally 60° apart): offsets ≈ {0.04, 0.94, 16.87, 34.12}°.
Moser lattice has 9 directions: {0, 16.78, 33.56, 60, 76.78, 93.56, 120, 136.78,
153.56}°. Best alignment: 6/12 of ours match (the 0° and 16.78° families);
the 0.90° and 34.08° families are floating hinges (34.08 ≠ arccos(5/6)=33.557).
→ Flexible-skeleton hypothesis (handoff §2.4, §5.A.1).

## G. Literature anchors (verified via web, June 2026)
- Records table source: arXiv:2406.15317v3 Table 2 (n=1..100). DB: codeberg.org/zsamboki/dbs-udg
- Upper bound 1.94n^(4/3): arXiv:2006.06285.
- Small-n exact values: arXiv:2412.11914 (and Schade 1993).
- 2026 lower-bound development: MathWorld Erdős Unit Distance Problem page
  (updated May 2026) describes a Sawin/OpenAI number-field construction beating
  the conjectured grid-type bound; primary PDF at cdn.openai.com (unit-distance-proof).
  RE-VERIFY before relying on it.
