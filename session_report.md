# Unit Distance Problem — Research Session Report

## Phase 3 (computational probing)
- Grid popular-distance counts track n·r2(k)/2; winners (d²=65, 325) are sums of two
  squares with max representations. Triangular lattice beats square grid 1.5x (unit
  group 6 vs 4). Growth fits n^(1+c/loglog n), c drifting toward ~0.5+.
- U/n^(4/3) declines on lattices: constructions nowhere near the upper bound.

## Bisector energy (reflection-symmetry energy)
- E = Σ_reflections |matched pairs|². Grid/tri-lattice: E ~ n^2.5 (exponent measured
  2.60→2.54). 1% jitter collapses E by 26x to ~n^2.0 (trivial). Random: trivial.
- Insight: unit-distance-rich sets and symmetry-rich sets coincide; both are brittle.

## Search (n=40)
- Free-coordinate annealing fails (rich configs are measure-zero).
- Tolerance exploit discovered: tangential near-coincident clusters fake K_{2,3}'s.
  Fixed with min-separation 0.2 + K_{2,3} audit + Gauss-Newton realizability proof.
- RESULT: 40 points, 132 exact unit distances (residual 1e-27, min sep 0.283).
  World record (Engel et al. 2024, beam search on Moser lattice): 137.
  We hit 96% of record from random init, no lattice prior, minutes of CPU.
- Structure: 4 triangular-lattice families. Two sit exactly on Moser lattice
  directions (0°, 16.78°); two float at hinge angles (0.90°, 34.08°) ≠ exact
  ML angles (33.56°). Interpretation: continuous search finds the FLEXIBLE
  skeleton; the last ~5 edges require locking hinges at exact arithmetic angles
  so extra coincidences fire. Density's last few percent IS number theory.

## Literature status (verified June 2026)
- Best upper bound: ~1.94 n^(4/3) (Ágoston–Pálvölgyi 2022). Asymptotic exponent
  4/3 unimproved since 1984.
- u(n) known exactly only for n ≤ 15. Densest-known to n=100: Engel–Hammond-Lee–
  Su–Varga–Zsámboki (arXiv:2406.15317), all on Moser lattice Z<1,ω1,ω3,ω1ω3>.
  Many records are Minkowski sums of small UDGs.
- BREAKING (2026): a new construction (Sawin / OpenAI collaboration) reportedly
  beats Erdős's conjectured n^(1+c/loglog n) lower bound using projections of
  higher-degree number-field lattices (cf. Ruhland 2024: lattice families with
  unboundedly many unit vectors; Radchenko 2021). The conjecture in its classic
  form (grid-type bound optimal) appears to be FALSE on the lower-bound side;
  the gap to n^(4/3) remains open.

## Files
- udg40_132edges.csv — coordinates of the 40-point, 132-edge configuration.
