# Reproducing the densest-known unit-distance graphs for 16 ≤ n ≤ 100: exact certificates, coincidence forcing, and the structure of record configurations

**Chirag Patnaik** (NakliTechie)
*Research conducted with and by Claude (Anthropic) multi-agent sessions; see §10.*
Draft v0.2 — 2026-06-10. Code, data, and exact certificates: https://github.com/NakliTechie/erdos

## Abstract

Let u(n) be the maximum number of unit distances among n points in the plane. The
densest known configurations for n ≤ 100 are due to Engel, Hammond-Lee, Su, Varga
and Zsámboki, who found them by diverse beam search restricted to the Moser
lattice ML = Z⟨1, ω₁, ω₃, ω₁ω₃⟩ and published both the record table and a
database of their searches' visited configurations (~66 million graphs). We
report an independent reproduction of this entire frontier: for every n with
16 ≤ n ≤ 100 we re-derive a configuration attaining the record count and certify
it in exact arithmetic over ℚ(√3, √11); at the four values where we enumerated
record classes and diffed them against the full database, our isomorphism
classes coincide exactly with theirs. (For three stubborn values the search was
guided by structural forensics of the published configurations; these
dependencies are tracked per result.) No record was beaten, and our extensive
beyond-record searches — including exhaustive local optimality checks and one
exact branch-and-bound result — support the hypothesis that the published
values are optimal over ML in this range. Along the way we develop:
(i) *coincidence forcing*, a constraint-side method that converts degenerate
near-miss distance clusters into Gauss–Newton constraints and, at n = 40 (where
we analyze the mechanism in full) shows how continuous search gets stuck a few
edges below lattice records; (ii) a first-order rigidity analysis showing that
a near-record configuration found by continuous search has exactly one internal
flex, along which the missing edges fire simultaneously at algebraic
coincidence angles; (iii) an exact integer toolkit for ML (canonical forms,
Minkowski sums, subset selection, beam search, simulated annealing) with which
most records fall in seconds; and (iv) reverse-engineered explicit
constructions for several records, including n = 64 (a flattened 6-cube),
n = 98 (a 7-generator {0,1}-sum whose 7-vertex Minkowski factor is a
cube-minus-corner, not a wheel or spindle), and n = 96, which is the densest
known 96-point subset of the 103-point Galois trace ball |z|² + |σ(z)|² ≤ 4.
The n = 96 record raises a methodological point: it contains point pairs at
distance ≈ 0.085, and it is the only n in [16, 100] at which *every* known
record configuration contains such close pairs — so any float-search pipeline
with a minimum-separation hygiene floor is structurally blind to it. We also
quantify a structure–randomness tradeoff: across five independent "locking"
transitions, gaining the final 2–4% of unit distances multiplies the
configuration's bisector (reflection-symmetry) energy by 3–4×.

## 1. Introduction

Erdős's unit distance problem asks for u(n), the maximum number of pairs at
distance exactly 1 among n distinct points in R². The asymptotic upper bound is
u(n) ≤ ~1.94·n^{4/3} (Ágoston–Pálvölgyi [3]; the 4/3 exponent is unimproved
since Spencer–Szemerédi–Trotter). Exact values are known for n ≤ 21
(Alexeev–Mixon–Parshall [4]). For 22 ≤ n ≤ 100 the state of the art is the
table of densest *known* configurations of Engel, Hammond-Lee, Su, Varga and
Zsámboki [1], produced by diverse beam search over the Moser lattice

  ML = Z⟨1, ω₁, ω₃, ω₁ω₃⟩ ⊂ C,  ω₁ = e^{iπ/3},  ω₃ = e^{i·arccos(5/6)},

a rank-4 Z-module, dense in the plane, with exactly 18 unit vectors (9
undirected edge directions). All of their records embed in ML. They published
their full database of visited configurations (~66M graphs, integer ML
coordinates) [2], which makes the frontier independently checkable — an
opportunity this note takes up.

On the lower-bound side, a 2026 development is worth flagging for context: an
OpenAI internal model produced, and Sawin and others made explicit and verified,
a construction beating Erdős's conjectured n^{1+c/log log n} bound for infinitely
many n via projections of number-field lattices of growing degree [5, 6] (see
also [7, 8]). That result is purely asymptotic; at n ≤ 100 its finite sections
are far below the beam-search records, and it plays no role here.

**What this note reports.** Starting from a single 40-point, 132-edge
configuration found by continuous (lattice-free) search in a prior session, we:

1. **(Mechanism)** explain *why* continuous search stalls a few edges short of
   lattice records, in full at n = 40: the found framework is flexible with
   exactly one internal degree of freedom, and the missing edges fire
   *simultaneously, along the flex*, at an algebraic coincidence point. We give
   two independent procedures (rotate-and-project homotopy; flex following on
   the rigidity matrix's null space) that both land on the same 135-edge
   configuration to 3.8·10⁻¹⁴, and a cheap constraint-side method —
   *coincidence forcing* — that achieves the same jump automatically and also
   broke search plateaus at n = 50 and 70 (§3).
2. **(Reproduction)** re-derive a record-count configuration for **every** of
   the 85 values n ∈ [16, 100], using an escalating ladder of exact-integer
   lattice searches (§4–5). 84 of the 85 are attained by configurations that
   also pass a conservative float-geometry audit; the one exception (n = 96) is
   forced — see point 4.
3. **(Constructions)** reverse-engineer explicit generative recipes for several
   records (§6): the n = 49 record is the Minkowski sum of two 6-wheels *on
   different sublattice rings* (same-ring sums collapse); a record-count
   configuration at n = 50 is that sum plus one point (an exhaustively-checked
   fact); n = 64 is a {0,1}-sum of 6 unit generators ("flattened 6-cube");
   n = 98 is a 7-generator {0,1}-sum whose published description ("4 edges ⊕
   7-vertex graph") hides 14 lattice collisions and a 7-vertex factor that is a
   cube-minus-corner rather than any wheel or spindle; records at consecutive n
   nest (every one of the 42 record classes at n = 90 is a one-vertex deletion
   of an n = 91 record).
4. **(A convention finding)** the unique record class at n = 96 — the densest
   known 96-point subset of the 103-point *Galois trace ball*
   |z|² + |σ(z)|² ≤ 4, where σ is the √11 ↦ −√11 conjugation — contains four
   point pairs at exact squared distance (23 − 4√33)/3, i.e. distance
   ≈ 0.08515. Any search pipeline with a minimum-separation hygiene floor (ours
   is 0.2, adopted to kill float-tolerance exploits) will *never* claim this
   record, no matter the budget. We argue for a dual-track standard: hygiene
   floors for float search, exact certificates for record claims (§7).
5. **(Dichotomy data)** across five independent locking transitions (n = 30, 40
   twice, 50, 70), gaining the last +2–4% of unit distances multiplied the
   configuration's bisector energy — a quantitative reflection-symmetry measure
   — by 3.0–4.2×, while a genuinely non-lattice near-record configuration sits
   at the low-symmetry extreme (§8).
6. **(Negative results)** documented dry beyond-record hunts at every n we
   pushed, including exhaustive 2- and 3-vertex relocation optimality of record
   classes we held, an exact branch-and-bound proof that the densest 30-point
   subset of the 49-point wheel sum has exactly 92 edges, and the observation
   that at n = 70 a vast "280-edge attractor" defeats every monotone method —
   only high-temperature exact-lattice annealing reassembles the record's
   four-hub core (§9).

Everything is reproducible from the repository: a test suite pins the
arithmetic (156 tests at the time of writing); every claimed configuration
ships with a float audit log and an exact certificate
(`data/frontier/`, `data/mlcoords/`); the §6 recipes are committed as
self-verifying scripts (`scripts/recipes/`); and each batch of results was
re-derived by adversarial verifier agents independent of the agents that
produced it (§10).

**What this note does not claim.** No new record counts. At the four deep-dive
values our class inventories matched the published ones exactly (we hold
2/1/5/3 classes at n = 30/40/50/70, verified complete against the database —
the database's record configurations at those n, 4/1/8/4 lattice embeddings,
collapse to the same classes); elsewhere we verified record counts, not class
inventories. That the frontier of [1] is reproducible — and at the checked
values complete — is itself the point: it is robust against several methods
quite different from their beam search.

## 2. Verification standards

Every count claimed in this note passed two independent layers.

**Float three-audit** (search hygiene; inherited from the prior session's
hard-won failures): (a) minimum pairwise separation ≥ 0.2; (b) K_{2,3}-freeness
— no two vertices share ≥ 3 common unit-neighbors (two unit circles meet in ≤ 2
points); (c) exact realizability — damped Gauss–Newton on *all* claimed edges
converges to total residual < 10⁻²⁴ with every edge within 10⁻¹² of unit. The
separation floor exists because, without it, Metropolis search discovers
near-coincident point clusters spread tangentially along unit circles (the
distance error is second-order, ~δ²/2), faking K_{m,m} subgraphs and absurd
counts; an early run produced a bogus "400 edges" at n = 40 this way.

**Exact certificate.** A configuration with integer ML coordinates is checked
symbolically: with basis (1, ω₁, ω₃, ω₁ω₃) and coordinate difference
(da, db, dc, dd), the integer invariants

  A = 12da + 6db + 10dc + 5dd, B = 6db + 5dd, C = 2dc + dd, D = −dd

satisfy 12·Re = A + D√33 and 12·Im = B√3 + C√11, hence

  144·|diff|² = (A² + 3B² + 11C² + 33D²) + 2(AD + BC)·√33.

A pair is at distance exactly 1 iff A² + 3B² + 11C² + 33D² = 144 and
AD + BC = 0 — a pure integer test, vectorizable, with a Fraction-arithmetic
reference path over the basis {1, √3, √11, √33} cross-checking it. Certificates
also verify exact pairwise distinctness and record the integer coordinates.
Recovering integer coordinates from a float configuration is itself nontrivial
(ML is dense in the plane, so rounding positions is meaningless): coordinates
are propagated along a spanning tree of the unit-distance graph after matching
each edge to one of the 18 unit vectors, then every non-tree edge is checked for
exact cycle consistency.

**Dual-track convention** (adopted by this project after the n = 96 finding,
§7): the float three-audit is required for any claim arising from float search;
an exact certificate suffices for a record claim regardless of float minimum
separation, with sub-floor pairs flagged. The tolerance exploit the floor
guards against is impossible under a certificate, which proves distinctness and
edge-exactness symbolically.

**Adversarial verification.** Every batch of results was re-verified by an
agent that did not produce it: fresh subprocess re-runs of both verification
scripts on every claimed file (493 files in the largest batch), independent
re-execution of claimed *recipes* from their textual description (not the
producing agent's scripts), and canonical-form class comparisons. This process
caught real errors — notably six audit-invalid raw ties produced by a driver
bug in which a raw edge count could stop a search stage before passing the
audit gate (§5) — and they were rejected and fixed rather than papered over.
The present draft was itself reviewed the same way: four adversarial reviewers
re-ran the reproduction entry points and recomputed the mathematics.

## 3. Continuous search, the flexible skeleton, and coincidence forcing

### 3.1 Baseline

The continuous baseline is Metropolis search over *circle-intersection moves*:
pick a vertex k and two others a, b with |ab| < 2; move k to a random
intersection of the unit circles around a and b (every accepted move creates ≥ 2
exact edges; surplus edges arise by coincidence); reject any placement violating
the separation floor; accept by the change in unit count under a geometric
temperature schedule. Free-coordinate annealing fails completely (dense
configurations are measure-zero in coordinate space); the structured move set is
essential. At n = 40 this reproducibly yields 122–132 edges across seeds from
random starts in minutes of CPU (the record is 137); our port of the prior
session's search regenerates its best 132-edge configuration bit-for-bit from
the same seed.

### 3.2 The near-miss signature

The 132-edge configuration has a striking diagnostic: among its 648 non-edge
pairs, the three nearest misses sit at *identical* distance — d = 0.9863168633,
|d − 1| = 1.368314·10⁻², agreeing to ~6·10⁻¹¹ across the three pairs — and the
next-nearest miss is at 6.7·10⁻². A plain warm-start control arm (24 runs
across a temperature × budget grid, ~1.56M steps total) never left 132: the
chains random-walk a wide neutral plateau (hundreds of accepted Δ=0 moves per
30k steps, zero up-moves), and the triple-degenerate near-miss floor is
*invariant* across the plateau. Three pairs pinned at one wrong algebraic
distance is the signature of a hinge: a rigid sub-structure rotated slightly
off the angle at which all three would become unit simultaneously.

### 3.3 Coincidence forcing

The constraint-side response is immediate: add the degenerate near-miss cluster
to the edge set and run the damped Gauss–Newton projection on all 135
constraints. It converges (total residual 1.3·10⁻²⁷), and the resulting
configuration passes the full audit: **132 → 135 audited edges at n = 40**, and
the projected configuration is fully Moser-aligned — all 9 of its edge
directions match ML directions, where the input had only 6 of 12. The floating
families snapped onto the lattice angles the moment the constraints demanded it.
A greedy loop (*force_coincidences*: cluster near-misses by |d−1| degeneracy,
attempt cluster-then-single augmentations, accept any strictly-improving audited
projection, repeat until dry) and an alternating loop with warm Metropolis
(*exploit_loop*) automate this. Applied across our campaign bests:

| n | search plateau | after forcing (+ exploit) | record |
|---|---|---|---|
| 40 | 132 | 135 → **136** | 137 |
| 40 (2nd config) | 132 | 135 → **136** (distinct class) | 137 |
| 50 | 177 | 179 → **181** | 183 |
| 70 | 266 | 274 (an **8-cluster** fired at once) → **277** | 281 |

At n = 70 the forcing step broke a plateau that 44 independent search seeds and
re-annealing ladders had failed to leave.

### 3.4 The rigidity mechanism

First-order rigidity analysis makes the picture precise. The rigidity matrix of
the 132-edge framework (132 × 80; row (i,j) carries p_i − p_j at i's coordinate
columns and its negation at j's) has rank 76: nullity 4 = 3 trivial motions +
**exactly one internal flex**, with a clean spectral gap (smallest structural
singular value 5.6·10⁻¹ vs 1.7·10⁻¹¹ for the flex; ratio 3.4·10¹⁰). The single
flex couples to *all four* edge-direction families simultaneously (angular
velocities −8.74, +19.94, −7.07, +8.08 deg per unit of configuration-space arc
length), which has a practical consequence: "lock family X to its lattice angle"
is gauge-ambiguous, because moving along the flex counter-rotates everything
else. Two independent procedures were run:

- **Rotate-and-project homotopy**: rigidly rotate a floating family by an
  increment about its hinge centroid, Gauss–Newton-project onto all 132 original
  edges, repeat. Locking the small floating family (6 edges) converges in 33
  increments with zero stalls and carries *all* families to ML angles — but the
  ML-locked endpoint itself *overshoots*: the near-miss triple crosses unit
  *during* the motion, at θ* = 0.0113507° (a regula-falsi search along the
  homotopy finds all three pairs simultaneously unit with spread 2.6·10⁻¹⁴),
  and is +1.7·10⁻⁴ past unit at the exact lattice angle. Locking alone, without
  the coincidence search, fired nothing in 0/28 warm searches.
- **Flex following**: walk the full one-dimensional flex path (predictor along
  the null-space direction, Gauss–Newton corrector), s ∈ [−0.402, +0.946],
  terminated at both ends by the separation floor with the flex alive
  throughout. Exactly eight non-edge pairs cross unit along the path, in four
  events: two *triple* fires (s = −0.040 and s = +0.570), each yielding an
  audited 135-edge fully-ML configuration — mutually non-congruent — at which
  the flex dies (the three new edges consume the degree of freedom, flex
  dimension drops to 0); and two *single* fires (s = +0.140, +0.390) yielding
  audited **133-edge configurations at genuinely non-Moser angles** (only 3 of
  their 13 directions match ML) — coincidences firing off-lattice, of
  independent interest for the question of non-ML near-record configurations.
  The per-vertex flex motion peaks at the vertices of the firing pairs.

The two procedures cross-validate to machine precision: the homotopy-bisection
135 and the flex-scan event-A 135 are the *same configuration* to 3.8·10⁻¹⁴.
Warm search from the 135s reaches 136 on 12/12 seeds; across all routes the
session produced 9 distinct 136-edge congruence classes, later grown to 18
collected classes by basin-hopping restarts. The 9 hinge-batch classes were
verified rigid, fully ML-locked, with nearest non-edge at 8.2·10⁻² (an ML
distance quantum), and every collected class is 1- and 2-relocation-optimal:
the 136 level is a closed attractor family from which 137 is unreachable by
≤3-vertex discrete moves or continuous deformation. In summary: *continuous
search finds a flexible skeleton; the last edges are simultaneous algebraic
coincidences along its flex; forcing the coincidence — from either the
constraint side or the parameter side — is what lattice-restricted search gets
for free.* The prior session's slogan "density's last few percent is number
theory" is, on this evidence, literally correct at n = 40.

## 4. Exact lattice machinery

The 136 → 137 step and everything after it required leaving floats entirely.
The toolkit (package `udg`, modules `mlgraph`, `subsetsearch`, `anneal`)
operates on integer 4-tuples with the §2 invariants and provides:

- **Canonical forms** under the full 12-element ML point symmetry group — the
  six rotations by ω₁^k together with the reflection z ↦ ω₃·z̄, which acts on
  coordinates as (a, b, c, d) ↦ (c + d, −d, a + b, −b) — composed with
  translation normalization. Canonical forms give exact congruence-class
  identities, tabu keys, and database diffs.
- **Constructions**: Minkowski sums; {0,1}-sums of generator sets ("cubes");
  a small library (triangular patches T(k), hexagonal patches H(k), 6-wheels on
  either sublattice ring, the Moser spindle — which embeds in ML as two
  60°-rhombi sharing a vertex with the second multiplied by ω₃, since
  |1 + ω₁|·|1 − ω₃| = √3 · (1/√3) = 1).
- **Search engines**, in the escalation order of the per-n driver:
  (a) *subset-in-closure*: densest-k-subgraph ILS inside an engineered ambient
  (a good Minkowski sum, its 1-step unit-vector closure, or a trace ball);
  (b) *diverse beam search* over exact coordinates (gain-ranked 1-hop candidate
  growth with per-density bucket caps — a reimplementation, on our invariants,
  of the spirit of [1]);
  (c) *exact-lattice simulated annealing* (Metropolis over 1–2-step relocations
  with exact integer edge deltas);
  (d) continuous polish + coincidence forcing on the float side, then
  re-certification.
- **The audit gate**: a raw edge count, however produced, can neither stop a
  search stage nor back a claim until the configuration passes the float audit
  (or, post-§7, carries an exact certificate). This rule is enforced in code; it
  exists because its absence produced six false claims in one afternoon.

Two illustrative facts about the constructions. First, the n = 49 record
(180 edges) is minkowski(W₆^{ω₁}, W₆^{ω₃}) — two 6-wheels on *different*
sublattice rings; the same-ring sum collapses (ω₁^k + ω₁^{k+2} = ω₁^{k+1}) to
the 19-point hexagon H(2) with 42 edges. Cross-ring structure is load-bearing
throughout the record table. Second, with the toolkit in hand, "the n = 50
record is the wheel sum plus one point" is an *exhaustively checkable*
statement: among all 360 candidate one-point extensions of the 49-point wheel
sum, exactly 36 gain 3 edges, yielding two of the five published record
classes (the other three arise by different routes).

## 5. Reproducing the frontier 16 ≤ n ≤ 100

**Targets.** We downloaded the published database [2] (1.32 GB; 6,307 arrays
keyed by batch and n; int8 ML coordinates in the same basis and column order as
ours, which we verified by exact arithmetic before relying on it) and scanned
all of it with the integer unit test: 66,442,373 configurations across 97
values of n, 64 minutes with 10 worker processes. The per-n maxima reproduce
the published Table 2 at every value we cross-checked, and serve as exact
targets (committed as `data/engel_targets.json`).

**Protocol.** The sweep covered 83 values (every n in [16, 100] except 50 and
70, which were already held from the deep-dive campaigns; the deep-dive values
30 and 40 were re-run inside the sweep as validations). For each n: run the
escalating driver (subset → beam → anneal → polish) with a 12-minute budget
(18 on a close-miss retry), audit-gated throughout; band-parallelized across
six agents; every claim re-verified afterwards by an adversarial verifier
(fresh audit subprocess on each of the 105 produced run directories, random-10
fresh exact re-certification, canonical-form checks on anything unusual).

**Results.** 76 of the 83 swept values were tied within the sweep protocol —
65 on the base 12-minute budget and 11 via its retry rules; 53 of the 76 in
under 60 seconds. The subset-in-closure stage won 58 (the 1-step closure of the
49-point wheel sum alone won 19 values), exact annealing won 18, and the beam
stage, squeezed between two stronger stages, won none. Four of the seven misses
fell the same day: n = 64 to a direct construction (§6), n = 76/89/92 to
retries. The last three (n = 90, 96, 98) required structural forensics (§6) —
n = 90 and 98 closed cleanly; n = 96 is the convention case (§7). Combined with
n = 50 and 70, **every one of the 85 values n ∈ [16, 100] is reproduced**: 84
audit-clean, and n = 96 by exact certificate with a close-pair flag.

**Class identity.** At the four deep-dive n we enumerated record classes by
repeated independent search and compared canonically against the full database
record inventory: the isomorphism-class correspondence is an exact bijection in
both directions (2/1/5/3 ↔ 2/1/5/3; the database's record configurations at
those n — 4/1/8/4 lattice embeddings — collapse to the same classes). Two
curiosities: at n = 30 the two record classes differ by relocating a *single*
degree-4 vertex, i.e., the record classes are adjacent in the relocation move
graph; at n = 40 the single record class was reached by 25+ independent search
arrivals across three method families, always the same class.

**Misses are where records are rare.** Of the seven first-pass misses, five
were values where the database holds ≤ 2 record configurations among ~1–1.8M
stored at that n. The hardest values for us were precisely the values where the
published record is a near-unique object.

## 6. A catalogue of record constructions

The closing of the last misses produced explicit recipes worth recording; each
is committed as a self-verifying script under `scripts/recipes/`.

**n = 64 = flattened 6-cube (252 edges).** The {0,1}-sum of 6 of the 18 unit
vectors. Enumerating 6-subsets in C(18,6) order, skipping sets containing a
vector and its negation, finds a valid set on the 17th candidate tried; all 64
subset-sums distinct, 252 exact edges, audit-clean. (Generators: (−2,1,2,−1),
(−1,−1,1,1), (−1,0,0,0), (−1,1,0,0), (0,0,−1,0), (0,0,0,−1).)

**n = 98 = (3-cube minus a corner) ⊕ 4-cube (429 edges).** The published
description "4 edges ⊕ 7-vertex UDG" conceals two traps. The naive reading —
generator 4-cube ⊕ wheel or spindle — produces *no* 98-point sums at all (the
sums collapse to 80/56/72/…). Translate-decomposition forensics on the actual
record (the unique such class in the database) recovers: the 4-edge factor is
the 16-point {0,1}-sum on generators {ω₁², −ω₁, ω₃ω₁², −ω₃ω₁} — equivalently a
rhombus plus its ω₃-rotation; the 7-vertex factor is the 8-point {0,1}-sum on
{1, (1+ω₁)(1−ω₃), (2−ω₁)(1−ω₃)} *minus its all-ones corner* — a 7-vertex,
11-edge graph with degree sequence (2,3,3,3,3,4,4) that is *not* the Moser
spindle and not a wheel. The sum has exactly 14 lattice collisions (7·16 − 14 =
98) and 429 edges. The whole record is thus a 7-generator {0,1}-sum — a direct
sibling of the n = 64 cube — and is regenerated from library primitives in one
line, landing canon-identically in the published class.

**n = 96 = densest known 96 of the 103-point Galois trace ball (418 edges).**
Let σ be the field automorphism √11 ↦ −√11 (fixing √3). The set of lattice
points with |z|² + |σ(z)|² ≤ 4 — in integer invariants,
A² + 3B² + 11C² + 33D² ≤ 288 — is a 103-point, 456-edge object
(box-independent; re-derived from scratch over a wider coefficient box by the
verifier). The unique record class at n = 96 is a 96-point subset of it, found
deterministically by subset ILS in about one second; this ambient/selection
structure was itself recovered by coset and trace-form dissection of the
database's record configuration (§10). The same ball yields record
configurations at k = 97 (423) and k = 95 (412); the chain stops at k = 98
(428 < 429). This is the cleanest description we know of any record in the
table, and it is also the problematic one — see §7.

**Nesting (n = 90 ↔ 91).** Every one of the 42 record classes at n = 90 is
obtained from one of the 78 record configurations at n = 91 by deleting a
degree-5 vertex (42/42 coverage). Consequently the n = 90 record is reachable
entirely from our own assets: our independently-found n = 91 tie has a unique
minimum-degree vertex, whose deletion gives a record class; and, in the other
direction, adding the deterministic best point to our n = 89 tie gives a
*different* record class. (One caveat for reproducers: the ambient-racing
front-end of the per-n driver is class-nondeterministic across re-runs, so the
committed chain is anchored to the saved, audited n = 91 artifact; the trim and
extend steps themselves are deterministic.) Records at consecutive n are not
isolated objects but a laddered family — consistent with how beam search visits
sub-configurations, and useful as a cheap closing move anywhere it holds.
(Engel-side, partial nesting was also confirmed at 96 ↔ 97 and for n = 98,
whose record is a subset of 4 of 7 record configurations at n = 99 and 9 of 37
at n = 100.)

**Hubs (n = 70, 281 edges).** The three record classes contain four
"super-center" vertices of degrees (16, 16, 15, 15) arranged in a 1/√3-rhombus
with unit long diagonal (the Moser spindle metric); each degree-16 hub realizes
16 of the 18 unit directions, and jointly the hubs use all 18. The large
280-edge attractor that defeats every monotone method (9+ distinct classes
collected, the held classes 2- and 3-relocation-optimal; large-neighborhood
destroy-rebuild returned 280 in 280/280 instrumented runs) lacks this
double-hub core entirely; only high-temperature exact-lattice annealing,
melting deeply enough to reassemble the hub rhombus, crossed the valley — and
then did so within minutes. We found exactly the three published classes.

## 7. The close-pair phenomenon and audit conventions

The unique n = 96 record class contains four point pairs at exact squared
distance (23 − 4√33)/3, i.e. distance ≈ 0.08515 (difference class
±(4,−1,−4,3)·ω₁^k). These pairs are not numerical artifacts: the certificate
proves all 418 edges exactly unit and all 96 points exactly distinct. But any
float-search pipeline with a minimum-separation hygiene floor above 0.085 —
adopted, in our case, after the tolerance exploit produced fake counts (§2) —
will *categorically never claim this record*, regardless of budget. Empirically
this is not a corner case: the same close pair produced audit-invalid raw
record-count ties that our verifier rejected at n ∈ {71, 72, 73, 90, 94}, and
record configurations carrying the pair also exist at n = 95 and 97. What is
special about n = 96 is that there the close pairs appear *unavoidable*: the
database's ~66M configurations contain exactly one 418, this one; exhaustive
search over all floor-respecting 96-subsets of the trace ball tops out at 414;
our dedicated floor-respecting hunts reached 416, and the sweep's audit-clean
best is 417.

We therefore adopted, and recommend, a **dual-track convention**: hygiene
floors apply to float-search *claims* (where they guard against a real exploit);
record claims are settled by **exact certificates** (where the exploit is
impossible), with sub-floor pairs flagged. The literature's counting already
implicitly takes the second view — the published u(96) lower bound *is* this
configuration. Pipelines that inherit a separation floor from search hygiene
should be aware they may be structurally blind to specific true records.

## 8. Bisector energy across the locking transitions

For a configuration P, the bisector energy E = Σ_ℓ m(ℓ)² sums, over distinct
perpendicular-bisector lines ℓ of point pairs, the squared number of pairs
sharing ℓ — a reflection-symmetry energy (E_nt restricts to lines with
m(ℓ) ≥ 2; for float configurations the line triples are bucketed, and the
values below are identical at 4 and 6 bucketing decimals). The forcing
transitions of §3 give matched before/after pairs at equal n — the controlled
comparison the structure-vs-randomness question wants:

| transition | edges | E_nt | factor |
|---|---|---|---|
| n=40, config A | 132 → 136 | 578 → 1760 | 3.0× |
| n=40, config B | 132 → 136 | 543 → 1681 | 3.1× |
| n=50 | 177 → 181 | 932 → 3292 | 3.5× |
| n=70 | 266 → 277 | 1984 → 8323 | 4.2× |
| n=30 | 91 → 93 | 294 → 938 / 993 | 3.2–3.4× |

(Random configurations score 0.) Gaining the final 2–4% of unit distances
multiplies reflection symmetry by 3–4×: the marginal record edges are purchased
with a disproportionate explosion of symmetric structure. The pre-forcing
configurations populate the interesting middle of the count-vs-symmetry Pareto
frontier — roughly 95–97% of record density at a quarter to a third of the
locked symmetry energy — and the n = 30, 91-edge configuration is the extreme
low-symmetry point among dense configurations measured here: it is certifiably
not a rotated ML configuration as floated (its exact-coordinate certification
fails with four cycle inconsistencies, and after the best global rotation its
direction-family residuals reach ~0.1 rad — structural, several orders above
its ~10⁻⁹ float noise). No low-symmetry configuration at full record density
appeared anywhere in this work; on present evidence the dichotomy
"record-dense ⟹ lattice-like" holds with a sharp knee.

## 9. Negative results and evidence of optimality

Documented dry hunts, with method and scale:

- **n = 40, 138**: exhaustive single-vertex relocation at 1/2/3-hop radius and
  exhaustive simultaneous pair relocation on the record class; full drop-3 (all
  9,880 triples) + beam rebuild; lift-to-41-and-return ladders; 60k+ basin-
  hopping restarts (and 56k+ more from the 137 itself); 6,690 Minkowski
  factor-pair sums; repeated wide (width-1500) beam runs, which reached 137 raw
  12/12 times but never 138; continuous polish + forcing in both directions.
  All dry; the 137 class is, additionally, an isolated basin (perturb-and-climb
  falls to the 136/134 levels and only ever returns to itself).
- **n = 50, 184**: ~1.4M exact local-search restarts across four engineered
  ambient supersets; exhaustive 1- and 2-relocation optimality of every 183;
  large-neighborhood funnels that always return to 183; a structural ladder
  argument (the n = 51 maximum we could reach, 188 — itself a tie of the
  published 188 — has minimum degree 5, so deletion returns 183).
- **n = 70, 282**: exhaustive two-ply and three-ply relocation over the 281
  classes held at the time; ladders from the 281s reached 286 at n = 71 (a tie
  of the table) and 290 at n = 72, with all drop-backs returning 281 (a
  certified 291 at n = 72, tying the table, was found separately as a
  byproduct); continued annealing melt waves.
- **n = 30 (exact)**: branch-and-bound proof that the densest 30-point subset
  of the 49-point wheel sum has exactly 92 edges. The committed proof
  (`scripts/bnb_n30_wheel49.py`, log in `docs/forensics/n30-bnb-proof.log`)
  explores ~6.7·10⁷ nodes in ~210 s with an admissible degree-based bound; an
  independent review re-implementation with a tighter bound needed ~1.8M
  nodes — the same theorem either way. The record 93 *requires* leaving the
  sum (both record classes keep only 16 of their 30 points inside it), which
  the closure ambient supplies.
- **n = 96 (clean)**: exhaustive floor-respecting subsets of the trace ball max
  at 414; dedicated floor-respecting hunts at 416; audit-clean sweep best 417.

None of this proves optimality; all of it is consistent with the published
table being exactly the ML optimum on 16 ≤ n ≤ 100, and the n = 30 result shows
exact subset optimality proofs at meaningful scale are within reach.

## 10. Provenance, methodology, and reproducibility

This work was carried out in one day (2026-06-10) as a sequence of autonomous
multi-agent research sessions: roughly thirty Claude agents (Anthropic)
organized in build/experiment/verify workflows, with strict separation between
agents that produce claims and agents that verify them. The repository
preserves the trail: the test suite (exact regression targets from the prior
session's data appendix, the tolerance-exploit fixture, toolkit validations
including the 3-line reproduction of the n = 49 record); per-value
configurations with audit logs and exact certificates (`data/frontier/`, one
directory per n ∈ [16, 100]); the forensics reports behind §6
(`docs/forensics/`); and a running results log (`RESULTS.md`).

The published database [2] was used for three things, explicitly tracked per
result: exact per-n targets (`data/engel_targets.json`); post-hoc
class-identity comparison; and, for three stubborn values (n = 90, 96, 98),
structural forensics whose findings were then re-expressed as generative
recipes executed by our own toolkit (the verifier re-derived each recipe from
its textual description before acceptance; for n = 90 the closing
constructions use no database information beyond the nesting hint and post-hoc
class identification). Where a result depends on database structure, the
dependency is stated where the result is.

Reproduction entry points: `uv run pytest` (suite);
`scripts/chase_n.py --n N` (per-n record chase; targets from
`data/engel_targets.json`); `scripts/audit_config.py` /
`scripts/ml_coords.py` (the two verification gates);
`scripts/force_coincidences.py`, `scripts/exploit_loop.py` (§3);
`scripts/recipes/build_n64_cube.py`, `build_n90_trim91.py`,
`build_n96_traceball.py`, `build_n98_recipe.py` (the §6 recipes, asserting
their own verification); `scripts/bnb_n30_wheel49.py` (the §9 proof).

## 11. Open problems

1. **Beat the table.** Every lever we tried inside ML on 16 ≤ n ≤ 100 was dry.
   The live directions: n > 100 (the published table stops; the toolkit
   extends); the lattices L_k = Z⟨1, ω₁, ω_k, ω₁ω_k⟩ for k ≠ 3 (the unit-vector
   counts in this family are unbounded — Ruhland [7] — and only some k have
   been searched); and non-ML configurations, for which our flex-path 133s at
   n = 40 (audited, 3 of 13 directions on ML) are the first specimens we know
   of above the continuous-search plateau. At n = 17 an optimal configuration
   not embeddable in the Moser ring is reported [4], so the lattice is not
   sacred.
2. **Prove something.** Extend the branch-and-bound subset proof from a fixed
   49-point ambient to closure ambients — even one *proven* ML-optimal value at
   an interesting n would be new.
3. **The dichotomy.** Make the §8 knee quantitative: is there a theorem of the
   form "u-count within ε of record ⟹ bisector energy ≥ f(ε)·E_lattice" for
   ML-free configurations? The five-transition data suggests f is steep.
4. **Close-pair geography.** Which record classes across the table contain
   sub-floor pairs? We know the ≈0.085 pair appears in record configurations at
   n = 95, 96, 97 and produced rejected raw ties at n ∈ {71, 72, 73, 90, 94};
   it is one difference-class phenomenon, (23 − 4√33)/3. A census would tell
   search designers where hygiene floors bite.

## References

[1] P. Engel, O. Hammond-Lee, Y. Su, D. Varga, P. Zsámboki. *Diverse beam
search to find densest-known planar unit distance graphs.* arXiv:2406.15317;
Experimental Mathematics (2025). Records table: Table 2.
[2] P. Zsámboki et al. Code: codeberg.org/zsamboki/dbs-udg. Database
(seen_graphs.npz, ~66M configurations): FigShare,
figshare.com/s/a7e6426c318e1d550366.
[3] P. Ágoston, D. Pálvölgyi. *An improved constant factor for the unit
distance problem.* arXiv:2006.06285 (2022). (Constant ≈1.94 — verify exact
statement against the published version before submission.)
[4] B. Alexeev, D. G. Mixon, H. Parshall. *The Erdős unit distance problem for
small point sets.* arXiv:2412.11914 (2024). Exact values u(n) for n ≤ 21; the
n = 17 non-ML optimum is reported there (statement verified via secondary
sources; confirm against the paper before submission).
[5] OpenAI. *Planar Point Sets with Many Unit Distances* (2026-05-20).
cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-proof.pdf
[6] W. Sawin. arXiv:2605.20579 (2026): explicit n^{1.014} lower bound. Human
verification: N. Alon, T. Bloom, W. T. Gowers, D. Litt, W. Sawin, et al.,
arXiv:2605.20695 (2026).
[7] H. Ruhland. arXiv:2410.16172 (2024). Lattice families with unboundedly
many unit vectors. (Title to be verified against arXiv before submission.)
[8] D. Radchenko. *Unit distances in number-field lattices.* Discrete &
Computational Geometry 66:269–272 (2021). (Bibliographic details inherited
from the project handoff; verify before submission.)
