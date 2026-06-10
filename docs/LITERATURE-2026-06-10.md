# Literature recon — record structure, exact values, 2026 lower bound

Date: 2026-06-10. Scope: HANDOFF §1, §5.C, §6 follow-ups. All URLs fetched and
read this session unless flagged otherwise.

---

## Q1. Record structure (Engel–Hammond-Lee–Su–Varga–Zsámboki)

**Paper:** "Diverse beam search to find densest-known planar unit distance graphs",
Peter Engel, Owen Hammond-Lee, Yiheng Su, Dániel Varga, Pál Zsámboki.
arXiv:2406.15317, **latest v3 (13 Jun 2025)**; published in *Experimental
Mathematics*, doi:10.1080/10586458.2025.2507956.
- abs: https://arxiv.org/abs/2406.15317
- html full text: https://arxiv.org/html/2406.15317v3

### Per-n facts at our target sizes (Table 2)

| n | edges | # isomorphism classes found | structural description in paper |
|---|-------|------------------------------|----------------------------------|
| 40 | 137 | **1** | none given |
| 50 | 183 | **5** | none given |
| 70 | 281 | **3** | none given |

The paper gives **no degree sequences, generating sets, or symmetry data for any
n > 15**; structural commentary is limited to the Minkowski-sum observations
below and figures (Fig 4: n=1–9; Fig 5: n=27/28/29; Fig 6: the two n=30
93-edge graphs; Fig 7: the n=49 180-edge graph; Figs 8–9: the n=39 and n=33
Minkowski-sum records).

### Which records are Minkowski sums

Every n the paper explicitly identifies (quotes condensed from v3):

- n=9: "disjoint Minkowski sum of two unit triangles"
- n=21: "disjoint Minkowski sum of a unit triangle and a 6-wheel"
- n=24, 28: "non-disjoint Minkowski sums of 5 edges"
- n=33: "Minkowski sum of a 4-vertex UDG — a triangle with an isolated center —
  and a 9-vertex UDG"
- **n=39: "Minkowski sum of a 7-vertex UDG and an 8-vertex UDG"** (adjacent to our n=40 target)
- **n=49: "disjoint Minkowski sum of two 6-wheels"** (7×7; adjacent to our n=50 target)
- n=64: "disjoint Minkowski sum of 6 edges" (flattened 6-cube, 2^6)
- n=98: "Minkowski sum of 4 edges and a 7-vertex UDG" (2^4 × 7, non-disjoint)

Aggregate: *"The fraction of Minkowski sums among the densest known UDGs is
44.2%, while the fraction of Minkowski sums among all the UDGs visited by the
beam search is 5.6%."* — Minkowski-sum structure is heavily enriched among
records (supports HANDOFF §5.A.3 seeding).

### Search-space facts relevant to us

- Moves are 100% Moser-lattice-internal: from parent graph, add v′ ∈ v +
  {±1, ±ω₁, ±ω₃, ±ω₁ω₃} (generator steps), plus **triangle completions and
  parallelogram completions**. Their search literally cannot leave ML — our
  continuous-move edge over them (HANDOFF §5.A.2) is confirmed real.
- On alternative lattices L_k: *"We checked some of these alternative lattices,
  but none gave better results than the Moser lattice."* "Some", not all —
  HANDOFF §5.A.5 (Ruhland-family sweep) remains open.
- Prior art for 30 < n ≤ 100 was essentially blank: the paper matched all
  published records for 15 < n ≤ 30 and the values above 30 are *their* new
  records (including a second, previously unpublished 93-edge class at n=30).

### Public artifacts (codeberg + FigShare)

Repo https://codeberg.org/zsamboki/dbs-udg contains only `main.py` (CuPy beam
search; `target_depth` = max n, 100 in the paper), `README.md`, `LICENSE` —
**no data files in the repo**. Running it regenerates `seen_graphs/`,
`toplists/` (per-n edge-count records), `visits/` (Zobrist hashes).

**Dataset:** single compressed NumPy file with graphs from 207 beam-search runs
(the "60M+ UDGs" DB), shared via FigShare:
- URL (from README): **https://figshare.com/s/a7e6426c318e1d550366**
- Format: `.npz`; **keys = vertex counts as strings** ("40", "50", "70", …);
  each value an integer array of shape **(m, n, 4)** = m graphs × n vertices ×
  Moser coordinates (a,b,c,d) meaning a·1 + b·ω₁ + c·ω₃ + d·ω₁ω₃,
  ω₁ = exp(iπ/3), ω₃ = exp(i·arccos(5/6)).
- **Access caveat (verified 2026-06-10):** the FigShare share page is behind an
  AWS WAF JavaScript challenge — curl/headless fetch returns HTTP 202
  challenge; the public API does not resolve this private-link token. **Open
  the link in a real browser to download.** File size not determinable
  headlessly (60M graphs ⇒ plausibly GBs; budget accordingly or pull via
  browser and extract only the needed keys — `np.load` on an `.npz` is lazy
  per-key, so after download you can read just "40"/"50"/"70" without
  decompressing the rest).
- No per-n standalone downloads exist. Alternatives to the big file: (a) email
  corresponding author zsamboki.pal@renyi.hu for the three record configs;
  (b) re-run `main.py` with `target_depth=70` (needs CUDA/CuPy — not available
  on Apple Silicon without effort).

---

## Q2. Exact values (arXiv:2412.11914)

**Paper:** "The Erdős unit distance problem for small point sets",
Boris Alexeev, Dustin G. Mixon, Hans Parshall. v2, 12 Feb 2025.
https://arxiv.org/abs/2412.11914

- Improved **upper** bounds on u(n) for n ∈ {16,…,30}.
- **u(n) is now known exactly for all n ≤ 21** (HANDOFF §1 said n ≤ 15 — update
  it). Quote: "When n ≤ 21, our bounds match the best known lower bounds, and
  we fully enumerate the densest unit-distance graphs in these cases."
  Values (OEIS A186705): u(16..21) = 41, 43, 46, 50, 54, 57.
- Methods: (i) combinatorial — efficient generation of forbidden-subgraph-free
  candidate graphs; (ii) algebraic — a **custom embedder** deciding
  unit-distance realizability, "more efficient in practice than tools such as
  cylindrical algebraic decomposition".
- Relevance to n=30–70: the exact methods are enumeration-based and stop near
  n=30 (only bounds, not exact values, for 22–30); they do **not** verify
  optimality at our n. Two takeaways for us: (a) their embedder is an
  independent exact-realizability oracle — same role as our Gauss–Newton +
  residual audit, useful as a cross-check method citation; (b) per MathWorld's
  summary of their enumeration, **one of the 7 maximally dense graphs at n=17
  does NOT embed in the Moser ring** — the first record-density UDG outside ML.
  Precedent that our Tier-2 goal (record-density non-ML config) is achievable.

---

## Q3. The 2026 lower-bound development — VERIFIED REAL

### What happened

On **20 May 2026** an OpenAI internal reasoning model autonomously produced a
proof **disproving Erdős's unit distance conjecture** (the conjectured matching
upper bound u(n) ≤ n^(1+C/log log n)). Three primary documents, all dated
2026-05-20:

1. **OpenAI, "Planar Point Sets with Many Unit Distances"** (author listed as
   "OpenAI"), PDF read in full this session:
   https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-proof.pdf
   - **Theorem 1.1 (verbatim):** "There exists an absolute constant δ > 0 and
     infinitely many positive integers n for which ν(n) ≥ n^(1+δ)."
   - Construction: infinite **unramified pro-3 class-field tower** over a cyclic
     cubic field (totally real fields L of growing degree, bounded root
     discriminant, prescribed completely-split primes ≡ 1 mod 4 via Chebotarev;
     Golod–Shafarevich + Shafarevich relation-rank estimate keep the tower
     infinite — the Hajir–Maire method). Set K = L(i); split primes give many
     u ∈ K^× with u·c(u)=1, i.e. |σ(u)| = 1 under **every** complex embedding;
     embed O_K as a Minkowski lattice in C^f, cut by a polydisc, **project to
     one complex coordinate** → planar set with n^(1+δ) unit distances.
   - Paper states the model was given an AI-written prompt, output graded by an
     AI pipeline, then "a draft was sent to external mathematicians, including
     several number theory experts, who confirmed the proof's correctness (and
     have already simplified and strengthened the argument)". The manuscript is
     a human-edited exposition; the verbatim model output is included.
2. **Will Sawin, "An explicit lower bound for the unit distance problem"**,
   arXiv:2605.20579 (v1, 20 May 2026): https://arxiv.org/abs/2605.20579
   — makes the exponent explicit: **ν(n) > n^1.014 for arbitrarily large n**
   ("improving on very recent work of a team at OpenAI, who proved the same
   result with an inexplicit exponent greater than 1").
3. **Alon, Bloom, Gowers, Litt, Sawin, Shankar, Tsimerman, Wang, Wood,
   "Remarks on the disproof of the unit distance conjecture"**,
   arXiv:2605.20695 (20 May 2026): https://arxiv.org/abs/2605.20695
   (also mirrored at cdn.openai.com/.../unit-distance-remarks.pdf) — "a short,
   digested, **human-verified** version of the recent OpenAI-generated
   counterexample … ideas that may, at least in retrospect, be attributed to
   Ellenberg–Venkatesh, Golod–Shafarevich, and Hajir–Maire–Ramakrishna."

### Verification-status table

| Claim | Status | Evidence |
|---|---|---|
| 2026 construction exists | **VERIFIED** | OpenAI PDF read directly; two arXiv papers (2605.20579, 2605.20695) |
| Disproves Erdős's n^(1+C/loglog n) conjectured upper bound | **VERIFIED** | Theorem 1.1 verbatim; "Remarks" abstract; MathWorld ("Refuted Conjectures" category) |
| Sawin explicit exponent n^1.014 | **VERIFIED** | arXiv:2605.20579 abstract |
| "Sawin / OpenAI collaboration" framing | **CORRECTED** | Not a collaboration: OpenAI proof first (autonomous model + human verification); Sawin's explicit bound is an independent same-day follow-up; Sawin is also a co-author of the 9-author verification "Remarks" |
| Peer-reviewed | **NO (but strongly human-verified)** | All three are preprints/PDFs, 3 weeks old. Verification = named external number theorists + the 9-author "Remarks" (Alon, Gowers, Wood, …). No journal acceptance found |
| MathWorld page updated May 2026 | **VERIFIED** | Page fetched 2026-06-10; cites OpenAI 2026, Alon et al. 2026, Sawin 2026, Pegg (Wolfram Community, May 21 2026) |
| Affects finite-n (n ≤ 100) record chasing | **NO — purely asymptotic** | Theorem is "infinitely many n" via towers with field degree f_j → ∞; n_j grows like exp(B·f_j). MathWorld on finite box pieces of the explicit construction (S. Yang / Pegg, m=5: 625 vertices, 2800 edges): "this box family still has only **linearly many** unit-distance pairs in the number of vertices". At n=100, even n^1.014 ≈ 107 ≪ record 439 |
| Upper bound changed | **NO** | Still O(n^4/3) (Spencer–Szemerédi–Trotter 1984; constant Ágoston–Pálvölgyi 2022) |

Could not verify (minor): the OpenAI blog post
(https://openai.com/index/model-disproves-discrete-geometry-conjecture/) and
FigShare both 403/WAF-block headless fetch — blog content unverified, but the
primary PDF supersedes it. Exact size of the FigShare dataset unknown.

---

## What this means for our record chase

1. **Is the n=40 record (137) a Minkowski sum?** The paper doesn't say —
   structure is undocumented for n=40/50/70 (only counts: 1, 5, and 3
   isomorphism classes respectively). But the flanking records ARE sums:
   **n=39 = (7-vertex UDG) ⊕ (8-vertex UDG)** and **n=49 = (6-wheel) ⊕
   (6-wheel)**, and 44.2% of all records are Minkowski sums. Concrete seeding
   plan (our inference, flagged as such):
   - **n=40:** seed with the n=39 sum structure (7-vertex ⊕ 8-vertex dense
     UDGs, e.g. 6-wheel ⊕ an 8-vertex 14-edge UDG) + 1 extra vertex placed at
     a max-coincidence circle intersection; also 40 = 2^3 × 5 (3 edges ⊕
     5-vertex UDG, possibly non-disjoint).
   - **n=50:** seed with the n=49 double-6-wheel sum + 1 vertex. With 5
     isomorphism classes at 183, the optimum is likely flexible — good news
     for continuous search.
   - **n=70:** 70 = 7 × 10 — try **6-wheel ⊕ (10-vertex 17-edge UDG)**
     (disjoint sum would give exactly 70 vertices); also 70 = 2 × 35
     (edge ⊕ 35-vertex record) and 64-record (6-cube) + 6 vertices.
2. **Can we fetch the record configs to compare against ours?** Yes, but only
   via the all-in-one FigShare npz (browser download required, WAF blocks
   curl): https://figshare.com/s/a7e6426c318e1d550366 — then
   `np.load(f)["40"]` etc. (shape (m, 40, 4) Moser coordinates; reconstruct
   points as a + b·ω₁ + c·ω₃ + d·ω₁ω₃ and select the max-edge-count graphs).
   Fallback: email zsamboki.pal@renyi.hu for just n ∈ {40,50,70}.
3. **Their search is ML-locked** (generator steps + triangle/parallelogram
   completions inside ML only) — our continuous circle-intersection moves
   genuinely explore space they cannot. And there is now precedent for
   record-density configs outside ML (one of the seven n=17 optima,
   Alexeev–Mixon–Parshall). Tier-2 goal stands.
4. **The 2026 development does not move our targets.** It kills the conjectured
   *asymptotic* upper-bound form, not the n^(4/3) upper bound, and provides no
   competitive finite-n configurations (linear edge density in finite
   sections). §5.C remains interesting as *reconnaissance* (rank-f Minkowski
   lattices of O_{L(i)} with many norm-one units, projected — small instances
   would need a degree-f totally real field with split primes; smallest
   interesting cases likely far exceed n=100). No urgency for the record
   chase; do not reprioritize over §5.A.
5. **HANDOFF updates needed** (do not edit in this recon session): §1 "exact
   values known for n ≤ 15" → **n ≤ 21** (Alexeev–Mixon–Parshall); §1 2026
   note can now cite the three primary documents above; "all known maximally
   dense UDGs embed in ML" → all except one n=17 optimum.

## Source list

- https://arxiv.org/abs/2406.15317 (v3) + https://arxiv.org/html/2406.15317v3
- https://codeberg.org/zsamboki/dbs-udg (+ raw README)
- https://figshare.com/s/a7e6426c318e1d550366 (dataset; browser-only)
- https://arxiv.org/abs/2412.11914 (v2)
- https://cdn.openai.com/pdf/74c24085-19b0-4534-9c90-465b8e29ad73/unit-distance-proof.pdf (read in full)
- https://arxiv.org/abs/2605.20579 (Sawin)
- https://arxiv.org/abs/2605.20695 (Alon et al. "Remarks")
- https://mathworld.wolfram.com/ErdosUnitDistanceProblem.html (fetched 2026-06-10)
- OEIS A186705 (u(n)), A385657 (counts of optima) — via MathWorld
