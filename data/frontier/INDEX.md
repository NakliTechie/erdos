# Frontier configurations, n = 16..100

Best verified unit-distance configurations at each n, staged from the
2026-06-10 closure session. One directory per n containing:

- `udg<N>_<E>edges.csv` — the configuration (float coordinates, header `x,y`);
- `udg<N>_<E>edges_audit.json` — the three-audit report (min-sep, K_2,3,
  Gauss-Newton exact realizability), as produced by `scripts/audit_config.py`
  / `udg.audit`;
- `udg<N>_<E>edges_cert.json` — the exact Moser-lattice certificate
  (`scripts/ml_coords.py`): integer ML coordinates plus an exact-arithmetic
  proof in Q(sqrt3, sqrt11) that every claimed edge has length exactly 1
  and all points are distinct.

Every file was re-verified in place before staging:
`uv run python scripts/audit_config.py data/frontier/n<N>/udg<N>_<E>edges.csv`
and `uv run python scripts/ml_coords.py <same csv>`.

Verification tally: **85/85 staged · 85/85 tie the Engel et al.
Table 2 target · 84/85 pass the float audit · 85/85 carry an
exact CERTIFIED lattice certificate.**

The single float-audit failure is **n = 96** and it is the documented expected
outcome (dual-track convention, decision 2026-06-10): the unique 418-edge
record class contains four point pairs at exact distance
sqrt((23 - 4*sqrt(33))/3) ~= 0.085146 < the 0.2 float-audit floor. The exact
certificate proves all 418 edges exactly unit and all 96 points exactly
distinct; see `docs/forensics/n96-traceball.md`. The audit-clean float-track
best at n = 96 remains 417.

Targets are the Engel et al. Table 2 densest-known values
(`data/engel_targets.json`). Source = the run artifact each staged CSV is a
byte-identical copy of (the `runs/` tree is not committed; generative recipes
for the non-sweep entries live in `scripts/recipes/`).

| n | edges | target | tie? | audit | cert | source path |
|---|-------|--------|------|-------|------|-------------|
| 16 | 41 | 41 | tie | pass | CERTIFIED | `runs/sweep/n16/udg16_41edges_final.csv` |
| 17 | 43 | 43 | tie | pass | CERTIFIED | `runs/sweep/n17/udg17_43edges_final.csv` |
| 18 | 46 | 46 | tie | pass | CERTIFIED | `runs/sweep/n18/udg18_46edges_final.csv` |
| 19 | 50 | 50 | tie | pass | CERTIFIED | `runs/sweep/n19/udg19_50edges_final.csv` |
| 20 | 54 | 54 | tie | pass | CERTIFIED | `runs/sweep/n20/udg20_54edges_final.csv` |
| 21 | 57 | 57 | tie | pass | CERTIFIED | `runs/sweep/n21/udg21_57edges_final.csv` |
| 22 | 60 | 60 | tie | pass | CERTIFIED | `runs/sweep/n22/udg22_60edges_final.csv` |
| 23 | 64 | 64 | tie | pass | CERTIFIED | `runs/sweep/n23/udg23_64edges_final.csv` |
| 24 | 68 | 68 | tie | pass | CERTIFIED | `runs/sweep/n24/udg24_68edges_final.csv` |
| 25 | 72 | 72 | tie | pass | CERTIFIED | `runs/sweep/n25/udg25_72edges_final.csv` |
| 26 | 76 | 76 | tie | pass | CERTIFIED | `runs/sweep/n26/udg26_76edges_final.csv` |
| 27 | 81 | 81 | tie | pass | CERTIFIED | `runs/sweep/n27/udg27_81edges_final.csv` |
| 28 | 85 | 85 | tie | pass | CERTIFIED | `runs/sweep/n28/udg28_85edges_final.csv` |
| 29 | 89 | 89 | tie | pass | CERTIFIED | `runs/sweep/n29/udg29_89edges_final.csv` |
| 30 | 93 | 93 | tie | pass | CERTIFIED | `runs/sweep/n30/udg30_93edges_final.csv` |
| 31 | 97 | 97 | tie | pass | CERTIFIED | `runs/sweep/n31/udg31_97edges_final.csv` |
| 32 | 101 | 101 | tie | pass | CERTIFIED | `runs/sweep/n32/udg32_101edges_final.csv` |
| 33 | 105 | 105 | tie | pass | CERTIFIED | `runs/sweep/n33/udg33_105edges_final.csv` |
| 34 | 109 | 109 | tie | pass | CERTIFIED | `runs/sweep/n34/udg34_109edges_final.csv` |
| 35 | 114 | 114 | tie | pass | CERTIFIED | `runs/sweep/n35/udg35_114edges_final.csv` |
| 36 | 119 | 119 | tie | pass | CERTIFIED | `runs/sweep/n36/udg36_119edges_final.csv` |
| 37 | 123 | 123 | tie | pass | CERTIFIED | `runs/sweep/n37/udg37_123edges_final.csv` |
| 38 | 128 | 128 | tie | pass | CERTIFIED | `runs/sweep/n38/udg38_128edges_final.csv` |
| 39 | 132 | 132 | tie | pass | CERTIFIED | `runs/sweep/n39/udg39_132edges_final.csv` |
| 40 | 137 | 137 | tie | pass | CERTIFIED | `runs/sweep/n40/udg40_137edges_final.csv` |
| 41 | 141 | 141 | tie | pass | CERTIFIED | `runs/sweep/n41/udg41_141edges_final.csv` |
| 42 | 146 | 146 | tie | pass | CERTIFIED | `runs/sweep/n42/udg42_146edges_final.csv` |
| 43 | 150 | 150 | tie | pass | CERTIFIED | `runs/sweep/n43/udg43_150edges_final.csv` |
| 44 | 155 | 155 | tie | pass | CERTIFIED | `runs/sweep/n44/udg44_155edges_final.csv` |
| 45 | 160 | 160 | tie | pass | CERTIFIED | `runs/sweep/n45/udg45_160edges_final.csv` |
| 46 | 164 | 164 | tie | pass | CERTIFIED | `runs/sweep/n46/udg46_164edges_final.csv` |
| 47 | 169 | 169 | tie | pass | CERTIFIED | `runs/sweep/n47/udg47_169edges_final.csv` |
| 48 | 174 | 174 | tie | pass | CERTIFIED | `runs/sweep/n48/udg48_174edges_final.csv` |
| 49 | 180 | 180 | tie | pass | CERTIFIED | `runs/sweep/n49/udg49_180edges_final.csv` |
| 50 | 183 | 183 | tie | pass | CERTIFIED | `runs/chase/n50/udg50_183edges_class0.csv` |
| 51 | 188 | 188 | tie | pass | CERTIFIED | `runs/sweep/n51/udg51_188edges_final.csv` |
| 52 | 192 | 192 | tie | pass | CERTIFIED | `runs/sweep/n52/udg52_192edges_final.csv` |
| 53 | 197 | 197 | tie | pass | CERTIFIED | `runs/sweep/n53/udg53_197edges_final.csv` |
| 54 | 202 | 202 | tie | pass | CERTIFIED | `runs/sweep/n54/udg54_202edges_final.csv` |
| 55 | 206 | 206 | tie | pass | CERTIFIED | `runs/sweep/n55/udg55_206edges_final.csv` |
| 56 | 211 | 211 | tie | pass | CERTIFIED | `runs/sweep/n56/udg56_211edges_final.csv` |
| 57 | 216 | 216 | tie | pass | CERTIFIED | `runs/sweep/n57/udg57_216edges_final.csv` |
| 58 | 221 | 221 | tie | pass | CERTIFIED | `runs/sweep/n58/udg58_221edges_final.csv` |
| 59 | 226 | 226 | tie | pass | CERTIFIED | `runs/sweep/n59/udg59_226edges_final.csv` |
| 60 | 231 | 231 | tie | pass | CERTIFIED | `runs/sweep/n60/udg60_231edges_final.csv` |
| 61 | 235 | 235 | tie | pass | CERTIFIED | `runs/sweep/n61/udg61_235edges_final.csv` |
| 62 | 240 | 240 | tie | pass | CERTIFIED | `runs/sweep/n62/udg62_240edges_final.csv` |
| 63 | 246 | 246 | tie | pass | CERTIFIED | `runs/sweep/n63_retry/udg63_246edges_final.csv` |
| 64 | 252 | 252 | tie | pass | CERTIFIED | `runs/sweep/n64fix/udg64_252edges.csv` |
| 65 | 256 | 256 | tie | pass | CERTIFIED | `runs/sweep/n65_retry/udg65_256edges_final.csv` |
| 66 | 261 | 261 | tie | pass | CERTIFIED | `runs/sweep/n66_retry/udg66_261edges_final.csv` |
| 67 | 266 | 266 | tie | pass | CERTIFIED | `runs/sweep/n67_retry/udg67_266edges_final.csv` |
| 68 | 271 | 271 | tie | pass | CERTIFIED | `runs/sweep/n68/udg68_271edges_final.csv` |
| 69 | 276 | 276 | tie | pass | CERTIFIED | `runs/sweep/n69/udg69_276edges_final.csv` |
| 70 | 281 | 281 | tie | pass | CERTIFIED | `data/udg70_281edges.csv` |
| 71 | 286 | 286 | tie | pass | CERTIFIED | `runs/sweep/n71_retry/udg71_286edges_final.csv` |
| 72 | 291 | 291 | tie | pass | CERTIFIED | `runs/sweep/n72_gatecheck/udg72_291edges_final.csv` |
| 73 | 296 | 296 | tie | pass | CERTIFIED | `runs/sweep/n73_retry/udg73_296edges_final.csv` |
| 74 | 301 | 301 | tie | pass | CERTIFIED | `runs/sweep/n74/udg74_301edges_final.csv` |
| 75 | 306 | 306 | tie | pass | CERTIFIED | `runs/sweep/n75/udg75_306edges_final.csv` |
| 76 | 312 | 312 | tie | pass | CERTIFIED | `runs/sweep/n76retry/udg76_312edges_final.csv` |
| 77 | 317 | 317 | tie | pass | CERTIFIED | `runs/sweep/n77_retry/udg77_317edges_final.csv` |
| 78 | 322 | 322 | tie | pass | CERTIFIED | `runs/sweep/n78/udg78_322edges_final.csv` |
| 79 | 327 | 327 | tie | pass | CERTIFIED | `runs/sweep/n79/udg79_327edges_final.csv` |
| 80 | 332 | 332 | tie | pass | CERTIFIED | `runs/sweep/n80/udg80_332edges_final.csv` |
| 81 | 338 | 338 | tie | pass | CERTIFIED | `runs/sweep/n81/udg81_338edges_final.csv` |
| 82 | 345 | 345 | tie | pass | CERTIFIED | `runs/sweep/n82/udg82_345edges_final.csv` |
| 83 | 350 | 350 | tie | pass | CERTIFIED | `runs/sweep/n83/udg83_350edges_final.csv` |
| 84 | 355 | 355 | tie | pass | CERTIFIED | `runs/sweep/n84/udg84_355edges_final.csv` |
| 85 | 360 | 360 | tie | pass | CERTIFIED | `runs/sweep/n85_retry/udg85_360edges_final.csv` |
| 86 | 365 | 365 | tie | pass | CERTIFIED | `runs/sweep/n86_retry/udg86_365edges_final.csv` |
| 87 | 370 | 370 | tie | pass | CERTIFIED | `runs/sweep/n87/udg87_370edges_final.csv` |
| 88 | 375 | 375 | tie | pass | CERTIFIED | `runs/sweep/n88/udg88_375edges_final.csv` |
| 89 | 380 | 380 | tie | pass | CERTIFIED | `runs/sweep/n89retry/udg89_380edges_final.csv` |
| 90 | 385 | 385 | tie | pass | CERTIFIED | `runs/last3/n90/udg90_385edges_trim91drop1.csv` |
| 91 | 390 | 390 | tie | pass | CERTIFIED | `runs/sweep/n91/udg91_390edges_final.csv` |
| 92 | 396 | 396 | tie | pass | CERTIFIED | `runs/sweep/n92retry/udg92_396edges_final.csv` |
| 93 | 401 | 401 | tie | pass | CERTIFIED | `runs/sweep/n93/udg93_401edges_final.csv` |
| 94 | 406 | 406 | tie | pass | CERTIFIED | `runs/sweep/n94_retry/udg94_406edges_final.csv` |
| 95 | 412 | 412 | tie | pass | CERTIFIED | `runs/sweep/n95/udg95_412edges_final.csv` |
| 96 | 418 | 418 | tie | FAIL (expected: min_sep 0.085146, cert-track) | CERTIFIED | `runs/last3/n96/udg96_418edges_traceball.csv` |
| 97 | 423 | 423 | tie | pass | CERTIFIED | `runs/sweep/n97/udg97_423edges_final.csv` |
| 98 | 429 | 429 | tie | pass | CERTIFIED | `runs/last3/n98/udg98_429edges_recipe.csv` |
| 99 | 434 | 434 | tie | pass | CERTIFIED | `runs/sweep/n99/udg99_434edges_final.csv` |
| 100 | 439 | 439 | tie | pass | CERTIFIED | `runs/sweep/n100/udg100_439edges_final.csv` |
