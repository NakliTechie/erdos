#!/usr/bin/env python
"""Minkowski-sum scan for the n=50 record chase.

1. Build a library of dense small ML UDGs (sizes 2..13) by exact beam
   search (add-best-point with branching) from several seeds, keeping the
   top-K symmetry-distinct configs per size; plus the named library pieces
   and their w3-twisted copies ((a,b,0,0) -> (0,0,a,b)) and conjugates.
2. For factor pairs (A, B) with |A|*|B| in [47, 53] (and a few triples),
   compute the exact Minkowski sum; complete to n=50 by greedy add-best
   (enumerated exactly) or exact drop-k (k <= 3, exhaustive over subsets).
3. Report every result with >= --report edges; checkpoint >= --save.

Usage: uv run python scripts/chase50_minkowski.py [--report 178] [--save 183]
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from udg.mlgraph import (
    MLConfig,
    _as_coeff_array,
    _unit_mask,
    exact_edge_count,
    degrees,
    minkowski,
    moser_spindle,
    save_csv,
    tri_patch,
    unit_edge,
    unit_rhombus,
    unit_triangle,
    wheel6,
)
from udg.moser import UNIT_COEFFS

from chase50_climb import canon_full  # noqa: E402

UC = np.array(UNIT_COEFFS, dtype=np.int64)

# record values u(n) for small n (OEIS A186705, known exactly to n=21)
SMALL_RECORDS = {2: 1, 3: 3, 4: 5, 5: 7, 6: 9, 7: 12, 8: 14, 9: 18, 10: 20,
                 11: 23, 12: 27, 13: 30}


def w3_twist(cfg: MLConfig) -> MLConfig | None:
    """Multiply by w3: only valid for Eisenstein-sublattice configs
    (c = d = 0): (a, b, 0, 0) -> (0, 0, a, b)."""
    pts = []
    for a, b, c, d in cfg.points:
        if c or d:
            return None
        pts.append((0, 0, a, b))
    return MLConfig(pts)


def conj_e(cfg: MLConfig) -> MLConfig | None:
    """Complex conjugation for Eisenstein configs: (a,b,0,0)->(a+b,-b,0,0)."""
    pts = []
    for a, b, c, d in cfg.points:
        if c or d:
            return None
        pts.append((a + b, -b, 0, 0))
    return MLConfig(pts)


def rot_w1_cfg(cfg: MLConfig, k: int) -> MLConfig:
    def rot(p):
        a, b, c, d = p
        return (-b, a + b, -d, c + d)
    pts = list(cfg.points)
    for _ in range(k):
        pts = [rot(p) for p in pts]
    return MLConfig(pts)


# ---------------------------------------------------------------------------
# beam search for dense small pieces
# ---------------------------------------------------------------------------

def beam_grow(max_n: int, width: int = 64) -> dict[int, list[MLConfig]]:
    """Grow configs point-by-point keeping `width` best (by edges, then
    compactness) symmetry-distinct configs per size. Returns per-size list."""
    start = MLConfig([(0, 0, 0, 0)])
    beam = [start]
    out: dict[int, list[MLConfig]] = {1: [start]}
    for n in range(2, max_n + 1):
        scored: dict[tuple, tuple[int, int, MLConfig]] = {}
        for cfg in beam:
            arr = cfg.as_array()
            cand = (arr[:, None, :] + UC[None, :, :]).reshape(-1, 4)
            cset = sorted({tuple(int(x) for x in r) for r in cand} - cfg._set)
            CA = _as_coeff_array(cset)
            gains = _unit_mask(CA, arr).sum(axis=1)
            base = exact_edge_count(cfg)
            # keep the top additions per parent
            order = np.argsort(-gains)[: width]
            for i in order:
                nc = cfg.with_point(cset[int(i)])
                cf = canon_full(nc.points)
                if cf in scored:
                    continue
                m = base + int(gains[int(i)])
                spread = max(abs(x) for p in nc.points for x in p)
                scored[cf] = (m, -spread, nc)
        ranked = sorted(scored.values(), key=lambda t: (-t[0], t[1]))
        beam = [t[2] for t in ranked[:width]]
        out[n] = [t[2] for t in ranked[: max(8, width // 4)]]
        best = ranked[0][0]
        rec = SMALL_RECORDS.get(n)
        print(f"  beam n={n}: best {best} edges"
              + (f" (record {rec})" if rec else ""), flush=True)
    return out


# ---------------------------------------------------------------------------
# completion: exact add-best / drop-k
# ---------------------------------------------------------------------------

def add_best_k(cfg: MLConfig, k: int, width: int = 48) -> MLConfig:
    """Beam over k exact best-addition steps (width-limited, exhaustive per
    step over ALL candidate positions); returns the best final config."""
    beam: list[tuple[int, MLConfig]] = [(exact_edge_count(cfg), cfg)]
    for _ in range(k):
        nxt: dict[tuple, tuple[int, MLConfig]] = {}
        for base, c in beam:
            arr = c.as_array()
            cand = (arr[:, None, :] + UC[None, :, :]).reshape(-1, 4)
            cset = sorted({tuple(int(x) for x in r) for r in cand} - c._set)
            CA = _as_coeff_array(cset)
            gains = _unit_mask(CA, arr).sum(axis=1)
            order = np.argsort(-gains)[:width]
            for i in order:
                nc = c.with_point(cset[int(i)])
                cf = canon_full(nc.points)
                if cf not in nxt:
                    nxt[cf] = (base + int(gains[int(i)]), nc)
        beam = sorted(nxt.values(), key=lambda t: -t[0])[:width]
    return beam[0][1]


def drop_k_exact(cfg: MLConfig, k: int) -> MLConfig:
    """Exhaustive best k-subset removal (maximize remaining edges)."""
    arr = cfg.as_array()
    A = _unit_mask(arr, arr)
    np.fill_diagonal(A, False)
    deg = A.sum(axis=1).astype(np.int64)
    n = len(cfg)
    best_lost, best_subset = None, None
    for sub in itertools.combinations(range(n), k):
        idx = list(sub)
        # edges lost = sum deg - edges within subset (those are double-counted)
        within = int(A[np.ix_(idx, idx)].sum()) // 2
        lost = int(deg[idx].sum()) - within
        if best_lost is None or lost < best_lost:
            best_lost, best_subset = lost, set(idx)
    pts = [p for i, p in enumerate(cfg.points) if i not in best_subset]
    return MLConfig(pts)


def complete_to_50(cfg: MLConfig) -> MLConfig | None:
    n = len(cfg)
    if n < 47 or n > 53:
        return None
    if n < 50:
        return add_best_k(cfg, 50 - n)
    if n > 50:
        return drop_k_exact(cfg, n - 50)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=int, default=178)
    ap.add_argument("--save", type=int, default=183)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--max-piece", type=int, default=16)
    ap.add_argument("--out", default="runs/chase/n50")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("growing piece library by beam search ...", flush=True)
    lib = beam_grow(args.max_piece, width=args.beam_width)

    # add named pieces
    named = {
        2: [unit_edge()],
        3: [unit_triangle()],
        4: [unit_rhombus()],
        7: [wheel6("w1"), wheel6("w3"), moser_spindle()],
        6: [tri_patch(2)],
        10: [tri_patch(3)],
    }
    for n, cfgs in named.items():
        lib.setdefault(n, []).extend(cfgs)

    # expand variants: w3-twist, conjugate, w1-rotations (rotations only for
    # the SECOND factor; dedupe by canonical form)
    variants: dict[int, list[MLConfig]] = {}
    for n, cfgs in lib.items():
        seen = {}
        allv = []
        for cfg in cfgs:
            vs = [cfg]
            tw = w3_twist(cfg)
            if tw is not None:
                vs.append(tw)
            cj = conj_e(cfg)
            if cj is not None:
                vs.append(cj)
                tw2 = w3_twist(cj)
                if tw2 is not None:
                    vs.append(tw2)
            for v in vs:
                for k in range(6):
                    r = rot_w1_cfg(v, k) if k else v
                    cf = canon_translate(r)
                    if cf not in seen:
                        seen[cf] = None
                        allv.append(r)
        variants[n] = allv[:36]
    for n in sorted(variants):
        print(f"  pieces n={n}: {len(variants[n])} variants", flush=True)

    # pair scan
    print("pair scan ...", flush=True)
    results = []
    seen_sums: set = set()
    sizes = sorted(variants)
    for na, nb in itertools.combinations_with_replacement(sizes, 2):
        if not (40 <= na * nb <= 56):
            continue
        for A in variants[na]:
            for B in variants[nb]:
                s = minkowski(A, B)
                ns = len(s)
                if ns < 47 or ns > 53:
                    continue
                cf = canon_full(s.points)
                if cf in seen_sums:
                    continue
                seen_sums.add(cf)
                done = complete_to_50(s)
                if done is None or len(done) != 50:
                    continue
                m = exact_edge_count(done)
                if m >= args.report:
                    results.append((m, na, nb, ns, done))
                    print(f"  {na}x{nb}: sum n={ns} -> completed 50 pts, {m} edges",
                          flush=True)
                    if m >= args.save:
                        tag = f"mink_{na}x{nb}_{m}_{len(results):03d}"
                        with open(outdir / f"udg50_{m}edges_{tag}.json", "w") as f:
                            json.dump({"n": 50, "exact_edges": m, "method": tag,
                                       "coords": [list(p) for p in done.points]}, f)
                        save_csv(done, outdir / f"udg50_{m}edges_{tag}.csv")
    # triples with the edge factor (2 x a x b)
    print("triple scan (2 x a x b) ...", flush=True)
    for na, nb in itertools.combinations_with_replacement(sizes, 2):
        if not (40 <= 2 * na * nb <= 56):
            continue
        for A in variants[na][:6]:
            for B in variants[nb][:6]:
                base = minkowski(A, B)
                for E in variants[2]:
                    s = minkowski(base, E)
                    ns = len(s)
                    if ns < 47 or ns > 53:
                        continue
                    cf = canon_full(s.points)
                    if cf in seen_sums:
                        continue
                    seen_sums.add(cf)
                    done = complete_to_50(s)
                    if done is None or len(done) != 50:
                        continue
                    m = exact_edge_count(done)
                    if m >= args.report:
                        results.append((m, 2 * na, nb, ns, done))
                        print(f"  2x{na}x{nb}: sum n={ns} -> 50 pts, {m} edges",
                              flush=True)
                        if m >= args.save:
                            tag = f"mink_2x{na}x{nb}_{m}_{len(results):03d}"
                            with open(outdir / f"udg50_{m}edges_{tag}.json", "w") as f:
                                json.dump({"n": 50, "exact_edges": m, "method": tag,
                                           "coords": [list(p) for p in done.points]}, f)
                            save_csv(done, outdir / f"udg50_{m}edges_{tag}.csv")

    results.sort(key=lambda t: -t[0])
    print(f"=== top results ({time.time()-t0:.0f}s) ===", flush=True)
    for m, na, nb, ns, _ in results[:20]:
        print(f"  {m} edges  ({na}x{nb}, sum n={ns})", flush=True)
    if not results:
        print("  (none above report threshold)", flush=True)


def canon_translate(cfg: MLConfig) -> tuple:
    pts = sorted(cfg.points)
    t = pts[0]
    return tuple((p[0]-t[0], p[1]-t[1], p[2]-t[2], p[3]-t[3]) for p in pts)


if __name__ == "__main__":
    main()
