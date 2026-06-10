"""Metropolis search over circle-intersection moves (faithful port of exp6_clean.py).

Moves are geometric (pitfall HANDOFF §3): a point is only ever re-placed at an
intersection of two unit circles around existing points, so every accepted move
creates >= 2 exact unit edges; surplus edges arise via coincidence. Candidate
placements violating the hard minimum separation are rejected outright.

Self-contained: stdlib + numpy only, no intra-package imports.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

TOL = 1e-9        # unit-distance tolerance for float configs
MIN_SEP = 0.2     # hard minimum separation (pitfall §3.1)


@dataclass
class SearchResult:
    best_count: int
    P: np.ndarray
    n: int
    steps: int
    seed: int


def circle_intersections(a: np.ndarray, b: np.ndarray) -> list[np.ndarray]:
    """Intersections of the two unit circles centered at a and b (port of exp6)."""
    d2 = ((a - b) ** 2).sum()
    if d2 >= 4.0 or d2 < 1e-12:
        return []
    d = np.sqrt(d2)
    mid = (a + b) / 2
    h2 = 1.0 - d2 / 4.0
    if h2 < 0:
        return []
    h = np.sqrt(h2)
    perp = np.array([-(b - a)[1], (b - a)[0]]) / d
    return [mid + h * perp, mid - h * perp]


def search(
    n: int = 40,
    steps: int = 150_000,
    seed: int = 0,
    *,
    min_sep: float = MIN_SEP,
    tol: float = TOL,
    box: float = 3.5,
    T0: float = 1.2,
    T1: float = 0.015,
    P0: np.ndarray | None = None,
) -> SearchResult:
    """Faithful port of exp6_clean.py search().

    Metropolis annealing over circle-intersection moves with a geometric
    temperature schedule T0 -> T1, hard min-sep rejection of every candidate
    placement, and incremental O(n) unit-count updates (moving point k only
    changes edges incident to k).

    P0: optional warm start (hinge-locking / Minkowski seeds). When given,
    n = len(P0) and the random box initialization is skipped.
    Determinism: same arguments (incl. seed) -> identical result.
    """
    rng = np.random.default_rng(seed)
    if P0 is not None:
        P = np.array(P0, dtype=np.float64, copy=True)
        n = len(P)
    else:
        # warm start options: random OR distorted lattice; random to be unbiased
        P = rng.uniform(0, box, size=(n, 2))

    def count_at(pt, P, k):
        d = np.sqrt(((P - pt) ** 2).sum(1))
        d[k] = np.inf
        if d.min() < min_sep:
            return None
        return int((np.abs(d - 1) < tol).sum())

    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    cur = int((np.abs(D - 1) < tol).sum() // 2)
    best = cur
    bestP = P.copy()
    for t in range(steps):
        T = T0 * (T1 / T0) ** (t / steps)
        k = rng.integers(n)
        a, b = rng.choice(n, 2, replace=False)
        if a == k or b == k:
            continue
        cands = circle_intersections(P[a], P[b])
        if not cands:
            continue
        cand = cands[rng.integers(len(cands))]
        cn = count_at(cand, P, k)
        if cn is None:
            continue
        dold = np.sqrt(((P - P[k]) ** 2).sum(1))
        dold[k] = np.inf
        co = int((np.abs(dold - 1) < tol).sum())
        delta = cn - co
        if delta >= 0 or rng.random() < np.exp(delta / T):
            P[k] = cand
            cur += delta
            if cur > best:
                best = cur
                bestP = P.copy()
    return SearchResult(best_count=best, P=bestP, n=n, steps=steps, seed=seed)


def _search_worker(args) -> SearchResult:
    """Top-level worker so ProcessPoolExecutor can pickle it (macOS spawn-safe)."""
    n, steps, seed, kw = args
    return search(n=n, steps=steps, seed=seed, **kw)


def multi_search(
    n: int,
    seeds: list[int],
    steps: int,
    *,
    processes: int | None = None,
    **kw,
) -> list[SearchResult]:
    """Fan search() out across seeds with a process pool.

    Results are returned in the order of `seeds`. Each seed gets its own
    independent rng, so per-seed results are identical to a serial search().
    """
    jobs = [(n, steps, seed, kw) for seed in seeds]
    with ProcessPoolExecutor(max_workers=processes) as ex:
        return list(ex.map(_search_worker, jobs))
