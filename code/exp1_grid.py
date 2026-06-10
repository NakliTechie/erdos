import numpy as np
from collections import Counter

def max_multiplicity(points):
    """Count pairwise squared distances, return (best_d2, count, total_pairs)."""
    P = np.asarray(points, dtype=np.int64)
    n = len(P)
    # chunked pairwise squared distances (integer-exact)
    cnt = Counter()
    chunk = 1000
    for i in range(0, n, chunk):
        A = P[i:i+chunk]
        D = (A[:, None, 0] - P[None, :, 0])**2 + (A[:, None, 1] - P[None, :, 1])**2
        # only upper triangle: mask pairs (global_i < j)
        gi = np.arange(i, i+len(A))[:, None]
        gj = np.arange(n)[None, :]
        vals = D[gi < gj]
        u, c = np.unique(vals, return_counts=True)
        for uu, cc in zip(u.tolist(), c.tolist()):
            cnt[uu] += cc
    best_d2, best_c = cnt.most_common(1)[0]
    return best_d2, best_c, n*(n-1)//2

print(f"{'m':>4} {'n':>6} {'best d^2':>9} {'U(n)':>8} {'U/n':>7} {'U/n^(4/3)':>10}")
for m in [5, 8, 12, 17, 24, 34, 48, 60]:
    pts = [(x, y) for x in range(m) for y in range(m)]
    d2, c, _ = max_multiplicity(pts)
    n = m*m
    print(f"{m:>4} {n:>6} {d2:>9} {c:>8} {c/n:>7.3f} {c/n**(4/3):>10.4f}")
