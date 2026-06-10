import numpy as np
from collections import Counter
import math

def r2(k):
    """Number of representations of k as a^2+b^2 (ordered, with signs)."""
    c = 0
    a = 0
    while a*a <= k:
        b2 = k - a*a
        b = math.isqrt(b2)
        if b*b == b2:
            c += (2 if a else 1) * (2 if b else 1) * (1 if a != b else 1)
            if a != b and a and b:
                c += 4  # (a,b) and (b,a) with signs... compute carefully below
        a += 1
    # simpler exact method:
    c = 0
    for x in range(-math.isqrt(k), math.isqrt(k)+1):
        y2 = k - x*x
        y = math.isqrt(y2)
        if y*y == y2:
            c += 1 if y == 0 else 2
    return c

def popular(points, topk=5):
    P = np.asarray(points, dtype=np.int64)
    n = len(P)
    cnt = Counter()
    chunk = 1200
    for i in range(0, n, chunk):
        A = P[i:i+chunk]
        D = (A[:, None, 0]-P[None, :, 0])**2 + (A[:, None, 1]-P[None, :, 1])**2
        gi = np.arange(i, i+len(A))[:, None]; gj = np.arange(n)[None, :]
        vals = D[gi < gj]
        u, c = np.unique(vals, return_counts=True)
        for uu, cc in zip(u.tolist(), c.tolist()):
            cnt[uu] += cc
    return n, cnt.most_common(topk)

# 1) verify: winner's count vs n*r2(k)/2 prediction (interior points see r2(k) neighbors)
print("=== square grid 50x50: top distances vs r2 prediction ===")
pts = [(x, y) for x in range(50) for y in range(50)]
n, top = popular(pts)
for d2, c in top:
    print(f"d^2={d2:>5}  count={c:>6}  r2={r2(d2):>3}  n*r2/2={n*r2(d2)//2:>6}  ratio={2*c/(n*r2(d2)):.3f}")

# 2) triangular lattice (Eisenstein integers): same test, squared norms x^2+xy+y^2
print("\n=== triangular lattice ~2500 pts: top distances ===")
tri = []
m = 50
for i in range(m):
    for j in range(m):
        tri.append((2*i + j, j))  # scaled: dist^2 = ((2i+j)-(2i'+j'))^2*1/4*... 
# use exact integer quadratic form instead: norm(a,b)=a^2+ab+b^2 via direct counting
P = [(i, j) for i in range(m) for j in range(m)]
cnt = Counter()
A = np.array(P, dtype=np.int64)
chunk = 1200
nn = len(A)
for i in range(0, nn, chunk):
    B = A[i:i+chunk]
    da = B[:, None, 0]-A[None, :, 0]; db = B[:, None, 1]-A[None, :, 1]
    D = da*da + da*db + db*db
    gi = np.arange(i, i+len(B))[:, None]; gj = np.arange(nn)[None, :]
    vals = D[gi < gj]
    u, c = np.unique(vals, return_counts=True)
    for uu, cc in zip(u.tolist(), c.tolist()):
        cnt[uu] += cc
for d2, c in cnt.most_common(5):
    print(f"norm={d2:>5}  count={c:>6}  count/n={c/nn:.3f}")

# 3) growth fit: U(n) = n * exp(c log n / log log n)?
print("\n=== growth fit on square grids ===")
data = [(25,48),(64,168),(144,456),(289,1136),(576,2832),(1156,6672),(2304,15864),(3600,28200)]
for n_, u in data:
    c_est = (math.log(u/n_)) * math.log(math.log(n_)) / math.log(n_)
    print(f"n={n_:>5}  U/n={u/n_:>7.3f}  implied c={c_est:.4f}")
