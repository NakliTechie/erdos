import numpy as np
from math import gcd

def bisector_energy(P):
    """P: integer points, shape (n,2). Returns (E, E_nontriv, num_lines, max_m, top lines).
    Bisector of (a,b): 2(b-a).x = |b|^2-|a|^2, normalized integer triple."""
    P = np.asarray(P, dtype=np.int64)
    n = len(P)
    i, j = np.triu_indices(n, k=1)
    A = 2*(P[j,0]-P[i,0]); B = 2*(P[j,1]-P[i,1])
    C = (P[j,0]**2+P[j,1]**2) - (P[i,0]**2+P[i,1]**2)
    # normalize each triple
    g = np.gcd(np.gcd(np.abs(A), np.abs(B)), np.abs(C))
    g[g==0] = 1
    A//=g; B//=g; C//=g
    # sign fix: first nonzero of (A,B) positive
    s = np.where(A!=0, np.sign(A), np.sign(B)).astype(np.int64)
    s[s==0]=1
    A*=s; B*=s; C*=s
    key = (A.astype(object)*(10**12) + B.astype(object))*(10**12) + C.astype(object)  # safe combo
    # use structured numpy unique instead (faster, exact)
    T = np.stack([A,B,C], axis=1)
    Tv = np.ascontiguousarray(T).view([('a',np.int64),('b',np.int64),('c',np.int64)])
    u, counts = np.unique(Tv, return_counts=True)
    E = int((counts.astype(object)**2).sum())
    E_nt = int((counts[counts>1].astype(object)**2).sum())
    return E, E_nt, len(u), counts.max(), len(i)

def report(name, P):
    E, Ent, L, mx, pairs = bisector_energy(P)
    n = len(P)
    print(f"{name:>14}: n={n:>5} pairs={pairs:>8} E={E:>12} E_nontriv={Ent:>12} "
          f"maxm={mx:>5}  E/n^2={E/n**2:>8.2f}  log_n(E)={np.log(E)/np.log(n):.3f}")

m = 40
grid = [(x,y) for x in range(m) for y in range(m)]
report("square grid", grid)

# triangular lattice embedded in integers: (2i+j, j) scaled -> use (2i+j, j) with metric distorted,
# but bisectors under Euclidean metric of the TRUE triangular lattice need irrational coords.
# Use exact rational trick: scale by 2 -> points (2i+j, j*sqrt3). Bisector equality involves sqrt3.
# Represent y-coords in units of sqrt3: point = (px, py*sqrt3), px,py ints.
# |b|^2-|a|^2 = (bx^2-ax^2) + 3(by^2-ay^2): integer. Bisector: 2(bx-ax)x + 2*3*(by-ay)*(y-part...) 
# line: 2(bx-ax)*X + 2(by-ay)*3*Yt = C where Y = Yt*sqrt3. All integer triples -> exact.
def tri_energy(m):
    pts = np.array([(2*i+j, j) for i in range(m) for j in range(m)], dtype=np.int64)
    n=len(pts)
    i,j = np.triu_indices(n,k=1)
    A = 2*(pts[j,0]-pts[i,0])
    B = 6*(pts[j,1]-pts[i,1])
    C = (pts[j,0]**2+3*pts[j,1]**2)-(pts[i,0]**2+3*pts[i,1]**2)
    g = np.gcd(np.gcd(np.abs(A),np.abs(B)),np.abs(C)); g[g==0]=1
    A//=g; B//=g; C//=g
    s = np.where(A!=0, np.sign(A), np.sign(B)).astype(np.int64); s[s==0]=1
    A*=s;B*=s;C*=s
    T = np.stack([A,B,C],axis=1)
    Tv = np.ascontiguousarray(T).view([('a',np.int64),('b',np.int64),('c',np.int64)])
    u,counts = np.unique(Tv,return_counts=True)
    E=int((counts.astype(object)**2).sum()); Ent=int((counts[counts>1].astype(object)**2).sum())
    print(f"{'tri lattice':>14}: n={n:>5} pairs={len(i):>8} E={E:>12} E_nontriv={Ent:>12} "
          f"maxm={counts.max():>5}  E/n^2={E/n**2:>8.2f}  log_n(E)={np.log(E)/np.log(n):.3f}")
tri_energy(m)

rng = np.random.default_rng(7)
rand = rng.integers(0, 10**7, size=(m*m,2))
report("random", rand)

# perturbed grid: grid + small random jitter (integer, +-1 on a 10x finer scale)
pg = (np.array(grid)*10 + rng.integers(-1,2,size=(m*m,2)))
report("perturbed grid", pg)
