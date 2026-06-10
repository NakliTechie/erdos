import numpy as np

rng = np.random.default_rng(42)

def smooth_score(D, sigma):
    """D: condensed upper-tri distances. Gaussian bump at d=1."""
    return np.exp(-((D-1.0)**2)/(2*sigma*sigma)).sum()

def anneal(n=40, steps=400_000, seed=0, box=4.0):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0, box, size=(n,2))
    # full distance matrix
    diff = P[:,None,:]-P[None,:,:]
    Dm = np.sqrt((diff**2).sum(-1))
    iu = np.triu_indices(n,1)

    def row_d(P, k):
        d = np.sqrt(((P - P[k])**2).sum(1)); d[k]=np.inf  # exclude self
        return d

    sigma0, sigma1 = 0.15, 0.006
    amp0, amp1 = 0.6, 0.01
    T0, T1 = 2.0, 0.01

    # current smooth score contribution bookkeeping: recompute per-row deltas
    def f(d, sigma): return np.exp(-((d-1.0)**2)/(2*sigma*sigma))
    # initialize per-row sums lazily; simpler: track total via row updates
    sigma = sigma0
    Fm = f(Dm, sigma); np.fill_diagonal(Fm, 0.0)
    total = Fm.sum()/2

    for t in range(steps):
        frac = t/steps
        sigma_new = sigma0*(sigma1/sigma0)**frac
        if abs(sigma_new-sigma)/sigma > 0.02:  # resync on sigma change
            sigma = sigma_new
            Fm = f(Dm, sigma); np.fill_diagonal(Fm,0.0)
            total = Fm.sum()/2
        amp = amp0*(amp1/amp0)**frac
        T = T0*(T1/T0)**frac
        k = rng.integers(n)
        old = P[k].copy()
        P[k] = old + rng.normal(0, amp, 2)
        dnew = np.sqrt(((P-P[k])**2).sum(1)); dnew[k]=np.inf
        fnew = f(dnew, sigma); fnew[k]=0.0
        delta = fnew.sum() - Fm[k].sum()
        if delta >= 0 or rng.random() < np.exp(delta/T):
            total += delta
            Dm[k,:]=dnew; Dm[:,k]=dnew
            Fm[k,:]=fnew; Fm[:,k]=fnew
        else:
            P[k]=old
    return P, Dm

def snap(P, tol=0.03, iters=200):
    """Gauss-Newton: push near-unit pairs to exactly 1."""
    P = P.copy(); n=len(P)
    diff = P[:,None,:]-P[None,:,:]; D=np.sqrt((diff**2).sum(-1))
    E = [(i,j) for i in range(n) for j in range(i+1,n) if abs(D[i,j]-1)<tol]
    for _ in range(iters):
        grad = np.zeros_like(P); res=0.0
        for (i,j) in E:
            v = P[i]-P[j]; d=np.linalg.norm(v); r=d-1.0; res+=r*r
            g = (r/d)*v
            grad[i]+=g; grad[j]-=g
        P -= 0.5*grad
        if res < 1e-22: break
    diff = P[:,None,:]-P[None,:,:]; D=np.sqrt((diff**2).sum(-1))
    cnt = sum(1 for (i,j) in E if abs(D[i,j]-1)<1e-9)
    return P, E, cnt, res

def analyze(P, E):
    """Angle spectrum of unit edges and lattice test."""
    angs=[]
    for (i,j) in E:
        v=P[j]-P[i]; a=np.degrees(np.arctan2(v[1],v[0]))%180.0
        angs.append(a)
    angs=np.array(sorted(angs))
    # cluster angles within 1 degree
    clusters=[]; cur=[angs[0]]
    for a in angs[1:]:
        if a-cur[-1]<1.0: cur.append(a)
        else: clusters.append(cur); cur=[a]
    clusters.append(cur)
    summary = sorted([(len(c), float(np.mean(c))) for c in clusters], reverse=True)
    return summary

best=None
for seed in range(4):
    P,_ = anneal(n=40, steps=250_000, seed=seed)
    P2,E,cnt,res = snap(P)
    print(f"seed {seed}: near-unit edges={len(E)}  exact-after-snap={cnt}  residual={res:.2e}")
    if best is None or cnt>best[0]:
        best=(cnt,P2,E)
cnt,P2,E = best
print(f"\nBEST n=40: {cnt} unit distances")
summ = analyze(P2,E)
print("edge-direction clusters (count, mean angle deg):")
for c,a in summ[:10]: print(f"  {c:>3} edges @ {a:7.2f} deg")
# triangular lattice benchmark for n=40 (best 40-point patch of tri lattice)
# greedy: take 40 lattice points minimizing... use a roughly hexagonal patch
import itertools
lat = [(i+0.5*j, j*np.sqrt(3)/2) for i in range(-5,6) for j in range(-5,6)]
lat = sorted(lat, key=lambda p: p[0]**2+p[1]**2)[:40]
L=np.array(lat)
DL=np.sqrt(((L[:,None,:]-L[None,:,:])**2).sum(-1))
ul = int((np.abs(DL-1)<1e-9).sum()//2)
print(f"\ntriangular-lattice 40-point patch benchmark: {ul} unit distances")
np.save('/home/claude/best40.npy', P2)
