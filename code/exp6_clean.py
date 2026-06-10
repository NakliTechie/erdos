import numpy as np
TOL = 1e-9
MINSEP = 0.2

def unit_edges(P):
    D = np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(-1))
    n=len(P)
    return [(i,j) for i in range(n) for j in range(i+1,n) if abs(D[i,j]-1)<TOL], D

def circle_intersections(a, b):
    d2 = ((a-b)**2).sum()
    if d2 >= 4.0 or d2 < 1e-12: return []
    d = np.sqrt(d2); mid=(a+b)/2; h2=1.0-d2/4.0
    if h2 < 0: return []
    h=np.sqrt(h2); perp=np.array([-(b-a)[1],(b-a)[0]])/d
    return [mid+h*perp, mid-h*perp]

def search(n=40, steps=150_000, seed=0):
    rng = np.random.default_rng(seed)
    # warm start options: random OR distorted lattice; use random to be unbiased
    P = rng.uniform(0, 3.5, size=(n,2))
    def count_at(pt, P, k):
        d = np.sqrt(((P-pt)**2).sum(1)); d[k]=np.inf
        if d.min() < MINSEP: return None
        return int((np.abs(d-1)<TOL).sum())
    D = np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(-1))
    cur = int((np.abs(D-1)<TOL).sum()//2)
    best=cur; bestP=P.copy()
    T0,T1 = 1.2, 0.015
    for t in range(steps):
        T = T0*(T1/T0)**(t/steps)
        k = rng.integers(n)
        a,b = rng.choice(n,2,replace=False)
        if a==k or b==k: continue
        cands = circle_intersections(P[a],P[b])
        if not cands: continue
        cand = cands[rng.integers(len(cands))]
        cn = count_at(cand, P, k)
        if cn is None: continue
        dold = np.sqrt(((P-P[k])**2).sum(1)); dold[k]=np.inf
        co = int((np.abs(dold-1)<TOL).sum())
        delta = cn-co
        if delta>=0 or rng.random()<np.exp(delta/T):
            P[k]=cand; cur+=delta
            if cur>best: best=cur; bestP=P.copy()
    return best,bestP

def audit(P):
    E,D = unit_edges(P)
    n=len(P)
    # min separation
    Dx=D.copy(); np.fill_diagonal(Dx,np.inf)
    # K_{2,3} check: any pair with >=3 common unit-neighbors?
    adj=[set() for _ in range(n)]
    for i,j in E: adj[i].add(j); adj[j].add(i)
    bad=0
    for i in range(n):
        for j in range(i+1,n):
            if len(adj[i]&adj[j])>=3: bad+=1
    return len(E), Dx.min(), bad

allres=[]
for seed in range(8):
    best,P = search(seed=seed)
    e,ms,bad = audit(P)
    print(f"seed {seed}: count={best} audited_edges={e} min_sep={ms:.3f} K23_violations={bad}")
    allres.append((e,P))
e,P = max(allres,key=lambda r:r[0])
np.save('/home/claude/clean40.npy', P)
print(f"\nBEST LEGIT n=40: {e} unit distances (tri-lattice patch benchmark: 96)")
