import numpy as np

TOL = 1e-7

def unit_count(P):
    D = np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(-1))
    return int((np.abs(D-1.0)<TOL).sum()//2)

def circle_intersections(a, b):
    """Intersections of unit circles centered at a and b."""
    d2 = ((a-b)**2).sum()
    if d2 >= 4.0 or d2 < 1e-12: return []
    d = np.sqrt(d2)
    mid = (a+b)/2
    h2 = 1.0 - d2/4.0
    if h2 < 0: return []
    h = np.sqrt(h2)
    perp = np.array([-(b-a)[1], (b-a)[0]])/d
    return [mid + h*perp, mid - h*perp]

def search(n=40, steps=120_000, seed=0, restarts_p=0.0):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0, 3.0, size=(n,2))
    # greedy warm start: rebuild each point as a circle intersection
    cur = unit_count(P)
    best = cur; bestP = P.copy()
    T0, T1 = 1.5, 0.02
    for t in range(steps):
        T = T0*(T1/T0)**(t/steps)
        k = rng.integers(n)
        a, b = rng.choice(n, 2, replace=False)
        if a==k or b==k: continue
        cands = circle_intersections(P[a], P[b])
        if not cands: continue
        cand = cands[rng.integers(len(cands))]
        # reject if it lands (nearly) on an existing point
        dmin = np.sqrt(((P-cand)**2).sum(1))
        dmin[k] = np.inf
        if dmin.min() < 1e-6: continue
        old = P[k].copy()
        # incremental count change
        dold = np.sqrt(((P-old)**2).sum(1)); dold[k]=np.inf
        co = int((np.abs(dold-1)<TOL).sum())
        P[k] = cand
        dnew = np.sqrt(((P-cand)**2).sum(1)); dnew[k]=np.inf
        cn = int((np.abs(dnew-1)<TOL).sum())
        delta = cn - co
        if delta >= 0 or rng.random() < np.exp(delta/T):
            cur += delta
            if cur > best:
                best = cur; bestP = P.copy()
        else:
            P[k] = old
    return best, bestP

results=[]
for seed in range(6):
    best, P = search(n=40, steps=120_000, seed=seed)
    print(f"seed {seed}: best unit count = {best}")
    results.append((best,P))
best, P = max(results, key=lambda r: r[0])
np.save('/home/claude/search40.npy', P)
print(f"\nBEST = {best}  (triangular 40-patch benchmark = 96)")

# structure analysis of the winner
D = np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(-1))
edges = [(i,j) for i in range(40) for j in range(i+1,40) if abs(D[i,j]-1)<TOL]
angs=[]
for (i,j) in edges:
    v=P[j]-P[i]; angs.append(np.degrees(np.arctan2(v[1],v[0]))%180)
angs=np.array(sorted(angs))
# cluster within 0.5 deg
clusters=[]; cur=[angs[0]]
for a in angs[1:]:
    if a-cur[-1]<0.5: cur.append(a)
    else: clusters.append(cur); cur=[a]
clusters.append(cur)
summ=sorted([(len(c), float(np.mean(c))) for c in clusters], reverse=True)
print("\nedge directions (count @ angle):")
for c,a in summ[:8]: print(f"  {c:>3} @ {a:7.2f}")
# differences between top angles - lattice signature = multiples of 60
if len(summ)>=3:
    top=[a for _,a in summ[:4]]
    print("pairwise angle gaps of top directions:", 
          [round((top[i]-top[j])%180,1) for i in range(len(top)) for j in range(i+1,len(top))])
