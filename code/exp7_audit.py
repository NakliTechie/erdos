import numpy as np
P = np.load('/home/claude/clean40.npy')
n=len(P)
D = np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(-1))
res = np.abs(D-1.0)
iu = np.triu_indices(n,1)
r = res[iu]
print("edge residual histogram (|d-1| of claimed edges):")
for lo,hi in [(0,1e-15),(1e-15,1e-12),(1e-12,1e-10),(1e-10,1e-9)]:
    c = int(((r>=lo)&(r<hi)).sum())
    if c: print(f"  [{lo:.0e},{hi:.0e}): {c}")
E = [(i,j) for i in range(n) for j in range(i+1,n) if res[i,j]<1e-9]
print("claimed edges:", len(E))

# Gauss-Newton snap on ALL claimed edges, damped
Q = P.copy()
for it in range(8000):
    grad=np.zeros_like(Q); tot=0.0
    for (i,j) in E:
        v=Q[i]-Q[j]; d=np.linalg.norm(v); rr=d-1.0; tot+=rr*rr
        g=(rr/max(d,1e-9))*v; grad[i]+=g; grad[j]-=g
    Q -= 0.08*grad
    if tot<1e-28: break
print(f"after snap: total residual={tot:.3e} (realizable iff ~0)")
D2=np.sqrt(((Q[:,None,:]-Q[None,:,:])**2).sum(-1))
exact = sum(1 for (i,j) in E if abs(D2[i,j]-1)<1e-12)
print(f"edges exact at 1e-12 after snap: {exact}/{len(E)}")
# also: did snap move points much / merge points?
move = np.sqrt(((Q-P)**2).sum(1)).max()
Dx=D2.copy(); np.fill_diagonal(Dx,np.inf)
print(f"max point movement: {move:.2e}, min separation after snap: {Dx.min():.3f}")
