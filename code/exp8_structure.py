import numpy as np
P=np.load('/home/claude/clean40.npy'); n=len(P)
D=np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(-1))
E=[(i,j) for i in range(n) for j in range(i+1,n) if abs(D[i,j]-1)<1e-9]
deg=np.zeros(n,int)
for i,j in E: deg[i]+=1; deg[j]+=1
print("degrees:", sorted(deg.tolist(), reverse=True))
print("extent:", P[:,0].min(), P[:,0].max(), P[:,1].min(), P[:,1].max())
# edge directions
angs=sorted([np.degrees(np.arctan2(*(P[j]-P[i])[::-1]))%180 for i,j in E])
cl=[]; cur=[angs[0]]
for a in angs[1:]:
    if a-cur[-1]<0.5: cur.append(a)
    else: cl.append(cur); cur=[a]
cl.append(cur)
summ=sorted([(len(c),round(float(np.mean(c)),2)) for c in cl],reverse=True)
print(f"{len(summ)} distinct directions; top:", summ[:8])
# distinct distances overall
iu=np.triu_indices(n,1)
dvals=np.round(D[iu],9)
u,c=np.unique(dvals,return_counts=True)
print(f"distinct distances: {len(u)} among 780 pairs; top multiplicities:", sorted(c.tolist(),reverse=True)[:8])
# are points on few concentric circles? distances from centroid
cen=P.mean(0)
rad=np.round(np.sqrt(((P-cen)**2).sum(1)),6)
ur,cr=np.unique(rad,return_counts=True)
print(f"distinct radii from centroid: {len(ur)}; multiplicities:", sorted(cr.tolist(),reverse=True)[:8])
