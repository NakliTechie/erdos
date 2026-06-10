import numpy as np
# 18 unit vectors of Moser lattice from the paper's Fig 2 coefficient list
w1 = np.exp(1j*np.pi/3)
w3 = np.exp(1j*np.arccos(5/6))
basis = np.array([1, w1, w3, w1*w3])
coeffs = [(-2,1,2,-1),(-1,-1,1,1),(-1,0,0,0),(-1,1,0,0),(-1,2,1,-2),(0,-1,0,0),
          (0,0,-1,0),(0,0,-1,1),(0,0,0,-1),(0,0,0,1),(0,0,1,-1),(0,0,1,0),
          (0,1,0,0),(1,-2,-1,2),(1,-1,0,0),(1,0,0,0),(1,1,-1,-1),(2,-1,-2,1)]
uv = np.array([ (np.array(c)*basis).sum() for c in coeffs ])
print("unit-length check:", np.allclose(np.abs(uv),1))
ml_dirs = sorted(set(np.round(np.degrees(np.angle(uv))%180, 3)))
print(f"Moser lattice: {len(ml_dirs)} undirected directions:")
print([round(d,2) for d in ml_dirs])

# our configuration's directions
P = np.load('/home/claude/clean40.npy'); n=len(P)
D = np.sqrt(((P[:,None,:]-P[None,:,:])**2).sum(-1))
E = [(i,j) for i in range(n) for j in range(i+1,n) if abs(D[i,j]-1)<1e-9]
ours = np.array(sorted(set(round(float(np.degrees(np.arctan2(*(P[j]-P[i])[::-1]))%180),2) for i,j in E)))
# merge near-duplicates
merged=[ours[0]]
for a in ours[1:]:
    if a-merged[-1]>0.3: merged.append(a)
print(f"\nours: {len(merged)} directions:", [round(a,2) for a in merged])
# best rotation alignment: try aligning each of ours to each ML dir, count matches
best=None
mld = np.array(ml_dirs)
for o in merged:
    for m in mld:
        rot = (o-m)%180
        ours_rot = (np.array(merged)-rot)%180
        # distance to nearest ML direction
        dmat = np.min(np.abs(ours_rot[:,None]-mld[None,:]) % 180, axis=1)
        dmat = np.minimum(dmat, 180-dmat)
        score = int((dmat<0.15).sum())
        if best is None or score>best[0]: best=(score,rot)
print(f"\nbest alignment: {best[0]}/{len(merged)} of our directions match Moser-lattice directions (rotation {best[1]:.2f} deg)")
