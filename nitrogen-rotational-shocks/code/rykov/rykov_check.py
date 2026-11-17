"""Check of a Rykov-model shock run: upstream-moving fraction beta(x), rotational and translational temperature of
the upstream-moving molecules, and the place of the negative part of the reduced distribution.
Usage: python rykov_check.py <run folder> ..."""
import sys, os, numpy as np

for d in sys.argv[1:]:
    v = np.loadtxt(os.path.join(d, "vnodes.txt")).ravel()
    nv = v.size
    a = np.fromfile(os.path.join(d, "reduced.bin")).reshape(-1, nv, 3)
    nx = a.shape[0]
    prof = np.genfromtxt(os.path.join(d, "profiles.csv"), delimiter=",", names=True)
    x = prof[prof.dtype.names[0]]
    F, H, R = a[:, :, 0], a[:, :, 1], a[:, :, 2]          # sums of wF, wH, w r^2 F over the radial nodes
    n = F.sum(1)
    n1, n2 = n[:5].mean(), n[-5:].mean()
    ns = (n - n1) / (n2 - n1)
    ic = int(np.argmin(abs(ns - 0.5))); xc = x[ic]
    up = v < 0
    nup = F[:, up].sum(1)
    beta = nup / n
    neg = np.where(F < 0, -F, 0.0)
    print(f"{d}: nx={nx} nv={nv} n2/n1={n2/n1:.4f} centre x={xc:.2f}")
    print(f"  negative part of F: total {neg.sum():.3e} of {abs(F).sum():.3e}; in v<0: {neg[:, up].sum():.3e}; "
          f"upstream of centre: {neg[:ic].sum():.3e}; upstream and v<0: {neg[:ic][:, up].sum():.3e}")
    T2 = None
    for nm in prof.dtype.names:
        if nm.lower() in ("t", "temperature", "t_total"):
            T2 = prof[nm][-5:].mean()
    if T2 is None:
        T2 = 1.0
    print(f"  columns: {prof.dtype.names}")
    for lev in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        idx = np.where(beta[:ic] >= lev)[0]
        if idx.size == 0 or idx[0] == 0:
            print(f"  beta={lev:g}: not reached"); continue
        i = idx[0]
        trot = H[i, up].sum() / nup[i]                     # two rotational degrees of freedom: H = e_rot F, e_rot = T
        tperp = 0.5 * R[i, up].sum() / nup[i]
        negl = neg[i, up].sum() / nup[i]
        print(f"  beta={lev:g}: x-xc={x[i]-xc:.2f}  Trot(up)/T2={trot/T2:.3f}  Tperp(up)/T2={tperp/T2:.3f}  "
              f"negative part / upstream-moving mass here: {negl:.2e}")
