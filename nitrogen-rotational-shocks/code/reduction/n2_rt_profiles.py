"""Nitrogen DSMC shocks in the units of Robben & Talbot (1966): L* = mu(T*)/(rho1 u1), T* = 2 T0/(gamma+1).
For the model gas (mu ~ T^omega, lengths in l1 = mu1 sqrt(R T1)/p1): L*/l1 = (T*/T1)^omega / (M sqrt(gamma)).
Prints the maximum-slope thicknesses of the density and of the rotational temperature and the distance between the
two profiles at their midpoints, next to the measured values.
Usage (in the DSMC folder): python n2_rt_profiles.py <runs folder> <case> [<case> ...]
       <case> = Mach number (files M<M>_w0.74_s*.txt) or a file prefix such as M8_z2 (files M8_z2_s*.txt)."""
import sys, glob, math
import numpy as np
from n2_check import read

EXP = {"7": dict(n=0.158, tr=0.115, lag=1.9), "12.9": dict(n=0.167, tr=0.107, lag=2.54)}   # L*/delta_m, and lag in L*
for Ms in sys.argv[2:]:
    pre = Ms if Ms.startswith("M") else f"M{Ms}_w0.74"
    runs = [read(s[:-4]) for s in sorted(glob.glob(f"{sys.argv[1]}/{pre}_s*.txt"))]
    h = runs[0][0]; M, gam, nc, dx, om = h["M"], h["gamma"], int(h["nc"]), h["dx"], h["omega"]
    n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); T2 = (2 * gam * M * M - (gam - 1)) / (gam + 1) / n2
    Lstar = ((1 + 0.5 * (gam - 1) * M * M) * 2 / (gam + 1)) ** om / (M * math.sqrt(gam))

    def prof(sel):
        mom = sum(runs[k][2] for k in sel); rot = sum(runs[k][4] for k in sel); ns = sum(runs[k][3] for k in sel)
        x = h["xmin"] + (np.arange(nc) + 0.5) * dx
        N = mom[:, 0]; n = (N * h["FN"] / (ns * dx) - 1) / (n2 - 1); tr = (rot[0] / N - 1) / (T2 - 1)
        k = np.ones(5) / 5; ns_, trs = np.convolve(n, k, "same"), np.convolve(tr, k, "same")     # 0.5 l1 running mean
        m = (x > -25) & (x < 15)
        dn = 1 / np.max(np.gradient(ns_, x)[m]); dt = 1 / np.max(np.gradient(trs, x)[m])
        xm = lambda f: float(np.interp(0.5, f[m], x[m]))
        return dn, dt, xm(n) - xm(tr)

    a = np.array([prof([k]) for k in range(len(runs))]); c = prof(range(len(runs)))
    se = a.std(0, ddof=1) / math.sqrt(len(runs)) if len(runs) > 1 else np.zeros(3)
    e = EXP.get(Ms)
    print(f"{pre} (Mach {M:g}, zrot {h['zrot']:g}): L* = {Lstar:.4f} l1; {len(runs)} seeds")
    print(f"   density thickness {c[0]:.3f}+-{se[0]:.3f} l1 = {c[0] / Lstar:.2f} L*   L*/delta = {Lstar / c[0]:.4f}" + (f"   (measured {e['n']})" if e else ""))
    print(f"   rotational-temperature thickness {c[1]:.3f}+-{se[1]:.3f} l1   L*/delta = {Lstar / c[1]:.4f}" + (f"   (measured {e['tr']})" if e else ""))
    print(f"   distance between midpoints {c[2]:.3f}+-{se[2]:.3f} l1 = {c[2] / Lstar:.2f} L*" + (f"   (measured {e['lag']})" if e else ""))
