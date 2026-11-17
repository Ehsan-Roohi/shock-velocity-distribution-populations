"""Hot rotational group of Robben & Talbot (1966) against the collision history in DSMC at the same normalised density.

Measured: the fraction of molecules in the group with temperature T' (their two-group description), and the part of
the molecules above a rotational energy E0 (from the level populations and the T' line).
DSMC (VHS, independent of the exchange rule): the fraction of molecules that have had at least one collision with a
molecule that is not part of the unscattered stream (generation >= 1), and by generation.
DSMC with the Larsen-Borgnakke rule: the fraction of molecules whose rotational energy has been exchanged, and the
part of the molecules above E0.

Usage: python hot_fraction.py <erh npz> <map npz> <label>:<experiment key> [...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robben_talbot_1966 as rt

d = np.load(sys.argv[1]); mp = np.load(sys.argv[2]); TH = rt.THETA; E0 = 0.3
for spec in sys.argv[3:]:
    lab, key = spec.split(":")
    x, n, erh = (d[f"{lab}_{k}"] for k in ("x", "n", "erh")); n2 = float(d[f"{lab}_n2"]); ns = (n - 1) / (n2 - 1); nb = erh.shape[-1]; DE = 8.0 / nb; E = (np.arange(nb) + 0.5) * DE
    xm, cnt, tag, nm = (mp[f"{lab}_{k}"] for k in ("x", "cnt", "tag", "n")); nsm = (nm - 1) / (float(mp[f"{lab}_n2"]) - 1); exch = 1 - tag.sum(1) / cnt.sum(1)
    rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); Te2 = ex[-1, 1]; en = (ex[:, 2] - rho1) / (rho2 - rho1)
    print(f"{lab} (measured T2 = {Te2:.0f} K): columns = station, n*, measured hot-group fraction, measured part above {E0} T2 | DSMC collided at least once, (one, two, three or more) | "
          f"Larsen-Borgnakke: exchanged fraction, part above {E0} T2 | measured hot group / collided")
    for j, (X, Tp, dk, last) in enumerate(rows):
        if not (0.02 < en[j] < 0.95): continue
        K = np.arange(len(dk)); kp = len(dk); EK = TH * K * (K + 1) / Te2; NK = (2 * K + 1) * np.array(dk, float)
        Ekp = TH * (kp - 0.5) * (kp + 0.5) / Te2; hi = (Tp / TH) * last; rho = NK.sum() + hi
        above = (NK[EK > E0].sum() + (hi if Ekp > E0 else hi * np.exp(-(E0 - Ekp) * Te2 / Tp))) / rho
        i = int(np.argmin(np.abs(ns - en[j]))); s = slice(max(i - 2, 0), i + 3); f = erh[:, s].sum(1); tot = f[0].sum(); g = f[2:6].sum(1) / tot
        im = int(np.argmin(np.abs(nsm - en[j]))); ex_lb = exch[max(im - 2, 0):im + 3].mean(); ab_lb = f[0][E > E0].sum() / tot
        print(f"   {X:5.2f}  n* {en[j]:.3f} | measured {ex[j, 4]:.3f}  {above:.3f} | collided {1 - g[0]:.3f} ({g[1]:.3f} {g[2]:.3f} {g[3]:.3f}) | LB {ex_lb:.3f}  {ab_lb:.3f} | ratio {ex[j, 4] / (1 - g[0]):.2f}")
