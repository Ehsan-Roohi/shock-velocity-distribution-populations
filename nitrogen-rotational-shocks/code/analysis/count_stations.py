"""Rotational energy of the molecules by exact number of collisions (dsmc_shock_vdf17), at given normalised
densities: share of the molecules, mean rotational energy, shares above 0.1, 0.3 and 1 kT2, and the temperature of
the exponential through the shares above 1 and 3 kT2, for the molecules with 0, 1, 2 and 3 or more collisions, for
the scattered molecules last struck by an unscattered one, and for all.

Usage: python count_stations.py <erh npz of n2_erh_reduce.py> <label> [<label> ...] [ns=0.02,0.07,0.14,0.24,0.39,0.58,0.76]"""
import sys, math
import numpy as np

labs = [a for a in sys.argv[2:] if "=" not in a]; opt = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
NS = [float(v) for v in opt.get("ns", "0.02,0.07,0.14,0.24,0.39,0.58,0.76").split(",")]; D = np.load(sys.argv[1])
ROWS = (("all", 0), ("0 collisions", 2), ("1 collision", 9), ("2 collisions", 10), ("3 or more", 11), ("last partner unscattered", 12), ("generation 1", 3))
for lab in labs:
    n, erh, erl = (D[f"{lab}_{k}"] for k in ("n", "erh", "erl")); ns = (n - 1) / (float(D[f"{lab}_n2"]) - 1); ec = (np.arange(160) + 0.5) * 0.05
    print(f"\n{lab}: rows = {erh.shape[0]}")
    for x in NS:
        i = int(np.argmin(np.abs(ns - x))); s = slice(max(i - 2, 0), i + 3); tot = erh[0, s].sum(); print(f"  n* = {x:.2f} (cell {ns[i]:.3f}):")
        for nm, r in ROWS:
            if r >= erh.shape[0]: continue
            h = erh[r, s].sum(0); hl = erl[r, s].sum(0); m = h.sum()
            if m < 50: continue
            c = h[::-1].cumsum()[::-1] / m; a1, a3 = c[20], c[60]; sl = 2 / math.log(a1 / a3) if a3 > 0 else float("nan")
            print(f"     {nm:25s} share {m / tot:.3f}  mean e {np.sum(h * ec) / m:.3f}  above 0.1: {1 - hl[:20].sum() / m:.3f}  0.3: {1 - hl[:60].sum() / m:.3f}  1: {a1:.3f}  2: {c[40]:.4f}  3: {a3:.4f}  T(1-3) {sl:.3f}")
