"""Seeds combined: counts, rotational energy and number of tagged molecules in every (cell, streamwise-velocity bin)
of the diatomic DSMC shocks (versions 11 and later), cut to the shock region, for maps in phase space.
Usage (in the DSMC folder): python n2_map_reduce.py <out.npz> <label>=<glob of .txt files> [...]"""
import sys, glob
import numpy as np
from n2_check import read

out = {}
for spec in sys.argv[2:]:
    lab, pat = spec.split("=", 1); stems = sorted(s[:-4] for s in glob.glob(pat))
    stems = [s for s in stems if "steps_done 215000" in open(s + ".txt").read()]
    cnt = None; erv = None; mom = None; ns = 0
    for s in stems:
        h, c, m, n_, rot, fly = read(s); nc, nb = int(h["nc"]), int(h["nb"])
        e = np.fromfile(s + ".erv.bin").reshape(2, nc, nb)
        cnt = c if cnt is None else cnt + c; erv = e if erv is None else erv + e; mom = m if mom is None else mom + m; ns = ns + n_
    M, gam, dx = h["M"], h["gamma"], h["dx"]; n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); T2 = (2 * gam * M * M - (gam - 1)) / (gam + 1) / n2
    x = h["xmin"] + (np.arange(nc) + 0.5) * dx; n = mom[:, 0] * h["FN"] / (ns * dx)
    ic = int(np.argmin(np.abs((n - 1) / (n2 - 1) - 0.5))); m = (x - x[ic] > -32) & (x - x[ic] < 14)
    v = h["vlo"] + (np.arange(nb) + 0.5) * h["dv"]
    out.update({f"{lab}_x": x[m] - x[ic], f"{lab}_v": v, f"{lab}_cnt": cnt[m].astype(np.float32), f"{lab}_er": erv[0][m].astype(np.float32), f"{lab}_tag": erv[1][m].astype(np.float32),
                f"{lab}_n": n[m], f"{lab}_T2": T2, f"{lab}_n2": n2, f"{lab}_M": M, f"{lab}_seeds": len(stems), f"{lab}_scale": h["FN"] / (dx * h["dv"] * float(ns))})
    print(lab, "seeds", len(stems), "cells", int(m.sum()), "velocity bins", nb, "total er / total:", erv[0].sum() / cnt.sum() / T2)
np.savez_compressed(sys.argv[1], **out)
