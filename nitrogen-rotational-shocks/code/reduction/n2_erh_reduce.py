"""Seeds combined: profiles and rotational-energy histograms of the diatomic DSMC shocks (dsmc_shock_vdf10), cut to
the shock region, in one compact file.
Usage (in the DSMC folder): python n2_erh_reduce.py <out.npz> <label>=<glob of .txt files> [<label>=<glob> ...]
Example: python n2_erh_reduce.py runs_n2/erh_tag.npz M7='runs_n2/tag/M7_w0.74_s*.txt' M12.9='runs_n2/tag/M12.9_w0.74_s*.txt'"""
import sys, glob
import numpy as np
from n2_check import read

out = {}; SMIN = 215000
for spec in sys.argv[2:]:
    if spec.startswith("min="):                         # number of steps a finished run must have (default 215000)
        SMIN = int(spec[4:]); continue
    lab, pat = spec.split("=", 1); stems = sorted(s[:-4] for s in glob.glob(pat))
    stems = [s for s in stems if f"steps_done {SMIN}" in open(s + ".txt").read()]
    runs = [read(s) for s in stems]
    h = runs[0][0]; M, gam, nc, dx = h["M"], h["gamma"], int(h["nc"]), h["dx"]
    n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); T2 = (2 * gam * M * M - (gam - 1)) / (gam + 1) / n2
    n2 = h.get("n2res", n2)                                # version 19: reservoir of the measured stream
    x = h["xmin"] + (np.arange(nc) + 0.5) * dx
    mom = sum(r[2] for r in runs); rot = sum(r[4] for r in runs); fly = sum(r[5] for r in runs); ns = sum(r[3] for r in runs)
    erh = sum(np.fromfile(s + ".erh.bin").reshape(-1, nc, 160) for s in stems)   # all, vx<0, generations 0..3 (and, from version 12, vx<0 in classes 1, 4, 3)
    import os
    if os.path.exists(stems[0] + ".erl.bin"):                                   # fine bins (0.005 T2), same layout
        erl = sum(np.fromfile(s + ".erl.bin").reshape(-1, nc, 160) for s in stems)
    else: erl = None
    N = mom[:, 0]; n = N * h["FN"] / (ns * dx); u = mom[:, 1] / N
    Tpar = mom[:, 2] / N - u ** 2; Tperp = 0.5 * mom[:, 5] / N
    ic = int(np.argmin(np.abs((n - 1) / (n2 - 1) - 0.5))); m = (x - x[ic] > -40) & (x - x[ic] < 15)
    d = dict(x=x[m] - x[ic], n=n[m], Ttr=((Tpar + 2 * Tperp) / 3)[m], Trot=(rot[0] / N)[m], N=N[m], beta=(fly[0] / N)[m],
             nup=rot[1:5].sum(0)[m], erot_up=rot[5:9].sum(0)[m], tag_n=rot[19:21][:, m], tag_e=rot[21:23][:, m], tag_k=rot[23:25][:, m],
             gen_n=rot[1:5][:, m], gen_e=rot[5:9][:, m], fly_n=rot[13][m], erh=erh[:, m, :])
    if erl is not None: d["erl"] = erl[:, m, :]
    if rot.shape[0] >= 35: d["cls_n"] = rot[25:30][:, m]; d["cls_e"] = rot[30:35][:, m]
    for k, v in d.items():
        out[f"{lab}_{k}"] = v
    out[f"{lab}_T2"] = T2; out[f"{lab}_n2"] = n2; out[f"{lab}_M"] = M; out[f"{lab}_zrot"] = h["zrot"]; out[f"{lab}_seeds"] = len(stems)
    print(lab, "seeds", len(stems), "T2", round(T2, 3), "cells kept", int(m.sum()), "histogram total / samples", erh[0].sum() / N.sum())
np.savez_compressed(sys.argv[1], **out)
