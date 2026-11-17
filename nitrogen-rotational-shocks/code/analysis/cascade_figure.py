"""Precursor of the nitrogen shock: DSMC against the linear recoil cascade (no fitted parameter).
Input: json written by dsmc/n2_cascade_compare.py. Usage: python cascade_figure.py <json> <output folder> <label> [...]"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = json.load(open(sys.argv[1])); outd = sys.argv[2]; labs = sys.argv[3:]; os.makedirs(outd, exist_ok=True)
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "axes.titlesize": 15, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14, "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.8})
COL = ["#1F4E79", "#2E8B57", "#C0392B", "#8E6BBE", "#E08A2B", "#555555"]
fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
print("case | decay rate d ln(beta)/d(AN): DSMC, cascade, ratio | decay length l1: DSMC, cascade")
for c, lab in zip(COL, labs):
    D, C = R[lab].get("dsmc"), R[lab].get("cascade")
    bs = sorted((float(k) for k in (D or C)["lev"]), reverse=True)
    if D:
        g = lambda q: np.array([D["lev"][str(b)][q] for b in bs])
        for a, q in zip(ax, ("trot", "ttr", "tag")):
            a.errorbar(bs, g(q), yerr=2 * g(q + "_se"), fmt="o", color=c, ms=4.5, mfc="none", capsize=2, lw=0.9)
    if C:
        g = lambda q: np.array([C["lev"][str(b)][q] for b in bs])
        for a, q in zip(ax, ("trot", "ttr", "tag")):
            a.plot(bs, g(q), "-", color=c, lw=1.4)
    if D and C:
        print(f"{lab:8s} | {D['slope']:.4f}+-{D['slope_se']:.4f}  {C['slope']:.4f}+-{C['slope_se']:.4f}  {C['slope'] / D['slope']:.3f} | {D['length']:.3f} {C['length']:.3f}")
for a, yl in zip(ax, (r"$T_{rot}^{-}/T_2$", r"$T_{tr}^{-}/T_2$", "cold-rotation share")):
    a.set_xscale("log"); a.set_xlim(2e-2, 6e-5); a.set_xlabel(r"$\beta$"); a.set_ylabel(yl)
ax[0].set_ylim(0.4, 0.8); ax[1].set_ylim(1.2, 1.8); ax[2].set_ylim(0.4, 0.8)
for a, s in zip(ax, "abc"):
    a.set_title(f"({s})", loc="left", fontsize=15)
hh = [Line2D([], [], color=c, lw=1.4) for c, _ in zip(COL, labs)] + [Line2D([], [], color="k", marker="o", mfc="none", ls=""), Line2D([], [], color="k", lw=1.4)]
fig.legend(hh, [l.replace("M", "Mach ").replace("_z", r", $Z_{rot}=$") for l in labs] + ["DSMC", "linear cascade"], loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.92))
name = "n2_precursor_cascade" + ("_zrot" if any("_z" in l for l in labs) else "")
fig.savefig(os.path.join(outd, name + ".png"), dpi=200); fig.savefig(os.path.join(outd, name + ".pdf"))
