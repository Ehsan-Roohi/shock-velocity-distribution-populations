"""Hot rotational group measured by Robben & Talbot (1966) against the collision history in DSMC, at equal normalised
density. Lines: molecules with at least one collision (independent of the exchange rule); the same weighted with the
probability q that one collision excites a cold molecule (classical trajectories: q = 0.83, band 0.75-0.90); molecules
whose rotational energy has been exchanged under the Larsen-Borgnakke rule with constant and with energy-dependent
probability.   Usage: python hot_fraction_figure.py <class npz> <out prefix> [q=0.83]"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robben_talbot_1966 as rt

d = np.load(sys.argv[1]); q0 = float(sys.argv[3][2:]) if len(sys.argv) > 3 else 0.83
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "axes.titlesize": 15, "legend.fontsize": 13.5, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.8})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.9), sharey=True, constrained_layout=True)


def curves(lab):
    n, cls, tag, cnt = (d[f"{lab}_{k}"] for k in ("n", "cls", "tag", "cnt")); ns = (n - 1) / (float(d[f"{lab}_n2"]) - 1); c = cls[:5] / cls[:5].sum(0)
    hot = lambda q: c[1] * q + (c[2] + c[4]) * (1 - (1 - q) ** 2) + c[3]
    return ns, 1 - c[0], hot, 1 - tag / cnt


for a, (lab, key, title) in zip(ax, (("M7", "7", "(a) Mach 7.0"), ("M12.9", "12.9", "(b) Mach 12.9"))):
    ns, coll, hot, exch = curves(lab); m = (ns > 0.005) & (ns < 0.99) & (np.arange(len(ns)) < np.argmax(ns > 0.99) + 1 if (ns > 0.99).any() else True)
    k = np.argsort(ns[m]); X = ns[m][k]
    a.fill_between(X, hot(0.75)[m][k], hot(0.90)[m][k], color="#1F4E79", alpha=0.18, lw=0)
    a.plot(X, coll[m][k], "-", color="k", lw=2.0, label="DSMC: at least one collision")
    a.plot(X, hot(q0)[m][k], "-", color="#1F4E79", lw=2.4, label="one collision excites with probability 0.83 (trajectories)")
    a.plot(X, exch[m][k], "--", color="#C0392B", lw=2.2, label=r"Larsen–Borgnakke, $Z_{rot}=5$: exchanged")
    ns2, _, _, ex2 = curves(lab + "pk14"); m2 = (ns2 > 0.005) & (ns2 < 0.99); k2 = np.argsort(ns2[m2])
    a.plot(ns2[m2][k2], ex2[m2][k2], ":", color="#C0392B", lw=2.4, label=r"energy-dependent probability, $Z_\infty=14$: exchanged")
    rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); en = (ex[:, 2] - rho1) / (rho2 - rho1); s = (en > 0.02) & (en < 0.97)
    a.plot(en[s], ex[s, 4], "o", color="#E08A2B", ms=10, mec="k", mew=1.0, label="Robben & Talbot (1966): hot group", zorder=5)
    a.set_xlim(0, 1); a.set_ylim(0, 1.03); a.set_xlabel(r"normalised density $n^*$"); a.set_title(title, loc="left")
    print(title, "measured:", np.round(en[s], 3).tolist(), np.round(ex[s, 4], 3).tolist())
    for e_, f_ in zip(en[s], ex[s, 4]):
        print(f"   n* {e_:.3f}: measured {f_:.3f}; collided {np.interp(e_, X, coll[m][k]):.3f}; with q = {q0}: {np.interp(e_, X, hot(q0)[m][k]):.3f}; LB {np.interp(e_, X, exch[m][k]):.3f}; energy-dependent {np.interp(e_, ns2[m2][k2], ex2[m2][k2]):.3f}")
ax[0].set_ylabel("fraction of the molecules")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc="outside lower center", ncol=2, fontsize=14)
fig.savefig(sys.argv[2] + ".png", dpi=200); fig.savefig(sys.argv[2] + ".pdf"); print("written", sys.argv[2] + ".png")
