"""Hot rotational group measured by Robben & Talbot (1966) against the fraction of molecules whose rotational energy
has been exchanged in DSMC, for three exchange rules, with the fraction of molecules that have had at least one
collision (independent of the rule) as the upper line.

Usage: python hot_fraction_rules_figure.py <out prefix> <class npz> [<class npz> ...]"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robben_talbot_1966 as rt

D = {}
for f in sys.argv[2:]:
    d = np.load(f); D.update({k: d[k] for k in d.files})
plt.rcParams.update({"font.size": 15, "axes.labelsize": 18, "axes.titlesize": 16, "legend.fontsize": 14, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.8})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 7.0), sharey=True, constrained_layout=True)
RULES = (("", "--", "#C0392B", 2.2, r"Larsen–Borgnakke, $Z_{rot}=5$"), ("zi10", "-", "#1F4E79", 2.6, r"energy-dependent probability, Boyd's $Z_\infty=18$"),
         ("C", "-.", "#2E8B57", 2.4, "state-dependent rule fitted to the trajectories"))


def curves(lab):
    n, cls, tag, cnt = (D[f"{lab}_{k}"] for k in ("n", "cls", "tag", "cnt")); ns = (n - 1) / (float(D[f"{lab}_n2"]) - 1); c = cls[:5] / cls[:5].sum(0)
    m = (ns > 0.005) & (ns < 0.99) & (np.arange(len(ns)) < np.argmax(ns > 0.99) + 1 if (ns > 0.99).any() else True); k = np.argsort(ns[m])
    return ns[m][k], (1 - c[0])[m][k], (1 - tag / cnt)[m][k]


for a, (mach, key, title) in zip(ax, (("M7", "7", "(a) Mach 7.0"), ("M12.9", "12.9", "(b) Mach 12.9"))):
    X, coll, _ = curves(mach + "zi10"); a.plot(X, coll, ":", color="k", lw=2.2, label="at least one collision")
    for suf, ls, col, lw, lab in RULES:
        X, _, ex = curves(mach + suf); a.plot(X, ex, ls, color=col, lw=lw, label=lab + ": exchanged")
    e = rt.reduce(rt.CASES[key]["rows"]); rho1 = e[0, 2]; rho2 = e[-3:, 2].mean(); en = (e[:, 2] - rho1) / (rho2 - rho1); s = (en > 0.02) & (en < 0.97)
    a.plot(en[s], e[s, 4], "o", color="#E08A2B", ms=11, mec="k", mew=1.0, label="Robben & Talbot (1966): hot group", zorder=5)
    a.set_xlim(0, 1); a.set_ylim(0, 1.03); a.set_xlabel(r"normalised density $n^*$"); a.set_title(title, loc="left")
ax[0].set_ylabel("fraction of the molecules")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc="outside lower center", ncol=2, fontsize=14.5)
fig.savefig(sys.argv[1] + ".png", dpi=200); fig.savefig(sys.argv[1] + ".pdf"); print("written", sys.argv[1] + ".png")
