"""Measured rotational level populations of nitrogen inside the shock (Robben & Talbot 1966, Tables I and II) against
the rotational-energy distribution of the Larsen-Borgnakke DSMC at the same normalised density, with the DSMC
distribution split by collision generation.

Both are shown as probability density per unit e_rot/kT2. Measurement: p = 10^{D_K} (T2/theta)/rho at
E_K = theta K(K+1), rho from their Eq. (9), and the straight line of temperature T' beyond the last tabulated level.

Usage: python rt_levels.py <erh npz> <output folder> <label>:<experiment key> [...]"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robben_talbot_1966 as rt

d = np.load(sys.argv[1]); outd = sys.argv[2]; os.makedirs(outd, exist_ok=True)
CG = ["#2C7FB8", "#8E6BBE", "#E08A2B", "#D7422B"]
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "axes.titlesize": 15, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14, "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.8})
TH = rt.THETA
for spec in sys.argv[3:]:
    lab, key = spec.split(":")
    x, n, erh = (d[f"{lab}_{k}"] for k in ("x", "n", "erh")); n2 = float(d[f"{lab}_n2"]); T2d = float(d[f"{lab}_T2"])
    nb = erh.shape[-1]; DE = 8.0 / nb; E = (np.arange(nb) + 0.5) * DE; ns = (n - 1) / (n2 - 1)
    rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); Te2 = ex[-1, 1]
    en = (ex[:, 2] - rho1) / (rho2 - rho1)
    pick = [j for j in range(len(rows)) if 0.04 < en[j] < 0.9 and len(rows[j][2]) > 3][:6]
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.0), sharex=True, sharey=True)
    print(f"{lab}: measured stations (turns) {[rows[j][0] for j in pick]} at normalised density {np.round(en[pick], 3).tolist()}")
    for a, j in zip(ax.ravel(), pick):
        X, Tp, dk, last = rows[j]; K = np.arange(len(dk)); kp = len(dk); rho = ex[j, 2]
        EK = TH * K * (K + 1) / Te2; pK = np.array(dk) * (Te2 / TH) / rho
        Ekp = TH * (kp - 0.5) * (kp + 0.5) / Te2; el = np.linspace(Ekp, 4.0, 40); pl = last * (Te2 / TH) / rho * np.exp(-(el - Ekp) * Te2 / Tp)
        i = int(np.argmin(np.abs(ns - en[j]))); s = slice(max(i - 2, 0), i + 3); f = erh[:, s].sum(1); tot = f[0].sum(); m = E < 4.0
        a.semilogy(E[m], f[0][m] / tot / DE, "k-", lw=1.7)
        for g in range(4):
            a.semilogy(E[m], np.maximum(f[2 + g][m], 1e-30) / tot / DE, "-", color=CG[g], lw=1.1)
        a.semilogy(EK, pK, "o", color="#C0392B", ms=5, mfc="none", mew=1.2); a.semilogy(el, pl, "--", color="#C0392B", lw=1.3)
        a.set_ylim(2e-4, 60); a.set_xlim(0, 4)
        a.text(0.97, 0.92, rf"$n^*={en[j]:.2f}$", transform=a.transAxes, ha="right")
        # ratio of the DSMC density to the measured one at the tabulated levels above 0.3 T2
        ok = EK > 0.3
        if ok.any():
            r = np.interp(EK[ok], E, f[0] / tot / DE) / pK[ok]
            print(f"   station {X}: n*={en[j]:.3f} (DSMC cell n*={ns[i]:.3f}); DSMC/measured at levels above 0.3 T2: " + " ".join(f"{v:.2f}" for v in r)
                  + f" ; at the start of the T' line (E={Ekp:.2f} T2): {np.interp(Ekp, E, f[0] / tot / DE) / (last * (Te2 / TH) / rho):.2f}")
    for a in ax[1]:
        a.set_xlabel(r"$e_{rot}/kT_2$")
    for a in ax[:, 0]:
        a.set_ylabel("probability density")
    hh = [Line2D([], [], color="#C0392B", marker="o", mfc="none", ls="--"), Line2D([], [], color="k", lw=1.7)] + [Line2D([], [], color=c, lw=1.3) for c in CG]
    fig.legend(hh, ["Robben & Talbot (1966): levels and $T'$ line", "DSMC, all molecules", "generation 0", "generation 1", "generation 2", "generation 3+"],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(outd, f"n2_levels_vs_experiment_{lab}.png"), dpi=200); fig.savefig(os.path.join(outd, f"n2_levels_vs_experiment_{lab}.pdf"))
