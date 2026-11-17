"""Rotational distribution inside the nitrogen shock: Larsen-Borgnakke DSMC, split by collision generation, against
the measurements of Robben & Talbot (1966). Comparisons are made against the normalised density, so the length scale of
the experiment does not enter.

Input: file written by dsmc/n2_erh_reduce.py (histograms of the rotational energy in bins of T2: all molecules,
upstream-moving molecules, collision generations 0, 1, 2, 3+).
T' = temperature of the straight high-energy part of the distribution, from the slope of log f(e) in a window of e.

Usage: python rt_compare.py <erh npz> <output folder> <label>:<experiment key> [...]     e.g. M7:7 M12.9:12.9 M8:-
       ("-" = no measurement for this case)"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robben_talbot_1966 as rt

d = np.load(sys.argv[1]); outd = sys.argv[2]; os.makedirs(outd, exist_ok=True)
pairs = [a.split(":") for a in sys.argv[3:]]
WIN = {"0.8-2.4": (0.8, 2.4), "1.6-3.2": (1.6, 3.2)}      # in T2; the measured lines start at 0.6-1.6 T2
CG = ["#2C7FB8", "#8E6BBE", "#E08A2B", "#D7422B"]         # generations 0, 1, 2, 3+
GN = ["generation 0 (unscattered stream)", "generation 1", "generation 2", r"generation 3+"]
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "axes.titlesize": 15, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14, "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.8})


def tail(E, f, lo, hi, nmin=3000):
    m = (E >= lo) & (E <= hi) & (f > 0)
    if m.sum() < 4 or f[m].sum() < nmin:
        return np.nan
    return -1 / np.polyfit(E[m], np.log(f[m]), 1, w=np.sqrt(f[m]))[0]


def case(lab, block=5):
    x, n, Trot, N, erh = (d[f"{lab}_{k}"] for k in ("x", "n", "Trot", "N", "erh"))
    T2, n2 = float(d[f"{lab}_T2"]), float(d[f"{lab}_n2"]); nb = erh.shape[-1]; DE = 8.0 / nb; E = (np.arange(nb) + 0.5) * DE
    ng = erh.shape[0] - 2
    rows = []
    for i in range(0, len(x) - block + 1, block):
        s = slice(i, i + block); f = erh[:, s].sum(1); tot = f[0].sum()
        if tot < 5000:
            continue
        r = dict(x=x[s].mean(), ns=(n[s].mean() - 1) / (n2 - 1), TR=(Trot[s] * N[s]).sum() / N[s].sum() / T2, f=f, E=E)
        for k, (lo, hi) in WIN.items():
            r["Tp_" + k] = tail(E, f[0], lo, hi); r["Tpu_" + k] = tail(E, f[1], lo, hi)
        r["up_mean"] = (E * f[1]).sum() / max(f[1].sum(), 1)
        if ng > 0:
            r["share"] = f[2:].sum(1) / tot
            r["Tg"] = np.array([(E * f[2 + g]).sum() / max(f[2 + g].sum(), 1) for g in range(ng)])
            r["Tpg"] = np.array([tail(E, f[2 + g], *WIN["0.8-2.4"]) for g in range(ng)])
        rows.append(r)
    return rows, T2, ng


def col(rows, k):
    return np.array([r[k] for r in rows])


# ---------- figure 1: comparison with the measurements ----------
fig, ax = plt.subplots(len(pairs), 3, figsize=(14, 4.5 * len(pairs)), squeeze=False)
for irow, (lab, key) in enumerate(pairs):
    rows, T2, ng = case(lab); ns = col(rows, "ns"); T1 = 1 / T2; a = ax[irow]
    a[0].plot(ns, col(rows, "Tp_0.8-2.4"), "-", color="#1F4E79", lw=1.6)
    a[0].plot(ns, col(rows, "Tp_1.6-3.2"), ":", color="#1F4E79", lw=1.3)
    a[1].plot(ns, (col(rows, "TR") - T1) / (1 - T1), "-", color="#1F4E79", lw=1.6)
    a[1].plot([0, 1], [0, 1], "-", color="0.8", lw=0.8)
    if ng > 0:
        a[2].plot(ns, 1 - np.array([r["share"][0] for r in rows]), "-", color="#1F4E79", lw=1.6)
    a[2].plot(ns, (col(rows, "TR") - T1) / (1 - T1), "--", color="#1F4E79", lw=1.2)
    if key != "-":
        ex = rt.reduce(rt.CASES[key]["rows"]); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); Te2 = ex[-1, 1]
        en = (ex[:, 2] - rho1) / (rho2 - rho1); TR1 = ex[0, 3] / Te2; ok = ex[:, 1] / Te2 > 0.3
        a[0].plot(en[ok], ex[ok, 1] / Te2, "o", color="#C0392B", ms=5, mfc="none")
        a[1].plot(en, (ex[:, 3] / Te2 - TR1) / (1 - TR1), "o", color="#C0392B", ms=5, mfc="none")
        a[2].plot(en, (ex[:, 3] / Te2 - TR1) / (1 - TR1), "o", color="#C0392B", ms=5, mfc="none")
        print(f"{lab}: measured T'/T2 at normalised density {np.round(en[ok], 3).tolist()}: {np.round(ex[ok, 1] / Te2, 3).tolist()}")
    for q, yl in zip(a, (r"$T'/T_2$", r"$(T_R-T_1)/(T_2-T_1)$", "fraction")):
        q.set_xlabel(r"$(n-n_1)/(n_2-n_1)$"); q.set_ylabel(yl); q.set_xlim(0, 1)
    a[0].set_ylim(0.5, 1.2); a[1].set_ylim(0, 1.05); a[2].set_ylim(0, 1.05)
    a[0].text(0.97, 0.06, f"Mach {float(d[f'{lab}_M']):g}", transform=a[0].transAxes, ha="right")
    sel = [int(np.argmin(np.abs(ns - t))) for t in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)]
    print(f"{lab} (Mach {float(d[f'{lab}_M']):g}, zrot {float(d[f'{lab}_zrot']):g}, {int(d[f'{lab}_seeds'])} seeds)  n*: " + " ".join(f"{ns[i]:.3f}" for i in sel))
    for k in WIN:
        print(f"   T'/T2, window {k} T2:   " + " ".join(f"{rows[i]['Tp_' + k]:.3f}" for i in sel))
    print("   T_R/T2:                    " + " ".join(f"{rows[i]['TR']:.3f}" for i in sel))
    print("   upstream-moving: mean      " + " ".join(f"{rows[i]['up_mean']:.3f}" for i in sel) + "   T' (0.8-2.4): " + " ".join(f"{rows[i]['Tpu_0.8-2.4']:.3f}" for i in sel))
    if ng > 0:
        for g in range(ng):
            print(f"   generation {g}: share " + " ".join(f"{rows[i]['share'][g]:.3f}" for i in sel) + "   mean e_rot/T2 " + " ".join(f"{rows[i]['Tg'][g]:.3f}" for i in sel)
                  + "   T' " + " ".join(f"{rows[i]['Tpg'][g]:.3f}" for i in sel))
hh = [Line2D([], [], color="#1F4E79", lw=1.6), Line2D([], [], color="#1F4E79", lw=1.3, ls=":"), Line2D([], [], color="#1F4E79", lw=1.2, ls="--"),
      Line2D([], [], color="#C0392B", marker="o", mfc="none", ls="")]
fig.legend(hh, [r"DSMC ($T'$ from $0.8$–$2.4\,T_2$; collided fraction in the right panels)", r"DSMC, $T'$ from $1.6$–$3.2\,T_2$",
                r"DSMC, normalised $T_R$", "Robben & Talbot (1966)"], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 1 - 0.16 / len(pairs) - 0.02))
fig.savefig(os.path.join(outd, "n2_rotational_distribution_vs_experiment.png"), dpi=200); fig.savefig(os.path.join(outd, "n2_rotational_distribution_vs_experiment.pdf"))

# ---------- figure 2: the distribution split by collision generation (first case) ----------
lab = pairs[0][0]; rows, T2, ng = case(lab)
if ng > 0:
    ns = col(rows, "ns"); fig, ax = plt.subplots(1, 3, figsize=(14, 4.9))
    j = int(np.argmin(np.abs(ns - 0.2))); r = rows[j]; E = r["E"]; tot = r["f"][0].sum(); DE = E[1] - E[0]; m = E < 5
    ax[0].semilogy(E[m], r["f"][0][m] / tot / DE, "k-", lw=1.8)
    for g in range(ng):
        ax[0].semilogy(E[m], np.maximum(r["f"][2 + g][m], 1e-30) / tot / DE, "-", color=CG[g], lw=1.3)
    ax[0].semilogy(E[m], np.exp(-E[m]) * r["f"][0][(E > 1.9) & (E < 2.1)].mean() / tot / DE * np.exp(2.0), "--", color="0.55", lw=0.9)
    ax[0].set_ylim(1e-5, 30); ax[0].set_xlabel(r"$e_{rot}/kT_2$"); ax[0].set_ylabel("probability density")
    ax[0].text(0.96, 0.9, rf"$n^*={ns[j]:.2f}$", transform=ax[0].transAxes, ha="right")
    sel = ns > 0.01
    for g in range(ng):
        ax[1].plot(ns[sel], np.array([q["Tg"][g] for q in rows])[sel], "-", color=CG[g], lw=1.5)
        ax[2].plot(ns[sel], np.array([q["share"][g] for q in rows])[sel], "-", color=CG[g], lw=1.5)
    ax[1].plot(ns[sel], col(rows, "TR")[sel], "k--", lw=1.2)
    for q, yl in zip(ax[1:], (r"$T_{rot}/T_2$", "share of the molecules")):
        q.set_xlabel(r"$(n-n_1)/(n_2-n_1)$"); q.set_ylabel(yl); q.set_xlim(0, 1); q.set_ylim(0, 1.05)
    hh = [Line2D([], [], color="k", lw=1.8)] + [Line2D([], [], color=c, lw=1.5) for c in CG[:ng]] + [Line2D([], [], color="0.55", lw=0.9, ls="--")]
    fig.legend(hh, ["all molecules"] + GN[:ng] + [r"slope of $T_2$"], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    for q, s in zip(ax, "abc"):
        q.set_title(f"({s})", loc="left", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(os.path.join(outd, f"n2_rotation_by_generation_{lab}.png"), dpi=200); fig.savefig(os.path.join(outd, f"n2_rotation_by_generation_{lab}.pdf"))
