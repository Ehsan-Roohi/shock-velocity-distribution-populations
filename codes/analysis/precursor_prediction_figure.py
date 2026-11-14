"""Decay of the upstream precursor: DSMC against the linear cascade model with the Rankine-Hugoniot source.

Panels (a)-(c): upstream-moving mass fraction beta at Mach 8 for hard spheres, omega = 0.81 and omega = 1, against
A (N - N0): DSMC (eight runs pooled, vdf_dsmc/DSMC_w*_M8_reduced_vdf.npz; N is the column density upstream of the
cell where n* = 0.5, as in dsmc/precursor_seeds.py) and the cascade model whose source is a half-range Maxwellian
of the downstream state (mean of four runs, cascade_fine/casc_M8_w*_s*.txt, tally cells of L/1280; N is the depth below the source plane,
n1 = 1). N0 is the column density at which beta = 1e-2 on each curve: the model does not fix the position of its
source plane relative to the shock, and the statistic of the decay-rate table (least-squares slope of ln beta
against A N over the cells with 1e-4 < beta < 1e-2) does not depend on the origin. The shaded band is that range.

Panel (d): ratio of the decay rate of the model to that of DSMC for Mach 8, 12, 25 and the three collision laws,
with the numbers of the decay-rate table: means over the runs and 95 % intervals of the means from
analysis/out/rate_seeds_dsmc.json, rate_seeds_dsmc_m25.json (DSMC, eight runs) and rate_seeds_casc_fine.json (model,
four runs). The model rates are recomputed here from the cascade files and compared with the JSON.

Usage: python precursor_prediction_figure.py <shock_data dir>
Writes paper/jfm_main/figures/fig23_precursor_prediction.pdf, analysis/fig23_precursor_prediction.png and
analysis/out/precursor_prediction_figure.json.
"""
import sys, os, json, glob, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.legend_handler import HandlerTuple

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.7,
                     "lines.linewidth": 1.2, "savefig.dpi": 220, "pdf.fonttype": 42})
LO, HI = 1e-4, 1e-2
T95 = {4: 3.182, 8: 2.365}
# file tag, omega, key in the JSON files, colour (as in fig11_dsmc_precursor), label
OMS = [("0p5", 0.5, "0.5", "C3", "hard spheres"), ("0p81", 0.81, "0.81", "C1", r"$\omega=0.81$"), ("1", 1.0, "1.0", "C0", r"$\omega=1$")]
MACHS = (8, 12, 25)
M_SHOWN = 8
C_BAND, C_DSMC_CI = "0.91", "0.72"


def vhs_A(om):
    sref = 15 * math.sqrt(math.pi) / (2 * (5 - 2 * om) * (7 - 2 * om))
    return sref * 4 ** (om - 0.5) / math.gamma(2.5 - om)


def dsmc_profile(f):
    """beta and column density of the pooled runs, defined as in dsmc/precursor_seeds.py (profile)."""
    z = np.load(f, allow_pickle=True)
    x = z["x_mfp"].astype(float); vx = z["vx"]; w = z["wx"]; G = z["G"]; dx = x[1] - x[0]
    rho = G @ w; beta = (G[:, vx < 0] @ w[vx < 0]) / rho
    r = json.loads(str(z["states"]))["rho2"]; ic = int(np.argmin(np.abs((rho - 1) / (r - 1) - 0.5)))
    N = np.concatenate([np.cumsum(rho[:ic][::-1])[::-1] * dx, np.zeros(len(x) - ic)])
    up = (np.arange(len(x)) < ic) & (x > x[0] + 5)             # upstream of the centre, clear of the inflow reservoir
    return x, beta, N, up, int(z["nseeds"])


def crossing(AN, beta, level):
    """A N at which beta falls through level (linear interpolation of ln beta; AN increasing, beta decreasing)."""
    lb = np.log(beta); i = int(np.where(lb < math.log(level))[0][0])
    return float(AN[i - 1] + (AN[i] - AN[i - 1]) * (math.log(level) - lb[i - 1]) / (lb[i] - lb[i - 1]))


def main(root):
    out = os.path.join(root, "analysis", "out")
    J = lambda n: json.load(open(os.path.join(out, n)))
    D = {**J("rate_seeds_dsmc.json"), **J("rate_seeds_dsmc_m25.json")}; C = J("rate_seeds_casc_fine.json")

    # ---- numbers of the decay-rate table, and check of the model rates against the cascade files
    rows = []
    print("omega  M   DSMC (8 runs)        model (4 runs)       model/DSMC  [model recomputed from files; DSMC pooled-run slope]")
    for tag, om, key, col, lab in OMS:
        A = vhs_A(om)
        for M in MACHS:
            d = D[f"M{M}_w{key}"]; c = C[f"casc_M{M}_w{key}"]
            fs = sorted(glob.glob(os.path.join(root, "cascade_fine", f"casc_M{M}_w{key}_s*.txt"))); sl = []
            for f in fs:
                a = np.loadtxt(f); k = (a[:, 1] < HI) & (a[:, 1] > LO)
                sl.append(-np.polyfit(-A * a[k, 0], np.log(a[k, 1]), 1)[0])
            sl = np.array(sl); m_re = float(sl.mean()); ci_re = float(T95[len(sl)] * sl.std(ddof=1) / math.sqrt(len(sl)))
            assert abs(m_re - c["mean"]) < 1e-9 and abs(ci_re - c["ci95_half"]) < 1e-9, (M, key, m_re, c["mean"])
            x, beta, N, up, ns = dsmc_profile(os.path.join(root, "vdf_dsmc", f"DSMC_w{tag}_M{M}_reduced_vdf.npz"))
            k = up & (beta < HI) & (beta > LO); pooled = float(-np.polyfit(A * N[k], np.log(beta[k]), 1)[0])
            r = dict(omega=om, mach=M, dsmc=d["mean"], dsmc_ci95=d["ci95_half"], dsmc_runs=d["n"], model=c["mean"], model_ci95=c["ci95_half"],
                     model_runs=c["n"], ratio=c["mean"] / d["mean"], ratio_ci95_dsmc=d["ci95_half"] / d["mean"], ratio_ci95_model=c["ci95_half"] / d["mean"],
                     ratio_ci95_combined=(c["mean"] / d["mean"]) * math.hypot(d["ci95_half"] / d["mean"], c["ci95_half"] / c["mean"]),
                     dsmc_pooled_slope=pooled, model_recomputed=m_re)
            rows.append(r)
            print(f"{om:<5g} {M:<3d} {d['mean']:.5f} +- {d['ci95_half']:.5f}   {c['mean']:.5f} +- {c['ci95_half']:.5f}   {r['ratio']:.4f} +- {r['ratio_ci95_combined']:.4f}"
                  f"   [{m_re:.5f} +- {ci_re:.5f}; {pooled:.5f}]")
    json.dump(dict(window=[LO, HI], mach_shown=M_SHOWN, cases=rows), open(os.path.join(out, "precursor_prediction_figure.json"), "w"), indent=1)

    # ---- figure
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0))
    pan = [axs[0, 0], axs[0, 1], axs[1, 0]]
    for ax, (tag, om, key, col, lab) in zip(pan, OMS):
        A = vhs_A(om)
        x, beta, N, up, ns = dsmc_profile(os.path.join(root, "vdf_dsmc", f"DSMC_w{tag}_M{M_SHOWN}_reduced_vdf.npz"))
        o = np.argsort(N[up]); AN = A * N[up][o]; b = beta[up][o]                       # increasing column density
        a = np.mean([np.loadtxt(f) for f in sorted(glob.glob(os.path.join(root, "cascade_fine", f"casc_M{M_SHOWN}_w{key}_s*.txt")))], 0)
        ANm = -A * a[:, 0]; bm = a[:, 1]
        n0, m0 = crossing(AN, b, HI), crossing(ANm, bm, HI)
        e_d, e_m = crossing(AN, b, LO), crossing(ANm, bm, LO)
        ylo = 1.5e-5
        last = np.where(b > ylo)[0]; xmax = AN[last[-1]] - n0 if b.min() > ylo else crossing(AN, b, ylo) - n0
        xmax = max(xmax, e_d - n0) * 1.04; xmin = -0.09 * xmax
        kd = (AN - n0 > xmin) & (b > 0); km = (ANm - m0 > xmin) & (ANm - m0 < xmax * 1.02) & (bm > 0)
        ax.axhspan(LO, HI, color=C_BAND, lw=0, zorder=0)
        ax.semilogy(AN[kd] - n0, b[kd], "-", color=col, lw=1.7, zorder=2)
        ax.semilogy(ANm[km] - m0, bm[km], "--", color="k", lw=1.0, dashes=(4, 2.2), zorder=3)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ylo, 4e-2)
        ax.set_xlabel(r"$A\,(N-N_0)$"); ax.set_ylabel(r"mass fraction $\beta$")
        ax.text(0.96, 0.94, lab + f", $M={M_SHOWN}$", transform=ax.transAxes, ha="right", va="top", fontsize=8.5)
        print(f"panel {lab}: A N0 DSMC {n0:.3f} (from the shock centre), model {m0:.3f} (depth); window ends at A(N-N0) = {e_d - n0:.3f} (DSMC), {e_m - m0:.3f} (model); "
              f"DSMC beta at the upstream end of the data {b[-1]:.2e}; runs pooled {ns}")

    # (d) ratio of the decay rates
    ax = axs[1, 1]; xt = []; wbox = 0.56
    ax.axhline(1.0, color="0.35", lw=0.7, zorder=1)
    for y in (0.98, 1.02):
        ax.axhline(y, color="0.35", lw=0.8, ls=":", zorder=1)
    for j, (tag, om, key, col, lab) in enumerate(OMS):
        for i, M in enumerate(MACHS):
            r = next(q for q in rows if q["omega"] == om and q["mach"] == M); xp = 3.7 * j + i; xt.append(xp)
            ax.add_patch(Rectangle((xp - wbox / 2, 1 - r["ratio_ci95_dsmc"]), wbox, 2 * r["ratio_ci95_dsmc"], fc=C_DSMC_CI, ec="none", zorder=2))
            ax.errorbar([xp], [r["ratio"]], yerr=[r["ratio_ci95_model"]], fmt="o", ms=4.2, mfc=col, mec="k", mew=0.5, ecolor="k", elinewidth=0.9, capsize=2.2, zorder=4)
        ax.text(3.7 * j + 1, 1.0335, lab, ha="center", va="center", fontsize=8.5)
    ax.set_xticks(xt); ax.set_xticklabels([str(M) for M in MACHS] * 3); ax.tick_params(axis="x", length=2.5)
    ax.set_xlim(-0.8, 3.7 * 2 + 2.8); ax.set_ylim(0.974, 1.038); ax.set_yticks([0.98, 0.99, 1.0, 1.01, 1.02, 1.03])
    ax.set_xlabel(r"Mach number $M$"); ax.set_ylabel("decay rate, model / DSMC")

    for i, ax in enumerate(np.ravel(axs)):
        ax.annotate("(%s)" % "abcd"[i], xy=(0, 1), xycoords="axes fraction", xytext=(-3, 5), textcoords="offset points", ha="right", va="bottom", fontsize=8.5)
    # legend above the panels; entries are listed column by column (three columns, two rows)
    h = [tuple(Line2D([], [], color=c[3], lw=1.7) for c in OMS),
         Line2D([], [], color="0.35", lw=0.8, ls=":"),
         Line2D([], [], color="k", ls="--", lw=1.0, dashes=(4, 2.2)),
         Line2D([], [], marker="o", ms=4.2, mfc="w", mec="k", mew=0.5, color="k", lw=0.9),
         Patch(fc=C_BAND, ec="0.6", lw=0.4),
         Patch(fc=C_DSMC_CI, ec="none")]
    l = ["DSMC", r"(d) $\pm2\%$",
         "cascade model, Rankine\u2013Hugoniot source", "(d) model / DSMC, 95% interval of the model",
         r"fitted range, $10^{-4}<\beta<10^{-2}$", "(d) 95% interval of DSMC"]
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.5,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.25)})
    fig.tight_layout(rect=(0, 0, 1, 0.915), h_pad=1.3, w_pad=1.2)
    fig.savefig(os.path.join(root, "paper", "jfm_main", "figures", "fig23_precursor_prediction.pdf"))
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig23_precursor_prediction.png"), dpi=220)
    print("written fig23_precursor_prediction")


if __name__ == "__main__":
    main(sys.argv[1])
