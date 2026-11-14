"""Mechanism of the upstream precursor in DSMC at Mach 8, for three viscosity exponents.

Top row: upstream-moving mass fraction beta ahead of the plane where beta = 1e-2, against the distance from that plane:
DSMC; the linear cascade model driven by the DSMC flux through the plane; the uncollided source molecules of the same
model (survival alone); and the DSMC molecules that carry the free-flight flag (flown from the plane without collision).
Bottom row: composition of the upstream-moving molecules: generation 1 (most recent collision involved an unscattered
stream molecule), stream molecules after exactly one collision, free flight in DSMC (flag) and in the model.

Usage: python mechanism_figure.py <shock_data dir>   ->  paper/figures/fig21_precursor_mechanism.{pdf,png}
"""
import sys, os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
                     "font.family": "serif", "mathtext.fontset": "cm", "savefig.dpi": 300, "pdf.fonttype": 42, "lines.linewidth": 1.3})
CASES = [("w0p5", "0.5", "M8_w0.5", r"hard spheres"), ("w0p81", "0.81", "M8_w0.81", r"$\omega=0.81$"), ("w1", "1", "M8_w1.0", r"$\omega=1$")]
C_DSMC, C_CASC, C_SURV, C_GEN, C_ONCE = "k", "#D55E00", "#0072B2", "#009E73", "#CC79A7"


def main(root):
    out = os.path.join(root, "analysis", "out")
    XR = json.load(open(os.path.join(out, "precursor_cascade.json")))["cascade_dsmc_source"]; FL = json.load(open(os.path.join(out, "free_flight.json")))
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex="col")
    for j, (tag, wtab, fkey, lab) in enumerate(CASES):
        z = np.load(os.path.join(root, "vdf_cls", f"DSMC_{tag}_M8_classes.npz"), allow_pickle=True)
        x = z["x"].astype(float); rho = z["rho"].astype(float); neg = z["vx"] < 0; dv = float(z["dv"])
        ng = z["Ggen"][:, :, neg].sum(2) * dv; no = z["Gown"][:, :, neg].sum(2) * dv; nup = ng.sum(0); beta = nup / rho
        xref = XR[f"{tag}_M8"]["x_ref"]; d = xref - x                                  # distance upstream of the reference plane
        a = np.mean([np.loadtxt(f) for f in sorted(glob.glob(os.path.join(root, "cascade", f"tab_M8_w{wtab}_s*.txt")))], 0)
        dc = -a[:, 0]; rc = np.interp(xref - dc, x, rho)
        f = FL[fkey]; lev = ["0.003", "0.001", "0.0003", "0.0001"]
        xf = np.array([f["pooled"][l]["x"] for l in lev]); df = xref - xf
        sh = np.array([f["levels"][l]["mean"][0] for l in lev]); ci = np.array([f["levels"][l]["ci95_half"][0] for l in lev])
        bf = np.interp(xf, x, beta)
        dmax = float(d[(d > 0) & (beta > 2.5e-5)].max()); k = (d >= 0) & (d <= dmax); kc = (dc <= dmax) & (a[:, 1] > 0)
        ax = axs[0, j]
        ax.semilogy(d[k], beta[k], color=C_DSMC, ls="-", lw=1.6, label="DSMC")
        ax.semilogy(dc[kc], a[kc, 1] / rc[kc], color=C_CASC, ls="--", label="cascade model")
        ax.semilogy(dc[kc], a[kc, 2] / rc[kc], color=C_SURV, ls=":", lw=1.6, label="model, survival alone")
        ax.errorbar(df, sh * bf, yerr=ci * bf, fmt="o", ms=3.8, mfc="w", mec=C_SURV, ecolor=C_SURV, elinewidth=0.9, capsize=2, label="DSMC, free flight")
        ax.set_ylim(3e-7, 2e-2); ax.set_title(lab, fontsize=9, pad=4)
        ax.text(0.95, 0.95, f"({'abc'[j]})", transform=ax.transAxes, va="top", ha="right", fontsize=8.5)
        ax = axs[1, j]
        ax.semilogy(d[k], ng[1, k] / nup[k], color=C_GEN, ls="-", label="generation 1")
        ax.semilogy(d[k], no[1, k] / nup[k], color=C_ONCE, ls="-.", label="stream molecules, one collision")
        ax.semilogy(dc[kc], a[kc, 2] / a[kc, 1], color=C_SURV, ls=":", lw=1.6, label="model, free flight")
        ax.errorbar(df, sh, yerr=ci, fmt="o", ms=3.8, mfc="w", mec=C_SURV, ecolor=C_SURV, elinewidth=0.9, capsize=2, label="DSMC, free flight")
        ax.set_ylim(0.01, 1.6)
        ax.text(0.05, 0.07, f"({'def'[j]})", transform=ax.transAxes, va="bottom", ha="left", fontsize=8.5)   # empty lower-left corner
        ax.set_xlim(0, dmax)
    axs[0, 0].set_ylabel(r"mass fraction $\beta$"); axs[1, 0].set_ylabel(r"share of $\beta$")
    fig.supxlabel(r"distance upstream of the plane where $\beta=10^{-2}$, in units of $\ell_1$", fontsize=9, y=0.012)
    h0, l0 = axs[0, 0].get_legend_handles_labels(); h1, l1 = axs[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1[:2], l0 + l1[:2], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.035, 1, 0.90), h_pad=1.0, w_pad=0.8)
    fd = os.path.join(root, "paper", "figures")
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(fd, "fig21_precursor_mechanism." + e))
    print("written fig21_precursor_mechanism")


if __name__ == "__main__":
    main(sys.argv[1])
