"""The three fitted populations shown on the streamwise distribution at three stations of the Mach-8 shock:
BGK model (top) and DSMC of hard spheres (bottom).

Usage: python decomposition_figure.py <shock_data dir>   ->  paper/figures/fig22_population_decomposition.{pdf,png}
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
                     "font.family": "serif", "mathtext.fontset": "cm", "savefig.dpi": 300, "pdf.fonttype": 42, "lines.linewidth": 1.3})
LEVELS = (0.1, 0.3, 0.6)
STY = [("hot", "#009E73", ":"), ("intermediate", "#D55E00", "-."), ("beam", "#0072B2", "--")]      # ordered by mean velocity


def main(root):
    fits = os.path.join(root, "analysis", "out", "popfits")
    cases = [("BGK model", os.path.join(root, "vdf_reduced", "M8_BGK_reduced_vdf.npz"), "M8_BGK"),
             ("DSMC, hard spheres", os.path.join(root, "vdf_dsmc", "DSMC_w0p5_M8_reduced_vdf.npz"), "DSMC_w0p5_M8")]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.1), sharex=True)
    for r, (name, path, tag) in enumerate(cases):
        c = sl.load_case(path); f = np.load(os.path.join(fits, tag + "_branch3.npz")); st = c["st"]
        du = st["u1"] - st["u2"]; v = c["vx"]; xi = sl.xi_hat(v, st)
        for j, lv in enumerate(LEVELS):
            i = int(np.argmin(np.abs(f["nstar"] - lv))); p = f["p3"][i].reshape(3, 4); k = int(f["idx"][i])
            p = p[np.argsort(p[:, 1])]                                             # hot, intermediate, beam
            sc = du / c["rho"][k]; ax = axs[r, j]
            gk = [np.exp(q[0]) / np.sqrt(2 * np.pi * np.exp(q[2])) * np.exp(-(v - q[1]) ** 2 / (2 * np.exp(q[2]))) for q in p]
            ax.plot(xi, sum(gk) * sc, color="0.6", lw=3.2, solid_capstyle="round", label="sum of the three")
            ax.plot(xi, c["G"][k] * sc, color="k", lw=1.0, label="kinetic solution")
            for (lab, col, ls), g in zip(STY, gk):
                ax.plot(xi, g * sc, color=col, ls=ls, lw=1.5, label=lab)
            fr = np.exp(p[:, 0]) / np.exp(p[:, 0]).sum(); w = c["wx"]
            res = float(np.sqrt(((sum(gk) - c["G"][k]) ** 2 * w).sum() / (c["G"][k] ** 2 * w).sum()))     # relative L2 residual of g
            ax.set_xlim(-2.6, 2.1); ax.set_ylim(0, None)
            ax.set_title(f"({'abcdef'[3 * r + j]}) $n^*={f['nstar'][i]:.2f}$, residual {100 * res:.1f}%\n" + "mass fractions " + ", ".join(f"{x:.2f}" for x in fr),
                         fontsize=8, loc="left", pad=3)        # kept above the axes, clear of the curves
            if r == 1:
                ax.set_xlabel(r"$\hat\xi=(\xi_x-u_2)/(u_1-u_2)$")
        axs[r, 0].set_ylabel(name + "\n" + r"$g\,(u_1-u_2)/\rho$")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.9, w_pad=0.8)
    fd = os.path.join(root, "paper", "figures")
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(fd, "fig22_population_decomposition." + e))
    print("written fig22_population_decomposition")


if __name__ == "__main__":
    main(sys.argv[1])
