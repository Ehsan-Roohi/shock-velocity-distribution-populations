"""Overview figure: the three Gaussian populations of a Mach-8 hard-sphere shock (DSMC and its fitted populations).

(a) density through the shock divided among the beam, the intermediate population and the hot population;
(b)-(d) the streamwise reduced distribution at three positions with the three fitted Gaussians.
A second output is the highlight image for the journal (one panel, aspect ratio 1.2:1, JPEG at 300 dpi).

Usage: python schematic_populations.py <shock_data dir>
  -> paper/jfm_main/figures/fig0_population_overview.{pdf,png}, paper/jfm_main/graphical_abstract.jpg
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl

RC = {"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
      "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.7, "savefig.dpi": 300, "pdf.fonttype": 42}
C_B, C_I, C_H = "#2C7FB8", "#8E6BBE", "#D7422B"          # beam (cold, fast), intermediate, hot
LEVELS = (0.1, 0.4, 0.7)
TAG = "DSMC_w0p5_M8"


def load(root):
    c = sl.load_case(os.path.join(root, "vdf_dsmc", TAG + "_reduced_vdf.npz"))
    f = np.load(os.path.join(root, "analysis", "out", "popfits", TAG + "_branch3.npz"))
    ok = np.isfinite(f["p3"]).all(1)                                                      # positions with a three-population fit
    f = {k: (f[k][ok] if np.ndim(f[k]) and len(f[k]) == len(ok) else f[k]) for k in f.files}
    P = np.array([p.reshape(3, 4)[np.argsort(p.reshape(3, 4)[:, 1])] for p in f["p3"]])     # hot, intermediate, beam
    return c, f, P


def paper_figure(root, c, f, P):
    plt.rcParams.update(RC)
    st = c["st"]; du = st["u1"] - st["u2"]; v = c["vx"]; xi = sl.xi_hat(v, st)
    x = f["x"] - np.interp(0.5, f["nstar"], f["x"]); n1 = float(c["rho"][0])             # origin where n* = 0.5
    mass = np.exp(P[:, :, 0]) / n1                                                        # station, (hot, intermediate, beam)
    nh, ni, nb = mass[:, 0], mass[:, 1], mass[:, 2]
    fig = plt.figure(figsize=(7.0, 5.3))
    axa = fig.add_axes([0.085, 0.50, 0.89, 0.30])
    axa.fill_between(x, 0, nb, color=C_B, alpha=0.85, lw=0)
    axa.fill_between(x, nb, nb + ni, color=C_I, alpha=0.85, lw=0)
    axa.fill_between(x, nb + ni, nb + ni + nh, color=C_H, alpha=0.85, lw=0)
    axa.plot(x, c["rho"][f["idx"]] / n1, "k", lw=1.4)
    r = st["u1"] / st["u2"]
    axa.set_xlim(x.min(), x.max()); axa.set_ylim(0, 1.13 * r); axa.set_yticks([0, 1, 2, 3, r]); axa.set_yticklabels(["0", "1", "2", "3", r"$n_2/n_1$"])
    axa.set_xlabel(r"$(x-x_c)/\ell_1$"); axa.set_ylabel(r"$n/n_1$")
    for lv, let in zip(LEVELS, "bcd"):
        i = int(np.argmin(np.abs(f["nstar"] - lv)))
        axa.plot([x[i], x[i]], [0, 1.13 * r], color="k", lw=0.7, ls=(0, (4, 3)))
        axa.annotate(f"({let})", xy=(x[i], 1), xycoords=("data", "axes fraction"), xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8.5)
    for x0, x1, lw, txt, ha, xt in ((0.10, 0.36, 1.2, "upstream: cold, fast", "left", 0.10),
                                    (0.80, 0.90, 3.0, "downstream: hot, slow", "right", 0.97)):
        fig.patches.append(FancyArrowPatch((x0, 0.862), (x1, 0.862), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=12, lw=lw, color="0.25"))
        fig.text(xt, 0.882, txt, ha=ha, va="bottom", fontsize=9)
    fig.text(0.012, 0.805, "(a)", ha="left", va="bottom", fontsize=9)
    h = [Patch(color=C_B, alpha=0.85), Patch(color=C_I, alpha=0.85), Patch(color=C_H, alpha=0.85), Line2D([], [], color="k", lw=1.4)]
    fig.legend(h, ["beam", "intermediate population", "hot population", "DSMC"], loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.53, 0.995), columnspacing=1.6, handlelength=1.8)
    w0, gap = 0.27, 0.04
    info = []
    for j, (lv, let) in enumerate(zip(LEVELS, "bcd")):
        i = int(np.argmin(np.abs(f["nstar"] - lv))); k = int(f["idx"][i]); sc = du / c["rho"][k]
        g = [np.exp(q[0]) / np.sqrt(2 * np.pi * np.exp(q[2])) * np.exp(-(v - q[1]) ** 2 / (2 * np.exp(q[2]))) * sc for q in P[i]]
        ax = fig.add_axes([0.085 + j * (w0 + gap), 0.085, w0, 0.27])
        for gk, col in zip(g, (C_H, C_I, C_B)):
            ax.fill_between(xi, 0, gk, color=col, alpha=0.55, lw=0); ax.plot(xi, gk, color=col, lw=1.0)
        gd = c["G"][k] * sc
        ax.plot(xi, gd, "k", lw=1.3)
        ax.set_xlim(-2.1, 1.9); ax.set_ylim(0, 1.12 * gd.max()); ax.set_yticks([])
        ax.set_xticks([0, 1]); ax.set_xticklabels([r"$u_2$", r"$u_1$"]); ax.set_xlabel(r"$\xi_x$")
        ax.set_title(f"({let})  $n^*={f['nstar'][i]:.1f}$", fontsize=9, loc="left")
        if j == 0:
            ax.set_ylabel(r"$g(\xi_x)$")
        fr = mass[i] / mass[i].sum()
        info.append(dict(nstar=float(f["nstar"][i]), x=float(x[i]), frac_hot_int_beam=[round(float(a), 3) for a in fr],
                         res_g=float(f["res3"][i][0])))
    out = os.path.join(root, "paper", "jfm_main", "figures")
    fig.savefig(os.path.join(out, "fig0_population_overview.pdf")); fig.savefig(os.path.join(out, "fig0_population_overview.png"))
    plt.close(fig)
    for d in info:
        print(d)
    ok = (f["nstar"] > 0.02) & (f["nstar"] < 0.98)
    print("largest residual of g, 0.02<n*<0.98:", float(f["res3"][ok, 0].max()),
          "| beam centre, width (units of u1-u2):", np.round(((P[ok, 2, 1] - st["u2"]) / du).mean(), 3), np.round((np.sqrt(np.exp(P[ok, 2, 2])) / du)[[0, len(P[ok]) // 2]], 3),
          "| sqrt(RT1), sqrt(RT2):", round(float(np.sqrt(st["T1"]) / du), 3), round(float(np.sqrt(st["T2"]) / du), 3))


def highlight(root, c, f, P):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans"})
    st = c["st"]; x = f["x"]; n1 = float(c["rho"][0]); mass = np.exp(P[:, :, 0]) / n1
    nh, ni, nb = mass[:, 0], mass[:, 1], mass[:, 2]
    k = x < 3.4; x = x[k]; nh, ni, nb = nh[k], ni[k], nb[k]
    fig = plt.figure(figsize=(6.0, 5.0)); ax = fig.add_axes([0.06, 0.125, 0.90, 0.80])
    ax.fill_between(x, 0, nb, color=C_B, lw=0); ax.fill_between(x, nb, nb + ni, color=C_I, lw=0)
    ax.fill_between(x, nb + ni, nb + ni + nh, color=C_H, lw=0)
    ax.plot(x, nb + ni + nh, color="#1a1a1a", lw=2.6, solid_capstyle="round")
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(0, 4.25); ax.set_xticks([]); ax.set_yticks([])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(1.8)
    ax.plot(1.0, 0.0, ">", color="k", ms=13, transform=ax.transAxes, clip_on=False, zorder=6)
    ax.set_xlabel("through the shock", fontsize=21, labelpad=10)
    ax.text(-2.55, 0.45, "beam", fontsize=30, fontweight="bold", color="white", ha="center", va="center")
    ax.text(2.2, 2.75, "hot", fontsize=34, fontweight="bold", color="white", ha="center", va="center")
    ax.text(1.75, 1.5, "intermediate", fontsize=22, fontweight="bold", color="white", ha="center", va="center")
    ax.annotate("", xy=(-0.25, 0.60), xytext=(0.75, 1.27), arrowprops=dict(arrowstyle="-|>", lw=2.4, color="white", mutation_scale=24, shrinkA=0, shrinkB=0))
    tmp = os.path.join(root, "paper", "jfm_main", "graphical_abstract.png")
    fig.savefig(tmp, dpi=300, facecolor="white"); plt.close(fig)
    im = Image.open(tmp).convert("RGB"); jpg = tmp[:-4] + ".jpg"
    im.save(jpg, "JPEG", quality=95, dpi=(300, 300), subsampling=0)
    im.resize((283, 236), Image.LANCZOS).save(tmp[:-4] + "_thumbnail_check.png")
    print(jpg, im.size, "aspect", round(im.size[0] / im.size[1], 3))


if __name__ == "__main__":
    root = sys.argv[1]
    c, f, P = load(root)
    paper_figure(root, c, f, P)
    if "--highlight" in sys.argv:
        highlight(root, c, f, P)
