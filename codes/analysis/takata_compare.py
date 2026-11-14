"""Scattered part of the hard-sphere distribution at high Mach number against the infinite-Mach result of
Takata, Aoki & Cercignani (2000).

Their Fig. 5(a) shows (u_-/rho_-) h~ against xi_1/u_-, where h~ is the marginal distribution of the part of
f that is not the delta function at the upstream velocity. The curve for X1/l_- = 0.2 (the upper envelope of
the panel) was traced from the bitmap of the published figure (experiments/takata2000_fig5a_X0p2.json).
Their solution keeps the exact collision operator between the delta and the non-delta part and replaces the
collisions within the non-delta part by a BGK term, so positions are not comparable; the DSMC station is
chosen where the density of the scattered molecules (all but generation 0) equals the area under their curve.

Usage: python takata_compare.py <shock_data dir>
Writes paper/figures/fig19_takata.* and analysis/out/takata_compare.json.
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generations as G

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.7, "lines.linewidth": 1.2,
                     "savefig.dpi": 220, "pdf.fonttype": 42})


def main(root):
    t = json.load(open(os.path.join(root, "experiments", "takata2000_fig5a_X0p2.json")))
    xt, yt = np.array(t["x"]), np.array(t["y"])
    area = float(np.trapezoid(yt, xt))
    S = dict(takata_area=area, takata_peak=float(yt.max()), takata_peak_at=float(xt[yt.argmax()]), takata_value_at_corner=float(np.interp(0.985, xt, yt)), cases={})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax[0].plot(xt, yt, "k", lw=1.6, label=r"Takata et al., $M=\infty$")
    for M, col, ls in ((8, "C2", (0, (6, 1.5, 1, 1.5))), (12, "C3", "--"), (25, "C0", "-")):
        f = os.path.join(root, "vdf_cls", f"DSMC_w0p5_M{M}_classes.npz")
        if not os.path.exists(f):
            continue
        d = G.load(f); u1 = d["st"]["u1"]
        sc = np.array([G.at_level(d, lv, d["rho"]) - G.at_level(d, lv, d["mgen"][0, :, 0]) for lv in G.GRID])     # density of scattered molecules
        lv = float(np.interp(area, sc, G.GRID))
        y = G.at_level(d, lv, d["G"] - d["Ggen"][0]) * u1; x = d["vx"] / u1
        yi = np.interp(xt, x, y)
        k = (xt > -1.6) & (xt < 1.6)
        rec = dict(nstar=lv, unscattered_fraction=float(np.interp(lv, G.GRID, d["prof"]["frac"][:, 0]) if "prof" in d else np.nan),
                   peak=float(y.max()), peak_at=float(x[y.argmax()]), rms_over_peak=float(np.sqrt(np.mean((yi[k] - yt[k]) ** 2)) / yt.max()),
                   max_abs_over_peak=float(np.abs(yi[k] - yt[k]).max() / yt.max()), value_at_corner=float(np.interp(0.985 - 2.5 / u1, x, y)))
        S["cases"][f"M{M}"] = rec
        ax[0].plot(x, y, color=col, ls=ls, label=f"DSMC, Mach {M}")
        ax[1].plot(xt[k], yi[k] - yt[k], color=col, ls=ls)
    ax[0].set_xlim(-2, 2); ax[0].set_ylim(0, None); ax[0].set_xlabel(r"$\xi_x/u_1$"); ax[0].set_ylabel(r"$(u_1/\rho_1)\,(g-g_0)$"); ax[0].set_title("(a)", fontsize=9)
    ax[1].axhline(0, color="0.6", lw=0.6); ax[1].set_xlim(-2, 2); ax[1].set_xlabel(r"$\xi_x/u_1$"); ax[1].set_ylabel("DSMC minus Takata et al."); ax[1].set_title("(b)", fontsize=9)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(root, "paper", "figures", "fig19_takata." + e))
    json.dump(S, open(os.path.join(root, "analysis", "out", "takata_compare.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
