"""Figure: stress and heat flux of the DSMC shocks split into the part carried by the unscattered stream and the
part carried by scattered molecules.

Top row (stress): (a) the two parts and their sum for hard spheres at Mach 8 and 25; (b) share of the stress carried
by the unscattered stream for the three collision laws at Mach 8 and 25, with its share of the mass.
Bottom row (heat flux): (c) the two parts and their sum for hard spheres at Mach 8 and 25; (d) the part carried by
the unscattered stream for the three collision laws at Mach 8 and, where the runs with the extra sums exist, Mach 25.

Usage: python collision_split_figure.py <shock_data dir> [class dirs, default: vdf_cls vdf_split]
       (later directories take precedence) -> paper/jfm_main/figures/fig25_collision_split.{pdf,png}
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collision_split import split

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.7,
                     "lines.linewidth": 1.3, "savefig.dpi": 300, "pdf.fonttype": 42})
# collision laws in the colours of the other DSMC figures; the two parts in colours of their own
LAWS = [("w0p5", "hard spheres", "C3"), ("w0p81", r"$\omega=0.81$", "C1"), ("w1", r"$\omega=1$", "C0")]
C_B, C_C, C_T = "#009E73", "#CC79A7", "k"
MACH = ((8, "-", 1.4), (25, "--", 1.0))


def letters(axs):
    for i, ax in enumerate(np.ravel(axs)):
        ax.annotate("(%s)" % "abcdefgh"[i], xy=(0, 1), xycoords="axes fraction", xytext=(-22, 6), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8.5)


def main(root, subs=("vdf_cls", "vdf_split")):
    S = {}
    for sub in subs:
        for tag, _, _ in LAWS:
            for M, _, _ in MACH:
                f = os.path.join(root, sub, f"DSMC_{tag}_M{M}_classes.npz")
                if os.path.exists(f):
                    s = split(f)
                    if (tag, M) in S and "q" in S[tag, M] and "q" not in s:
                        continue
                    S[tag, M] = s
    have_q = "q" in S["w0p5", 8]
    nrow = 2 if have_q else 1
    H = 2.75 * nrow + 0.55
    fig, axs = plt.subplots(nrow, 2, figsize=(7.0, H), squeeze=False)
    k = lambda s: (s["nstar"] > 0.005) & (s["nstar"] < 0.995)

    # (a) stress parts, hard spheres
    ax = axs[0, 0]
    for M, ls, lw in MACH:
        s = S["w0p5", M]; m = k(s); ns = s["nstar"][m]; sg = s["sigma"][:, m]
        lab = (lambda t: t if M == 8 else None)
        ax.plot(ns, sg.sum(0), ls, color=C_T, lw=lw, label=lab("total"))
        ax.plot(ns, sg[0], ls, color=C_B, lw=lw, label=lab("unscattered stream"))
        ax.plot(ns, sg[1:].sum(0), ls, color=C_C, lw=lw, label=lab("scattered molecules"))
    ax.set_ylabel(r"$\sigma_{xx}/(\rho_1u_1^2)$"); ax.set_xlim(0, 1); ax.set_ylim(0, None)
    # (b) share of the stress carried by the unscattered stream
    ax = axs[0, 1]
    for tag, lab, cl in LAWS:
        for M, ls, lw in MACH:
            s = S[tag, M]; m = k(s) & (s["sigma"].sum(0) > 0.02 * s["sigma"].sum(0).max())
            ax.plot(s["nstar"][m], (s["sigma"][0] / s["sigma"].sum(0))[m], ls, color=cl, lw=lw, label=lab if M == 8 else None)
        s = S[tag, 8]; m = k(s)
        ax.plot(s["nstar"][m], s["share"][0][m], ":", color=cl, lw=1.0)
    ax.set_ylabel("share of the unscattered stream"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    if have_q:
        # (c) heat-flux parts, hard spheres
        ax = axs[1, 0]
        for M, ls, lw in MACH:
            s = S["w0p5", M]
            if "q" not in s:
                continue
            m = k(s); ns = s["nstar"][m]; q = s["q"][:, m]
            ax.plot(ns, q.sum(0), ls, color=C_T, lw=lw); ax.plot(ns, q[0], ls, color=C_B, lw=lw); ax.plot(ns, q[1:].sum(0), ls, color=C_C, lw=lw)
        ax.axhline(0, color="0.6", lw=0.6); ax.set_ylabel(r"$q_x/(\rho_1u_1^3/2)$"); ax.set_xlim(0, 1)
        # (d) heat flux carried by the unscattered stream, three collision laws
        ax = axs[1, 1]
        for tag, lab, cl in LAWS:
            for M, ls, lw in MACH:
                s = S[tag, M]
                if "q" not in s:
                    continue
                m = k(s); ax.plot(s["nstar"][m], s["q"][0][m], ls, color=cl, lw=lw)
        ax.set_ylabel(r"$q_x^{(0)}/(\rho_1u_1^3/2)$"); ax.set_xlim(0, 1); ax.set_ylim(0, None)
    for ax in axs[-1]:
        ax.set_xlabel(r"$n^*$")
    letters(axs)
    # legends above the top panels, clear of the curves
    h0, l0 = axs[0, 0].get_legend_handles_labels()
    hM = [Line2D([], [], color="0.3", ls="-", lw=1.4), Line2D([], [], color="0.3", ls="--", lw=1.0)]
    axs[0, 0].legend(h0 + hM, l0 + ["Mach 8", "Mach 25"], loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=2, frameon=False,
                     columnspacing=1.2, handlelength=2.2, borderaxespad=0)
    h1, l1 = axs[0, 1].get_legend_handles_labels()
    h1 += [Line2D([], [], color="0.3", ls=":", lw=1.0)]; l1 += ["mass share, Mach 8"]
    axs[0, 1].legend(h1, l1, loc="lower left", bbox_to_anchor=(0.0, 1.03), ncol=2, frameon=False, columnspacing=1.2,
                     handlelength=2.2, borderaxespad=0)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.42 / H, top=1 - 0.72 / H, wspace=0.27, hspace=0.24)
    out = os.path.join(root, "paper", "jfm_main", "figures")
    fig.savefig(os.path.join(out, "fig25_collision_split.pdf")); fig.savefig(os.path.join(out, "fig25_collision_split.png"))
    print("written", "with heat flux; Mach-25 heat flux for:" if have_q else "stress only",
          [t for t, _, _ in LAWS if "q" in S[t, 25]] if have_q else "")


if __name__ == "__main__":
    main(sys.argv[1], tuple(sys.argv[2:]) or ("vdf_cls", "vdf_split"))
