"""Overview figure of the Letter (Fig. 1): the Mach 7.0 shock with the density split by the number of counted collisions
(DSMC with the trajectory kernel, main runs), molecule glyphs whose rotation grows with the count, a molecule sent back
upstream (precursor), and an inset sketch of the rotational Boltzmann plot with the slope temperatures of the cold
stream, of the molecules after one collision and of those with three or more.

Usage: python make_schematic.py <n2 data folder> <output folder> [src=rin]"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Arc, Patch
from matplotlib.transforms import ScaledTranslation, blended_transform_factory

N2, OUT = sys.argv[1], sys.argv[2]; opt = dict(a.split("=") for a in sys.argv[3:]); SRC = opt.get("src", "rin")
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "font.family": "serif",
                     "mathtext.fontset": "cm", "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
BLUE, PURPLE, ORANGE, RED, GREY = "#1F4E79", "#8E6BBE", "#E08A2B", "#C0392B", "#7F7F7F"
W1 = 3.4

V = np.load(os.path.join(N2, f"erh_{SRC}.npz")); lab = "M7k"
x, n, erh = V[f"{lab}_x"], V[f"{lab}_n"], V[f"{lab}_erh"]; n2 = float(V[f"{lab}_n2"])
k5 = np.ones(5) / 5; sm = lambda a: np.convolve(np.pad(a, 2, mode="edge"), k5, "valid")
tot = erh[0].sum(1); sh = [sm(erh[r].sum(1) / tot) for r in (2, 9, 10, 11)]
ns = (sm(n) - 1) / (n2 - 1); ns[:3] = ns[3]; ns[-3:] = ns[-4]; xc = x[int(np.argmin(np.abs(ns - 0.5)))]; X = x - xc; m = (X > -7.2) & (X < 5.2)
layers = [sm(n) * s for s in sh]


plt.rcParams.update({"axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.6})
fig = plt.figure(figsize=(W1, 2.1)); ax = fig.add_axes([0.13, 0.19, 0.835, 0.6])
ax.stackplot(X[m], *[l[m] for l in layers], colors=[BLUE, PURPLE, ORANGE, RED], alpha=0.8, lw=0.5, edgecolor="white")
ax.plot(X[m], sm(n)[m], "k-", lw=1.2)
ax.set_xlim(-7, 5); ax.set_ylim(0, 6.3); ax.set_xticks([-6, -4, -2, 0, 2, 4]); ax.set_yticks([0, 2, 4, 6])
ax.set_xlabel(r"$(x-x_c)/\ell_1$", labelpad=1); ax.set_ylabel(r"$n/n_1$", labelpad=2)
ax.tick_params(length=2.5, width=0.6, pad=2)
ax.annotate("", xy=(-5.0, 0.45), xytext=(-6.7, 0.45), arrowprops=dict(arrowstyle="-|>", color="w", lw=1.1, mutation_scale=8), zorder=8)
STRIP = blended_transform_factory(ax.transData, fig.transFigure)          # x in data units, y in figure units


def molecule(xd, yf, col, spin, ang=30.0, label=None):
    """Dumbbell in the strip above the axes (x in data units, y in figure units), drawn in inches so that it stays
    round; spin 0..1 sets the length of the rotation arrow."""
    tr = fig.dpi_scale_trans + ScaledTranslation(xd, yf, STRIP)
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang)); h = 0.04
    fig.add_artist(plt.Line2D([-h * c, h * c], [-h * s, h * s], color="k", lw=0.9, transform=tr, zorder=6, solid_capstyle="round"))
    for sg in (-1, 1):
        fig.add_artist(Circle((sg * h * c, sg * h * s), 0.027, fc=col, ec="k", lw=0.5, transform=tr, zorder=7))
    if spin > 0:
        span = 60 + 220 * spin; r = 0.085
        fig.add_artist(Arc((0, 0), 2 * r, 2 * r, theta1=ang + 25, theta2=ang + 25 + span, color="0.25", lw=0.7, transform=tr, zorder=6))
        t2 = np.radians(ang + 25 + span); p = np.array([r * np.cos(t2), r * np.sin(t2)]); d = np.array([-np.sin(t2), np.cos(t2)])
        fig.add_artist(FancyArrowPatch(p - 0.018 * d, p + 0.011 * d, arrowstyle="-|>", mutation_scale=4.5, color="0.25", lw=0.5, transform=tr, zorder=6))
    if label:
        fig.text(0.13, 0.0, label, color=col, fontsize=8, va="center", ha="left", transform=tr)


YS = 0.905
fig.add_artist(FancyArrowPatch((-6.8, YS - 0.088), (4.6, YS - 0.088), arrowstyle="-|>", mutation_scale=7, color=GREY, lw=0.7, transform=STRIP, zorder=2))
molecule(-6.0, YS, BLUE, 0.0, 25, label="0")
molecule(-2.6, YS, PURPLE, 0.3, 60, label="1")
molecule(-0.3, YS, ORANGE, 0.55, 140, label="2")
molecule(2.7, YS, RED, 1.0, 100, label="3+")

# inset: rotational Boltzmann plot (log of the population per state against the level energy), sketched with the
# slope temperatures of the cold stream, of the molecules after one collision and of those with three or more
ins = fig.add_axes([0.2, 0.44, 0.235, 0.29]); E = np.linspace(0.12, 2.0, 50)
T1 = 28.3 / 296.0
ins.plot(E[E < 0.42], 1.4 - E[E < 0.42] / T1, "-", color=BLUE, lw=1.4)
ins.plot(E, -2.2 - E / 0.62, "-", color=PURPLE, lw=1.4)
ins.plot(E, -1.2 - E / 0.92, "-", color=RED, lw=1.4)
ins.set_xlim(0, 2.45); ins.set_ylim(-6.6, 1.2); ins.axis("off")
ins.annotate("", xy=(2.45, -6.6), xytext=(0, -6.6), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.6, mutation_scale=6), annotation_clip=False)
ins.annotate("", xy=(0, 1.2), xytext=(0, -6.6), arrowprops=dict(arrowstyle="-|>", color="k", lw=0.6, mutation_scale=6), annotation_clip=False)
ins.text(2.45, -6.6 - 0.35, r"$E_K$", fontsize=8, ha="right", va="top")
ins.text(1.15, -1.25, r"$0.92\,T_2$", color=RED, fontsize=8, va="center", ha="left")
ins.text(0.25, -5.45, r"$0.62\,T_2$", color=PURPLE, fontsize=8, va="center", ha="left")
ins.text(0.44, 0.35, r"$T_1$", color=BLUE, fontsize=8, va="center", ha="left")
for e in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "fig0." + e), dpi=300)
print("written fig0; x_c", round(float(xc), 2), "; shares at the left edge", [round(float(s[m][0]), 3) for s in sh])
