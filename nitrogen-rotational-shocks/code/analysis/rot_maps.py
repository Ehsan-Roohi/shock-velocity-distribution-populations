"""Rotational state of the molecules in phase space (position, streamwise velocity) for the nitrogen shock:
mean rotational energy from DSMC and from the Rykov model, the fraction of molecules that still carry the rotational
energy of the free stream (DSMC), and the mean rotational energy against velocity at two stations of the precursor.

Usage: python rot_maps.py <n2 folder> <Mach> <output folder> [<dsmc map npz> ...]
Reads <n2>/map_lb.npz (dsmc/n2_map_reduce.py) and <n2>/rykov_map_M<M>.npz; further map files (other exchange rules,
written with the same label M<M>) are added as lines in panel (d): give them as name=file."""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter

root, Ms, outd = sys.argv[1], sys.argv[2], sys.argv[3]; extra = [a.split("=", 1) for a in sys.argv[4:]]; os.makedirs(outd, exist_ok=True)
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "axes.titlesize": 15, "legend.fontsize": 13, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.8})
CM = plt.get_cmap("turbo"); lab = f"M{Ms}"; gam = 1.4; M = float(Ms)
n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); u1 = M * np.sqrt(gam); u2 = u1 / n2; T2 = (2 * gam * M * M - (gam - 1)) / (gam + 1) / n2
xh = lambda v: (v - u2) / (u1 - u2)


def load(path):
    d = np.load(path); x, v = d[f"{lab}_x"], d[f"{lab}_v"]; cnt, er, tag = (d[f"{lab}_{k}"].astype(float) for k in ("cnt", "er", "tag"))
    return x, v, cnt, er, tag, float(d[f"{lab}_scale"])


def smooth(c, q, fine=(1.5, 1.0), coarse=(18.0, 4.0), c0=1500.0):
    """Ratio q/c with both smoothed, lightly where the counts are large and strongly where they are small."""
    ca, cb = gaussian_filter(c, fine), gaussian_filter(c, coarse); qa, qb = gaussian_filter(q, fine), gaussian_filter(q, coarse)
    w = cb / (cb + c0)
    return (w * qa + (1 - w) * qb) / np.maximum(w * ca + (1 - w) * cb, 1e-30), cb


def image(a, x, y, q, vmin, vmax, weight, ws=1.6):
    rgba = CM(Normalize(vmin, vmax, clip=True)(np.nan_to_num(q, nan=vmin)))
    w = np.clip(gaussian_filter(np.clip(np.nan_to_num(weight, nan=0.0), 0, 1), ws), 0, 1)[..., None]
    rgb = rgba[..., :3] * w + (1 - w); dx, dy = x[1] - x[0], y[1] - y[0]
    a.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", aspect="auto", interpolation="bicubic", extent=(x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2))
    return ScalarMappable(Normalize(vmin, vmax), CM)


ramp = lambda q, lo, width: np.clip((q - lo) / width, 0, 1)
x, v, cnt, er, tag, scale = load(os.path.join(root, "map_lb.npz"))
trd, cb = smooth(cnt, er); trd /= T2; tgd, _ = smooth(cnt, tag); wd = ramp(np.log10(np.maximum(cb, 1e-30)), np.log10(25.0), 0.9)
r = np.load(os.path.join(root, f"rykov_map_M{Ms}.npz")); xr = r["x"] - np.interp(0.5, (r["rho"] - 1) / (n2 - 1), r["x"]); vr = r["v"]; dv = vr[1] - vr[0]
F = np.maximum(gaussian_filter(r["F"].astype(float) / dv, (0.0, 0.8)), 0.0); H = np.maximum(gaussian_filter(r["H"].astype(float) / dv, (0.0, 0.8)), 0.0)
with np.errstate(invalid="ignore", divide="ignore"):
    trr = np.where(F > 1e-30, H / np.maximum(F, 1e-30) / T2, 0.0); lgr = np.log10(np.maximum(F * (u1 - u2), 1e-30))
XL = (-20, 10); VL = (-1.6, 2.2)
fig, ax = plt.subplots(2, 2, figsize=(13, 9.4), constrained_layout=True)
sm = image(ax[0, 0], x, xh(v), trd, 0.0, 1.6, wd); ax[0, 0].set_title("(a) DSMC (Larsen–Borgnakke): rotational temperature", loc="left")
sm = image(ax[0, 1], xr, xh(vr), trr, 0.0, 1.6, ramp(lgr, -7.5, 1.0), ws=(3.0, 3.0)); ax[0, 1].set_title("(b) Rykov model: rotational temperature", loc="left")
cb1 = fig.colorbar(sm, ax=ax[0, :], pad=0.015, ticks=np.arange(0, 1.61, 0.4)); cb1.set_label(r"$T_{rot}(x,\xi_x)/T_2$")
sm = image(ax[1, 0], x, xh(v), tgd, 0.0, 1.0, wd); ax[1, 0].set_title("(c) DSMC: share with free-stream rotation", loc="left")
cb2 = fig.colorbar(sm, ax=ax[1, 0], pad=0.02, ticks=np.arange(0, 1.01, 0.25)); cb2.set_label("fraction")
for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
    a.axhline(xh(0.0), color="k", lw=0.9, ls="--"); a.set_xlim(*XL); a.set_ylim(*VL); a.set_ylabel(r"$\hat\xi=(\xi_x-u_2)/(u_1-u_2)$")
ax[1, 0].set_xlabel(r"$(x-x_c)/\ell_1$")
# (d) mean rotational energy against velocity at two stations of the precursor
a = ax[1, 1]; beta = cnt[:, v < 0].sum(1) / cnt.sum(1); COL = ["#1F4E79", "#C0392B", "#2E8B57", "#8E6BBE"]
sets = [("DSMC", (x, v, cnt, er))] + [(nm, load(pth)[:4]) for nm, pth in extra]
for k, (nm, (xx, vv, cc, ee)) in enumerate(sets):
    bb = cc[:, vv < 0].sum(1) / cc.sum(1)
    for lv, ls in ((1e-2, "-"), (1e-3, "--")):
        i = int(np.argmin(np.abs(np.log(np.maximum(bb[xx < 0], 1e-30)) - np.log(lv)))); s = slice(max(i - 5, 0), i + 6)
        c1 = gaussian_filter(cc[s].sum(0), 1.5); e1 = gaussian_filter(ee[s].sum(0), 1.5); ok = c1 > 300
        a.plot(xh(vv[ok]), e1[ok] / c1[ok] / T2, ls, color=COL[k], lw=1.8, label=f"{nm}, " + (r"$\beta=10^{-2}$" if lv == 1e-2 else r"$\beta=10^{-3}$"))
br = F[:, vr < 0].sum(1) / np.maximum(F.sum(1), 1e-30)
for lv, ls in ((1e-2, "-"), (1e-3, "--")):
    i = int(np.argmin(np.abs(np.log(np.maximum(br[xr < 0], 1e-30)) - np.log(lv)))); ok = F[i] * (u1 - u2) > 1e-7
    a.plot(xh(vr[ok]), trr[i][ok], ls, color="0.45", lw=1.5, label="Rykov, " + (r"$\beta=10^{-2}$" if lv == 1e-2 else r"$\beta=10^{-3}$"))
a.plot([xh(0.0)] * 2, [0, 0.98], color="k", lw=0.9, ls="--"); a.set_xlim(*VL); a.set_ylim(0, 1.5); a.set_xlabel(r"$\hat\xi$"); a.set_ylabel(r"$T_{rot}(\xi_x)/T_2$")
a.set_title("(d) rotational temperature against velocity", loc="left"); a.legend(frameon=False, fontsize=13, ncol=2, loc="upper right", columnspacing=1.2, handlelength=1.8)
ax[0, 0].set_xticklabels([]); ax[0, 1].set_xticklabels([]); ax[0, 1].set_yticklabels([])
fig.savefig(os.path.join(outd, f"n2_rot_maps_M{Ms}.png"), dpi=200); fig.savefig(os.path.join(outd, f"n2_rot_maps_M{Ms}.pdf"), dpi=300)
# numbers
i = int(np.argmin(np.abs(np.log(np.maximum(beta[x < 0], 1e-30)) - np.log(1e-3)))); s = slice(i - 5, i + 6); c1 = cnt[s].sum(0); e1 = er[s].sum(0); t1 = tag[s].sum(0)
for lo, hi in ((-1.6, -0.9), (-0.9, -0.5), (-0.5, xh(0.0)), (xh(0.0), 0.3), (0.3, 0.8)):
    m = (xh(v) > lo) & (xh(v) <= hi)
    print(f"beta = 1e-3, scaled velocity in ({lo:.2f}, {hi:.2f}]: share of molecules {c1[m].sum() / c1.sum():.2e}, T_rot/T2 {e1[m].sum() / c1[m].sum() / T2:.3f}, share with free-stream rotation {t1[m].sum() / c1[m].sum():.3f}")
print("written", os.path.join(outd, f"n2_rot_maps_M{Ms}.png"))
