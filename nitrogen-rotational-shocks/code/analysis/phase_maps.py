"""Colour maps of the nitrogen shock in phase space: the streamwise velocity distribution g(x, xi_x) from DSMC and
from the Rykov model, the distribution of the rotational energy through the shock (DSMC), and the rotational
temperature of the molecules as a function of position and velocity (Rykov model).

The maps are drawn as smooth images (bicubic interpolation) that fade to white where there is no data. The DSMC arrays
are smoothed with a Gaussian kernel whose width grows where the number of samples per bin is small, so that the
statistical scatter of the far tails does not show.

Usage: python phase_maps.py <n2 folder> <Mach> <output folder>
Reads <n2>/vdf_n2/DSMC_w0p74_M<M>_reduced_vdf.npz, <n2>/rykov_map_M<M>.npz, <n2>/erh_tagfix.npz."""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter

root, Ms, outd = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(outd, exist_ok=True)
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "axes.titlesize": 15, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14, "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.8})
CM = plt.get_cmap("turbo")


def smooth_counts(c, fine, coarse, c0=300.0):
    """Counts smoothed lightly where they are large and strongly where they are small; also the strongly smoothed ones."""
    a, b = gaussian_filter(c, fine), gaussian_filter(c, coarse)
    w = b / (b + c0)
    return w * a + (1 - w) * b, b


def image(a, x, y, q, vmin, vmax, weight):
    """Smooth image of q(x, y) on uniform grids, blended into white with the weight (1 = full colour)."""
    rgba = CM(Normalize(vmin, vmax, clip=True)(np.nan_to_num(q, nan=vmin)))
    w = np.clip(gaussian_filter(np.clip(np.nan_to_num(weight, nan=0.0), 0, 1), 1.6), 0, 1)[..., None]      # soft edge
    rgb = rgba[..., :3] * w + (1 - w)
    dx, dy = x[1] - x[0], y[1] - y[0]
    a.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", aspect="auto", interpolation="bicubic",
             extent=(x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2))
    return ScalarMappable(Normalize(vmin, vmax), CM)


def ramp(q, lo, width):
    return np.clip((q - lo) / width, 0, 1)


z = np.load(os.path.join(root, "vdf_n2", f"DSMC_w0p74_M{Ms.replace('.', 'p')}_reduced_vdf.npz")); st = json.loads(str(z["states"]))
u1, u2, T2, n2 = st["u1"], st["u2"], st["T2"], st["rho2"]; cs = float(z["count_scale"])
xd = z["x_mfp"]; xd = xd - np.interp(0.5, z["nstar"], xd); vd = z["vx"]
Gc, Gb = smooth_counts(z["G"] * cs, (1.5, 1.0), (10.0, 3.0))
r = np.load(os.path.join(root, f"rykov_map_M{Ms}.npz")); xr = r["x"] - np.interp(0.5, (r["rho"] - 1) / (n2 - 1), r["x"]); vr = r["v"]
dv = vr[1] - vr[0]; F = r["F"].astype(float) / dv; H = r["H"].astype(float) / dv      # stored as sums over the radial nodes times dv
F = np.maximum(gaussian_filter(F, (0.0, 0.8)), 0.0); H = np.maximum(gaussian_filter(H, (0.0, 0.8)), 0.0)   # removes the grid-scale ripple beside the beam
e = np.load(os.path.join(root, "erh_tagfix.npz")); lab = f"M{Ms}"; xe = e[f"{lab}_x"]; erh = e[f"{lab}_erh"][0]
nb = erh.shape[-1]; DE = 8.0 / nb; E = (np.arange(nb) + 0.5) * DE
Ec, Eb = smooth_counts(erh, (2.0, 0.6), (10.0, 2.5)); pe = Ec / erh.sum(1, keepdims=True) / DE

xh = lambda v: (v - u2) / (u1 - u2)
XL = (-25, 10); VL = (-1.6, 2.2); LO, HI = -6.5, 0.0
fig, ax = plt.subplots(2, 2, figsize=(13, 9.2), constrained_layout=True)
with np.errstate(invalid="ignore", divide="ignore"):
    lgd = np.log10(np.maximum(Gc / cs * (u1 - u2), 1e-30)); lgr = np.log10(np.maximum(F * (u1 - u2), 1e-30))
    wd = np.minimum(ramp(lgd, LO, 0.8), ramp(np.log10(np.maximum(Gb, 1e-30)), math_lo := np.log10(2.0), 0.8)); wr = ramp(lgr, LO, 0.8)
for a, x, v, lg, w, ttl in ((ax[0, 0], xd, vd, lgd, wd, r"(a) DSMC (Larsen–Borgnakke): $g(x,\xi_x)$"), (ax[0, 1], xr, vr, lgr, wr, r"(b) Rykov model: $g(x,\xi_x)$")):
    m = (x > XL[0] - 2) & (x < XL[1] + 2)
    sm = image(a, x[m], xh(v), lg[m], LO, HI, w[m])
    a.contour(x[m], xh(v), gaussian_filter(np.maximum(lg[m], -9.0), 1.5).T, levels=np.arange(-4, 0.1, 1.0), colors="k", linewidths=0.3, alpha=0.55)
    a.axhline(xh(0.0), color="w", lw=1.0, ls="--"); a.set_xlim(*XL); a.set_ylim(*VL)
    a.set_ylabel(r"$\hat\xi=(\xi_x-u_2)/(u_1-u_2)$"); a.set_title(ttl, fontsize=15, loc="left")
    a.text(-24, xh(0.0) - 0.27, "upstream-moving", color="w", fontsize=12)
cb = fig.colorbar(sm, ax=ax[0, :], pad=0.015, ticks=np.arange(-6, 0.1, 1)); cb.set_label(r"$\log_{10}\,[\,g\,(u_1-u_2)/n_1]$")

with np.errstate(invalid="ignore", divide="ignore"):
    lp = np.log10(np.maximum(pe, 1e-30)); wp = np.minimum(ramp(lp, -3.5, 0.6), ramp(np.log10(np.maximum(Eb, 1e-30)), np.log10(2.0), 0.8))
m = (xe > XL[0] - 2) & (xe < XL[1] + 2)
sm = image(ax[1, 0], xe[m], E, lp[m], -3.5, 1.5, wp[m])
ax[1, 0].contour(xe[m], E, gaussian_filter(np.maximum(lp[m], -9.0), 1.5).T, levels=np.arange(-3, 1.1, 1.0), colors="k", linewidths=0.3, alpha=0.55)
ax[1, 0].set_xlim(*XL); ax[1, 0].set_ylim(0, 5); ax[1, 0].set_ylabel(r"$e_{rot}/kT_2$")
ax[1, 0].set_title("(c) DSMC: rotational-energy distribution", fontsize=15, loc="left")
cb = fig.colorbar(sm, ax=ax[1, 0], pad=0.02, ticks=np.arange(-3, 1.1, 1)); cb.set_label(r"$\log_{10}$ probability density")

with np.errstate(invalid="ignore", divide="ignore"):
    tr = np.where(F > 1e-30, H / np.maximum(F, 1e-30) / T2, 0.0)
m = (xr > XL[0] - 2) & (xr < XL[1] + 2)
sm = image(ax[1, 1], xr[m], xh(vr), tr[m], 0.0, 1.6, ramp(lgr[m], -7.5, 1.0))
ax[1, 1].contour(xr[m], xh(vr), np.where(lgr[m] > -7.0, tr[m], np.nan).T, levels=[0.5, 1.0], colors="k", linewidths=0.35, alpha=0.7)
ax[1, 1].axhline(xh(0.0), color="k", lw=0.9, ls="--"); ax[1, 1].set_xlim(*XL); ax[1, 1].set_ylim(*VL)
ax[1, 1].set_ylabel(r"$\hat\xi$"); ax[1, 1].set_title("(d) Rykov model: rotational temperature", fontsize=15, loc="left")
cb = fig.colorbar(sm, ax=ax[1, 1], pad=0.02, ticks=np.arange(0, 1.61, 0.4)); cb.set_label(r"$T_{rot}(x,\xi_x)/T_2$")
for a in ax[1]:
    a.set_xlabel(r"$(x-x_c)/\ell_1$")
fig.savefig(os.path.join(outd, f"n2_phase_maps_M{Ms}.png"), dpi=220); fig.savefig(os.path.join(outd, f"n2_phase_maps_M{Ms}.pdf"), dpi=300)
print("written", os.path.join(outd, f"n2_phase_maps_M{Ms}.png"))
