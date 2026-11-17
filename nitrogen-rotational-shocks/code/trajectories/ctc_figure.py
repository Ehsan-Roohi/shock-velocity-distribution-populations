"""Figure of the single-collision result: rotational energy of a cold stream molecule after the collision that sends it
upstream, from classical trajectories and from the Larsen-Borgnakke and Pullin rules on the same pairs; and, against
the deflection angle, the mean rotational energy given to the stream molecule, the probability that it is excited,
and the differential cross-section next to the isotropic VHS value.

Usage: python ctc_figure.py <out prefix> <raw npz from ctc_kernel.py> [...]   (files without the deflection angle are
used for panels (a) and (b) only)"""
import sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "axes.titlesize": 15, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.8})
SUM = [np.load(f[8:]) for f in sys.argv[2:] if f.startswith("summary=")]; D = [np.load(f) for f in sys.argv[2:] if not f.startswith("summary=")]; T2 = float(D[0]["par_T2"]); om = float(D[0]["par_omega"]); bmax = float(D[0]["par_bmax"]); MU = 14.0
cat = lambda k, src: np.concatenate([d[k] for d in src])
C = dict(ctc="#1F4E79", lb="#C0392B", pu="#2E8B57")
fig, ax = plt.subplots(2, 2, figsize=(13, 9.6), constrained_layout=True)
edges = np.concatenate([[0, 0.05, 0.1, 0.2, 0.3], np.arange(0.5, 6.01, 0.25)]); mid = 0.5 * (edges[1:] + edges[:-1])
sets = [("classical trajectories", "v1x", "e1", "w", C["ctc"], "-"), ("Larsen–Borgnakke", "lb_v1x", "lb_e1", "wv", C["lb"], "--"), ("Pullin", "pu_v1x", "pu_e1", "wv", C["pu"], ":")]
xs = np.linspace(0, 6, 601)
for name, kv, ke, kw, col, ls in sets:
    vx, e, w = cat(kv, D), cat(ke, D) / T2, cat(kw, D); m = vx < 0; h = np.histogram(e[m], edges, weights=w[m])[0] / w[m].sum() / np.diff(edges)
    ax[0, 0].semilogy(mid, h, ls, color=col, lw=2.2, marker="o" if ls == "-" else None, ms=4, label=name)
    cum = np.array([(w[m] * (e[m] < c)).sum() for c in xs]) / w[m].sum(); ax[0, 1].plot(xs, cum, ls, color=col, lw=2.2, label=name)
    if name.startswith("classical"):
        k = (mid > 0.6) & (mid < 4.5) & (h > 0); p = np.polyfit(mid[k], np.log(h[k]), 1); print(f"slope of the tail of the trajectory distribution between 0.6 and 4.5 T2: temperature {-1 / p[0]:.3f} T2; mean {np.average(e[m], weights=w[m]):.3f} T2")
ax[0, 0].set_xlim(0, 6); ax[0, 0].set_ylim(3e-3, 30); ax[0, 0].set_xlabel(r"$e_{rot}'/kT_2$"); ax[0, 0].set_ylabel("probability density")
ax[0, 0].set_title("(a) stream molecule sent upstream: rotational energy", loc="left"); ax[0, 0].legend(frameon=False, loc="upper right")
ax[0, 1].set_xlim(0, 4); ax[0, 1].set_ylim(0, 1.02); ax[0, 1].set_xlabel(r"$e_{rot}'/kT_2$"); ax[0, 1].set_ylabel("cumulative probability")
ax[0, 1].set_title("(b) cumulative distribution", loc="left"); ax[0, 1].legend(frameon=False, loc="lower right")
# deflection
Dc = [d for d in D if "cchi" in d.files]
if Dc:
    c, e1, e10, g, b, drift = (cat(k, Dc) for k in ("cchi", "e1", "e1_0", "w", "b", "drift")); w = g * (np.abs(drift) < 1e-3); et = 0.5 * MU * g ** 2
    sig_vhs = math.pi * 4.17 ** 2 * (273.0 / et) ** (om - 0.5) / math.gamma(2.5 - om); vhs_sr = np.average(sig_vhs, weights=g) / (4 * math.pi)
    ed = np.linspace(-1, 1, 21); cc = 0.5 * (ed[1:] + ed[:-1]); k = np.clip(np.digitize(c, ed) - 1, 0, 19); W = np.bincount(k, w, 20)
    me = np.bincount(k, w * e1, 20) / W / T2; p01 = np.bincount(k, w * (e1 > 0.1 * T2), 20) / W; dsig = math.pi * bmax ** 2 * W / w.sum() / (2 * math.pi * 0.1)
    lb_mean = np.average((et + e10) / (1 + 2.5 - om), weights=w) / T2 / float(Dc[0]["par_zrot"]) ; lb_p = 1 / float(Dc[0]["par_zrot"])
if not Dc and SUM:
    q = SUM[0]; cc, me, p01, dsig, vhs_sr = q["cos"], q["me1"], q["p01"], q["dsig"], float(q["vhs_sr"]); zr = float(D[0]["par_zrot"]); lb_p = 1 / zr
    lb_mean = np.average((0.5 * MU * D[0]["w"] ** 2 + D[0]["e1_0"]) / (1 + 2.5 - om), weights=D[0]["w"]) / T2 / zr
if Dc or SUM:
    a = ax[1, 0]; a.plot(cc, me, "-o", color=C["ctc"], lw=2.2, ms=5, label=r"trajectories: mean $e_{rot}'/kT_2$"); a.plot(cc, p01, "-s", color="#8E6BBE", lw=2.0, ms=5, label=r"trajectories: share with $e_{rot}'>0.1\,kT_2$")
    a.axhline(lb_mean, color=C["lb"], ls="--", lw=1.8, label="Larsen–Borgnakke: mean"); a.axhline(lb_p, color=C["lb"], ls=":", lw=2.0, label="Larsen–Borgnakke: share")
    a.set_xlim(-1, 1); a.set_ylim(0, 1.75); a.set_xlabel(r"$\cos\chi$"); a.set_title("(c) excitation of the stream molecule against deflection", loc="left"); a.legend(frameon=False, loc="upper center", ncol=1, fontsize=13)
    a = ax[1, 1]; a.semilogy(cc, dsig, "-o", color=C["ctc"], lw=2.2, ms=5, label="trajectories"); a.axhline(vhs_sr, color=C["lb"], ls="--", lw=1.8, label="VHS, isotropic")
    a.set_xlim(-1, 1); a.set_ylim(0.8, 200); a.set_xlabel(r"$\cos\chi$"); a.set_ylabel(r"$d\sigma/d\Omega$  [$\mathrm{\AA^2/sr}$]"); a.set_title("(d) differential cross-section", loc="left"); a.legend(frameon=False, loc="upper left")
fig.savefig(sys.argv[1] + ".png", dpi=200); fig.savefig(sys.argv[1] + ".pdf")
print("written", sys.argv[1] + ".png")
