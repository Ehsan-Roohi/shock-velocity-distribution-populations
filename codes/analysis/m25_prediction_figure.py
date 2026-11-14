"""Prediction of the Mach-25 DSMC shock from Mach 8 and 12, shown as profiles in physical space.

The three predictions are those of the paper, obtained by importing the existing functions (nothing is re-derived
here): three populations (P; predictor.py, predictor_models.py), direct combination of the scaled distributions (D)
and the same with the beam treated separately (H; predictor_direct.py), all through methods_figure.case().
The position x(n*) of each prediction follows from the moment equation (predictor.x_of_nstar); the predicted and the
DSMC profiles are aligned at n* = 0.5 and nothing else is taken from the Mach-25 simulation.

Rows: hard spheres and omega = 1. Columns: n*(x); heat flux and stress against x; the same against n*.
The errors printed at the end are compared with analysis/out/predictor_direct.json and predictor_models_summary.json
(the numbers of tables tab:predictor2, tab:predictor3 and tab:methods of the manuscript).

Usage: python m25_prediction_figure.py <shock_data dir>
Writes paper/jfm_main/figures/fig24_m25_prediction.pdf and analysis/fig24_m25_prediction.png.
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl
import predictor as P
import predictor_direct as PD
import predictor_models as PM
import methods_figure as MF

GRID = P.GRID
RC = {"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
      "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.7, "savefig.dpi": 300, "pdf.fonttype": 42,
      "lines.linewidth": 1.3}
# colours of fig. 20 (methods_figure.STY); the DSMC reference is the solid line, so the predictions are broken lines
STY = {"P": dict(color=MF.STY["P"]["color"], ls="--", lw=1.3, label="three populations (P)"),
       "D": dict(color=MF.STY["D"]["color"], ls=":", lw=1.3, label="direct combination (D)"),
       "H": dict(color=MF.STY["H"]["color"], ls="-.", lw=1.3, label="separate beam (H)")}
GASES = [("DSMC, hard sphere", "hard spheres"), ("DSMC, omega=1", r"$\omega=1$")]


def predict(root, label, tgt=25, lo=8, hi=12):
    """Predictions P, D and H of Mach tgt from Mach lo and hi for one DSMC gas (label: key of predictor_models.SETS)."""
    om, folder, pat, _ = PM.SETS[label]
    fits = os.path.join(root, "analysis", "out", "popfits")
    cs = {m: sl.load_case(os.path.join(root, folder, pat.format(m=m) + "_reduced_vdf.npz")) for m in (tgt, lo, hi)}
    fit = lambda m: np.load(os.path.join(fits, cs[m]["name"] + "_branch3.npz"))
    out, ref, xd, delta = MF.case(cs[tgt], cs[lo], cs[hi], fit(lo), fit(hi), lambda F: F["rho"] * F["T"] ** (1 - om), label)
    return dict(c=cs[tgt], out=out, ref=ref, xd=xd, delta=delta, om=om, label=label, mach=tgt, sources=[lo, hi])


def xspace(xd, xp, ref, F):
    """Errors in physical space, defined as in predictor.extra_checks and predictor_direct.run (rms over the common
    range of x, in percent of the density jump or of the peak), and the same differences as relative L2 norms."""
    lo, hi = max(xd[0], xp[0]), min(xd[-1], xp[-1]); X = np.linspace(lo, hi, 300)
    r = dict(nstar_rms=float(100 * np.sqrt(np.mean((np.interp(X, xp, GRID) - np.interp(X, xd, GRID)) ** 2))))
    for key in ("q", "sigma"):
        fd, fp = np.interp(X, xd, ref[key]), np.interp(X, xp, F[key])
        r[key + "_rms_peak"] = float(100 * np.sqrt(np.mean((fp - fd) ** 2)) / np.abs(ref[key]).max())
        r[key + "_L2"] = float(100 * np.linalg.norm(fp - fd) / np.linalg.norm(fd))
    return r


def equal_nstar(ref, F):
    """Errors at equal n*: relative L2 norm over 0.03 <= n* <= 0.93 (tables of the manuscript) and rms in percent of the peak."""
    r = {}
    for key in ("q", "sigma"):
        d = F[key] - ref[key]
        r[key + "_L2"] = float(100 * np.linalg.norm(d) / np.linalg.norm(ref[key]))
        r[key + "_rms_peak"] = float(100 * np.sqrt(np.mean(d ** 2)) / np.abs(ref[key]).max())
    return r


def errors(pr):
    width = lambda xx: float(np.interp(0.8, GRID, xx) - np.interp(0.2, GRID, xx))
    res = {}
    for m in "PDH":
        o = pr["out"][m]
        res[m] = dict(equal_nstar=equal_nstar(pr["ref"], o["F"]), x=xspace(pr["xd"], o["xp"], pr["ref"], o["F"]),
                      g_max=float(100 * o["eg"].max()), g_n049_n051=[float(100 * np.interp(v, GRID, o["eg"])) for v in (0.49, 0.51)],
                      width_ratio=width(o["xp"]) / width(pr["xd"]), monotone=bool(np.all(o["dxdn"] > 0)),
                      x_max_over_delta=float(np.abs(o["xp"] - pr["xd"]).max() / pr["delta"]))
    return res


def check_tables(root, prs):
    """Compare with the stored results behind the manuscript tables."""
    jd = json.load(open(os.path.join(root, "analysis", "out", "predictor_direct.json")))
    jp = json.load(open(os.path.join(root, "analysis", "out", "predictor_models_summary.json")))
    rows = []
    for pr in prs:
        key = f"{pr['label']}|M{pr['mach']} from {pr['sources'][0]},{pr['sources'][1]}"; e = errors(pr)
        stored = {"P": {a: jp[key][b] for a, b in (("q", "q_L2_pred"), ("sigma", "sigma_L2_pred"), ("qx", "xspace_q_rms_pred"), ("sx", "xspace_sigma_rms_pred"))},
                  "D": {a: jd[key][b] for a, b in (("q", "q_L2"), ("sigma", "sigma_L2"), ("qx", "xspace_q_rms"), ("sx", "xspace_sigma_rms"))},
                  "H": {a: jd[key + "|hybrid"][b] for a, b in (("q", "q_L2"), ("sigma", "sigma_L2"), ("qx", "xspace_q_rms"), ("sx", "xspace_sigma_rms"))}}
        for m in "PDH":
            here = dict(q=e[m]["equal_nstar"]["q_L2"], sigma=e[m]["equal_nstar"]["sigma_L2"], qx=e[m]["x"]["q_rms_peak"], sx=e[m]["x"]["sigma_rms_peak"])
            dmax = max(abs(here[a] - stored[m][a]) for a in here)
            rows.append((key, m, here, stored[m], dmax))
            print(f"{key:36s} {m}: equal n*  q {here['q']:.3f} (stored {stored[m]['q']:.3f})  sigma {here['sigma']:.3f} ({stored[m]['sigma']:.3f})   "
                  f"in x  q {here['qx']:.3f} ({stored[m]['qx']:.3f})  sigma {here['sx']:.3f} ({stored[m]['sx']:.3f})   largest difference {dmax:.1e}", flush=True)
    return rows


def dsmc_profile(pr, lo=0.004, hi=0.985):
    """DSMC profiles on the cells of the simulation, with x measured from the plane n* = 0.5 (same origin as xd)."""
    c = pr["c"]; k, ns, _, _ = PD.on_grid(c)
    xc = float(np.interp(0.5, GRID, pr["ref"]["x"]))
    j = k[(ns >= lo) & (ns <= hi)]
    return dict(x=c["x"][j] - xc, nstar=c["nstar"][j], q=c["q"][j], sigma=c["sigma"][j])


def main(root):
    prs = [predict(root, label) for label, _ in GASES]
    check_tables(root, prs)
    plt.rcParams.update(RC)
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.55))
    show = "PH"
    for r, (pr, (_, gas)) in enumerate(zip(prs, GASES)):
        d = dsmc_profile(pr); ref = pr["ref"]; qs, ss = np.abs(ref["q"]).max(), np.abs(ref["sigma"]).max()
        a0, a1, a2 = axs[r]
        a0.plot(d["x"], d["nstar"], "k-", lw=1.6)
        a1.plot(d["x"], d["q"] / qs, "k-", lw=1.6); a1.plot(d["x"], d["sigma"] / ss, "k-", lw=1.6)
        a2.plot(GRID, ref["q"] / qs, "k-", lw=1.6); a2.plot(GRID, ref["sigma"] / ss, "k-", lw=1.6)
        for m in show:
            o = pr["out"][m]; F = o["F"]; st = {k: v for k, v in STY[m].items() if k != "label"}
            a0.plot(o["xp"], GRID, **st)
            a1.plot(o["xp"], F["q"] / qs, **st); a1.plot(o["xp"], F["sigma"] / ss, **st)
            a2.plot(GRID, F["q"] / qs, **st); a2.plot(GRID, F["sigma"] / ss, **st)
        w = pr["xd"][-1] - pr["xd"][0]; xl = (pr["xd"][0] - 0.28 * w, pr["xd"][-1] + 0.20 * w)
        for ax in (a0, a1):
            ax.set_xlim(*xl); ax.set_xlabel(r"$(x-x_c)/\ell_1$")
        a0.set_ylim(-0.03, 1.03); a0.set_ylabel(r"$n^*$")
        a2.set_xlim(0, 1); a2.set_xlabel(r"$n^*$")
        for ax in (a1, a2):
            ax.set_ylim(-1.12, 1.12); ax.axhline(0, color="0.6", lw=0.6, zorder=0)
            ax.set_ylabel(r"$q_x/|q_x|_{\max}$,  $\sigma_{xx}/\sigma_{xx,\max}$")
        # labels in regions without curves: upstream of the shock in (b), (e); at large n* in (c), (f)
        a1.text(0.05, 0.90, r"$\sigma_{xx}$", transform=a1.transAxes, ha="left", va="center", fontsize=9)
        a1.text(0.05, 0.10, r"$q_x$", transform=a1.transAxes, ha="left", va="center", fontsize=9)
        a2.text(0.95, 0.90, r"$\sigma_{xx}$", transform=a2.transAxes, ha="right", va="center", fontsize=9)
        a2.text(0.95, 0.10, r"$q_x$", transform=a2.transAxes, ha="right", va="center", fontsize=9)
        a0.text(0.05, 0.90, gas, transform=a0.transAxes, ha="left", va="center", fontsize=8.5)
    for i, ax in enumerate(axs.ravel()):                      # panel letters outside the axes
        ax.annotate("(%s)" % "abcdef"[i], xy=(0, 1), xycoords="axes fraction", xytext=(-3, 5), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color="k", lw=1.6, label="DSMC, Mach 25")] + [Line2D([], [], color=STY[m]["color"], ls=STY[m]["ls"], lw=STY[m]["lw"], label=STY[m]["label"]) for m in show]
    fig.legend(handles=h, loc="upper center", ncol=len(h), frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.8, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.8, h_pad=1.0)
    fig.savefig(os.path.join(root, "paper", "jfm_main", "figures", "fig24_m25_prediction.pdf"))
    fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig24_m25_prediction.png"))
    for pr in prs:
        e = errors(pr)
        print(pr["label"], "delta/l1 = %.3f" % pr["delta"], {m: dict(width_ratio=round(e[m]["width_ratio"], 3), nstar_x=round(e[m]["x"]["nstar_rms"], 2),
                                                                    x_max_over_delta=round(e[m]["x_max_over_delta"], 3)) for m in "PDH"})


if __name__ == "__main__":
    main(sys.argv[1])
