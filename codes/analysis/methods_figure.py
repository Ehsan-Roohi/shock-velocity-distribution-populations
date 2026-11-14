"""Station-by-station errors of the three predictions of the Mach-12 shock from Mach 6 and 8:
three populations (P), directly combined scaled distributions (D), the same with the beam separate (H).
Top row: BGK model, the three solutions recomputed with the reduced solver. Bottom row: DSMC, omega = 1.
Columns: residual of g, error of the heat flux, error of the stress, error of the position, deviation of the energy flux.

Usage: python methods_figure.py <shock_data dir>   ->  paper/figures/fig20_methods_errors.{pdf,png}
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl
import predictor as P
import predictor_direct as PD
import predictor_models as PM

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
                     "font.family": "serif", "mathtext.fontset": "cm", "savefig.dpi": 300, "pdf.fonttype": 42, "lines.linewidth": 1.3})
GRID = P.GRID
STY = {"P": dict(color="#D55E00", ls="-", label="three populations (P)"),            # colour-blind-safe palette, distinct line styles
       "D": dict(color="#0072B2", ls="--", label="direct combination (D)"),
       "H": dict(color="#009E73", ls="-.", label="separate beam (H)")}


def populations(c, cl, ch, fl, fh, nu_of):
    st = c["st"]; s = lambda M: 1 / M ** 2
    wH = (s(st["M1"]) - s(cl["st"]["M1"])) / (s(ch["st"]["M1"]) - s(cl["st"]["M1"]))
    Qp = (1 - wH) * P.scaled(fl, cl["st"]) + wH * P.scaled(fh, ch["st"])
    n, u, a, b = P.unscale(Qp, st); F = P.mixture_fields(n, u, a, b)
    xp, dxdn = P.x_of_nstar(F, nu_of(F))
    k, ns, _, _ = PD.on_grid(c); v, w = c["vx"], c["wx"]; sw = np.sqrt(w); eg = []
    for i in range(len(GRID)):
        gref = PD._interp_rows(GRID[i], ns, c["G"][k])
        p = np.stack([np.log(n[i]), u[i], np.log(a[i]), np.log(b[i])], 1).ravel()
        eg.append(np.linalg.norm((sl.pop_model(p, v)[0].sum(0) - gref) * sw) / np.linalg.norm(gref * sw))
    e = (0.5 * n * u * (u ** 2 + 3 * a + 2 * b)).sum(1); e1 = st["u1"] * (0.5 * st["u1"] ** 2 + 2.5)
    return dict(eg=np.array(eg), F=F, xp=xp, dxdn=dxdn, energy=e / e1)


def case(c, cl, ch, fl, fh, nu_of, label):
    out = {"P": populations(c, cl, ch, fl, fh, nu_of)}
    for m, fits in (("D", (None, None)), ("H", (fl, fh))):
        PD.KEEP = {}; PD.run(c, cl, ch, nu_of, label, *fits); out[m] = dict(PD.KEEP)
    PD.KEEP = None
    ref, xd = out["D"]["ref"], out["D"]["xd"]
    delta = 1 / np.max(1 / np.gradient(xd, GRID))
    return out, ref, xd, delta


def main(root):
    red, fits = os.path.join(root, "vdf_reduced"), os.path.join(root, "analysis", "out", "popfits")
    L = lambda t: sl.load_case(os.path.join(red, f"{t}_BGK_reduced_vdf.npz")); Fb = lambda t: np.load(os.path.join(fits, f"{t}_BGK_branch3.npz"))
    rows = [("BGK, recomputed",) + case(L("M12r"), L("M6r"), L("M8r"), Fb("M6r"), Fb("M8r"), lambda F: 1.0, "BGK")]
    om, folder, pat, _ = PM.SETS["DSMC, omega=1"]
    G = lambda m: sl.load_case(os.path.join(root, folder, pat.format(m=m) + "_reduced_vdf.npz"))
    cs = {m: G(m) for m in (12, 6, 8)}; Fd = lambda m: np.load(os.path.join(fits, cs[m]["name"] + "_branch3.npz"))
    rows.append((r"DSMC, $\omega=1$",) + case(cs[12], cs[6], cs[8], Fd(6), Fd(8), lambda F: F["rho"] * F["T"] ** (1 - om), "DSMC, omega=1"))

    fig, axs = plt.subplots(2, 5, figsize=(7.4, 3.9), sharex=True)
    titles = [r"$g$ (%)", r"$q_x$ (%)", r"$\sigma_{xx}$ (%)", r"position ($\delta$)", "energy (%)"]
    for r, (name, out, ref, xd, delta) in enumerate(rows):
        for m in "PDH":
            o = out[m]; F = o["F"]; st = STY[m]
            axs[r, 0].semilogy(GRID, 100 * o["eg"], color=st["color"], ls=st["ls"])
            axs[r, 1].plot(GRID, 100 * (F["q"] - ref["q"]) / np.abs(ref["q"]).max(), color=st["color"], ls=st["ls"])
            axs[r, 2].plot(GRID, 100 * (F["sigma"] - ref["sigma"]) / np.abs(ref["sigma"]).max(), color=st["color"], ls=st["ls"])
            axs[r, 3].plot(GRID, (o["xp"] - xd) / delta, color=st["color"], ls=st["ls"])
            axs[r, 4].plot(GRID, 100 * (o["energy"] - 1), color=st["color"], ls=st["ls"])
        for j in range(5):
            ax = axs[r, j]
            if j:
                ax.axhline(0, color="0.6", lw=0.6, zorder=0)
            lab = f"({'abcdefghij'[5 * r + j]})"                       # panel labels above the axes, clear of the curves
            if r == 0:
                ax.set_title(lab + " " + titles[j], fontsize=8.5, pad=4, loc="left")
            else:
                ax.set_title(lab, fontsize=8.5, pad=3, loc="left"); ax.set_xlabel(r"$n^*$")
        axs[r, 0].set_ylabel(name, fontsize=8.5)
        axs[r, 0].set_ylim(0.05, 40)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=STY[m]["color"], ls=STY[m]["ls"], label=STY[m]["label"]) for m in "PDH"],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=2.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.6, h_pad=0.8)
    fd = os.path.join(root, "paper", "figures")
    for e in ("pdf", "png"):
        fig.savefig(os.path.join(fd, "fig20_methods_errors." + e))
    for name, out, ref, xd, delta in rows:
        print(name, {m: dict(g_max=round(float(100 * out[m]["eg"].max()), 2), flux=round(float(100 * np.abs(out[m]["energy"] - 1).max()), 2),
                             x_err=round(float(np.abs((out[m]["xp"] - xd) / delta).max()), 3)) for m in "PDH"})


if __name__ == "__main__":
    main(sys.argv[1])
