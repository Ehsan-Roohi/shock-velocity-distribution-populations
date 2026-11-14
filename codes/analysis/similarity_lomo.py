"""Leave-one-Mach-out test in Rankine-Hugoniot-scaled variables.

For a withheld Mach number, two predictions are built from the two neighbouring Mach numbers only,
at fixed n* and linearly in 1/M^2:
  (a) interpolated populations: the K = 3 branch parameters (mass fractions, xi_hat means,
      log T_par/T2, log T_perp/T2);
  (b) interpolated scaled distributions: g (u1-u2)/rho and h (u1-u2)/(rho T2) on a common xi_hat grid.
Only the RH states of the withheld case are used. Both are compared with the withheld DVM solution
and with the K = 3 fit of the withheld state.

Usage: python similarity_lomo.py <shock_data dir>
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl

plt.rcParams.update({"font.size": 9, "legend.fontsize": 7, "font.family": "serif", "mathtext.fontset": "cm", "savefig.dpi": 220, "pdf.fonttype": 42})
TESTS = [("M3", "M2p5", "M4"), ("M4", "M3", "M5"), ("M5", "M4", "M6_gl3fine"), ("M6_gl3fine", "M5", "M8"), ("M8", "M6_gl3fine", "M12_gl3medium")]
GRID = np.linspace(0.03, 0.93, 46)
XIG = np.linspace(-6, 4, 1601)


def scaled_params(f, st):
    p = f["p3"].reshape(len(f["nstar"]), 3, 4)
    ok = np.isfinite(p[:, 0, 0]); o = np.argsort(f["nstar"][ok]); ns = f["nstar"][ok][o]; p = p[ok][o]
    n = np.exp(p[:, :, 0]); frac = n / n.sum(1, keepdims=True)
    xi = (p[:, :, 1] - st["u2"]) / (st["u1"] - st["u2"])
    Q = np.stack([frac, xi, p[:, :, 2] - np.log(st["T2"]), p[:, :, 3] - np.log(st["T2"])], 2)
    return np.array([[[np.interp(g, ns, Q[:, k, j]) for j in range(4)] for k in range(3)] for g in GRID])


def to_params(Q, nstar, st):
    rho = 1 + nstar * (st["rho2"] - 1)
    frac = np.clip(Q[:, 0], 1e-9, None); frac = frac / frac.sum()
    return np.stack([np.log(frac * rho), st["u2"] + Q[:, 1] * (st["u1"] - st["u2"]), Q[:, 2] + np.log(st["T2"]), Q[:, 3] + np.log(st["T2"])], 1).ravel()


def scaled_gh(c):
    st = c["st"]; xi = sl.xi_hat(c["vx"], st); du = st["u1"] - st["u2"]
    g = np.empty((len(GRID), len(XIG))); h = np.empty_like(g)
    for i, ns in enumerate(GRID):
        s = int(np.argmin(np.abs(c["nstar"] - ns)))
        g[i] = np.interp(XIG, xi, c["G"][s] * du / c["rho"][s], left=0, right=0)
        h[i] = np.interp(XIG, xi, c["H"][s] * du / (c["rho"][s] * st["T2"]), left=0, right=0)
    return g, h


def main(root):
    red, fits = os.path.join(root, "vdf_reduced"), os.path.join(root, "analysis/out/popfits")
    load = lambda t: (sl.load_case(os.path.join(red, f"{t}_BGK_reduced_vdf.npz")), np.load(os.path.join(fits, f"{t}_BGK_branch3.npz")))
    out = {}
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.3), constrained_layout=True)
    row = 0
    for tgt, lo, hi in TESTS:
        (c, f), (cl, fl), (ch, fh) = load(tgt), load(lo), load(hi)
        st = c["st"]; s = lambda M: 1 / M ** 2; du = st["u1"] - st["u2"]
        wH = (s(st["M1"]) - s(cl["st"]["M1"])) / (s(ch["st"]["M1"]) - s(cl["st"]["M1"]))
        Qp = (1 - wH) * scaled_params(fl, cl["st"]) + wH * scaled_params(fh, ch["st"])
        Qf = scaled_params(f, st)
        (gl, hl), (gh_, hh) = scaled_gh(cl), scaled_gh(ch)
        gd, hd = (1 - wH) * gl + wH * gh_, (1 - wH) * hl + wH * hh
        v, w = c["vx"], c["wx"]; sw = np.sqrt(w); xi = sl.xi_hat(v, st)
        names = ("pop", "fit", "dir")
        E = {f"{a}_{m}": [] for a in names for m in ("g", "q", "sigma", "M400")}
        E.update({m: [] for m in ("q", "sigma", "M400")})
        keep = {}
        for i, ns in enumerate(GRID):
            j = int(np.argmin(np.abs(c["nstar"] - ns))); g = c["G"][j]; gn = np.linalg.norm(g * sw)
            pp, pf = to_params(Qp[i], c["nstar"][j], st), to_params(Qf[i], c["nstar"][j], st)
            gp, gf = sl.pop_model(pp, v)[0].sum(0), sl.pop_model(pf, v)[0].sum(0)
            gdir = np.interp(xi, XIG, gd[i]) * c["rho"][j] / du
            hdir = np.interp(xi, XIG, hd[i]) * c["rho"][j] * st["T2"] / du
            md = {k: float(a[0]) for k, a in sl.moments(v, w, gdir[None, :], hdir[None, :]).items()}
            mp, mf = sl.pop_moments(pp), sl.pop_moments(pf)
            for a, gg, mm in (("pop", gp, mp), ("fit", gf, mf), ("dir", gdir, md)):
                E[f"{a}_g"].append(np.linalg.norm((gg - g) * sw) / gn)
                for m in ("q", "sigma", "M400"):
                    E[f"{a}_{m}"].append(mm[m])
            for m in ("q", "sigma", "M400"):
                E[m].append(c[m][j])
            if abs(ns - 0.1) < 0.011 or abs(ns - 0.5) < 0.011:
                keep[round(float(ns), 1)] = (g, gp, gdir)
        A = {k: np.array(vv) for k, vv in E.items()}
        res = dict(mach=st["M1"], neighbours=[cl["st"]["M1"], ch["st"]["M1"]])
        for a in names:
            res[f"g_max_{a}"] = float(100 * A[f"{a}_g"].max()); res[f"g_mean_{a}"] = float(100 * A[f"{a}_g"].mean())
            for m in ("q", "sigma", "M400"):
                res[f"{m}_L2_{a}"] = float(100 * np.linalg.norm(A[f"{a}_{m}"] - A[m]) / np.linalg.norm(A[m]))
        out[tgt] = res
        print(tgt, json.dumps({k: (round(v_, 2) if isinstance(v_, float) else v_) for k, v_ in res.items()}))
        if tgt in ("M3", "M6_gl3fine"):
            for col, lv in enumerate((0.1, 0.5)):
                g, gp, gdir = keep[lv]; ax = axs[row, col]
                ax.plot(xi, g / g.max(), "k-", lw=1.7, label="DVM (withheld)")
                ax.plot(xi, gp / g.max(), "C3--", label="interpolated populations")
                ax.plot(xi, gdir / g.max(), "C0:", lw=1.4, label="interpolated scaled $g$")
                ax.set_xlim(-1.6, 1.9); ax.set_title(f"$M={st['M1']:g}$, $n^*={lv}$", fontsize=8, loc="left", pad=3)
                ax.set_xlabel(r"$\hat\xi$"); ax.set_ylabel(r"$g/\max g$")
            ax = axs[row, 2]
            for m, col_, lab in (("q", "C3", r"$q_x$"), ("M400", "C0", r"$M_{400}^{\rm neq}$")):
                nrm = np.abs(A[m]).max()
                ax.plot(GRID, A[m] / nrm, "-", color="k" if m == "q" else "0.55", lw=1.7, label=lab + " DVM")
                ax.plot(GRID, A["pop_" + m] / nrm, "--", color=col_, label=lab + " populations")
                ax.plot(GRID, A["dir_" + m] / nrm, ":", color=col_, lw=1.4, label=lab + " scaled $g,h$")
            ax.set_xlabel(r"$n^*$"); ax.set_ylabel("moment / peak")
            ax.set_title(f"$M={st['M1']:g}$ from $M={cl['st']['M1']:g}$ and ${ch['st']['M1']:g}$", fontsize=8, loc="left", pad=3)
            if row == 0:
                fig.legend(*axs[0, 1].get_legend_handles_labels(), frameon=False, loc="outside upper left", ncol=1, fontsize=6.8, handlelength=2.2)
                fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="outside upper right", ncol=2, fontsize=6.8, handlelength=2.2, columnspacing=1.2)
            row += 1
    fd = os.path.join(root, "paper", "figures"); os.makedirs(fd, exist_ok=True)
    fig.savefig(os.path.join(fd, "fig9_leave_one_mach_out.pdf")); fig.savefig(os.path.join(fd, "fig9_leave_one_mach_out.png"))
    json.dump(out, open(os.path.join(root, "analysis", "out", "lomo_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
