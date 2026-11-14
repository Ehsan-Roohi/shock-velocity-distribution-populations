"""Three-population predictor of the shock structure at a Mach number that was not used.

Step 1 (distribution versus n*): the K = 3 population parameters, in Rankine-Hugoniot-scaled form
(mass fractions, xi_hat means, T_par/T2, T_perp/T2), are taken from two other Mach numbers and
combined linearly in 1/M^2 at fixed n*. The target may lie between them (interpolation) or beyond
them (extrapolation). Only the RH states of the target are used.

Step 2 (position): the map x(n*) follows from one moment equation of the kinetic model. For
phi = xi_x^2 the relaxation models give   d/dx int xi_x^3 g dxi_x = - nu sigma_xx ,   so that
    dx/dn* = (d Phi_2 / d n*) / (- nu sigma_xx),   Phi_2 = sum_k n_k (u_k^3 + 3 u_k T_par,k),
with every quantity an algebraic function of the population parameters. nu = 1 for the BGK shocks
(constant relaxation time) and nu = rho T^(1-omega) for mu ~ T^omega.

The result is a shock profile and a velocity distribution at the target Mach number without a
kinetic solution at that Mach number. It is compared with the DVM solution.

Usage: python predictor.py <shock_data dir>
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl

plt.rcParams.update({"font.size": 9, "legend.fontsize": 6.8, "font.family": "serif", "mathtext.fontset": "cm", "savefig.dpi": 220, "pdf.fonttype": 42})
GRID = np.linspace(0.03, 0.93, 46)
TESTS = [("M3", "M2p5", "M4", "interp"), ("M4", "M3", "M5", "interp"), ("M5", "M4", "M6_gl3fine", "interp"),
         ("M6_gl3fine", "M5", "M8", "interp"), ("M8", "M6_gl3fine", "M12_gl3medium", "interp"),
         ("M8", "M5", "M6_gl3fine", "extrap"), ("M12_gl3medium", "M6_gl3fine", "M8", "extrap"),
         ("M12_gl3medium", "M5", "M8", "extrap"), ("M12_gl3medium", "M5", "M6_gl3fine", "extrap")]


def scaled(f, st):
    p = f["p3"].reshape(len(f["nstar"]), 3, 4)
    ok = np.isfinite(p[:, 0, 0]); o = np.argsort(f["nstar"][ok]); ns = f["nstar"][ok][o]; p = p[ok][o]
    n = np.exp(p[:, :, 0]); frac = n / n.sum(1, keepdims=True)
    Q = np.stack([frac, (p[:, :, 1] - st["u2"]) / (st["u1"] - st["u2"]), np.exp(p[:, :, 2]) / st["T2"], np.exp(p[:, :, 3]) / st["T2"]], 2)
    return np.array([[[np.interp(g, ns, Q[:, k, j]) for j in range(4)] for k in range(3)] for g in GRID])


def unscale(Q, st):
    """Q (grid, 3, 4) -> physical n_k, u_k, a_k, b_k on the n* grid."""
    rho = 1 + GRID * (st["rho2"] - 1)
    frac = np.clip(Q[:, :, 0], 1e-9, None); frac = frac / frac.sum(1, keepdims=True)
    return frac * rho[:, None], st["u2"] + Q[:, :, 1] * (st["u1"] - st["u2"]), np.clip(Q[:, :, 2], 1e-4, None) * st["T2"], np.clip(Q[:, :, 3], 1e-4, None) * st["T2"]


def mixture_fields(n, u, a, b):
    rho = n.sum(1); U = (n * u).sum(1) / rho; d = u - U[:, None]
    T = (n * (a + d ** 2 + 2 * b)).sum(1) / (3 * rho)
    return dict(rho=rho, u=U, T=T, q=(0.5 * n * d * (d ** 2 + 3 * a + 2 * b)).sum(1),
                sigma=((2.0 / 3.0) * n * (a + d ** 2 - b)).sum(1), Phi2=(n * (u ** 3 + 3 * u * a)).sum(1),
                mass=(n * u).sum(1), mom=(n * (u ** 2 + a)).sum(1))


def x_of_nstar(F, nu):
    dxdn = np.gradient(F["Phi2"], GRID) / (-nu * F["sigma"])
    x = np.concatenate([[0.0], np.cumsum(0.5 * (dxdn[1:] + dxdn[:-1]) * np.diff(GRID))])
    return x - np.interp(0.5, GRID, x), dxdn


def extra_checks(F, n, u, a, b, st, ref, xd, xp, dxdn, name):
    """Conserved fluxes of the mixture, errors of the profiles in physical space (predicted positions, both
    solutions aligned at n* = 0.5, nothing else taken from the target), and where dx/dn* fails to be positive."""
    e = (0.5 * n * u * (u ** 2 + 3 * a + 2 * b)).sum(1); e1 = st["u1"] * (0.5 * st["u1"] ** 2 + 2.5)
    out = {f"flux_mass_{name}": float(100 * np.max(np.abs(F["mass"] / st["u1"] - 1))),
           f"flux_mom_{name}": float(100 * np.max(np.abs(F["mom"] / (st["u1"] ** 2 + 1) - 1))),
           f"flux_energy_{name}": float(100 * np.max(np.abs(e / e1 - 1)))}
    if np.all(dxdn > 0):
        lo, hi = max(xd[0], xp[0]), min(xd[-1], xp[-1]); X = np.linspace(lo, hi, 300)
        nd, npr = np.interp(X, xd, GRID), np.interp(X, xp, GRID)
        out[f"xspace_nstar_rms_{name}"] = float(100 * np.sqrt(np.mean((npr - nd) ** 2)))
        for key in ("q", "sigma"):
            fd, fp = np.interp(X, xd, ref[key]), np.interp(X, xp, F[key])
            out[f"xspace_{key}_rms_{name}"] = float(100 * np.sqrt(np.mean((fp - fd) ** 2)) / np.abs(ref[key]).max())
        fd, fp = np.interp(X, xd, ref["T"]), np.interp(X, xp, F["T"])
        out[f"xspace_T_rms_{name}"] = float(100 * np.sqrt(np.mean((fp - fd) ** 2)) / (st["T2"] - st["T1"]))
        out[f"xspace_overlap_{name}"] = float((hi - lo) / (xd[-1] - xd[0]))
    else:
        bad = dxdn <= 0; dPhi = np.gradient(F["Phi2"], GRID)
        out[f"nonmonotone_nstar_{name}"] = [float(GRID[bad].min()), float(GRID[bad].max())]
        out[f"nonmonotone_cause_{name}"] = "dPhi2/dn* changes sign" if np.any(dPhi[bad] >= 0) else "sigma_xx changes sign"
        out[f"nonmonotone_sigma_over_peak_{name}"] = float(F["sigma"][bad].min() / np.abs(F["sigma"]).max())
    return out


def smoothed_monotone(Q, st, nu_of):
    """Is dx/dn* positive when the scaled parameters are smoothed over five grid points in n*?"""
    k = np.ones(5) / 5; Qs = Q.copy()
    for i in range(Q.shape[1]):
        for j in range(Q.shape[2]):
            pad = np.r_[[Q[0, i, j]] * 2, Q[:, i, j], [Q[-1, i, j]] * 2]; Qs[:, i, j] = np.convolve(pad, k, mode="valid")
    n, u, a, b = unscale(Qs, st); F = mixture_fields(n, u, a, b)
    return bool(np.all(x_of_nstar(F, nu_of(F))[1] > 0))


def main(root):
    red, fits = os.path.join(root, "vdf_reduced"), os.path.join(root, "analysis/out/popfits")
    load = lambda t: (sl.load_case(os.path.join(red, f"{t}_BGK_reduced_vdf.npz")), np.load(os.path.join(fits, f"{t}_BGK_branch3.npz")))
    cache = {}
    def get(t):
        if t not in cache:
            c, f = load(t); cache[t] = (c, scaled(f, c["st"]))
        return cache[t]
    out = {}; keep = {}
    for tgt, lo, hi, kind in TESTS:
        (c, Qf), (cl, Ql), (ch, Qh) = get(tgt), get(lo), get(hi)
        st = c["st"]; s = lambda M: 1 / M ** 2
        wH = (s(st["M1"]) - s(cl["st"]["M1"])) / (s(ch["st"]["M1"]) - s(cl["st"]["M1"]))
        Qp = (1 - wH) * Ql + wH * Qh
        v, w = c["vx"], c["wx"]; sw = np.sqrt(w)
        o = np.argsort(c["nstar"])
        ref = {k: np.interp(GRID, c["nstar"][o], c[k][o]) for k in ("q", "sigma", "T", "x")}
        Phi2_dvm = np.interp(GRID, c["nstar"][o], (c["G"] @ (v ** 3 * w))[o])
        xd = ref["x"] - np.interp(0.5, GRID, ref["x"])
        dxdn_d = np.gradient(xd, GRID); delta_d = 1 / np.max(1 / dxdn_d); width = lambda xx: float(np.interp(0.8, GRID, xx) - np.interp(0.2, GRID, xx))
        res = dict(mach=st["M1"], sources=[cl["st"]["M1"], ch["st"]["M1"]], kind=kind, weight_high=float(wH))
        # consistency of the moment equation on the DVM solution itself (nu = 1)
        xm, _ = x_of_nstar(dict(Phi2=Phi2_dvm, sigma=ref["sigma"]), 1.0)
        res["moment_eq_check_x_rms_over_delta"] = float(np.sqrt(np.mean((xm - xd) ** 2)) / delta_d)
        for name, Q in (("pred", Qp), ("fit", Qf)):
            n, u, a, b = unscale(Q, st)
            F = mixture_fields(n, u, a, b)
            xp, dxdn = x_of_nstar(F, 1.0)
            ok = np.all(dxdn > 0)
            delta_p = 1 / np.max(1 / dxdn) if ok else float("nan")
            eg = []
            for i, ns in enumerate(GRID):
                j = int(np.argmin(np.abs(c["nstar"] - ns)))
                p = np.stack([np.log(n[i]), u[i], np.log(a[i]), np.log(b[i])], 1).ravel()
                gp = sl.pop_model(p, v)[0].sum(0)
                eg.append(np.linalg.norm((gp - c["G"][j]) * sw) / np.linalg.norm(c["G"][j] * sw))
                if name == "pred" and (abs(ns - 0.1) < 0.011 or abs(ns - 0.5) < 0.011):
                    keep[(tgt, lo, hi, round(float(ns), 1))] = (c["G"][j], gp)
            res.update({f"g_max_{name}": float(100 * np.max(eg)), f"g_mean_{name}": float(100 * np.mean(eg)),
                        f"q_L2_{name}": float(100 * np.linalg.norm(F["q"] - ref["q"]) / np.linalg.norm(ref["q"])),
                        f"sigma_L2_{name}": float(100 * np.linalg.norm(F["sigma"] - ref["sigma"]) / np.linalg.norm(ref["sigma"])),
                        f"T_max_{name}": float(100 * np.max(np.abs(F["T"] - ref["T"]) / ref["T"])),
                        f"flux_mass_{name}": float(100 * np.max(np.abs(F["mass"] / st["u1"] - 1))),
                        f"flux_mom_{name}": float(100 * np.max(np.abs(F["mom"] / (st["u1"] ** 2 + 1) - 1))),
                        f"monotone_{name}": bool(ok),
                        f"delta_ratio_{name}": float(delta_p / delta_d), f"width_ratio_{name}": width(xp) / width(xd),
                        f"x_rms_over_delta_{name}": float(np.sqrt(np.mean((xp - xd) ** 2)) / delta_d)})
            res.update(extra_checks(F, n, u, a, b, st, ref, xd, xp, dxdn, name))
            if not ok:
                res[f"monotone_smoothed_{name}"] = smoothed_monotone(Q, st, lambda F_: 1.0)
            if name == "pred":
                keep[(tgt, lo, hi, "prof")] = (xd, xp, ref, F, delta_d)
        out[f"{tgt}_from_{lo}_{hi}"] = res
        print(f"{kind:6s} M{st['M1']:g} from {cl['st']['M1']:g},{ch['st']['M1']:g} (w={wH:5.2f}): g max {res['g_max_pred']:5.2f}% (fit {res['g_max_fit']:4.2f}) "
              f"q {res['q_L2_pred']:5.2f}% sigma {res['sigma_L2_pred']:5.2f}% Tmax {res['T_max_pred']:4.2f}% | flux m/p {res['flux_mass_pred']:4.2f}/{res['flux_mom_pred']:4.2f}% | "
              f"width ratio {res['width_ratio_pred']:.3f} (fit {res['width_ratio_fit']:.3f}) monotone {int(res['monotone_pred'])} x rms/delta {res['x_rms_over_delta_pred']:.3f} (fit {res['x_rms_over_delta_fit']:.3f}; DVM eq. check {res['moment_eq_check_x_rms_over_delta']:.3f})", flush=True)
    json.dump(out, open(os.path.join(root, "analysis", "out", "predictor_summary.json"), "w"), indent=1)

    shows = [("M6_gl3fine", "M5", "M8", "Mach 6 from 5 and 8 (interpolation)"), ("M12_gl3medium", "M6_gl3fine", "M8", "Mach 12 from 6 and 8 (extrapolation)")]
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.0), constrained_layout=True)
    for row, (tgt, lo, hi, title) in enumerate(shows):
        c = get(tgt)[0]; st = c["st"]; xi = sl.xi_hat(c["vx"], st)
        xd, xp, ref, F, dl = keep[(tgt, lo, hi, "prof")]
        ax = axs[row, 0]
        ax.plot(xd / dl, GRID, "k-", lw=1.7, label="DVM"); ax.plot(xp / dl, GRID, "C3--", label="three-population model")
        ax.set_xlabel(r"$(x-x_c)/\delta$"); ax.set_ylabel(r"$n^*$"); ax.set_title(title, fontsize=7.5, loc="left")
        ax = axs[row, 1]
        for m, cl_, lab in (("q", "C3", r"$q_x$"), ("sigma", "C0", r"$\sigma_{xx}$")):
            nrm = np.abs(ref[m]).max()
            ax.plot(xd / dl, ref[m] / nrm, "-", color="k" if m == "q" else "0.55", lw=1.7)
            ax.plot(xp / dl, F[m] / nrm, "--", color=cl_, label=lab)
        ax.set_xlabel(r"$(x-x_c)/\delta$"); ax.set_ylabel("moment / peak")
        for col, lv in ((2, 0.1), (3, 0.5)):
            g, gp = keep[(tgt, lo, hi, lv)]; ax = axs[row, col]
            ax.plot(xi, g / g.max(), "k-", lw=1.7); ax.plot(xi, gp / g.max(), "C3--")
            ax.set_xlim(-1.6, 1.9); ax.set_xlabel(r"$\hat\xi$"); ax.set_ylabel(r"$g/\max g$")
            ax.set_title(f"$n^*={lv}$", fontsize=8, pad=3)
        if row == 0:
            from matplotlib.lines import Line2D
            fig.legend(handles=[Line2D([], [], color="k", lw=1.7, label="DVM (grey: $\\sigma_{xx}$)"),
                                Line2D([], [], color="C3", ls="--", label="three-population model, $n^*$, $g$ and $q_x$"),
                                Line2D([], [], color="C0", ls="--", label="three-population model, $\\sigma_{xx}$")],
                       frameon=False, loc="outside upper center", ncol=3, fontsize=6.8, columnspacing=1.4, handlelength=2.0)
    fd = os.path.join(root, "paper", "figures")
    fig.savefig(os.path.join(fd, "fig12_predictor.pdf")); fig.savefig(os.path.join(fd, "fig12_predictor.png"))


if __name__ == "__main__":
    main(sys.argv[1])
