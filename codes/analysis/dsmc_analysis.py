"""Analysis of the DSMC velocity distributions (hard sphere and VHS) against the relaxation models.

Usage: python dsmc_analysis.py <shock_data dir>
Reads vdf_dsmc/, vdf_reduced/, vdf_models/, models_jfm/DSMC_M*_profiles.npz, analysis/out/popfits/.
Writes paper/figures/fig10_dsmc_distributions.*, fig11_dsmc_precursor.* and analysis/out/dsmc_summary.json.
"""
import sys, os, json, math
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 6.8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.7, "lines.linewidth": 1.2,
                     "savefig.dpi": 220, "pdf.fonttype": 42})
XI = r"$\hat\xi=(\xi_x-u_2)/(u_1-u_2)$"
OM = [("0p5", 0.5, "C3", "DSMC, hard sphere"), ("0p81", 0.81, "C1", r"DSMC, $\omega=0.81$"), ("1", 1.0, "C0", r"DSMC, $\omega=1$")]


def at(c, lv):
    return int(np.argmin(np.abs(c["nstar"] - lv)))


def avg_g(c, lv, half=0.01):
    """g at n* = lv, normalised by rho: average over the cells with |n* - lv| < half where there are
    at least two of them, otherwise linear interpolation in n* between the two bracketing cells
    (thin hard-sphere shocks, where no cell centre may lie within the window)."""
    k = np.abs(c["nstar"] - lv) < half
    if k.sum() >= 2:
        return c["G"][k].sum(0) / c["rho"][k].sum()
    core = np.where((c["nstar"] > 0.005) & (c["nstar"] < 0.995))[0]
    idx = np.arange(core.min() - 1, core.max() + 2)
    j = idx[np.argmax(c["nstar"][idx] >= lv)]
    a = (lv - c["nstar"][j - 1]) / (c["nstar"][j] - c["nstar"][j - 1])
    return ((1 - a) * c["G"][j - 1] + a * c["G"][j]) / ((1 - a) * c["rho"][j - 1] + a * c["rho"][j])


def smooth(y, n=3):
    return np.convolve(y, np.ones(n) / n, mode="same")


def phi_star(AN, omega, T2):
    """Saddle-point exponent for molecules of the downstream Maxwellian crossing a column density N
    against a loss rate n A c^(2-2 omega): min over y of y^2/2 + a / y^m, with m = 2 omega - 1 and
    a = A N T2^(-m/2). For m = 1 this is (3/2) a^(2/3); for m -> 0 (hard sphere) it tends to a."""
    m = 2 * omega - 1
    a = AN * T2 ** (-m / 2)
    if m < 1e-6:
        return a
    return (0.5 + 1 / m) * (a * m) ** (2 / (m + 2))


def lam_star(AN, omega, st):
    """Survival exponent of a molecule that moves upstream with streamwise speed w through a column AN of the
    stream, which it meets at the relative speed w + u1, weighted with the downstream Maxwellian drifting at u2:
    min over w of (w + u2)^2 / 2 T2 + AN (w + u1)^(2 - 2 omega) / w   (golden-section search, vectorised)."""
    AN = np.atleast_1d(np.asarray(AN, float)); u1, u2, T2 = st["u1"], st["u2"], st["T2"]; ex = 2 - 2 * omega
    f = lambda w: (w + u2) ** 2 / (2 * T2) + AN * (w + u1) ** ex / w
    a = np.full(AN.shape, 1e-3); b = np.full(AN.shape, 80 * math.sqrt(T2)); gr = (math.sqrt(5) - 1) / 2
    for _ in range(80):
        c = b - gr * (b - a); d = a + gr * (b - a); left = f(c) < f(d)
        b = np.where(left, d, b); a = np.where(left, a, c)
    return f(0.5 * (a + b))


def main(root):
    figdir = os.path.join(root, "paper", "figures")
    dd = os.path.join(root, "vdf_dsmc")
    D = {}
    for tag, om, _, _ in OM:
        for M in (2, 3, 4, 5, 6, 8, 10, 12):
            p = os.path.join(dd, f"DSMC_w{tag}_M{M}_reduced_vdf.npz")
            if os.path.exists(p):
                c = sl.load_case(p); z = np.load(p)
                c["direct"] = {k[7:]: z[k] for k in z.files if k.startswith("direct_")}
                c["count_scale"] = float(z["count_scale"])
                c["nstar"] = (c["direct"]["rho"] - 1) / (c["st"]["rho2"] - 1)
                D[(om, M)] = c
    REL = {("BGK", 3): sl.load_case(os.path.join(root, "vdf_reduced", "M3_BGK_reduced_vdf.npz")),
           ("BGK", 5): sl.load_case(os.path.join(root, "vdf_reduced", "M5_BGK_reduced_vdf.npz")),
           ("BGK", 8): sl.load_case(os.path.join(root, "vdf_reduced", "M8_BGK_reduced_vdf.npz")),
           ("BGK", 12): sl.load_case(os.path.join(root, "vdf_reduced", "M12_gl3medium_BGK_reduced_vdf.npz")),
           ("Shakhov", 3): sl.load_case(os.path.join(root, "vdf_models", "Shakhov_M3_reduced_vdf.npz")),
           ("Shakhov", 5): sl.load_case(os.path.join(root, "vdf_models", "Shakhov_M5_reduced_vdf.npz")),
           ("Shakhov081", 8): sl.load_case(os.path.join(root, "vdf_models", "Shakhov_w081_M8_reduced_vdf.npz")),
           ("ESBGK", 3): sl.load_case(os.path.join(root, "vdf_models", "ESBGK_M3_reduced_vdf.npz"))}
    S = {"validation": {}, "cases": {}, "precursor": {}}

    # ---- validation against the earlier DSMC (Bird's DS1, VHS omega = 1): moments only
    for M in (2, 3, 5):
        p = os.path.join(root, "models_jfm", f"DSMC_M{M}_profiles.npz")
        if (1.0, M) not in D or not os.path.exists(p):
            continue
        z = np.load(p); st = sl.rh_states(M); c = D[(1.0, M)]; d = c["direct"]
        ns_b = (z["rho"] - 1) / (st["rho2"] - 1); r_b = z["pxx"] / (0.5 * (z["pyy"] + z["pzz"]))
        grid = np.linspace(0.03, 0.97, 48)
        ob = np.argsort(ns_b); om_ = np.argsort(c["nstar"])
        rb = np.interp(grid, ns_b[ob], smooth(r_b, 9)[ob]); rm = np.interp(grid, c["nstar"][om_], (d["Tpar"] / d["Tperp"])[om_])
        qb = np.interp(grid, ns_b[ob], z["q"][ob]); qm = np.interp(grid, c["nstar"][om_], d["q"][om_])
        S["validation"][f"M{M}"] = dict(peak_this=float(rm.max()), peak_bird=float(rb.max()),
                                         max_abs_diff_TparTperp=float(np.abs(rm - rb).max()),
                                         qmin_this=float(qm.min()), qmin_bird=float(qb.min()),
                                         q_shape_L2_pct=float(100 * np.linalg.norm(qm / np.abs(qm).max() - qb / np.abs(qb).max()) / np.linalg.norm(qb / np.abs(qb).max())))

    # ---- per-case summary
    for (om, M), c in sorted(D.items()):
        st = c["st"]; d = c["direct"]; x = c["x"]
        delta = (st["rho2"] - 1) / np.abs(np.gradient(smooth(d["rho"], 9), x))[10:-10].max()
        neg = c["vx"] < 0; beta = (c["G"][:, neg] @ c["wx"][neg]) / c["rho"]
        o = np.argsort(c["nstar"])
        eps = sl.two_maxwellian_residual(c)
        core = (c["nstar"] > 0.02) & (c["nstar"] < 0.98)
        cnt = c["G"] * c["count_scale"]
        noise = np.sqrt(cnt.sum(1)) / np.sqrt((cnt ** 2).sum(1))
        bim_c = any(len(find_peaks(np.r_[0, smooth(avg_g(c, lv), 5), 0], prominence=0.03 * avg_g(c, lv).max())[0]) >= 2 for lv in np.arange(0.3, 0.75, 0.05))
        bim_u = any(len(find_peaks(np.r_[0, smooth(avg_g(c, lv), 5), 0], prominence=0.005 * avg_g(c, lv).max())[0]) >= 2 for lv in (0.03, 0.05, 0.1, 0.15))
        rec = dict(omega=om, mach=M, delta=float(delta), TparTperp_max=float(smooth(d["Tpar"] / d["Tperp"], 5).max()),
                   rh_pair_max=float(100 * eps[core].max()), rh_pair_n=float(c["nstar"][core][np.argmax(eps[core])]),
                   noise_floor_pct=float(100 * np.median(noise[core])),
                   beta_n05=float(np.interp(0.05, c["nstar"][o], beta[o])), beta_n20=float(np.interp(0.2, c["nstar"][o], beta[o])),
                   bimodal_center=bool(bim_c), bimodal_upstream=bool(bim_u))
        fp = os.path.join(root, "analysis", "out", "popfits", f"{c['name']}_popfit.npz")
        if os.path.exists(fp):
            f = np.load(fp); k = (f["nstar"] > 0.02) & (f["nstar"] < 0.98)
            for K in (1, 2, 3, 4):
                rec[f"K{K}_max"] = float(100 * np.nanmax(f[f"res{K}"][k, 0]))
                for m in ("q", "sigma", "M400"):
                    rec[f"K{K}_{m}"] = float(100 * np.nanmax(np.abs(f[f"{m}_fit{K}"][k] - f[m][k])) / float(f[m + "max"]))
            p = f["p3"].reshape(len(f["nstar"]), 3, 4); ok = np.isfinite(p[:, 0, 0]); oo = np.argsort(f["nstar"][ok])
            n = np.exp(p[ok][oo][:, :, 0]); frac = n / n.sum(1, keepdims=True)
            xi = (p[ok][oo][:, :, 1] - st["u2"]) / (st["u1"] - st["u2"])
            a = np.exp(p[ok][oo][:, :, 2]) / st["T2"]; b = np.exp(p[ok][oo][:, :, 3]) / st["T2"]
            rec["pop3_n10"] = [[float(np.interp(0.1, f["nstar"][ok][oo], y[:, j])) for y in (frac, xi, a, b)] for j in range(3)]
        S["cases"][f"w{om:g}_M{M}"] = rec

    # ---- Fig 10: distributions and anisotropy
    fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.9), constrained_layout=True, sharey="row")
    relsets = {3: [("BGK", "k", "-", r"BGK, const. $\tau$"), ("Shakhov", "0.45", "--", r"Shakhov")],
               5: [("BGK", "k", "-", r"BGK, const. $\tau$"), ("Shakhov", "0.45", "--", r"Shakhov")],
               8: [("BGK", "k", "-", r"BGK, const. $\tau$"), ("Shakhov081", "0.45", "--", r"Shakhov")],
               12: [("BGK", "k", "-", r"BGK, const. $\tau$")]}
    for col, M in enumerate((3, 5, 8, 12)):
        for row, lv in enumerate((0.1, 0.5)):
            ax = axs[row, col]
            for key, cl, ls, lab in relsets[M]:
                c = REL[(key, M)]; g = avg_g(c, lv, 0.004)
                ax.semilogy(sl.xi_hat(c["vx"], c["st"]), np.clip(g, 1e-9, None) * (c["st"]["u1"] - c["st"]["u2"]), ls, color=cl, lw=1.0, label=lab,
                            marker="." if M == 12 else None, ms=2.5)
            for tag, om, cl, lab in OM:
                if (om, M) in D:
                    c = D[(om, M)]; g = avg_g(c, lv)
                    ax.semilogy(sl.xi_hat(c["vx"], c["st"]), np.clip(g, 1e-9, None) * (c["st"]["u1"] - c["st"]["u2"]), color=cl, lw=1.1, label=lab)
            ax.set_xlim(-3.2, 2.2); ax.set_ylim(2e-4, 4); ax.text(0.04, 0.91, f"$M={M}$, $n^*={lv}$", transform=ax.transAxes, fontsize=7)
            if row == 1: ax.set_xlabel(XI)
            if col == 0: ax.set_ylabel(r"$g\,(u_1-u_2)/\rho$")
        ax = axs[2, col]
        for key, cl, ls, lab in relsets[M]:
            c = REL[(key, M)]; k = (c["nstar"] > 0.005) & (c["nstar"] < 0.995)
            ax.plot(c["nstar"][k], (c["Tpar"] / c["Tperp"])[k], ls, color=cl, lw=1.0)
        for tag, om, cl, lab in OM:
            if (om, M) in D:
                c = D[(om, M)]; d = c["direct"]; k = (c["nstar"] > 0.005) & (c["nstar"] < 0.995)
                ax.plot(c["nstar"][k], smooth(d["Tpar"] / d["Tperp"], 5)[k], color=cl, lw=1.1)
        ax.set_xlabel(r"$n^*$")
        if col == 0: ax.set_ylabel(r"$T_\parallel/T_\perp$")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), frameon=False, loc="outside upper center", ncol=5, columnspacing=0.9, handlelength=1.6)
    fig.savefig(os.path.join(figdir, "fig10_dsmc_distributions.pdf")); fig.savefig(os.path.join(figdir, "fig10_dsmc_distributions.png")); plt.close(fig)

    # ---- Fig 11: decay of the backward-moving mass
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9), constrained_layout=True)
    LS = {5: "--", 8: "-", 12: ":"}
    for (om, M), c in sorted(D.items()):
        if M < 5:
            continue
        st = c["st"]; d = c["direct"]; x = c["x"]
        neg = c["vx"] < 0; beta = (c["G"][:, neg] @ c["wx"][neg]) / c["rho"]
        ic = at(c, 0.5); col = np.concatenate([[0.0], np.cumsum(0.5 * (d["rho"][1:] + d["rho"][:-1]) * np.diff(x))])
        Ncol = col[ic] - col
        k = (Ncol > 0) & (beta > 3e-7) & (x > x[0] + 5.0)          # leave out the cells next to the inflow reservoir
        cl = {0.5: "C3", 0.81: "C1", 1.0: "C0"}[om]
        sref = 15 * math.sqrt(math.pi) / (2 * (5 - 2 * om) * (7 - 2 * om)); A = sref * 4 ** (om - 0.5) / math.gamma(2.5 - om)
        svar = lam_star(A * np.maximum(Ncol, 0), om, st)
        sold = phi_star(A * np.maximum(Ncol, 0), om, st["T2"])
        if M in LS:
            axs[0].semilogy(Ncol[k], beta[k], LS[M], color=cl)
            axs[1].semilogy(svar[k], beta[k], LS[M], color=cl)
        rec = dict(power_predicted=2 / (2 * om + 1), sigma_ref=float(sref), beta_at_inflow=float(beta[50]))
        rec["column_to_beta_1e-5"] = float(np.interp(-5.0, np.log10(beta[k]), Ncol[k])) if beta[k].min() < 1e-5 else None
        for name, lo in (("", 2e-5), ("_narrow", 2e-4)):
            kk = k & (beta < 1e-2) & (beta > lo)
            if kk.sum() > 8:
                xs_ = Ncol[kk]; lb = np.log(beta[kk]); rms = lambda z: float(np.sqrt(np.mean((np.polyval(np.polyfit(z, lb, 1), z) - lb) ** 2)))
                rec.update({"slope_vs_column" + name: float(np.polyfit(xs_, lb, 1)[0]), "rms_linear" + name: rms(xs_),
                            "rms_two_thirds" + name: rms(xs_ ** (2 / 3)), "rms_predicted_power" + name: rms(xs_ ** (2 / (2 * om + 1))),
                            "slope_vs_lambda" + name: float(np.polyfit(svar[kk], lb, 1)[0]), "rms_lambda" + name: rms(svar[kk]),
                            "slope_vs_phi" + name: float(np.polyfit(sold[kk], lb, 1)[0]), "rms_phi" + name: rms(sold[kk]),
                            "x_upstream_end" + name: float(x[kk].min()), "column_max" + name: float(xs_.max())})
        S["precursor"][f"w{om:g}_M{M}"] = rec
    for M in (8, 12):
        c = REL[("BGK", M)]; st = c["st"]; neg = c["vx"] < 0; beta = (c["G"][:, neg] @ c["wx"][neg]) / c["rho"]
        ic = at(c, 0.5); opt = np.concatenate([[0.0], np.cumsum(np.diff(c["x"]))]); th = opt[ic] - opt; k = (th > 0) & (beta > 3e-7)
        sv = lam_star(np.maximum(th, 0), 1.0, st)                     # constant collision frequency: the column is the optical distance
        sold = 1.5 * (th / math.sqrt(st["T2"])) ** (2 / 3)
        axs[1].semilogy(sv[k], beta[k], LS[M], color="k", lw=1.0)
        kk = k & (beta < 1e-2) & (beta > 2e-5)
        S["precursor"][f"BGK_M{M}"] = dict(slope_vs_lambda=float(np.polyfit(sv[kk], np.log(beta[kk]), 1)[0]), slope_vs_phi=float(np.polyfit(sold[kk], np.log(beta[kk]), 1)[0]))
    xx = np.linspace(0, 20, 20); axs[1].semilogy(xx, 0.5 * np.exp(-xx), "-", color="0.75", lw=0.8)
    hc = [Line2D([], [], color=cl, label=lab) for _, _, cl, lab in OM] + [Line2D([], [], color="k", label=r"BGK, const. $\tau$")]
    hm = [Line2D([], [], color="0.3", ls=LS[M], label=f"$M={M}$") for M in (5, 8, 12)]
    axs[0].set_xlabel(r"column density $N=\int_x^{x_c}n\,dx'$"); axs[0].set_ylabel(r"upstream-moving mass $\beta$")
    axs[0].set_ylim(3e-6, 1); axs[0].set_xlim(0, 110); axs[0].legend(handles=hm, frameon=False, fontsize=6.3, loc="upper right")
    axs[1].set_xlabel(r"survival exponent $\Lambda(N)$"); axs[1].set_ylim(3e-6, 1); axs[1].set_xlim(0, 20)
    fig.legend(handles=hc + [Line2D([], [], color="0.75", lw=0.8, label=r"$\propto e^{-\Lambda}$")], frameon=False, fontsize=6.8, loc="outside upper center", ncol=5, columnspacing=1.0, handlelength=1.6)
    M = 12
    for tag, om, cl, lab in OM:
        if (om, M) in D:
            c = D[(om, M)]; g = avg_g(c, 0.05)
            axs[2].semilogy(sl.xi_hat(c["vx"], c["st"]), np.clip(g, 1e-9, None) * (c["st"]["u1"] - c["st"]["u2"]), color=cl, label=lab)
    c = REL[("BGK", M)]; axs[2].semilogy(sl.xi_hat(c["vx"], c["st"]), avg_g(c, 0.05, 0.004) * (c["st"]["u1"] - c["st"]["u2"]), "k.-", lw=0.8, ms=2.5, label=r"BGK, const. $\tau$")
    axs[2].set_xlim(-3.2, 2.2); axs[2].set_ylim(2e-4, 4); axs[2].set_xlabel(XI); axs[2].text(0.03, 0.92, f"$M={M}$, $n^*=0.05$", transform=axs[2].transAxes, fontsize=8)
    for ax, t in zip(axs, "abc"):
        ax.set_title(f"({t})", loc="left", fontsize=8, pad=2)
    fig.savefig(os.path.join(figdir, "fig11_dsmc_precursor.pdf")); fig.savefig(os.path.join(figdir, "fig11_dsmc_precursor.png")); plt.close(fig)

    json.dump(S, open(os.path.join(root, "analysis", "out", "dsmc_summary.json"), "w"), indent=1)
    print("VALIDATION", json.dumps(S["validation"]))
    for kname, r in S["precursor"].items():
        print("PRECURSOR", kname, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()})
    hdr = ["delta", "TparTperp_max", "rh_pair_max", "rh_pair_n", "noise_floor_pct", "K1_max", "K2_max", "K3_max", "K2_q", "K3_q", "K2_M400", "K3_M400", "beta_n05", "beta_n20"]
    print(f"{'case':10s} " + " ".join(f"{h[:9]:>9s}" for h in hdr) + "  bim(c/u)  pop3@0.1")
    for k, r in S["cases"].items():
        print(f"{k:10s} " + " ".join(f"{r.get(h, float('nan')):9.3f}" for h in hdr) + f"  {int(r['bimodal_center'])}/{int(r['bimodal_upstream'])}  "
              + ("  ".join("[" + ",".join(f"{v:.2f}" for v in pop) + "]" for pop in r["pop3_n10"]) if "pop3_n10" in r else ""))


if __name__ == "__main__":
    main(sys.argv[1])
