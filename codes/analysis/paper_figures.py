"""Figures and summary tables for the shock-VDF paper.

Usage: python paper_figures.py <shock_data dir>
Reads vdf_reduced/, vdf_models/, models_jfm/ (DSMC), analysis/out/popfits/; writes paper/figures/ and
analysis/out/summary_tables.json.
"""
import sys, os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.7,
                     "lines.linewidth": 1.2, "savefig.dpi": 220, "pdf.fonttype": 42})
BGK = ["M1p5", "M2", "M2p5", "M3", "M4", "M5", "M6_gl3fine", "M8", "M12_gl3medium"]
XI = r"$\hat\xi=(\xi_x-u_2)/(u_1-u_2)$"
POPS = ["hot", "intermediate", "beam"]


def main(root):
    red, mod, fits = (os.path.join(root, d) for d in ("vdf_reduced", "vdf_models", "analysis/out/popfits"))
    figdir = os.path.join(root, "paper", "figures"); os.makedirs(figdir, exist_ok=True)
    C = {t: sl.load_case(os.path.join(red, f"{t}_BGK_reduced_vdf.npz")) for t in BGK}
    F = {t: np.load(os.path.join(fits, f"{t}_BGK_popfit.npz")) for t in BGK}
    F3 = {t: dict(np.load(os.path.join(fits, f"{t}_BGK_branch3.npz"))) for t in BGK}
    MODELS = ["Shakhov_M2", "Shakhov_M3", "Shakhov_M4", "Shakhov_M5", "ESBGK_M2", "ESBGK_M3", "Shakhov_w081_M8"]
    CM = {t: sl.load_case(os.path.join(mod, f"{t}_reduced_vdf.npz")) for t in MODELS}
    FM = {t: np.load(os.path.join(fits, f"{t}_popfit.npz")) for t in MODELS}
    cmap = plt.cm.viridis(np.linspace(0.0, 0.92, len(BGK)))
    Ms = [C[t]["st"]["M1"] for t in BGK]
    T = {}

    def save(fig, name):
        fig.savefig(os.path.join(figdir, name + ".pdf")); fig.savefig(os.path.join(figdir, name + ".png")); plt.close(fig)

    def at(c, lv):
        return int(np.argmin(np.abs(c["nstar"] - lv)))

    def letters(axs, inside=()):
        """Panel letters (a), (b), ... outside the top-left corner of each axes, clear of curves, titles and legends."""
        for i, ax in enumerate(np.ravel(axs)):
            if i in inside:                                   # a legend occupies the space above this panel
                ax.text(0.04, 0.93, "(%s)" % "abcdefgh"[i], transform=ax.transAxes, ha="left", va="top", fontsize=8.5); continue
            ax.annotate("(%s)" % "abcdefgh"[i], xy=(0, 1), xycoords="axes fraction", xytext=(-3, 5), textcoords="offset points",
                        ha="right", va="bottom", fontsize=8.5)

    # ---- Fig 1: maps of g in (x, xi_hat)
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True, constrained_layout=True)
    for ax, t in zip(axs, ["M3", "M5", "M8"]):
        c = C[t]; st = c["st"]
        xc = c["x"][at(c, 0.5)]
        delta = (st["rho2"] - 1) / np.abs(np.gradient(c["rho"], c["x"])).max()
        xg = np.linspace(-2.6, 2.3, 400); xi0 = sl.xi_hat(c["vx"], st)
        Z = np.array([np.interp(xg, xi0, np.log10(np.clip(row / row.max(), 1e-6, None))) for row in c["G"]])
        im = ax.pcolormesh((c["x"] - xc) / delta, xg, np.clip(Z.T, -4, 0),
                           shading="auto", cmap="magma", vmin=-4, vmax=0, rasterized=True)
        ax.set_xlim(-5, 3); ax.set_ylim(-2.5, 2.2); ax.set_title(f"$M={st['M1']:g}$", fontsize=9)
        ax.set_xlabel(r"$(x-x_c)/\delta$")
    axs[0].set_ylabel(XI)
    fig.colorbar(im, ax=axs, label=r"$\log_{10}(g/\max_{\xi} g)$", pad=0.015, fraction=0.04)
    save(fig, "fig1_vdf_maps")

    # ---- Fig 2: g profiles at fixed n*
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.1), sharex=True, sharey=True)
    lv_cols = plt.cm.plasma(np.linspace(0, 0.85, 5))
    for ax, t in zip(axs.flat, ["M2", "M3", "M4", "M5", "M8", "M12_gl3medium"]):
        c = C[t]; st = c["st"]; xi = sl.xi_hat(c["vx"], st)
        for lv, cl in zip([0.05, 0.2, 0.5, 0.8, 0.95], lv_cols):
            s = at(c, lv); ax.plot(xi, c["G"][s] / c["G"][s].max(), color=cl, label=f"$n^*={lv:g}$")
        s = at(c, 0.5)
        A = np.stack([sl.maxwell1d(c["vx"], 1, st["u1"], 1), sl.maxwell1d(c["vx"], st["rho2"], st["u2"], st["T2"])], 1)
        from scipy.optimize import nnls
        co = nnls(A * np.sqrt(c["wx"])[:, None], c["G"][s] * np.sqrt(c["wx"]))[0]
        ax.plot(xi, (A @ co) / c["G"][s].max(), "k--", lw=0.8, label="two RH Maxwellians, $n^*=0.5$")
        ax.set_xlim(-1.6, 1.9); ax.set_title(f"$M={st['M1']:g}$", fontsize=9, pad=3)
    for ax in axs[-1]: ax.set_xlabel(XI)
    for ax in axs[:, 0]: ax.set_ylabel(r"$g/\max g$")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), frameon=False, loc="upper center", ncol=6, fontsize=6.8,
               columnspacing=1.0, handlelength=1.6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.945)); save(fig, "fig2_vdf_profiles")

    # ---- Fig 3: two-population descriptions
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.5), constrained_layout=True)
    for cl, t in zip(cmap, BGK):
        f = F[t]; k = (f["nstar"] > 0.01) & (f["nstar"] < 0.99)
        axs[0].plot(f["nstar"][k], 100 * f["eps_rh2"][k], color=cl, label=f"{f['mach']:g}")
        axs[1].plot(f["nstar"][k], 100 * f["res2"][k, 0], color=cl)
    axs[0].set_ylabel(r"residual of $g$ (%)"); axs[0].set_title("two RH Maxwellians", fontsize=8.5)
    axs[1].set_title("two free populations", fontsize=8.5)
    for ax in axs[:2]: ax.set_xlabel(r"$n^*$")
    h3, l3 = axs[0].get_legend_handles_labels()
    fig.legend(h3, [f"$M={l}$" for l in l3], frameon=False, loc="outside lower center", ncol=9, fontsize=6.8, columnspacing=1.0, handlelength=1.4)
    for K, mk in zip((1, 2, 3, 4), "osD^"):
        axs[2].semilogy(Ms, [100 * np.nanmax(F[t][f"res{K}"][(F[t]["nstar"] > 0.01) & (F[t]["nstar"] < 0.99), 0]) for t in BGK],
                        marker=mk, ms=3.5, label=f"$K={K}$")
    axs[2].semilogy(Ms, [100 * np.nanmax(F[t]["eps_rh2"][(F[t]["nstar"] > 0.01) & (F[t]["nstar"] < 0.99)]) for t in BGK], "k--", marker="x", ms=4, label="RH pair")
    axs[2].set_xlabel("$M$"); axs[2].set_ylabel(r"max residual over shock (%)")
    axs[2].legend(frameon=False, ncol=3, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.01), columnspacing=0.9, handlelength=1.8)
    axs[2].set_xticks([2, 4, 6, 8, 10, 12]); letters(axs); save(fig, "fig3_population_count")
    T["max_resid_pct"] = {t: {**{f"K{K}": float(100 * np.nanmax(F[t][f"res{K}"][(F[t]["nstar"] > 0.01) & (F[t]["nstar"] < 0.99), 0])) for K in (1, 2, 3, 4)},
                               "RHpair": float(100 * np.nanmax(F[t]["eps_rh2"][(F[t]["nstar"] > 0.01) & (F[t]["nstar"] < 0.99)]))} for t in BGK}

    # moment errors of the K-population fits (relative to the profile maximum)
    T["moment_err_pct"] = {}
    for t in BGK:
        f = F[t]; k = (f["nstar"] > 0.01) & (f["nstar"] < 0.99)
        T["moment_err_pct"][t] = {f"K{K}": {m: float(100 * np.nanmax(np.abs(f[f"{m}_fit{K}"][k] - f[m][k])) / float(f[m + "max"]))
                                             for m in ("q", "sigma", "M300", "M400")} for K in (1, 2, 3, 4)}

    # ---- Fig 4: population-count map and moment errors
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.7), constrained_layout=True)
    grid = np.linspace(0.02, 0.98, 49)
    for ax, eps in zip(axs[:2], (0.03, 0.01)):
        Z = np.zeros((len(BGK), len(grid)))
        for i, t in enumerate(BGK):
            f = F[t]; kk = np.full(len(f["nstar"]), 5)
            for K in (4, 3, 2, 1):
                kk[f[f"res{K}"][:, 0] <= eps] = K
            o = np.argsort(f["nstar"])
            Z[i] = np.round(np.interp(grid, f["nstar"][o], kk[o]))
        im = ax.pcolormesh(np.linspace(0.01, 0.99, len(grid) + 1), np.arange(len(BGK) + 1) - 0.5, Z, cmap=plt.get_cmap("YlOrRd", 5), vmin=0.5, vmax=5.5, shading="flat")
        ax.set_yticks(range(len(BGK))); ax.set_yticklabels([f"{m:g}" for m in Ms]); ax.set_xlabel(r"$n^*$")
        ax.set_title(f"$K$ for residual $\\leq${100*eps:g}%", fontsize=8.5)
    axs[0].set_ylabel("$M$")
    cb = fig.colorbar(im, ax=axs[:2], ticks=[1, 2, 3, 4, 5], pad=0.02, fraction=0.05, label="populations $K$"); cb.ax.set_yticklabels(["1", "2", "3", "4", ">4"])
    for m, mk, lab in (("q", "o", r"$q_x$"), ("sigma", "s", r"$\sigma_{xx}$"), ("M400", "^", r"$M_{400}^{\rm neq}$")):
        for K, ls in ((2, "--"), (3, "-")):
            axs[2].semilogy(Ms, [T["moment_err_pct"][t][f"K{K}"][m] for t in BGK], ls, marker=mk, ms=3.5,
                            color={"q": "C0", "sigma": "C1", "M400": "C3"}[m], label=f"{lab}, $K={K}$")
    axs[2].set_xlabel("$M$"); axs[2].set_ylabel("max moment error (% of peak)"); axs[2].legend(frameon=False, fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01), columnspacing=0.8, handlelength=1.6)
    letters(axs, inside=(2,)); save(fig, "fig4_count_map")

    # ---- Fig 5: K = 3 population parameters versus n*
    fig, axs = plt.subplots(4, 3, figsize=(7.0, 7.2), sharex=True)
    sel = ["M3", "M4", "M5", "M6_gl3fine", "M8", "M12_gl3medium"]
    cs = plt.cm.viridis(np.linspace(0, 0.92, len(sel)))
    T["pop3"] = {}
    for cl, t in zip(cs, sel):
        f = F3[t]; st = C[t]["st"]; p = f["p3"].reshape(len(f["nstar"]), 3, 4)
        k = (f["nstar"] > 0.02) & (f["nstar"] < 0.90) & np.isfinite(p[:, 0, 0])
        n = np.exp(p[:, :, 0]); frac = n / n.sum(1, keepdims=True)
        xi = (p[:, :, 1] - st["u2"]) / (st["u1"] - st["u2"])
        a = np.exp(p[:, :, 2]) / st["T2"]; b = np.exp(p[:, :, 3]) / st["T2"]
        for j in range(3):
            for r, y in enumerate((frac[:, j], xi[:, j], a[:, j], b[:, j])):
                if t in ("M3", "M4"):       # downstream of n* = 0.5 the populations overlap: decomposition not unique
                    k1 = k & (f["nstar"] <= 0.5); k2 = k & (f["nstar"] >= 0.5)
                    axs[r, j].plot(f["nstar"][k1], y[k1], color=cl, label=f"{f['mach']:g}")
                    axs[r, j].plot(f["nstar"][k2], y[k2], ":", color=cl, lw=1.0)
                else:
                    axs[r, j].plot(f["nstar"][k], y[k], color=cl, label=f"{f['mach']:g}")
        T["pop3"][t] = {f"n{lv}": [[float(np.interp(lv, f["nstar"][k], y[k, j])) for y in (frac, xi, a, b)] for j in range(3)] for lv in (0.1, 0.5)}
    for j in range(3): axs[0, j].set_title(POPS[j], fontsize=9); axs[3, j].set_xlabel(r"$n^*$")
    for r, lab in enumerate((r"mass fraction $n_k/\rho$", r"mean velocity $\hat\xi_k$", r"$T_{\parallel k}/T_2$", r"$T_{\perp k}/T_2$")):
        axs[r, 0].set_ylabel(lab)
    h5, l5 = axs[0, 1].get_legend_handles_labels()
    fig.legend(h5, [f"$M={l}$" for l in l5], frameon=False, loc="lower center", ncol=6, fontsize=7, columnspacing=1.2, handlelength=1.6, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.4, rect=(0, 0.03, 1, 1)); save(fig, "fig5_population_parameters")

    # ---- Fig 6: transport carried by each population
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.0), sharex=True)
    for col, t in enumerate(["M3", "M5", "M12_gl3medium"]):
        f = F3[t]; k = (f["nstar"] > 0.01) & (f["nstar"] < 0.99)
        for j, cl in zip(range(3), ("C3", "C2", "C0")):
            axs[0, col].plot(f["nstar"][k], f["qk3"][k, j] / f["qmax"], color=cl, label=POPS[j])
            axs[1, col].plot(f["nstar"][k], f["sigmak3"][k, j] / f["sigmamax"], color=cl)
        axs[0, col].plot(f["nstar"][k], f["q"][k] / f["qmax"], "k--", lw=0.9, label="DVM total")
        axs[1, col].plot(f["nstar"][k], f["sigma"][k] / f["sigmamax"], "k--", lw=0.9)
        axs[0, col].set_title(f"$M={f['mach']:g}$", fontsize=9); axs[1, col].set_xlabel(r"$n^*$")
    axs[0, 0].set_ylabel(r"$q_{x,k}/\max|q_x|$"); axs[1, 0].set_ylabel(r"$\sigma_{xx,k}/\max\sigma_{xx}$")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), frameon=False, loc="upper center", ncol=4, fontsize=7,
               columnspacing=1.2, handlelength=1.6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.945)); save(fig, "fig6_transport_by_population")

    # ---- Fig 7: precursor
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
    T["precursor"] = {}
    for cl, t in zip(cmap, BGK):
        c = C[t]; st = c["st"]
        if st["M1"] >= 2.5:
            s_ = at(c, 0.05); axs[0].semilogy(sl.xi_hat(c["vx"], st), c["G"][s_] / c["rho"][s_] * (st["u1"] - st["u2"]), color=cl, label=f"{st['M1']:g}")
    axs[0].set_xlim(-3.5, 2.2); axs[0].set_ylim(1e-5, 5); axs[0].set_xlabel(XI); axs[0].set_ylabel(r"$g\,(u_1-u_2)/\rho$ at $n^*=0.05$")
    h7, l7 = axs[0].get_legend_handles_labels()
    axs[0].legend(h7, [f"$M={l}$" for l in l7], frameon=False, ncol=3, fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 1.01), columnspacing=0.9, handlelength=1.4)

    def decay(c, nu):
        neg = c["vx"] < 0
        beta = (np.clip(c["G"][:, neg], 0, None) @ c["wx"][neg]) / c["rho"]
        ic = at(c, 0.5)
        opt = np.concatenate([[0.0], np.cumsum(0.5 * (nu[1:] + nu[:-1]) * np.diff(c["x"]))])
        return opt[ic] - opt, beta - 0.5 * math.erfc(c["st"]["u1"] / math.sqrt(2.0))

    for cl, t in zip(cmap, BGK):
        c = C[t]; st = c["st"]
        d, be = decay(c, np.ones(len(c["x"])))
        k = (d > 0) & (be > 1e-9)
        sv = (d[k] / np.sqrt(st["T2"])) ** (2 / 3)
        if st["M1"] >= 3:
            # the curve is drawn above five times the difference between the numerical and the analytic upstream value
            fl = abs(float(be[int(np.argmax(d))])); kp = be[k] > 5 * fl
            print(f"   fig7(b) M={st['M1']:g}: upstream offset of beta {fl:.2e}, curve drawn above {5*fl:.2e}")
            axs[1].semilogy(sv[kp], be[k][kp], color=cl)
        kk = (be[k] < 1e-2) & (be[k] > 1e-6)
        if kk.sum() > 5 and st["M1"] >= 3:
            sl_, ic_ = np.polyfit(sv[kk], np.log(be[k][kk]), 1)
            T["precursor"][t] = dict(mach=st["M1"], slope_23=float(sl_), beta_n05=float(np.interp(0.05, c["nstar"], be)), beta_n20=float(np.interp(0.2, c["nstar"], be)))
    for t, ls, lab, om in (("Shakhov_M5", "--", r"Shakhov $\mu\propto T$, $M=5$", 1.0), ("Shakhov_w081_M8", "-.", r"Shakhov $\omega=0.81$, $M=8$", 0.81)):
        c = CM[t]; st = c["st"]
        d, be = decay(c, c["rho"] * c["T"] ** (1 - om)); k = (d > 0) & (be > 1e-9)
        sv = (d[k] / np.sqrt(st["T2"])) ** (2 / 3)
        axs[1].semilogy(sv, be[k], "k" + ls, lw=0.9, label=lab)
        kk = (be[k] < 1e-2) & (be[k] > 1e-6)
        if kk.sum() > 5:
            T["precursor"][t] = dict(mach=st["M1"], slope_23=float(np.polyfit(sv[kk], np.log(be[k][kk]), 1)[0]))
    st = sl.rh_states(12.0); xi_ = -np.logspace(-3, 2.2, 4000) * np.sqrt(st["T2"]); sgrid = np.linspace(0.3, 11, 60)
    hs = [st["rho2"] * abs(np.trapezoid(sl.maxwell1d(xi_, 1.0, st["u2"], st["T2"]) * np.exp(-(sv_ ** 1.5) * np.sqrt(st["T2"]) / np.abs(xi_)), xi_)) for sv_ in sgrid]
    axs[1].semilogy(sgrid, hs, ":", color="C3", lw=1.4, label="half-space model, $M=12$")
    T["precursor"]["half_space_slope"] = float(np.polyfit(sgrid[(np.array(hs) < 1e-2) & (np.array(hs) > 1e-6)], np.log(np.array(hs)[(np.array(hs) < 1e-2) & (np.array(hs) > 1e-6)]), 1)[0])
    axs[1].set_xlabel(r"$s=[\,\theta/\sqrt{T_2}\,]^{2/3}$"); axs[1].set_ylabel(r"excess mass $\beta-\beta_1$")
    axs[1].set_ylim(1e-7, 1); axs[1].set_xlim(0, 11)
    axs[1].legend(frameon=False, fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1)
    c = C["M3"]; st = c["st"]; s_ = at(c, 0.05)
    axs[2].semilogy(sl.xi_hat(c["vx"], st), c["G"][s_] / c["rho"][s_] * (st["u1"] - st["u2"]), "k-", label=r"BGK, constant $\tau$")
    for t, cl, lab in (("Shakhov_M3", "C0", r"Shakhov, $\mu\propto T$"), ("ESBGK_M3", "C3", r"ES-BGK, $\mu\propto T$")):
        c = CM[t]; st = c["st"]; s_ = at(c, 0.05)
        axs[2].semilogy(sl.xi_hat(c["vx"], st), np.clip(c["G"][s_], 1e-12, None) / c["rho"][s_] * (st["u1"] - st["u2"]), color=cl, label=lab)
    axs[2].set_xlim(-3.5, 2.2); axs[2].set_ylim(1e-5, 5); axs[2].set_xlabel(XI); axs[2].text(0.04, 0.92, "$M=3$, $n^*=0.05$", transform=axs[2].transAxes, fontsize=8)
    axs[2].legend(frameon=False, fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1)
    letters(axs); save(fig, "fig7_precursor")

    # ---- Fig 8: model sensitivity
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.5), constrained_layout=True)
    sets = [(3, [("M3", C, "k", "BGK"), ("Shakhov_M3", CM, "C0", "Shakhov"), ("ESBGK_M3", CM, "C3", "ES-BGK")]),
            (5, [("M5", C, "k", "BGK"), ("Shakhov_M5", CM, "C0", "Shakhov")]),
            (8, [("M8", C, "k", "BGK"), ("Shakhov_w081_M8", CM, "C2", r"Shakhov, $\omega=0.81$")])]
    for col, (M, lst) in enumerate(sets):
        for t, D, cl, lab in lst:
            c = D[t]; st = c["st"]; xi = sl.xi_hat(c["vx"], st)
            for lv, ls in ((0.1, "-"), (0.5, "--")):
                s = at(c, lv); axs[0, col].plot(xi, c["G"][s] / c["G"][s].max(), ls, color=cl, label=lab if lv == 0.1 else None)
            k = (c["nstar"] > 0.005) & (c["nstar"] < 0.995)
            axs[1, col].plot(c["nstar"][k], (c["Tpar"] / c["Tperp"])[k], color=cl, label=lab)
        axs[0, col].set_xlim(-1.6, 1.9); axs[0, col].text(0.03, 0.9, f"$M={M}$", transform=axs[0, col].transAxes, fontsize=9)
        axs[0, col].set_xlabel(XI); axs[1, col].set_xlabel(r"$n^*$")
        dp = os.path.join(root, "models_jfm", f"DSMC_M{M}_profiles.npz")
        if os.path.exists(dp):
            z = np.load(dp); ns = (z["rho"] - 1) / (sl.rh_states(M)["rho2"] - 1); k = (ns > 0.01) & (ns < 0.99)
            axs[1, col].plot(ns[k][::6], (z["pxx"] / (0.5 * (z["pyy"] + z["pzz"])))[k][::6], ".", ms=2.5, color="0.45", label=r"DSMC, $\omega=1$")
        axs[0, col].legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, columnspacing=0.8, handlelength=1.5); axs[1, col].legend(frameon=False, fontsize=6.5, loc="upper right", handlelength=1.3, borderaxespad=0.3, labelspacing=0.3)   # clear of the curves
    axs[0, 0].set_ylabel(r"$g/\max g$"); axs[1, 0].set_ylabel(r"$T_\parallel/T_\perp$")
    save(fig, "fig8_model_sensitivity")

    # model comparison table: residuals and K=3 populations at n*=0.1
    T["models"] = {}
    for t, (c, f) in {**{k: (C[k], F[k]) for k in ("M2", "M3", "M4", "M5", "M8")}, **{k: (CM[k], FM[k]) for k in MODELS}}.items():
        st = c["st"]; k = (f["nstar"] > 0.01) & (f["nstar"] < 0.99); p = f["p3"].reshape(len(f["nstar"]), 3, 4)
        ok = k & np.isfinite(p[:, 0, 0]); n = np.exp(p[:, :, 0]); frac = n / n.sum(1, keepdims=True)
        xi = (p[:, :, 1] - st["u2"]) / (st["u1"] - st["u2"])
        neg = c["vx"] < 0; beta = (c["G"][:, neg] @ c["wx"][neg]) / c["rho"]
        T["models"][t] = dict(model=c["model"], mach=st["M1"], TparTperp_max=float((c["Tpar"] / c["Tperp"]).max()),
                              rh_pair_max=float(100 * np.nanmax(f["eps_rh2"][k])),
                              **{f"K{K}_max": float(100 * np.nanmax(f[f"res{K}"][k, 0])) for K in (1, 2, 3, 4)},
                              hot_frac_n10=float(np.interp(0.1, f["nstar"][ok], frac[ok, 0])), hot_xi_n10=float(np.interp(0.1, f["nstar"][ok], xi[ok, 0])),
                              mid_frac_n10=float(np.interp(0.1, f["nstar"][ok], frac[ok, 1])), mid_xi_n10=float(np.interp(0.1, f["nstar"][ok], xi[ok, 1])),
                              beta_n05=float(np.interp(0.05, c["nstar"], beta)), beta_n20=float(np.interp(0.2, c["nstar"], beta)),
                              bimodal_center=bool(_bimodal(c, 0.3, 0.7)))
    T["bimodal_range"] = {t: _bimodal_range(C[t]) for t in BGK}
    json.dump(T, open(os.path.join(root, "analysis", "out", "summary_tables.json"), "w"), indent=1)
    print(json.dumps({k: T[k] for k in ("bimodal_range", "precursor")}))


def _peaks(g):
    from scipy.signal import find_peaks
    return len(find_peaks(np.r_[0.0, g, 0.0], prominence=1e-3 * g.max())[0])


def _bimodal(c, lo, hi):
    k = np.where((c["nstar"] > lo) & (c["nstar"] < hi))[0]
    return any(_peaks(c["G"][s]) >= 2 for s in k[::max(1, len(k) // 60)])


def _bimodal_range(c):
    k = np.where((c["nstar"] > 0.02) & (c["nstar"] < 0.98))[0]
    b = np.array([_peaks(c["G"][s]) >= 2 for s in k]); ns = c["nstar"][k]
    if not b.any():
        return None
    up = ns[b & (ns < 0.3)]; ce = ns[b & (ns >= 0.3)]
    return dict(upstream=[float(up.min()), float(up.max())] if len(up) else None,
                center=[float(ce.min()), float(ce.max())] if len(ce) else None)


if __name__ == "__main__":
    main(sys.argv[1])
