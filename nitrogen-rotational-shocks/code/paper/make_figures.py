"""Figures of the letter on the rotational populations of a nitrogen shock wave.

  fig1  one collision at the shock front (classical trajectories against the Larsen-Borgnakke rule)
  fig2  measured hot rotational group against the collision history in DSMC
  fig3  rotational-energy distribution inside the shock, split by collision generation, with the measured levels
  fig4  upstream precursor: decay against the linear recoil cascade, and composition

Usage: python make_figures.py <n2 data folder> <output folder> [ctc=<suffix of the trajectory set, default M12.9>] [ctc7=<default M7>]"""
import sys, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

N2 = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True); sys.path.insert(0, N2)
import robben_talbot_1966 as rt
opt = dict(a.split("=") for a in sys.argv[3:]); S129 = opt.get("ctc", "M12.9"); S7 = opt.get("ctc7", "M7")
RIN = opt.get("src", "v17") == "rin"      # main runs: stream entering with the measured free-stream rotational distribution
F_ERH = "erh_rin.npz" if RIN else "erh_v17.npz"; F_FIRST = "first_rin.npz" if RIN else "first_v17.npz"; F_LB = "erh_rin.npz" if RIN else "erh_ref.npz"
F_PRE = "letter_precursor_rin.npz" if RIN else "letter_precursor.npz"; F_PREK = "letter_precursor_k_rin.npz" if RIN else "letter_precursor_k.npz"
plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 10, "legend.fontsize": 9, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.7, "lines.linewidth": 1.5, "xtick.direction": "in", "ytick.direction": "in",
                     "xtick.top": True, "ytick.right": True, "legend.handlelength": 2.2, "legend.columnspacing": 1.2, "legend.handletextpad": 0.5})
BLUE, RED, ORANGE, GREEN, GREY = "#1F4E79", "#C0392B", "#E08A2B", "#2E8B57", "#7F7F7F"
W1, W2 = 3.4, 7.0


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png"), dpi=300); plt.close(fig); print("written", name)


def tag(a, s, x=0.035, y=0.9):
    a.text(x, y, s, transform=a.transAxes, fontsize=10.5, va="center")


# ---------------------------------------------------------------- figure 1: one collision (version after the outside review)
def fig1():
    d = np.load(os.path.join(N2, "ctc_cluster", f"cond_{S129}.npz")); S, fS, et, ed = d["S"], d["bin_fS"], d["bin_et"], d["et_edges"]
    use = [j for j in range(len(et)) if np.isfinite(et[j]) and 2 <= ed[j] < 10]; cm = plt.get_cmap("viridis"); cols = [cm(0.92 * q / (len(use) - 1)) for q in range(len(use))]
    fig, ax = plt.subplots(2, 1, figsize=(W1, 4.0))
    for j, c in zip(use, cols):
        m = (S >= 0.01) & (S <= 0.6) & (fS[j] > 0); ax[0].semilogy(S[m], fS[j][m], "-", color=c, lw=1.4, label=f"{ed[j]:.0f}-{ed[j + 1]:.0f}")
    s0 = float(np.mean([0.2 / math.log(fS[j][20] / fS[j][60]) for j in use])); a0 = float(np.mean([fS[j][20] * math.exp(0.1 / s0) for j in use]))
    x = np.linspace(0, 0.6, 50); ax[0].semilogy(x, a0 * np.exp(-x / s0), "k--", lw=1.3, dashes=(4, 2), label=rf"${a0:.2f}\,e^{{-x/{s0:.2f}}}$")
    ax[0].set_xlim(0, 0.6); ax[0].set_ylim(4e-3, 2); ax[0].set_xlabel(r"share of the pair energy, $x$", labelpad=1); ax[0].set_ylabel(r"$\sigma(e_{\rm rot}'>xE_{\rm tot})\,/\,\sigma_{\rm VHS}$")
    ax[0].legend(loc="lower left", frameon=False, fontsize=8.2, ncol=2, borderaxespad=0.3, labelspacing=0.2, columnspacing=0.8, handlelength=1.5, title=r"$E_t/kT_2$", title_fontsize=8.6)
    tag(ax[0], "(a)", x=0.88, y=0.9)
    print(f"   fig1a: share law {a0:.3f} exp(-x/{s0:.3f}); per bin " + " ".join(f"{0.2 / math.log(fS[j][20] / fS[j][60]):.3f}" for j in use))
    F = np.load(os.path.join(N2, F_FIRST)); E = np.arange(201) * 0.05
    for lab, col, ls, name in (("M7k", BLUE, "-", "trajectory kernel, Mach 7.0"), ("M12.9k", BLUE, "--", "trajectory kernel, Mach 12.9"),
                               ("M7lb", RED, "-", r"LB rule $p(E_c)$, Mach 7.0"), ("M12.9lb", RED, "--", r"LB rule $p(E_c)$, Mach 12.9")):
        f = F[f"{lab}_first"]; c = np.concatenate([f[0][::-1].cumsum()[::-1], [0.0]]) / f[2].sum(); m = (E <= 4.0) & (c > 0)
        ax[1].semilogy(E[m], c[m], ls, color=col, lw=1.6 if ls == "-" else 1.3, label=name)
        print(f"   fig1b {lab}: above 0.3, 1, 2, 3 kT2: " + " ".join(f"{c[int(round(v / 0.05))]:.4f}" for v in (0.3, 1, 2, 3)) + f"; slope {float(F[lab + '_slope']):.3f}, amplitude {float(F[lab + '_amp']):.3f}, mean {float(F[lab + '_mean']):.3f}")
    sl = 0.5 * (float(F["M7k_slope"]) + float(F["M12.9k_slope"])); am = 0.5 * (float(F["M7k_amp"]) + float(F["M12.9k_amp"])); e = np.linspace(0.5, 4, 20)
    ax[1].semilogy(e, am * np.exp(-e / sl), "-", color=GREY, lw=3.4, alpha=0.4, zorder=1, label=rf"${am:.2f}\,\exp(-E/{sl:.2f}\,kT_2)$")
    ax[1].set_xlim(0, 4); ax[1].set_ylim(1e-4, 40); ax[1].set_xlabel(r"$E/kT_2$", labelpad=1); ax[1].set_ylabel(r"$P(e_{\rm rot}'>E)$")
    ax[1].legend(loc="upper right", frameon=False, fontsize=7.9, borderaxespad=0.25, labelspacing=0.18, handlelength=1.8); tag(ax[1], "(b)", x=0.04, y=0.92)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.983, bottom=0.098, hspace=0.32); save(fig, "fig1")


# ---------------------------------------------------------------- (before the outside review) one collision of the precursor ensemble
def fig1_old():
    d = np.load(os.path.join(N2, "ctc_cluster", f"exc_{S129}.npz")); d7 = np.load(os.path.join(N2, "ctc_cluster", f"exc_{S7}.npz")); a = np.load(os.path.join(N2, "ctc_cluster", f"an_{S129}.npz"))
    E = d["E"]; sv = float(d["sig_vhs"]); sv7 = float(d7["sig_vhs"])
    fig, ax = plt.subplots(2, 1, figsize=(W1, 4.3))
    m = (E <= 5.0) & (E >= 0.05); i1 = int(np.argmin(np.abs(E - 1.0))); i3 = int(np.argmin(np.abs(E - 3.0)))
    fit = [(2.0 / math.log(z["ctc"][i1] / z["ctc"][i3]), s_) for z, s_ in ((d, sv), (d7, sv7))]                # slope temperature between 1 and 3 kT2
    amp = [z["ctc"][i1] / s_ * math.exp(1.0 / t) for (t, s_), z in zip(fit, (d, d7))]
    th = 0.5 * (fit[0][0] + fit[1][0]); A = 0.5 * (amp[0] + amp[1])                                             # mean of the two Mach numbers
    print(f"   fits: Mach 12.9 amplitude {amp[0]:.3f}, temperature {fit[0][0]:.3f} T2; Mach 7 amplitude {amp[1]:.3f}, temperature {fit[1][0]:.3f} T2")
    ax[0].semilogy(E[m], A * np.exp(-E[m] / th), "-", color=GREY, lw=3.6, alpha=0.45, zorder=1)
    ax[0].semilogy(E[m], d["ctc"][m] / sv, "-", color=BLUE, lw=1.7, zorder=3)
    k = np.where(m & (E >= 0.3))[0][3::9]; ax[0].semilogy(E[k], d7["ctc"][k] / sv7, "o", color=BLUE, mfc="white", ms=3.6, mew=0.9, zorder=4)
    ax[0].semilogy(E[m], d["boyd"][m] / sv, "-.", color=ORANGE, lw=1.5, zorder=2); ax[0].semilogy(E[m], d["lb"][m] / sv, "--", color=RED, lw=1.5, zorder=2)
    ax[0].set_xlim(0, 5); ax[0].set_ylim(3e-5, 2.2); ax[0].set_xlabel(r"$E/kT_2$", labelpad=1); ax[0].set_ylabel(r"$\sigma(e_{\rm rot}'>E)\,/\,\sigma_{\rm VHS}$")
    tag(ax[0], "(a)", x=0.88, y=0.9)
    hh = [Line2D([], [], color=BLUE, lw=1.7), Line2D([], [], color=BLUE, marker="o", mfc="white", ms=3.6, mew=0.9, ls=""), Line2D([], [], color=GREY, lw=3.6, alpha=0.45),
          Line2D([], [], color=ORANGE, ls="-.", lw=1.5), Line2D([], [], color=RED, ls="--", lw=1.5)]
    ll = ["trajectories, Mach 12.9", "trajectories, Mach 7.0", rf"${A:.2f}\,\exp(-E/{th:.2f}\,kT_2)$", r"LB rule, $p(E_c)$", r"LB rule, $p=0.2$"]
    ax[0].legend(hh, ll, loc="lower left", frameon=False, fontsize=8.6, borderaxespad=0.25, labelspacing=0.25)
    c = a["cos"]; ax[1].plot(c, a["me1"], "-", color=BLUE, lw=1.7, label=r"$\langle e_{\rm rot}'\rangle/kT_2$"); ax[1].plot(c, a["p01"], "--", color=GREEN, lw=1.7, label=r"$P(e_{\rm rot}'>0.1\,kT_2)$")
    ax[1].set_xlim(-1, 1); ax[1].set_ylim(0, 1.15); ax[1].set_xlabel(r"cosine of the deflection angle, $\cos\chi$", labelpad=1); ax[1].set_ylabel("after one collision")
    ax[1].legend(loc="lower left", frameon=False, fontsize=8.6, handlelength=1.8, borderaxespad=0.25, labelspacing=0.25); tag(ax[1], "(b)", x=0.88, y=0.9)
    ax[1].set_xticks([-1, -0.5, 0, 0.5, 1])
    fig.subplots_adjust(left=0.185, right=0.955, top=0.985, bottom=0.085, hspace=0.27)
    save(fig, "fig1"); print(f"   exponential: amplitude {A:.3f}, temperature {th:.3f} T2 (Mach 12.9 set {S129})")


# ---------------------------------------------------------------- figure 2: hot group and collision history
def curves(D, lab):
    n, cls, tg, cnt = (D[f"{lab}_{k}"] for k in ("n", "cls", "tag", "cnt")); ns = (n - 1) / (float(D[f"{lab}_n2"]) - 1); c = cls[:5] / cls[:5].sum(0)
    m = (ns > 0.005) & (ns < 0.99) & (np.arange(len(ns)) < np.argmax(ns > 0.99) + 1 if (ns > 0.99).any() else True); k = np.argsort(ns[m])
    return ns[m][k], (1 - c[0])[m][k], (1 - tg / cnt)[m][k]


def fig2():
    """measured hot group against the shares of molecules with at least one, two and three collisions (kernel runs)"""
    d = np.load(os.path.join(N2, F_ERH)); fig, ax = plt.subplots(2, 1, figsize=(W1, 3.7), sharex=True)
    for a, (mach, key, title) in zip(ax, (("M7k", "7", "(a) Mach 7.0"), ("M12.9k", "12.9", "(b) Mach 12.9"))):
        n, erh = d[f"{mach}_n"], d[f"{mach}_erh"]; ns = (n - 1) / (float(d[f"{mach}_n2"]) - 1); t = erh.sum(-1); tot = np.maximum(t[0], 1e-30)
        m = (ns > 0.005) & (ns < 0.99) & (np.arange(len(ns)) < np.argmax(ns > 0.99) + 1 if (ns > 0.99).any() else True); k = np.argsort(ns[m]); X = ns[m][k]
        s1 = (1 - t[2] / tot)[m][k]; s2 = ((t[10] + t[11]) / tot)[m][k]; s3 = (t[11] / tot)[m][k]
        a.plot(X, s1, ":", color="k", lw=1.6); a.plot(X, s2, "--", color=GREY, lw=1.4); a.plot(X, s3, "-", color=BLUE, lw=1.9)
        e = rt.reduce(rt.CASES[key]["rows"]); rho1 = e[0, 2]; rho2 = e[-3:, 2].mean(); en = (e[:, 2] - rho1) / (rho2 - rho1); s = (en > 0.02) & (en < 0.97)
        a.plot(en[s], e[s, 4], "o", color=ORANGE, ms=5.5, mec="k", mew=0.7, zorder=5)
        r = [np.sqrt(np.mean((np.interp(en[s], X, v) - e[s, 4]) ** 2)) for v in (s1, s2, s3)]
        print(f"   fig2 {mach}: rms difference from the measured hot group: >=1 collision {r[0]:.3f}, >=2 {r[1]:.3f}, >=3 {r[2]:.3f}")
        a.set_xlim(0, 1); a.set_ylim(0, 1.04); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); a.text(0.96, 0.1, title, transform=a.transAxes, ha="right", fontsize=10)
    ax[1].set_xlabel(r"normalised density $n^*$", labelpad=1); fig.text(0.025, 0.494, "fraction of molecules", rotation=90, va="center", ha="center")
    hh = [Line2D([], [], color=ORANGE, marker="o", mec="k", mew=0.7, ms=5.5, ls=""), Line2D([], [], color="k", ls=":", lw=1.6), Line2D([], [], color=GREY, ls="--", lw=1.4), Line2D([], [], color=BLUE, lw=1.9)]
    ll = ["measured hot group", "one or more collisions", "two or more", "three or more"]
    fig.legend(hh, ll, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.545, 1.0), fontsize=8.3, handlelength=1.7, columnspacing=0.9)
    fig.subplots_adjust(left=0.16, right=0.955, top=0.892, bottom=0.096, hspace=0.07)
    save(fig, "fig2")


def fig2_old():
    D = {}
    for f in ("class_exch.npz", "class_ref.npz"):
        z = np.load(os.path.join(N2, f)); D.update({k: z[k] for k in z.files})
    fig, ax = plt.subplots(2, 1, figsize=(W1, 4.2), sharex=True)
    for a, (mach, key, title) in zip(ax, (("M7", "7", "(a) Mach 7.0"), ("M12.9", "12.9", "(b) Mach 12.9"))):
        X, coll, _ = curves(D, mach + "zi10"); a.plot(X, coll, ":", color="k", lw=1.6)
        X, _, ex = curves(D, mach + "zi10"); a.plot(X, ex, "-", color=BLUE, lw=1.8)
        X, _, ex = curves(D, mach); a.plot(X, ex, "--", color=RED, lw=1.5)
        e = rt.reduce(rt.CASES[key]["rows"]); rho1 = e[0, 2]; rho2 = e[-3:, 2].mean(); en = (e[:, 2] - rho1) / (rho2 - rho1); s = (en > 0.02) & (en < 0.97)
        a.plot(en[s], e[s, 4], "o", color=ORANGE, ms=5.5, mec="k", mew=0.7, zorder=5)
        a.set_xlim(0, 1); a.set_ylim(0, 1.04); a.set_ylabel("fraction of molecules"); a.text(0.96, 0.1, title, transform=a.transAxes, ha="right", fontsize=10)
    ax[1].set_xlabel(r"normalised density $n^*$", labelpad=1)
    hh = [Line2D([], [], color=ORANGE, marker="o", mec="k", mew=0.7, ms=5.5, ls=""), Line2D([], [], color="k", ls=":", lw=1.6), Line2D([], [], color=BLUE, lw=1.8), Line2D([], [], color=RED, ls="--", lw=1.5)]
    ll = ["measured hot group", "collided at least once", r"exchanged, $p(E_c)$", r"exchanged, $p=0.2$"]
    fig.legend(hh, ll, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.545, 1.0), fontsize=8.3, handlelength=1.7, columnspacing=0.9)
    fig.subplots_adjust(left=0.16, right=0.955, top=0.905, bottom=0.085, hspace=0.07)
    save(fig, "fig2")


# ---------------------------------------------------------------- figure 3: measured populations as a sum of three fixed shapes
def three_shapes(key, exc, kmax=24):
    """Stations of Robben & Talbot, their level populations (K = 0..kmax; T' line above the tabulated levels) and the
    three shapes fixed in advance: measured free stream, one gas-kinetic collision (trajectories), Boltzmann at T2."""
    TH = rt.THETA; K = np.arange(kmax + 1); G = 2 * K + 1; EK = TH * K * (K + 1)
    rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); T2 = ex[-1, 1]; en = (ex[:, 2] - rho1) / (rho2 - rho1)

    def levels(row):
        X, Tp, dk, last = row; kp = len(dk); A = last * np.exp(TH * (kp - 0.5) * (kp + 0.5) / Tp)
        d = np.array([dk[k] if k < kp else A * np.exp(-EK[k] / Tp) for k in K]); N = G * d
        return N / (N.sum() + A * Tp / TH * np.exp(-TH * (kmax + 0.5) * (kmax + 1.5) / Tp)), kp

    d = np.load(exc); E = d["E"] * T2; sv = float(d["sig_vhs"]); sm = float(d["sigmax"]); edge = TH * np.arange(kmax + 2) ** 2
    f1 = (d["ctc"] - d["before"] * (1 - sv / sm)) / sv; c = np.interp(edge, E, f1); S1 = np.maximum(c[:-1] - c[1:], 0) / c[0]      # excess over the VHS cross-section: unchanged molecules
    flb = (d["lb"] - 0.8 * sv * d["before"] / sm) / (0.2 * sv); c = np.interp(edge, E, flb); SLB = np.maximum(c[:-1] - c[1:], 0) / c[0]  # molecules the LB rule excites
    S0, _ = levels(rows[0]); S2 = G * np.exp(-EK / T2); S2 = S2 / (S2.sum() + T2 / TH * np.exp(-TH * (kmax + 0.5) * (kmax + 1.5) / T2))
    P = [levels(r) for r in rows]
    return dict(K=K, G=G, EK=EK, T2=T2, en=en, rows=rows, P=[p for p, _ in P], kp=[k for _, k in P], S=np.stack([S0, S1, S2], 1), SLB=SLB, TH=TH)


def fig_shapes(key="7", lab="M7zi10", want=(0.074, 0.243)):
    from scipy.optimize import nnls
    q = three_shapes(key, os.path.join(N2, "ctc_cluster", f"exc_{S7 if key == '7' else S129}.npz")); K, G, EK, T2, en, S, TH = (q[k] for k in ("K", "G", "EK", "T2", "en", "S", "TH"))
    dens = lambda p: p / G * T2 / TH                                   # population per unit degeneracy and unit e/kT2
    sel = [j for j in range(len(en)) if 0.02 < en[j] < 0.97]; W = np.array([nnls(S, q["P"][j])[0] for j in sel])
    res = lambda B, j: np.linalg.norm(q["P"][j] - B @ nnls(B, q["P"][j])[0]) / np.linalg.norm(q["P"][j])
    tab = [j for j in sel if q["kp"][j] > 0]
    r3 = [res(S, j) for j in tab]; r2 = [res(S[:, [0, 2]], j) for j in tab]; rl = [res(np.stack([S[:, 0], q["SLB"], S[:, 2]], 1), j) for j in tab]
    print(f"   fig3 Mach {key}: stations with tabulated levels n* = " + " ".join(f"{en[j]:.2f}" for j in tab))
    print("      residual, three shapes: " + " ".join(f"{v:.3f}" for v in r3) + f" (mean {np.mean(r3):.3f}); two shapes: " + " ".join(f"{v:.3f}" for v in r2) + f" (mean {np.mean(r2):.3f}); LB-excited shape as second: mean {np.mean(rl):.3f}")
    print("      sum of the weights: " + " ".join(f"{v:.3f}" for v in W.sum(1)))
    C = np.load(os.path.join(N2, "class_ref.npz")); n, cls = C[f"{lab}_n"], C[f"{lab}_cls"]; ns = (n - 1) / (float(C[f"{lab}_n2"]) - 1); c = cls[:5] / cls[:5].sum(0)
    m = (ns > 0.002) & (ns < 0.995) & (np.arange(len(ns)) < np.argmax(ns > 0.995) + 1 if (ns > 0.995).any() else True); o = np.argsort(ns[m]); X = ns[m][o]
    D = [c[0][m][o], (c[1] + c[2] + c[4])[m][o], c[3][m][o]]
    print("      weights (stream, one collision, thermal) and DSMC shares at the stations:")
    for j, w in zip(sel, W):
        print(f"         n* = {en[j]:.3f}: " + " ".join(f"{v:.3f}" for v in w) + " | " + " ".join(f"{np.interp(en[j], X, v):.3f}" for v in D))
    if key != "7": return
    CS = [BLUE, ORANGE, RED]; names = ["free stream", "one collision", r"thermal at $T_2$"]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 3.0))
    for a, wnt, lt in zip(ax[:2], want, "ab"):
        j = int(np.argmin(np.abs(en - wnt))); p = q["P"][j]; kp = q["kp"][j]; w = nnls(S, p)[0]; x = EK / T2; mm = x <= 4.3
        for i in range(3):
            if w[i] > 1e-4: a.semilogy(x[mm], dens(w[i] * S[:, i])[mm], "-", color=CS[i], lw=1.3)
        a.semilogy(x[mm], dens(S @ w)[mm], "k-", lw=1.9)
        a.semilogy(x[:kp], dens(p)[:kp], "o", color="k", ms=4.4, mfc="white", mew=1.0, zorder=6); a.semilogy(x[kp:][x[kp:] <= 4.3], dens(p)[kp:][x[kp:] <= 4.3], "--", color="k", lw=1.1, dashes=(2, 2), zorder=5)
        a.set_ylim(3e-4, 40); a.set_xlim(0, 4); a.set_xlabel(r"$e_{\rm rot}/kT_2$", labelpad=1); a.text(0.95, 0.9, rf"({lt}) $n^*={en[j]:.2f}$", transform=a.transAxes, ha="right", fontsize=10)
        print(f"      panel {lt}: n* = {en[j]:.3f}, weights " + " ".join(f"{v:.3f}" for v in w) + f", residual {np.linalg.norm(p - S @ w) / np.linalg.norm(p):.3f}")
    ax[0].set_ylabel("probability density"); ax[1].set_yticklabels([]); ax[0].set_xticks([0, 1, 2, 3])
    a = ax[2]
    for i, mk in zip(range(3), ("o", "s", "^")):
        a.plot(X, D[i], "-", color=CS[i], lw=1.5); a.plot(en[sel], W[:, i], mk, color=CS[i], ms=5.2, mec="k", mew=0.6, zorder=5)
    a.set_xlim(0, 1); a.set_ylim(0, 1.04); a.set_xlabel(r"normalised density $n^*$", labelpad=1); a.set_ylabel("share of the molecules"); a.text(0.5, 0.9, "(c)", transform=a.transAxes, ha="center", fontsize=10)
    hh = [Line2D([], [], color="k", marker="o", mfc="white", ms=4.4, mew=1.0, ls="--", lw=1.1), Line2D([], [], color="k", lw=1.9)] + [Line2D([], [], color=cc, lw=1.5) for cc in CS]
    fig.legend(hh, ["measured levels and $T'$ line", "sum of the three shapes"] + names, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9,
               columnspacing=2.0, handlelength=2.0, handletextpad=0.5)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.83, bottom=0.165, wspace=0.06)
    pos = a.get_position(); a.set_position([pos.x0 + 0.075, pos.y0, pos.width - 0.075, pos.height])
    save(fig, "figS_shapes")


# ---------------------------------------------------------------- (not in the letter) levels by generation in DSMC
def fig_generations(lab="M7zi10", key="7", want=(0.074, 0.243, 0.579)):
    d = np.load(os.path.join(N2, "erh_ref.npz")); x, n, erh = (d[f"{lab}_{k}"] for k in ("x", "n", "erh")); n2 = float(d[f"{lab}_n2"])
    nb = erh.shape[-1]; DE = 8.0 / nb; E = (np.arange(nb) + 0.5) * DE; ns = (n - 1) / (n2 - 1); TH = rt.THETA
    rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); Te2 = ex[-1, 1]; en = (ex[:, 2] - rho1) / (rho2 - rho1)
    CG = [BLUE, "#8E6BBE", ORANGE, RED]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 3.0), sharey=True)
    for a, w, lt in zip(ax, want, "abc"):
        j = int(np.argmin(np.abs(en - w))); X, Tp, dk, last = rows[j]; K = np.arange(len(dk)); kp = len(dk); rho = ex[j, 2]
        EK = TH * K * (K + 1) / Te2; pK = np.array(dk) * (Te2 / TH) / rho
        Ekp = TH * (kp - 0.5) * (kp + 0.5) / Te2; el = np.linspace(Ekp, 4.0, 40); pl = last * (Te2 / TH) / rho * np.exp(-(el - Ekp) * Te2 / Tp)
        i = int(np.argmin(np.abs(ns - en[j]))); s = slice(max(i - 2, 0), i + 3); f = erh[:, s].sum(1); tot = f[0].sum(); m = E < 4.0
        for g in range(4):
            a.semilogy(E[m], np.maximum(f[2 + g][m], 1e-30) / tot / DE, "-", color=CG[g], lw=1.2)
        a.semilogy(E[m], f[0][m] / tot / DE, "k-", lw=1.8)
        a.semilogy(EK, pK, "o", color=RED, ms=4.2, mfc="white", mew=1.0, zorder=6); a.semilogy(el, pl, "--", color=RED, lw=1.3, zorder=6)
        a.set_ylim(3e-4, 40); a.set_xlim(0, 4); a.set_xlabel(r"$e_{\rm rot}/kT_2$", labelpad=1); a.text(0.95, 0.9, rf"({lt}) $n^*={en[j]:.2f}$", transform=a.transAxes, ha="right", fontsize=10)
        print(f"   fig3 panel {lt}: station X = {X}, n* = {en[j]:.3f}, DSMC cell n* = {ns[i]:.3f}, T' = {Tp} K = {Tp / Te2:.2f} T2")
    ax[0].set_ylabel("probability density")
    hh = [Line2D([], [], color=RED, marker="o", mfc="white", ms=4.2, mew=1.0, ls="--", lw=1.3), Line2D([], [], color="k", lw=1.8)] + [Line2D([], [], color=c, lw=1.4) for c in CG]
    fig.legend(hh, ["measured levels and $T'$ line", "DSMC, all molecules", "generation 0", "generation 1", "generation 2", "generation 3+"], loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.53, 1.0), fontsize=9, columnspacing=2.0, handlelength=2.0, handletextpad=0.5)
    for a in ax[:2]: a.set_xticks([0, 1, 2, 3])
    fig.subplots_adjust(left=0.085, right=0.985, top=0.83, bottom=0.165, wspace=0.06)
    save(fig, "figS_generations")


# ---------------------------------------------------------------- figure 3: populations predicted with no fitted number
def dsmc_levels(d, lab, nstar, T2, S0, kmax=24, groups=(2, 3, 4, 5)):
    """Level populations of DSMC at the normalised density nstar: all molecules and four groups (histogram rows; the
    default is the generations 0-3, rows 2, 9, 10, 11 of version 17 are 0, 1, 2, 3+ collisions), with the unscattered
    molecules (first group) given the measured free-stream shape S0."""
    TH = rt.THETA; n, erh, erl = (d[f"{lab}_{k}"] for k in ("n", "erh", "erl")); ns = (n - 1) / (float(d[f"{lab}_n2"]) - 1)
    i = int(np.argmin(np.abs(ns - nstar))); s = slice(max(i - 2, 0), i + 3); tot = erh[0, s].sum(); edge = TH * np.arange(kmax + 2) ** 2 / T2
    ef = np.arange(161) * 0.005; ec = np.arange(161) * 0.05

    def pops(row):
        cf = np.concatenate([[0], np.cumsum(erl[row, s].sum(0))]); cc = np.concatenate([[0], np.cumsum(erh[row, s].sum(0))])
        C = np.where(edge <= 0.8, np.interp(edge, ef, cf), cf[-1] + np.interp(edge, ec, cc) - np.interp(0.8, ec, cc)); return (C[1:] - C[:-1]) / tot

    gen = [pops(r) for r in groups]
    if S0 is None: return pops(0), gen                  # stream already entering with the measured distribution
    w0 = erh[groups[0], s].sum() / tot; allp = pops(0) - gen[0] + w0 * S0; gen[0] = w0 * S0
    return allp, gen


def fig3(key="7", want=(0.074, 0.243), kl="M7k", ll="M7zi10", exc=None, name="fig3", nmin=0.02, ymax=1.6):
    q = three_shapes(key, os.path.join(N2, "ctc_cluster", f"exc_{exc or S7}.npz")); K, G, EK, T2, en, TH = (q[k] for k in ("K", "G", "EK", "T2", "en", "TH")); S0 = q["S"][:, 0]
    dk = np.load(os.path.join(N2, F_ERH)); dl = np.load(os.path.join(N2, F_LB)); dens = lambda p: p / G * T2 / TH; x = EK / T2
    if RIN: S0 = None
    CG = [BLUE, "#8E6BBE", ORANGE, RED]; GR = (2, 9, 10, 11)
    fig, ax = plt.subplots(1, 3, figsize=(W2, 3.0))
    for a, wnt, lt in zip(ax[:2], want, "ab"):
        j = int(np.argmin(np.abs(en - wnt))); p = q["P"][j]; kp = q["kp"][j]; tot, gen = dsmc_levels(dk, kl, en[j], T2, S0, groups=GR); lb, _ = dsmc_levels(dl, ll, en[j], T2, S0); mm = x <= 4.3
        for g in range(4):
            a.semilogy(x[mm], np.maximum(dens(gen[g])[mm], 1e-30), "-", color=CG[g], lw=1.2)
        a.semilogy(x[mm], dens(lb)[mm], "--", color=GREY, lw=1.5, dashes=(4, 2)); a.semilogy(x[mm], dens(tot)[mm], "k-", lw=1.9)
        a.semilogy(x[:kp], dens(p)[:kp], "o", color="k", ms=4.4, mfc="white", mew=1.0, zorder=6); a.semilogy(x[kp:][x[kp:] <= 4.3], dens(p)[kp:][x[kp:] <= 4.3], ":", color="k", lw=1.3, zorder=5)
        a.set_ylim(3e-4, 40); a.set_xlim(0, 4); a.set_xlabel(r"$e_{\rm rot}/kT_2$", labelpad=1); a.text(0.95, 0.9, rf"({lt}) $n^*={en[j]:.2f}$", transform=a.transAxes, ha="right", fontsize=10)
    ax[0].set_ylabel("probability density"); ax[1].set_yticklabels([]); ax[0].set_xticks([0, 1, 2, 3])
    a = ax[2]; rk, rl = [], []
    for j in range(len(en)):
        kp = q["kp"][j]
        if not (nmin < en[j] < 0.97) or kp == 0: continue
        p = q["P"][j]; tot, _ = dsmc_levels(dk, kl, en[j], T2, S0, groups=GR); lb, _ = dsmc_levels(dl, ll, en[j], T2, S0)
        a.plot(K[:kp] + 0.12, lb[:kp] / p[:kp], "s", color=GREY, ms=4.0, mfc="white", mew=0.9, zorder=3); a.plot(K[:kp] - 0.12, tot[:kp] / p[:kp], "o", color="k", ms=4.0, zorder=4)
        rk += (tot[:kp] / p[:kp]).tolist(); rl += (lb[:kp] / p[:kp]).tolist()
    a.axhline(1.0, color="k", lw=0.7); a.set_xlim(-0.7, 12.7); a.set_xticks([0, 4, 8, 12]); a.set_ylim(0.5, ymax); a.set_xlabel(r"rotational level $K$", labelpad=1); a.set_ylabel("predicted / measured"); a.text(0.86, 0.9, "(c)", transform=a.transAxes, fontsize=10)
    rk, rl = np.array(rk), np.array(rl)
    print(f"   fig3: predicted/measured over {len(rk)} tabulated levels of the Mach {key} stations: trajectory kernel {rk.min():.2f}-{rk.max():.2f}, rms deviation {np.sqrt(np.mean((rk - 1) ** 2)):.3f}; "
          f"LB with p(Ec) {rl.min():.2f}-{rl.max():.2f}, rms {np.sqrt(np.mean((rl - 1) ** 2)):.3f}")
    hh = [Line2D([], [], color="k", marker="o", mfc="white", ms=4.4, mew=1.0, ls=":", lw=1.3), Line2D([], [], color="k", lw=1.9), Line2D([], [], color=GREY, lw=1.5, ls="--")] + [Line2D([], [], color=c, lw=1.4) for c in CG]
    ll = ["measured levels and $T'$ line", "DSMC, trajectory kernel", r"DSMC, LB rule with $p(E_c)$", "no collision", "one collision", "two collisions", "three or more"]
    fig.legend(hh, ll, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8.8, columnspacing=1.4, handlelength=1.9, handletextpad=0.5)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.83, bottom=0.165, wspace=0.06)
    pos = a.get_position(); a.set_position([pos.x0 + 0.075, pos.y0, pos.width - 0.075, pos.height])
    save(fig, name)


# ---------------------------------------------------------------- figure 4: precursor
def fig4():
    d = np.load(os.path.join(N2, F_PRE))
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.8))
    for lab, cl, col, mk, name in (("M7b", "M7c", BLUE, "o", "Mach 7.0"), ("M12.9b", "M12.9c", RED, "s", "Mach 12.9")):
        AN, b = d[f"{lab}_AN"], d[f"{lab}_beta"]; ok = (AN > 0) & (b > 2e-6) & (b < 0.2); k = np.where(ok)[0]; k = k[np.argsort(AN[k])]
        ax[0].semilogy(AN[k][::6], b[k][::6], mk, color=col, mfc="white", ms=3.8, mew=0.9, zorder=3)
        ca, cb = d[f"{cl}_AN"], d[f"{cl}_beta"]; o = np.argsort(ca); ca, cb = ca[o], cb[o]; good = cb > 0
        # the cascade is linear: shift along the depth so that it passes through the DSMC point beta = 1e-2
        a0 = np.interp(math.log(1e-2), np.log(b[k])[::-1], AN[k][::-1]); c0 = np.interp(math.log(1e-2), np.log(cb[good])[::-1], ca[good][::-1])
        mm = good & (cb < 1e-2) & (cb > 2e-6); ax[0].semilogy(ca[mm] - c0 + a0, cb[mm], "-", color=col, lw=1.6, zorder=4)
        print(f"   fig4 {name}: decay rate DSMC {float(d[lab + '_slope']):.4f} +- {float(d[lab + '_slope_se']):.4f}, cascade {float(d[cl + '_slope']):.4f}; ratio {float(d[cl + '_slope']) / float(d[lab + '_slope']):.4f}; "
              f"decay length DSMC {float(d[lab + '_length']):.3f} l1")
    ax[0].set_xlim(0, 22); ax[0].set_ylim(1e-5, 0.2); ax[0].set_xlabel(r"scaled column density $A\mathcal{N}$", labelpad=1); ax[0].set_ylabel(r"fraction $\beta$")
    hh = [Line2D([], [], color=BLUE, marker="o", mfc="white", ms=3.8, mew=0.9, ls=""), Line2D([], [], color=RED, marker="s", mfc="white", ms=3.8, mew=0.9, ls=""), Line2D([], [], color="k", lw=1.6)]
    ax[0].legend(hh, ["DSMC, Mach 7.0", "DSMC, Mach 12.9", "linear cascade"], loc="lower left", frameon=False, fontsize=8.8, borderaxespad=0.3, labelspacing=0.3); tag(ax[0], "(a)", x=0.88, y=0.9)
    # composition of the upstream-moving molecules against beta
    edges = np.logspace(-4.25, -1.75, 11); mid = np.sqrt(edges[1:] * edges[:-1])
    for lab, ls in (("M7b", "-"), ("M12.9b", "--")):
        b = d[f"{lab}_beta"]; cn = d[f"{lab}_cn"]; xx = d[f"{lab}_x"]; up = xx < 0; sh = np.zeros((3, len(mid)))
        for q in range(len(mid)):
            w = up & (b >= edges[q]) & (b < edges[q + 1]); c = cn[:, w].sum(1); sh[:, q] = np.array([c[1], c[4], c[2] + c[3]]) / c.sum()
        for r, col in zip(range(3), (BLUE, ORANGE, GREY)):
            ax[1].semilogx(mid, sh[r], ls, color=col, lw=1.7 if r < 2 else 1.4)
        print(f"   fig4 {lab}: shares (stream molecule after one collision, scattered molecule last hit by a stream molecule, others) at beta = {mid[-1]:.1e}: " + " ".join(f"{v:.3f}" for v in sh[:, -1])
              + f"; at beta = {mid[0]:.1e}: " + " ".join(f"{v:.3f}" for v in sh[:, 0]))
    ax[1].set_xlim(1.5e-2, 7e-5); ax[1].set_ylim(0, 0.86); ax[1].set_xlabel(r"upstream-moving fraction $\beta$", labelpad=1); ax[1].set_ylabel("share")
    hh = [Line2D([], [], color=BLUE, lw=1.7), Line2D([], [], color=ORANGE, lw=1.7), Line2D([], [], color=GREY, lw=1.4)]
    ax[1].legend(hh, ["stream molecule, one collision", "last partner a stream molecule", "others"], loc="upper left", frameon=False, fontsize=8.6, borderaxespad=0.3, labelspacing=0.3)
    tag(ax[1], "(b)", x=0.88, y=0.9)
    fig.subplots_adjust(left=0.175, right=0.955, top=0.983, bottom=0.096, hspace=0.3)
    save(fig, "fig4")


# ---------------------------------------------------------------- figure 5 (End Matter): one collision at fixed collision energy
def fig5():
    d = np.load(os.path.join(N2, "ctc_cluster", f"cond_{S129}.npz")); E, S, fE, fS, et = d["E"], d["S"], d["bin_fE"], d["bin_fS"], d["bin_et"]; ed = d["et_edges"]
    use = [j for j in range(len(et)) if np.isfinite(et[j]) and 2 <= ed[j] < 10]; cm = plt.get_cmap("viridis"); cols = [cm(0.92 * q / (len(use) - 1)) for q in range(len(use))]
    fig, ax = plt.subplots(2, 1, figsize=(W1, 4.5))
    for j, c in zip(use, cols):
        m = (E >= 0.05) & (E <= 5.0) & (fE[j] > 0); ax[0].semilogy(E[m], fE[j][m], "-", color=c, lw=1.4, label=f"{ed[j]:.0f}-{ed[j + 1]:.0f}")
        m = (S >= 0.01) & (S <= 0.62) & (fS[j] > 0); ax[1].semilogy(S[m], fS[j][m], "-", color=c, lw=1.4)
    m = (E >= 0.05) & (E <= 5.0); ax[0].semilogy(E[m], d["after"][m], "k--", lw=1.3, dashes=(4, 2), label="all")
    s0 = float(np.mean([0.2 / math.log(fS[j][20] / fS[j][60]) for j in use])); a0 = float(np.mean([fS[j][20] * math.exp(0.1 / s0) for j in use]))
    x = np.linspace(0, 0.62, 50); ax[1].semilogy(x, a0 * np.exp(-x / s0), "k--", lw=1.3, dashes=(4, 2), label=rf"${a0:.2f}\,\exp(-x/{s0:.2f})$")
    print(f"   fig5: share distribution between 0.1 and 0.3 of the pair energy: mean slope {s0:.3f}, amplitude {a0:.3f}; per bin slope " + " ".join(f"{0.2 / math.log(fS[j][20] / fS[j][60]):.3f}" for j in use)
          + "; slope temperature over Et per bin " + " ".join(f"{d['bin_slope'][j] / et[j]:.3f}" for j in use) + "; mean gain over Et " + " ".join(f"{d['bin_gain'][j] / et[j]:.3f}" for j in use))
    ax[0].set_xlim(0, 5); ax[0].set_ylim(2e-4, 3); ax[0].set_xlabel(r"$E/kT_2$", labelpad=1); ax[0].set_ylabel(r"$\sigma(e_{\rm rot}'>E)\,/\,\sigma_{\rm VHS}$")
    h0, l0 = ax[0].get_legend_handles_labels()
    ax[0].legend(h0[-1:], ["whole ensemble"], loc="lower left", frameon=False, fontsize=8.6, borderaxespad=0.4, handlelength=1.8); tag(ax[0], "(a)", x=0.88, y=0.9)
    ax[1].set_xlim(0, 0.62); ax[1].set_ylim(2e-3, 6); ax[1].set_xlabel(r"share of the pair energy, $x=E/E_{\rm tot}$", labelpad=1); ax[1].set_ylabel(r"$\sigma(e_{\rm rot}'>xE_{\rm tot})\,/\,\sigma_{\rm VHS}$")
    h1, l1 = ax[1].get_legend_handles_labels()
    ax[1].legend(h1, l1, loc="lower left", frameon=False, fontsize=9, borderaxespad=0.4)
    ax[1].add_artist(ax[1].get_legend()); ax[1].legend(h0[:-1], l0[:-1], loc="upper right", ncol=3, frameon=False, fontsize=8.0, borderaxespad=0.25, labelspacing=0.15, columnspacing=0.8,
                                                     handlelength=1.4, title=r"bins of $E_t/kT_2$", title_fontsize=8.4)
    tag(ax[1], "(b)", x=0.04, y=0.92)
    fig.subplots_adjust(left=0.2, right=0.975, top=0.985, bottom=0.095, hspace=0.3); save(fig, "figS_energy")


# ---------------------------------------------------------------- supplemental figures (sm=1)
def figS_kernel():
    k = np.load(os.path.join(N2, "kcell", "kernel_tests.npz")); fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9))
    for nm, col, lab in (("M12p9", BLUE, "Mach 12.9"), ("M7", RED, "Mach 7.0")):
        E = k[nm + "_E"]; m = (E >= 0.05) & (E <= 5.0); idx = np.where(m)[0][6::9]
        ax[0].semilogy(E[m], k[nm + "_direct"][m], "-", color=col, lw=1.6, label=f"trajectories, {lab}")
        ax[0].semilogy(E[idx], k[nm + "_kernel"][idx], "o", color=col, mfc="white", ms=4.0, mew=0.9, label=f"kernel, {lab}")
    ax[0].set_xlim(0, 5); ax[0].set_ylim(2e-5, 3); ax[0].set_xlabel(r"$E/kT_2$", labelpad=1); ax[0].set_ylabel(r"$\sigma(e_{\rm rot}'>E)\,/\,\sigma_{\rm VHS}$")
    ax[0].legend(loc="lower left", frameon=False, fontsize=8.8, borderaxespad=0.3, labelspacing=0.3); tag(ax[0], "(a)", x=0.9, y=0.9)
    h = k["rel_hist"]; he = k["eq_hist"]; a = ax[1]
    a.plot(h[:, 0], h[:, 1], "-", color=RED, lw=1.7, label=r"$T_{\rm tr}$, cold rotation at the start"); a.plot(h[:, 0], h[:, 2], "-", color=BLUE, lw=1.7, label=r"$T_{\rm rot}$, cold rotation at the start")
    a.plot(he[:, 0], he[:, 1], "--", color=RED, lw=1.1, label=r"$T_{\rm tr}$, equilibrium at the start"); a.plot(he[:, 0], he[:, 2], ":", color=BLUE, lw=1.4, label=r"$T_{\rm rot}$, equilibrium at the start")
    a.set_xlim(0, 12); a.set_ylim(0, 700); a.set_xlabel("VHS collisions per molecule", labelpad=1); a.set_ylabel("temperature (K)")
    a.legend(loc="upper right", frameon=False, fontsize=8.8, borderaxespad=0.3, labelspacing=0.3); tag(a, "(b)", x=0.9, y=0.1)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.97, bottom=0.165, wspace=0.24); save(fig, "figS_kernel")


def figS_molecules():
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9)); cm = plt.get_cmap("viridis")
    for (s_, lab), c in zip((("elo20", "0.20"), (S129, "0.345 (nitrogen)"), ("elo40", "0.40"), ("elo50", "0.50"), ("elo65", "0.65")), (0.0, 0.25, 0.5, 0.7, 0.9)):
        d = np.load(os.path.join(N2, "ctc_cluster", f"exc_{s_}.npz")); E = d["E"]; m = (E >= 0.05) & (E <= 5.0); ax[0].semilogy(E[m], d["ctc"][m] / float(d["sig_vhs"]), "-", color=cm(c), lw=1.6, label=lab)
    for (s_, lab), col, ls in zip(((S129, r"$T_2=305$ K"), ("simT2", r"$T_2=610$ K"), ("simT4", r"$T_2=1220$ K")), (BLUE, GREEN, RED), ("-", "--", "-.")):
        d = np.load(os.path.join(N2, "ctc_cluster", f"exc_{s_}.npz")); E = d["E"]; m = (E >= 0.05) & (E <= 5.0); ax[1].semilogy(E[m], d["ctc"][m] / float(d["sig_vhs"]), ls, color=col, lw=1.6, label=lab)
    for a, t, lt in zip(ax, ("elongation", "downstream temperature"), ("(a)", "(b)")):
        a.set_xlim(0, 5); a.set_ylim(2e-5, 3); a.set_xlabel(r"$E/kT_2$", labelpad=1); a.legend(loc="lower left", frameon=False, fontsize=8.8, borderaxespad=0.3, labelspacing=0.3, title=t, title_fontsize=8.8); tag(a, lt, x=0.9, y=0.9)
    ax[0].set_ylabel(r"$\sigma(e_{\rm rot}'>E)\,/\,\sigma_{\rm VHS}$")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.97, bottom=0.165, wspace=0.2); save(fig, "figS_molecules")


def figS_precursor():
    d = np.load(os.path.join(N2, F_PRE)); dk = np.load(os.path.join(N2, F_PREK)); edges = np.logspace(-4.25, -1.75, 11); mid = np.sqrt(edges[1:] * edges[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True)
    for a, mach, title in zip(ax, ("M7", "M12.9"), ("(a) Mach 7.0", "(b) Mach 12.9")):
        for src, suf, col, ls, lab in ((dk, "k", "k", "-", "trajectory kernel"), (d, "b", BLUE, "--", r"LB rule, $p(E_c)$"), (d, "l", RED, "-.", r"LB rule, $p=0.2$")):
            L = mach + suf; b = src[f"{L}_beta"]; up = src[f"{L}_x"] < 0; T2 = float(src[f"{L}_T2"]); v = []
            for q in range(len(mid)):
                w = up & (b >= edges[q]) & (b < edges[q + 1]); v.append(src[f"{L}_eg"][w].sum() / max(src[f"{L}_cg"][w].sum(), 1) / T2)
            a.semilogx(mid, v, ls, color=col, lw=1.7, label=lab)
        a.set_xlim(1.5e-2, 7e-5); a.set_ylim(0, 1.45); a.set_xlabel(r"upstream-moving fraction $\beta$", labelpad=1); a.text(0.04, 0.08, title, transform=a.transAxes, fontsize=10)
    ax[0].set_ylabel(r"rotational energy $e^-_{\rm rot}/kT_2$"); h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 1.0), fontsize=9)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.165, wspace=0.06); save(fig, "figS_precursor")


fig1(); fig2(); fig3(); fig_shapes("7", "M7zi10"); fig_shapes("12.9", "M12.9zi10"); fig4()
if opt.get("sm"):
    fig5(); figS_kernel(); figS_molecules(); figS_precursor()
    fig3("12.9", want=(0.143, 0.308), kl="M12.9k", ll="M12.9zi10", exc=S129, name="figS_levels_M12p9", nmin=0.13, ymax=2.3)
