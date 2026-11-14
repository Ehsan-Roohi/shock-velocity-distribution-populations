"""The upstream precursor as a recoil cascade: DSMC collision classes, pure-loss law, linear cascade model.

Inputs
  vdf_cls/DSMC_w*_M*_classes.npz       DSMC with collision-history tags (dsmc/dsmc_classes_to_npz.py)
  cascade/casc_M*_w*_s*.txt            linear cascade, half-range Maxwellian source (dsmc/cascade_linear.cpp)
  cascade/tab_M*_w*_s*.txt             linear cascade driven by the DSMC backward flux at beta = 0.01
  cascade/sources/index.json           positions of those source planes (analysis/cascade_sources.py)

Quantities (window 1e-4 < beta < 1e-2, beta = mass fraction of molecules with xi_x < 0):
  slope of ln(beta) against A N, N the column density between the station and the shock centre (DSMC) or the
  depth (linear model, n1 = 1): all upstream-moving molecules, and those that have had no contact with the
  stream (DSMC: generation 3+; linear model: primaries);
  the same slope for the pure-loss estimate  Phi(N) = min_c [ (c+u2)^2 / 2T2 + A N (c+u1)^(2-2w) / c ].

Usage: python precursor_cascade.py <shock_data dir>
Writes paper/figures/fig16_precursor_cascade.* and analysis/out/precursor_cascade.json.
"""
import sys, os, json, glob, re, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.7, "lines.linewidth": 1.2,
                     "savefig.dpi": 220, "pdf.fonttype": 42})
LO, HI = 1e-4, 1e-2
OMS = [("0p5", 0.5, "hard sphere"), ("0p81", 0.81, r"$\omega=0.81$"), ("1", 1.0, r"$\omega=1$")]


def vhs_A(om):
    return 15 * math.sqrt(math.pi) / (2 * (5 - 2 * om) * (7 - 2 * om)) * 4 ** (om - 0.5) / math.gamma(2.5 - om)


def phi_loss(AN, om, st):
    """Pure-loss exponent by golden-section search over the speed c of the upstream-moving molecule."""
    u1, u2, T2 = st["u1"], st["u2"], st["T2"]; ex = 2 - 2 * om
    f = lambda c: (c + u2) ** 2 / (2 * T2) + AN * (c + u1) ** ex / c
    a, b = 1e-3, 80 * math.sqrt(T2); gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(90):
        if f(c) < f(d): b = d
        else: a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return f(0.5 * (a + b)), 0.5 * (a + b)


def load_cls(f):
    z = np.load(f, allow_pickle=True)
    st = json.loads(str(z["states"])); x = z["x"].astype(float); vx = z["vx"]; dv = float(z["dv"]); rho = z["rho"].astype(float)
    neg = vx < 0
    bg = z["Ggen"][:, :, neg].sum(2) * dv / rho; bo = z["Gown"][:, :, neg].sum(2) * dv / rho
    ic = int(np.argmin(np.abs(z["nstar"] - 0.5))); dx = x[1] - x[0]
    N = np.concatenate([np.cumsum(rho[:ic][::-1])[::-1] * dx, np.zeros(len(x) - ic)])
    return dict(st=st, x=x, om=float(z["omega"]), M=float(z["mach"]), beta=bg.sum(0), surv=bg[3], once=bo[1], N=N, ic=ic, nstar=z["nstar"].astype(float))


def load_casc(pattern):
    fs = sorted(glob.glob(pattern))
    if not fs:
        return None
    a = np.mean([np.loadtxt(f) for f in fs], 0)
    return dict(x=a[:, 0], all=a[:, 1], prim=a[:, 2], scat=a[:, 3], recoil=a[:, 4], speed=a[:, 5], nseeds=len(fs))


def slope(xv, y, k):
    k = k & (y > 0)
    return float(np.polyfit(xv[k], np.log(y[k]), 1)[0]) if k.sum() > 3 else None


def main(root):
    figdir = os.path.join(root, "paper", "figures")
    S = dict(window=[LO, HI], dsmc={}, cascade_maxwell_source={}, cascade_dsmc_source={})
    C = {}
    for f in sorted(glob.glob(os.path.join(root, "vdf_cls", "DSMC_w*_M*_classes.npz"))):
        tag = re.search(r"DSMC_(w.*_M.*)_classes", os.path.basename(f)).group(1)
        d = load_cls(f)
        if d["M"] < 5:
            continue
        C[tag] = d; A = vhs_A(d["om"]); x = d["x"]
        k = (np.arange(len(x)) < d["ic"]) & (d["beta"] < HI) & (d["beta"] > LO) & (x > x[0] + 5)
        AN = A * d["N"]
        ph = np.array([phi_loss(a, d["om"], d["st"])[0] for a in AN[k]])
        i3 = np.where(k)[0][np.argmin(np.abs(np.log(d["beta"][k]) - math.log(1e-3)))]
        S["dsmc"][tag] = dict(mach=d["M"], omega=d["om"], A=A, AN_range=[float(AN[k].min()), float(AN[k].max())],
                              slope_all=slope(AN, d["beta"], k), slope_no_stream_contact=slope(AN, d["surv"], k),
                              slope_pure_loss=float(-np.polyfit(AN[k], ph, 1)[0]),
                              saddle_speed_over_u1=float(phi_loss(AN[k].mean(), d["om"], d["st"])[1] / d["st"]["u1"]),
                              share_no_stream_contact=float(d["surv"][i3] / d["beta"][i3]),
                              share_stream_once=float(d["once"][i3] / d["beta"][i3]))

    for f in sorted({re.sub(r"_s\d+\.txt$", "", p) for p in glob.glob(os.path.join(root, "cascade", "casc_M*_w*_s*.txt"))}):
        m = re.search(r"casc_M(.+)_w(.+)$", os.path.basename(f)); M, om = float(m.group(1)), float(m.group(2))
        c = load_casc(f + "_s*.txt"); A = vhs_A(om); AN = -A * c["x"]
        k = (c["all"] < HI) & (c["all"] > LO)
        i3 = np.where(k)[0][np.argmin(np.abs(np.log(c["all"][k]) - math.log(1e-3)))]
        rec = dict(mach=M, omega=om, seeds=c["nseeds"], slope_all=slope(AN, c["all"], k), slope_primaries=slope(AN, c["prim"], k),
                   share_primaries=float(c["prim"][i3] / c["all"][i3]), share_recoil=float(c["recoil"][i3] / c["all"][i3]),
                   mean_upstream_speed_over_u1=float(c["speed"][k].mean() / (M * math.sqrt(5 / 3))))
        f = (M * math.sqrt(5 / 3)) ** (2 * om - 1); rec["windows"] = {}
        for lo, hi in ((1e-4, 1e-2), (1e-6, 1e-4), (1e-8, 1e-6)):
            kk = (c["all"] < hi) & (c["all"] > lo)
            if kk.sum() > 3:
                rec["windows"][f"{lo:.0e}_{hi:.0e}"] = dict(rate_all_scaled=-slope(AN, c["all"], kk) * f, share_recoil=float(c["recoil"][kk].sum() / c["all"][kk].sum()),
                                                            share_primaries=float(c["prim"][kk].sum() / c["all"][kk].sum()), mean_speed_over_u1=float(c["speed"][kk].mean() / (M * math.sqrt(5 / 3))))
        # stretched-exponential form  ln(beta) = a - b * Ntilde^p  over 1e-7 < beta < 1e-2
        kk = (c["all"] < 1e-2) & (c["all"] > 1e-7); Nt = AN[kk] / f; lb = np.log(c["all"][kk]); fits = {}
        for pw in np.round(np.arange(0.40, 1.21, 0.01), 2):
            co = np.polyfit(Nt ** pw, lb, 1); fits[float(pw)] = float(np.sqrt(np.mean((np.polyval(co, Nt ** pw) - lb) ** 2)))
        best = min(fits, key=fits.get)
        rec["stretched_fit"] = dict(p_best=best, rms_best=fits[best], rms_p1=fits[1.0], rms_p067=fits[0.67], p_heuristic=2 / (2 * om + 1),
                                    rms_heuristic=fits[round(2 / (2 * om + 1), 2)], p_within_twice_best=[min(k for k in fits if fits[k] < 2 * fits[best]), max(k for k in fits if fits[k] < 2 * fits[best])])
        S["cascade_maxwell_source"][f"M{M:g}_w{om:g}"] = rec

    idx = json.load(open(os.path.join(root, "cascade", "sources", "index.json")))
    T = {}
    for key, info in idx.items():
        c = load_casc(os.path.join(root, "cascade", f"tab_{key}_s*.txt"))
        tag = "w" + key.split("_w")[1].replace(".", "p") + "_" + key.split("_")[0]
        if c is None or tag not in C:
            continue
        d = C[tag]; xc = info["x_ref"] + c["x"]; T[tag] = (xc, c, info)
        bd = np.interp(xc, d["x"], d["beta"])
        k = (bd < 0.8 * HI) & (bd > LO) & (xc > d["x"][0] + 5)
        r = np.log(c["all"][k] / bd[k])
        S["cascade_dsmc_source"][tag] = dict(x_ref=info["x_ref"], seeds=c["nseeds"], max_abs_log_ratio=float(np.abs(r).max()), mean_log_ratio=float(r.mean()),
                                              rms_log_ratio=float(np.sqrt(np.mean(r ** 2))), x_range=[float(xc[k].min()), float(xc[k].max())])

    # ---- figure ----
    fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.9))
    MCOL = {8: "C0", 12: "C3", 25: "C2"}
    for j, (otag, om, lab) in enumerate(OMS):
        for M in (8, 12, 25):
            tag = f"w{otag}_M{M}"
            if tag not in C:
                continue
            d = C[tag]; x = d["x"]; up = (np.arange(len(x)) < d["ic"]) & (x > x[0] + 5) & (d["beta"] > 2e-5)
            xr = T[tag][2]["x_ref"] if tag in T else x[up][np.argmin(np.abs(np.log(d["beta"][up]) - math.log(1e-2)))]
            ax[0, j].semilogy(x[up] - xr, d["beta"][up], color=MCOL[M])
            s = up & (d["surv"] > 0)
            ax[0, j].semilogy(x[s] - xr, d["surv"][s], color=MCOL[M], ls=":", lw=1.0)
            if tag in T:
                xc, c, info = T[tag]; kk = (c["all"] > 2e-5) & (xc > x[0] + 5)
                ax[0, j].semilogy(xc[kk] - xr, c["all"][kk], color="k", ls="--", lw=0.9)
            kb = up & (d["beta"] < 3e-2)
            ax[1, j].semilogx(d["beta"][kb], d["surv"][kb] / d["beta"][kb], color=MCOL[M], ls=":", lw=1.0)
            ax[1, j].semilogx(d["beta"][kb], d["once"][kb] / d["beta"][kb], color=MCOL[M], ls="-")
        ax[0, j].set_ylim(2e-5, 3e-2); ax[0, j].set_xlabel(r"$(x-x_{\rm ref})/\ell_1$"); ax[0, j].set_title(f"({'abc'[j]}) {lab}", fontsize=9)
        ax[0, j].set_xlim(None, 1)
        ax[1, j].set_xlim(3e-2, 5e-5); ax[1, j].set_ylim(0, 0.6); ax[1, j].set_xlabel(r"$\beta$"); ax[1, j].set_title(f"({'def'[j]}) {lab}", fontsize=9)
    ax[0, 0].set_ylabel(r"$\beta$"); ax[1, 0].set_ylabel(r"share of $\beta$")
    h = [plt.Line2D([], [], color=MCOL[M], label=f"DSMC, Mach {M}") for M in (8, 12, 25)] + \
        [plt.Line2D([], [], color="k", ls="--", lw=0.9, label="linear cascade from the DSMC flux at $x_{\\rm ref}$"),
         plt.Line2D([], [], color="0.4", ls=":", lw=1.0, label="no contact with the stream"),
         plt.Line2D([], [], color="0.4", ls="-", label="stream molecules after one collision (d-f)")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, "fig16_precursor_cascade." + e))
    json.dump(S, open(os.path.join(root, "analysis", "out", "precursor_cascade.json"), "w"), indent=1)

    print("DSMC: slope all | no stream contact | pure loss || linear cascade (Maxwell source): all | primaries")
    for tag, r in S["dsmc"].items():
        key = f"M{r['mach']:g}_w{r['omega']:g}"; c = S["cascade_maxwell_source"].get(key)
        cs = f"{c['slope_all']:+.3f} | {c['slope_primaries']:+.3f}" if c else "   --"
        print(f"  {tag:12s} {r['slope_all']:+.3f} | {r['slope_no_stream_contact']:+.3f} | {-r['slope_pure_loss']:+.3f} || {cs}   shares at 1e-3: no contact {r['share_no_stream_contact']:.3f}, one-collision stream {r['share_stream_once']:.3f}")
    print("linear cascade from the DSMC flux:")
    for tag, r in S["cascade_dsmc_source"].items():
        print(f"  {tag:12s} ln(model/DSMC): mean {r['mean_log_ratio']:+.3f} rms {r['rms_log_ratio']:.3f} max {r['max_abs_log_ratio']:.3f} over x {r['x_range']}")


if __name__ == "__main__":
    main(sys.argv[1])
