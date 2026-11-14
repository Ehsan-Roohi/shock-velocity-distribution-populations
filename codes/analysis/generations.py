"""Collision-history decomposition of the DSMC shock waves.

Every simulated molecule carries a generation: 0 for a molecule of the incoming stream that has not yet
collided with anything but other such molecules; after any other collision both partners take the
smaller of their two generations plus one (capped at 3). Generation 1 is therefore the set of molecules
whose most recent collision involved an unscattered molecule, generation 2 those one further collision
away from the stream, and 3+ the rest. The files vdf_cls/DSMC_w*_M*_classes.npz hold the reduced
distribution and the moments of each generation (dsmc/dsmc_classes_to_npz.py).

Usage: python generations.py <shock_data dir>
Writes paper/figures/fig13_generation_fractions.*, fig14_generation_distributions.*,
fig15_generations_and_populations.*, fig16_precursor_composition.* and analysis/out/generations_summary.json.
"""
import sys, os, json, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.7, "lines.linewidth": 1.2,
                     "savefig.dpi": 220, "pdf.fonttype": 42})
XI = r"$\hat\xi=(\xi_x-u_2)/(u_1-u_2)$"
GCOL = ["C0", "C1", "C2", "C3"]
GLAB = ["generation 0 (unscattered)", "generation 1", "generation 2", "generation 3+"]
OMS = [("0p5", 0.5, "hard sphere"), ("0p81", 0.81, r"$\omega=0.81$"), ("1", 1.0, r"$\omega=1$")]
LEVELS = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9)
GRID = np.linspace(0.02, 0.95, 94)


def load(path):
    z = np.load(path, allow_pickle=True)
    st = json.loads(str(z["states"]))
    d = dict(st=st, x=z["x"].astype(float), vx=z["vx"].astype(float), dv=float(z["dv"]), nstar=z["nstar"].astype(float),
             rho=z["rho"].astype(float), G=z["G"].astype(float), Ggen=z["Ggen"].astype(float), mgen=z["mgen"].astype(float),
             mown=z["mown"].astype(float), M=float(z["mach"]), omega=float(z["omega"]))
    d["xi"] = sl.xi_hat(d["vx"], st)
    # monotone core of the density profile: from the last cell below n* = 0.005 to the first above 0.995
    lo = np.where(d["nstar"] < 0.005)[0]; lo = lo[lo < np.argmax(d["nstar"] > 0.5)].max()
    hi = np.argmax(d["nstar"] > 0.995) if np.any(d["nstar"] > 0.995) else len(d["x"]) - 1
    d["core"] = np.arange(lo, hi + 1)
    return d


def at_level(d, lv, arr):
    """Linear interpolation of arr (first axis = cell) to the position where n* = lv."""
    c = d["core"]; ns = d["nstar"][c]
    j = int(np.argmax(ns >= lv)); j = max(j, 1)
    a = (lv - ns[j - 1]) / (ns[j] - ns[j - 1])
    return (1 - a) * arr[c[j - 1]] + a * arr[c[j]]


def profiles(d):
    """Generation mass fractions, mean velocities (scaled) and temperatures (over T2) on the n* grid."""
    st = d["st"]
    n = d["mgen"][:, :, 0].T                     # [cell, gen]
    nu = (d["mgen"][:, :, 0] * np.nan_to_num(d["mgen"][:, :, 1])).T
    e2 = (d["mgen"][:, :, 0] * np.nan_to_num(d["mgen"][:, :, 2] + d["mgen"][:, :, 1] ** 2)).T
    tp = (d["mgen"][:, :, 0] * np.nan_to_num(d["mgen"][:, :, 3])).T
    out = dict(frac=[], xi=[], Tpar=[], Tperp=[])
    for lv in GRID:
        nn = at_level(d, lv, n); rr = at_level(d, lv, d["rho"])
        uu = at_level(d, lv, nu) / nn
        out["frac"].append(nn / rr); out["xi"].append((uu - st["u2"]) / (st["u1"] - st["u2"]))
        out["Tpar"].append((at_level(d, lv, e2) / nn - uu ** 2) / st["T2"]); out["Tperp"].append(at_level(d, lv, tp) / nn / st["T2"])
    return {k: np.array(v) for k, v in out.items()}


def gauss_pops(root, tag):
    f = os.path.join(root, "analysis", "out", "popfits", f"DSMC_{tag}_branch3.npz")
    if not os.path.exists(f):
        return None
    b = np.load(f)
    p = b["p3"].reshape(len(b["nstar"]), 3, 4)
    ok = np.all(np.isfinite(b["p3"]), 1)
    mid = np.argmin(np.abs(np.where(ok, b["nstar"], 9) - 0.5))
    order = np.argsort(p[mid, :, 1])            # slow, intermediate, fast at mid-shock; branch order is kept elsewhere
    return dict(nstar=b["nstar"][ok], n=np.exp(p[ok][:, order, 0]), u=p[ok][:, order, 1], a=np.exp(p[ok][:, order, 2]),
                b=np.exp(p[ok][:, order, 3]))


def main(root):
    figdir = os.path.join(root, "paper", "figures"); os.makedirs(figdir, exist_ok=True)
    C = {}
    for f in sorted(glob.glob(os.path.join(root, "vdf_cls", "DSMC_w*_M*_classes.npz"))):
        tag = re.search(r"DSMC_(w.*_M.*)_classes", os.path.basename(f)).group(1)
        C[tag] = load(f); C[tag]["prof"] = profiles(C[tag])
    S = dict(levels=list(LEVELS), cases={})
    for tag, d in C.items():
        pr = d["prof"]; k = [int(np.argmin(np.abs(GRID - lv))) for lv in LEVELS]
        S["cases"][tag] = dict(mach=d["M"], omega=d["omega"], fractions=pr["frac"][k].round(4).tolist(), xi_hat=pr["xi"][k].round(4).tolist(),
                               Tpar_over_T2=pr["Tpar"][k].round(4).tolist(), Tperp_over_T2=pr["Tperp"][k].round(4).tolist(),
                               max_fraction=pr["frac"].max(0).round(4).tolist(),
                               nstar_of_max=[float(GRID[i]) for i in pr["frac"].argmax(0)])

    # ---- fast Gaussian population against generation 0, and stream molecules after exactly one collision ----
    S["fast_vs_generation0"] = {}; S["own_count"] = {}
    for tag, d in C.items():
        st = d["st"]; rows = {}
        for lv in (0.1, 0.2, 0.35, 0.5, 0.7):
            n = at_level(d, lv, d["mown"][:, :, 0].T); rr = at_level(d, lv, d["rho"])
            nu = at_level(d, lv, (d["mown"][:, :, 0] * np.nan_to_num(d["mown"][:, :, 1])).T) / n
            e2 = at_level(d, lv, (d["mown"][:, :, 0] * np.nan_to_num(d["mown"][:, :, 2] + d["mown"][:, :, 1] ** 2)).T) / n
            rows[f"{lv}"] = dict(fractions_k0_k1_k2_k3_lastpartnerstream=(n / rr).round(4).tolist(),
                                 xi_hat=((nu - st["u2"]) / (st["u1"] - st["u2"])).round(3).tolist(), Tpar_over_T2=((e2 - nu ** 2) / st["T2"]).round(3).tolist())
        S["own_count"][tag] = rows
        gp = gauss_pops(root, tag)
        if gp is None:
            continue
        k = (gp["nstar"] > 0.05) & (gp["nstar"] < 0.9)
        wf = (gp["n"][k] / gp["n"][k].sum(1, keepdims=True))[:, 2]; g0 = np.interp(gp["nstar"][k], GRID, d["prof"]["frac"][:, 0])
        S["fast_vs_generation0"][tag] = dict(mach=d["M"], omega=d["omega"], max_excess=float((wf - g0).max()), min_excess=float((wf - g0).min()),
                                             at_nstar=float(gp["nstar"][k][np.argmax(wf - g0)]))

    # ---- collapse of the generation fractions on the normalized density ----
    S["collapse"] = {}
    for otag, om, _ in OMS:
        ms = sorted(d["M"] for t, d in C.items() if t.startswith(f"w{otag}_"))
        if not ms:
            continue
        ref = C[f"w{otag}_M{max(ms):g}".replace(".", "p")]["prof"]
        S["collapse"][otag] = {f"M{m:g}_vs_M{max(ms):g}": dict(
            max_abs_fraction_difference=float(np.abs(C[f"w{otag}_M{m:g}".replace('.', 'p')]["prof"]["frac"] - ref["frac"]).max()),
            max_abs_xi_difference_gen1=float(np.abs(C[f"w{otag}_M{m:g}".replace('.', 'p')]["prof"]["xi"][:, 1] - ref["xi"][:, 1]).max()))
            for m in ms[:-1]}

    # ---- figure 13: fractions and mean velocities of the generations ----
    mset = [m for m in (3, 5, 8, 12, 25) if any(f"_M{m}" == t[t.index("_M"):] for t in C)]
    ls = dict(zip(mset, [(0, (1, 1)), (0, (4, 1.5)), (0, (6, 1.5, 1, 1.5)), "-", (0, (8, 1))][-len(mset):]))
    lw = dict(zip(mset, np.linspace(0.8, 1.5, len(mset))))
    fig, ax = plt.subplots(3, 3, figsize=(7.0, 6.2), sharex=True)
    for j, (otag, om, lab) in enumerate(OMS):
        for m in mset:
            t = f"w{otag}_M{m}"
            if t not in C:
                continue
            pr = C[t]["prof"]
            for g in range(4):
                ax[0, j].plot(GRID, pr["frac"][:, g], color=GCOL[g], ls=ls[m], lw=lw[m])
                ax[1, j].plot(GRID, pr["xi"][:, g], color=GCOL[g], ls=ls[m], lw=lw[m])
                ax[2, j].plot(GRID, pr["Tpar"][:, g], color=GCOL[g], ls=ls[m], lw=lw[m])
        ax[0, j].set_title(f"({'abc'[j]}) {lab}", fontsize=9); ax[2, j].set_xlabel(r"$n^*$")
        ax[0, j].set_ylim(0, 1); ax[1, j].set_ylim(-1.5, 1.2); ax[2, j].set_ylim(0, 2.2)
    ax[0, 0].set_ylabel("mass fraction"); ax[1, 0].set_ylabel(r"mean velocity $\hat\xi$"); ax[2, 0].set_ylabel(r"$T_\parallel/T_2$")
    h = [plt.Line2D([], [], color=GCOL[g], label=GLAB[g]) for g in range(4)] + \
        [plt.Line2D([], [], color="k", ls=ls[m], lw=lw[m], label=f"Mach {m}") for m in mset]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, "fig13_generation_fractions." + e))
    plt.close(fig)

    # ---- figure 14: distributions by generation with the three Gaussian populations ----
    MD = 8
    lv3 = (0.1, 0.3, 0.6)
    fig, ax = plt.subplots(3, 3, figsize=(7.0, 6.0), sharex=True)
    for i, (otag, om, lab) in enumerate(OMS):
        t = f"w{otag}_M{MD}"; d = C[t]; gp = gauss_pops(root, t)
        sc = d["st"]["u1"] - d["st"]["u2"]
        for j, lv in enumerate(lv3):
            rho = at_level(d, lv, d["rho"]); base = np.zeros_like(d["xi"])
            for g in (3, 2, 1, 0):
                y = at_level(d, lv, d["Ggen"][g]) * sc / rho
                ax[i, j].fill_between(d["xi"], base, base + y, color=GCOL[g], alpha=0.55, lw=0)
                base = base + y
            ax[i, j].plot(d["xi"], base, "k", lw=0.6)
            if gp is not None:
                s = int(np.argmin(np.abs(gp["nstar"] - lv)))
                for k in range(3):
                    y = sl.maxwell1d(d["vx"], gp["n"][s, k], gp["u"][s, k], gp["a"][s, k]) * sc / rho
                    ax[i, j].plot(d["xi"], y, color="k", ls=["--", "-.", ":"][k], lw=0.9)
            top = base.max()
            ax[i, j].set_ylim(0, min(top * 1.05, 2.6)); ax[i, j].set_xlim(-2.2, 1.6)
            ax[i, j].set_title(f"{lab}, $n^*={lv}$", fontsize=8.5)
            if i == 2:
                ax[i, j].set_xlabel(XI)
        ax[i, 0].set_ylabel(r"$(u_1-u_2)\,g/\rho$")
    h = [plt.Rectangle((0, 0), 1, 1, color=GCOL[g], alpha=0.55, label=GLAB[g]) for g in range(4)] + \
        [plt.Line2D([], [], color="k", ls=s, lw=0.9, label=l) for s, l in (("--", "Gaussian population, slow"), ("-.", "intermediate"), (":", "fast"))]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, "fig14_generation_distributions." + e))
    plt.close(fig)

    # ---- figure 15: Gaussian populations against generations ----
    S["populations_vs_generations"] = {}
    fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.3), sharex=True)
    for j, (otag, om, lab) in enumerate(OMS):
        t = f"w{otag}_M{MD}"; d = C[t]; gp = gauss_pops(root, t); pr = d["prof"]; st = d["st"]
        if gp is None:
            continue
        k = (gp["nstar"] > 0.02) & (gp["nstar"] < 0.95)
        wts = gp["n"][k] / gp["n"][k].sum(1, keepdims=True); xs = gp["nstar"][k]
        xih = (gp["u"][k] - st["u2"]) / (st["u1"] - st["u2"])
        grp = [pr["frac"][:, 3], pr["frac"][:, 1] + pr["frac"][:, 2], pr["frac"][:, 0]]
        gxi = [pr["xi"][:, 3], (pr["frac"][:, 1] * pr["xi"][:, 1] + pr["frac"][:, 2] * pr["xi"][:, 2]) / (pr["frac"][:, 1] + pr["frac"][:, 2]), pr["xi"][:, 0]]
        col = ["C3", "C4", "C0"]
        for q in range(3):
            ax[0, j].plot(GRID, grp[q], color=col[q]); ax[0, j].plot(xs, wts[:, q], color=col[q], ls="--")
            ax[1, j].plot(GRID, gxi[q], color=col[q]); ax[1, j].plot(xs, xih[:, q], color=col[q], ls="--")
        ax[0, j].set_title(f"({'abc'[j]}) {lab}", fontsize=9); ax[1, j].set_xlabel(r"$n^*$"); ax[0, j].set_ylim(0, 1); ax[1, j].set_ylim(-0.8, 1.2)
        gi = lambda a: np.interp(xs, GRID, a)
        S["populations_vs_generations"][t] = dict(
            max_abs_weight_difference=dict(fast_vs_gen0=float(np.abs(wts[:, 2] - gi(grp[2])).max()),
                                           intermediate_vs_gen1plus2=float(np.abs(wts[:, 1] - gi(grp[1])).max()),
                                           intermediate_vs_gen1=float(np.abs(wts[:, 1] - gi(pr["frac"][:, 1])).max()),
                                           slow_vs_gen3=float(np.abs(wts[:, 0] - gi(grp[0])).max())),
            weights_at_levels={f"{lv}": dict(gaussian_slow_mid_fast=wts[int(np.argmin(np.abs(xs - lv)))].round(3).tolist(),
                                              generations=pr["frac"][int(np.argmin(np.abs(GRID - lv)))].round(3).tolist(),
                                              xi_gaussian=xih[int(np.argmin(np.abs(xs - lv)))].round(3).tolist(),
                                              xi_generations=pr["xi"][int(np.argmin(np.abs(GRID - lv)))].round(3).tolist(),
                                              Tpar_gaussian_over_T2=(gp["a"][k][int(np.argmin(np.abs(xs - lv)))] / st["T2"]).round(3).tolist(),
                                              Tpar_generations_over_T2=pr["Tpar"][int(np.argmin(np.abs(GRID - lv)))].round(3).tolist())
                               for lv in (0.1, 0.2, 0.35, 0.5, 0.7)})
    ax[0, 0].set_ylabel("mass fraction"); ax[1, 0].set_ylabel(r"mean velocity $\hat\xi$")
    h = [plt.Line2D([], [], color=c, label=l) for c, l in (("C0", "generation 0 / fast population"), ("C4", "generations 1+2 / intermediate"), ("C3", "generation 3+ / slow"))] + \
        [plt.Line2D([], [], color="k", label="collision history"), plt.Line2D([], [], color="k", ls="--", label="Gaussian fit")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, "fig15_generations_and_populations." + e))
    plt.close(fig)

    # ---- figure 16: who are the molecules that move upstream? ----
    S["backward"] = {}
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
    for j, (otag, om, lab) in enumerate(OMS):
        t = f"w{otag}_M{MD}"; d = C[t]
        neg = d["vx"] < 0
        bk = d["Ggen"][:, :, neg].sum(2) * d["dv"]          # [gen, cell] density of molecules with xi_x < 0
        tot = bk.sum(0); c = d["core"]; c = c[(d["nstar"][c] > 1e-3)]
        cfar = np.arange(max(c.min() - 400, 0), c.max() + 1)
        cfar = cfar[tot[cfar] / d["rho"][cfar] > 3e-6]
        for g in range(4):
            ax[j].semilogy(d["x"][cfar], np.maximum(bk[g, cfar] / d["rho"][cfar], 1e-12), color=GCOL[g])
        ax[j].semilogy(d["x"][cfar], tot[cfar] / d["rho"][cfar], "k", lw=0.7)
        ax[j].set_ylim(1e-6, 1); ax[j].set_xlabel(r"$x/\ell_1$"); ax[j].set_title(f"({'abc'[j]}) {lab}", fontsize=9)
        rec = {}
        for b in (1e-2, 1e-3, 1e-4):
            i = cfar[np.argmin(np.abs(np.log(tot[cfar] / d["rho"][cfar]) - np.log(b)))]
            rec[f"beta={b:g}"] = dict(x=float(d["x"][i]), nstar=float(d["nstar"][i]), share_by_generation=(bk[:, i] / tot[i]).round(3).tolist(),
                                      all_molecules_by_generation=(d["mgen"][:, i, 0] / d["rho"][i]).round(4).tolist())
        S["backward"][t] = rec
    ax[0].set_ylabel(r"density of molecules with $\xi_x<0$, over $\rho$")
    h = [plt.Line2D([], [], color=GCOL[g], label=GLAB[g]) for g in range(4)] + [plt.Line2D([], [], color="k", lw=0.7, label="all")]
    fig.legend(handles=h, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(figdir, "fig16_precursor_composition." + e))
    plt.close(fig)

    # ---- loss of unscattered molecules for omega = 1: d(n0 u0)/dx = -A n0 (rho - n0), A = 2 ----
    S["erosion_maxwell"] = {}
    for t, d in C.items():
        if abs(d["omega"] - 1) > 1e-9:
            continue
        n0 = d["mgen"][0, :, 0]; flux = n0 * np.nan_to_num(d["mgen"][0, :, 1]); dx = d["x"][1] - d["x"][0]
        c = d["core"]; c = c[(d["nstar"][c] > 0.002) & (n0[c] / d["rho"][c] > 0.02)]
        rhs = -2.0 * n0 * (d["rho"] - n0)
        # integrated form between the two ends of the window, which avoids differentiating noisy data
        lhs = flux[c[-1]] - flux[c[0]]; pred = 0.5 * (rhs[c[:-1]] + rhs[c[1:]]).sum() * dx
        loc = np.gradient(flux, dx)[c]; k = slice(None, None, max(len(c) // 40, 1))
        S["erosion_maxwell"][t] = dict(flux_drop=float(lhs), integral_of_rate=float(pred), ratio=float(lhs / pred),
                                       local_ratio_median=float(np.median(loc[k] / rhs[c][k])))

    with open(os.path.join(root, "analysis", "out", "generations_summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    print(json.dumps(S["collapse"], indent=1)); print(json.dumps(S["erosion_maxwell"], indent=1))
    print(json.dumps(S["populations_vs_generations"], indent=1)); print(json.dumps(S["backward"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
