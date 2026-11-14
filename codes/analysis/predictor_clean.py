"""Retest of the predictions that use the Mach-8 BGK solution, with BGK shocks recomputed for the reduced distributions
(bgk_reduced_shock.py), whose discrete Maxwellian has no transverse quadrature error.

  1. Compares each recomputed solution (tags M6r, M8r, M12r) with the archived one: upstream T_par/T_perp, conserved
     fluxes, differences of g and h at equal n*, of the moments, and of the upstream-moving mass fraction.
  2. Repeats the predictions of Mach 12 from Mach 6 and 8 (and Mach 8 from 6 and 12, Mach 6 from 5 and 8) with the
     recomputed solutions, by the three methods (three populations, direct, separate beam).

Usage: python predictor_clean.py <shock_data dir>        (needs vdf_reduced/M{6,8,12}r_BGK_reduced_vdf.npz and their
                                                          popfits in analysis/out/popfits)
Output: analysis/out/predictor_clean.json
"""
import sys, os, json, shutil, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl
import predictor as P
import predictor_direct as PD
import predictor_models as PM

PAIRS = [("M8r", "M8"), ("M6r", "M6_gl3fine"), ("M12r", "M12_gl3medium")]
TESTS = [("M12r", "M6r", "M8r", "extrap"), ("M12_gl3medium", "M6_gl3fine", "M8r", "extrap"), ("M12_gl3medium", "M6_gl3fine", "M8", "extrap"),
         ("M8r", "M6r", "M12r", "interp"), ("M8", "M6_gl3fine", "M12_gl3medium", "interp"),
         ("M6_gl3fine", "M5", "M8r", "interp"), ("M6_gl3fine", "M5", "M8", "interp"), ("M12_gl3medium", "M5", "M8r", "extrap")]
LEVELS = np.array([0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9])


def compare(root, new, old):
    red = os.path.join(root, "vdf_reduced")
    a, b = (sl.load_case(os.path.join(red, f"{t}_BGK_reduced_vdf.npz")) for t in (new, old))
    st = a["st"]; res = {}
    for nm, c in (("new", a), ("old", b)):
        v, w = c["vx"], c["wx"]
        i0 = int(np.argmin(np.abs(c["nstar"] - 1e-4))) if np.any(c["nstar"] > 1e-4) else 20
        i0 = max(5, min(i0, int(np.argmax(c["nstar"] > 1e-5))))
        mass = c["G"] @ (w * v); mom = c["G"] @ (w * v * v); en = 0.5 * (c["G"] @ (w * v ** 3) + c["H"] @ (w * v))
        res[nm] = dict(aniso_inflow=float(c["Tpar"][5] / c["Tperp"][5]), aniso_peak=float((c["Tpar"] / c["Tperp"]).max()),
                       flux_dev=[float(100 * np.abs(f / f[5] - 1).max()) for f in (mass, mom, en)],
                       rho2_ratio=float(c["rho"][-20] / st["rho2"]), T2_ratio=float(c["T"][-20] / st["T2"]),
                       q_peak=float(np.abs(c["q"]).max()), sigma_peak=float(np.abs(c["sigma"]).max()),
                       delta=float((st["rho2"] - st["rho1"]) / np.max(np.gradient(c["rho"], c["x"]))))
    xi = np.linspace(-6, 4, 2001); vv = st["u2"] + xi * (st["u1"] - st["u2"])
    oa, ob = np.argsort(a["nstar"]), np.argsort(b["nstar"])
    dg, dh, dbeta, beta = [], [], [], []
    for lv in LEVELS:
        row = []
        for c, o in ((a, oa), (b, ob)):
            j = o[int(np.argmin(np.abs(c["nstar"][o] - lv)))]
            # linear interpolation in n* between the two neighbouring stations
            k = int(np.searchsorted(c["nstar"][o], lv)); k = min(max(k, 1), len(o) - 1)
            j0, j1 = o[k - 1], o[k]; t = (lv - c["nstar"][j0]) / (c["nstar"][j1] - c["nstar"][j0])
            g = (1 - t) * np.interp(vv, c["vx"], c["G"][j0]) + t * np.interp(vv, c["vx"], c["G"][j1])
            h = (1 - t) * np.interp(vv, c["vx"], c["H"][j0]) + t * np.interp(vv, c["vx"], c["H"][j1])
            row.append((g, h))
        (ga, ha), (gb, hb) = row
        dg.append(float(100 * np.linalg.norm(ga - gb) / np.linalg.norm(gb))); dh.append(float(100 * np.linalg.norm(ha - hb) / np.linalg.norm(hb)))
        ba, bb = (float(np.trapezoid(np.where(vv < 0, g, 0.0), vv) / np.trapezoid(g, vv)) for g in (ga, gb))
        beta.append([ba, bb])
    G = np.linspace(0.03, 0.93, 46)
    prof = lambda c, o, k: np.interp(G, c["nstar"][o], c[k][o])
    for k in ("q", "sigma", "T", "Tpar", "Tperp"):
        pa, pb = prof(a, oa, k), prof(b, ob, k)
        res[f"{k}_L2_diff"] = float(100 * np.linalg.norm(pa - pb) / np.linalg.norm(pb))
    res.update(levels=LEVELS.tolist(), g_diff=dg, h_diff=dh, beta_new_old=beta, mach=st["M1"])
    fits = os.path.join(root, "analysis", "out", "popfits")
    try:
        fa, fb = (np.load(os.path.join(fits, f"{t}_BGK_branch3.npz")) for t in (new, old))
        Qa, Qb = P.scaled(fa, a["st"]), P.scaled(fb, b["st"])
        res["pop_param_max_diff"] = [float(np.abs(Qa[:, :, k] - Qb[:, :, k]).max()) for k in range(Qa.shape[2])]
        i = int(np.argmin(np.abs(P.GRID - 0.1)))
        res["pop_at_0p1_new"] = Qa[i].round(3).tolist(); res["pop_at_0p1_old"] = Qb[i].round(3).tolist()
    except (FileNotFoundError, OSError) as e:
        res["pop_param_max_diff"] = None; print("   no branch fits:", e)
    return res


def main(root):
    out = dict(compare={}, tests={})
    for new, old in PAIRS:
        out["compare"][new] = r = compare(root, new, old)
        print(f"{new} vs {old}: inflow T_par/T_perp {r['new']['aniso_inflow']:.5f} (old {r['old']['aniso_inflow']:.4f}); peak {r['new']['aniso_peak']:.3f} (old {r['old']['aniso_peak']:.3f}); "
              f"flux dev % {np.round(r['new']['flux_dev'], 3).tolist()}; delta {r['new']['delta']:.3f} (old {r['old']['delta']:.3f})")
        print("   g diff % at n* levels", np.round(r["g_diff"], 2).tolist(), "\n   h diff %", np.round(r["h_diff"], 2).tolist())
        print("   profile L2 diff %: " + "  ".join(f"{k} {r[k + '_L2_diff']:.2f}" for k in ("q", "sigma", "T", "Tpar", "Tperp")), "| beta(new, old):", np.round(r["beta_new_old"], 4).tolist())
        if r.get("pop_param_max_diff"):
            print("   max difference of scaled population parameters:", np.round(r["pop_param_max_diff"], 3).tolist())
    # predictions: run the two existing drivers in a scratch tree so that their summary files are not overwritten
    tmp = tempfile.mkdtemp(prefix="clean_")
    for d in ("vdf_reduced", os.path.join("analysis", "out", "popfits"), os.path.join("paper", "figures")):
        os.makedirs(os.path.join(tmp, d), exist_ok=True)
    tags = sorted({t for test in TESTS for t in test[:3]})
    for t in tags:
        shutil.copy(os.path.join(root, "vdf_reduced", f"{t}_BGK_reduced_vdf.npz"), os.path.join(tmp, "vdf_reduced"))
        for suf in ("branch3", "popfit"):
            shutil.copy(os.path.join(root, "analysis", "out", "popfits", f"{t}_BGK_{suf}.npz"), os.path.join(tmp, "analysis", "out", "popfits"))
    P.TESTS = TESTS
    try:
        P.main(tmp)
    except KeyError:
        pass                     # the figure of the standard tests is not made here
    PD.BGK_TESTS = [t[:3] for t in TESTS]; PM.SETS = {}
    PD.main(tmp)
    A = json.load(open(os.path.join(tmp, "analysis", "out", "predictor_summary.json"))); D = json.load(open(os.path.join(tmp, "analysis", "out", "predictor_direct.json")))
    keys = ["g_max", "q_L2", "sigma_L2", "T_max", "width_ratio", "flux_mass", "flux_mom", "flux_energy", "xspace_nstar_rms", "xspace_q_rms", "xspace_sigma_rms", "monotone"]
    for tgt, lo, hi, _ in TESTS:
        k = f"{tgt}_from_{lo}_{hi}"
        p = {a[:-5]: b for a, b in A[k].items() if a.endswith("_pred")}
        out["tests"][k] = dict(P=p, D=D["BGK|" + k], H=D["BGK|" + k + "|hybrid"], fit_g_max=A[k]["g_max_fit"])
        for nm, r in (("P", p), ("D", D["BGK|" + k]), ("H", D["BGK|" + k + "|hybrid"])):
            print(f"{nm} {k:42s}", " ".join((f"{r[q]:6.2f}" if isinstance(r.get(q), float) else f"{str(r.get(q)):>6s}") for q in keys))
    json.dump(out, open(os.path.join(root, "analysis", "out", "predictor_clean.json"), "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1])
