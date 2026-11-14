"""Three-population predictor (see predictor.py) applied to the Shakhov solutions and to DSMC.

The position step uses  dx/dn* = (d Phi_2/dn*) / (-nu sigma_xx)  with nu = rho T^(1-omega). This is
exact for the Shakhov model and for Maxwell molecules (omega = 1); for a VHS gas with omega < 1 the
stress production is not exactly -(p/mu) sigma_xx far from equilibrium, so there the same formula is
an approximation and the check on the simulation itself measures its error.

Usage: python predictor_models.py <shock_data dir>
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl
import predictor as P

GRID = P.GRID
# (target, source, source): the nine tests of the BGK table (Mach 2 in place of 2.5) and two more at Mach 10 and 12
DSMC_TESTS = [(3, 2, 4), (4, 3, 5), (5, 4, 6), (6, 5, 8), (8, 6, 12), (10, 8, 12),
              (8, 5, 6), (12, 6, 8), (12, 5, 8), (12, 8, 10), (12, 5, 6),
              (12, 8, 25), (25, 8, 12), (25, 10, 12), (25, 6, 12), (25, 5, 8)]    # Mach 25: skipped where the simulation is not available
SETS = {
    "Shakhov, mu~T": (1.0, "vdf_models", "Shakhov_M{m}", [(3, 2, 4), (4, 3, 5), (5, 3, 4), (2, 3, 4)]),
    "DSMC, omega=1": (1.0, "vdf_dsmc", "DSMC_w1_M{m}", DSMC_TESTS),
    "DSMC, omega=0.81": (0.81, "vdf_dsmc", "DSMC_w0p81_M{m}", DSMC_TESTS),
    "DSMC, hard sphere": (0.5, "vdf_dsmc", "DSMC_w0p5_M{m}", DSMC_TESTS),
}


def main(root):
    fits = os.path.join(root, "analysis/out/popfits")
    out = {}
    for label, (om, folder, pat, tests) in SETS.items():
        cache = {}
        def get(m):
            if m not in cache:
                name = pat.format(m=m)
                c = sl.load_case(os.path.join(root, folder, name + "_reduced_vdf.npz"))
                f = np.load(os.path.join(fits, c["name"] + "_branch3.npz"))
                cache[m] = (c, P.scaled(f, c["st"]))
            return cache[m]
        for tgt, lo, hi in tests:
            try:
                (c, Qf), (cl, Ql), (ch, Qh) = get(tgt), get(lo), get(hi)
            except (FileNotFoundError, OSError):
                continue
            st = c["st"]; s = lambda M: 1 / M ** 2
            wH = (s(tgt) - s(lo)) / (s(hi) - s(lo))
            kind = "interp" if 0 <= wH <= 1 else "extrap"
            v, w = c["vx"], c["wx"]; sw = np.sqrt(w)
            core = np.where((c["nstar"] > 0.005) & (c["nstar"] < 0.97))[0]
            k = np.arange(core.min() - 2, core.max() + 3)          # contiguous cells across the shock, no smoothing
            ns_all = np.maximum.accumulate(c["nstar"][k])
            ref = {key: np.interp(GRID, ns_all, c[key][k]) for key in ("q", "sigma", "T", "rho", "x")}
            Phi2 = np.interp(GRID, ns_all, (c["G"] @ (v ** 3 * w))[k])
            Gref = np.array([np.interp(GRID, ns_all, c["G"][k, i]) for i in range(len(v))]).T   # g at exact n*
            xd = ref["x"] - np.interp(0.5, GRID, ref["x"])
            dxdn_d = np.gradient(xd, GRID); delta = 1 / np.max(1 / dxdn_d)
            width = lambda xx: float(np.interp(0.8, GRID, xx) - np.interp(0.2, GRID, xx))
            xm, _ = P.x_of_nstar(dict(Phi2=Phi2, sigma=ref["sigma"]), ref["rho"] * ref["T"] ** (1 - om))
            res = dict(model=label, mach=tgt, sources=[lo, hi], kind=kind,
                       closure_check_width_ratio=width(xm) / width(xd),
                       closure_check_x_rms=float(np.sqrt(np.mean((xm - xd) ** 2)) / delta))
            Qnear = Qh if abs(s(hi) - s(tgt)) < abs(s(lo) - s(tgt)) else Ql      # baseline: scaled parameters of the nearer source, unchanged
            for name, Q in (("pred", (1 - wH) * Ql + wH * Qh), ("fit", Qf), ("base", Qnear)):
                n, u, a, b = P.unscale(Q, st)
                F = P.mixture_fields(n, u, a, b)
                xp, dxdn = P.x_of_nstar(F, F["rho"] * F["T"] ** (1 - om))
                eg = []
                for i, ns in enumerate(GRID):
                    g = Gref[i]
                    p = np.stack([np.log(n[i]), u[i], np.log(a[i]), np.log(b[i])], 1).ravel()
                    gp = sl.pop_model(p, v)[0].sum(0) * (g @ w) / n[i].sum()
                    eg.append(np.linalg.norm((gp - g) * sw) / np.linalg.norm(g * sw))
                res.update({f"g_max_{name}": float(100 * np.max(eg)),
                            f"q_L2_{name}": float(100 * np.linalg.norm(F["q"] - ref["q"]) / np.linalg.norm(ref["q"])),
                            f"sigma_L2_{name}": float(100 * np.linalg.norm(F["sigma"] - ref["sigma"]) / np.linalg.norm(ref["sigma"])),
                            f"T_max_{name}": float(100 * np.max(np.abs(F["T"] - ref["T"]) / ref["T"])),
                            f"monotone_{name}": bool(np.all(dxdn > 0)),
                            f"width_ratio_{name}": width(xp) / width(xd),
                            f"x_rms_{name}": float(np.sqrt(np.mean((xp - xd) ** 2)) / delta)})
                res.update(P.extra_checks(F, n, u, a, b, st, ref, xd, xp, dxdn, name))
                if not np.all(dxdn > 0):
                    res[f"monotone_smoothed_{name}"] = P.smoothed_monotone(Q, st, lambda F_: F_["rho"] * F_["T"] ** (1 - om))
            out[f"{label}|M{tgt} from {lo},{hi}"] = res
            print(f"{label:18s} {kind:6s} M{tgt} from {lo},{hi}: g max {res['g_max_pred']:5.2f}% (fit {res['g_max_fit']:4.2f}) q {res['q_L2_pred']:5.2f}% "
                  f"sigma {res['sigma_L2_pred']:5.2f}% Tmax {res['T_max_pred']:4.2f}% | width ratio {res['width_ratio_pred']:.3f} (fit {res['width_ratio_fit']:.3f}; "
                  f"closure on data {res['closure_check_width_ratio']:.3f}) x rms {res['x_rms_pred']:.3f} (fit {res['x_rms_fit']:.3f}; data {res['closure_check_x_rms']:.3f}) monotone {int(res['monotone_pred'])}", flush=True)
    json.dump(out, open(os.path.join(root, "analysis", "out", "predictor_models_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
