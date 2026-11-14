"""Stability of the three-population fit.

For selected cases and stations, the K = 3 solution of the continuous branch is compared with
  (a) fits of two statistically independent halves of the DSMC data (seeds 1-4 and 5-8), started from it;
  (b) fits started from random perturbations of it (30 starts; solutions whose cost is within 5 % of the lowest
      are kept and their spread is reported, together with the number of starts that end elsewhere);
  (c) fits with the weight of the h residual changed by factors 1/4 and 4, and a fit of g alone for the
      streamwise parameters.
Reported per population: mass fraction, scaled mean velocity, T_par/T2, T_perp/T2, and its contribution to the
heat flux and to the stress in units of the peak of the total.

Usage: python fit_stability.py <popfit dir> <full reduced dir> <half A dir> <half B dir> <out json> <case> [...]
       case = file stem, e.g. DSMC_w0p5_M8
"""
import sys, os, json
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl

LEVELS = (0.1, 0.2, 0.35, 0.5, 0.7)


def fit(v, w, g, h, p0, wh=1.0, nfev=600):
    sw = np.sqrt(w); gn, hn = np.linalg.norm(g * sw), np.linalg.norm(h * sw) / np.sqrt(wh)
    try:
        r = least_squares(lambda p: sl._res_jac(p, v, sw, g, h, gn, hn)[0], p0, jac=lambda p: sl._res_jac(p, v, sw, g, h, gn, hn)[1], method="lm", max_nfev=nfev)
    except Exception:
        return None
    if not np.all(np.isfinite(r.x)):
        return None
    gk, hk = sl.pop_model(r.x, v)
    return r.x, float(np.linalg.norm((gk.sum(0) - g) * sw) / gn), float(r.cost)


def describe(p, st, qmax, smax):
    q = p.reshape(3, 4); order = np.argsort(q[:, 1]); q = q[order]            # slow, intermediate, fast
    n = np.exp(q[:, 0]); m = sl.pop_moments(q.ravel())
    return dict(frac=(n / n.sum()).tolist(), xi=((q[:, 1] - st["u2"]) / (st["u1"] - st["u2"])).tolist(), Tpar=(np.exp(q[:, 2]) / st["T2"]).tolist(),
                Tperp=(np.exp(q[:, 3]) / st["T2"]).tolist(), q_share=(np.asarray(m["q_k"]) / qmax).tolist(), sigma_share=(np.asarray(m["sigma_k"]) / smax).tolist())


def at_level(c, lv):
    core = np.where((c["nstar"] > 0.005) & (c["nstar"] < 0.995))[0]; k = np.arange(core.min() - 1, core.max() + 2)
    ns = np.maximum.accumulate(c["nstar"][k]); j = max(int(np.argmax(ns >= lv)), 1); a = (lv - ns[j - 1]) / (ns[j] - ns[j - 1])
    return (1 - a) * c["G"][k[j - 1]] + a * c["G"][k[j]], (1 - a) * c["H"][k[j - 1]] + a * c["H"][k[j]]


def main(fitdir, full, ha, hb, out, cases):
    rng = np.random.default_rng(7)
    R = {}
    for name in cases:
        c = sl.load_case(os.path.join(full, name + "_reduced_vdf.npz")); st = c["st"]; v, w = c["vx"], c["wx"]
        b = np.load(os.path.join(fitdir, name + "_branch3.npz"))
        ok = np.all(np.isfinite(b["p3"]), 1)
        qmax, smax = float(np.abs(c["q"]).max()), float(np.abs(c["sigma"]).max())
        halves = [sl.load_case(os.path.join(d, name + "_reduced_vdf.npz")) if os.path.exists(os.path.join(d, name + "_reduced_vdf.npz")) else None for d in (ha, hb)]
        R[name] = {}
        for lv in LEVELS:
            j = np.where(ok)[0][np.argmin(np.abs(b["nstar"][ok] - lv))]
            p0 = b["p3"][j]
            g, h = at_level(c, float(b["nstar"][j]))
            base = fit(v, w, g, h, p0)
            rec = dict(nstar=float(b["nstar"][j]), base=describe(base[0], st, qmax, smax), base_residual=base[1])
            # (a) independent halves
            hv = []
            for hc in halves:
                if hc is None:
                    continue
                gg, hh = at_level(hc, float(b["nstar"][j])); r = fit(v, w, gg, hh, base[0])
                if r is not None:
                    hv.append(describe(r[0], st, qmax, smax))
            if len(hv) == 2:
                rec["halves_max_abs_difference"] = {k: float(np.abs(np.array(hv[0][k]) - np.array(hv[1][k])).max()) for k in hv[0]}
            # (b) random starts
            sols = []; nfail = 0
            for _ in range(30):
                q = base[0].copy().reshape(3, 4)
                q[:, 0] += rng.normal(0, 0.5, 3); q[:, 1] += rng.normal(0, 0.25, 3) * (st["u1"] - st["u2"]); q[:, 2:] += rng.normal(0, 0.5, (3, 2))
                r = fit(v, w, g, h, q.ravel())
                if r is None:
                    nfail += 1
                else:
                    sols.append(r)
            best = min(s[2] for s in sols)
            good = [describe(s[0], st, qmax, smax) for s in sols if s[2] <= 1.05 * best]
            rec["random_starts"] = dict(n_started=30, n_failed=nfail, n_within_5pct_of_lowest_cost=len(good), lowest_cost_over_base=float(best / base[2]),
                                        spread={k: float(np.ptp(np.array([d[k] for d in good]), axis=0).max()) for k in good[0]},
                                        other_minima_residuals=sorted({round(s[1], 4) for s in sols if s[2] > 1.05 * best})[:6])
            # (c) weight of the h residual
            rec["weight"] = {}
            for wh in (0.25, 4.0):
                r = fit(v, w, g, h, base[0], wh=wh)
                if r is not None:
                    d = describe(r[0], st, qmax, smax)
                    rec["weight"][f"{wh:g}"] = {k: float(np.abs(np.array(d[k]) - np.array(rec["base"][k])).max()) for k in d}
            R[name][f"{lv}"] = rec
            print(name, lv, "frac", np.round(rec["base"]["frac"], 3), "q share", np.round(rec["base"]["q_share"], 3), "halves", {k: round(x, 4) for k, x in rec.get("halves_max_abs_difference", {}).items()},
                  "starts", rec["random_starts"]["n_within_5pct_of_lowest_cost"], "spread", {k: round(x, 4) for k, x in rec["random_starts"]["spread"].items()}, flush=True)
    json.dump(R, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:])
