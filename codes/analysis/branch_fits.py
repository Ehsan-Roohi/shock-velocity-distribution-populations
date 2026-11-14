"""Continuous three-population branch.

The lowest-residual K = 3 fits (run_population_fits.py) can switch between different local minima
from one station to the next where populations overlap. Here a single branch is followed: the fit
starts at the most upstream station from the best solution found there and marches downstream,
each station warm-started only from its upstream neighbour. Outputs have the same keys as the
K = 3 entries of the popfit files.

Usage: python branch_fits.py <popfit dir> <reduced file> [...]
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl


def main(fitdir, files):
    for path in files:
        c = sl.load_case(path)
        f = np.load(os.path.join(fitdir, c["name"] + "_popfit.npz"))
        idx, v, w = f["idx"], c["vx"], c["wx"]
        o = np.argsort(c["x"][idx]); idx = idx[o]
        start = next(j for j in range(len(idx)) if np.all(np.isfinite(f["p3"][o][j])) and c["nstar"][idx[j]] > 0.01)
        P = np.full((len(idx), 12), np.nan); R = np.full((len(idx), 2), np.nan)
        p = f["p3"][o][start].copy()
        for j in range(start, len(idx)):
            s = idx[j]
            r = sl.fit_populations(v, w, c["G"][s], c["H"][s], [p], max_nfev=400)
            if r is None:
                # the march from the upstream neighbour failed numerically: retry from slightly broadened populations,
                # then from the best solution stored for this station with its populations matched to the branch by velocity
                alt = []
                for dlt in (0.1, 0.3):
                    q = p.copy().reshape(3, 4); q[:, 2:] += dlt; alt.append(q.ravel())
                pb = f["p3"][o][j]
                if np.all(np.isfinite(pb)):
                    qb = pb.reshape(3, 4); cur = p.reshape(3, 4)
                    order = [int(np.argmin(np.abs(qb[:, 1] - cur[k, 1]))) for k in range(3)]
                    if len(set(order)) == 3:
                        alt.append(qb[order].ravel())
                for q in alt:
                    r = sl.fit_populations(v, w, c["G"][s], c["H"][s], [q], max_nfev=400)
                    if r is not None:
                        break
            if r is None:
                continue
            p = r[0]; P[j] = p; R[j] = (r[2], r[3])   # order kept: no re-sorting along the branch
        mom = [sl.pop_moments(q) if np.all(np.isfinite(q)) else None for q in P]
        best = f["res3"][o][:, 0]
        out = dict(idx=idx, x=c["x"][idx], nstar=c["nstar"][idx], mach=c["st"]["M1"], p3=P, res3=R,
                   qk3=np.array([m["q_k"] if m else np.full(3, np.nan) for m in mom]),
                   sigmak3=np.array([m["sigma_k"] if m else np.full(3, np.nan) for m in mom]))
        for key in ("q", "sigma", "M300", "M400"):
            out[f"{key}_fit3"] = np.array([m[key] if m else np.nan for m in mom])
            out[key] = c[key][idx]; out[key + "max"] = np.abs(c[key]).max()
        np.savez(os.path.join(fitdir, c["name"] + "_branch3.npz"), **out)
        k = (out["nstar"] > 0.01) & (out["nstar"] < 0.99) & np.isfinite(R[:, 0])
        worse = np.sum(R[k, 0] > 1.5 * best[k] + 1e-4)
        print(f"{c['name']:20s} branch max res_g {100*np.nanmax(R[k,0]):5.2f}% (best-of-seeds {100*np.nanmax(best[k]):5.2f}%), "
              f"stations worse than 1.5x best: {worse}/{k.sum()}; max moment err q {100*np.nanmax(np.abs(out['q_fit3'][k]-out['q'][k]))/out['qmax']:5.2f}% "
              f"sigma {100*np.nanmax(np.abs(out['sigma_fit3'][k]-out['sigma'][k]))/out['sigmamax']:5.2f}% M400 {100*np.nanmax(np.abs(out['M400_fit3'][k]-out['M400'][k]))/out['M400max']:5.2f}%", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
