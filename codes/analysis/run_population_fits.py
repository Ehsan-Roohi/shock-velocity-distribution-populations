"""Population (bi-Maxwellian mixture) fits across the whole shock for every case.

For K = 1..KMAX populations and ~100 stations per case, g and h are fitted jointly. Each K is
swept upstream->downstream and back with warm starts from the neighbouring station, RH-informed
templates and a greedy seed built from the K-1 solution; the lowest-cost fit is kept.

Usage: python run_population_fits.py <out dir> <reduced file> [<reduced file> ...]
"""
import sys, os, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl

KMAX = 4
NSTAT = 100


def stations(case):
    ns = case["nstar"]
    core = np.where((ns > 0.002) & (ns < 0.998))[0]
    lo, hi = core.min(), core.max()
    a = [int(np.argmin(np.abs(ns - t))) for t in np.linspace(0.002, 0.998, int(0.6 * NSTAT))]
    b = list(np.linspace(lo, hi, int(0.4 * NSTAT)).astype(int))
    return np.array(sorted(set(a + b)))


def run_case(args):
    path, out = args
    t0 = time.time()
    c = sl.load_case(path)
    st, v, w = c["st"], c["vx"], c["wx"]
    idx = stations(c)
    n = len(idx)
    P = {K: [None] * n for K in range(1, KMAX + 1)}
    C = {K: np.full(n, np.inf) for K in range(1, KMAX + 1)}
    R = {K: np.full((n, 2), np.nan) for K in range(1, KMAX + 1)}

    def attempt(K, j, seeds):
        s = idx[j]
        seeds = [p for p in seeds if p is not None and len(p) == 4 * K]
        if not seeds:
            return
        r = sl.fit_populations(v, w, c["G"][s], c["H"][s], seeds)
        if r is not None and r[1] < C[K][j]:
            P[K][j], C[K][j], R[K][j] = sl.sort_by_velocity(r[0]), r[1], (r[2], r[3])

    for K in range(1, KMAX + 1):
        for sweep in (range(n), range(n - 1, -1, -1), range(n)):
            for j in sweep:
                s = idx[j]
                loc = (c["rho"][s], c["u"][s], c["T"][s])
                seeds = [sl.seed_templates(K, loc, st)]
                for jj in (j - 1, j + 1):
                    if 0 <= jj < n:
                        seeds.append(P[K][jj])
                seeds.append(P[K][j])
                if K > 1 and P[K - 1][j] is not None:
                    seeds.append(sl.add_population(P[K - 1][j], v, w, c["G"][s], st))
                attempt(K, j, seeds)
    save = dict(idx=idx, x=c["x"][idx], nstar=c["nstar"][idx], mach=st["M1"],
                rho=c["rho"][idx], u=c["u"][idx], T=c["T"][idx], q=c["q"][idx], sigma=c["sigma"][idx],
                M300=c["M300"][idx], M400=c["M400"][idx], Tpar=c["Tpar"][idx], Tperp=c["Tperp"][idx],
                qmax=np.abs(c["q"]).max(), sigmamax=np.abs(c["sigma"]).max(),
                M300max=np.abs(c["M300"]).max(), M400max=np.abs(c["M400"]).max(),
                eps_rh2=sl.two_maxwellian_residual(c)[idx])
    for K in range(1, KMAX + 1):
        par = np.array([p if p is not None else np.full(4 * K, np.nan) for p in P[K]])
        mom = [sl.pop_moments(p) if p is not None else None for p in P[K]]
        save[f"p{K}"] = par
        save[f"res{K}"] = R[K]
        for key in ("q", "sigma", "M300", "M400"):
            save[f"{key}_fit{K}"] = np.array([m[key] if m else np.nan for m in mom])
        save[f"qk{K}"] = np.array([m["q_k"] if m else np.full(K, np.nan) for m in mom])
        save[f"sigmak{K}"] = np.array([m["sigma_k"] if m else np.full(K, np.nan) for m in mom])
    np.savez(os.path.join(out, c["name"] + "_popfit.npz"), **save)
    med = [float(np.nanmedian(R[K][:, 0])) for K in range(1, KMAX + 1)]
    mx = [float(np.nanmax(R[K][:, 0])) for K in range(1, KMAX + 1)]
    return f"{c['name']:22s} n={n:3d} {time.time()-t0:6.0f}s  max res_g K=1..4: " + " ".join(f"{100*a:6.2f}%" for a in mx) + "  median: " + " ".join(f"{100*a:5.2f}%" for a in med)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    jobs = [(p, out) for p in sys.argv[2:]]
    with Pool(min(6, len(jobs))) as pool:
        for line in pool.imap_unordered(run_case, jobs):
            print(line, flush=True)
