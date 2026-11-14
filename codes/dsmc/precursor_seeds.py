"""Decay of the upstream-moving mass fraction, seed by seed.

For every case found in <runs dir> (files M<M>_w<omega>_s<seed>.{txt,bin}) and every seed, beta(x) and the column
density N(x) upstream of the station where n* = 0.5 are computed from that seed alone, and
    ln(beta) = a - b N^p
is fitted in a window of beta with p free (grid in p, linear least squares for a and b) and with p fixed at 1, 2/3
and 2/(2 omega + 1). Reported: p of every seed, mean, standard deviation, 95 % interval of the mean, and the
root-mean-square deviation of ln(beta) for each fixed p (mean over seeds); the same for the pooled data.

Usage: python precursor_seeds.py <runs dir> <out json> [min_steps]
"""
import sys, os, glob, re, json, math
import numpy as np

GAM = 5.0 / 3.0
WINDOWS = {"1e-2..2e-5": (2e-5, 1e-2), "1e-2..2e-4": (2e-4, 1e-2), "3e-3..2e-5": (2e-5, 3e-3)}
PGRID = np.round(np.arange(0.40, 1.301, 0.005), 3)
T95 = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def read(stem):
    h = dict(l.split() for l in open(stem + ".txt") if l.strip())
    nc, nb, NB = (int(h[k]) for k in ("nc", "nb", "NB"))
    a = np.fromfile(stem + ".bin")
    return h, a[:nc * nb].reshape(nc, nb), a[-NB:].sum()


def profile(h, cnt, ns):
    M = float(h["M"]); dx, dv, FN = float(h["dx"]), float(h["dv"]), float(h["FN"])
    r = (GAM + 1) * M ** 2 / ((GAM - 1) * M ** 2 + 2)
    x = float(h["xmin"]) + (np.arange(cnt.shape[0]) + 0.5) * dx; v = float(h["vlo"]) + (np.arange(cnt.shape[1]) + 0.5) * dv
    tot = cnt.sum(1); rho = tot * FN / (dx * ns); beta = cnt[:, v < 0].sum(1) / np.maximum(tot, 1)
    nstar = (rho - 1) / (r - 1); ic = int(np.argmin(np.abs(nstar - 0.5)))
    N = np.concatenate([np.cumsum(rho[:ic][::-1])[::-1] * dx, np.zeros(len(x) - ic)])
    return x, beta, N, ic


def fits(x, beta, N, ic, om):
    out = {}
    for wn, (lo, hi) in WINDOWS.items():
        k = (np.arange(len(x)) < ic) & (beta < hi) & (beta > lo) & (x > x[0] + 5)
        if k.sum() < 12:
            continue
        lb = np.log(beta[k]); Nn = N[k]
        rms = {}
        for p in PGRID:
            co = np.polyfit(Nn ** p, lb, 1); rms[float(p)] = float(np.sqrt(np.mean((np.polyval(co, Nn ** p) - lb) ** 2)))
        pb = min(rms, key=rms.get)
        near = lambda q: rms[float(PGRID[np.argmin(np.abs(PGRID - q))])]
        out[wn] = dict(p=pb, rms=rms[pb], rms_p1=near(1.0), rms_p23=near(2 / 3), rms_heur=near(2 / (2 * om + 1)), n=int(k.sum()),
                       x_upstream_end=float(x[k].min()), column_max=float(Nn.max()))
    return out


def main(runs, outj, min_steps=215000):
    cases = sorted({re.sub(r"_s\d+\.txt$", "", os.path.basename(p)) for p in glob.glob(os.path.join(runs, "M*_w*_s*.txt"))})
    R = {}
    for case in cases:
        per = []; pool = None
        for stem in sorted(p[:-4] for p in glob.glob(os.path.join(runs, case + "_s*.txt"))):
            h, cnt, ns = read(stem)
            if int(h["steps_done"]) < min_steps:
                continue
            om = float(h["omega"]); M = float(h["M"])
            if M < 5:
                break
            per.append(fits(*profile(h, cnt, ns), om))
            pool = [cnt, ns, h] if pool is None else [pool[0] + cnt, pool[1] + ns, h]
        if not per:
            continue
        rec = dict(mach=M, omega=om, seeds=len(per), xmin=float(pool[2]["xmin"]), pooled=fits(*profile(pool[2], pool[0], pool[1]), om), windows={})
        for wn in WINDOWS:
            ps = np.array([d[wn]["p"] for d in per if wn in d])
            if len(ps) < 2:
                continue
            sd = float(ps.std(ddof=1)); n = len(ps)
            rec["windows"][wn] = dict(p_seeds=ps.tolist(), p_mean=float(ps.mean()), p_std=sd, ci95_half=float(T95.get(n, 2.0) * sd / math.sqrt(n)),
                                      rms_free=float(np.mean([d[wn]["rms"] for d in per if wn in d])), rms_p1=float(np.mean([d[wn]["rms_p1"] for d in per if wn in d])),
                                      rms_p23=float(np.mean([d[wn]["rms_p23"] for d in per if wn in d])), rms_heur=float(np.mean([d[wn]["rms_heur"] for d in per if wn in d])))
        R[case] = rec
        w = rec["windows"].get("1e-2..2e-5") or rec["windows"].get("1e-2..2e-4")
        if w:
            print(f"{case:14s} seeds {rec['seeds']} xmin {rec['xmin']:g}  p = {w['p_mean']:.3f} +- {w['ci95_half']:.3f} (std {w['p_std']:.3f})  pooled p {rec['pooled'].get('1e-2..2e-5', rec['pooled'].get('1e-2..2e-4', {})).get('p')}  rms free/p=1/2/3/heur {w['rms_free']:.3f} {w['rms_p1']:.3f} {w['rms_p23']:.3f} {w['rms_heur']:.3f}", flush=True)
    json.dump(R, open(outj, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 215000)
