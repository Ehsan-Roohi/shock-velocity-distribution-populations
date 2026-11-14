"""Decay rate of the upstream-moving mass fraction, seed by seed, in DSMC and in the linear cascade model.

The statistic of the decay-rate table: the least-squares slope of ln(beta) against A*N over all cells (DSMC) or
tally cells (cascade) with LO < beta < HI, every cell with equal weight; A is the VHS collision-rate constant and
N the column density (DSMC: upstream of the station n* = 0.5; cascade: depth below the source plane, n1 = 1).
It is computed for every independent run; reported are the mean and the 95 % interval of the mean.

Usage: python rate_seeds.py dsmc <runs dir> <out json>            (files M<M>_w<omega>_s<seed>.{txt,bin})
       python rate_seeds.py cascade <dir> <prefix> <out json>     (files <prefix>_M<M>_w<omega>_s<seed>.txt)
"""
import sys, os, glob, re, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LO, HI = 1e-4, 1e-2
T95 = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def vhs_A(om):
    sref = 15 * math.sqrt(math.pi) / (2 * (5 - 2 * om) * (7 - 2 * om))
    return sref * 4 ** (om - 0.5) / math.gamma(2.5 - om)


def stat(v):
    v = np.array(v, float); n = len(v); sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    return dict(values=v.tolist(), mean=float(v.mean()), std=sd, ci95_half=float(T95.get(n, 2.0) * sd / math.sqrt(n)) if n > 1 else None, n=n)


def dsmc(runs, outj):
    import precursor_seeds as PS
    R = {}
    for case in sorted({re.sub(r"_s\d+\.txt$", "", os.path.basename(p)) for p in glob.glob(os.path.join(runs, "M*_w*_s*.txt"))}):
        sl = []; ncell = []
        for stem in sorted(p[:-4] for p in glob.glob(os.path.join(runs, case + "_s*.txt"))):
            h, cnt, ns = PS.read(stem); M, om = float(h["M"]), float(h["omega"])
            if M < 5 or int(h["steps_done"]) < 215000:
                continue
            x, beta, N, ic = PS.profile(h, cnt, ns)
            k = (np.arange(len(x)) < ic) & (beta < HI) & (beta > LO) & (x > x[0] + 5)
            if k.sum() < 8:
                continue
            sl.append(-np.polyfit(vhs_A(om) * N[k], np.log(beta[k]), 1)[0]); ncell.append(int(k.sum()))
        if len(sl) > 1:
            R[case] = dict(mach=M, omega=om, cells=ncell, **stat(sl))
            print(f"{case:14s} n={len(sl)} rate {R[case]['mean']:.4f} +- {R[case]['ci95_half']:.4f} (std {R[case]['std']:.4f}) cells {min(ncell)}-{max(ncell)}", flush=True)
    json.dump(R, open(outj, "w"), indent=1)


def cascade(d, prefix, outj):
    R = {}
    for case in sorted({re.sub(r"_s\d+\.txt$", "", os.path.basename(p)) for p in glob.glob(os.path.join(d, prefix + "_M*_w*_s*.txt"))}):
        m = re.search(r"_M(.+)_w(.+)$", case); M, om = float(m.group(1)), float(m.group(2)); sl = []; meta = []
        for f in sorted(glob.glob(os.path.join(d, case + "_s*.txt"))):
            a = np.loadtxt(f); k = (a[:, 1] < HI) & (a[:, 1] > LO)
            if k.sum() < 8:
                continue
            sl.append(-np.polyfit(-vhs_A(om) * a[k, 0], np.log(a[k, 1]), 1)[0])
            head = open(f).readlines()[:2]; meta.append(" ".join(l.strip("# \n") for l in head))
        if len(sl) > 1:
            R[case] = dict(mach=M, omega=om, header=meta[0], **stat(sl))
            print(f"{case:22s} n={len(sl)} rate {R[case]['mean']:.4f} +- {R[case]['ci95_half']:.4f} (std {R[case]['std']:.4f})", flush=True)
    json.dump(R, open(outj, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "dsmc":
        dsmc(sys.argv[2], sys.argv[3])
    else:
        cascade(sys.argv[2], sys.argv[3], sys.argv[4])
