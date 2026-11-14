"""Free-flight flag of dsmc_shock_vdf5: share of the upstream-moving molecules that have flown without any collision
from a reference plane (plane 1: the station where beta = 1e-2; plane 2: x = 0; plane 3: half-way), at the stations
where beta takes given values. Per run, mean over runs and 95 % interval of the mean; and from the pooled runs.

Usage: python fly_analysis.py <runs dir> <out json>      (files M<M>_w<omega>_s<seed>.{txt,bin,fly.bin})
"""
import sys, os, glob, re, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precursor_seeds as PS

LEVELS = (3e-3, 1e-3, 3e-4, 1e-4)
T95 = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}


def shares(h, cnt, ns, fly):
    x, beta, N, ic = PS.profile(h, cnt, ns)
    k = np.where((np.arange(len(x)) < ic) & (x > x[0] + 5) & (beta > 3e-6))[0]
    out = {}
    for b in LEVELS:
        i = k[np.argmin(np.abs(np.log(beta[k]) - math.log(b)))]; s = slice(i - 2, i + 3)       # five cells (0.5 l1)
        out[f"{b:g}"] = dict(x=float(x[i]), column=float(N[i]), flagged=[float(fly[j, s].sum() / fly[0, s].sum()) for j in (1, 2, 3)],
                             beta_check=float(fly[0, s].sum() / cnt[s].sum()))
    return out


MIN_STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 215000


def main(runs, outj):
    R = {}
    for case in sorted({re.sub(r"_s\d+\.txt$", "", os.path.basename(p)) for p in glob.glob(os.path.join(runs, "M*_w*_s*.txt"))}):
        per = []; pool = None
        for stem in sorted(p[:-4] for p in glob.glob(os.path.join(runs, case + "_s*.txt"))):
            h, cnt, ns = PS.read(stem)
            if int(h["steps_done"]) < MIN_STEPS or not os.path.exists(stem + ".fly.bin"):
                continue
            fly = np.fromfile(stem + ".fly.bin").reshape(4, -1)
            per.append(shares(h, cnt, ns, fly))
            pool = [cnt, ns, fly, h] if pool is None else [pool[0] + cnt, pool[1] + ns, pool[2] + fly, h]
        if len(per) < 2:
            continue
        rec = dict(mach=float(pool[3]["M"]), omega=float(pool[3]["omega"]), runs=len(per), pooled=shares(pool[3], pool[0], pool[1], pool[2]), levels={})
        for b in LEVELS:
            key = f"{b:g}"; v = np.array([p[key]["flagged"] for p in per]); n = len(v)
            rec["levels"][key] = dict(mean=v.mean(0).tolist(), ci95_half=(T95.get(n, 2.0) * v.std(0, ddof=1) / math.sqrt(n)).tolist())
        R[case] = rec
        print(case, "runs", len(per), " | ".join(f"beta={b:g}: plane1 {100*rec['levels'][f'{b:g}']['mean'][0]:.2f}+-{100*rec['levels'][f'{b:g}']['ci95_half'][0]:.2f}%  x=0 {100*rec['levels'][f'{b:g}']['mean'][1]:.3f}%  half {100*rec['levels'][f'{b:g}']['mean'][2]:.2f}%" for b in LEVELS), flush=True)
    json.dump(R, open(outj, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
