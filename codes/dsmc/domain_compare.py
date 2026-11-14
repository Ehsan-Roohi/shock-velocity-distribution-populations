"""Effect of the length of the upstream domain on the precursor: pooled beta(N) of the standard runs against runs
with a longer domain (same Mach number and omega).

Usage: python domain_compare.py <standard runs dir> <long runs dir> <out json>
"""
import sys, os, glob, re, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import precursor_seeds as PS


def pooled(runs, case, min_steps=215000):
    tot = None
    for stem in sorted(p[:-4] for p in glob.glob(os.path.join(runs, case + "_s*.txt"))):
        h, cnt, ns = PS.read(stem)
        if int(h["steps_done"]) < min_steps:
            continue
        tot = [cnt, ns, h, 1] if tot is None else [tot[0] + cnt, tot[1] + ns, h, tot[3] + 1]
    return tot


def main(std, lng, outj):
    R = {}
    for case in sorted({re.sub(r"_s\d+\.txt$", "", os.path.basename(p)) for p in glob.glob(os.path.join(lng, "M*_w*_s*.txt"))}):
        a, b = pooled(std, case), pooled(lng, case)
        if a is None or b is None:
            continue
        om = float(a[2]["omega"])
        xa, ba, Na, ia = PS.profile(a[2], a[0], a[1]); xb, bb, Nb, ib = PS.profile(b[2], b[0], b[1])
        rec = dict(seeds_standard=a[3], seeds_long=b[3], xmin_standard=float(a[2]["xmin"]), xmin_long=float(b[2]["xmin"]),
                   fits_standard=PS.fits(xa, ba, Na, ia, om), fits_long=PS.fits(xb, bb, Nb, ib, om))
        for lo in (2e-5, 2e-4):
            ka = (np.arange(len(xa)) < ia) & (ba < 1e-2) & (ba > lo) & (xa > xa[0] + 5)
            lb_long = np.interp(Na[ka][::-1], Nb[:ib][::-1], np.log(np.maximum(bb[:ib], 1e-300))[::-1])[::-1]
            d = lb_long - np.log(ba[ka])
            rec[f"ln_ratio_long_over_standard_beta>{lo:g}"] = dict(mean=float(d.mean()), rms=float(np.sqrt(np.mean(d ** 2))), max_abs=float(np.abs(d).max()),
                                                                 at_low_end=float(d[np.argmin(ba[ka])]))
        # upstream state seen by the two domains: beta just inside the inflow margin
        rec["beta_5_inside_inflow"] = dict(standard=float(ba[50]), long=float(bb[50]))
        R[case] = rec
        print(case, json.dumps({k: v for k, v in rec.items() if not k.startswith("fits")}), flush=True)
        for nm in ("fits_standard", "fits_long"):
            print("   ", nm, {w: (round(v["p"], 3), round(v["rms"], 3), round(v["rms_p23"], 3), round(v["rms_heur"], 3)) for w, v in rec[nm].items()})
    json.dump(R, open(outj, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
