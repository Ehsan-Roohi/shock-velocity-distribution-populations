"""Profiles of the upstream precursor for the letter figures: for every DSMC set the upstream-moving fraction beta
against position and against A times the column density to the shock centre, the rotational and translational energy
of the upstream-moving molecules, the share that still carries free-stream rotation and the shares and rotational
energy of the collision classes; for every cascade set beta against A times depth.

Usage (in the DSMC folder): python n2_letter_precursor.py <out.npz> d:<label>=<glob of .txt> ... c:<label>=<glob of cascade .txt> ..."""
import sys, glob, math
import numpy as np
from n2_check import read

LO, HI = 1e-4, 1e-2


def vhs_A(om):
    return 15 * math.sqrt(math.pi) / (2 * (5 - 2 * om) * (7 - 2 * om)) * 4 ** (om - 0.5) / math.gamma(2.5 - om)


out = {}
for spec in sys.argv[2:]:
    kind, rest = spec.split(":", 1); lab, pat = rest.split("=", 1)
    if kind == "d":
        stems = sorted(s[:-4] for s in glob.glob(pat)); stems = [s for s in stems if "steps_done 215000" in open(s + ".txt").read()]
        runs = [read(s) for s in stems]; h = runs[0][0]; M, gam, nc, dx, om = h["M"], h["gamma"], int(h["nc"]), h["dx"], h["omega"]; NR = int(h["NR"])
        n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); T2 = (2 * gam * M * M - (gam - 1)) / (gam + 1) / n2
        n2 = h.get("n2res", n2)

        def prof(sel):
            mom = sum(runs[k][2] for k in sel); rot = sum(runs[k][4] for k in sel); fly = sum(runs[k][5] for k in sel); ns = sum(runs[k][3] for k in sel)
            x = h["xmin"] + (np.arange(nc) + 0.5) * dx; N = mom[:, 0]; n = N * h["FN"] / (ns * dx)
            ic = int(np.argmin(np.abs((n - 1) / (n2 - 1) - 0.5))); beta = fly[0] / N
            cs = np.cumsum(n[::-1])[::-1]; AN = vhs_A(om) * (cs - cs[ic]) * dx
            k = np.where((np.arange(nc) < ic) & (np.arange(nc) > 50) & (beta > 0))[0]; w = k[(beta[k] > LO) & (beta[k] < HI)]
            return x - x[ic], n, beta, AN, rot, float(np.polyfit(AN[w], np.log(beta[w]), 1)[0]), float(1 / np.polyfit(x[w], np.log(beta[w]), 1)[0])

        x, n, beta, AN, rot, slope, length = prof(range(len(runs))); per = np.array([prof([k])[5:] for k in range(len(runs))])
        se = per.std(0, ddof=1) / math.sqrt(len(runs)) if len(runs) > 1 else np.zeros(2)
        out.update({f"{lab}_x": x, f"{lab}_n": n, f"{lab}_beta": beta, f"{lab}_AN": AN, f"{lab}_slope": slope, f"{lab}_length": length, f"{lab}_slope_se": se[0],
                    f"{lab}_length_se": se[1], f"{lab}_T2": T2, f"{lab}_n2": n2, f"{lab}_cg": rot[1:5].sum(0), f"{lab}_eg": rot[5:9].sum(0), f"{lab}_tg": rot[15:19].sum(0)})
        if NR >= 25: out.update({f"{lab}_tn": rot[19:21], f"{lab}_te": rot[21:23]})
        if NR >= 35: out.update({f"{lab}_cn": rot[25:30], f"{lab}_ce": rot[30:35]})
        print(f"{lab}: Mach {M:g}, {len(runs)} seeds, decay rate d ln(beta)/d(A N) {slope:.4f} +- {se[0]:.4f}, decay length {length:.3f} +- {se[1]:.3f} l1")
    else:
        files = sorted(glob.glob(pat)); a = sum(np.loadtxt(f) for f in files) / len(files)
        hd = open(files[0]).readline().split(); om = float(hd[hd.index("omega") + 1]); xx, b = a[:, 0], a[:, 1]; AN = vhs_A(om) * (-xx); w = (b > LO) & (b < HI)
        per = [float(np.polyfit(AN[w], np.log(np.loadtxt(f)[:, 1][w]), 1)[0]) for f in files]
        slope = float(np.polyfit(AN[w], np.log(b[w]), 1)[0]); out.update({f"{lab}_AN": AN, f"{lab}_beta": b, f"{lab}_x": xx, f"{lab}_slope": slope, f"{lab}_slope_se": float(np.std(per, ddof=1) / math.sqrt(len(per)))})
        print(f"{lab}: cascade, {len(files)} seeds, decay rate {slope:.4f} +- {out[lab + '_slope_se']:.4f}, decay length {1 / np.polyfit(xx[w], np.log(b[w]), 1)[0]:.3f} l1")
np.savez(sys.argv[1], **out)
