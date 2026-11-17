"""Nitrogen DSMC shocks, seeds combined: upstream-moving fraction beta(x), decay length, shares of the generations
among the upstream-moving molecules, their rotational and translational temperatures, free-flight share.
Usage: python n2_prod.py <folder> <out.json> <Mach> [<Mach> ...]"""
import sys, math, glob, json
import numpy as np
from n2_check import read

folder, outj = sys.argv[1], sys.argv[2]
res = {}
for Ms in sys.argv[3:]:
    stems = sorted(s[:-4] for s in glob.glob(f"{folder}/M{Ms}_w0.74_s*.txt"))
    runs = [read(s) for s in stems]
    h = runs[0][0]; M, gam, nc, dx = h["M"], h["gamma"], int(h["nc"]), h["dx"]
    x = h["xmin"] + (np.arange(nc) + 0.5) * dx
    n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); T2 = (2 * gam * M * M - (gam - 1)) / (gam + 1) / n2

    def quantities(sel):
        mom = sum(runs[k][2] for k in sel); rot = sum(runs[k][4] for k in sel); fly = sum(runs[k][5] for k in sel)
        ns = sum(runs[k][3] for k in sel)
        N = mom[:, 0]; n = N * h["FN"] / (ns * dx)
        nstar = (n - 1) / (n2 - 1); ic = int(np.argmin(np.abs(nstar - 0.5)))
        return N, n, rot, fly, ic

    def at_levels(sel):
        N, n, rot, fly, ic = quantities(sel)
        beta = fly[0] / N; j = np.arange(nc); k = np.where((j < ic) & (j > 50) & (beta > 0))[0]
        out = {}
        for b in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
            i = k[np.argmin(np.abs(np.log(beta[k]) - math.log(b)))]; s = slice(i - 3, i + 4)
            cg = rot[1:5, s].sum(1); eg = rot[5:9, s].sum(1); tg = rot[15:19, s].sum(1)
            out[b] = dict(x=float(x[i] - x[ic]), share=(cg / cg.sum()).tolist(), trot_gen=(eg / np.maximum(cg, 1) / T2).tolist(),
                          trot=float(eg.sum() / cg.sum() / T2), ttr=float(tg.sum() / cg.sum() * 2 / 3 / T2),
                          ttr_gen=(tg / np.maximum(cg, 1) * 2 / 3 / T2).tolist(),
                          fly_share=float(rot[13, s].sum() / cg.sum()), trot_fly=float(rot[14, s].sum() / max(rot[13, s].sum(), 1) / T2),
                          trot_local=float(rot[0, s].sum() / N[s].sum()), nsamp=float(cg.sum()))
        # decay exponent of beta between 1e-2 and 1e-4: beta ~ exp(-(x_c - x)/L)
        i1 = k[np.argmin(np.abs(np.log(beta[k]) - math.log(1e-2)))]; i2 = k[np.argmin(np.abs(np.log(beta[k]) - math.log(1e-4)))]
        p = np.polyfit(x[i2:i1 + 1], np.log(beta[i2:i1 + 1]), 1)
        return out, float(1 / p[0])

    allsel = range(len(runs)); lev, L = at_levels(allsel)
    per = [at_levels([k]) for k in allsel]
    print(f"Mach {Ms}: {len(runs)} seeds, T2={T2:.3f}, 1/T2={1 / T2:.4f}; decay length of beta (1e-2..1e-4): {L:.3f} "
          f"(seeds: {' '.join(f'{q[1]:.3f}' for q in per)})")
    for b, r in lev.items():
        sc = np.std([q[0][b]["trot"] for q in per], ddof=1) / math.sqrt(len(per))
        sct = np.std([q[0][b]["ttr"] for q in per], ddof=1) / math.sqrt(len(per))
        r["trot_se"] = float(sc); r["ttr_se"] = float(sct)
        print(f"  beta={b:g} x-xc={r['x']:.2f} | gen shares " + " ".join(f"{v:.3f}" for v in r["share"])
              + f" | Trot/T2 all {r['trot']:.3f}+-{sc:.3f} by gen " + " ".join(f"{v:.3f}" for v in r["trot_gen"])
              + f" | Ttr/T2 all {r['ttr']:.3f}+-{sct:.3f} by gen " + " ".join(f"{v:.2f}" for v in r["ttr_gen"])
              + f" | free-flight share {r['fly_share']:.3f} Trot {r['trot_fly']:.3f} | local Trot/T1 {r['trot_local']:.3f} | N {r['nsamp']:.3g}")
    res[Ms] = dict(T2=T2, decay_length=L, decay_length_seeds=[q[1] for q in per], levels={str(b): r for b, r in lev.items()})
json.dump(res, open(outj, "w"), indent=1)
