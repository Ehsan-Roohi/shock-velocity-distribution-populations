"""Upstream precursor of the diatomic shock: DSMC (Larsen-Borgnakke) against the linear recoil cascade.

For every case: decay rate of the upstream-moving fraction beta (slope of ln beta against A*N in 1e-4 < beta < 1e-2;
N = column density between the station and the shock centre in DSMC, depth below the source in the cascade), and at
fixed beta: rotational and translational temperature of the upstream-moving molecules, share of the molecules that
still carry the rotational energy of the free stream (tag), their rotational temperature and that of the others.
DSMC also: shares of the collision classes; cascade also: shares of primaries and of former stream molecules.

Usage (in the DSMC folder): python n2_cascade_compare.py <out.json> <label>=<dsmc glob of .txt>=<cascade glob> [...]
Example: python n2_cascade_compare.py runs_n2/cascade_compare.json "M12=runs_n2/tagfix/M12_w0.74_s*.txt=casc_n2/casc_M12_z5_s*.txt" """
import sys, glob, math, json
import numpy as np
from n2_check import read

LEV = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4); LO, HI = 1e-4, 1e-2


def vhs_A(om):
    return 15 * math.sqrt(math.pi) / (2 * (5 - 2 * om) * (7 - 2 * om)) * 4 ** (om - 0.5) / math.gamma(2.5 - om)


def se(v):
    v = np.array(v, float); return float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")


def dsmc_one(runs, sel):
    h = runs[0][0]; M, gam, nc, nb, dx, om = h["M"], h["gamma"], int(h["nc"]), int(h["nb"]), h["dx"], h["omega"]
    n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); T2 = (2 * gam * M * M - (gam - 1)) / (gam + 1) / n2
    mom = sum(runs[k][2] for k in sel); rot = sum(runs[k][4] for k in sel); fly = sum(runs[k][5] for k in sel); ns = sum(runs[k][3] for k in sel)
    cu = sum(runs[k][6] for k in sel)
    x = h["xmin"] + (np.arange(nc) + 0.5) * dx; N = mom[:, 0]; n = N * h["FN"] / (ns * dx)
    ic = int(np.argmin(np.abs((n - 1) / (n2 - 1) - 0.5))); beta = fly[0] / N
    AN = vhs_A(om) * (np.cumsum(n[::-1])[::-1] - np.cumsum(n[::-1])[::-1][ic]) * dx        # A * column density up to the centre
    k = np.where((np.arange(nc) < ic) & (np.arange(nc) > 50) & (beta > 0))[0]
    w = k[(beta[k] > LO) & (beta[k] < HI)]
    out = dict(slope=float(np.polyfit(AN[w], np.log(beta[w]), 1)[0]), length=float(1 / np.polyfit(x[w], np.log(beta[w]), 1)[0]), lev={})
    for b in LEV:
        i = k[np.argmin(np.abs(np.log(beta[k]) - math.log(b)))]; s = slice(i - 3, i + 4)
        cg = rot[1:5, s].sum(); tn = rot[19:21, s].sum(1); te = rot[21:23, s].sum(1); c = cu[:5, s].sum(1)
        out["lev"][b] = dict(x=float(x[i] - x[ic]), trot=float(rot[5:9, s].sum() / cg / T2), ttr=float(rot[15:19, s].sum() / cg * 2 / 3 / T2),
                             tag=float(tn[1] / tn.sum()), trot_tag_T1=float(te[1] / max(tn[1], 1)), trot_untag=float(te[0] / max(tn[0], 1) / T2),
                             cls=(c / c.sum()).tolist())
    return out


def casc_one(files):
    a = sum(np.loadtxt(f) for f in files) / len(files)          # equal number of primaries per seed
    hd = open(files[0]).readline().split(); M = float(hd[hd.index("M") + 1]); om = float(hd[hd.index("omega") + 1])
    x, b = a[:, 0], a[:, 1]; AN = vhs_A(om) * (-x)
    w = (b > LO) & (b < HI)
    out = dict(slope=float(np.polyfit(AN[w], np.log(b[w]), 1)[0]), length=float(1 / np.polyfit(x[w], np.log(b[w]), 1)[0]), lev={})
    for lv in LEV:
        i = int(np.argmin(np.abs(np.log(np.maximum(b, 1e-300)) - math.log(lv)))); s = slice(max(i - 5, 0), i + 6); wt = a[s, 1]
        tg = a[s, 14].sum() / wt.sum(); trot = (a[s, 6] * wt).sum() / wt.sum()
        out["lev"][lv] = dict(x=float(x[i]), trot=float(trot), ttr=float((a[s, 10] * wt).sum() / wt.sum()), tag=float(tg),
                              trot_untag=float((trot - tg * (1.0 / out_T2(M))) / max(1 - tg, 1e-9)),
                              primary=float(a[s, 2].sum() / wt.sum()), scattered=float(a[s, 3].sum() / wt.sum()), recoil=float(a[s, 4].sum() / wt.sum()),
                              trot_primary=float((a[s, 7] * a[s, 2]).sum() / a[s, 2].sum()), trot_scattered=float((a[s, 8] * a[s, 3]).sum() / a[s, 3].sum()),
                              trot_recoil=float((a[s, 9] * a[s, 4]).sum() / a[s, 4].sum()))
    return out


def out_T2(M, gam=1.4):
    n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); return (2 * gam * M * M - (gam - 1)) / (gam + 1) / n2


R = {}
for spec in sys.argv[2:]:
    lab, dpat, cpat = spec.split("=")
    stems = sorted(s[:-4] for s in glob.glob(dpat)); cf = sorted(glob.glob(cpat)); rec = {}
    if stems:
        runs = []
        for s in stems:
            r = read(s); h = r[0]; nc, nb = int(h["nc"]), int(h["nb"])
            cc = np.fromfile(s + ".cls.bin", dtype=np.float32, count=9 * nc * nb).reshape(9, nc, nb)
            v = h["vlo"] + (np.arange(nb) + 0.5) * h["dv"]
            runs.append(tuple(r) + (cc[:, :, v < 0].sum(2, dtype=float),))
        D = dsmc_one(runs, range(len(runs))); per = [dsmc_one(runs, [k]) for k in range(len(runs))]
        D["slope_se"] = se([p["slope"] for p in per]); D["length_se"] = se([p["length"] for p in per])
        for b in LEV:
            for q in ("trot", "ttr", "tag", "trot_untag"):
                D["lev"][b][q + "_se"] = se([p["lev"][b][q] for p in per])
        rec["dsmc"] = D; rec["dsmc_seeds"] = len(runs)
    if cf:
        C = casc_one(cf); per = [casc_one([f]) for f in cf]
        C["slope_se"] = se([p["slope"] for p in per]); C["length_se"] = se([p["length"] for p in per])
        for b in LEV:
            for q in ("trot", "ttr", "tag"):
                C["lev"][b][q + "_se"] = se([p["lev"][b][q] for p in per])
        rec["cascade"] = C; rec["cascade_seeds"] = len(cf)
    R[lab] = rec
    print(f"=== {lab}: DSMC seeds {rec.get('dsmc_seeds', 0)}, cascade seeds {rec.get('cascade_seeds', 0)}")
    if "dsmc" in rec and "cascade" in rec:
        D, C = rec["dsmc"], rec["cascade"]
        print(f"  decay rate d ln(beta)/d(A N): DSMC {D['slope']:.4f}+-{D['slope_se']:.4f}  cascade {C['slope']:.4f}+-{C['slope_se']:.4f}  ratio {C['slope'] / D['slope']:.3f}"
              f"   | decay length in l1: DSMC {D['length']:.3f}  cascade {C['length']:.3f}")
    for b in LEV:
        ln = f"  beta={b:g}:"
        if "dsmc" in rec:
            d = rec["dsmc"]["lev"][b]
            ln += (f" DSMC Trot {d['trot']:.3f}+-{d['trot_se']:.3f} Ttr {d['ttr']:.3f}+-{d['ttr_se']:.3f} tagged {d['tag']:.3f}+-{d['tag_se']:.3f}"
                   f" (Trot tagged/T1 {d['trot_tag_T1']:.2f}, others/T2 {d['trot_untag']:.3f}) classes once/twice/hot/last-beam "
                   + "/".join(f"{d['cls'][j]:.3f}" for j in (1, 2, 3, 4)))
        if "cascade" in rec:
            c = rec["cascade"]["lev"][b]
            ln += (f" || cascade Trot {c['trot']:.3f}+-{c['trot_se']:.3f} Ttr {c['ttr']:.3f}+-{c['ttr_se']:.3f} tagged {c['tag']:.3f}+-{c['tag_se']:.3f}"
                   f" (others/T2 {c['trot_untag']:.3f}) primary/scattered/recoil {c['primary']:.3f}/{c['scattered']:.3f}/{c['recoil']:.3f}"
                   f" Trot prim/scat/recoil {c['trot_primary']:.2f}/{c['trot_scattered']:.2f}/{c['trot_recoil']:.2f}")
        print(ln)
json.dump({k: {a: ({str(x): y for x, y in b.items()} if False else b) for a, b in v.items()} for k, v in R.items()}, open(sys.argv[1], "w"), indent=1, default=str)
