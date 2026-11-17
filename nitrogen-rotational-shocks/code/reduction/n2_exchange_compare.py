"""Nitrogen DSMC shocks computed with different rules for the exchange of rotational energy, side by side:
thicknesses and distance between the density and rotational-temperature profiles in the units of Robben & Talbot,
and the state of the upstream-moving molecules in the precursor (rotational and translational energy, share that
still carries the rotational energy of the free stream, shares and rotational energy by collision class).

Usage (in the DSMC folder): python n2_exchange_compare.py <out.json> <label>=<glob of .txt files> [...]
Only seeds that have reached the last step (steps_done = nsteps of the first file, or min=<steps>) are used."""
import sys, glob, math, json
import numpy as np
from n2_check import read

EXP = {7.0: dict(n=0.158, tr=0.115, lag=1.9), 12.9: dict(n=0.167, tr=0.107, lag=2.54)}
args = [a for a in sys.argv[2:] if not a.startswith("min=")]; smin = [int(a[4:]) for a in sys.argv[2:] if a.startswith("min=")]; smin = smin[0] if smin else 215000
res = {}
for spec in args:
    lab, pat = spec.split("=", 1); stems = sorted(s[:-4] for s in glob.glob(pat)); runs = []
    for s in stems:
        try:
            r = read(s)
        except Exception as e:
            print("skipped", s, e); continue
        if r[0]["steps_done"] >= smin: runs.append(r)
    if not runs:
        print(lab, "no complete seed"); continue
    h = runs[0][0]; M, gam, nc, dx, om = h["M"], h["gamma"], int(h["nc"]), h["dx"], h["omega"]; NR = int(h["NR"])
    n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); T2 = (2 * gam * M * M - (gam - 1)) / (gam + 1) / n2
    n2 = h.get("n2res", n2); TR1, TR2 = (h["rin_me"], h["T2res"]) if h.get("freeze", 0) > 0 else (1.0, T2)   # version 19: ends of the frozen stream
    Lstar = ((1 + 0.5 * (gam - 1) * M * M) * 2 / (gam + 1)) ** om / (M * math.sqrt(gam)); x = h["xmin"] + (np.arange(nc) + 0.5) * dx

    def prof(sel):
        mom = sum(runs[k][2] for k in sel); rot = sum(runs[k][4] for k in sel); ns = sum(runs[k][3] for k in sel)
        N = mom[:, 0]; n = (N * h["FN"] / (ns * dx) - 1) / (n2 - 1); tr = (rot[0] / N - TR1) / (TR2 - TR1)
        k = np.ones(5) / 5; ns_, trs = np.convolve(n, k, "same"), np.convolve(tr, k, "same")
        m = (x > -25) & (x < 15)
        xm = lambda f: float(np.interp(0.5, f[m], x[m]))
        return 1 / np.max(np.gradient(ns_, x)[m]), 1 / np.max(np.gradient(trs, x)[m]), xm(n) - xm(tr)

    a = np.array([prof([k]) for k in range(len(runs))]); c = prof(range(len(runs)))
    se = a.std(0, ddof=1) / math.sqrt(len(runs)) if len(runs) > 1 else np.zeros(3)
    e = EXP.get(round(M, 1)); rule = "Pullin" if "/pullin" in pat else ("energy-dependent probability, Zinf = " + pat.split("_zi")[1].split("_")[0] if "_zi" in pat else ("deflection-dependent probability" if "/defl" in pat else "Larsen-Borgnakke, Zrot = %g" % h["zrot"]))
    print(f"{lab}: Mach {M:g}, {rule}, {len(runs)} seeds, L* = {Lstar:.4f} l1")
    print(f"   L*/delta density {Lstar / c[0]:.4f} (+-{Lstar * se[0] / c[0] ** 2:.4f})" + (f" [measured {e['n']}]" if e else "") +
          f"; L*/delta rotational temperature {Lstar / c[1]:.4f} (+-{Lstar * se[1] / c[1] ** 2:.4f})" + (f" [measured {e['tr']}]" if e else "") +
          f"; distance between midpoints {c[2] / Lstar:.2f} (+-{se[2] / Lstar:.2f}) L*" + (f" [measured {e['lag']}]" if e else ""))
    out = dict(M=M, rule=rule, seeds=len(runs), Lstar=Lstar, dens=Lstar / c[0], trot=Lstar / c[1], lag=c[2] / Lstar, se=[float(v) for v in se], T2=T2)
    mom = sum(r[2] for r in runs); rot = sum(r[4] for r in runs); fly = sum(r[5] for r in runs); ns = sum(r[3] for r in runs)
    N = mom[:, 0]; n = N * h["FN"] / (ns * dx); ic = int(np.argmin(np.abs((n - 1) / (n2 - 1) - 0.5))); beta = fly[0] / N; j = np.arange(nc)
    k = np.where((j < ic) & (j > 50) & (beta > 0))[0]
    for b in (1e-2, 1e-3):
        i = k[np.argmin(np.abs(np.log(beta[k]) - math.log(b)))]; s = slice(i - 2, i + 3)
        cg = rot[1:5, s].sum(1); eg = rot[5:9, s].sum(1); tg = rot[15:19, s].sum(1)
        line = f"   beta = {b:g} (x - xc = {x[i] - x[ic]:.2f}): upstream-moving T_rot/T2 {eg.sum() / cg.sum() / T2:.3f}, (2/3)E_tr/T2 {tg.sum() / cg.sum() * 2 / 3 / T2:.3f}"
        d = dict(x=float(x[i] - x[ic]), trot=float(eg.sum() / cg.sum() / T2), ttr=float(tg.sum() / cg.sum() * 2 / 3 / T2))
        if NR >= 25:
            tn = rot[19:21, s].sum(1); te = rot[21:23, s].sum(1)
            line += f"; share with free-stream rotation {tn[1] / tn.sum():.3f} (T_rot/T2 {te[1] / tn[1] / T2:.3f}; others {te[0] / tn[0] / T2:.3f})"
            d.update(tag=float(tn[1] / tn.sum()), trot_tag=float(te[1] / tn[1] / T2), trot_other=float(te[0] / tn[0] / T2))
        print(line)
        if NR >= 35:
            cn = rot[25:30, s].sum(1); ce = rot[30:35, s].sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                tc = ce / cn / T2
            print("      classes (stream / once-collided stream / twice / three or more / scattered by stream): shares " + " ".join(f"{v:.3f}" for v in cn / cn.sum()) + "; T_rot/T2 " + " ".join(f"{v:.3f}" for v in tc))
            d.update(class_share=[float(v) for v in cn / cn.sum()], class_trot=[float(v) for v in np.nan_to_num(tc)])
        out[f"beta{b:g}"] = d
    # decay rate of beta between 1e-2 and 1e-4
    kk = k[(beta[k] < 1e-2) & (beta[k] > 1e-4)]
    if len(kk) > 5:
        p = np.polyfit(x[kk], np.log(beta[kk]), 1); out["decay_length"] = float(1 / p[0]); print(f"   decay length of beta between 1e-2 and 1e-4: {1 / p[0]:.3f} l1")
    res[lab] = out
json.dump(res, open(sys.argv[1], "w"), indent=1)
