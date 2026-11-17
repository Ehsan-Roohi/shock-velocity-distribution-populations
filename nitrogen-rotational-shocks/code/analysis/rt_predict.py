"""Level populations predicted by DSMC, with no fitted number, against the measured ones of Robben & Talbot (1966).

At each measured station (placed by the normalised density) the DSMC distribution of rotational energy is turned into
populations of the levels K (classical level K = energies between theta K^2 and theta (K+1)^2). Two predictions:
  raw        all DSMC molecules (the simulated free stream is in equilibrium at T1);
  stream     the unscattered molecules (generation 0) are given the measured free-stream distribution instead, because
             the free stream of the experiment is frozen out of equilibrium.
Also the non-negative three-shape weights (measured free stream, one collision, thermal) of the DSMC populations, to
set beside those of the measurement.

Usage: python rt_predict.py <out.json> <erh npz> <label>:<experiment key>:<exc npz of the one-collision shape> [...] [kmax=24]"""
import sys, os, json
import numpy as np
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robben_talbot_1966 as rt

KMAX = 24
for a in sys.argv[3:]:
    if a.startswith("kmax="): KMAX = int(a[5:])
TH = rt.THETA; K = np.arange(KMAX + 1); G = 2 * K + 1; EK = TH * K * (K + 1); D = np.load(sys.argv[2]); out = {}


def levels(row):
    X, Tp, dk, last = row; kp = len(dk); A = last * np.exp(TH * (kp - 0.5) * (kp + 0.5) / Tp)
    d = np.array([dk[k] if k < kp else A * np.exp(-EK[k] / Tp) for k in K]); N = G * d
    return N / (N.sum() + A * Tp / TH * np.exp(-TH * (KMAX + 0.5) * (KMAX + 1.5) / Tp)), kp


def pops(fine, coarse, T2):
    """Level populations (not normalised) from the histograms of e/T2: fine bins of 0.005 up to 0.8, coarse of 0.05 up to 8."""
    ef = np.arange(161) * 0.005; ec = np.arange(161) * 0.05; cf = np.concatenate([[0], np.cumsum(fine)]); cc = np.concatenate([[0], np.cumsum(coarse)])
    edge = TH * np.arange(KMAX + 2) ** 2 / T2
    C = np.where(edge <= 0.8, np.interp(edge, ef, cf), cf[-1] + np.interp(edge, ec, cc) - np.interp(0.8, ec, cc))
    return C[1:] - C[:-1]


for spec in [a for a in sys.argv[3:] if not a.startswith("kmax=")]:
    lab, key, exc = spec.split(":", 2)
    n, erh, erl = (D[f"{lab}_{k}"] for k in ("n", "erh", "erl")); ns = (n - 1) / (float(D[f"{lab}_n2"]) - 1)
    rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); T2 = ex[-1, 1]; en = (ex[:, 2] - rho1) / (rho2 - rho1)
    S0, _ = levels(rows[0]); d = np.load(exc); E = d["E"] * T2; sv = float(d["sig_vhs"]); sm = float(d["sigmax"]); edge = TH * np.arange(KMAX + 2) ** 2
    f1 = (d["ctc"] - d["before"] * (1 - sv / sm)) / sv; c = np.interp(edge, E, f1); S1 = np.maximum(c[:-1] - c[1:], 0) / c[0]
    S2 = G * np.exp(-EK / T2); S2 = S2 / (S2.sum() + T2 / TH * np.exp(-TH * (KMAX + 0.5) * (KMAX + 1.5) / T2)); B = np.stack([S0, S1, S2], 1)
    print(f"\n{lab} against Mach {key}: station, residual of the raw DSMC populations, residual with the measured free stream for the unscattered molecules, range of predicted/measured at the tabulated levels,")
    print("   three-shape weights of the measurement | of the DSMC populations (stream, one collision, thermal); share of unscattered molecules in DSMC")
    rec = []
    for j, row in enumerate(rows):
        if not (0.02 < en[j] < 0.97): continue
        p, kp = levels(row); i = int(np.argmin(np.abs(ns - en[j]))); s = slice(max(i - 2, 0), i + 3); tot = erh[0, s].sum()
        pa = pops(erl[0, s].sum(0), erh[0, s].sum(0), 1.0 * T2 / T2 * T2) / tot if False else pops(erl[0, s].sum(0), erh[0, s].sum(0), T2) / tot
        p0 = pops(erl[2, s].sum(0), erh[2, s].sum(0), T2) / tot; w0 = erh[2, s].sum() / tot; ps = pa - p0 + w0 * S0
        r_raw = np.linalg.norm(p - pa) / np.linalg.norm(p); r_s = np.linalg.norm(p - ps) / np.linalg.norm(p); kk = slice(0, max(kp, 1)); ratio = ps[kk] / p[kk]
        wm, _ = nnls(B, p); wd, _ = nnls(B, ps)
        print(f"   n* = {en[j]:.3f} (cell {ns[i]:.3f}): {r_raw:.3f}  {r_s:.3f}  {ratio.min():.2f}-{ratio.max():.2f} | {wm[0]:.3f} {wm[1]:.3f} {wm[2]:.3f} | {wd[0]:.3f} {wd[1]:.3f} {wd[2]:.3f}; unscattered {w0:.3f}")
        rec.append(dict(nstar=float(en[j]), res_raw=float(r_raw), res_stream=float(r_s), w_meas=wm.tolist(), w_dsmc=wd.tolist(), unscattered=float(w0), pred=ps.tolist(), pred_raw=pa.tolist(), meas=p.tolist(), kp=int(kp)))
    tab = [r for r in rec if r["kp"] > 0]
    print(f"   mean residual at the stations with tabulated levels: raw {np.mean([r['res_raw'] for r in tab]):.3f}, with the measured free stream {np.mean([r['res_stream'] for r in tab]):.3f}")
    out[lab] = rec
json.dump(out, open(sys.argv[1], "w"), indent=1)
