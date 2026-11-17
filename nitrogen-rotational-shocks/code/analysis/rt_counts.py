"""Measured level populations of Robben & Talbot (1966) against the populations of the DSMC molecules sorted by the
exact number of collisions (dsmc_shock_vdf17, trajectory kernel).

At every measured station:
  * DSMC shares of the molecules with 0, 1, 2 and 3 or more collisions, and the temperature of each group;
  * the measured populations fitted with non-negative weights of three shapes fixed in advance: the measured free
    stream (S0), the distribution left by the first collision of stream molecules in the simulation (S1, record
    <out>.first.bin), and Boltzmann at T2 (S2); weights set beside the DSMC shares of 0, 1-2, and 3+ collisions;
  * the same with S1 replaced by the single-collision distribution of the trajectory ensemble with molecules moving
    upstream (exc npz), for comparison;
  * which group supplies the populations of the measured hot line (levels above K', the last tabulated level).

Usage: python rt_counts.py <erh_v17 npz> <first_v17 npz> <label> <experiment key 7|12.9> <exc npz> [kmax=24] [nmin=0.02]"""
import sys, os, math, json
import numpy as np
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robben_talbot_1966 as rt

erhf, firstf, lab, key, excf = sys.argv[1:6]; opt = dict(a.split("=") for a in sys.argv[6:]); KMAX = int(opt.get("kmax", 24)); NMIN = float(opt.get("nmin", 0.02))
TH = rt.THETA; K = np.arange(KMAX + 1); G = 2 * K + 1; EK = TH * K * (K + 1)
D = np.load(erhf); F = np.load(firstf); n, erh, erl = (D[f"{lab}_{k}"] for k in ("n", "erh", "erl")); ns = (n - 1) / (float(D[f"{lab}_n2"]) - 1)
rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); T2 = ex[-1, 1]; en = (ex[:, 2] - rho1) / (rho2 - rho1)


def levels(row):
    X, Tp, dk, last = row; kp = len(dk); A = last * np.exp(TH * (kp - 0.5) * (kp + 0.5) / Tp)
    d = np.array([dk[k] if k < kp else A * np.exp(-EK[k] / Tp) for k in K]); N = G * d
    return N / (N.sum() + A * Tp / TH * np.exp(-TH * (KMAX + 0.5) * (KMAX + 1.5) / Tp)), kp, Tp


def pops(fine, coarse, dfine=0.005, dcoarse=0.05):
    """level populations from histograms of e/T2 (fine bins up to 0.8, coarse beyond)"""
    nf = int(round(0.8 / dfine)); ef = np.arange(nf + 1) * dfine; ec = np.arange(len(coarse) + 1) * dcoarse
    cf = np.concatenate([[0], np.cumsum(fine[:nf])]); cc = np.concatenate([[0], np.cumsum(coarse)]); edge = TH * np.arange(KMAX + 2) ** 2 / T2
    C = np.where(edge <= 0.8, np.interp(edge, ef, cf), cf[-1] + np.interp(edge, ec, cc) - np.interp(0.8, ec, cc))
    return C[1:] - C[:-1]


S0, _, _ = levels(rows[0]); fl = "M7k" if key == "7" else "M12.9k"; fr = F[f"{fl}_first"]
S1 = pops(fr[1], fr[0]); S1 = S1 / S1.sum()
d = np.load(excf); E = d["E"] * T2; sv = float(d["sig_vhs"]); sm = float(d["sigmax"]); edge = TH * np.arange(KMAX + 2) ** 2
f1 = (d["ctc"] - d["before"] * (1 - sv / sm)) / sv; c = np.interp(edge, E, f1); S1u = np.maximum(c[:-1] - c[1:], 0) / c[0]
S2 = G * np.exp(-EK / T2); S2 = S2 / (S2.sum() + T2 / TH * np.exp(-TH * (KMAX + 0.5) * (KMAX + 1.5) / T2))
B = np.stack([S0, S1, S2], 1); Bu = np.stack([S0, S1u, S2], 1); B2 = B[:, [0, 2]]
res = lambda M, p: (nnls(M, p)[0], np.linalg.norm(p - M @ nnls(M, p)[0]) / np.linalg.norm(p))
out = []
print(f"{lab} against Mach {key} (T2 = {T2:.0f} K). First-collision shape: mean e {np.sum(S1 * EK) / T2:.3f} kT2")
print("station | DSMC shares 0, 1, 2, 3+ coll. | weights with first-collision shape (residual) | with upstream-ensemble shape (residual) | two shapes | share of the measured hot line (E above level K') from 0, 1, 2, 3+ coll. in DSMC | measured T'/T2, DSMC slope there")
for j, row in enumerate(rows):
    p, kp, Tp = levels(row)
    if not (NMIN < en[j] < 0.97) or kp == 0: continue
    i = int(np.argmin(np.abs(ns - en[j]))); s = slice(max(i - 2, 0), i + 3); tot = erh[0, s].sum()
    sh = [erh[r, s].sum() / tot for r in (2, 9, 10, 11)]
    w, r3 = res(B, p); wu, r3u = res(Bu, p); _, r2 = res(B2, p)
    ek = TH * kp * kp / T2; ie = int(ek / 0.05)                                   # energy of the lowest level of the T' line, in T2
    hot = [erh[r, s].sum(0)[ie:].sum() for r in (2, 9, 10, 11)]; ht = sum(hot)
    h = erh[0, s].sum(0); cum = h[::-1].cumsum()[::-1]; e1, e2 = ie, min(ie + 30, 159); sl = (e2 - e1) * 0.05 / math.log(cum[e1] / cum[e2])
    print(f"n* = {en[j]:.2f}: {sh[0]:.2f} {sh[1]:.2f} {sh[2]:.2f} {sh[3]:.2f} | {w[0]:.2f} {w[1]:.2f} {w[2]:.2f} ({100 * r3:.1f}%) | {wu[0]:.2f} {wu[1]:.2f} {wu[2]:.2f} ({100 * r3u:.1f}%) | {100 * r2:.1f}% | "
          + " ".join(f"{v / ht:.2f}" for v in hot) + f" | {Tp / T2:.2f}, {sl:.2f} (from {ek:.2f} kT2)")
    out.append(dict(nstar=float(en[j]), shares=sh, w=w.tolist(), res3=r3, wu=wu.tolist(), res3u=r3u, res2=r2, hot=[v / ht for v in hot], Tp=Tp / T2, slope=sl, ek=ek))
print(f"mean residual: first-collision shape {100 * np.mean([o['res3'] for o in out]):.1f}%, upstream-ensemble shape {100 * np.mean([o['res3u'] for o in out]):.1f}%, two shapes {100 * np.mean([o['res2'] for o in out]):.1f}%")
w1 = np.array([o["w"][1] for o in out]); s12 = np.array([o["shares"][1] + o["shares"][2] for o in out]); w0 = np.array([o["w"][0] for o in out]); s0 = np.array([o["shares"][0] for o in out])
w2 = np.array([o["w"][2] for o in out]); s3 = np.array([o["shares"][3] for o in out])
print(f"weights minus DSMC shares, rms over the stations: free stream {np.sqrt(np.mean((w0 - s0) ** 2)):.3f}, first-collision shape vs 1-2 collisions {np.sqrt(np.mean((w1 - s12) ** 2)):.3f}, thermal vs 3+ {np.sqrt(np.mean((w2 - s3) ** 2)):.3f}")
json.dump(dict(rows=out, S1=S1.tolist(), S1u=S1u.tolist()), open(f"rt_counts_{lab}.json", "w"), indent=1)
