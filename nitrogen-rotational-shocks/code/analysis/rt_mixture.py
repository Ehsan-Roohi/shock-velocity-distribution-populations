"""Measured level populations of Robben & Talbot (1966) written as a sum of three shapes fixed in advance:
  S0  the measured free-stream distribution (first station of the table),
  S1  the distribution after one gas-kinetic collision from the classical trajectories (ctc_excite.py),
  S2  a Boltzmann distribution at the downstream temperature.
(a) Non-negative least-squares weights at every station, next to the DSMC shares of molecules with no collision,
    exactly one collision and two or more collisions at the same normalised density.
(b) The populations predicted with the DSMC shares as weights (no fitted number) against the measured ones.

Usage: python rt_mixture.py <exc npz Mach 7> <exc npz Mach 12.9> <class npz> [kmax=24] [suffix=zi10]"""
import sys, os
import numpy as np
from scipy.optimize import nnls
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robben_talbot_1966 as rt

KMAX = 24; SUF = "zi10"
for a in sys.argv[4:]:
    if a.startswith("kmax="): KMAX = int(a[5:])
    if a.startswith("suffix="): SUF = a[7:]
TH = rt.THETA; K = np.arange(KMAX + 1); G = 2 * K + 1; EK = TH * K * (K + 1); C = np.load(sys.argv[3])


def levels(row):
    X, Tp, dk, last = row; kp = len(dk); A = last * np.exp(TH * (kp - 0.5) * (kp + 0.5) / Tp)
    d = np.array([dk[k] if k < kp else A * np.exp(-EK[k] / Tp) for k in K]); N = G * d
    tail = A * Tp / TH * np.exp(-TH * (KMAX + 0.5) * (KMAX + 1.5) / Tp)
    return N / (N.sum() + tail), kp


def kernel(exc, T2):
    """Level populations after one gas-kinetic collision: the trajectories sample a larger cross-section than the VHS
    one; the excess is taken as molecules left in their initial state."""
    d = np.load(exc); E = d["E"] * T2; sv = float(d["sig_vhs"]); sm = float(d["sigmax"])
    f = (d["ctc"] - d["before"] * (1 - sv / sm)) / sv; edge = TH * np.arange(KMAX + 2) ** 2; c = np.interp(edge, E, f); p = np.maximum(c[:-1] - c[1:], 0)
    return p / c[0], float(c[-1] / c[0])                        # second value: share above the last level


for key, exc, lab in (("7", sys.argv[1], "M7" + SUF), ("12.9", sys.argv[2], "M12.9" + SUF)):
    rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); T2 = ex[-1, 1]; en = (ex[:, 2] - rho1) / (rho2 - rho1)
    S0, _ = levels(rows[0]); S1, above = kernel(exc, T2); S2 = G * np.exp(-EK / T2); S2 = S2 / (S2.sum() + T2 / TH * np.exp(-TH * (KMAX + 0.5) * (KMAX + 1.5) / T2))
    n, cls = C[f"{lab}_n"], C[f"{lab}_cls"]; ns = (n - 1) / (float(C[f"{lab}_n2"]) - 1); c = cls[:5] / cls[:5].sum(0)
    m = (ns > 0.002) & (ns < 0.995) & (np.arange(len(ns)) < np.argmax(ns > 0.995) + 1 if (ns > 0.995).any() else True); o = np.argsort(ns[m])
    sh = lambda v, x: float(np.interp(x, ns[m][o], v[m][o]))
    B = np.stack([S0, S1, S2], 1)
    print(f"\nMach {key}: one-collision shape K = 0..9: " + " ".join(f"{v:.3f}" for v in S1[:10]) + f"; mean {(S1 * EK).sum() / T2:.3f} T2 below K = {KMAX} (share above: {above:.3f})")
    print("   n*    | fitted weights: stream, one collision, thermal | residual | DSMC shares: no collision, exactly one, two or more | residual with the DSMC shares | worst level ratio (K <= tabulated)")
    for j, row in enumerate(rows):
        if not (0.02 < en[j] < 0.97): continue
        p, kp = levels(row); w, _ = nnls(B, p); r = np.linalg.norm(p - B @ w) / np.linalg.norm(p)
        d = np.array([sh(c[0], en[j]), sh(c[1], en[j]), sh(c[2] + c[3] + c[4], en[j])]); q = B @ d; rd = np.linalg.norm(p - q) / np.linalg.norm(p)
        kk = slice(0, max(kp, 1)); ratio = q[kk] / p[kk]
        print(f"   {en[j]:.3f} | {w[0]:.3f} {w[1]:.3f} {w[2]:.3f} (sum {w.sum():.3f}) | {r:.3f} | {d[0]:.3f} {d[1]:.3f} {d[2]:.3f} | {rd:.3f} | predicted/measured from {ratio.min():.2f} to {ratio.max():.2f}")
    # which second shape do the measurements prefer? residual of the non-negative fit for several choices, and a grouping of the DSMC classes
    dd = np.load(exc); E = dd["E"] * T2; sv = float(dd["sig_vhs"]); sm = float(dd["sigmax"]); edge = TH * np.arange(KMAX + 2) ** 2
    flb = (dd["lb"] - 0.8 * sv * dd["before"] / sm) / (0.2 * sv); cl = np.interp(edge, E, flb); SLB = np.maximum(cl[:-1] - cl[1:], 0) / cl[0]       # molecules the LB rule excites
    alt = {"two shapes: stream + thermal": [S0, S2], "stream + one collision (trajectories) + thermal": [S0, S1, S2], "stream + Boltzmann at 0.74 T2 + thermal": None, "stream + excited by the LB rule + thermal": [S0, SLB, S2]}
    SB = G * np.exp(-EK / (0.74 * T2)); SB = SB / (SB.sum() + 0.74 * T2 / TH * np.exp(-TH * (KMAX + 0.5) * (KMAX + 1.5) / (0.74 * T2))); alt["stream + Boltzmann at 0.74 T2 + thermal"] = [S0, SB, S2]
    sel = [j for j in range(len(rows)) if 0.02 < en[j] < 0.97 and len(rows[j][2]) > 0]
    print("   residual of the non-negative fit at the stations with tabulated levels (n* = " + " ".join(f"{en[j]:.2f}" for j in sel) + "):")
    for name, shapes in alt.items():
        BB = np.stack(shapes, 1); rr = []
        for j in sel:
            p, _ = levels(rows[j]); w, _ = nnls(BB, p); rr.append(np.linalg.norm(p - BB @ w) / np.linalg.norm(p))
        print(f"      {name:50s} " + " ".join(f"{v:.3f}" for v in rr) + f"   mean {np.mean(rr):.3f}")
    print("   fitted weight of the one-collision shape against DSMC shares (class 1 = stream molecule after one collision; 2 = two collisions; 4 = scattered, last partner a stream molecule; 3 = three or more):")
    for j in sel:
        p, _ = levels(rows[j]); w, _ = nnls(B, p)
        print(f"      n* = {en[j]:.3f}: fitted stream / one collision / thermal {w[0]:.3f} {w[1]:.3f} {w[2]:.3f} | DSMC class 0: {sh(c[0], en[j]):.3f}; classes 1+2+4: {sh(c[1] + c[2] + c[4], en[j]):.3f}; class 3: {sh(c[3], en[j]):.3f}"
              f" | generations 0, 1, 2, 3+: " + " ".join(f"{sh(cls[5 + g] / cls[5:9].sum(0), en[j]):.3f}" for g in range(4)))
