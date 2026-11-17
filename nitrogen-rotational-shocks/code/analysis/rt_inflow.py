"""Measured free-stream rotational distribution of Robben & Talbot (1966), first station, as an inflow file for
dsmc_shock_vdf18 (rin=): one line per level K = 0..kmax with the energy range of the classical level,
theta K^2 to theta (K+1)^2, in units of k T1, and the population (tabulated levels, then the T' line).

Usage: python rt_inflow.py <out.txt> <experiment key 7|12.9> <T1 in K> [kmax=24]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robben_talbot_1966 as rt

out, key, T1 = sys.argv[1], sys.argv[2], float(sys.argv[3]); opt = dict(a.split("=") for a in sys.argv[4:]); KMAX = int(opt.get("kmax", 24))
TH = rt.THETA; K = np.arange(KMAX + 1); G = 2 * K + 1; EK = TH * K * (K + 1)
X, Tp, dk, last = rt.CASES[key]["rows"][0]; kp = len(dk); A = last * np.exp(TH * (kp - 0.5) * (kp + 0.5) / Tp)
N = G * np.array([dk[k] if k < kp else A * np.exp(-EK[k] / Tp) for k in K]); N /= N.sum()
lo, hi = TH * K ** 2 / T1, TH * (K + 1) ** 2 / T1
with open(out, "w") as f:
    for a, b, c in zip(lo, hi, N): f.write(f"{a:.6f} {b:.6f} {c:.8e}\n")
me = float(np.sum(N * 0.5 * (lo + hi))); print(f"Mach {key}: {kp} tabulated levels, T' = {Tp} K above; mean energy {me:.3f} k T1 = {me * T1:.1f} K (Boltzmann at T1: {T1:.1f} K)")
