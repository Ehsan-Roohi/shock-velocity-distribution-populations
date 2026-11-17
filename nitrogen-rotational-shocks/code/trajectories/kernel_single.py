"""Single-collision excitation function of exchange rules next to the classical trajectories: share of the VHS
collisions that leave a cold stream molecule with rotational energy above E, for the ensemble of ctc_kernel.py
(stream molecule against an upstream-moving molecule of the precursor).

Rules: Larsen-Borgnakke with constant probability, with the energy-dependent probability of Boyd (1993), and the
partial-exchange rule of Pullin (1978) with active fraction phi.

Usage: python kernel_single.py <exc npz of ctc_excite.py> [key=value ...]   (keys as in ctc_kernel.py; phi=0.3,0.5,...)"""
import sys, math
import numpy as np

d = np.load(sys.argv[1]); E = d["E"]; sv = float(d["sig_vhs"]); T1 = float(d["T1"]); T2 = float(d["T2"])
P = dict(u1=797.0, Th=442.0, Trh=400.0, pcold=0.55, omega=0.74, zinf=10.0, tstar=91.5); phis = [0.3, 0.4, 0.5, 0.6, 0.7]
for a in sys.argv[2:]:
    k, v = a.split("=")
    if k == "phi": phis = [float(x) for x in v.split(",")]
    else: P[k] = float(v)
om = P["omega"]; eta = 2.5 - om; VU = 91.18; MM = 28.0; MU = 14.0; n = 2000000; rng = np.random.default_rng(3)
v1 = rng.normal(0, math.sqrt(T1 / MM), (n, 3)); v1[:, 0] += P["u1"] / VU
v2 = rng.normal(0, math.sqrt(P["Th"] / MM), (n, 3)); v2[:, 0] = -np.abs(v2[:, 0])
e1 = -T1 * np.log(1 - rng.random(n)); cold = rng.random(n) < P["pcold"]; e2 = np.where(cold, -T1 * np.log(1 - rng.random(n)), -P["Trh"] * np.log(1 - rng.random(n)))
g = np.linalg.norm(v1 - v2, axis=1); et = 0.5 * MU * g ** 2; wv = g ** (2 - 2 * om)


def exc(e):
    k = np.argsort(e); cw = np.cumsum(wv[k]); idx = np.searchsorted(e[k], E * T2, side="right")
    return 1 - np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0) / cw[-1]


def pullin(f):
    s1 = rng.beta(f, 1 - f, n); s2 = rng.beta(f, 1 - f, n); s3 = rng.beta(f * eta, (1 - f) * eta, n)
    pool = s1 * e1 + s2 * e2 + s3 * et; s5 = rng.beta(2 * f, f * eta, n); s4 = rng.beta(f, f, n)
    return (1 - s1) * e1 + s4 * s5 * pool


def lb(p):
    ch = rng.random(n) < p; ec = et + e1
    return np.where(ch, ec * (1 - rng.random(n) ** (1 / eta)), e1)


c1 = math.gamma(3.5 - om) / math.gamma(3 - om) * math.pi ** 1.5 / 2; c2 = math.gamma(3.5 - om) / math.gamma(2.5 - om) * (math.pi ** 2 / 4 + 2)
pb = np.minimum(1.0, (1 + c1 * np.sqrt(P["tstar"] / (et + e1)) + c2 * P["tstar"] / (et + e1)) / P["zinf"])
ii = [int(np.argmin(np.abs(E - x))) for x in (0.1, 0.3, 1.0, 2.0, 3.0)]
sl = lambda f, a, b: (b - a) / math.log(f[int(np.argmin(np.abs(E - a)))] / f[int(np.argmin(np.abs(E - b)))])
rows = [("trajectories", d["ctc"] / sv, float(d["gain_ctc"]) / T2)]
rows += [("Larsen-Borgnakke p = 0.2", exc(lb(0.2)), None), ("energy-dependent (Boyd)", exc(lb(pb)), None)]
rows += [(f"Pullin phi = {f:g}", exc(pullin(f)), None) for f in phis]
print(f"T1 {T1} K, T2 {T2} K; share of the VHS collisions that leave the stream molecule above 0.1, 0.3, 1, 2, 3 T2 | slope temperature/T2 over 0.5-1.5 and 1.5-3")
for name, f, gain in rows:
    print(f"   {name:26s} " + " ".join(f"{f[i]:6.3f}" for i in ii) + f" | {sl(f, 0.5, 1.5):5.2f} {sl(f, 1.5, 3.0):5.2f}")
