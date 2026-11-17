"""Velocity grid for the Rykov shock solver at a given Mach number (gamma = 7/5, velocities in sqrt(R T1)).

Streamwise nodes: uniform (midpoint rule), spacing dv, from u2 - c sqrt(T2) to max(u1 + 7, u2 + c sqrt(T2)).
Radial nodes: four-point Gauss-Legendre rule on panels that are narrow near the axis (upstream beam, width 1) and
widen towards c sqrt(T2). The weight of a node is dv * 2 pi r * (radial weight), so that sum w F is the density.

Usage: python make_grid.py <Mach> <out file> [dv=0.25] [c=7.5]      -> prints nv and nr
"""
import sys, math
import numpy as np

M = float(sys.argv[1]); out = sys.argv[2]
dv = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
c = float(sys.argv[4]) if len(sys.argv) > 4 else 7.5
g = 1.4; u1 = M * math.sqrt(g); r2 = (g + 1) * M * M / ((g - 1) * M * M + 2); T2 = (2 * g * M * M - (g - 1)) / (g + 1) / r2; u2 = u1 / r2
s2 = math.sqrt(T2)
vlo, vhi = u2 - c * s2, max(u1 + 7.0, u2 + c * s2)
nv = int(math.ceil((vhi - vlo) / dv)); v = vlo + (np.arange(nv) + 0.5) * dv
rmax = c * s2
edges = [0.0]; h = 0.75
while edges[-1] < rmax:
    edges.append(edges[-1] + h)
    if edges[-1] >= 3.0:
        h *= 1.35
edges[-1] = max(edges[-1], rmax)
xg, wg = np.polynomial.legendre.leggauss(4)
r = np.concatenate([0.5 * (a + b) + 0.5 * (b - a) * xg for a, b in zip(edges[:-1], edges[1:])])
wr = np.concatenate([0.5 * (b - a) * wg for a, b in zip(edges[:-1], edges[1:])])
nr = len(r)
with open(out, "w") as f:
    f.write(f"{nv * nr}\n")
    for a in v:
        for b, wb in zip(r, wr):
            f.write(f"{a:.18e} {b:.18e} {dv * 2 * math.pi * b * wb:.18e}\n")
V, R = np.meshgrid(v, r, indexing="ij"); W = dv * 2 * math.pi * R * wr[None, :]
mx = lambda n, u, T: n / (2 * math.pi * T) ** 1.5 * np.exp(-((V - u) ** 2 + R ** 2) / (2 * T))
for name, n, u, T in (("upstream", 1.0, u1, 1.0), ("downstream", r2, u2, T2)):
    F = mx(n, u, T); rho = (W * F).sum(); uu = (W * V * F).sum() / rho; tt = (W * ((V - uu) ** 2 + R ** 2) * F).sum() / (3 * rho)
    print(f"  {name}: density error {rho / n - 1:+.2e}, velocity error {uu - u:+.2e}, temperature error {tt / T - 1:+.2e}")
print(f"M={M:g} nv={nv} nr={nr} K={nv * nr} v in [{vlo:.2f},{vhi:.2f}] rmax={edges[-1]:.2f} u1={u1:.4f} u2={u2:.4f} T2={T2:.4f}")
