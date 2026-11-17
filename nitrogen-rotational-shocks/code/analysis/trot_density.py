"""Mean rotational temperature against density through the shock, both normalised between their upstream and
downstream values: measurements at seven Mach numbers (Robben & Talbot 1966, electron beam, Mach 7.0 and 12.9; Ramos et
al. 2000, Raman, Mach 7.7 to 15.3, shock waves A-E of a free jet) next to DSMC with different exchange rules.

For the free-jet shocks the upstream state is the point of minimum density and temperature (point 1 of the authors)
and the downstream state is their point 2 (end of the shock, dashed line of their Fig. 3).

Usage: python trot_density.py <out.json> <label>=<erh npz>:<key in the npz> [...]"""
import sys, json
import numpy as np
import robben_talbot_1966 as rt
import ramos_2000 as R

NS = np.array([0.1, 0.2, 0.3, 0.5, 0.7]); out = {}


def show(name, ns, ts):
    o = np.argsort(ns); ns, ts = np.asarray(ns)[o], np.asarray(ts)[o]; ok = (ns > 0.01) & (ns < 0.99)
    v = np.interp(NS, ns[ok], ts[ok]); out[name] = dict(nstar=ns[ok].tolist(), tstar=ts[ok].tolist(), at=v.tolist())
    print(f"   {name:44s} " + " ".join(f"{x:6.3f}" for x in v) + f"   | n* at T* = 0.5: {np.interp(0.5, ts[ok], ns[ok]):.3f}")


print("normalised rotational temperature T* at normalised density n* = " + " ".join(f"{x:g}" for x in NS))
print("measurements")
for key in ("7", "12.9"):
    ex = rt.reduce(rt.CASES[key]["rows"]); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); T1 = ex[0, 3]; T2 = ex[-3:, 3].mean()
    show(f"Robben & Talbot, Mach {key}", (ex[:, 2] - rho1) / (rho2 - rho1), (ex[:, 3] - T1) / (T2 - T1))
for s in R.SHOCKS:
    z, n, zt, T = R.fig3(s); z1, z2 = R.FIG3_DASHED_Z_OVER_D[s]; m = (z >= z1 - 0.3) & (z <= z2 + 0.05); mt = (zt >= z1 - 0.3) & (zt <= z2 + 0.05)
    n1 = n[m].min(); T1 = R.FIG3_TROT_MIN_K[s][0]; n2 = float(np.interp(z2, z, n)); T2 = float(np.interp(z2, zt, T)); zz = z[m]; o = np.argsort(zt[mt])
    show(f"Ramos et al., shock {s}, Mach {R.FIG6_TABLE[s]['M1']} (n2/n1 {n2 / n1:.2f}, T2 {T2:.0f} K)", (n[m] - n1) / (n2 - n1), (np.interp(zz, zt[mt][o], T[mt][o]) - T1) / (T2 - T1))
print("DSMC")
for spec in sys.argv[2:]:
    name, rest = spec.split("=", 1); fn, lab = rest.rsplit(":", 1); d = np.load(fn); n, tr = d[f"{lab}_n"], d[f"{lab}_Trot"]; n2 = float(d[f"{lab}_n2"]); T2 = float(d[f"{lab}_T2"])
    show(name, (n - 1) / (n2 - 1), (tr - 1) / (T2 - 1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
