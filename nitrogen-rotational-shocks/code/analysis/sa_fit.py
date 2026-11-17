"""Fit of the state-dependent acceptance a(x) = ainf + (a0 - ainf) exp(-x/x0), x = min(e, e')/Ec, of the Larsen-Borgnakke
exchange to the mean rotational-energy changes of the classical-trajectory table.

For a molecule with rotational energy e in a collision of relative energy Et (Ec = Et + e), the rule gives per VHS
collision the mean change  D(Et, e) = int pi(e'|Ec) a(min(e, e')/Ec) (e' - e) de',  pi = (eta/Ec)(1 - e'/Ec)^(eta-1).
The trajectories give  (sigma_max/sigma_VHS(Et)) <e' - e>.

Usage: python sa_fit.py <json of ctc_table_analyse.py>   or   python sa_fit.py logs <log files of ctc_table.py>"""
import sys, math, json, re
import numpy as np
from scipy.optimize import least_squares

om = 0.74; eta = 2.5 - om; bmax = 5.5
sv = lambda et: math.pi * 4.17 ** 2 * (273.0 / et) ** (om - 0.5) / math.gamma(2.5 - om)
rows = []          # (Et, e of the molecule, e of the partner, mean change per VHS collision)
if sys.argv[1] == "logs":
    pat = re.compile(r"Et (\S+) K, e1 (\S+) K, e2 (\S+) K;.*mean change of e1 (\S+) K, of e2 (\S+) K")
    for f in sys.argv[2:]:
        for ln in open(f):
            m = pat.search(ln)
            if m:
                et, e1, e2, d1, d2 = (float(v) for v in m.groups()); r = math.pi * bmax ** 2 / sv(et)
                rows.append((et, e1, e2, r * d1));
                if e1 != e2: rows.append((et, e2, e1, r * d2))
                else: rows[-1] = (et, e1, e2, r * 0.5 * (d1 + d2))
else:
    for q in json.load(open(sys.argv[1])):
        r = math.pi * bmax ** 2 / sv(q["Et"])
        if q["e1"] == q["e2"]: rows.append((q["Et"], q["e1"], q["e2"], r * 0.5 * (q["d1"] + q["d2"])))
        else: rows.append((q["Et"], q["e1"], q["e2"], r * q["d1"])); rows.append((q["Et"], q["e2"], q["e1"], r * q["d2"]))
rows = np.array(rows); u = np.linspace(0, 1, 4001)[1:-1]


def model(par, et, e):
    a0, ainf, x0 = par; ec = et + e; ep = ec * u; pi = eta * (1 - u) ** (eta - 1)
    a = ainf + (a0 - ainf) * np.exp(-np.minimum(e, ep) / ec / x0)
    y = pi * a * (ep - e); return float(((y[1:] + y[:-1]) * np.diff(u)).sum() / 2)


def resid(par, sel=None):
    R = rows if sel is None else rows[sel]
    return np.array([(model(par, et, e) - d) / et for et, e, e2, d in R])


print(f"{len(rows)} entries; relative energies {sorted(set(rows[:, 0]))}")
fit = least_squares(resid, [0.45, 0.1, 0.1], bounds=([0, 0, 0.005], [1, 1, 5]))
a0, ainf, x0 = fit.x; print(f"all entries: a0 = {a0:.3f}, ainf = {ainf:.3f}, x0 = {x0:.3f}; rms residual {np.sqrt(np.mean(fit.fun ** 2)):.4f} of Et")
lb = least_squares(lambda p: resid([p[0], p[0], 1.0]), [0.2], bounds=([0], [1])); print(f"constant probability (ordinary rule): {lb.x[0]:.3f}; rms residual {np.sqrt(np.mean(lb.fun ** 2)):.4f} of Et")
for et in sorted(set(rows[:, 0])):
    s = rows[:, 0] == et; f1 = least_squares(lambda p: resid(p, s), [0.45, 0.1, 0.1], bounds=([0, 0, 0.005], [1, 1, 5]))
    print(f"   Et = {et:g} K alone: a0 = {f1.x[0]:.3f}, ainf = {f1.x[1]:.3f}, x0 = {f1.x[2]:.3f}; rms residual {np.sqrt(np.mean(f1.fun ** 2)):.4f}")
print("\n  Et     e    partner   trajectories   state-dependent rule   ordinary rule (best constant)      [mean change of e per VHS collision, K]")
for et, e, e2, d in rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]:
    print(f"{et:6g} {e:6g} {e2:8g} {d:12.1f} {model(fit.x, et, e):16.1f} {model([lb.x[0], lb.x[0], 1.0], et, e):20.1f}")
