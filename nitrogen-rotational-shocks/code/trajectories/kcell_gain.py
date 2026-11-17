"""Single collision of the precursor ensemble through the trajectory kernel (kcell.py), counted by the GAIN of
rotational energy of the stream molecule (e' - e > E) and by crossing the level E from below, next to the same from
the direct trajectories (ctc_condition.py), in units of the rate-averaged VHS cross-section.

Usage: python kcell_gain.py <out.npz> <cond npz Mach 12.9> <cond npz Mach 7> <parts c_*.npz>"""
import sys, math
import numpy as np
from kcell import KCell

out, c129, c7 = sys.argv[1:4]; KT = KCell(sys.argv[4:]); rng = np.random.default_rng(5); res = {}


def above(x, wt, grid):
    k = np.argsort(x); cw = np.cumsum(wt[k]); idx = np.searchsorted(x[k], grid, side="right")
    return cw[-1] - np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0)


for name, cond, P in (("M12p9", c129, dict(u1=797.0, Th=442.0, Trh=400.0, T1=9.16)), ("M7", c7, dict(u1=759.0, Th=414.0, Trh=379.0, T1=28.3))):
    d = np.load(cond); T2 = float(d["T2"]); T1 = P["T1"]; E = d["E"]; sv = float(d["sig_vhs"]); VU = 91.18; MM = 28.0; MU = 14.0; nn = 3000000
    v1 = rng.normal(0, math.sqrt(T1 / MM), (nn, 3)); v1[:, 0] += P["u1"] / VU; v2 = rng.normal(0, math.sqrt(P["Th"] / MM), (nn, 3)); v2[:, 0] = -np.abs(v2[:, 0])
    e1 = -T1 * np.log(1 - rng.random(nn)); cold = rng.random(nn) < 0.55; e2 = np.where(cold, -T1 * np.log(1 - rng.random(nn)), -P["Trh"] * np.log(1 - rng.random(nn)))
    g = np.linalg.norm(v1 - v2, axis=1); cr = 2 * np.sqrt(0.5 * MU * g ** 2); scr, tab = KT.sigma_cr(cr, e1, e2); e1n, e2n, etn, cc, null, moved = KT.collide(cr, e1, e2, rng)
    e1n = np.where(null | ~tab, e1, e1n); w = g * np.where(tab, scr / cr, 0.0); norm = np.sum(g) * sv
    fa = above(e1n, w, E * T2) / norm; fg = above(e1n - e1, w, E * T2) / norm
    fc = np.array([np.sum(w[(e1n > x * T2) & (e1 <= x * T2)]) for x in E]) / norm
    res.update({f"{name}_E": E, f"{name}_after": fa, f"{name}_gain": fg, f"{name}_cross": fc, f"{name}_gain_direct": d["gain"], f"{name}_cross_direct": d["cross"], f"{name}_after_direct": d["after"]})
    i = lambda x: int(np.argmin(np.abs(E - x)))
    for nm, a, b in (("after", fa, d["after"]), ("crossing", fc, d["cross"]), ("gain", fg, d["gain"])):
        print(f"{name} {nm:9s} at 0.1, 0.3, 1, 2, 3 kT2: kernel " + " ".join(f"{a[i(x)]:.3f}" for x in (0.1, 0.3, 1, 2, 3)) + " | direct " + " ".join(f"{b[i(x)]:.3f}" for x in (0.1, 0.3, 1, 2, 3))
              + f" | slope 1-3 kernel {2 / math.log(a[i(1)] / a[i(3)]):.3f}, direct {2 / math.log(b[i(1)] / b[i(3)]):.3f}")
    print(f"{name} mean gain per VHS collision: kernel {np.sum(w * (e1n - e1)) / norm / T2:.3f} kT2")
np.savez(out, **res)
