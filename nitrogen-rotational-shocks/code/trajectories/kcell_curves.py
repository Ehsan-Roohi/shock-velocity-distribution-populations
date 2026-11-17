"""Curves for the tests of the trajectory kernel (kcell.py) shown in the supplemental material:
  - the single-collision excitation function of the precursor ensemble through the kernel, on the energy grid of the
    direct trajectories (ctc_excite.py), for the Mach 12.9 and Mach 7 conditions;
  - the relaxation of a homogeneous gas started with cold rotation (T_tr and T_rot against the number of VHS
    collisions per molecule), and a gas started in equilibrium.

Usage: python kcell_curves.py <out.npz> <exc npz Mach 12.9> <exc npz Mach 7> <parts c_*.npz> [T=300] [n=200000]"""
import sys, math
import numpy as np
from kcell import KCell, sig_vhs

args = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
out, exc129, exc7 = args[:3]; KT = KCell(args[3:]); T0 = float(opt.get("T", 300)); n = int(float(opt.get("n", 200000))); rng = np.random.default_rng(2); res = {}


def turn(g, cc):
    gm = np.linalg.norm(g, axis=1); u = g / gm[:, None]; a = np.cross(u, rng.normal(size=u.shape)); a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(u, a); ph = 2 * math.pi * rng.random(len(g)); sc = np.sqrt(np.maximum(1 - cc ** 2, 0))
    return gm[:, None] * (cc[:, None] * u + sc[:, None] * (np.cos(ph)[:, None] * a + np.sin(ph)[:, None] * b))


def run(Ttr, Trot, ncoll):
    v = rng.normal(0, math.sqrt(Ttr), (n, 3)); e = -Trot * np.log(1 - rng.random(n)); crmax = 0.0; N = 0.0; hist = [(0.0, (v * v).sum(1).mean() / 3, e.mean())]
    while N < ncoll:
        o = rng.permutation(n); a, b = o[:n // 2], o[n // 2:2 * (n // 2)]; g = v[a] - v[b]; cr = np.linalg.norm(g, axis=1)
        scr, tab = KT.sigma_cr(cr, e[a], e[b]); crmax = max(crmax, 1.05 * scr.max()); acc = rng.random(len(cr)) * crmax < scr
        N += float(np.mean(sig_vhs(0.25 * cr * cr) * cr) / crmax); k = np.where(acc & tab)[0]
        if len(k):
            e1n, e2n, etn, cc, null, moved = KT.collide(cr[k], e[a[k]], e[b[k]], rng); m = ~null; kk = k[m]
            gn = turn(g[kk] * (2 * np.sqrt(etn[m]) / cr[kk])[:, None], cc[m]); vc = 0.5 * (v[a[kk]] + v[b[kk]])
            v[a[kk]] = vc + 0.5 * gn; v[b[kk]] = vc - 0.5 * gn; e[a[kk]] = e1n[m]; e[b[kk]] = e2n[m]
        k = np.where(acc & ~tab)[0]
        if len(k):
            ct = 2 * rng.random(len(k)) - 1; st = np.sqrt(1 - ct ** 2); ph = 2 * math.pi * rng.random(len(k)); gn = cr[k][:, None] * np.stack([ct, st * np.cos(ph), st * np.sin(ph)], 1)
            vc = 0.5 * (v[a[k]] + v[b[k]]); v[a[k]] = vc + 0.5 * gn; v[b[k]] = vc - 0.5 * gn
        hist.append((N, (v * v).sum(1).mean() / 3, e.mean()))
    return np.array(hist), e


h, e = run(T0, T0, 20.0); s = h[:, 0] > 8; res["eq_hist"] = h; res["eq_ratio"] = h[s, 2].mean() / h[s, 1].mean(); res["eq_err"] = 2.2 / math.sqrt(n)
res["eq_above2"] = np.mean(e > 2 * e.mean()); res["eq_below01"] = np.mean(e < 0.1 * e.mean()); res["eq_m2"] = np.mean(e * e) / (2 * e.mean() ** 2)
print(f"equilibrium at {T0:g} K: T_rot/T_tr = {res['eq_ratio']:.4f} +- {res['eq_err']:.4f}; share above 2 T {res['eq_above2']:.4f} ({math.exp(-2):.4f}), below 0.1 T {res['eq_below01']:.4f} ({1 - math.exp(-0.1):.4f}); <e^2>/2<e>^2 {res['eq_m2']:.4f}")
Ttr0 = (2.5 * T0 - 0.1 * T0) / 1.5; h, e = run(Ttr0, 0.1 * T0, 12.0); dT = h[:, 1] - h[:, 2]; m = dT > 0.05 * dT[0]; m2 = dT > 0.5 * dT[0]
res["rel_hist"] = h; res["rel_rate"] = -np.polyfit(h[m, 0], np.log(dT[m]), 1)[0]; res["rel_rate_early"] = -np.polyfit(h[m2, 0], np.log(dT[m2]), 1)[0]
print(f"relaxation from T_rot = {0.1 * T0:g} K: rate per VHS collision {res['rel_rate']:.4f} (first half {res['rel_rate_early']:.4f}); equivalent LB probability {res['rel_rate'] / 1.063:.3f} ({res['rel_rate_early'] / 1.063:.3f})")
for name, exc, P in (("M12p9", exc129, dict(u1=797.0, Th=442.0, Trh=400.0)), ("M7", exc7, dict(u1=759.0, Th=414.0, Trh=379.0))):
    d = np.load(exc); T1 = float(d["T1"]); T2 = float(d["T2"]); E = d["E"]; sv = float(d["sig_vhs"]); VU = 91.18; MM = 28.0; MU = 14.0; nn = 3000000
    v1 = rng.normal(0, math.sqrt(T1 / MM), (nn, 3)); v1[:, 0] += P["u1"] / VU; v2 = rng.normal(0, math.sqrt(P["Th"] / MM), (nn, 3)); v2[:, 0] = -np.abs(v2[:, 0])
    e1 = -T1 * np.log(1 - rng.random(nn)); cold = rng.random(nn) < 0.55; e2 = np.where(cold, -T1 * np.log(1 - rng.random(nn)), -P["Trh"] * np.log(1 - rng.random(nn)))
    g = np.linalg.norm(v1 - v2, axis=1); cr = 2 * np.sqrt(0.5 * MU * g ** 2); scr, tab = KT.sigma_cr(cr, e1, e2); e1n, e2n, etn, cc, null, moved = KT.collide(cr, e1, e2, rng)
    e1n = np.where(null | ~tab, e1, e1n); w = g * np.where(tab, scr / cr, 0.0); o = np.argsort(e1n); cw = np.cumsum(w[o]); idx = np.searchsorted(e1n[o], E * T2, side="right")
    f = (cw[-1] - np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0)) / np.sum(g) / sv; res[f"{name}_E"] = E; res[f"{name}_kernel"] = f; res[f"{name}_direct"] = d["ctc"] / sv
    i = lambda x: int(np.argmin(np.abs(E - x))); fd = d["ctc"] / sv
    print(f"{name}: above 0.1, 0.3, 1, 2, 3 kT2: kernel " + " ".join(f"{f[i(x)]:.3f}" for x in (0.1, 0.3, 1, 2, 3)) + f" (slope {2 / math.log(f[i(1)] / f[i(3)]):.3f}); direct "
          + " ".join(f"{fd[i(x)]:.3f}" for x in (0.1, 0.3, 1, 2, 3)) + f" (slope {2 / math.log(fd[i(1)] / fd[i(3)]):.3f})")
np.savez(out, **res)
