"""Tests of the cell kernel (kcell.py, the rule of dsmc_shock_vdf16.cpp) in a homogeneous gas.

  equilibrium   start at T_tr = T_rot = T and collide: the temperatures must stay equal and the rotational energy
                exponential (the kernel satisfies detailed balance by construction);
  relaxation    start with cold rotation: decay rate of T_tr - T_rot per VHS collision, expressed as the constant
                Larsen-Borgnakke probability with the same rate (rate = 1.063 p per VHS collision);
  one collision the precursor ensemble of ctc_kernel.py (stream molecule against an upstream-moving molecule) through
                the kernel, next to the direct trajectories (ctc_excite.py), if exc=<npz> is given.

Usage: python kcell_test.py <parts c_*.npz> [T=300] [n=200000] [ncoll=25] [fake=<energy index>] [exc=<exc npz>] [synth=1]
synth=1 replaces the trajectories by random outcomes (software test of the detailed balance only)."""
import sys, math
import numpy as np
from kcell import KCell, sig_vhs, OM

files = [a for a in sys.argv[1:] if "=" not in a]; opt = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
T0 = float(opt.get("T", 300)); n = int(float(opt.get("n", 200000))); NCOLL = float(opt.get("ncoll", 25)); rng = np.random.default_rng(1)
if opt.get("synth"):
    import ctc_kcell_grid as cg                                    # grid definitions only
    import tempfile, os
    r2 = np.random.default_rng(5); res = {}
    for q, (k, s) in enumerate(cg.JOBS):
        m = 300; x1 = r2.dirichlet([1.0, 1.0, 2.0], m)[:, :2]; stay = r2.random(m) < 0.4 + 0.3 * (k % 2)
        a, b = cg.SRC[s]; own = np.stack([0.5 * (cg.XE[a] + cg.XE[a + 1]) * np.ones(m), np.minimum(0.5 * (cg.XE[b] + cg.XE[b + 1]), 0.99 - 0.5 * (cg.XE[a] + cg.XE[a + 1])) * np.ones(m)], 1)
        xf = np.where(stay[:, None], own, x1); res[f"j{q}"] = np.stack([xf[:, 0], xf[:, 1], np.where(stay, 0.97, 2 * r2.random(m) - 1)]).astype(np.float32)
    fn = os.path.join(tempfile.gettempdir(), "kcell_synth.npz")
    np.savez(fn, etot=np.array(cg.ETOT), bmax=np.array(cg.BMAX), xe=np.array(cg.XE), src=np.array(cg.SRC), xtmin=cg.XTMIN, **res); files = [fn]
KT = KCell(files, fake=int(opt["fake"]) if "fake" in opt else None)
print(f"kernel: {KT.nS} source cells, {len(KT.dest)} transitions; largest probability of leaving a cell {KT.maxoff:.3f}, rows scaled {KT.scaled}")


def turn(g, cc, rng):
    gm = np.linalg.norm(g, axis=1); u = g / gm[:, None]; a = np.cross(u, rng.normal(size=u.shape)); a /= np.linalg.norm(a, axis=1)[:, None]
    b = np.cross(u, a); ph = 2 * math.pi * rng.random(len(g)); sc = np.sqrt(np.maximum(1 - cc ** 2, 0))
    return gm[:, None] * (cc[:, None] * u + sc[:, None] * (np.cos(ph)[:, None] * a + np.sin(ph)[:, None] * b))


def run(Ttr, Trot, ncoll, label):
    v = rng.normal(0, math.sqrt(Ttr), (n, 3)); e = -Trot * np.log(1 - rng.random(n)); crmax = 0.0; N = 0.0; hist = []; ntab = 0; nmove = 0
    e0 = 1.5 * (v * v).sum(1).mean() / 3 + e.mean()
    while N < ncoll:
        o = rng.permutation(n); a, b = o[:n // 2], o[n // 2:2 * (n // 2)]; g = v[a] - v[b]; cr = np.linalg.norm(g, axis=1)
        scr, tab = KT.sigma_cr(cr, e[a], e[b]); crmax = max(crmax, 1.05 * scr.max()); acc = rng.random(len(cr)) * crmax < scr
        N += float(np.mean(sig_vhs(0.25 * cr * cr) * cr) / crmax)
        k = np.where(acc & tab)[0]
        if len(k):
            e1n, e2n, etn, cc, null, moved = KT.collide(cr[k], e[a[k]], e[b[k]], rng); m = ~null; kk = k[m]; ntab += int(m.sum()); nmove += int(moved[m].sum())
            gn = turn(g[kk] * (2 * np.sqrt(etn[m]) / cr[kk])[:, None], cc[m], rng); vc = 0.5 * (v[a[kk]] + v[b[kk]])
            v[a[kk]] = vc + 0.5 * gn; v[b[kk]] = vc - 0.5 * gn; e[a[kk]] = e1n[m]; e[b[kk]] = e2n[m]
        k = np.where(acc & ~tab)[0]
        if len(k):
            ct = 2 * rng.random(len(k)) - 1; st = np.sqrt(1 - ct ** 2); ph = 2 * math.pi * rng.random(len(k)); gn = cr[k][:, None] * np.stack([ct, st * np.cos(ph), st * np.sin(ph)], 1)
            vc = 0.5 * (v[a[k]] + v[b[k]]); v[a[k]] = vc + 0.5 * gn; v[b[k]] = vc - 0.5 * gn
        hist.append((N, (v * v).sum(1).mean() / 3, e.mean()))
    h = np.array(hist); e1 = 1.5 * h[-1, 1] + h[-1, 2]
    print(f"{label}: {N:.1f} VHS collisions per molecule, {ntab} table collisions of which {nmove / max(ntab, 1):.3f} changed the cell; energy per molecule {e0:.3f} -> {e1:.3f} K")
    return h, e, v


h, e, v = run(T0, T0, NCOLL, f"equilibrium at {T0:g} K"); s = h[:, 0] > 0.4 * h[-1, 0]; ttr, trot = h[s, 1].mean(), h[s, 2].mean()
print(f"   T_tr {ttr:.2f} K, T_rot {trot:.2f} K, ratio T_rot/T_tr {trot / ttr:.4f} (statistical error about {2.2 / math.sqrt(n):.4f})")
print(f"   rotational energy: share above 2 T_rot {np.mean(e > 2 * e.mean()):.4f} (exponential {math.exp(-2):.4f}), below 0.1 T_rot {np.mean(e < 0.1 * e.mean()):.4f} ({1 - math.exp(-0.1):.4f}); "
      f"<e^2>/2<e>^2 = {np.mean(e * e) / (2 * e.mean() ** 2):.4f}; <v^4>/(15 T^2) = {np.mean((v * v).sum(1) ** 2) / (15 * ((v * v).sum(1).mean() / 3) ** 2):.4f}")
Ttr0 = (2.5 * T0 - 0.1 * T0) / 1.5; h, e, v = run(Ttr0, 0.1 * T0, min(NCOLL, 12), f"relaxation, T_tr {Ttr0:.0f} K, T_rot {0.1 * T0:.0f} K"); dT = h[:, 1] - h[:, 2]; m = dT > 0.05 * dT[0]
k = -np.polyfit(h[m, 0], np.log(dT[m]), 1)[0]; m2 = dT > 0.5 * dT[0]; k2 = -np.polyfit(h[m2, 0], np.log(dT[m2]), 1)[0]
print(f"   decay rate of T_tr - T_rot per VHS collision {k:.4f} (first half {k2:.4f}); equivalent constant exchange probability {k / 1.063:.3f} ({k2 / 1.063:.3f}); final T_tr {h[-1, 1]:.1f}, T_rot {h[-1, 2]:.1f} K")
if "exc" in opt:                                                    # one collision of the precursor ensemble through the kernel
    d = np.load(opt["exc"]); T1 = float(d["T1"]); T2 = float(d["T2"]); E = d["E"]; sv = float(d["sig_vhs"]); P = dict(u1=797.0, Th=442.0, Trh=400.0, pcold=0.55)
    for a_ in sys.argv[1:]:
        if a_.split("=")[0] in P: P[a_.split("=")[0]] = float(a_.split("=")[1])
    VU = 91.18; MM = 28.0; MU = 14.0; nn = 1500000
    v1 = rng.normal(0, math.sqrt(T1 / MM), (nn, 3)); v1[:, 0] += P["u1"] / VU; v2 = rng.normal(0, math.sqrt(P["Th"] / MM), (nn, 3)); v2[:, 0] = -np.abs(v2[:, 0])
    e1 = -T1 * np.log(1 - rng.random(nn)); cold = rng.random(nn) < P["pcold"]; e2 = np.where(cold, -T1 * np.log(1 - rng.random(nn)), -P["Trh"] * np.log(1 - rng.random(nn)))
    g = np.linalg.norm(v1 - v2, axis=1); et = 0.5 * MU * g ** 2; cr = 2 * np.sqrt(et)
    scr, tab = KT.sigma_cr(cr, e1, e2); e1n, e2n, etn, cc, null, moved = KT.collide(cr, e1, e2, rng); e1n = np.where(null | ~tab, e1, e1n)
    w = g * np.where(tab, scr / cr, 0.0)                            # rate weight g * cross-section (A^2)
    f = np.array([np.sum(w * (e1n > x * T2)) for x in E]) / np.sum(g) / sv
    ii = [int(np.argmin(np.abs(E - x))) for x in (0.1, 0.3, 1.0, 2.0, 3.0)]; i1, i3 = ii[2], ii[4]
    print(f"one collision at T2 = {T2:g} K: share of the VHS collisions that leave the stream molecule above 0.1, 0.3, 1, 2, 3 T2, and slope temperature between 1 and 3 T2:")
    print("   kernel       " + " ".join(f"{f[i]:6.3f}" for i in ii) + f" | {2.0 / math.log(f[i1] / f[i3]):.3f}")
    fd = d["ctc"] / sv; print("   trajectories " + " ".join(f"{fd[i]:6.3f}" for i in ii) + f" | {2.0 / math.log(fd[i1] / fd[i3]):.3f}")
