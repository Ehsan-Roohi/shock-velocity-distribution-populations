"""State of the unscattered stream ahead of and inside the shock, and the conserved fluxes (dsmc_shock_vdf18-20).
For each label (runs pooled): at given distances from the density midpoint and at given normalised densities n*,
the share of unscattered molecules, their mean velocity (units u1), translational temperature (units T1), and mean
rotational energy (units kT1, against the mean of the populations that enter); then the fluxes of mass and momentum
(range over the domain, relative to the upstream values) and the energy flux at the downstream end over that at the
upstream end.
Usage (in the DSMC folder): python n2_stream_state.py <label>=<glob of .txt files> [...]"""
import sys, glob, math
import numpy as np
from n2_check import read


def cls_moments(stem, nc, nb):
    raw = open(stem + ".cls.bin", "rb").read(); off = 9 * nc * nb * 4
    return np.frombuffer(raw[off:off + 9 * nc * 4 * 8], dtype=np.float64).reshape(9, nc, 4)


for spec in sys.argv[1:]:
    lab, pat = spec.split("=", 1); stems = sorted(s[:-4] for s in glob.glob(pat))
    stems = [s for s in stems if "steps_done 215000" in open(s + ".txt").read()]
    runs = [read(s) for s in stems]; h = runs[0][0]
    M, gam, nc, dx, nb = h["M"], h["gamma"], int(h["nc"]), h["dx"], int(h["nb"])
    u1 = M * math.sqrt(gam); n2 = h.get("n2res", (gam + 1) * M * M / ((gam - 1) * M * M + 2)); me = h.get("rin_me", 1.0)
    x = h["xmin"] + (np.arange(nc) + 0.5) * dx
    mom = sum(r[2] for r in runs); rot = sum(r[4] for r in runs); ns = sum(r[3] for r in runs); cm = sum(cls_moments(s, nc, nb) for s in stems)
    N = mom[:, 0]; n = N * h["FN"] / (ns * dx); u = mom[:, 1] / N; nstar = (n - 1) / (n2 - 1)
    k = np.ones(5) / 5; nss = np.convolve(nstar, k, "same"); ic = int(np.argmin(np.abs(nss - 0.5))); xc = x[ic]
    c0 = cm[0]; w = c0[:, 0]
    T2b = (2 * gam * M * M - (gam - 1)) / (gam + 1) / ((gam + 1) * M * M / ((gam - 1) * M * M + 2))   # unit of the energy bins
    erl = sum(np.fromfile(s + ".erl.bin").reshape(-1, nc, 160) for s in stems)[2]; erh = sum(np.fromfile(s + ".erh.bin").reshape(-1, nc, 160) for s in stems)[2]
    el = (np.arange(160) + 0.5) * 0.005 * T2b; eh = (np.arange(160) + 0.5) * 0.05 * T2b      # generation 0 = unscattered
    erot0 = lambda s: ((erl[s].sum(0) * el).sum() + (erh[s].sum(0)[16:] * eh[16:]).sum()) / (erl[s].sum() + erh[s].sum(0)[16:].sum())
    print(f"{lab}: Mach {M:g}, {len(stems)} runs, freeze {int(h.get('freeze', 0))}, entering rotational energy {me:.3f} kT1, n2 {n2:.4f}")
    pts = [("x-xc=%g" % d, int(np.argmin(np.abs(x - xc - d)))) for d in (-30, -15, -10, -6)]
    for t in (0.02, 0.07, 0.14, 0.3):
        j = np.where((nss >= t) & (np.arange(nc) < ic))[0]
        if len(j): pts.append(("n*=%g" % t, int(j[0])))
    for name, i in pts:
        s = slice(i - 2, i + 3); ww = w[s].sum()
        uu = c0[s, 1].sum() / ww; tp = c0[s, 2].sum() / ww - uu ** 2; tq = 0.5 * c0[s, 3].sum() / ww
        print(f"   {name:>10s} (x-xc {x[i] - xc:6.2f}): unscattered share {ww / N[s].sum():.3f}, u/u1 {uu / u1:.4f}, "
              f"T_tr/T1 {(tp + 2 * tq) / 3:.3f} (par {tp:.3f}, perp {tq:.3f}), e_rot/kT1 {erot0(s):.3f}")
    mass = n * u; mome = n * mom[:, 2] / N; ens = n * (0.5 * (mom[:, 3] + mom[:, 6]) / N) + n * u * rot[0] / N
    up, dn = slice(20, 120), slice(nc - 120, nc - 20); m1 = mass[up].mean(); p1 = mome[up].mean()
    print(f"   fluxes: mass {mass.min() / m1:.4f} to {mass.max() / m1:.4f}; momentum {mome.min() / p1:.4f} to {mome.max() / p1:.4f}; "
          f"energy downstream/upstream {ens[dn].mean() / ens[up].mean():.4f}; downstream n {n[dn].mean():.4f} (reservoir {n2:.4f})")
