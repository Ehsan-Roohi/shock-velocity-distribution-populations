"""Combine the seeds of each DSMC case and write reduced distributions in the common format.

Reads runs/M<M>_w<omega>_s<seed>.{txt,bin} written by dsmc_shock_vdf.cpp. For every case it writes
<out>/DSMC_w<omega>_M<M>_reduced_vdf.npz (same keys as the kinetic-model files, plus direct moment
sums and four time blocks) and prints checks: Rankine-Hugoniot states at both ends, conserved
fluxes, drift between the first and last time block, and the number of samples.

Usage: python dsmc_to_reduced.py <runs dir> <out dir>
"""
import sys, os, glob, json, re
import numpy as np

GAM = 5.0 / 3.0


def rh(M):
    r = (GAM + 1) * M ** 2 / ((GAM - 1) * M ** 2 + 2); p = (2 * GAM * M ** 2 - (GAM - 1)) / (GAM + 1)
    u1 = M * np.sqrt(GAM)
    return dict(rho1=1.0, u1=float(u1), T1=1.0, rho2=float(r), u2=float(u1 / r), T2=float(p / r), gamma=GAM, M1=float(M))


def read(stem):
    h = dict(l.split() for l in open(stem + ".txt") if l.strip())
    nc, nb, NB, NM = (int(h[k]) for k in ("nc", "nb", "NB", "NM"))
    a = np.fromfile(stem + ".bin")
    cnt = a[:nc * nb].reshape(nc, nb); hs = a[nc * nb:2 * nc * nb].reshape(nc, nb)
    mom = a[2 * nc * nb:2 * nc * nb + NB * nc * NM].reshape(NB, nc, NM); ns = a[-NB:]
    return h, cnt, hs, mom, ns


def central(m, FN, dx, ns):
    """Moments from direct sums m[..., 7] = sum of 1, v, v2, v3, v4, w2, v*w2 over samples."""
    n = m[..., 0] * FN / (dx * ns)
    a1, a2, a3, a4, b0, b1 = (m[..., k] / m[..., 0] for k in (1, 2, 3, 4, 5, 6))
    u = a1; Tpar = a2 - u ** 2; Tperp = 0.5 * b0; T = (Tpar + 2 * Tperp) / 3
    c3 = a3 - 3 * u * a2 + 2 * u ** 3
    c4 = a4 - 4 * u * a3 + 6 * u ** 2 * a2 - 3 * u ** 4
    q = 0.5 * n * (c3 + (b1 - u * b0))
    return dict(rho=n, u=u, T=T, Tpar=Tpar, Tperp=Tperp, q=q, sigma=n * (Tpar - T), M300=n * c3, M400=n * c4 - 3 * n * T ** 2)


def main(runs, out):
    os.makedirs(out, exist_ok=True)
    cases = sorted({re.sub(r"_s\d+\.txt$", "", os.path.basename(p)) for p in glob.glob(os.path.join(runs, "M*_w*_s*.txt"))})
    for case in cases:
        tot = None; nseed = 0
        for stem in sorted(p[:-4] for p in glob.glob(os.path.join(runs, case + "_s*.txt"))):
            h, cnt, hs, mom, ns = read(stem)
            if int(h["steps_done"]) < 215000:
                print("  incomplete:", os.path.basename(stem), h["steps_done"]); continue
            tot = [cnt, hs, mom, ns] if tot is None else [tot[0] + cnt, tot[1] + hs, tot[2] + mom, tot[3] + ns]
            nseed += 1
        if tot is None:
            continue
        cnt, hs, mom, ns = tot
        M, om, dx, dv, FN = (float(h[k]) for k in ("M", "omega", "dx", "dv", "FN"))
        nc, nb = cnt.shape; st = rh(M)
        x = float(h["xmin"]) + (np.arange(nc) + 0.5) * dx
        vx = float(h["vlo"]) + (np.arange(nb) + 0.5) * dv; wx = np.full(nb, dv)
        G = cnt * FN / (dx * dv * ns.sum()); H = hs * FN / (dx * dv * ns.sum())
        d = central(mom.sum(0), FN, dx, ns.sum())
        blk = [central(mom[b], FN, dx, ns[b]) for b in range(len(ns))]
        nstar = (d["rho"] - 1) / (st["rho2"] - 1)
        J = d["rho"] * d["u"]; P = d["rho"] * (d["u"] ** 2 + d["Tpar"])
        E = d["rho"] * d["u"] * (0.5 * d["u"] ** 2 + 2.5 * d["T"]) + d["u"] * d["sigma"] + d["q"]
        Js, Ps, Es = st["u1"], st["u1"] ** 2 + 1, st["u1"] * (0.5 * st["u1"] ** 2 + 2.5)
        xc = lambda r: float(np.interp(0.5, (r["rho"] - 1) / (st["rho2"] - 1), x))
        tag = f"DSMC_w{om:g}_M{M:g}".replace(".", "p")
        np.savez(os.path.join(out, tag + "_reduced_vdf.npz"), x_mfp=x, vx=vx, wx=wx, G=G, H=H, K=np.zeros_like(G),
                 states=json.dumps(st), mach=M, model=f"DSMC, VHS omega={om:g}", nstar=nstar,
                 **{"direct_" + k: v for k, v in d.items()},
                 block_rho=np.array([b["rho"] for b in blk]), block_q=np.array([b["q"] for b in blk]),
                 block_Tpar=np.array([b["Tpar"] for b in blk]), nseeds=nseed, nsamples=ns.sum(),
                 count_scale=dx * dv * ns.sum() / FN)   # counts per bin = G * count_scale
        e = slice(5, 60); o = slice(-60, -5)
        print(f"{tag:18s} seeds {nseed}  samples/cell(upstream) {mom.sum(0)[e, 0].mean():.2e}  "
              f"up rho,u/u1,T {d['rho'][e].mean():.4f} {d['u'][e].mean()/st['u1']:.4f} {d['T'][e].mean():.4f}  "
              f"down rho/RH,T/RH {d['rho'][o].mean()/st['rho2']:.4f} {d['T'][o].mean()/st['T2']:.4f}  "
              f"flux dev {100*np.abs(J/Js-1).max():.2f}/{100*np.abs(P/Ps-1).max():.2f}/{100*np.abs(E/Es-1).max():.2f}%  "
              f"x_c blocks {' '.join(f'{xc(b):+.3f}' for b in blk)}  TparTperp max {np.max(d['Tpar']/d['Tperp']):.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
