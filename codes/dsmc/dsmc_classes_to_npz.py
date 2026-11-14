"""Combine the seeds of version-2 DSMC runs and write the collision-history decomposition.

Reads <runs>/M<M>_w<omega>_s<seed>.{txt,bin,cls.bin} written by dsmc_shock_vdf2.cpp and writes, for every
case, <out>/DSMC_w<omega>_M<M>_classes.npz with
  x, vx, nstar, states                      positions, velocity bins, normalized density, RH states
  G                [nc, nb]                 total reduced distribution g (as in the *_reduced_vdf.npz files)
  Gown             [5, nc, nb]              g of the own-count classes 0, 1, 2, 3+ and "struck by the beam"
  Ggen             [4, nc, nb]              g of generations 0, 1, 2, 3+
  mown, mgen       [5 or 4, nc, 4]          class density, mean velocity, T_par and T_perp
The generation of a molecule is the number of collisions that separates it from the unscattered beam.

Usage: python dsmc_classes_to_npz.py <runs dir> <out dir> [min_steps]
"""
import sys, os, glob, json, re
import numpy as np

GAM = 5.0 / 3.0


def rh(M):
    r = (GAM + 1) * M ** 2 / ((GAM - 1) * M ** 2 + 2); p = (2 * GAM * M ** 2 - (GAM - 1)) / (GAM + 1)
    u1 = M * np.sqrt(GAM)
    return dict(rho1=1.0, u1=float(u1), T1=1.0, rho2=float(r), u2=float(u1 / r), T2=float(p / r), gamma=GAM, M1=float(M))


def main(runs, out, min_steps=215000):
    os.makedirs(out, exist_ok=True)
    cases = sorted({re.sub(r"_s\d+\.txt$", "", os.path.basename(p)) for p in glob.glob(os.path.join(runs, "M*_w*_s*.txt"))})
    for case in cases:
        tot = None; nseed = 0
        for stem in sorted(p[:-4] for p in glob.glob(os.path.join(runs, case + "_s*.txt"))):
            h = dict(l.split() for l in open(stem + ".txt") if l.strip())
            if int(h["steps_done"]) < min_steps or not os.path.exists(stem + ".cls.bin"):
                print("  skipped:", os.path.basename(stem), h["steps_done"]); continue
            nc, nb, NB, NM, NCLS, NCM = (int(h[k]) for k in ("nc", "nb", "NB", "NM", "NCLS", "NCM"))
            a = np.fromfile(stem + ".bin"); cnt = a[:nc * nb].reshape(nc, nb); ns = a[-NB:].sum()
            cc = np.fromfile(stem + ".cls.bin", dtype=np.float32, count=NCLS * nc * nb).reshape(NCLS, nc, nb).astype(np.float64)
            cm = np.fromfile(stem + ".cls.bin", dtype=np.float64, offset=4 * NCLS * nc * nb).reshape(NCLS, nc, NCM)
            assert np.allclose(cc[:5].sum(0), cnt) and np.allclose(cc[5:].sum(0), cnt), "class histograms do not add up"
            tot = [cnt, cc, cm, ns] if tot is None else [tot[0] + cnt, tot[1] + cc, tot[2] + cm, tot[3] + ns]
            nseed += 1
        if tot is None:
            continue
        cnt, cc, cm, ns = tot
        M, om, dx, dv, FN = (float(h[k]) for k in ("M", "omega", "dx", "dv", "FN"))
        st = rh(M); scale = FN / (dx * dv * ns)
        x = float(h["xmin"]) + (np.arange(nc) + 0.5) * dx; vx = float(h["vlo"]) + (np.arange(nb) + 0.5) * dv
        n = cm[:, :, 0] * FN / (dx * ns)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cm[:, :, 1] / cm[:, :, 0]; Tpar = cm[:, :, 2] / cm[:, :, 0] - u ** 2; Tperp = 0.5 * cm[:, :, 3] / cm[:, :, 0]
        mom = np.stack([n, u, Tpar, Tperp], 2)
        rho = n[5:].sum(0); nstar = (rho - 1) / (st["rho2"] - 1)
        tag = f"DSMC_w{om:g}_M{M:g}".replace(".", "p")
        np.savez_compressed(os.path.join(out, tag + "_classes.npz"), x=x, vx=vx, dv=dv, nstar=nstar, rho=rho, states=json.dumps(st),
                            mach=M, omega=om, G=(cnt * scale).astype(np.float32), Gown=(cc[:5] * scale).astype(np.float32),
                            Ggen=(cc[5:] * scale).astype(np.float32), mown=mom[:5], mgen=mom[5:], nseeds=nseed,
                            count_scale=1.0 / scale)
        j = int(np.argmin(np.abs(nstar - 0.2)))
        print(f"{tag:20s} seeds {nseed}  generation fractions at n*=0.2: " + " ".join(f"{v:.3f}" for v in n[5:, j] / rho[j]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 215000)
