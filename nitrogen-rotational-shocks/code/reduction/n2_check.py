"""Check of a diatomic DSMC shock (dsmc_shock_vdf7): Rankine-Hugoniot states, rotational and translational
temperatures at the two ends, conserved fluxes, and the positions where the upstream-moving mass fraction beta takes
given values. Also the rotational temperature of the upstream-moving molecules by generation at those positions.

Usage: python n2_check.py <run stem> [<run stem> ...]        (stem = path without .txt)
"""
import sys, math
import numpy as np


def read(stem):
    h = {}
    for ln in open(stem + ".txt"):
        k, v = ln.split(); h[k] = float(v)
    nc, nb, NB, NM = int(h["nc"]), int(h["nb"]), int(h["NB"]), int(h["NM"])
    a = np.fromfile(stem + ".bin")
    cnt = a[:nc * nb].reshape(nc, nb); mom = a[2 * nc * nb:2 * nc * nb + NB * nc * NM].reshape(NB, nc, NM).sum(0)
    ns = a[2 * nc * nb + NB * nc * NM:].sum()
    rot = np.fromfile(stem + ".rot.bin").reshape(int(h["NR"]), nc)
    fly = np.fromfile(stem + ".fly.bin").reshape(4, nc)
    return h, cnt, mom, ns, rot, fly


def main(stem):
    h, cnt, mom, ns, rot, fly = read(stem)
    M, gam, nc, dx = h["M"], h["gamma"], int(h["nc"]), h["dx"]
    x = h["xmin"] + (np.arange(nc) + 0.5) * dx
    u1 = M * math.sqrt(gam); n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); T2 = (2 * gam * M * M - (gam - 1)) / (gam + 1) / n2; u2 = u1 / n2
    N = mom[:, 0]; n = N * h["FN"] / (ns * dx); u = mom[:, 1] / N
    Tpar = mom[:, 2] / N - u ** 2; Tperp = 0.5 * mom[:, 5] / N; Ttr = (Tpar + 2 * Tperp) / 3; Trot = rot[0] / N
    # fluxes (per unit n1): mass n u; momentum n (u^2 + Tpar); energy n [ <v (v^2 + w^2)>/2 + <v er> ] -- the last needs
    # <v er>, not sampled; in equilibrium ends it is u Trot, so the energy flux is checked at the two ends only.
    mass = n * u; mome = n * mom[:, 2] / N
    ens = n * (0.5 * (mom[:, 3] + mom[:, 6]) / N)
    up, dn = slice(20, 120), slice(nc - 120, nc - 20)
    print(f"{stem}: M={M:g} gamma={gam:g} steps={int(h['steps_done'])}  RH: n2={n2:.4f} T2={T2:.4f} u2={u2:.4f}")
    print(f"  upstream:   n={n[up].mean():.4f} u={u[up].mean():.4f} (u1={u1:.4f}) Ttr={Ttr[up].mean():.4f} Trot={Trot[up].mean():.4f}")
    print(f"  downstream: n={n[dn].mean():.4f} u={u[dn].mean():.4f} Ttr={Ttr[dn].mean():.4f} Trot={Trot[dn].mean():.4f} Tpar/Tperp={np.mean(Tpar[dn] / Tperp[dn]):.4f}")
    print(f"  mass flux/n1u1: min {mass.min() / u1:.4f} max {mass.max() / u1:.4f}; momentum flux / (u1^2+1): min {mome.min() / (u1 ** 2 + 1):.4f} max {mome.max() / (u1 ** 2 + 1):.4f}")
    e1 = ens[up].mean() + (n * u * Trot)[up].mean(); e2 = ens[dn].mean() + (n * u * Trot)[dn].mean()
    print(f"  energy flux downstream/upstream: {e2 / e1:.4f}")
    nstar = (n - 1) / (n2 - 1); ic = int(np.argmin(np.abs(nstar - 0.5)))
    xc = x[ic]; j = np.arange(nc)
    i10, i90 = int(np.argmin(np.abs(nstar - 0.1))), int(np.argmin(np.abs(nstar - 0.9)))
    slope = np.max(np.gradient(n, x)); print(f"  centre x={xc:.2f}; width (n2-n1)/max slope = {(n2 - 1) / slope:.2f}; x(n*=0.1)={x[i10]:.2f} x(n*=0.9)={x[i90]:.2f}; peak Ttr/T2={Ttr.max() / T2:.3f} peak Tpar/T2={Tpar.max() / T2:.3f}")
    beta = fly[0] / N
    k = np.where((j < ic) & (j > 50) & (beta > 0))[0]
    for b in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4):
        if beta[k].min() > b:
            print(f"  beta={b:g}: not reached inside the domain"); continue
        i = k[np.argmin(np.abs(np.log(beta[k]) - math.log(b)))]; s = slice(i - 2, i + 3)
        cg = rot[1:5, s].sum(1); eg = rot[5:9, s].sum(1); tg = rot[15:19, s].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            tr = eg / cg; tt = tg / cg * 2 / 3
        print(f"  beta={b:g}: x={x[i]:.2f} (x-xc={x[i] - xc:.2f})  shares of gen 0-3 among upstream-moving: " + " ".join(f"{v:.3f}" for v in cg / cg.sum())
              + " | Trot/T2 by gen: " + " ".join(f"{v:.3f}" for v in tr / T2) + f" | all: {eg.sum() / cg.sum() / T2:.3f}"
              + " | (2/3)E_tr/T2 by gen: " + " ".join(f"{v:.2f}" for v in tt / T2) + f" | local Trot of all molecules/T1: {Trot[s].mean():.4f}")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
