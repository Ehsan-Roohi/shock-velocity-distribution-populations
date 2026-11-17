"""Profiles of the nitrogen shock from DSMC (seeds combined) and from the Rykov model, on a common form.
Run in the DSMC folder: python rykov/compare_reduce.py runs_n2/prod rykov/runs compare_n2.npz 7 8 12"""
import sys, glob, math
import numpy as np
sys.path.insert(0, ".")
from n2_check import read

dfold, rfold, outf = sys.argv[1:4]
out = {}
for Ms in sys.argv[4:]:
    runs = [read(s[:-4]) for s in sorted(glob.glob(f"{dfold}/M{Ms}_w0.74_s*.txt"))]
    h = runs[0][0]; M, gam, nc, dx = h["M"], h["gamma"], int(h["nc"]), h["dx"]
    n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); T2 = (2 * gam * M * M - (gam - 1)) / (gam + 1) / n2
    x = h["xmin"] + (np.arange(nc) + 0.5) * dx
    mom = sum(r[2] for r in runs); rot = sum(r[4] for r in runs); fly = sum(r[5] for r in runs); ns = sum(r[3] for r in runs)
    N = mom[:, 0]; n = N * h["FN"] / (ns * dx); u = mom[:, 1] / N
    Tpar = mom[:, 2] / N - u ** 2; Tperp = 0.5 * mom[:, 5] / N; Ttr = (Tpar + 2 * Tperp) / 3; Trot = rot[0] / N
    ic = int(np.argmin(np.abs((n - 1) / (n2 - 1) - 0.5)))
    cu = rot[1:5].sum(0); eu = rot[5:9].sum(0); tu = rot[15:19].sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = dict(x=x - x[ic], n=n, u=u, Ttr=Ttr, Tpar=Tpar, Tperp=Tperp, Trot=Trot, beta=fly[0] / N, Trot_up=eu / cu, Etr_up=tu / cu, nup=cu)
    # seed scatter of Trot (for the downstream relaxation fit)
    for k, v in d.items():
        out[f"dsmc_M{Ms}_{k}"] = v
    out[f"M{Ms}_T2"] = T2; out[f"M{Ms}_n2"] = n2
    # Rykov
    rd = f"{rfold}/M{Ms}"
    v = np.loadtxt(f"{rd}/vnodes.txt").ravel(); a = np.fromfile(f"{rd}/reduced.bin").reshape(-1, v.size, 3)
    p = np.genfromtxt(f"{rd}/profiles.csv", delimiter=",", names=True)
    F, H, R = a[:, :, 0], a[:, :, 1], a[:, :, 2]; nr = F.sum(1); up = v < 0
    nu_ = F[:, up].sum(1); icr = int(np.argmin(np.abs((p["rho"] - 1) / (n2 - 1) - 0.5)))
    Tpar_r = (F * v ** 2).sum(1) / nr - p["u"] ** 2; Tperp_r = 0.5 * R.sum(1) / nr
    with np.errstate(invalid="ignore", divide="ignore"):
        d = dict(x=p["x"] - p["x"][icr], n=p["rho"], u=p["u"], Ttr=p["Ttr"], Tpar=Tpar_r, Tperp=Tperp_r, Trot=p["Trot"], beta=nu_ / nr,
                 Trot_up=H[:, up].sum(1) / nu_, Etr_up=0.5 * ((F[:, up] * v[up] ** 2).sum(1) + R[:, up].sum(1)) / nu_)
    for k, vv in d.items():
        out[f"ryk_M{Ms}_{k}"] = vv
    print(Ms, "DSMC centre", x[ic], "Rykov centre", p["x"][icr], "reduced n vs profile n max diff", np.abs(nr - p["rho"]).max())
np.savez_compressed(outf, **out)
