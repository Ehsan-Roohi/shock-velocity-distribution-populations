"""Source tables for the linear recoil cascade (dsmc/cascade_linear.cpp) taken from the DSMC shocks.

For every case the flux of upstream-moving molecules through the plane x_ref, where the backward mass
fraction beta equals BETA_REF, is written as rows  vx  flux  Tperp  (flux = |vx| g dv, Tperp = h / 2g).
The file name carries the case; cascade/sources/index.json lists x_ref and the distance to the inflow side.

Usage: python cascade_sources.py <shock_data dir>
"""
import sys, os, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl

BETA_REF = 0.01


def main(root):
    out = os.path.join(root, "cascade", "sources"); os.makedirs(out, exist_ok=True)
    index = {}
    files = sorted(glob.glob(os.path.join(root, "vdf_dsmc", "DSMC_w*_M*_reduced_vdf.npz")) + glob.glob(os.path.join(root, "vdf_m25", "DSMC_w*_M25_reduced_vdf.npz")))
    seen = set()
    for f in files:
        name = os.path.basename(f).replace("_reduced_vdf.npz", "")
        m = re.match(r"DSMC_w(.+)_M(.+)", name); om = float(m.group(1).replace("p", ".")); M = float(m.group(2).replace("p", "."))
        if M < 8 or name in seen:
            continue
        seen.add(name)
        c = sl.load_case(f); v, w = c["vx"], c["wx"]; neg = v < 0
        beta = (c["G"][:, neg] @ w[neg]) / c["rho"]
        up = np.where(c["x"] < c["x"][np.argmin(np.abs(c["nstar"] - 0.5))])[0]
        i = up[np.argmin(np.abs(np.log(np.maximum(beta[up], 1e-30)) - np.log(BETA_REF)))]
        k = np.arange(i - 2, i + 3)                         # five cells (0.5 l1) to reduce noise
        g = c["G"][k].mean(0)[neg]; h = c["H"][k].mean(0)[neg]
        ok = g > 0
        tab = np.stack([v[neg][ok], -v[neg][ok] * g[ok] * w[neg][ok], h[ok] / (2 * g[ok])], 1)
        np.savetxt(os.path.join(out, f"src_M{M:g}_w{om:g}.txt"), tab, fmt="%.8e")
        index[f"M{M:g}_w{om:g}"] = dict(x_ref=float(c["x"][i]), beta_ref=float(beta[i]), nstar_ref=float(c["nstar"][i]), rho_ref=float(c["rho"][i]),
                                         L=float(c["x"][i] - c["x"][0] - 5.0), backward_flux=float(tab[:, 1].sum()), u1=c["st"]["u1"])
        print(name, index[f"M{M:g}_w{om:g}"])
    json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
