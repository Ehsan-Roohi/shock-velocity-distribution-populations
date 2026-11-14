"""One summary record per DSMC case (all folders), for the tables of the JFM manuscript.

Same definitions as in dsmc_analysis.py: maximum-slope thickness, peak anisotropy, largest residual of the
best non-negative combination of the two Rankine-Hugoniot Maxwellians and of the K = 2, 3 population fits
over 0.02 < n* < 0.98, largest errors of the K = 3 heat flux and fourth-order moment relative to their
peaks, and the upstream-moving mass fraction at n* = 0.05 and 0.2.

Usage: python dsmc_table.py <shock_data dir>     -> analysis/out/dsmc_table.json
"""
import sys, os, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl


def smooth(y, n=3):
    return np.convolve(y, np.ones(n) / n, mode="same")


def record(root, path):
    c = sl.load_case(path); z = np.load(path)
    d = {k[7:]: z[k] for k in z.files if k.startswith("direct_")}
    st = c["st"]; x = c["x"]
    c["nstar"] = (d["rho"] - 1) / (st["rho2"] - 1)
    delta = (st["rho2"] - 1) / np.abs(np.gradient(smooth(d["rho"], 9), x))[10:-10].max()
    neg = c["vx"] < 0; beta = (c["G"][:, neg] @ c["wx"][neg]) / c["rho"]
    core0 = np.where((c["nstar"] > 0.005) & (c["nstar"] < 0.995))[0]; k = np.arange(core0.min() - 2, core0.max() + 3)
    ns = np.maximum.accumulate(c["nstar"][k])
    eps = sl.two_maxwellian_residual(c)
    core = (c["nstar"] > 0.02) & (c["nstar"] < 0.98)
    m = re.match(r"DSMC_w(.+)_M(.+)", c["name"])
    rec = dict(omega=float(m.group(1).replace("p", ".")), mach=float(m.group(2).replace("p", ".")), delta=float(delta),
               TparTperp_max=float(smooth(d["Tpar"] / d["Tperp"], 5).max()),
               Tpar_over_T2_max=float(d["Tpar"][k].max() / st["T2"]),
               Tpar_over_T2_max_exact=float((1 + st["u1"] ** 2) ** 2 / (4 * st["u1"] ** 2) / st["T2"]),
               rh_pair_max=float(100 * eps[core].max()), rh_pair_n=float(c["nstar"][core][np.argmax(eps[core])]),
               beta_n05=float(np.interp(0.05, ns, beta[k])), beta_n20=float(np.interp(0.2, ns, beta[k])),
               up_T=float(d["T"][20:60].mean()) if "T" in d else None)
    fp = os.path.join(root, "analysis", "out", "popfits", f"{c['name']}_popfit.npz")
    if os.path.exists(fp):
        f = np.load(fp); kk = (f["nstar"] > 0.02) & (f["nstar"] < 0.98)
        for K in (2, 3):
            rec[f"K{K}_max"] = float(100 * np.nanmax(f[f"res{K}"][kk, 0]))
            for mm in ("q", "sigma", "M400"):
                rec[f"K{K}_{mm}"] = float(100 * np.nanmax(np.abs(f[f"{mm}_fit{K}"][kk] - f[mm][kk])) / float(f[mm + "max"]))
    return c["name"], rec


def main(root):
    out = {}
    for folder in ("vdf_dsmc", "vdf_m25", "vdf_dsmc_exp"):
        for p in sorted(glob.glob(os.path.join(root, folder, "DSMC_w*_M*_reduced_vdf.npz"))):
            name = os.path.basename(p).replace("_reduced_vdf.npz", "")
            if name in out:
                continue
            n, r = record(root, p); r["folder"] = folder; out[n] = r
            print(n, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
    json.dump(out, open(os.path.join(root, "analysis", "out", "dsmc_table.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
