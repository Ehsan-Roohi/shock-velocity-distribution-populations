"""Resolution checks of the DSMC shocks at Mach 8.

Baseline: dx = 0.1, dt = 0.0035, 200 molecules per upstream cell, 8 seeds (vdf_dsmc/).
Variants (vdf_dsmc_conv/<A..D>/, 4 seeds each):
  A  hard sphere, dx/2 and dt/2      B  hard sphere, 50 molecules per cell
  C  hard sphere, dt/2               D  omega = 1, dx/2 and dt/2

For every variant the differences from the baseline are listed together with the level expected
from statistical scatter alone (Poisson counts for the distributions).

Usage: python dsmc_resolution.py <shock_data dir>
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl

XG = np.arange(-12.0, 8.0001, 0.1)
LEVELS = (0.1, 0.5, 0.9)


def prep(path):
    c = sl.load_case(path)
    z = np.load(path)
    for k in ("rho", "u", "T", "Tpar", "Tperp", "q", "sigma"):
        c["d_" + k] = z["direct_" + k]
    c["cs"] = float(z["count_scale"])
    ns = (c["d_rho"] - 1) / (c["st"]["rho2"] - 1)
    k = np.where((ns > 0.02) & (ns < 0.98))[0]
    k = np.arange(k.min(), k.max() + 1)
    xc = float(np.interp(0.5, ns[k], c["x"][k]))
    c["xs"] = c["x"] - xc; c["ns"] = ns; c["xc"] = xc
    c["prof"] = {key: np.interp(XG, c["xs"], val) for key, val in
                 (("ns", ns), ("T", c["d_T"]), ("ratio", c["d_Tpar"] / c["d_Tperp"]), ("q", c["d_q"]), ("sigma", c["d_sigma"]))}
    c["width"] = float(np.interp(0.8, ns[k], c["x"][k]) - np.interp(0.2, ns[k], c["x"][k]))
    xf = np.arange(-12, 8, 0.5)
    c["delta"] = float(1 / np.max(np.gradient(np.interp(xf, c["xs"], ns), xf)))
    c["g"] = {}; c["gvar"] = {}; c["beta"] = {}
    v, w = c["vx"], c["wx"]
    for lv in LEVELS:
        j = k[np.argmax(ns[k] >= lv)]                     # first cell at or beyond the level
        a = (lv - ns[j - 1]) / (ns[j] - ns[j - 1])        # linear interpolation in n* between two cells
        g = (1 - a) * c["G"][j - 1] + a * c["G"][j]
        c["g"][lv] = g
        c["gvar"][lv] = ((1 - a) ** 2 * c["G"][j - 1] + a ** 2 * c["G"][j]) / c["cs"]   # Poisson variance
        c["beta"][lv] = float((g * w)[v < 0].sum() / (g * w).sum())
    return c


def main(root):
    base = {0.5: prep(os.path.join(root, "vdf_dsmc", "DSMC_w0p5_M8_reduced_vdf.npz")),
            1.0: prep(os.path.join(root, "vdf_dsmc", "DSMC_w1_M8_reduced_vdf.npz"))}
    out = {}
    for name, om, what in (("A", 0.5, "hard sphere, dx/2 and dt/2"), ("B", 0.5, "hard sphere, 50 molecules per cell"),
                           ("C", 0.5, "hard sphere, dt/2"), ("D", 1.0, "omega=1, dx/2 and dt/2")):
        tag = "DSMC_w0p5_M8" if om == 0.5 else "DSMC_w1_M8"
        c = prep(os.path.join(root, "vdf_dsmc_conv", name, tag + "_reduced_vdf.npz")); b = base[om]
        r = dict(variant=what,
                 width=c["width"], width_base=b["width"], width_change_pct=100 * (c["width"] / b["width"] - 1),
                 delta=c["delta"], delta_base=b["delta"],
                 nstar_max_abs=float(np.max(np.abs(c["prof"]["ns"] - b["prof"]["ns"]))),
                 T_max_rel_pct=float(100 * np.max(np.abs(c["prof"]["T"] / b["prof"]["T"] - 1))),
                 ratio_peak=float(c["prof"]["ratio"].max()), ratio_peak_base=float(b["prof"]["ratio"].max()),
                 q_L2_pct=float(100 * np.linalg.norm(c["prof"]["q"] - b["prof"]["q"]) / np.linalg.norm(b["prof"]["q"])),
                 q_min_change_pct=float(100 * (c["prof"]["q"].min() / b["prof"]["q"].min() - 1)),
                 sigma_L2_pct=float(100 * np.linalg.norm(c["prof"]["sigma"] - b["prof"]["sigma"]) / np.linalg.norm(b["prof"]["sigma"])),
                 sigma_max_change_pct=float(100 * (c["prof"]["sigma"].max() / b["prof"]["sigma"].max() - 1)))
        w = b["wx"]
        for lv in LEVELS:
            d = c["g"][lv] - b["g"][lv]
            nrm = np.sqrt((b["g"][lv] ** 2 * w).sum())
            r[f"g_L2_pct_n{lv}"] = float(100 * np.sqrt((d ** 2 * w).sum()) / nrm)
            r[f"g_noise_pct_n{lv}"] = float(100 * np.sqrt(((c["gvar"][lv] + b["gvar"][lv]) * w).sum()) / nrm)
            r[f"beta_n{lv}"] = c["beta"][lv]; r[f"beta_base_n{lv}"] = b["beta"][lv]
        out[name] = r
        print(f"{name} ({what})")
        print(f"   width x(0.8)-x(0.2): {r['width']:.3f} vs {r['width_base']:.3f} ({r['width_change_pct']:+.2f}%)   max-slope thickness {r['delta']:.3f} vs {r['delta_base']:.3f}")
        print(f"   profiles vs baseline: n* max abs {r['nstar_max_abs']:.4f}  T max {r['T_max_rel_pct']:.2f}%  q L2 {r['q_L2_pct']:.2f}% (min {r['q_min_change_pct']:+.2f}%)  "
              f"sigma L2 {r['sigma_L2_pct']:.2f}% (max {r['sigma_max_change_pct']:+.2f}%)  Tpar/Tperp peak {r['ratio_peak']:.3f} vs {r['ratio_peak_base']:.3f}")
        print("   g difference (expected from scatter): " + "  ".join(f"n*={lv}: {r[f'g_L2_pct_n{lv}']:.2f}% ({r[f'g_noise_pct_n{lv}']:.2f}%)" for lv in LEVELS))
        print("   backward mass fraction: " + "  ".join(f"n*={lv}: {r[f'beta_n{lv}']:.5f} vs {r[f'beta_base_n{lv}']:.5f}" for lv in LEVELS), flush=True)
    json.dump(out, open(os.path.join(root, "analysis", "out", "dsmc_resolution_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
