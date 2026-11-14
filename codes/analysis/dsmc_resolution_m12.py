"""Resolution checks of the DSMC shocks at Mach 12 and Mach 25.

Baseline: dx = 0.1, 200 molecules per upstream cell, dt = 0.002 (Mach 12) or 0.001 (Mach 25) (vdf_dsmc/).
Variants (vdf_dsmc_conv/<name>/, 4 seeds each; <M> = 12 or 25):
  A<M>  hard sphere, dx/2 and dt/2          B<M>  hard sphere, 400 molecules per cell
  D<M>  omega = 1, dx/2 and dt/2            E<M>  omega = 1, 400 molecules per cell

Same measures as dsmc_resolution.py (Mach 8), plus the precursor: the distances between the planes where the mass
fraction of upstream-moving molecules is 1e-2, 1e-3 and 1e-4, which set the decay rate discussed in the paper.

Usage: python dsmc_resolution_m12.py <shock_data dir> [Mach number, default 12]
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsmc_resolution import prep, LEVELS

BETAS = (1e-2, 1e-3, 1e-4)


def precursor(c):
    """x (relative to the shock centre) where beta falls to each level, searched upstream of the centre."""
    v, w = c["vx"], c["wx"]
    beta = (c["G"] * w)[:, v < 0].sum(1) / np.maximum((c["G"] * w).sum(1), 1e-300)
    xs = c["xs"]; k = np.where(xs < 0)[0]; out = {}
    lb = np.log(np.maximum(beta[k], 1e-300))
    for b in BETAS:
        j = np.where(lb >= np.log(b))[0]
        j = j.min() if len(j) else None                       # first cell (from upstream) at or above the level
        if j is None or j == 0:
            out[b] = float("nan"); continue
        a = (np.log(b) - lb[j - 1]) / (lb[j] - lb[j - 1])
        out[b] = float(xs[k][j - 1] + a * (xs[k][j] - xs[k][j - 1]))
    return out


def main(root, M="12"):
    base = {0.5: prep(os.path.join(root, "vdf_dsmc", f"DSMC_w0p5_M{M}_reduced_vdf.npz")),
            1.0: prep(os.path.join(root, "vdf_dsmc", f"DSMC_w1_M{M}_reduced_vdf.npz"))}
    pb = {om: precursor(b) for om, b in base.items()}
    out = {}
    for name, om, what in (("A" + M, 0.5, "hard sphere, dx/2 and dt/2"), ("B" + M, 0.5, "hard sphere, 400 molecules per cell"),
                           ("D" + M, 1.0, "omega=1, dx/2 and dt/2"), ("E" + M, 1.0, "omega=1, 400 molecules per cell")):
        tag = f"DSMC_w0p5_M{M}" if om == 0.5 else f"DSMC_w1_M{M}"
        f = os.path.join(root, "vdf_dsmc_conv", name, tag + "_reduced_vdf.npz")
        if not os.path.exists(f):
            print(name, "missing", f); continue
        c = prep(f); b = base[om]; pc = precursor(c)
        r = dict(variant=what, width=c["width"], width_base=b["width"], width_change_pct=100 * (c["width"] / b["width"] - 1),
                 delta=c["delta"], delta_base=b["delta"], delta_change_pct=100 * (c["delta"] / b["delta"] - 1),
                 nstar_max_abs=float(np.max(np.abs(c["prof"]["ns"] - b["prof"]["ns"]))),
                 T_max_rel_pct=float(100 * np.max(np.abs(c["prof"]["T"] / b["prof"]["T"] - 1))),
                 ratio_peak=float(c["prof"]["ratio"].max()), ratio_peak_base=float(b["prof"]["ratio"].max()),
                 q_L2_pct=float(100 * np.linalg.norm(c["prof"]["q"] - b["prof"]["q"]) / np.linalg.norm(b["prof"]["q"])),
                 sigma_L2_pct=float(100 * np.linalg.norm(c["prof"]["sigma"] - b["prof"]["sigma"]) / np.linalg.norm(b["prof"]["sigma"])))
        r["ratio_peak_change_pct"] = 100 * (r["ratio_peak"] / r["ratio_peak_base"] - 1)
        w = b["wx"]
        for lv in LEVELS:
            d = c["g"][lv] - b["g"][lv]; nrm = np.sqrt((b["g"][lv] ** 2 * w).sum())
            r[f"g_L2_pct_n{lv}"] = float(100 * np.sqrt((d ** 2 * w).sum()) / nrm)
            r[f"g_noise_pct_n{lv}"] = float(100 * np.sqrt(((c["gvar"][lv] + b["gvar"][lv]) * w).sum()) / nrm)
        # precursor: lengths over which beta falls by one and two decades
        for lo in (1e-3, 1e-4):
            Lc, Lb = pc[1e-2] - pc[lo], pb[om][1e-2] - pb[om][lo]
            r[f"precursor_length_1e-2_to_{lo:g}"] = Lc; r[f"precursor_length_base_1e-2_to_{lo:g}"] = Lb
            r[f"precursor_length_change_pct_{lo:g}"] = 100 * (Lc / Lb - 1)
        out[name] = r
        print(f"{name} ({what})")
        print(f"   width x(0.8)-x(0.2): {r['width']:.3f} vs {r['width_base']:.3f} ({r['width_change_pct']:+.2f}%)   max-slope thickness {r['delta']:.3f} vs {r['delta_base']:.3f} ({r['delta_change_pct']:+.2f}%)")
        print(f"   profiles: n* max abs {r['nstar_max_abs']:.4f}  T max {r['T_max_rel_pct']:.2f}%  q L2 {r['q_L2_pct']:.2f}%  sigma L2 {r['sigma_L2_pct']:.2f}%  Tpar/Tperp peak {r['ratio_peak']:.3f} vs {r['ratio_peak_base']:.3f} ({r['ratio_peak_change_pct']:+.2f}%)")
        print("   g difference (expected from scatter): " + "  ".join(f"n*={lv}: {r[f'g_L2_pct_n{lv}']:.2f}% ({r[f'g_noise_pct_n{lv}']:.2f}%)" for lv in LEVELS))
        print("   precursor length beta 1e-2 -> 1e-3: %.3f vs %.3f (%+.2f%%)   1e-2 -> 1e-4: %.3f vs %.3f (%+.2f%%)" % (
            r["precursor_length_1e-2_to_0.001"], r["precursor_length_base_1e-2_to_0.001"], r["precursor_length_change_pct_0.001"],
            r["precursor_length_1e-2_to_0.0001"], r["precursor_length_base_1e-2_to_0.0001"], r["precursor_length_change_pct_0.0001"]), flush=True)
    json.dump(out, open(os.path.join(root, "analysis", "out", f"dsmc_resolution_m{M}.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "12")
