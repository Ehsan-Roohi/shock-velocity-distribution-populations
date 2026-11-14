"""Resolution tests of the reduced BGK solver: each variant (coarser cells, fewer or more velocities) against the
solution used. Usage: python bgk_resolution.py <shock_data dir>   ->  analysis/out/bgk_resolution.json"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predictor_clean as PC

root = sys.argv[1]; out = {}
for new, old in (("M8r_x2", "M8r"), ("M8r_nv161", "M8r"), ("M8r_nv641", "M8r"), ("M12r_x2", "M12r"), ("M8r_cont", "M8r")):     # M8r_cont: 150 more relaxation times
    if not os.path.exists(os.path.join(root, "vdf_reduced", f"{new}_BGK_reduced_vdf.npz")):
        continue
    out[new] = r = PC.compare(root, new, old)
    print(f"{new} vs {old}: flux dev % {np.round(r['new']['flux_dev'], 3).tolist()} (ref {np.round(r['old']['flux_dev'], 3).tolist()}); delta {r['new']['delta']:.4f} vs {r['old']['delta']:.4f}; peak aniso {r['new']['aniso_peak']:.4f} vs {r['old']['aniso_peak']:.4f}")
    print("   g diff %", np.round(r["g_diff"], 3).tolist(), " h diff %", np.round(r["h_diff"], 3).tolist())
    print("   profiles %: " + "  ".join(f"{k} {r[k + '_L2_diff']:.3f}" for k in ("q", "sigma", "T", "Tpar", "Tperp")))
json.dump(out, open(os.path.join(root, "analysis", "out", "bgk_resolution.json"), "w"), indent=1)
