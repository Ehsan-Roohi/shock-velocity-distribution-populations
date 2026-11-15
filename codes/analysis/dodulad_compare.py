"""Hard-sphere DSMC shock profiles against the Boltzmann-equation solution of Dodulad & Tcheremissine,
Comput. Math. Math. Phys. 53 (2013) 827 (Russian original: Zh. Vychisl. Mat. Mat. Fiz. 53(6), 1008-1026), table on
p. 1020: density n, temperature T and longitudinal temperature T_xx in units of the upstream values, against x in
upstream mean free paths lambda_1 = 1/(sqrt(2) pi d^2 n_1), x = 0 where the normalised density is 0.5.
Their stated accuracy: 1 % for local values, 3 % for the shock width.

Our lengths are in l_1 = mu_1 sqrt(R T_1)/p_1; for hard spheres lambda_1 = 16/(5 sqrt(2 pi)) l_1 = 1.2766 l_1.

Usage: python dodulad_compare.py <shock_data dir>
Writes analysis/out/dodulad_compare.json and analysis/out/dodulad_compare.{png,pdf}."""
import sys, os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# x/lambda_1, n, T, Txx  (Mach 3 and Mach 8 columns of the table)
M3 = """-10.0 1.000 1.000 1.000; -9.2 1.000 1.000 1.000; -8.4 1.000 1.000 1.000; -7.6 1.000 1.003 1.003; -6.8 1.000 1.003 1.008;
-6.2 1.000 1.008 1.013; -5.6 1.002 1.013 1.027; -5.0 1.004 1.029 1.053; -4.4 1.008 1.056 1.104; -3.8 1.016 1.109 1.205;
-3.2 1.030 1.216 1.405; -2.8 1.050 1.339 1.627; -2.4 1.080 1.517 1.952; -2.0 1.128 1.771 2.397; -1.6 1.204 2.104 2.947;
-1.2 1.322 2.493 3.523; -0.8 1.496 2.880 3.992; -0.4 1.726 3.200 4.235; 0.0 2.000 3.419 4.251; 0.4 2.278 3.547 4.133;
0.8 2.516 3.613 3.989; 1.2 2.696 3.645 3.872; 1.6 2.816 3.659 3.792; 2.0 2.892 3.667 3.739; 2.4 2.936 3.667 3.709;
2.8 2.962 3.669 3.691; 3.2 2.978 3.669 3.680; 3.8 2.990 3.667 3.672; 4.4 2.996 3.667 3.669; 5.0 2.998 3.667 3.669;
5.6 3.000 3.667 3.667; 6.2 3.000 3.667 3.667; 6.8 3.000 3.667 3.667; 7.6 3.000 3.667 3.667; 8.4 3.000 3.667 3.667;
9.2 3.000 3.667 3.667; 10.0 3.000 3.667 3.667"""
M8 = """-10.0 1.000 1.000 1.000; -9.2 1.000 1.000 1.000; -8.4 1.000 1.000 1.000; -7.6 1.000 1.000 1.020; -6.8 1.000 1.020 1.040;
-6.2 1.000 1.040 1.079; -5.6 1.000 1.079 1.139; -5.0 1.003 1.159 1.298; -4.4 1.006 1.318 1.596; -3.8 1.011 1.676 2.252;
-3.2 1.025 2.371 3.563; -2.8 1.042 3.226 5.153; -2.4 1.071 4.557 7.598; -2.0 1.121 6.564 11.174; -1.6 1.206 9.386 15.924;
-1.2 1.350 12.824 21.210; -0.8 1.587 16.301 25.443; -0.4 1.945 18.964 27.112; 0.0 2.410 20.415 26.277; 0.4 2.896 20.932 24.449;
0.8 3.288 21.051 22.879; 1.2 3.542 21.031 21.905; 1.6 3.683 20.971 21.369; 2.0 3.753 20.932 21.110; 2.4 3.787 20.912 20.991;
2.8 3.804 20.892 20.932; 3.2 3.813 20.892 20.892; 3.8 3.818 20.872 20.872; 4.4 3.818 20.872 20.872; 5.0 3.821 20.872 20.872;
5.6 3.821 20.872 20.872; 6.2 3.821 20.872 20.872; 6.8 3.821 20.872 20.872; 7.6 3.821 20.872 20.872; 8.4 3.821 20.872 20.872;
9.2 3.821 20.872 20.872; 10.0 3.821 20.872 20.872"""
TAB = {3: np.array([[float(v) for v in r.split()] for r in M3.replace("\n", " ").split(";")]),
       8: np.array([[float(v) for v in r.split()] for r in M8.replace("\n", " ").split(";")])}
LAM = 16 / (5 * math.sqrt(2 * math.pi))          # lambda_1 / l_1 for hard spheres

root = sys.argv[1]; out = os.path.join(root, "analysis", "out"); os.makedirs(out, exist_ok=True)
plt.rcParams.update({"font.size": 15, "axes.labelsize": 17, "axes.titlesize": 15, "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
                     "font.family": "serif", "mathtext.fontset": "cm", "axes.linewidth": 0.8})
fig, ax = plt.subplots(2, 3, figsize=(13, 8.0)); S = {}
for row, M in enumerate((3, 8)):
    z = np.load(os.path.join(root, "vdf_dsmc", f"DSMC_w0p5_M{M}_reduced_vdf.npz")); st = json.loads(str(z["states"]))
    x = z["x_mfp"]; n, T, Tp = z["direct_rho"], z["direct_T"], z["direct_Tpar"]; n2, T2 = st["rho2"], st["T2"]
    xc = float(np.interp(0.5, (n - 1) / (n2 - 1), x)); xl = (x - xc) / LAM
    k = np.ones(3) / 3; sm = lambda q: np.convolve(q, k, "same")               # 0.3 l_1 running mean of the cell values
    t = TAB[M]; xt = t[:, 0]; rec = {}
    for j, (name, q, ref, jump) in enumerate((("n", n, t[:, 1], n2 - 1), ("T", T, t[:, 2], T2 - 1), ("Txx", Tp, t[:, 3], T2 - 1))):
        qi = np.interp(xt, xl, sm(q)); dif = qi - ref
        rec[name] = dict(max_abs=float(np.abs(dif).max()), max_rel_local=float(np.abs(dif / ref).max()), max_over_jump=float(np.abs(dif).max() / jump),
                         at_x=float(xt[np.argmax(np.abs(dif))]))
        a = ax[row, j]; m = (xl > -9) & (xl < 7)
        a.plot(xl[m], q[m], "-", color="#1F4E79", lw=1.8, label="DSMC (this work)")
        a.plot(xt, ref, "o", color="#C0392B", ms=6, mfc="none", mew=1.3, label="Boltzmann equation (Dodulad & Tcheremissine 2013)")
        a.set_xlim(-9, 7); a.set_ylabel({"n": r"$n/n_1$", "T": r"$T/T_1$", "Txx": r"$T_{xx}/T_1$"}[name])
        if row == 1:
            a.set_xlabel(r"$x/\lambda_1$")
        a.set_title(f"Mach {M}", loc="left")
    # width: maximum slope of the density from each source
    wd = (n2 - 1) / np.max(np.gradient(sm(n), xl)[(xl > -9) & (xl < 7)])
    xf = np.linspace(-6, 6, 2401); cs = np.interp(xf, xt, t[:, 1])
    from scipy.interpolate import CubicSpline
    sp = CubicSpline(xt, t[:, 1]); wt = (n2 - 1) / np.max(sp(xf, 1))
    rec["inverse_width_dsmc"] = float(1 / wd); rec["inverse_width_table"] = float(1 / wt)
    rec["peak_Txx_dsmc"] = float(Tp.max()); rec["peak_Txx_table"] = float(t[:, 3].max())
    rec["peak_T_over_T2_dsmc"] = float(sm(T).max() / T2); rec["peak_T_over_T2_table"] = float(t[:, 2].max() / T2)
    S[f"M{M}"] = rec
    print(f"Mach {M}: largest difference DSMC - table: " + "; ".join(f"{k} {v['max_abs']:.3f} ({100 * v['max_rel_local']:.2f}% of the local value, {100 * v['max_over_jump']:.2f}% of the jump, at x={v['at_x']:.1f})" for k, v in rec.items() if isinstance(v, dict)))
    print(f"   inverse width lambda_1/delta: DSMC {1 / wd:.4f}, table (spline) {1 / wt:.4f}, difference {100 * (wt / wd - 1):+.1f}%;  peak Txx/T1: DSMC {Tp.max():.2f}, table {t[:, 3].max():.2f};"
          f"  peak T/T2: DSMC {sm(T).max() / T2:.4f}, table {t[:, 2].max() / T2:.4f}")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(out, "dodulad_compare.png"), dpi=200); fig.savefig(os.path.join(out, "dodulad_compare.pdf"))
json.dump(S, open(os.path.join(out, "dodulad_compare.json"), "w"), indent=1)
