"""Hard-sphere DSMC profiles against the direct Boltzmann solution of Ohwada, Phys. Fluids A 5, 217 (1993).

Ohwada's Tables III and V give density and temperature for hard spheres at Mach 2 and 3 on the
coordinate x1 = X / ((sqrt(pi)/2) l0), where l0 = 1/(sqrt(2) pi d^2 n0) is the upstream mean free path.
With the first-approximation hard-sphere viscosity, l0 = 16/(5 sqrt(2 pi)) ell_1 = 1.2766 ell_1, so
X/ell_1 = 1.1313 x1. The two solutions are aligned at the position of n* = 0.5.

Usage: python ohwada_compare.py <shock_data dir>
"""
import sys, os, json, math
import numpy as np

X1 = np.array([-17.446, -12.688, -8.833, -6.087, -4.321, -3.241, -2.551, -2.045, -1.611, -1.203, -0.800, -0.400, 0.000,
               0.400, 0.800, 1.203, 1.611, 2.045, 2.551, 3.241, 4.321, 6.087, 8.833, 12.688, 17.446])
RHO = {2: [1.000, 1.000, 1.001, 1.007, 1.034, 1.088, 1.155, 1.230, 1.314, 1.410, 1.518, 1.633, 1.749, 1.857, 1.952, 2.032, 2.098,
           2.151, 2.195, 2.234, 2.265, 2.281, 2.286, 2.287, 2.287],
       3: [1.000, 1.000, 1.000, 1.003, 1.020, 1.063, 1.131, 1.223, 1.345, 1.507, 1.715, 1.957, 2.207, 2.435, 2.618, 2.752, 2.845,
           2.907, 2.949, 2.978, 2.994, 2.999, 3.000, 3.001, 3.001]}
TEMP = {2: [1.000, 1.000, 1.001, 1.021, 1.098, 1.232, 1.371, 1.496, 1.609, 1.712, 1.801, 1.875, 1.933, 1.976, 2.007, 2.029, 2.045,
            2.056, 2.065, 2.071, 2.075, 2.077, 2.077, 2.077, 2.077],
        3: [1.000, 1.000, 1.001, 1.024, 1.146, 1.423, 1.789, 2.174, 2.554, 2.904, 3.189, 3.393, 3.520, 3.592, 3.631, 3.651, 3.661,
            3.666, 3.668, 3.668, 3.667, 3.666, 3.666, 3.666, 3.666]}
SCALE = (math.sqrt(math.pi) / 2) * 16 / (5 * math.sqrt(2 * math.pi))      # X/ell_1 per unit x1


def main(root):
    out = {}
    for M in (2, 3):
        z = np.load(os.path.join(root, "vdf_dsmc", f"DSMC_w0p5_M{M}_reduced_vdf.npz"))
        st = json.loads(str(z["states"])); x = z["x_mfp"]; rho = z["direct_rho"]; T = z["direct_T"]
        ns = (rho - 1) / (st["rho2"] - 1)
        k = np.where((ns > 0.02) & (ns < 0.98))[0]; k = np.arange(k.min(), k.max() + 1)
        xc = float(np.interp(0.5, ns[k], x[k]))
        ro, To = np.array(RHO[M]), np.array(TEMP[M]); nso = (ro - 1) / (st["rho2"] - 1)
        ko = np.where((nso > 0.02) & (nso < 0.98))[0]; ko = np.arange(ko.min() - 1, ko.max() + 2)
        xo = SCALE * (X1 - float(np.interp(0.5, nso[ko], X1[ko])))
        rd = np.interp(xo, x - xc, rho); Td = np.interp(xo, x - xc, T)
        tn = lambda t: (t - 1) / (st["T2"] - 1)
        xf = np.arange(-12, 8, 0.25)
        thick = lambda xx, nn: float(1 / np.max(np.gradient(np.interp(xf, xx, nn), xf)))
        w80 = lambda xx, nn, kk: float(np.interp(0.8, nn[kk], xx[kk]) - np.interp(0.2, nn[kk], xx[kk]))
        r = dict(max_abs_nstar=float(np.max(np.abs((rd - ro) / (st["rho2"] - 1)))),
                 max_abs_Tnorm=float(np.max(np.abs(tn(Td) - tn(To)))),
                 max_rel_rho_pct=float(100 * np.max(np.abs(rd / ro - 1))), max_rel_T_pct=float(100 * np.max(np.abs(Td / To - 1))),
                 width_dsmc=w80(x - xc, ns, k), width_ohwada=w80(xo, nso, ko),
                 Tmax_over_T2_dsmc=float(T.max() / st["T2"]), Tmax_over_T2_ohwada=float(To.max() / To[-1]))
        r["width_ratio"] = r["width_dsmc"] / r["width_ohwada"]
        out[f"M{M}"] = r
        print(f"M={M}: max |dn*| {r['max_abs_nstar']:.4f}  max |dT_norm| {r['max_abs_Tnorm']:.4f}  max rel rho {r['max_rel_rho_pct']:.2f}%  "
              f"max rel T {r['max_rel_T_pct']:.2f}%  width x(0.8)-x(0.2): DSMC {r['width_dsmc']:.3f}, Ohwada {r['width_ohwada']:.3f} (ratio {r['width_ratio']:.3f})")
        for a, b, c_, d_ in zip(xo[3:22], ro[3:22], rd[3:22], Td[3:22] - To[3:22]):
            print(f"     x/ell1 {a:+7.3f}  rho Ohwada {b:.3f}  DSMC {c_:.3f}  dT {d_:+.3f}")
    json.dump(out, open(os.path.join(root, "analysis", "out", "ohwada_compare.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
