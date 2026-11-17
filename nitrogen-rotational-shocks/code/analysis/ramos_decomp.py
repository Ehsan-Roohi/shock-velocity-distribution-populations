"""Three fixed shapes against the Raman level populations of Ramos et al. (Phys. Rev. E 62, 4940, 2000), shock wave B
(Mach 10.6), spectrum (b) recorded inside the shock (digitised Boltzmann plot, ramos_2000.py).

Populations per rotational state X_J (the nuclear-spin weights are already divided out in the published plot) are
turned into p_J proportional to (2J+1) X_J and written as a non-negative sum of
  S0  the free stream: Boltzmann at the measured upstream rotational temperature (13.9 K, plot a),
  S1  the distribution after one gas-kinetic collision from the classical trajectories, in units of k T2,
  S2  Boltzmann at the downstream temperature T2.
Levels above the last measured one follow the measured asymptote of the upper levels (fitted here to J >= 10).
Printed next to the fractions the authors obtained from their own decomposition of the spectrum (cold, scattered, warm).

Usage: python ramos_decomp.py <exc npz of the one-collision shape> [T2=262] [jmax=40] [dropJ0=0]"""
import sys, math
import numpy as np
from scipy.optimize import nnls
import ramos_2000 as R

opt = dict(a.split("=") for a in sys.argv[2:]); T2 = float(opt.get("T2", 262.0)); JMAX = int(opt.get("jmax", 40)); drop0 = int(opt.get("dropJ0", 0))
TH = R.BETA_CM_K * R.B_ROT_CM1; J = np.arange(JMAX + 1); G = 2 * J + 1; EJ = TH * J * (J + 1)
Jb, xb, yb = R.fig4_series("b"); hi = Jb >= 10; sl, ic = np.polyfit(xb[hi], yb[hi], 1); Tw = -TH / sl
lnX = np.array([yb[list(Jb).index(j)] if j in Jb else ic + sl * j * (j + 1) for j in J]); p = G * np.exp(lnX - lnX.max()); p /= p.sum()
print(f"spectrum (b), z/D = {R.FIG4_POSITIONS['b']['z_over_D']}: {len(Jb)} measured levels (J = {Jb.min()}..{Jb.max()}); slope temperature of J >= 10: {Tw:.0f} K "
      f"(paper: {R.FIG4_B_ASYMPTOTES_K['warm']:.0f} K); mean rotational energy {np.sum(p * EJ):.0f} K (paper: 130 +- 10 K); share beyond the measured levels {p[J > Jb.max()].sum():.3f}")
S0 = G * np.exp(-EJ / 13.9); S0 /= S0.sum()
d = np.load(sys.argv[1]); E = d["E"] * T2; sv = float(d["sig_vhs"]); sm = float(d["sigmax"]); edge = TH * np.arange(JMAX + 2) ** 2
f1 = (d["ctc"] - d["before"] * (1 - sv / sm)) / sv; c = np.interp(edge, E, f1); S1 = np.maximum(c[:-1] - c[1:], 0) / c[0]
S2 = G * np.exp(-EJ / T2); S2 /= S2.sum(); m = np.ones(len(J), bool); m[0] = not drop0
for name, B in (("free stream + thermal", np.stack([S0, S2], 1)), ("free stream + one collision + thermal", np.stack([S0, S1, S2], 1))):
    w, _ = nnls(B[m], p[m]); r = np.linalg.norm(p[m] - B[m] @ w) / np.linalg.norm(p[m]); meas = m & (J <= Jb.max()); ratio = (B @ w)[meas] / p[meas]
    print(f"   {name:40s}: weights " + " ".join(f"{v:.3f}" for v in w) + f" (sum {w.sum():.3f}); residual {r:.3f}; fitted/measured at the measured levels {ratio.min():.2f} to {ratio.max():.2f}")
F = R.FIG5_FRACTIONS; print(f"   fractions of the authors (their Fig. 5): cold {F['x_c']}, scattered {F['x_s']}, warm {F['x_w']}")
z, n, zt, T = R.fig3("B"); nb = float(np.interp(18.9, z[(z > 17.5) & (z < 20)], n[(z > 17.5) & (z < 20)])); n2rh = 2.4 * 10.6 ** 2 / (0.4 * 10.6 ** 2 + 2)
n2m = float(np.mean(n[(z > 20.1) & (z < 21.6)])); print(f"   density at z/D = 18.9: n/n1 = {nb:.2f}; normalised density {(nb - 1) / (n2m - 1):.2f} with the measured post-shock density ({n2m:.2f} n1), "
                                                         f"{(nb - 1) / (n2rh - 1):.2f} with the Rankine-Hugoniot value ({n2rh:.2f} n1)")
