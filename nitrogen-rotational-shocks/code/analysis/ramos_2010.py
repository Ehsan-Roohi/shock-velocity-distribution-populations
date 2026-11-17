"""Digitised data: A. Ramos, G. Tejeda, J. M. Fernandez, S. Montero, "Nonequilibrium Processes in Supersonic Jets
of N2, H2, and N2 + H2 Mixtures: (II) Shock Waves", J. Phys. Chem. A 114, 7761 (2010).

Only the PURE N2 jet is recorded here: normal shock wave (Mach disk) at z/D ~ 8 of a jet from a D = 310 um nozzle,
p0 = 1 bar, T0 = 295 K, background pressure pb = 6.1 mbar; M1 = 7.9 at the onset of the shock.

SOURCE TYPE.  The figures in the PDF are 1-bit RASTER images at 1200 dpi (clean computer-generated bitmaps, no
skew; no vector paths).  Marker centres were found by image processing (maximal inscribed disk for the filled
circles, outline matching for the open squares), axes calibrated by least squares on the tick marks (7 ticks in z/D,
4 in n, 6 in T_R; residuals 0.4-0.5 px), and the result drawn back over the bitmap and inspected.

-----------------------------------------------------------------------------------------------------------
FIG1_N2_DENSITY   Fig. 1 (top), filled circles "N2": number density along the jet axis.
    columns: z/D,  n in 10^24 m^-3,  cross flag (1 = the point carries a cross symbol, "points in N2 shock wave
    where Kn > 0.05"),  nominal z in mm (see below).   Linear axes: z/D 4 to 20, n 0 to 0.25e24 m^-3.  22 points.
FIG1_N2_TROT      Fig. 1 (bottom), "N2": rotational temperature along the jet axis.
    columns: z/D,  T_R in K,  square flag,  cross flag,  nominal z in mm.   Linear axes: z/D 4 to 20, T_R 0 to 310 K.
    square flag 0: filled circle, T_R from the Boltzmann distribution of the rotational populations;
    square flag 1: open square, <T_R> = beta * sum_J P_J E_J (eq 1 of the paper), used inside the shock where the
                   populations are not Boltzmann.  4 squares; the three with a cross have Kn > 0.05.
    20 filled circles + 4 open squares.  The square at z/D = 7.90 has no counterpart in the density panel.
    Digitisation uncertainty (1 sigma): z/D +/- 0.01; n +/- 0.0003e24 m^-3; T_R +/- 0.4 K (circles), +/- 0.6 K (squares).
    (1 px = 0.005 z/D = 0.0001e24 m^-3 = 0.12 K.)  The paper's own uncertainties are larger: n 5-10 % (2 sigma),
    T_R(N2) about 1 % random.  The +/- sigma error bars drawn on some density points were not digitised.
    Nominal z: the digitised abscissae fall, within 0.004 z/D, on multiples of 0.05 mm when multiplied by D = 0.310 mm;
    the last column gives that multiple.  This regular grid is an inference from the data, not stated in the paper.
FIG5A_N2_KNUDSEN  Fig. 5a, filled circles: local Knudsen number Kn = lambda/L, L = n/(dn/dz), along the axis of
    the pure N2 jet.  columns: z/D, Kn, flag (1 = marker touches the lower frame, less accurate).
    Axes: z/D linear 0 to 20; Kn logarithmic (base 10), 1e-4 ... 1.  Only points with z/D > 2.4 are given: the
    eight or so points at smaller z/D overlap each other and were not separated (they are far upstream of the shock).
    Digitisation uncertainty: z/D +/- 0.03; Kn +/- 1.5 % (multiplicative).
The H2 data of the same figures (open circles, triangles) and the mixture, barrel-shock and background-gas
figures (Figs. 2-4, 5b, 6-8) are not recorded here.

NOT IN THE PAPER.  There is no figure with rotational populations or Boltzmann plots of N2 inside this shock:
    "The Raman spectra of N2 recorded inside the present shock waves, and the corresponding Boltzmann plots showing
    the departure from a single Boltzmann distribution, are similar to those reported in Figure 4 of ref 11"
    (ref 11 = Phys. Rev. E 62, 4940 (2000); see ramos_2000.py).
SUPPORTING INFORMATION.  The paper states that tables "with the experimental data of the number densities and the
    rotational temperatures plotted in Figures 1-4, 7, and 8" are available from pubs.acs.org.  They are not part
    of the PDF used here; if obtained they supersede the digitised values below.

Usage:  python ramos_2010.py     prints a summary.
"""
import numpy as np

# ------------------------------------------------------------------------------------------------------------
# constants and conditions stated in the paper
# ------------------------------------------------------------------------------------------------------------
D_NOZZLE_UM = 310.0          # "through a cylindrical nozzle of diameter D = 310 um and internal length L ~ 1.8 mm"
L_NOZZLE_MM = 1.8
P0_BAR = 1.0                 # "at stagnation pressure p0 = 1 bar and temperature T0 = 295 K,
T0_K = 295.0                 #  against a background pressure pb = 6.1 +/- 0.1 mbar"
PB_MBAR = 6.1
PB_ERR_MBAR = 0.1
Z_MACH_DISK_OVER_D_EQ2 = 8.6     # "zM/D = 0.67 sqrt(p0/pb) ... for p0 = 1 bar and pb = 6.1 mbar is zM/D ~ 8.6" (vertical dashed line)
Z_POINT1_OVER_D = 7.7        # "The minimum density prior to the N2 shock wave, point 1, is located at z/D ~ 7.7"
MACH_1 = 7.9                 # "At this point the Mach number is M1 = 7.9 according to Miller's parametrization, taking M = 1 at z/D = -0.4"
GAMMA = 7.0 / 5.0
N2_OVER_N1_RANKINE_HUGONIOT = 5.55   # "n2/n1 = 2.4 M1^2/(2 + 0.4 M1^2), yields n2 = 5.55 n1 ... ~10% greater than the experimental density of N2 at point 2"
SHOCK_THICKNESS_UM = 120.0   # "delta = (n2 - n1)/max(dn/dz), is delta ~ 120 um for N2"
LAMBDA_1_UM = 18.0           # "yielding for the present pure N2 jet lambda1 = 18 um"   (eq 3, VSS mean free path)
LAMBDA1_OVER_DELTA = 0.15    # "its inverse shock thickness is lambda1/delta = 0.15, about one-half of the value reported by Alsmeyer ... for the same M = 8"
T_R_MAX_K = 292.0            # "Once the rotational equilibrium is recovered, N2 reaches a maximum T_R ~ 292 K (point 2')"
P_AT_POINT_2PRIME_MBAR = 7.3 # "At this point the estimated pressure p = nkT (assuming T = T_R) is ~ 7.3 mbar"
KN_MAX_NORMAL_SHOCK = 0.25   # "The maxima of Knudsen number are ~0.25 in the normal shock wave and ~0.34 in the barrel shock wave"
KN_MAX_BARREL_SHOCK = 0.34
KN_NON_BOLTZMANN = 0.05      # "Non-Boltzmann distributions of the rotational populations of N2 are observed ... where the local Knudsen number is higher than 0.05"
BETA_CM_K = 1.4388           # "beta = hc/k = 1.4388 cm K"   (eq 1: <T_R> = beta sum_J P_J E_J)
D_REF_ANGSTROM = 4.11        # "For N2, d_ref = 4.11 A at T_ref = 273 K, and omega = 0.74"   (eq 3)
T_REF_K = 273.0
OMEGA_VSS = 0.74
SPATIAL_RESOLUTION_D = 0.05  # "representative spatial resolution was about 0.05D (~15 um) along and across the expansion axis"
DENSITY_UNCERTAINTY_2SIGMA = (0.05, 0.10)   # "the total uncertainty (2 sigma) of n is estimated in 5-10%"
TROT_N2_RANDOM_ERROR = 0.01  # "The standard deviation due to random errors in the N2 rotational temperatures is ~1%"

# ------------------------------------------------------------------------------------------------------------
# Fig. 1: "Number density n (top) and rotational temperature T_R (bottom) along the axis of the pure H2 and N2 jets
# across the normal shock wave. ... Open squares in N2 temperatures indicate <T_R> according to eq 1. Cross symbols
# show the points in N2 shock wave where Kn > 0.05."
# ------------------------------------------------------------------------------------------------------------
FIG1_N2_DENSITY = np.array([      # z/D, n (1e24 m^-3), cross (Kn > 0.05), nominal z (mm)
    [4.515, 0.10928, 0, 1.40],
    [5.808, 0.06579, 0, 1.80],
    [6.451, 0.05336, 0, 2.00],
    [7.095, 0.04293, 0, 2.20],
    [7.746, 0.03680, 0, 2.40],
    [8.063, 0.04393, 1, 2.50],
    [8.225, 0.09213, 1, 2.55],
    [8.387, 0.16567, 0, 2.60],
    [8.710, 0.17890, 0, 2.70],
    [9.031, 0.18193, 0, 2.80],
    [9.358, 0.18194, 0, 2.90],
    [9.674, 0.18446, 0, 3.00],
    [10.325, 0.18301, 0, 3.20],
    [10.968, 0.18302, 0, 3.40],
    [11.614, 0.18195, 0, 3.60],
    [12.255, 0.17851, 0, 3.80],
    [12.906, 0.18115, 0, 4.00],
    [14.191, 0.18446, 0, 4.40],
    [15.486, 0.17890, 0, 4.80],
    [16.773, 0.17957, 0, 5.20],
    [18.066, 0.17016, 0, 5.60],
    [19.353, 0.16117, 0, 6.00],
])
FIG1_N2_TROT = np.array([      # z/D, T_R (K), open square (<T_R>, eq 1), cross (Kn > 0.05), nominal z (mm)
    [4.515, 35.1, 0, 0, 1.40],
    [5.164, 31.8, 0, 0, 1.60],
    [5.809, 29.2, 0, 0, 1.80],
    [6.451, 26.7, 0, 0, 2.00],
    [7.096, 24.9, 0, 0, 2.20],
    [7.745, 23.9, 1, 0, 2.40],
    [7.901, 29.9, 1, 1, 2.45],
    [8.062, 59.9, 1, 1, 2.50],
    [8.227, 210.1, 1, 1, 2.55],
    [8.389, 279.9, 0, 0, 2.60],
    [8.710, 287.6, 0, 0, 2.70],
    [9.032, 290.9, 0, 0, 2.80],
    [9.358, 292.1, 0, 0, 2.90],
    [9.674, 290.2, 0, 0, 3.00],
    [10.326, 289.4, 0, 0, 3.20],
    [10.970, 289.4, 0, 0, 3.40],
    [11.613, 284.8, 0, 0, 3.60],
    [12.256, 279.1, 0, 0, 3.80],
    [12.906, 277.1, 0, 0, 4.00],
    [14.192, 269.7, 0, 0, 4.40],
    [15.486, 262.3, 0, 0, 4.80],
    [16.774, 254.0, 0, 0, 5.20],
    [18.066, 238.8, 0, 0, 5.60],
    [19.352, 227.0, 0, 0, 6.00],
])

# ------------------------------------------------------------------------------------------------------------
# Fig. 5a: "Knudsen number, eq 4 and 5, for the pure N2 jet, (a) along the jet axis, taking l = z"
# ------------------------------------------------------------------------------------------------------------
FIG5A_N2_KNUDSEN = np.array([      # z/D, Kn, flag (1 = marker touches the lower frame)
    [2.58, 7.473e-03, 0],
    [3.22, 7.609e-03, 0],
    [4.51, 1.247e-02, 0],
    [5.80, 1.539e-02, 0],
    [6.45, 1.540e-02, 0],
    [7.11, 1.696e-02, 0],
    [7.74, 1.132e-02, 0],
    [8.07, 2.486e-01, 0],
    [8.23, 1.812e-01, 0],
    [8.40, 3.914e-02, 0],
    [8.70, 3.406e-03, 0],
    [9.05, 6.071e-04, 0],
    [9.36, 5.373e-04, 0],
    [9.69, 3.717e-04, 0],
    [10.33, 1.513e-04, 0],
    [10.97, 1.132e-04, 0],
    [11.62, 4.555e-04, 0],
    [12.25, 7.820e-05, 1],
    [12.90, 4.379e-04, 0],
    [14.20, 1.094e-04, 0],
    [15.50, 2.518e-04, 0],
    [16.79, 4.377e-04, 0],
    [18.07, 1.019e-03, 0],
])


def density():
    """z/D and number density in m^-3 of the pure N2 jet (Fig. 1 top)."""
    return FIG1_N2_DENSITY[:, 0], FIG1_N2_DENSITY[:, 1] * 1e24


def rotational_temperature():
    """z/D, T_R (K) and a boolean array that is True for the <T_R> (open square, eq 1) points (Fig. 1 bottom)."""
    return FIG1_N2_TROT[:, 0], FIG1_N2_TROT[:, 1], FIG1_N2_TROT[:, 2] == 1


if __name__ == "__main__":
    z, n = density()
    i = int(np.argmin(n))
    print("J. Phys. Chem. A 114, 7761 (2010) - pure N2 jet, normal shock")
    print("  density: %d points, z/D %.2f..%.2f; minimum n1 = %.3e m^-3 at z/D = %.3f; maximum %.3e m^-3; max/min = %.2f"
          % (len(z), z.min(), z.max(), n[i], z[i], n.max(), n.max() / n[i]))
    print("  Rankine-Hugoniot n2 = 5.55 n1 = %.3e m^-3 (paper: ~10%% above the measured density at point 2)" % (5.55 * n[i]))
    zt, T, sq = rotational_temperature()
    print("  T_R: %d points (%d open squares); minimum %.1f K at z/D = %.3f; maximum %.1f K at z/D = %.3f"
          % (len(zt), int(sq.sum()), T.min(), zt[np.argmin(T)], T.max(), zt[np.argmax(T)]))
    for a, b in zip(zt[sq], T[sq]):
        print("     <T_R> square: z/D = %.3f  %.1f K" % (a, b))
    k = FIG5A_N2_KNUDSEN
    print("  Kn: %d points; maximum %.3f at z/D = %.2f" % (len(k), k[:, 1].max(), k[np.argmax(k[:, 1]), 0]))
    print("  largest departure of z*D from the 0.05 mm grid: %.4f mm"
          % np.abs(np.r_[FIG1_N2_DENSITY[:, 0] * 0.310 - FIG1_N2_DENSITY[:, 3], FIG1_N2_TROT[:, 0] * 0.310 - FIG1_N2_TROT[:, 4]]).max())
