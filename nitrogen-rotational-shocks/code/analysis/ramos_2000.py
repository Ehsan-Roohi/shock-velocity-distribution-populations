"""Digitised data: A. Ramos, B. Mate, G. Tejeda, J. M. Fernandez, S. Montero,
"Raman spectroscopy of hypersonic shock waves", Phys. Rev. E 62, 4940 (2000).

Five normal shock waves (Mach disks) A-E in a free jet of N2 (nozzle D = 313 um, P0 = 100 kPa, T0 = 295 K),
probed along the jet axis by spontaneous Raman scattering.  Everything below was read from the publisher's PDF.

SOURCE TYPE.  Every figure in the PDF is a 1-bit RASTER scan at about 300 dpi (no vector paths), slightly sheared.
Marker centres were located by image processing on the native bitmap; axes were calibrated from the tick marks
with an affine map (so the shear is removed); every series was drawn back over the bitmap and inspected.

-----------------------------------------------------------------------------------------------------------
FIG4_*   Fig. 4 (bottom): Boltzmann plots of shock wave B.
    abscissa  J(J+1)  (linear, 0 to 450; markers sit at the integer values, J is given explicitly)
    ordinate  ln X, NATURAL logarithm (axis label "ln X"), X = P_J / [g_J (2J+1)]  (caption).  The slope of the drawn
              straight line (a) gives 13.92 K with ln, against 13.9 +/- 0.1 K printed, which confirms ln (not log10).
    The absolute level of ln X (-56 ... -63) is NOT that of populations normalised to sum 1; see NOTE 1.
    FIG4_A  open triangles, 5 points J = 0..4   "At point 1", z/D = 16.0, T_rot = 13.9 +/- 0.1 K
    FIG4_B  filled circles, 17 points J = 0..16 "In between", z/D = 18.9, <T_rot> = 130 +/- 10 K
    FIG4_C  open circles, 19 points J = 0, 3..20 "At point 2", z/D = 20.1, T_rot = 262 +/- 3 K
    FIG4_C_HIDDEN  the J = 1 and J = 2 open circles of (c): they lie under the J = 2 filled circle of (b), the frame
            and the J = 0 circle.  J = 1 shows about half of its outline, J = 2 is entirely covered (it must lie
            within about 1.5 px of the filled circle, otherwise its rim would show).  Values come from a model fit
            of the crowded bitmap and are kept OUT of FIG4_C; use them only with the larger uncertainty given.
    Digitisation uncertainty (1 sigma): ln X +/- 0.015 for (b) and (c) [1 px = 0.0152; marker abscissae reproduce
            J(J+1) to +/- 0.3, i.e. 0.6 px], +/- 0.02 for (a); FIG4_C_HIDDEN +/- 0.03 (J = 1), +/- 0.04 (J = 2).
            Triangles: the data point is taken at the centroid of the triangle (this makes the five triangles
            scatter evenly about the drawn line; the box-centre convention would put all of them 0.06 higher).
    Independent check: relative ln X computed from the PEAK HEIGHTS of the spectra in the top panel of Fig. 4
            (not integrated intensities) agrees with the plotted points within about 0.04 for the strong lines
            (J up to about 9 in b and c), including the two hidden points of (c): -58.13 and -58.18 from the
            spectrum against -58.14 and -58.14 from the bitmap model.  Exception, NOTE 2: the J = 0 point of (b).

FIG3_*   Fig. 3: relative number density and rotational temperature along the axis, shocks A to E.
    abscissa  z/D (linear, 0 to 80; D = 313 um; the top axis of the figure is z in mm)
    left axis   n/n1, number density relative to the minimum density n1 at point 1 (linear, 1 to 9): FILLED circles
    right axis  rotational temperature in K (linear, 0 to 300): OPEN circles.  Between the two dashed lines
                (points 1 and 2) the plotted temperature is <T_rot> of Eq. (5); outside it is the Boltzmann slope.
    FIG3_DENSITY[s]  array (N, 3): z/D, n/n1, flag        FIG3_TROT[s]  array (N, 3): z/D, T (K), flag
    flag 0  isolated marker (automatic centre)
    flag 1  marker overlapping other markers or a drawn curve; centre from a model fit of the local bitmap
    flag 2  marker largely hidden, or member of a merged group whose number of members is itself inferred
    Digitisation uncertainty (1 sigma): flag 0: z/D +/- 0.05, n/n1 +/- 0.015, T +/- 0.5 K;
            flags 1 and 2: z/D +/- 0.1, n/n1 +/- 0.03, T +/- 1 K.
            Checks: (i) in panels B, C, D the abscissae fall on a 0.1 mm grid with 0.035 z/D rms scatter and less
            than 0.045 z/D offset; (ii) the lowest open circles reproduce the printed T_rot^min within 0.3 K and
            the lowest filled circle of each panel comes out at n/n1 = 0.99-1.01; (iii) n/n1 times one constant per
            shock reproduces the dots of the log-log Fig. 2 with 0.3-0.6 % rms; (iv) two independent centre
            estimators agree to 0.15 px rms on isolated markers.  The ordinate calibration uses the tick marks of
            both axes and the two horizontal frame lines of each panel (n/n1 = 9 and n/n1 = 1, i.e. T = 0); the
            scan is distorted by up to 1 px near the bottom of panels D and E, which is inside the quoted errors.
    FIG3_A_UNRESOLVED_TRACE  panel A, just after point 2 (z/D 8.8 to 11.5): the filled circles overlap into one
            continuous band and cannot be separated.  The array samples the CENTRELINE of that band (z/D, n/n1);
            these are not individual data points.  The first three samples also overlap line II and the label line.
    FIG3_DASHED_Z_OVER_D  the two vertical dashed lines of each panel (points 1 and 2), +/- 0.05.
    FIG3_TROT_MIN_K       the T_rot^min printed in each panel.  No Mach number is printed in Fig. 3.
    Lines I (isentropic expansion) and II (Rankine-Hugoniot jump) of the figure are not digitised.
    Points possibly lost: markers exactly concentric with another marker of the same kind cannot be seen.

N1_ABS_1E22_M3   absolute number density at the density minimum (point 1) of each shock, from Fig. 2 (log-log,
            10^22 m^-3 against z/D): obtained as the constant that maps the Fig. 3 n/n1 onto the Fig. 2 dots
            (30 to 53 matched points per shock).  Digitisation +/- 1 %; the paper quotes +/- 10 % for absolute
            densities.  Absolute density of any Fig. 3 point = n/n1 * N1_ABS_1E22_M3[shock] * 1e22 m^-3.

FIG5_FRACTIONS   Fig. 5: the three molar fractions printed in the figure (spectrum b, shock B, z/D = 18.9).
FIG6_TABLE       table printed inside Fig. 6: M1, delta, lambda1, Kn_max for A-E (exact transcription).
            The data points of Fig. 6 (reduced density against z~/lambda1) are not digitised.

NOTE 1 (level of ln X).  The three plots are not normalised to sum_J P_J = 1.  sum_J g_J (2J+1) X_J over the plotted
    levels is 7.07e-24 (a), 8.35e-24 (b), 2.50e-23 (c, with its two hidden points; 2.28e-23 without); the ratio
    c/a = 3.5 is close to the density ratio 3.8 read from Fig. 3 between z/D = 20.1 and 16.0, so X seems to scale
    with the local density.  This is an observation made while digitising, not a statement of the paper.
    Use populations() to renormalise.
NOTE 2 (J = 0 of plot b).  The J = 0 filled circle is plotted at ln X = -56.78.  The peak heights of spectrum (b)
    in the same figure, and the three-fraction decomposition of Fig. 5, both put it about 0.35-0.4 lower relative
    to J >= 1 (J = 1..6 agree with the spectrum within 0.04).  The plotted value is recorded as published.
NOTE 3.  Fig. 4 and Fig. 5 quote <T_rot> = 130 +/- 10 K at z/D = 18.9; the open circle of Fig. 3 (panel B) nearest
    to that position is at z/D = 18.87 with 149 K.  Eq. (5) summed over the plotted levels of (b) gives 111 K
    (levels J > 16 are not plotted).

Usage:  python ramos_2000.py     prints a summary and the consistency numbers quoted above.
"""
import numpy as np

# ------------------------------------------------------------------------------------------------------------
# constants and conditions stated in the paper
# ------------------------------------------------------------------------------------------------------------
D_NOZZLE_UM = 313.0          # "expansion of N2 through a nozzle of diameter D = 313 um, and internal length L = 1 mm"
L_NOZZLE_MM = 1.0
P0_KPA = 100.0               # "under nominal constant stagnation pressure P0 = 100 kPa and temperature T0 = 295 K"
T0_K = 295.0
P0_EFF_KPA = 80.0            # "This fit yields an effective stagnation pressure P0eff = 80 kPa, 20% smaller than the nominal one"
GAMMA = 7.0 / 5.0            # "isentropic expansion of an ideal gas with a heat capacity ratio gamma = 7/5"
T_NOZZLE_EXIT_K = 246.0      # "complete rethermalization (nozzle exit temperature Te ~ 246 K), denoting the end of the shock wave"
BETA_CM_K = 1.4388           # "beta = hc/k = 1.4388 cm K"
B_ROT_CM1 = 1.98973          # "B = 1.98973 cm-1 the rotational constant of N2"
D_REF_ANGSTROM = 4.11        # "d_ref = 4.11 A is the variable soft sphere (VSS) molecular diameter of N2 at ... T_ref = 273 K,
T_REF_K = 273.0              #  and omega = 0.74 its viscosity index"   (mean free path, Eq. 8)
OMEGA_VSS = 0.74
SHOCKS = ("A", "B", "C", "D", "E")
RESIDUAL_PRESSURE_PA = {"A": 420.0, "B": 100.0, "C": 50.0, "D": 28.0, "E": 18.0}     # "for Pr = 420, 100, 50, 28, and 18 Pa, respectively"
NOMINAL_LOCATION_D = {"A": 9.0, "B": 18.0, "C": 27.0, "D": 36.0, "E": 45.0}          # "located at distances of about 9, 18, 27, 36, and 45 D"
MACH_RANGE = (7.7, 15.3)     # "span the Mach number range 7.7 <= M <= 15.3, according to the usual estimate"
ACCURACY_TROT_REL = 0.05     # "high quality rotational temperature (+/-5%), and absolute density data (+/-10%)"
ACCURACY_DENSITY_REL = 0.10
SPATIAL_RESOLUTION_UM = (10.0, 15.0)   # "effective spatial resolution 10 < a < 15 um" along z; z set to +/-1 um


def g_J(J):
    """Nuclear-spin weight of N2 (14N2): 6 for even J, 3 for odd J.  The paper only calls g_J "the spin statistical
    weight" (Eq. 4) without printing the values; these are the standard ones."""
    J = np.asarray(J)
    return np.where(J % 2 == 0, 6, 3)


# ------------------------------------------------------------------------------------------------------------
# Fig. 6: table printed in the figure   ("M1 and lambda1 are Mach number and mean free path at the onset of the
# shock wave, delta the shock wave thickness, and Knmax = max[(lambda/n)(dn/dz)]")
# header of the table: "M1   delta (um)   lambda1   Kn^max" - the unit (um) is printed between delta and lambda1
# and is taken to apply to both (lambda1 from Eq. 8 with the densities of Fig. 2 is consistent with um).
# ------------------------------------------------------------------------------------------------------------
FIG6_TABLE = {
    "A": dict(M1=7.7, delta_um=95.0, lambda1_um=21.8, Kn_max=0.33),
    "B": dict(M1=10.6, delta_um=400.0, lambda1_um=86.7, Kn_max=0.35),
    "C": dict(M1=12.2, delta_um=770.0, lambda1_um=161.5, Kn_max=0.42),
    "D": dict(M1=13.6, delta_um=1110.0, lambda1_um=256.2, Kn_max=0.54),
    "E": dict(M1=15.3, delta_um=1640.0, lambda1_um=433.6, Kn_max=0.59),
}

# ------------------------------------------------------------------------------------------------------------
# Fig. 5: "Breakdown of a non-Boltzmann rotational Raman spectrum of N2, recorded within shock wave B, spectrum (b)
# of Fig. 4, into the simulated spectra of xc, xs, and xw molar fractions."  Numbers printed in the panels.
# ------------------------------------------------------------------------------------------------------------
FIG5_FRACTIONS = dict(
    shock="B", z_over_D=18.9,
    x_c=0.41, T_c_K=14.0,          # "x_c = 0.41   T_rot = 14 K"      cold molecules
    x_s=0.09, T_s_K=59.0,          # "x_s = 0.09   <T_rot> = 59 K"    scattered molecules (temperature in the sense of Eq. 5)
    x_w=0.50, T_w_K=236.0,         # "x_w = 0.50   T_rot = 236 K"     warm (rethermalised) molecules
)
# text, Sec. V B: "For J <= 4 plot (b) tends asymptotically to a straight line corresponding to a temperature
# T_rot = 14 K, while for J >= 8 it tends asymptotically to another straight line corresponding to T_rot = 235 K."
FIG4_B_ASYMPTOTES_K = dict(cold=14.0, warm=235.0)      # (the figure label of Fig. 5 gives 236 K for the warm part)

# ------------------------------------------------------------------------------------------------------------
# Fig. 4 (bottom): Boltzmann plots, shock wave B.   columns: J, ln X
# ------------------------------------------------------------------------------------------------------------
FIG4_POSITIONS = {      # labels printed in the top panel of Fig. 4
    "a": dict(where="At point 1", z_over_D=16.0, T_rot_K=13.9, T_err_K=0.1, kind="Boltzmann slope"),
    "b": dict(where="In between", z_over_D=18.9, T_rot_K=130.0, T_err_K=10.0, kind="<T_rot>, Eq. (5)"),
    "c": dict(where="At point 2", z_over_D=20.1, T_rot_K=262.0, T_err_K=3.0, kind="Boltzmann slope"),
}
FIG4_A = np.array([      # open triangles, spectrum (a), z/D = 16.0
    [0, -56.458],
    [1, -56.895],
    [2, -57.669],
    [3, -58.960],
    [4, -60.554],
])
FIG4_B = np.array([      # filled circles, spectrum (b), z/D = 18.9
    [0, -56.780],
    [1, -57.525],
    [2, -58.138],
    [3, -58.925],
    [4, -59.453],
    [5, -59.906],
    [6, -60.086],
    [7, -60.330],
    [8, -60.611],
    [9, -60.865],
    [10, -61.153],
    [11, -61.495],
    [12, -61.651],
    [13, -62.047],
    [14, -62.291],
    [15, -62.502],
    [16, -63.085],
])
FIG4_C = np.array([      # open circles, spectrum (c), z/D = 20.1   (J = 1, 2 hidden: see FIG4_C_HIDDEN)
    [0, -58.009],
    [3, -58.228],
    [4, -58.238],
    [5, -58.506],
    [6, -58.485],
    [7, -58.693],
    [8, -58.871],
    [9, -59.070],
    [10, -59.218],
    [11, -59.601],
    [12, -59.663],
    [13, -60.022],
    [14, -60.411],
    [15, -60.638],
    [16, -61.000],
    [17, -61.744],
    [18, -61.858],
    [19, -61.827],
    [20, -62.906],
])
FIG4_C_HIDDEN = np.array([      # J, ln X, 1-sigma uncertainty; model-fit estimates, NOT plotted-marker readings
    [1, -58.139, 0.03],      # about half of the outline visible
    [2, -58.136, 0.04],      # entirely under the J = 2 filled circle of (b), which is at ln X = -58.138
])
# straight lines drawn in the figure, measured on the bitmap: ln X = intercept + slope * J(J+1)
FIG4_DRAWN_LINE_A = dict(slope=-0.20561, intercept=-56.476, T_K=13.92)      # through the triangles; printed T_rot = 13.9 +/- 0.1 K
FIG4_DRAWN_LINE_C = dict(slope=-0.011053, intercept=-58.050, T_K=259.0)     # through the open circles; printed T_rot = 262 +/- 3 K

# ------------------------------------------------------------------------------------------------------------
# Fig. 3: "Relative number densities (filled circles), and rotational temperatures (open circles) of N2 shock
# waves along the expansion axis. Vertical dashed lines indicate the limits of the shock waves."
# ------------------------------------------------------------------------------------------------------------
FIG3_TROT_MIN_K = {"A": (22.7, 0.1), "B": (13.9, 0.1), "C": (11.5, 0.2), "D": (10.4, 0.1), "E": (10.1, 0.2)}   # printed "T_rot^min = ..." (value, +/-)
FIG3_DASHED_Z_OVER_D = {      # (point 1, point 2), measured; +/- 0.05
    "A": (7.98, 8.82),
    "B": (17.05, 20.05),
    "C": (24.05, 28.05),
    "D": (32.01, 38.03),
    "E": (42.04, 49.48),
}
N1_ABS_1E22_M3 = {"A": 3.0856, "B": 0.6381, "C": 0.3496, "D": 0.1888, "E": 0.1039}      # density at point 1, 10^22 m^-3, from Fig. 2
FIG3_DENSITY = {      # filled circles: z/D, n/n1, flag
    "A": np.array([
        [3.161, 6.728, 0],
        [3.803, 4.838, 0],
        [4.754, 2.952, 0],
        [5.717, 2.033, 0],
        [6.399, 1.612, 0],
        [6.689, 1.443, 0],
        [6.986, 1.305, 0],
        [7.360, 1.184, 2],
        [7.666, 1.093, 2],
        [8.010, 1.004, 2],
        [8.164, 0.990, 2],
        [8.305, 1.119, 2],
        [8.460, 1.795, 0],
        [8.644, 3.679, 0],
        [12.113, 5.166, 0],
        [12.813, 5.162, 0],
        [13.428, 5.207, 0],
        [14.123, 5.154, 0],
        [14.718, 5.134, 0],
        [15.365, 5.167, 0],
        [16.000, 5.315, 0],
        [16.661, 5.466, 0],
        [17.289, 5.554, 0],
        [17.929, 5.430, 0],
        [18.606, 5.291, 0],
        [19.232, 5.160, 0],
        [19.861, 5.043, 0],
        [20.520, 5.027, 0],
        [21.464, 5.232, 0],
        [22.418, 5.531, 0],
        [23.210, 5.770, 0],
        [24.009, 6.000, 0],
        [24.833, 6.087, 0],
        [25.635, 6.076, 0],
        [27.215, 6.077, 0],
        [28.826, 5.959, 0],
        [29.635, 5.858, 0],
        [30.419, 5.731, 0],
        [31.216, 5.805, 0],
        [32.029, 5.968, 0],
        [33.627, 6.370, 0],
        [35.187, 6.420, 0],
        [36.798, 6.440, 0],
        [38.413, 6.225, 0],
        [39.994, 6.092, 0],
        [41.588, 6.094, 0],
        [43.177, 6.362, 0],
        [44.755, 6.395, 0],
        [46.333, 6.389, 0],
        [47.957, 6.141, 0],
        [51.136, 5.964, 0],
        [52.748, 6.196, 0],
        [54.327, 6.135, 0],
        [57.539, 5.800, 0],
        [60.699, 5.644, 0],
        [63.904, 5.659, 0],
        [67.033, 5.405, 0],
        [70.306, 5.299, 0],
        [73.497, 5.206, 0],
        [76.709, 4.880, 0],
    ]),
    "B": np.array([
        [6.382, 7.831, 0],
        [6.693, 6.988, 0],
        [7.337, 5.770, 0],
        [7.934, 4.862, 0],
        [9.539, 3.223, 0],
        [12.784, 1.728, 0],
        [15.994, 1.127, 0],
        [16.988, 1.002, 2],
        [17.604, 1.008, 2],
        [17.912, 1.005, 2],
        [18.267, 1.047, 2],
        [18.574, 1.231, 0],
        [18.873, 1.524, 0],
        [19.229, 2.020, 0],
        [19.522, 2.771, 0],
        [19.811, 3.706, 0],
        [20.154, 4.366, 1],
        [20.500, 4.474, 1],
        [20.784, 4.737, 1],
        [21.146, 4.774, 1],
        [21.453, 4.892, 0],
        [21.750, 5.082, 2],
        [22.068, 5.100, 2],
        [22.376, 5.066, 2],
        [23.064, 5.315, 0],
        [23.685, 5.550, 0],
        [24.342, 5.666, 0],
        [25.000, 5.897, 0],
        [25.618, 6.005, 0],
        [26.249, 6.113, 0],
        [26.948, 6.351, 0],
        [27.547, 6.398, 0],
        [28.161, 6.482, 0],
        [28.828, 6.482, 0],
        [29.443, 6.380, 0],
        [30.381, 6.277, 0],
        [32.046, 6.052, 0],
        [33.599, 5.707, 0],
        [35.196, 5.551, 0],
        [36.762, 5.382, 0],
        [38.406, 5.341, 0],
        [41.580, 5.303, 0],
        [43.192, 5.236, 0],
        [44.753, 5.286, 0],
        [46.329, 5.362, 0],
        [47.921, 5.483, 0],
        [49.543, 5.736, 0],
        [51.139, 5.926, 0],
        [52.740, 6.094, 0],
        [54.305, 5.977, 0],
        [55.917, 5.900, 0],
        [57.529, 5.852, 0],
        [59.094, 5.686, 0],
        [60.660, 5.520, 0],
        [62.307, 5.351, 0],
        [63.902, 5.333, 0],
        [65.466, 5.216, 0],
        [67.113, 5.054, 0],
        [70.254, 5.023, 0],
        [73.458, 5.057, 0],
        [75.102, 5.026, 0],
        [76.681, 5.010, 0],
    ]),
    "C": np.array([
        [7.946, 8.849, 0],
        [9.555, 5.865, 0],
        [12.769, 3.148, 0],
        [15.981, 2.065, 0],
        [19.182, 1.467, 0],
        [20.814, 1.245, 0],
        [22.364, 1.033, 0],
        [23.976, 0.999, 0],
        [24.949, 1.113, 0],
        [25.579, 1.329, 0],
        [26.208, 1.623, 0],
        [26.540, 2.075, 0],
        [26.875, 2.389, 0],
        [27.189, 2.896, 0],
        [27.489, 3.360, 0],
        [27.856, 3.681, 0],
        [28.032, 3.951, 1],
        [28.459, 4.087, 0],
        [28.787, 4.384, 0],
        [29.117, 4.507, 1],
        [29.416, 4.567, 1],
        [30.093, 4.842, 0],
        [30.723, 5.039, 0],
        [31.354, 5.230, 0],
        [32.006, 5.321, 0],
        [32.600, 5.534, 0],
        [33.586, 5.864, 0],
        [35.154, 6.313, 0],
        [36.774, 6.648, 0],
        [37.757, 6.804, 0],
        [38.400, 6.883, 0],
        [40.009, 6.790, 0],
        [41.580, 6.379, 0],
        [43.165, 6.050, 0],
        [44.719, 5.666, 0],
        [46.332, 5.544, 0],
        [47.937, 5.034, 0],
        [49.536, 4.838, 0],
        [51.083, 4.768, 0],
        [52.696, 4.671, 0],
        [54.293, 4.567, 0],
        [55.889, 4.470, 0],
        [57.485, 4.409, 0],
        [59.076, 4.583, 0],
        [60.650, 4.774, 0],
        [62.290, 4.945, 0],
        [63.862, 5.227, 0],
        [65.434, 5.509, 0],
        [67.076, 5.562, 0],
        [68.685, 5.661, 0],
        [70.280, 5.630, 0],
        [71.860, 5.536, 0],
        [73.424, 5.468, 0],
        [75.085, 5.401, 0],
        [76.648, 5.313, 0],
    ]),
    "D": np.array([
        [12.816, 5.831, 0],
        [15.934, 3.823, 0],
        [19.179, 2.710, 0],
        [22.402, 1.894, 0],
        [25.633, 1.420, 0],
        [28.789, 1.134, 0],
        [30.373, 1.034, 0],
        [32.004, 0.998, 0],
        [33.575, 1.044, 0],
        [34.520, 1.296, 0],
        [35.163, 1.645, 0],
        [35.803, 2.157, 0],
        [36.462, 2.546, 0],
        [37.051, 3.514, 0],
        [37.708, 4.086, 0],
        [38.363, 4.933, 0],
        [38.944, 5.168, 0],
        [39.620, 5.501, 0],
        [40.252, 5.658, 0],
        [40.878, 6.020, 0],
        [41.549, 6.656, 0],
        [42.175, 6.593, 0],
        [42.795, 6.967, 0],
        [43.127, 7.195, 1],
        [43.462, 7.249, 1],
        [44.069, 7.675, 0],
        [44.715, 7.798, 0],
        [45.349, 7.880, 0],
        [46.006, 8.262, 0],
        [46.302, 8.087, 0],
        [46.627, 7.928, 0],
        [47.276, 8.156, 0],
        [47.880, 8.385, 0],
        [48.545, 8.286, 0],
        [49.166, 8.793, 0],
        [49.517, 8.363, 1],
        [49.811, 8.305, 1],
        [50.474, 8.498, 0],
        [51.083, 8.188, 0],
        [52.667, 7.899, 0],
        [54.235, 7.620, 0],
        [55.885, 7.301, 0],
        [57.458, 6.784, 0],
        [59.035, 6.081, 0],
        [60.602, 5.828, 0],
        [62.271, 5.379, 0],
        [63.853, 5.207, 0],
        [65.405, 4.902, 0],
        [67.024, 4.522, 0],
        [68.640, 4.278, 0],
        [70.222, 4.110, 0],
        [71.814, 4.231, 0],
        [73.422, 4.359, 0],
        [75.065, 4.374, 0],
        [76.641, 4.450, 0],
    ]),
    "E": np.array([
        [15.959, 6.853, 0],
        [19.205, 4.926, 0],
        [22.442, 3.458, 0],
        [25.669, 2.586, 0],
        [30.399, 1.891, 0],
        [38.461, 1.133, 0],
        [40.037, 1.094, 0],
        [41.635, 1.007, 0],
        [43.205, 1.050, 0],
        [43.980, 1.315, 0],
        [44.783, 1.397, 0],
        [45.583, 1.696, 0],
        [46.334, 1.944, 0],
        [47.177, 2.469, 0],
        [47.944, 3.158, 0],
        [48.726, 3.854, 0],
        [49.091, 4.275, 0],
        [49.489, 4.670, 0],
        [50.292, 5.198, 0],
        [51.108, 5.829, 0],
        [51.877, 6.373, 0],
        [52.698, 6.790, 0],
        [53.521, 7.130, 0],
        [54.279, 7.415, 0],
        [55.067, 7.840, 0],
        [55.898, 8.354, 0],
        [56.682, 8.607, 0],
        [57.477, 8.629, 0],
        [58.285, 8.646, 0],
        [59.058, 8.404, 0],
        [59.885, 8.521, 0],
        [60.669, 8.366, 0],
        [62.279, 8.395, 0],
        [63.851, 7.919, 0],
        [67.046, 6.892, 0],
        [70.246, 6.365, 0],
        [73.451, 5.609, 0],
        [76.653, 5.002, 0],
    ]),
}
FIG3_TROT = {      # open circles: z/D, T (K), flag
    "A": np.array([
        [6.384, 25.7, 2],
        [7.013, 23.9, 1],
        [7.661, 22.7, 1],
        [8.008, 30.9, 1],
        [8.168, 60.5, 0],
        [8.325, 126.8, 0],
        [8.452, 210.7, 0],
        [8.604, 262.1, 0],
        [8.776, 275.0, 1],
        [8.931, 272.9, 1],
        [9.252, 275.0, 1],
        [9.556, 275.2, 1],
        [10.225, 278.0, 1],
        [11.181, 272.0, 0],
        [12.796, 254.6, 0],
        [14.414, 241.8, 0],
        [15.208, 235.5, 0],
        [15.989, 231.9, 0],
        [16.824, 225.7, 1],
        [17.611, 223.4, 1],
        [18.435, 217.9, 0],
        [19.215, 207.3, 1],
        [20.007, 204.8, 1],
        [20.833, 201.7, 1],
        [21.619, 201.5, 1],
        [22.412, 201.1, 1],
        [24.024, 201.9, 0],
        [25.635, 199.2, 0],
        [27.236, 194.6, 1],
        [28.815, 190.9, 1],
        [30.422, 186.4, 0],
        [32.022, 182.8, 1],
        [35.207, 189.2, 0],
        [38.408, 183.8, 0],
        [44.773, 185.9, 0],
        [51.156, 182.4, 1],
        [57.529, 188.1, 0],
        [63.918, 191.8, 0],
        [70.298, 196.2, 0],
        [76.699, 197.4, 0],
    ]),
    "B": np.array([
        [15.979, 13.6, 1],
        [16.947, 14.0, 1],
        [17.592, 14.6, 1],
        [17.928, 20.0, 1],
        [18.244, 46.9, 0],
        [18.557, 89.8, 0],
        [18.870, 148.6, 0],
        [19.208, 193.6, 0],
        [19.567, 221.7, 0],
        [19.869, 241.9, 0],
        [20.194, 261.3, 2],
        [20.519, 260.0, 2],
        [20.832, 261.0, 2],
        [21.154, 275.5, 0],
        [21.798, 268.4, 1],
        [22.440, 266.9, 1],
        [23.084, 261.1, 0],
        [23.708, 266.0, 0],
        [24.362, 256.4, 1],
        [24.995, 260.5, 1],
        [25.641, 257.4, 1],
        [26.271, 261.8, 1],
        [27.260, 257.9, 0],
        [28.831, 241.6, 0],
        [30.430, 236.8, 0],
        [32.022, 225.3, 0],
        [35.206, 208.8, 0],
        [38.409, 202.7, 0],
        [41.580, 197.0, 0],
        [44.764, 195.2, 0],
        [47.940, 195.1, 0],
        [51.146, 191.7, 0],
        [54.322, 202.7, 0],
        [57.515, 192.8, 0],
        [60.703, 197.6, 0],
        [63.901, 196.9, 0],
        [67.094, 197.5, 0],
        [70.297, 198.6, 0],
        [73.496, 203.1, 0],
        [76.702, 205.7, 0],
    ]),
    "C": np.array([
        [20.779, 11.9, 1],
        [21.740, 11.5, 1],
        [22.380, 11.5, 1],
        [23.033, 12.1, 1],
        [23.648, 13.3, 1],
        [24.300, 25.2, 0],
        [24.938, 49.3, 0],
        [25.591, 61.3, 0],
        [26.243, 156.0, 0],
        [26.890, 200.8, 0],
        [27.217, 226.9, 0],
        [27.520, 237.4, 1],
        [27.836, 233.9, 1],
        [28.163, 248.3, 0],
        [28.513, 258.8, 1],
        [28.806, 258.7, 1],
        [29.127, 264.9, 1],
        [29.452, 258.4, 1],
        [29.773, 254.9, 1],
        [30.402, 259.5, 1],
        [31.052, 254.8, 0],
        [32.002, 253.0, 0],
        [33.601, 250.3, 0],
        [35.183, 255.9, 0],
        [36.787, 256.9, 0],
        [38.389, 250.3, 0],
        [41.582, 242.0, 0],
        [44.754, 222.9, 0],
        [47.945, 211.8, 0],
        [51.126, 197.6, 0],
        [54.309, 196.6, 0],
        [57.502, 184.7, 0],
        [60.682, 189.2, 0],
        [63.885, 197.1, 0],
        [67.066, 207.7, 0],
        [70.291, 207.6, 0],
        [73.474, 207.6, 0],
        [76.685, 202.6, 0],
    ]),
    "D": np.array([
        [28.774, 10.2, 1],
        [30.388, 12.7, 0],
        [31.986, 14.4, 1],
        [32.621, 16.2, 1],
        [33.266, 17.9, 1],
        [33.582, 48.7, 0],
        [33.886, 55.5, 0],
        [34.210, 85.4, 0],
        [34.846, 119.6, 0],
        [35.469, 164.2, 1],
        [35.795, 168.4, 1],
        [36.120, 179.3, 0],
        [36.423, 220.3, 0],
        [37.074, 232.3, 1],
        [37.686, 229.3, 1],
        [38.350, 237.0, 0],
        [38.975, 244.1, 0],
        [39.628, 269.6, 1],
        [40.267, 267.0, 1],
        [40.917, 262.0, 0],
        [41.538, 268.2, 1],
        [42.198, 268.5, 1],
        [42.823, 269.6, 1],
        [43.476, 252.1, 1],
        [44.083, 257.0, 1],
        [44.723, 247.8, 1],
        [45.339, 253.5, 1],
        [46.005, 268.0, 1],
        [46.622, 256.4, 1],
        [47.269, 257.8, 1],
        [47.903, 254.0, 1],
        [48.547, 258.0, 1],
        [49.493, 259.1, 0],
        [51.100, 251.7, 0],
        [54.276, 239.1, 0],
        [57.467, 197.2, 0],
        [60.652, 194.1, 0],
        [63.851, 180.7, 0],
        [67.025, 182.0, 0],
        [70.233, 175.1, 0],
        [73.433, 178.0, 0],
        [76.640, 180.0, 0],
    ]),
    "E": np.array([
        [35.246, 10.4, 0],
        [36.821, 10.6, 1],
        [38.434, 10.2, 1],
        [40.047, 10.2, 1],
        [41.622, 14.7, 0],
        [42.433, 19.7, 1],
        [43.211, 21.4, 1],
        [43.983, 47.5, 0],
        [44.773, 52.0, 0],
        [45.552, 63.4, 0],
        [46.331, 75.4, 0],
        [47.116, 133.7, 1],
        [47.914, 189.8, 0],
        [48.719, 225.3, 0],
        [49.510, 234.2, 0],
        [50.317, 265.9, 1],
        [51.114, 268.0, 1],
        [51.897, 275.3, 1],
        [52.689, 275.3, 1],
        [53.502, 280.9, 1],
        [54.306, 279.4, 1],
        [55.105, 282.8, 0],
        [55.876, 278.3, 2],
        [57.486, 268.5, 0],
        [59.062, 269.1, 0],
        [60.662, 262.9, 0],
        [62.257, 248.1, 0],
        [63.867, 255.0, 1],
        [67.053, 230.2, 0],
        [70.260, 225.1, 0],
        [73.461, 213.1, 0],
        [76.655, 210.2, 0],
    ]),
}
FIG3_A_UNRESOLVED_TRACE = np.array([      # panel A: centreline of the merged band of filled circles, z/D, n/n1 (not individual points)
    [8.81, 4.888],
    [9.10, 4.955],
    [9.40, 5.014],
    [9.69, 5.041],
    [9.98, 5.036],
    [10.27, 5.056],
    [10.57, 5.080],
    [10.86, 5.109],
    [11.15, 5.120],
    [11.44, 5.100],
])


def fig4_series(name):
    """J (int array), J(J+1), ln X of Boltzmann plot 'a', 'b' or 'c' (hidden points of c not included)."""
    a = {"a": FIG4_A, "b": FIG4_B, "c": FIG4_C}[name]
    J = a[:, 0].astype(int)
    return J, J * (J + 1), a[:, 1]


def populations(name, include_hidden=False):
    """Fractional populations P_J = g_J (2J+1) X_J renormalised to sum 1 over the PLOTTED levels only.
    Returns J, P_J.  With include_hidden=True the two model-fit points of (c) are added."""
    a = {"a": FIG4_A, "b": FIG4_B, "c": FIG4_C}[name]
    if include_hidden and name == "c":
        a = np.vstack([a, FIG4_C_HIDDEN[:, :2]])
        a = a[np.argsort(a[:, 0])]
    J = a[:, 0].astype(int)
    w = g_J(J) * (2 * J + 1) * np.exp(a[:, 1])
    return J, w / w.sum()


def mean_rot_temperature(J, P):
    """<T_rot> = beta * sum_J P_J B J(J+1)   (Eq. 5 of the paper), in K."""
    J = np.asarray(J)
    return BETA_CM_K * B_ROT_CM1 * float(np.sum(P * J * (J + 1)))


def fig3(shock):
    """z/D and n/n1 of the filled circles, z/D and T of the open circles, for shock 'A'..'E' (all flags)."""
    d, t = FIG3_DENSITY[shock], FIG3_TROT[shock]
    return d[:, 0], d[:, 1], t[:, 0], t[:, 1]


def absolute_density(shock):
    """z/D and number density in m^-3 of the Fig. 3 filled circles of a shock."""
    d = FIG3_DENSITY[shock]
    return d[:, 0], d[:, 1] * N1_ABS_1E22_M3[shock] * 1e22


if __name__ == "__main__":
    print("Phys. Rev. E 62, 4940 (2000) - digitised data")
    for k in "abc":
        J, x, y = fig4_series(k)
        s, i = np.polyfit(x, y, 1)
        Jp, P = populations(k, include_hidden=True)
        print("  Fig. 4 plot %s: %2d points, J = %d..%d; straight-line fit of all points T = %.1f K; "
              "sum g(2J+1)X = %.3e; Eq. (5) over the plotted levels: %.1f K"
              % (k, len(J), J.min(), J.max(), -BETA_CM_K * B_ROT_CM1 / s,
                 np.sum(g_J(J) * (2 * J + 1) * np.exp(y)), mean_rot_temperature(Jp, P)))
    J, x, y = fig4_series("b")
    for lab, sel in (("J <= 2", J <= 2), ("J >= 8", J >= 8)):
        s, i = np.polyfit(x[sel], y[sel], 1)
        print("  plot b, straight line through %s: %.1f K" % (lab, -BETA_CM_K * B_ROT_CM1 / s))
    for s in SHOCKS:
        d, t = FIG3_DENSITY[s], FIG3_TROT[s]
        print("  Fig. 3 %s: %d filled (flags 0/1/2: %d/%d/%d), %d open (%d/%d/%d); min n/n1 %.3f at z/D %.2f; "
              "min T %.1f K (printed %.1f); max n/n1 %.2f; dashed lines %.2f, %.2f; n1 = %.3fe22 m^-3; M1 = %.1f"
              % (s, len(d), *(int(np.sum(d[:, 2] == f)) for f in (0, 1, 2)), len(t), *(int(np.sum(t[:, 2] == f)) for f in (0, 1, 2)),
                 d[:, 1].min(), d[np.argmin(d[:, 1]), 0], t[:, 1].min(), FIG3_TROT_MIN_K[s][0], d[:, 1].max(),
                 *FIG3_DASHED_Z_OVER_D[s], N1_ABS_1E22_M3[s], FIG6_TABLE[s]["M1"]))
