"""Ab initio N2-N2 pair potential of R. Hellmann, Mol. Phys. 111, 387 (2013), version V_12^B (CCSD(T)/CBS with core,
relativistic and half of the CCSDT(Q) corrections), for rigid rotors. Five sites per molecule on the axis; for each
site pair
    V_ij(R) = A_ij exp(-alpha_ij R) - f6(b_ij R) C6_ij / R^6 + q_i q_j / R,
    f6(x) = 1 - exp(-x) sum_{k=0..6} x^k / k!   (Tang-Toennies),
summed over the 25 site pairs. Units: Angstrom and Kelvin (charges in sqrt(K A)); parameters from the supplementary
material of the paper. The rigid-rotor bond length of that work is the zero-point averaged 1.1014 A.

make(I, MU) returns (deriv, potential) with the same interface as the two-centre Lennard-Jones functions of
ctc_kernel.py: r = position of molecule 1 relative to molecule 2, n1, n2 = unit axes, w1, w2 = angular velocities."""
import math
import numpy as np

Z = np.array([-0.680065710389, -0.447763006688, 0.0, 0.447763006688, 0.680065710389])          # site positions on the axis
Q = np.array([-832.77884541, 1601.24507755, -1536.93246428, 1601.24507755, -832.77884541])     # site charges
TYPE = "ABCBA"
PAR = {("A", "A"): (0.973347918383e7, 3.06144571072, 2.58031350518, 0.298807116692e7),
       ("A", "B"): (-0.954555977809e7, 2.58710992361, 3.45760438302, -0.608284467163e7),
       ("A", "C"): (0.122158259267e8, 2.96686681629, 2.46746232590, 0.490318811890e7),
       ("B", "B"): (0.299460243665e7, 2.15319940621, 2.42577961527, 0.146889670654e8),
       ("B", "C"): (-0.819908034347e7, 2.84661195657, 2.02508542307, -0.129841807274e8),
       ("C", "C"): (0.163947777734e8, 2.99548316813, 1.97117981681, 0.107874613877e8)}
BOND = 1.1014
FACT = np.array([1.0 / math.factorial(k) for k in range(7)])


def _pair(i, j):
    a, b = TYPE[i], TYPE[j]
    A, al, bb, c6 = PAR[(a, b)] if (a, b) in PAR else PAR[(b, a)]
    return A, al, bb, c6, Q[i] * Q[j]


PAIRS = [(i, j) + _pair(i, j) for i in range(5) for j in range(5)]


def site_energy(R, A, al, bb, c6, qq):
    """V(R) and -(dV/dR)/R for one site pair (arrays of distances R)."""
    x = bb * R; ex = np.exp(-x)
    poly = FACT[0] + x * (FACT[1] + x * (FACT[2] + x * (FACT[3] + x * (FACT[4] + x * (FACT[5] + x * FACT[6])))))
    f6 = 1.0 - ex * poly; df6 = bb * ex * x ** 6 * FACT[6]                      # d f6 / dR
    r6 = R ** -6; rep = A * np.exp(-al * R)
    V = rep - f6 * c6 * r6 + qq / R
    dV = -al * rep - c6 * (df6 * r6 - 6.0 * f6 * r6 / R) - qq / (R * R)
    return V, -dV / R


def make(I, MU):
    def deriv(r, v, n1, w1, n2, w2):
        F = np.zeros_like(r); t1 = np.zeros_like(r); t2 = np.zeros_like(r)
        for i, j, A, al, bb, c6, qq in PAIRS:
            a = Z[i] * n1; b = Z[j] * n2; d = r + a - b; R = np.sqrt((d * d).sum(1))
            _, g = site_energy(R, A, al, bb, c6, qq); f = g[:, None] * d                     # force on the site of molecule 1
            F += f; t1 += np.cross(a, f); t2 -= np.cross(b, f)
        return v, F / MU, np.cross(w1, n1), t1 / I, np.cross(w2, n2), t2 / I

    def potential(r, n1, n2):
        V = np.zeros(len(r))
        for i, j, A, al, bb, c6, qq in PAIRS:
            d = r + Z[i] * n1 - Z[j] * n2; V += site_energy(np.sqrt((d * d).sum(1)), A, al, bb, c6, qq)[0]
        return V

    return deriv, potential


def configuration(R, th1, th2, phi):
    """r, n1, n2 for the centre-of-mass distance R and the angles (degrees) of the paper: theta1, theta2 between the
    molecular axes and the line of centres, phi the dihedral angle."""
    t1, t2, p = (np.radians(np.asarray(a, float)) for a in (th1, th2, phi)); n = len(t1)
    n1 = np.stack([np.sin(t1), np.zeros(n), np.cos(t1)], 1)
    n2 = np.stack([np.sin(t2) * np.cos(p), np.sin(t2) * np.sin(p), np.cos(t2)], 1)
    r = np.zeros((n, 3)); r[:, 2] = -np.asarray(R, float)                                    # molecule 2 at +R on the z axis
    return r, n1, n2
