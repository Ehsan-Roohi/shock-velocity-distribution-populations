"""Measured rotational distributions of nitrogen in a shock wave: Robben & Talbot, Phys. Fluids 9, 653 (1966),
Tables I (Mach 7.0, L* = 4.70 X) and II (Mach 12.9, L* = 4.23 X), transcribed from the paper.
Each row: X (turns), T' (K) of the straight high-K part, list of 10^{D_K} for K = 0, 1, 2, ..., and the last entry
10^{D_{K'-1/2}} at K' - 1/2 (the lower limit of the part represented by T'). N_K = (2K+1) 10^{D_K} (their Eq. 7).
Density and rotational energy follow their Eqs. (9) and (11).

Usage: python robben_talbot_1966.py            (prints density, rotational temperature and the two-group split)"""
import numpy as np

THETA = 2.88     # K, characteristic rotational temperature of nitrogen

# X, T', [10^D_K for K = 0..K'-1], value at K' - 1/2
M7 = [
    (3.0, 46, [163, 138, 98, 60, 32, 14.6, 7.1], 4.6),
    (4.0, 58, [163, 138, 98, 60, 32, 14.6, 7.1], 4.6),
    (4.5, 130, [150, 112, 95, 59, 32, 17, 8.8, 4.7, 2.63], 2.04),
    (5.0, 240, [124, 111, 88, 62, 38, 21.6, 12.9, 8.7, 6.1, 4.3, 3.0, 2.13, 1.61], 1.42),
    (5.25, 240, [117, 105, 86, 63, 42, 26.4, 17.6, 12.9, 9.6, 7.3, 5.37], 4.51),
    (5.5, 235, [111, 105, 82, 61.4, 44.7, 33.3, 24.5, 19.5, 14.8], 12.9),
    (5.75, 247, [98, 91, 78, 62, 50, 41, 33.1, 27.2], 24.5),
    (6.0, 252, [95, 91, 81, 69, 59, 52], 47.4),
    (6.25, 270, [90, 88, 81, 74], 70.8),
    (6.5, 280, [], 93.3),
    (6.75, 292, [], 92.3),
    (7.0, 293, [], 97.3),
    (8.0, 296, [], 96),
]
M129 = [
    (3.0, 65, [400, 184, 77, 29, 10.3, 3.7, 1.63], 1.12),
    (3.5, 70, [400, 184, 77, 29, 11.5, 4.7, 2.46], 1.78),
    (4.0, 260, [355, 221, 93, 37.3, 14.8, 7.16, 4.41, 2.88, 2.09], 1.83),
    (4.5, 260, [272, 190, 92, 44, 22.4, 12.5, 8.2, 6.12, 4.55], 3.98),
    (4.75, 260, [252, 171, 79, 44, 26.3, 16.6, 11.6, 8.5, 6.7], 5.65),
    (5.0, 260, [184, 140, 82, 47, 32.4, 23, 16, 12.3], 11.2),
    (5.25, 261, [170, 127, 71, 57, 41, 31, 24, 17.5], 15.9),
    (5.5, 265, [115, 98, 74, 61, 50, 41, 33.5, 28.2], 25.8),
    (5.75, 271, [96, 89, 78, 68, 60, 53, 44.6, 37.3], 33.9),
    (6.0, 282, [90, 86, 80, 73, 66, 59], 55.2),
    (6.25, 293, [], 86.1),
    (6.5, 302, [], 89.1),
    (7.0, 305, [], 90.2),
]
CASES = {"7": dict(rows=M7, Lstar_per_turn=4.70, mach=7.0), "12.9": dict(rows=M129, Lstar_per_turn=4.23, mach=12.9)}


def reduce(rows):
    """Density (arbitrary units), rotational energy per molecule (K), T', and the hot-group fraction: the part of
    the molecules represented by a Boltzmann distribution at T' extended to K = 0, from the value at K' - 1/2."""
    out = []
    for X, Tp, d, last in rows:
        K = np.arange(len(d)); d = np.array(d, float); kp = len(d)
        e = lambda k: THETA * k * (k + 1)
        rho_low = ((2 * K + 1) * d).sum(); E_low = (e(K) * (2 * K + 1) * d).sum()
        rho_hi = (Tp / THETA) * last
        E_hi = (THETA * (kp - 0.5) * (kp + 0.5) + Tp) * (Tp / THETA) * last
        rho = rho_low + rho_hi; ER = E_low + E_hi
        # hot group: N_K = (2K+1) A exp(-theta K(K+1)/T'), A fixed by the value at K' - 1/2; its total number is A T'/theta
        A = last * np.exp(THETA * (kp - 0.5) * (kp + 0.5) / Tp)
        out.append((X, Tp, rho, ER / rho, min(A * Tp / THETA / rho, 1.0)))
    return np.array(out)


if __name__ == "__main__":
    for name, c in CASES.items():
        r = reduce(c["rows"]); rho1 = r[0, 2]; rho2 = r[-3:, 2].mean(); T2 = r[-1, 1]
        print(f"Mach {name}: downstream T' = {T2:.0f} K, density ratio from the table {rho2 / rho1:.2f}")
        print("   X   x/L*   T'/T2  (rho-rho1)/(rho2-rho1)  T_R/T2  hot-group fraction")
        for X, Tp, rho, TR, fh in r:
            print(f"  {X:5.2f} {X * c['Lstar_per_turn']:6.2f}  {Tp / T2:5.3f}   {(rho - rho1) / (rho2 - rho1):6.3f}   {TR / T2:6.3f}   {fh:6.3f}")
