"""Model-free count of the rotational populations in the measured shock: singular values of the matrix of level
populations (stations x levels) of Robben & Talbot (1966), and a non-negative factorisation into two and three
components. The components are compared with a Boltzmann distribution at the free-stream temperature and with the
single-collision distribution of the classical trajectories (ctc_excite.py).

Levels above the last tabulated one follow the T' line of the tables, so the shape of the upper part of the hot
component is the measured T' by construction; the information is in the levels tabulated one by one.

Usage: python rt_modes.py <exc npz for Mach 7> <exc npz for Mach 12.9> [kmax=24]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robben_talbot_1966 as rt

KMAX = 24
for a in sys.argv[3:]:
    if a.startswith("kmax="): KMAX = int(a[5:])
TH = rt.THETA; K = np.arange(KMAX + 1); G = 2 * K + 1; EK = TH * K * (K + 1)


def levels(row):
    """Normalised populations N_K (K = 0..KMAX) of one station: tabulated levels, then the T' line."""
    X, Tp, dk, last = row; kp = len(dk); A = last * np.exp(TH * (kp - 0.5) * (kp + 0.5) / Tp)
    d = np.array([dk[k] if k < kp else A * np.exp(-EK[k] / Tp) for k in K]); N = G * d
    # molecules above KMAX (T' line, continuum estimate) are added to the normalisation only
    tail = A * Tp / TH * np.exp(-TH * (KMAX + 0.5) * (KMAX + 1.5) / Tp)
    return N / (N.sum() + tail), kp


def nmf(M, r, it=20000, seed=0):
    rng = np.random.default_rng(seed); W = rng.random((M.shape[0], r)) + 0.1; H = rng.random((r, M.shape[1])) + 0.1
    for _ in range(it):
        H *= (W.T @ M) / (W.T @ W @ H + 1e-30); W *= (M @ H.T) / (W @ H @ H.T + 1e-30)
    s = H.sum(1); return W * s, H / s[:, None]                       # components normalised to unit sum


def kernel(exc, T2):
    """Level populations of the single-collision distribution: share of sigma(e' > E) between the classical level edges."""
    d = np.load(exc); E = d["E"] * T2; f = d["ctc"] / d["ctc"][np.argmin(np.abs(d["E"] - 0.0))]
    edge = TH * np.arange(KMAX + 2) ** 2; c = np.interp(edge, E, f); p = c[:-1] - c[1:]
    return p / p.sum()


for key, exc in (("7", sys.argv[1]), ("12.9", sys.argv[2])):
    rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); T2 = ex[-1, 1]; en = (ex[:, 2] - rho1) / (rho2 - rho1)
    T1 = T2 / {"7": 10.469, "12.9": 33.30}[key]
    P = []; kps = []
    for row in rows:
        p, kp = levels(row); P.append(p); kps.append(kp)
    M = np.array(P); s = np.linalg.svd(M, compute_uv=False); e2 = s ** 2 / (s ** 2).sum()
    print(f"\nMach {key}: {len(rows)} stations, levels 0..{KMAX}; normalised density of the stations " + " ".join(f"{v:.2f}" for v in en))
    print("   singular values / first: " + " ".join(f"{v:.4f}" for v in s[:6] / s[0]) + "   share of the squared norm left after 1, 2, 3 modes: "
          + " ".join(f"{1 - e2[:k].sum():.2e}" for k in (1, 2, 3)))
    cold = G * np.exp(-EK / T1); cold /= cold.sum(); k1 = kernel(exc, T2); hot2 = G * np.exp(-EK / T2); hot2 /= hot2.sum()
    for r in (2, 3):
        W, H = nmf(M, r); res = np.linalg.norm(M - W @ H) / np.linalg.norm(M); o = np.argsort([(H[i] * EK).sum() for i in range(r)]); W, H = W[:, o], H[o]
        print(f"   {r} non-negative components: relative residual {res:.4f}; worst station {np.max(np.linalg.norm(M - W @ H, axis=1) / np.linalg.norm(M, axis=1)):.4f}")
        for i in range(r):
            me = (H[i] * EK).sum() / T2; hi = K >= 8
            slope = -1 / np.polyfit(EK[hi], np.log(np.maximum(H[i][hi] / G[hi], 1e-300)), 1)[0] / T2 if H[i][hi].sum() > 1e-3 else float("nan")
            print(f"      component {i + 1}: mean rotational energy {me:.3f} T2; slope temperature above K = 8: {slope:.2f} T2; populations K = 0..9: " + " ".join(f"{v:.3f}" for v in H[i][:10]))
            print(f"         its weight at the stations: " + " ".join(f"{v:.2f}" for v in W[:, i]))
    print("   reference shapes, K = 0..9:")
    print("      Boltzmann at T1          : " + " ".join(f"{v:.3f}" for v in cold[:10]) + f"   (mean {(cold * EK).sum() / T2:.3f} T2)")
    print("      one collision (trajectories): " + " ".join(f"{v:.3f}" for v in k1[:10]) + f"   (mean {(k1 * EK).sum() / T2:.3f} T2)")
    print("      Boltzmann at T2          : " + " ".join(f"{v:.3f}" for v in hot2[:10]))
    # two fixed shapes (cold stream + one-collision distribution): least-squares weights at every station and what is left
    B = np.stack([cold, k1], 1)
    print("   fit of every station with [Boltzmann at T1, one-collision distribution] and with [Boltzmann at T1, Boltzmann at T2]: weight of the second, relative residual")
    for j, row in enumerate(rows):
        if not (0.02 < en[j] < 0.97): continue
        w, *_ = np.linalg.lstsq(B, M[j], rcond=None); r1 = np.linalg.norm(M[j] - B @ w) / np.linalg.norm(M[j])
        B2 = np.stack([cold, hot2], 1); w2, *_ = np.linalg.lstsq(B2, M[j], rcond=None); r2 = np.linalg.norm(M[j] - B2 @ w2) / np.linalg.norm(M[j])
        print(f"      n* = {en[j]:.3f} (levels tabulated up to K = {kps[j] - 1}): one-collision {w[1]:.3f}, residual {r1:.3f} | Boltzmann T2 {w2[1]:.3f}, residual {r2:.3f} | hot group of the tables {ex[j, 4]:.3f}")
