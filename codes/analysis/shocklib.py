"""Shared tools for the normal-shock VDF study (reduced distributions g, h on a vx grid).

Conventions: rho1 = T1 = R = 1, u1 = M sqrt(gamma), gamma = 5/3. Lengths are in units of the
upstream relaxation length l1 = sqrt(R T1) mu1 / p1. g = int f dvy dvz, h = int (vy^2+vz^2) f dvy dvz.
"""
import json, os
import numpy as np
from scipy.optimize import least_squares, nnls

GAMMA = 5.0 / 3.0


def rh_states(M, g=GAMMA):
    r = (g + 1) * M ** 2 / ((g - 1) * M ** 2 + 2)
    p = (2 * g * M ** 2 - (g - 1)) / (g + 1)
    u1 = M * np.sqrt(g)
    return dict(rho1=1.0, u1=float(u1), T1=1.0, rho2=float(r), u2=float(u1 / r), T2=float(p / r), gamma=g, M1=float(M))


def moments(vx, wx, G, H):
    rho = G @ wx
    u = (G * vx) @ wx / rho
    c = vx[None, :] - u[:, None]
    Exx = (G * c ** 2) @ wx
    Eperp = H @ wx
    T = (Exx + Eperp) / (3 * rho)
    return dict(rho=rho, u=u, T=T, Tpar=Exx / rho, Tperp=Eperp / (2 * rho),
                q=0.5 * ((G * c ** 3) @ wx + (H * c) @ wx), sigma=Exx - rho * T,
                M300=(G * c ** 3) @ wx, M400=(G * c ** 4) @ wx - 3 * rho * T ** 2)


def load_case(path):
    z = np.load(path, allow_pickle=False)
    st = json.loads(str(z["states"]))
    d = dict(path=path, name=os.path.basename(path).replace("_reduced_vdf.npz", ""), st=st,
             x=z["x_mfp"].astype(float), vx=z["vx"].astype(float), wx=z["wx"].astype(float),
             G=z["G"].astype(float), H=z["H"].astype(float),
             model=str(z["model"]) if "model" in z.files else "BGK, constant tau")
    d.update(moments(d["vx"], d["wx"], d["G"], d["H"]))
    d["nstar"] = (d["rho"] - st["rho1"]) / (st["rho2"] - st["rho1"])
    return d


def xi_hat(v, st):
    return (v - st["u2"]) / (st["u1"] - st["u2"])


def maxwell1d(v, rho, u, T):
    return rho / np.sqrt(2 * np.pi * T) * np.exp(-(v - u) ** 2 / (2 * T))


def two_maxwellian_residual(case):
    """Best nonnegative combination of the two RH Maxwellians at every station (g only)."""
    st, v, w, G = case["st"], case["vx"], case["wx"], case["G"]
    sw = np.sqrt(w)
    A = np.stack([maxwell1d(v, st["rho1"], st["u1"], st["T1"]) * sw, maxwell1d(v, st["rho2"], st["u2"], st["T2"]) * sw], 1)
    eps = np.empty(len(G))
    for s in range(len(G)):
        b = G[s] * sw
        eps[s] = nnls(A, b)[1] / np.linalg.norm(b)
    return eps


# ---------------------------------------------------------------- population (bi-Maxwellian) fits

def _unpack(p):
    p = p.reshape(-1, 4)
    return np.exp(p[:, 0]), p[:, 1], np.exp(p[:, 2]), np.exp(p[:, 3])


def pop_model(p, v):
    n, u, a, b = _unpack(p)
    gk = n[:, None] * np.exp(-(v[None, :] - u[:, None]) ** 2 / (2 * a[:, None])) / np.sqrt(2 * np.pi * a[:, None])
    return gk, 2 * b[:, None] * gk


def _res_jac(p, v, sw, g, h, gn, hn):
    n, u, a, b = _unpack(p)
    gk, hk = pop_model(p, v)
    r = np.concatenate([(gk.sum(0) - g) * sw / gn, (hk.sum(0) - h) * sw / hn])
    K, Nv = gk.shape
    J = np.zeros((2 * Nv, 4 * K))
    d = v[None, :] - u[:, None]
    dg = [gk, gk * d / a[:, None], gk * (d ** 2 / (2 * a[:, None]) - 0.5)]
    for k in range(K):
        for j in range(3):
            J[:Nv, 4 * k + j] = dg[j][k] * sw / gn
            J[Nv:, 4 * k + j] = 2 * b[k] * dg[j][k] * sw / hn
        J[Nv:, 4 * k + 3] = hk[k] * sw / hn
    return r, J


def fit_populations(v, w, g, h, seeds, max_nfev=300):
    """Joint weighted L2 fit of g and h by K bi-Maxwellian populations. Returns (params, cost, res_g, res_h)."""
    sw = np.sqrt(w)
    gn, hn = np.linalg.norm(g * sw), np.linalg.norm(h * sw)
    best = None
    for p0 in seeds:
        try:
            r = least_squares(lambda p: _res_jac(p, v, sw, g, h, gn, hn)[0], p0,
                              jac=lambda p: _res_jac(p, v, sw, g, h, gn, hn)[1], method="lm", max_nfev=max_nfev)
        except Exception:
            continue
        if not np.all(np.isfinite(r.x)):
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return None
    gk, hk = pop_model(best.x, v)
    return best.x, best.cost, np.linalg.norm((gk.sum(0) - g) * sw) / gn, np.linalg.norm((hk.sum(0) - h) * sw) / hn


def pop_moments(p):
    """Analytic moments of the mixture and additive per-population contributions to q and sigma."""
    n, u, a, b = _unpack(p)
    rho = n.sum(); U = (n * u).sum() / rho; d = u - U
    T = (n * (a + d ** 2 + 2 * b)).sum() / (3 * rho)
    qk = 0.5 * n * d * (d ** 2 + 3 * a + 2 * b)
    sk = (2.0 / 3.0) * n * (a + d ** 2 - b)
    m4 = (n * (d ** 4 + 6 * d ** 2 * a + 3 * a ** 2)).sum() - 3 * rho * T ** 2
    m3 = (n * (d ** 3 + 3 * a * d)).sum()
    return dict(rho=rho, u=U, T=T, q=qk.sum(), sigma=sk.sum(), M300=m3, M400=m4, q_k=qk, sigma_k=sk)


def seed_templates(K, loc, st):
    """RH-informed initial guesses for K populations at a station with local (rho, u, T)."""
    r1, u1, T1, r2, u2, T2 = (st[k] for k in ("rho1", "u1", "T1", "rho2", "u2", "T2"))
    rho, U, T = loc
    ns = float(np.clip((rho - r1) / (r2 - r1), 0.02, 0.98))
    if K == 1:
        pops = [(rho, U, T, T)]
    else:
        pops = [(rho * ns, u2, T2, T2), (rho * (1 - ns), u1, T1, T1),
                (0.3 * rho, 0.5 * (u1 + u2) + 0.15 * (u1 - u2), 0.45 * T2, 0.45 * T2),
                (0.05 * rho, u2 - 0.8 * (u1 - u2), 0.4 * T2, 0.85 * T2),
                (0.05 * rho, U, 2 * T, T)][:K]
    return np.array([[np.log(max(n, 1e-8)), u, np.log(a), np.log(b)] for n, u, a, b in pops]).ravel()


def add_population(p, v, w, g, st):
    """Greedy seed for K+1: put a new population at the peak of the positive residual of g."""
    gk, _ = pop_model(p, v)
    r = g - gk.sum(0)
    i = int(np.argmax(r * np.sqrt(w)))
    mass = max(float(np.clip(r, 0, None) @ w), 1e-6 * float(g @ w))
    a = 0.25 * st["T2"]
    return np.concatenate([p, [np.log(0.5 * mass), v[i], np.log(a), np.log(a)]])


def sort_by_velocity(p):
    q = p.reshape(-1, 4)
    return q[np.argsort(q[:, 1])].ravel()
