"""Steady normal shock of the BGK model with constant relaxation time, solved for the reduced distributions.

    xi_x dg/dx = (g_M - g)/tau,   xi_x dh/dx = (h_M - h)/tau,   g_M = rho/sqrt(2 pi T) exp(-(xi_x-u)^2/2T),  h_M = 2 T g_M,

in units rho1 = T1 = R = tau = 1 (lengths in l1). Because the transverse velocities are integrated analytically, the
transverse temperature of the discrete Maxwellian is exact; the solutions of the three-dimensional velocity grid, which
this replaces at Mach 8, carry a quadrature error there (T_par/T_perp = 1.18 at the inflow).

Scheme: first-order upwind transport followed by exact exponential relaxation, marched in time from an existing
solution until the profiles stop changing. Output in the format of the *_reduced_vdf.npz files.

Usage: python bgk_reduced_shock.py <initial reduced npz> <out npz> [t_end] [nv]
"""
import sys, json, time
import numpy as np


def maxwell(rho, u, T, v):
    g = rho[:, None] / np.sqrt(2 * np.pi * T[:, None]) * np.exp(-(v[None, :] - u[:, None]) ** 2 / (2 * T[:, None]))
    return g, 2 * T[:, None] * g


def moments(g, h, v, w):
    rho = g @ w; u = (g @ (w * v)) / rho
    T = ((g * (v[None, :] - u[:, None]) ** 2 + h) @ w) / (3 * rho)
    return rho, u, T


def main(init, out, t_end=300.0, nv=321, xskip=1):
    z = np.load(init, allow_pickle=False); st = json.loads(str(z["states"]))
    x = z["x_mfp"].astype(float)[::xskip]; nx = len(x); dx = x[1] - x[0]       # xskip > 1: coarser spatial grid (resolution test)
    v0 = z["vx"].astype(float)
    v = np.linspace(v0.min(), v0.max(), nv); w = np.full(nv, v[1] - v[0]); w[0] *= 0.5; w[-1] *= 0.5
    g = np.array([np.interp(v, v0, r) for r in z["G"].astype(float)[::xskip]]); h = np.array([np.interp(v, v0, r) for r in z["H"].astype(float)[::xskip]])
    print(f"nx {nx} dx {dx:.5f} nv {nv} v in [{v[0]:.4f}, {v[-1]:.4f}] dv {v[1]-v[0]:.5f}", flush=True)
    one = np.array([1.0])
    gL, hL = maxwell(one * st["rho1"], one * st["u1"], one * st["T1"], v)
    gR, hR = maxwell(one * st["rho2"], one * st["u2"], one * st["T2"], v)
    dt = 0.9 * dx / np.abs(v).max(); cn = dt * v / dx; decay = np.exp(-dt); pos = v > 0
    nsteps = int(np.ceil(t_end / dt)); t0 = time.time()
    rho_prev = moments(g, h, v, w)[0]; hist = []; step_prev = 0
    for step in range(1, nsteps + 1):
        for f, fl, fr in ((g, gL, gR), (h, hL, hR)):
            left = np.vstack([fl, f[:-1]]); right = np.vstack([f[1:], fr])
            f -= cn[None, :] * np.where(pos[None, :], f - left, right - f)
        rho, u, T = moments(g, h, v, w)
        gm, hm = maxwell(rho, u, T, v)
        g *= decay; g += (1 - decay) * gm
        h *= decay; h += (1 - decay) * hm
        if step % 2000 == 0 or step == nsteps:
            # rate of change over the interval actually elapsed (the last interval is shorter than 2000 steps)
            rate = float(np.abs(rho - rho_prev).max() / ((step - step_prev) * dt)); rho_prev = rho.copy(); step_prev = step
            ns = (rho - 1) / (st["rho2"] - 1); xc = float(np.interp(0.5, ns, x))
            hist.append((step * dt, rate, xc))
            print(f"t {step*dt:8.2f}  max |d rho/dt| {rate:.3e}  x_c {xc:+.4f}  elapsed {time.time()-t0:6.0f}s", flush=True)
            if rate < 2e-8 and step * dt > 40:
                break
    rho, u, T = moments(g, h, v, w)
    c = v[None, :] - u[:, None]
    Tpar = (g * c ** 2) @ w / rho; Tperp = (h @ w) / (2 * rho)
    mass = g @ (w * v); mom = g @ (w * v * v); en = 0.5 * (g @ (w * v ** 3) + h @ (w * v))
    print("upstream T_par/T_perp", float(Tpar[20] / Tperp[20]), " downstream rho/RH, T/RH", float(rho[-20] / st["rho2"]), float(T[-20] / st["T2"]))
    print("flux deviations (%)", [float(100 * np.abs(a / a[10] - 1).max()) for a in (mass, mom, en)])
    np.savez_compressed(out, x_mfp=x, vx=v, wx=w, G=g, H=h, states=json.dumps(st), model="BGK, constant tau, reduced solver",
                        history=np.array(hist))
    print("written", out)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], float(a[3]) if len(a) > 3 else 300.0, int(a[4]) if len(a) > 4 else 321, int(a[5]) if len(a) > 5 else 1)
