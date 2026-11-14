"""Steady-state check of the stored distributions through the integral (characteristic) form.

A steady relaxation-model shock satisfies  xi dg/dx = nu(x) (g_target - g),  so along a characteristic
    g(x, xi) = g(x_b, xi) exp(-s) + int g_target exp(-(s - s')) ds',   ds = nu dx / |xi|,
integrating from the upstream boundary for xi > 0 and from the downstream boundary for xi < 0.
The target and nu are built from the stored moments of the same solution, so the difference between
the reconstructed and the stored g, h measures departure from a steady solution of the kinetic model
(time dependence plus discretization error of the solver).

Models: BGK (nu = 1), Shakhov (nu = rho T^(1-omega), Pr = 2/3), ES-BGK (nu = rho/1.5, alpha = -1/2).
Usage: python integral_check.py <out dir> <reduced file> [...]
"""
import sys, os, csv
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl


def targets(c, kind, omega):
    v = c["vx"][None, :]
    rho, u, T, q, sig = (c[k][:, None] for k in ("rho", "u", "T", "q", "sigma"))
    cx = v - u
    if kind == "bgk":
        g = rho / np.sqrt(2 * np.pi * T) * np.exp(-cx ** 2 / (2 * T)); h = 2 * T * g
        nu = np.ones(len(c["rho"]))
    elif kind == "shakhov":
        gm = rho / np.sqrt(2 * np.pi * T) * np.exp(-cx ** 2 / (2 * T))
        A = (1 - 2.0 / 3.0) * q * cx / (5 * rho * T * T)
        g = gm * (1 + A * (cx ** 2 / T - 3)); h = 2 * T * gm * (1 + A * (cx ** 2 / T - 1))
        nu = c["rho"] * c["T"] ** (1 - omega)
    else:  # ES-BGK, alpha = -1/2
        al = -0.5
        Tp = T + al * sig / rho; Tt = T - al * sig / (2 * rho)
        g = rho / np.sqrt(2 * np.pi * Tp) * np.exp(-cx ** 2 / (2 * Tp)); h = 2 * Tt * g
        nu = c["rho"] / 1.5
    return g, h, nu


def reconstruct(x, v, S, nu, bL, bR):
    """Exponential integration with a source linear in optical depth on each cell."""
    out = np.empty_like(S)
    pos, neg = v > 1e-12, v < -1e-12
    zero = ~(pos | neg)
    out[:, zero] = S[:, zero]
    dx = np.diff(x); a_cell = 0.5 * (nu[1:] + nu[:-1]) * dx

    def step(fprev, s0, s1, a):
        e = np.exp(-a)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(a > 1e-8, (1 - e) / a, 1 - a / 2)
        return fprev * e + s0 * (p - e) + s1 * (1 - p)

    f = bL[pos].copy(); out[0, pos] = f
    for i in range(1, len(x)):
        f = step(f, S[i - 1, pos], S[i, pos], a_cell[i - 1] / v[pos]); out[i, pos] = f
    f = bR[neg].copy(); out[-1, neg] = f
    for i in range(len(x) - 2, -1, -1):
        f = step(f, S[i + 1, neg], S[i, neg], a_cell[i] / (-v[neg])); out[i, neg] = f
    return out


def main(out, files):
    os.makedirs(out, exist_ok=True)
    rows = []
    for p in files:
        c = sl.load_case(p)
        m = c["model"].lower()
        kind = "esbgk" if "es-bgk" in m else ("shakhov" if "shakhov" in m else "bgk")
        omega = 0.81 if "0.81" in m else 1.0
        gT, hT, nu = targets(c, kind, omega)
        Gi = reconstruct(c["x"], c["vx"], gT, nu, c["G"][0], c["G"][-1])
        Hi = reconstruct(c["x"], c["vx"], hT, nu, c["H"][0], c["H"][-1])
        w = c["wx"]
        eg = np.sqrt(((Gi - c["G"]) ** 2) @ w) / np.sqrt((c["G"] ** 2) @ w)
        eh = np.sqrt(((Hi - c["H"]) ** 2) @ w) / np.sqrt((c["H"] ** 2) @ w)
        core = (c["nstar"] > 0.01) & (c["nstar"] < 0.99)
        mi = sl.moments(c["vx"], w, Gi, Hi)
        dq = np.abs(mi["q"] - c["q"]).max() / np.abs(c["q"]).max()
        ds = np.abs(mi["sigma"] - c["sigma"]).max() / np.abs(c["sigma"]).max()
        np.savez(os.path.join(out, c["name"] + "_integral.npz"), x=c["x"], nstar=c["nstar"], eg=eg, eh=eh,
                 vx=c["vx"], G_int_sel=Gi[::max(1, len(Gi) // 200)], idx_sel=np.arange(len(Gi))[::max(1, len(Gi) // 200)])
        rows.append(dict(case=c["name"], model=c["model"], mach=c["st"]["M1"], g_resid_max=float(eg[core].max()),
                         g_resid_median=float(np.median(eg[core])), h_resid_max=float(eh[core].max()),
                         q_dev=float(dq), sigma_dev=float(ds)))
        r = rows[-1]
        print(f"{r['case'][:22]:22s} g max {100*r['g_resid_max']:6.2f}%  median {100*r['g_resid_median']:6.2f}%  h max {100*r['h_resid_max']:6.2f}%  q {100*dq:6.2f}%  sigma {100*ds:6.2f}%", flush=True)
    with open(os.path.join(out, "integral_check.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
