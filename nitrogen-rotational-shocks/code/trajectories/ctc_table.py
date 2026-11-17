"""Table of classical-trajectory outcomes of N2-N2 collisions on a grid of pre-collision states, for building and
testing rules for the exchange of rotational energy.

State = (relative translational energy Et, rotational energies e1 <= e2), all in K. For each state N trajectories with
random orientations, rotation phases and impact parameter (uniform in b^2 up to bmax) are run with the definitions of
ctc_kernel.py (rigid rotors, two Lennard-Jones sites each). A trajectory starts at a separation of 14 A and is
integrated until the molecules are more than 10 A apart and receding (slow collisions that form a short-lived pair
are continued for up to 12 further passes). Stored per trajectory: cosine of the deflection angle, final rotational
energies, final relative translational energy, impact parameter, relative energy change, final separation.

Usage: python ctc_table.py <out.npz> <first state> <number of states> [N=4000] [seed=1] [eps=.. sig=.. bond=.. bmax=..]
       python ctc_table.py list            prints the grid"""
import sys, os
import numpy as np

ET = [100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]; ER = [3.0, 30.0, 100.0, 300.0, 800.0, 2000.0]
GRID = [(et, ER[i], ER[j]) for et in ET for i in range(len(ER)) for j in range(i, len(ER))]
if sys.argv[1] == "list":
    print(len(GRID), "states"); [print(k, s) for k, s in enumerate(GRID)]; sys.exit()
out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); kv = dict(a.split("=") for a in sys.argv[4:]); N = int(float(kv.pop("N", 4000))); seed = int(kv.pop("seed", 1))
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "ctc_kernel.py"), encoding="utf-8").read().split("T1, T2 = P[\"T1\"]")[0]
sys.argv = ["ctc", "unused.npz"] + [f"{k}={v}" for k, v in kv.items()] + [f"seed={seed + first}"]
ns = {}; exec(compile(src, "ctc_kernel_defs", "exec"), ns)
MU, P, I, rng, deriv, potential, rotor, unit = (ns[k] for k in ("MU", "P", "I", "rng", "deriv", "potential", "rotor", "unit"))


def advance(y, dt, nsteps):
    for _ in range(nsteps):
        k1 = deriv(*y)
        k2 = deriv(*[a + 0.5 * dt * k for a, k in zip(y, k1)])
        k3 = deriv(*[a + 0.5 * dt * k for a, k in zip(y, k2)])
        k4 = deriv(*[a + dt * k for a, k in zip(y, k3)])
        y = [a + dt / 6 * (p + 2 * q + 2 * s + u) for a, p, q, s, u in zip(y, k1, k2, k3, k4)]
        y[2] = unit(y[2]); y[4] = unit(y[4])
        y[3] -= (y[3] * y[2]).sum(1)[:, None] * y[2]; y[5] -= (y[5] * y[4]).sum(1)[:, None] * y[4]
    return y


def run_state(et, e1, e2, n, nsteps=3000, D=14.0, passes=12):
    g = np.sqrt(2 * et / MU); gh = np.zeros((n, 3)); gh[:, 0] = 1.0
    perp = unit(np.cross(gh, rng.normal(size=(n, 3)))); b = P["bmax"] * np.sqrt(rng.random(n))
    r = -D * gh + b[:, None] * perp; v = g * gh
    n1, w1 = rotor(n, np.full(n, e1)); n2, w2 = rotor(n, np.full(n, e2))
    E0 = et + e1 + e2 + potential(r, n1, n2); dt = 2 * D / g / nsteps
    y = advance([r, v, n1, w1, n2, w2], dt, nsteps)
    for _ in range(passes):
        rr = np.linalg.norm(y[0], axis=1); todo = ~((rr > 10.0) & ((y[0] * y[1]).sum(1) > 0))
        if not todo.any(): break
        sub = advance([a[todo] for a in y], dt, nsteps)
        for a, s in zip(y, sub): a[todo] = s
    r, v, n1, w1, n2, w2 = y; rr = np.linalg.norm(r, axis=1); done = (rr > 10.0) & ((r * v).sum(1) > 0)
    e1f = 0.5 * I * (w1 * w1).sum(1); e2f = 0.5 * I * (w2 * w2).sum(1); etf = 0.5 * MU * (v * v).sum(1)
    drift = (etf + e1f + e2f + potential(r, n1, n2) - E0) / E0
    return v[:, 0] / np.linalg.norm(v, axis=1), e1f, e2f, etf, b, drift, np.where(done, rr, 0.0)


res = {}
for k in range(first, min(first + count, len(GRID))):
    et, e1, e2 = GRID[k]; c, e1f, e2f, etf, b, drift, rfin = run_state(et, e1, e2, N)
    res[f"s{k}"] = np.stack([c, e1f, e2f, etf, b, drift, rfin]).astype(np.float32)
    ok = (np.abs(drift) < 1e-3) & (rfin > 0)
    print(f"state {k}: Et {et:g} K, e1 {e1:g} K, e2 {e2:g} K; {N} trajectories, {N - ok.sum()} not usable (largest energy change {np.abs(drift).max():.1e}); "
          f"mean change of e1 {np.mean(e1f[ok] - e1):+.2f} K, of e2 {np.mean(e2f[ok] - e2):+.2f} K per collision inside bmax = {P['bmax']} A", flush=True)
np.savez_compressed(out, grid=np.array(GRID), bmax=P["bmax"], eps=P["eps"], sig=P["sig"], bond=P["bond"], **res)
