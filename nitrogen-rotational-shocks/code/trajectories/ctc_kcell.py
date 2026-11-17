"""Classical trajectories for a collision kernel with exact detailed balance (Markov kernel between cells of the
pre-collision state), for use inside DSMC.

State of a colliding pair at fixed total energy Etot = Et + e1 + e2 (conserved): the shares x1 = e1/Etot and
x2 = e2/Etot of the two rotors. The shares are binned with the edges XE; a cell is an ordered pair of bins (a, b).
For every total energy ETOT[k] and every source cell with a <= b, N trajectories are started from shares drawn inside
the cell with the density of colliding pairs at equilibrium, proportional to the translational share 1 - x1 - x2
(two rotational degrees of freedom per molecule, collision rate proportional to the relative speed), with
1 - x1 - x2 >= XTMIN. Stored per usable trajectory (float32): final shares x1', x2' and cos(chi).

The kernel is assembled from these files by kcell.py (transition matrix between cells, symmetrised with the
equilibrium weights of the cells).

Usage: python ctc_kcell.py <out.npz> <first job> <number of jobs> [N=2500] [seed=1] [eps=.. sig=.. bond=..]
       python ctc_kcell.py list                 (a job is one source cell at one total energy)"""
import sys, os
import numpy as np

ETOT = [200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0]
BMAX = [7.5, 7.0, 6.5, 6.0, 6.0, 6.0, 6.0]                       # A
XE = [0.0, 0.002, 0.005, 0.010, 0.018, 0.030, 0.045, 0.065, 0.090, 0.120, 0.155, 0.195, 0.240, 0.290, 0.345, 0.405, 0.470, 0.540, 0.620, 0.720, 0.850, 1.0]
XTMIN = 0.02
NB = len(XE) - 1
SRC = [(a, b) for a in range(NB) for b in range(a, NB) if XE[a] + XE[b] < 1 - XTMIN - 1e-9]
JOBS = [(k, s) for k in range(len(ETOT)) for s in range(len(SRC))]
if sys.argv[1] == "list":
    print(len(JOBS), "jobs:", len(SRC), "source cells for each of", len(ETOT), "total energies;", NB, "bins of the share"); sys.exit()
out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); kv = dict(a.split("=") for a in sys.argv[4:]); N = int(float(kv.pop("N", 2500))); seed = int(kv.pop("seed", 1))
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "ctc_kernel.py"), encoding="utf-8").read().split("T1, T2 = P[\"T1\"]")[0]
tab = open(os.path.join(here, "ctc_table.py"), encoding="utf-8").read(); tab = tab[tab.index("def advance"):tab.index("def run_state")]
sys.argv = ["ctc", "unused.npz"] + [f"{k}={v}" for k, v in kv.items()] + [f"seed={seed + first}"]
ns = {}; exec(compile(src, "ctc_kernel_defs", "exec"), ns); exec(compile(tab, "ctc_table_defs", "exec"), ns)
MU, P, I, rng, potential, rotor, unit, advance = (ns[k] for k in ("MU", "P", "I", "rng", "potential", "rotor", "unit", "advance"))


def sample_cell(a, b, n):
    """Shares inside the cell (a, b) with density proportional to 1 - x1 - x2, restricted to 1 - x1 - x2 >= XTMIN."""
    x1 = np.empty(n); x2 = np.empty(n); todo = np.ones(n, bool); top = 1 - XE[a] - XE[b]
    while todo.any():
        m = int(todo.sum()); u1 = XE[a] + (XE[a + 1] - XE[a]) * rng.random(m); u2 = XE[b] + (XE[b + 1] - XE[b]) * rng.random(m); xt = 1 - u1 - u2
        ok = (xt >= XTMIN) & (rng.random(m) * top < xt); w = np.where(todo)[0][ok]; x1[w] = u1[ok]; x2[w] = u2[ok]; todo[w] = False
    return x1, x2


def run(et, e1, e2, bmax, nsteps=3000, D=14.0, passes=14):
    n = len(et); g = np.sqrt(2 * et / MU); gh = np.zeros((n, 3)); gh[:, 0] = 1.0
    perp = unit(np.cross(gh, rng.normal(size=(n, 3)))); b = bmax * np.sqrt(rng.random(n))
    r = -D * gh + b[:, None] * perp; v = g[:, None] * gh
    n1, w1 = rotor(n, e1); n2, w2 = rotor(n, e2)
    E0 = et + e1 + e2 + potential(r, n1, n2); dt = (2 * D / g / nsteps)[:, None]
    y = advance([r, v, n1, w1, n2, w2], dt, nsteps)
    for _ in range(passes):
        rr = np.linalg.norm(y[0], axis=1); todo = ~((rr > 10.0) & ((y[0] * y[1]).sum(1) > 0))
        if not todo.any(): break
        sub = advance([a[todo] for a in y], dt[todo], nsteps)
        for a, s in zip(y, sub): a[todo] = s
    r, v, n1, w1, n2, w2 = y; rr = np.linalg.norm(r, axis=1); done = (rr > 10.0) & ((r * v).sum(1) > 0)
    e1f = 0.5 * I * (w1 * w1).sum(1); e2f = 0.5 * I * (w2 * w2).sum(1); etf = 0.5 * MU * (v * v).sum(1)
    drift = (etf + e1f + e2f + potential(r, n1, n2) - E0) / E0
    return v[:, 0] / np.linalg.norm(v, axis=1), e1f, e2f, etf, done & (np.abs(drift) < 1e-3)


res = {}
for q in range(first, min(first + count, len(JOBS))):
    k, s = JOBS[q]; a, b = SRC[s]; E = ETOT[k]; x1, x2 = sample_cell(a, b, N)
    c, e1f, e2f, etf, ok = run((1 - x1 - x2) * E, x1 * E, x2 * E, BMAX[k]); tot = e1f + e2f + etf
    res[f"j{q}"] = np.stack([e1f[ok] / tot[ok], e2f[ok] / tot[ok], c[ok]]).astype(np.float32)
    print(f"job {q}: Etot {E:g} K, cell ({a}, {b}) = shares [{XE[a]:g}, {XE[a + 1]:g}) x [{XE[b]:g}, {XE[b + 1]:g}); {N} trajectories, {N - int(ok.sum())} not usable; "
          f"mean change of the shares {np.mean(e1f[ok] / tot[ok] - x1[ok]):+.4f} {np.mean(e2f[ok] / tot[ok] - x2[ok]):+.4f}; cos(chi) < 0.9 in {np.mean(c[ok] < 0.9):.3f}", flush=True)
np.savez_compressed(out, etot=np.array(ETOT), bmax=np.array(BMAX), xe=np.array(XE), src=np.array(SRC), xtmin=XTMIN, eps=P["eps"], sig=P["sig"], bond=P["bond"], **res)
