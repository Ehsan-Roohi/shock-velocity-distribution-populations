"""Collision kernel from classical trajectories for use inside DSMC: outcomes of N2-N2 collisions on a grid of
pre-collision states given by the total energy of the pair and the shares of the two rotors.

Node = (Etot, x1, x2) with Etot = Et + e1 + e2 (K, conserved in the collision) and x1 = e1/Etot <= x2 = e2/Etot,
x1 + x2 <= 0.95. For each node N trajectories (definitions of ctc_kernel.py and ctc_table.py: rigid rotors with two
Lennard-Jones sites, random orientations and phases, impact parameter uniform in b^2 up to bmax, which depends on
Etot) are run. Stored per usable trajectory (float32): change of the two shares, dx1 = (e1' - e1)/Etot and
dx2 = (e2' - e2)/Etot, and the cosine of the deflection angle.

Usage: python ctc_ktable.py <out.npz> <first node> <number of nodes> [N=8000] [seed=1] [eps=.. sig=.. bond=..]
       python ctc_ktable.py list"""
import sys, os
import numpy as np

ETOT = [100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 12800.0]
BMAX = [8.0, 8.0, 7.0, 6.5, 6.0, 6.0, 6.0, 6.0]                      # A, for each total energy
X = [0.0, 0.005, 0.015, 0.035, 0.075, 0.15, 0.27, 0.42, 0.60, 0.80]
PAIRS = [(i, j) for i in range(len(X)) for j in range(i, len(X)) if X[i] + X[j] <= 0.95 + 1e-9]
NODES = [(k, i, j) for k in range(len(ETOT)) for (i, j) in PAIRS]
if sys.argv[1] == "list":
    print(len(NODES), "nodes;", len(PAIRS), "pairs of shares for each of", len(ETOT), "total energies"); sys.exit()
out, first, count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); kv = dict(a.split("=") for a in sys.argv[4:]); N = int(float(kv.pop("N", 8000))); seed = int(kv.pop("seed", 1))
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "ctc_kernel.py"), encoding="utf-8").read().split("T1, T2 = P[\"T1\"]")[0]
tab = open(os.path.join(here, "ctc_table.py"), encoding="utf-8").read(); tab = tab[tab.index("def advance"):tab.index("res = {}")]
sys.argv = ["ctc", "unused.npz"] + [f"{k}={v}" for k, v in kv.items()] + [f"seed={seed + first}"]
ns = {}; exec(compile(src, "ctc_kernel_defs", "exec"), ns); exec(compile(tab, "ctc_table_defs", "exec"), ns)
P, run_state = ns["P"], ns["run_state"]

res = {}
for q in range(first, min(first + count, len(NODES))):
    k, i, j = NODES[q]; E = ETOT[k]; e1, e2 = X[i] * E, X[j] * E; et = E - e1 - e2; P["bmax"] = BMAX[k]
    c, e1f, e2f, etf, b, drift, rfin = run_state(et, e1, e2, N)
    ok = (np.abs(drift) < 1e-3) & (rfin > 0)
    res[f"n{q}"] = np.stack([(e1f[ok] - e1) / E, (e2f[ok] - e2) / E, c[ok]]).astype(np.float32)
    hard = c[ok] < 0.9
    print(f"node {q}: Etot {E:g} K, shares {X[i]:g} {X[j]:g}, bmax {BMAX[k]:g} A; {N} trajectories, {N - ok.sum()} not usable (largest energy change {np.abs(drift[ok]).max():.1e}); "
          f"mean change of the shares {np.mean(e1f[ok] - e1) / E:+.4f} {np.mean(e2f[ok] - e2) / E:+.4f}; cross-section with cos(chi) < 0.9: {np.pi * BMAX[k] ** 2 * hard.mean():.1f} A^2", flush=True)
np.savez_compressed(out, etot=np.array(ETOT), bmax=np.array(BMAX), x=np.array(X), pairs=np.array(PAIRS), eps=P["eps"], sig=P["sig"], bond=P["bond"], **res)
