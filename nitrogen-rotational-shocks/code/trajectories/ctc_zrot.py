"""Rotational collision number of the two-centre Lennard-Jones nitrogen model from classical trajectories, as a check
of the potential used in ctc_kernel.py. Both molecules have Maxwellian velocities at T and rotational energies at Tr < T;
the mean gain of rotational energy per collision gives the relaxation time through dTr/dt = n <g pi bmax^2 dEr>, and
Z = tau_rot / tau_c with dTr/dt = (T - Tr)/tau_rot and tau_c the VHS mean collision time (d_ref = 4.17 A at 273 K,
omega = 0.74). Usage: python ctc_zrot.py [T=300] [Tr=200] [N=40000] [seed=1] [eps=..] [sig=..] [bond=..] [bmax=..]"""
import sys, math, importlib.util, os
import numpy as np

args = dict(a.split("=") for a in sys.argv[1:]); T = float(args.pop("T", 300)); Tr = float(args.pop("Tr", 200)); N = int(float(args.pop("N", 40000))); seed = int(args.pop("seed", 1))
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "ctc_kernel.py"), encoding="utf-8").read().split("T1, T2 = P[\"T1\"]")[0]      # definitions only
sys.argv = ["ctc", "unused.npz"] + [f"{k}={v}" for k, v in args.items()] + [f"seed={seed}"]
ns = {}; exec(compile(src, "ctc_kernel_defs", "exec"), ns)
rng, MM, MU, P, traj = ns["rng"], ns["MM"], ns["MU"], ns["P"], ns["trajectories"]
tot_w = 0.0; tot_de = 0.0; tot_de2 = 0.0; done = 0; worst = 0.0
while done < N:
    n = min(20000, N - done)
    v1 = rng.normal(0, math.sqrt(T / MM), (n, 3)); v2 = rng.normal(0, math.sqrt(T / MM), (n, 3))
    e1 = -Tr * np.log(1 - rng.random(n)); e2 = -Tr * np.log(1 - rng.random(n)); g = v1 - v2; gm = np.linalg.norm(g, axis=1)
    gf, e1f, e2f, b, drift = traj(g, e1, e2)
    de = 0.5 * ((e1f - e1) + (e2f - e2)); tot_w += gm.sum(); tot_de += (gm * de).sum(); tot_de2 += (gm * de) ** 2 @ np.ones(n); done += n; worst = max(worst, np.abs(drift).max())
    print(done, "trajectories", flush=True)
gbar = tot_w / done; gde = tot_de / done; err = math.sqrt(max(tot_de2 / done - gde ** 2, 0) / done)
sig_b = math.pi * P["bmax"] ** 2
# VHS collision frequency per molecule: nu = n * sigma_T(T) * <g>-type factor; for VHS, nu = 4 n d_ref^2 sqrt(pi k T_ref/m) (T/T_ref)^(1-omega)
om = P["omega"]; dref, Tref = 4.17, 273.0
nu_over_n = 4 * dref ** 2 * math.sqrt(math.pi * Tref / MM) * (T / Tref) ** (1 - om)
rate_over_n = sig_b * gde                                      # dTr/dt / n   (two rotational degrees of freedom: Er = k Tr)
Z = (T - Tr) / rate_over_n * nu_over_n
print(f"T = {T} K, Tr = {Tr} K, {done} trajectories, bmax {P['bmax']} A, sites eps {P['eps']} K sigma {P['sig']} A bond {P['bond']} A; largest energy change {worst:.1e}")
print(f"mean gain of rotational energy per molecule, rate-weighted: {gde / gbar:.3f} K per collision inside bmax (relative statistical error {err / abs(gde):.2%})")
print(f"rotational collision number Z = tau_rot/tau_c(VHS) = {Z:.2f}")
