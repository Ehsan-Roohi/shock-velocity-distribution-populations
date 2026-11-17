"""Classical-trajectory calculation of single N2-N2 collisions under the conditions of the shock precursor, to obtain,
independently of any exchange rule, the rotational energy that a cold stream molecule carries after the collision
that sends it upstream.

Molecules: rigid rotors with two Lennard-Jones sites each (two-centre model; site-site eps/k, sigma and bond length
are inputs; defaults 37.3 K, 3.31 A, 1.09 A). Units: mass in amu, length in A, energy in K (k = 1), so the time unit is
1.0967 ps and the velocity unit 91.18 m/s.

Ensemble (laboratory frame of the shock, x downstream):
  molecule 1: stream molecule, velocity (u1, 0, 0) + thermal at T1, rotation at T1;
  molecule 2: upstream-moving molecule, half-range Maxwellian (vx < 0) at Th, rotation at T1 with probability pcold and
              at Trh otherwise;
  impact parameter uniform in b^2 up to bmax, orientations and rotation phases random; each trajectory is weighted by
  the relative speed (collision rate).
Reported for the collisions after which molecule 1 moves upstream (vx' < 0): the distribution of its rotational
energy in units of T2, for the trajectories and, on the same pre-collision pairs, for the Larsen-Borgnakke rule
(probability 1/zrot) and for the Pullin rule (active fraction phi) with isotropic scattering.

Usage: python ctc_kernel.py <out.npz> [N=100000] [seed=1] [key=value ...]
Keys: T1, T2, u1 (m/s), Th, Trh, pcold, eps, sig, bond, bmax, zrot, phi, omega, pot (1: ab initio potential of
Hellmann, see n2_hellmann.py); numerical parameters for accuracy
tests: nsteps (integration steps per trajectory, default 3000) and D (half-length of the path in A, default 14)."""
import sys, math
import numpy as np

P = dict(T1=9.16, T2=305.0, u1=797.0, Th=442.0, Trh=400.0, pcold=0.55, eps=37.3, sig=3.31, bond=1.09, bmax=5.5, zrot=5.0, phi=0.2635, omega=0.74, nsteps=3000.0, D=14.0)
out = sys.argv[1]; N = 100000; seed = 1
for a in sys.argv[2:]:
    k, v = a.split("=")
    if k == "N": N = int(float(v))
    elif k == "seed": seed = int(v)
    else: P[k] = float(v)
rng = np.random.default_rng(seed)
VU = 91.18                                # m/s per velocity unit
MA = 14.0; MM = 28.0; MU = 14.0           # atom, molecule, reduced mass (amu)
L = P["bond"]; I = MA * L * L / 2; EPS, SIG2 = P["eps"], P["sig"] ** 2
SGN = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def unit(n):
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def deriv(r, v, n1, w1, n2, w2):
    F = np.zeros_like(r); t1 = np.zeros_like(r); t2 = np.zeros_like(r)
    for sa, sb in SGN:
        a = 0.5 * sa * L * n1; b = 0.5 * sb * L * n2
        d = r + a - b; d2 = (d * d).sum(1); s6 = (SIG2 / d2) ** 3
        f = (24 * EPS * (2 * s6 * s6 - s6) / d2)[:, None] * d            # force on the site of molecule 1
        F += f; t1 += np.cross(a, f); t2 -= np.cross(b, f)
    return v, F / MU, np.cross(w1, n1), t1 / I, np.cross(w2, n2), t2 / I


def potential(r, n1, n2):
    V = np.zeros(len(r))
    for sa, sb in SGN:
        d = r + 0.5 * sa * L * n1 - 0.5 * sb * L * n2; s6 = (SIG2 / (d * d).sum(1)) ** 3
        V += 4 * EPS * (s6 * s6 - s6)
    return V


if P.get("pot", 0.0) == 1.0:     # pot=1: ab initio N2-N2 potential of Hellmann (2013), version B; use with bond=1.1014
    import n2_hellmann
    deriv, potential = n2_hellmann.make(I, MU)


def rotor(n, erot):
    """Orientation and angular velocity (perpendicular to the axis) for a given rotational energy."""
    ax = unit(rng.normal(size=(n, 3))); p = unit(np.cross(ax, rng.normal(size=(n, 3))))
    return ax, p * np.sqrt(2 * erot / I)[:, None]


def trajectories(g, e1, e2, nsteps=3000, D=14.0):
    n = len(g); gm = np.linalg.norm(g, axis=1); gh = g / gm[:, None]
    perp = unit(np.cross(gh, rng.normal(size=(n, 3)))); b = P["bmax"] * np.sqrt(rng.random(n))
    r = -D * gh + b[:, None] * perp; v = g.copy()
    n1, w1 = rotor(n, e1); n2, w2 = rotor(n, e2)
    E0 = 0.5 * MU * gm ** 2 + e1 + e2 + potential(r, n1, n2)
    dt = (2 * D / gm / nsteps)[:, None]
    y = [r, v, n1, w1, n2, w2]
    for _ in range(nsteps):
        k1 = deriv(*y)
        k2 = deriv(*[a + 0.5 * dt * k for a, k in zip(y, k1)])
        k3 = deriv(*[a + 0.5 * dt * k for a, k in zip(y, k2)])
        k4 = deriv(*[a + dt * k for a, k in zip(y, k3)])
        y = [a + dt / 6 * (p + 2 * q + 2 * s + u) for a, p, q, s, u in zip(y, k1, k2, k3, k4)]
        y[2] = unit(y[2]); y[4] = unit(y[4])
        y[3] -= (y[3] * y[2]).sum(1)[:, None] * y[2]; y[5] -= (y[5] * y[4]).sum(1)[:, None] * y[4]
    r, v, n1, w1, n2, w2 = y
    e1f = 0.5 * I * (w1 * w1).sum(1); e2f = 0.5 * I * (w2 * w2).sum(1)
    E1 = 0.5 * MU * (v * v).sum(1) + e1f + e2f + potential(r, n1, n2)
    return v, e1f, e2f, b, (E1 - E0) / E0


T1, T2 = P["T1"], P["T2"]; res = {k: [] for k in ("v1x", "e1", "e2", "b", "w", "drift", "e1_0", "e2_0", "cchi", "gout", "lb_v1x", "lb_e1", "pu_v1x", "pu_e1", "wv")}
done = 0; BATCH = 20000; eta = 2.5 - P["omega"]
while done < N:
    n = min(BATCH, N - done)
    v1 = rng.normal(0, math.sqrt(T1 / MM), (n, 3)); v1[:, 0] += P["u1"] / VU
    v2 = rng.normal(0, math.sqrt(P["Th"] / MM), (n, 3)); v2[:, 0] = -np.abs(v2[:, 0])
    e1 = -T1 * np.log(1 - rng.random(n))
    cold = rng.random(n) < P["pcold"]; e2 = np.where(cold, -T1 * np.log(1 - rng.random(n)), -P["Trh"] * np.log(1 - rng.random(n)))
    g = v1 - v2; gm = np.linalg.norm(g, axis=1); vcm = 0.5 * (v1 + v2)
    gf, e1f, e2f, b, drift = trajectories(g, e1, e2, nsteps=int(P["nsteps"]), D=P["D"])
    res["v1x"].append((vcm + 0.5 * gf)[:, 0]); res["e1"].append(e1f); res["e2"].append(e2f); res["b"].append(b); res["w"].append(gm)
    res["drift"].append(drift); res["e1_0"].append(e1); res["e2_0"].append(e2)
    gfm = np.linalg.norm(gf, axis=1); res["cchi"].append((g * gf).sum(1) / (gm * gfm)); res["gout"].append(gfm)
    # the two exchange rules on the same pairs, isotropic scattering, VHS rate weight g^(2-2 omega)
    et = 0.5 * MU * gm ** 2
    mu = 2 * rng.random(n) - 1; ph = 2 * math.pi * rng.random(n); sq = np.sqrt(1 - mu ** 2); ehat = np.stack([mu, sq * np.cos(ph), sq * np.sin(ph)], 1)
    # Larsen-Borgnakke, each partner with probability 1/zrot
    a1, a2, at = e1.copy(), e2.copy(), et.copy()
    for arr in (a1, a2):
        ch = rng.random(n) < 1 / P["zrot"]; ec = at[ch] + arr[ch]; new = ec * (1 - rng.random(ch.sum()) ** (1 / eta)); arr[ch] = new; at[ch] = ec - new
    res["lb_v1x"].append((vcm + 0.5 * np.sqrt(2 * at / MU)[:, None] * ehat)[:, 0]); res["lb_e1"].append(a1)
    # Pullin
    f = P["phi"]; s1 = rng.beta(f, 1 - f, n); s2 = rng.beta(f, 1 - f, n); s3 = rng.beta(f * eta, (1 - f) * eta, n)
    pool = s1 * e1 + s2 * e2 + s3 * et; s5 = rng.beta(2 * f, f * eta, n); s4 = rng.beta(f, f, n)
    p1 = (1 - s1) * e1 + s4 * s5 * pool; pt = (1 - s3) * et + (1 - s5) * pool
    res["pu_v1x"].append((vcm + 0.5 * np.sqrt(2 * pt / MU)[:, None] * ehat)[:, 0]); res["pu_e1"].append(p1)
    res["wv"].append(gm ** (2 - 2 * P["omega"]))
    done += n
    print(f"{done} trajectories; largest relative energy change in this batch {np.abs(drift).max():.2e}", flush=True)
R = {k: np.concatenate(v) for k, v in res.items()}
np.savez_compressed(out, **R, **{"par_" + k: v for k, v in P.items()})


def summary(name, vx, e, w):
    m = vx < 0; ww = w[m]; ee = e[m] / T2
    q = lambda c: float((ww * (ee < c)).sum() / ww.sum())
    print(f"{name:22s}: recoil share {(w * m).sum() / w.sum():.4f} of the weighted collisions; rotational energy of the recoiled stream molecule / T2: "
          f"mean {(ww * ee).sum() / ww.sum():.3f}; fraction below 0.1: {q(0.1):.3f}, below 0.3: {q(0.3):.3f}, below 1: {q(1.0):.3f}, above 2: {1 - q(2.0):.3f}")


print(f"conditions: T1 {T1} K, T2 {T2} K, u1 {P['u1']} m/s; T1/T2 = {T1 / T2:.4f}; sites eps {EPS} K, sigma {P['sig']} A, bond {L} A; mean relative energy {np.mean(0.5 * MU * R['w'] ** 2) / T2:.2f} T2")
summary("classical trajectories", R["v1x"], R["e1"], R["w"])
summary("Larsen-Borgnakke", R["lb_v1x"], R["lb_e1"], R["wv"])
summary("Pullin", R["pu_v1x"], R["pu_e1"], R["wv"])
