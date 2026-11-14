"""Convert the reduced-distribution checkpoints of the other kinetic models (JFM campaign) to the
common format used for the BGK sweep, and report acceptance numbers for every case.

Usage: python convert_models.py <models_jfm dir> <vdf_reduced dir (BGK)> <out dir> <acceptance csv>
"""
import sys, os, json, csv, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shocklib import rh_states, moments, load_case

SH = dict(M2=("Shakhov_M2_x1280", 2, 1280, 30, 321, 16), M3=("Shakhov_M3_x1920", 3, 1920, 45, 801, 40),
          M4=("Shakhov_M4_x2560", 4, 2560, 60, 801, 40), M5=("Shakhov_M5_x3200", 5, 3200, 75, 801, 40))
ES = dict(M2=("ESBGK_M2_v2", 2, 640, 30, 801, 40), M3=("ESBGK_M3_v2", 3, 960, 45, 801, 40))


def grid(nx, L, nv, vmax):
    v = np.linspace(-vmax, vmax, nv); w = np.full(nv, v[1] - v[0]); w[[0, -1]] *= 0.5
    dx = 2 * L / nx
    return -L + (np.arange(nx) + 0.5) * dx, v, w


def save(out, name, model, M, x, v, w, G, H, K=None):
    st = rh_states(M)
    m = moments(v, w, G, H)
    np.savez(os.path.join(out, f"{name}_reduced_vdf.npz"), x_mfp=x, vx=v, wx=w, G=G, H=H,
             K=K if K is not None else np.zeros_like(G), states=json.dumps(st), mach=float(M), model=model,
             nstar=(m["rho"] - 1) / (st["rho2"] - 1))


def acceptance(c):
    st, v, w, G, H, x = c["st"], c["vx"], c["wx"], c["G"], c["H"], c["x"]
    r1, u1, T1 = st["rho1"], st["u1"], st["T1"]
    J = G @ (v * w); P = G @ (v ** 2 * w); E = 0.5 * (G @ (v ** 3 * w) + H @ (v * w))
    Js, Ps, Es = r1 * u1, r1 * u1 ** 2 + r1 * T1, 0.5 * r1 * u1 ** 3 + 2.5 * r1 * T1 * u1
    gabs = np.abs(G).max(1)
    drho = np.gradient(c["rho"], x); delta = (st["rho2"] - r1) / np.abs(drho).max()
    neg = np.clip(-G, 0, None) @ w / c["rho"]
    return dict(case=c["name"], model=c["model"], mach=st["M1"],
                flux_mass=float(np.abs(J / Js - 1).max()), flux_mom=float(np.abs(P / Ps - 1).max()),
                flux_energy=float(np.abs(E / Es - 1).max()),
                rho2_over_RH=float(c["rho"][-1] / st["rho2"]), T2_over_RH=float(c["T"][-1] / st["T2"]),
                TparTperp_up=float(c["Tpar"][0] / c["Tperp"][0]),
                vx_edge=float(max((np.abs(G[:, 0]) / gabs).max(), (np.abs(G[:, -1]) / gabs).max())),
                neg_mass_max=float(neg.max()), delta_over_l1=float(delta),
                pts_per_delta=float(delta / np.median(np.diff(x))),
                TparTperp_max=float((c["Tpar"] / c["Tperp"]).max()))


def main(src, bgk, out, csvpath):
    os.makedirs(out, exist_ok=True)
    for tag, (stem, M, nx, L, nv, vmax) in SH.items():
        f = np.load(os.path.join(src, stem + "_checkpoint.npz"))["f"]
        x, v, w = grid(nx, L, nv, vmax)
        save(out, f"Shakhov_{tag}", "Shakhov, Pr=2/3, mu~T", M, x, v, w, f[0], f[1], f[2])
    for tag, (stem, M, nx, L, nv, vmax) in ES.items():
        f = np.load(os.path.join(src, stem + "_checkpoint.npz"))["f"]
        x, v, w = grid(nx, L, nv, vmax)
        save(out, f"ESBGK_{tag}", "ES-BGK, Pr=2/3, mu~T", M, x, v, w, f[0], f[1], f[2])
    z = np.load(os.path.join(src, "M8_shakhov_w081_checkpoint.npz"))
    om = 0.81; mu0 = 30 * np.sqrt(2 * np.pi) / (4 * (7 - 2 * om) * (5 - 2 * om))
    v = np.linspace(-40, 40, 401); w = np.full(401, v[1] - v[0]); w[[0, -1]] *= 0.5
    xv = -50 + (np.arange(1200) + 0.5) * 80 / 1200
    save(out, "Shakhov_w081_M8", "Shakhov, Pr=2/3, mu~T^0.81", 8, xv / mu0, v, w, z["g"], z["h"])
    rows = []
    for p in sorted(glob.glob(os.path.join(bgk, "*_reduced_vdf.npz"))) + sorted(glob.glob(os.path.join(out, "*_reduced_vdf.npz"))):
        rows.append(acceptance(load_case(p)))
    with open(csvpath, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    keys = ["flux_mass", "flux_mom", "flux_energy", "T2_over_RH", "TparTperp_up", "vx_edge", "neg_mass_max", "delta_over_l1", "pts_per_delta", "TparTperp_max"]
    print(f"{'case':22s} " + " ".join(f"{k[:11]:>11s}" for k in keys))
    for r in rows:
        print(f"{r['case'][:22]:22s} " + " ".join(f"{r[k]:11.3e}" for k in keys))


if __name__ == "__main__":
    main(*sys.argv[1:5])
