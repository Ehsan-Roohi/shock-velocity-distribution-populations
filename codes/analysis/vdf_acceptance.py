"""Acceptance tests for the extracted normal-shock VDFs.

For each case (reduced VDF file) report:
  flux_dev_*   max_x |F(x) - F_RH| / |F_RH| for mass, momentum and energy fluxes, computed from
               G and H (a steady 1D shock must carry constant fluxes)
  rh_*         downstream density and temperature relative to Rankine-Hugoniot
  Tpar/Tperp   at the upstream and downstream boundaries (must be 1 in equilibrium)
  vx_edge      max over x of g at the two ends of the vx box, relative to max g at that station
  vperp_edge   max over the stored 3D stations of f on the outer transverse nodes, relative to max f
  back_excess  backward-moving mass at the upstream boundary minus that of the upstream Maxwellian,
               relative to rho1 (precursor reaching the inflow boundary)
  grad_ends    |d rho/dx| at both ends relative to its maximum (domain long enough?)
  pts_per_delta  grid points per maximum-slope density thickness; delta/lambda1
Usage: python vdf_acceptance.py <vdf_reduced dir> <out csv>
"""
import sys, os, json, csv
import numpy as np
from math import erf, sqrt

CASES = ["M1p5", "M2", "M2p5", "M3", "M4", "M5", "M6", "M6_gl3medium", "M6_gl3fine",
         "M8", "M12", "M12_gl2coarse", "M12_gl3medium"]


def check(path):
    z = np.load(path, allow_pickle=False)
    st = json.loads(str(z["states"]))
    vx, wx, G, H = z["vx"], z["wx"], z["G"], z["H"]
    x, rho, u = z["x_mfp"], z["rho"], z["ux"]
    r1, u1, T1, r2, T2 = st["rho1"], st["u1"], st["T1"], st["rho2"], st["T2"]
    J = G @ (vx * wx)
    P = G @ (vx ** 2 * wx)
    E = 0.5 * (G @ (vx ** 3 * wx) + H @ (vx * wx))
    Js, Ps, Es = r1 * u1, r1 * u1 ** 2 + r1 * T1, 0.5 * r1 * u1 ** 3 + 2.5 * r1 * T1 * u1
    c = vx[None, :] - u[:, None]
    Tpar = (G * c ** 2) @ wx / rho
    Tperp = (H @ wx) / (2 * rho)
    T = (Tpar + 2 * Tperp) / 3
    gmax = G.max(axis=1)
    vx_edge = max(float((G[:, 0] / gmax).max()), float((G[:, -1] / gmax).max()))
    F3 = z["F3_sel"].astype(np.float64)
    edge = np.concatenate([F3[:, :, [0, -1], :].reshape(len(F3), -1), F3[:, :, :, [0, -1]].reshape(len(F3), -1)], axis=1)
    vperp_edge = float((edge.max(axis=1) / F3.reshape(len(F3), -1).max(axis=1)).max())
    back = float(G[0][vx < 0] @ wx[vx < 0])
    back_maxw = r1 * 0.5 * (1 + erf((0 - u1) / sqrt(2 * T1)))
    drho = np.gradient(rho, x)
    delta = (r2 - r1) / np.abs(drho).max()
    dx = float(np.median(np.diff(x)))
    return dict(case=os.path.basename(path).split("_BGK")[0], mach=st["M1"],
                flux_dev_mass=float(np.abs(J / Js - 1).max()),
                flux_dev_momentum=float(np.abs(P / Ps - 1).max()),
                flux_dev_energy=float(np.abs(E / Es - 1).max()),
                rh_rho2=float(rho[-1] / r2), rh_T2=float(T[-1] / T2),
                Tpar_Tperp_up=float(Tpar[0] / Tperp[0]), Tpar_Tperp_down=float(Tpar[-1] / Tperp[-1]),
                vx_edge=vx_edge, vperp_edge=vperp_edge,
                back_excess=(back - back_maxw) / r1,
                grad_up=float(abs(drho[0]) / np.abs(drho).max()), grad_down=float(abs(drho[-1]) / np.abs(drho).max()),
                delta_over_lambda1=float(delta), pts_per_delta=float(delta / dx))


def main(src, out):
    rows = [check(os.path.join(src, f"{c}_BGK_reduced_vdf.npz")) for c in CASES
            if os.path.exists(os.path.join(src, f"{c}_BGK_reduced_vdf.npz"))]
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    hdr = ["case", "flux_dev_mass", "flux_dev_momentum", "flux_dev_energy", "rh_rho2", "rh_T2",
           "Tpar_Tperp_up", "vx_edge", "vperp_edge", "back_excess", "grad_up", "grad_down",
           "delta_over_lambda1", "pts_per_delta"]
    print(" ".join(f"{h[:13]:>13s}" for h in hdr))
    for r in rows:
        print(" ".join(f"{r[h]:>13s}" if isinstance(r[h], str) else f"{r[h]:13.3e}" for h in hdr))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
