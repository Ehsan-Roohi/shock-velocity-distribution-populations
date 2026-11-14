"""Robustness of the collision-history results.

(1) Labels against numerical parameters and domain length, Mach 8 (baseline vdf_cls/, dx = 0.1, 200 molecules
    per cell, upstream domain 60):
      vdf_conv2/A  hard sphere, half cell and half time step      vdf_conv2/B  hard sphere, 50 molecules per cell
      vdf_conv2/F  omega 0.81, half cell and half time step       vdf_conv2/E  omega 0.81, 50 molecules per cell
      vdf_dom/     hard sphere and omega 0.81 with an upstream domain of 30
(2) Scattering law: variable-soft-sphere runs with the same viscosity (vdf_vss/a1p4, vdf_vss/a2: classes and
    reduced files) against the VHS runs at the same Mach number and omega, and the linear cascade with the
    same law (cascade/vss_M*_w*_a*_s*.txt).

Usage: python robustness.py <shock_data dir>     -> analysis/out/robustness.json
"""
import sys, os, json, glob, re, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generations as G
import precursor_cascade as PC
import dsmc_table as DT

LO, HI = PC.LO, PC.HI


def gen_profiles(f):
    d = G.load(f); d["prof"] = G.profiles(d); return d


def own_once(d):
    """Mass fraction of stream molecules after exactly one collision on the n* grid."""
    n = d["mown"][:, :, 0].T
    return np.array([G.at_level(d, lv, n)[1] / G.at_level(d, lv, d["rho"]) for lv in G.GRID])


def precursor(f, A):
    d = PC.load_cls(f); x = d["x"]
    k = (np.arange(len(x)) < d["ic"]) & (d["beta"] < HI) & (d["beta"] > LO) & (x > x[0] + 5)
    fsc = d["st"]["u1"] ** (2 * d["om"] - 1); AN = A * d["N"]
    i3 = np.where(k)[0][np.argmin(np.abs(np.log(d["beta"][k]) - math.log(1e-3)))]
    return dict(rate_all=-PC.slope(AN, d["beta"], k) * fsc, rate_generation3=-PC.slope(AN, d["surv"], k) * fsc,
                share_generation3=float(d["surv"][i3] / d["beta"][i3]), share_stream_once=float(d["once"][i3] / d["beta"][i3]),
                length_two_decades=float(np.interp(-math.log(1e-2), -np.log(d["beta"][k])[::-1], x[k][::-1]) - np.interp(-math.log(1e-4), -np.log(d["beta"][k])[::-1], x[k][::-1])))


def main(root):
    out = dict(labels={}, vss={})
    base = {t: gen_profiles(os.path.join(root, "vdf_cls", f"DSMC_{t}_classes.npz")) for t in ("w0p5_M8", "w0p81_M8", "w0p81_M12")}
    for folder, lab in (("vdf_conv2/A", "half cell and time step"), ("vdf_conv2/B", "50 molecules per cell"), ("vdf_conv2/F", "half cell and time step"),
                        ("vdf_conv2/E", "50 molecules per cell"), ("vdf_dom", "upstream domain 30")):
        for f in sorted(glob.glob(os.path.join(root, folder, "DSMC_w*_M8_classes.npz"))):
            t = re.search(r"DSMC_(w.*_M.*)_classes", os.path.basename(f)).group(1)
            d = gen_profiles(f); b = base[t]
            out["labels"][f"{t} | {lab}"] = dict(max_abs_generation_fraction_difference=float(np.abs(d["prof"]["frac"] - b["prof"]["frac"]).max()),
                                                 by_generation=np.abs(d["prof"]["frac"] - b["prof"]["frac"]).max(0).round(4).tolist(),
                                                 max_abs_once_collided_difference=float(np.abs(own_once(d) - own_once(b)).max()),
                                                 max_abs_gen1_velocity_difference=float(np.abs(d["prof"]["xi"][:, 1] - b["prof"]["xi"][:, 1]).max()))
    for sub in sorted(glob.glob(os.path.join(root, "vdf_vss", "a*"))):
        al = float(os.path.basename(sub)[1:].replace("p", "."))
        fac = (al + 1) * (al + 2) / (6 * al)
        for f in sorted(glob.glob(os.path.join(sub, "DSMC_w*_M*_classes.npz"))):
            t = re.search(r"DSMC_(w.*_M.*)_classes", os.path.basename(f)).group(1)
            d = gen_profiles(f); b = base.get(t)
            om = d["omega"]; A0 = PC.vhs_A(om)
            rec = dict(alpha=al, mach=d["M"], omega=om, precursor=precursor(f, A0 * fac),
                       precursor_vhs=precursor(os.path.join(root, "vdf_cls", f"DSMC_{t}_classes.npz"), A0),
                       fractions_at_levels={f"{lv}": d["prof"]["frac"][int(np.argmin(np.abs(G.GRID - lv)))].round(3).tolist() for lv in (0.1, 0.2, 0.35, 0.5)},
                       gen1_xi_at_levels={f"{lv}": float(d["prof"]["xi"][int(np.argmin(np.abs(G.GRID - lv))), 1]) for lv in (0.1, 0.2, 0.5)},
                       gen1_Tpar_at_levels={f"{lv}": float(d["prof"]["Tpar"][int(np.argmin(np.abs(G.GRID - lv))), 1]) for lv in (0.1, 0.2, 0.5)})
            if b is not None:
                rec["max_abs_generation_fraction_difference_from_vhs"] = float(np.abs(d["prof"]["frac"] - b["prof"]["frac"]).max())
                rec["vhs_fractions_at_levels"] = {f"{lv}": b["prof"]["frac"][int(np.argmin(np.abs(G.GRID - lv)))].round(3).tolist() for lv in (0.1, 0.2, 0.35, 0.5)}
                rec["vhs_gen1_xi_at_levels"] = {f"{lv}": float(b["prof"]["xi"][int(np.argmin(np.abs(G.GRID - lv))), 1]) for lv in (0.1, 0.2, 0.5)}
            red = os.path.join(sub, f"DSMC_{t}_reduced_vdf.npz")
            if os.path.exists(red):
                _, r = DT.record(root, red); rec["table"] = {k: r[k] for k in ("delta", "TparTperp_max", "rh_pair_max", "beta_n05", "beta_n20")}
                vh = os.path.join(root, "vdf_dsmc", f"DSMC_{t}_reduced_vdf.npz")
                if os.path.exists(vh):
                    _, r0 = DT.record(root, vh); rec["table_vhs"] = {k: r0[k] for k in ("delta", "TparTperp_max", "rh_pair_max", "beta_n05", "beta_n20")}
            c = PC.load_casc(os.path.join(root, "cascade", f"vss_M{d['M']:g}_w{om:g}_a{al:g}_s*.txt"))
            if c is not None:
                AN = -A0 * fac * c["x"]; k = (c["all"] < HI) & (c["all"] > LO); fsc = d["st"]["u1"] ** (2 * om - 1)
                rec["cascade"] = dict(rate_all=-PC.slope(AN, c["all"], k) * fsc, rate_primaries=-PC.slope(AN, c["prim"], k) * fsc,
                                      share_recoil=float(c["recoil"][k].sum() / c["all"][k].sum()), mean_speed_over_u1=float(c["speed"][k].mean() / d["st"]["u1"]))
            out["vss"][f"{t} | alpha {al:g}"] = rec
    json.dump(out, open(os.path.join(root, "analysis", "out", "robustness.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
