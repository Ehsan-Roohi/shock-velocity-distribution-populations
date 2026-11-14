"""Accuracy of the predictions placed beside the accuracy of the DSMC reference.

For each case (Mach 25 hard spheres; Mach 12 hard spheres and omega = 1; Mach 25 omega = 1 without resolution runs):
  prediction - reference   methods P, D, H (m25_prediction_figure.predict -> methods_figure.case)
  scatter of the reference from its four time blocks (block_rho, block_q, block_Tpar of the reduced files)
  change of the reference  half cell and half time step; 400 molecules per cell (vdf_dsmc_conv/)
all with the same measures of the heat flux q and the stress sigma:
  equal n*  relative L2 norm over 0.03 <= n* <= 0.93 (the measure of the manuscript tables) and rms in percent of the peak
  in x      profiles aligned at n* = 0.5, over the x-range of 0.03 <= n* <= 0.93: rms in percent of the peak
            (the measure of the manuscript tables) and relative L2 norm
The values of dsmc_resolution_m12.py (relative L2 norm over -12 <= x <= 8 of the directly summed moments, aligned at
n* = 0.5) are copied from analysis/out/dsmc_resolution_m{12,25}.json and recomputed here as a check.

Scatter: the four blocks are consecutive quarters of the sampling time, each summed over all runs of the case. With
f_b the profile of block b and <f> their mean, the scatter of the full sample is
    sqrt( sum_b |f_b - <f>|^2 / (B (B-1)) ) / |<f>|        (standard error of the mean of B = 4 blocks),
with the same norm as the error measure. Each block is reduced with its own density (its own n* and its own plane
n* = 0.5). The blocks hold rho, q and T_par but not T_perp; the stress of a block is obtained from them and from the
mass and energy fluxes of the full sample, which are independent of time:
    u_b = J/rho_b,  2 T_perp,b = 2 (E - q_b)/(rho_b u_b) - 3 T_par,b - u_b^2,  sigma_b = (2/3) rho_b (T_par,b - T_perp,b),
with J(x) = rho u and E(x) = q + rho u (u^2 + 3 T_par + 2 T_perp)/2 of the full sample. The mean of the four block
stresses is compared with the directly summed stress as a check.

Usage: python prediction_vs_reference_accuracy.py <shock_data dir>  ->  analysis/out/prediction_vs_reference_accuracy.json
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl
import predictor_direct as PD
import predictor_models as PM
import m25_prediction_figure as MP
from dsmc_resolution import prep, XG

GRID = MP.GRID
CASES = [dict(name="Mach 25, hard sphere", label="DSMC, hard sphere", mach=25, sources=[(8, 12)], variants=[("A25", "half cell and half time step"), ("B25", "400 molecules per cell")], res="dsmc_resolution_m25.json"),
         dict(name="Mach 12, hard sphere", label="DSMC, hard sphere", mach=12, sources=[(6, 8), (8, 10)], variants=[("A12", "half cell and half time step"), ("B12", "400 molecules per cell")], res="dsmc_resolution_m12.json"),
         dict(name="Mach 12, omega=1", label="DSMC, omega=1", mach=12, sources=[(6, 8), (8, 10)], variants=[("D12", "half cell and half time step"), ("E12", "400 molecules per cell")], res="dsmc_resolution_m12.json"),
         dict(name="Mach 25, omega=1", label="DSMC, omega=1", mach=25, sources=[(8, 12)], variants=[], res=None)]


def grid_profiles(c):
    """q, sigma and x of a solution at the levels GRID of n*, exactly as the reference of predictor_direct.run."""
    k, ns, _, _ = PD.on_grid(c)
    ref = {key: np.interp(GRID, ns, c[key][k]) for key in ("q", "sigma", "x")}
    return ref, ref["x"] - np.interp(0.5, GRID, ref["x"]), k


def spread(A):
    """Standard error of the mean of the rows of A, as relative L2 norm and as rms over the peak, in percent."""
    B = len(A); m = A.mean(0); s2 = ((A - m) ** 2).sum() / (B * (B - 1))
    return dict(L2=float(100 * np.sqrt(s2) / np.linalg.norm(m)), rms_peak=float(100 * np.sqrt(s2 / A.shape[1]) / np.abs(m).max()))


def block_scatter(path, c, k):
    z = np.load(path); st = c["st"]; x = c["x"]
    d = {key: z["direct_" + key] for key in ("rho", "u", "Tpar", "Tperp", "q", "sigma")}
    J = d["rho"] * d["u"]; E = d["q"] + 0.5 * d["rho"] * d["u"] * (d["u"] ** 2 + 3 * d["Tpar"] + 2 * d["Tperp"])
    blocks = []
    for rho, q, Tpar in zip(z["block_rho"], z["block_q"], z["block_Tpar"]):
        u = J / rho; Tperp = 0.5 * (2 * (E - q) / (rho * u) - 3 * Tpar - u ** 2)
        blocks.append(dict(rho=rho, q=q, sigma=(2.0 / 3.0) * rho * (Tpar - Tperp)))
    nb = len(blocks); out = dict(blocks=nb, runs=int(z["nseeds"]))
    # (1) equal n* and (2) in x over the range of 0.03 <= n* <= 0.93, each block with its own n*(x)
    prof = []
    for b in blocks:
        ns = np.maximum.accumulate((b["rho"][k] - 1) / (st["rho2"] - 1))
        xb = np.interp(GRID, ns, x[k]); prof.append(dict(xd=xb - np.interp(0.5, GRID, xb), q=np.interp(GRID, ns, b["q"][k]), sigma=np.interp(GRID, ns, b["sigma"][k])))
    X = np.linspace(max(p["xd"][0] for p in prof), min(p["xd"][-1] for p in prof), 300)
    out["equal_nstar"] = {key: spread(np.array([p[key] for p in prof])) for key in ("q", "sigma")}
    out["x"] = {key: spread(np.array([np.interp(X, p["xd"], p[key]) for p in prof])) for key in ("q", "sigma")}
    out["x"]["nstar_rms"] = float(100 * np.sqrt(((np.array([np.interp(X, p["xd"], GRID) for p in prof]) - np.mean([np.interp(X, p["xd"], GRID) for p in prof], 0)) ** 2).sum() / (nb * (nb - 1) * len(X))))
    # (3) the measure of dsmc_resolution.py: every cell on -12 <= x <= 8, aligned at n* = 0.5
    res = {"q": [], "sigma": []}
    for b in blocks:
        ns = (b["rho"] - 1) / (st["rho2"] - 1); kk = np.where((ns > 0.02) & (ns < 0.98))[0]; kk = np.arange(kk.min(), kk.max() + 1)
        xs = x - np.interp(0.5, ns[kk], x[kk])
        for key in res:
            res[key].append(np.interp(XG, xs, b[key]))
    out["x_resolution_measure"] = {key: spread(np.array(v))["L2"] for key, v in res.items()}
    # check of the block stresses: their mean against the directly summed stress, same cells
    sm = np.mean([b["sigma"] for b in blocks], 0)
    out["block_stress_check_L2_pct"] = float(100 * np.linalg.norm((sm - d["sigma"])[k]) / np.linalg.norm(d["sigma"][k]))
    return out


def poisson_g(p, lv=0.5):
    w = p["wx"]
    return float(100 * np.sqrt((p["gvar"][lv] * w).sum()) / np.sqrt((p["g"][lv] ** 2 * w).sum()))


def refined_direct(root):
    """Method D (no fit involved) for Mach 25 from Mach 8 and 12, hard spheres, with the solutions at half cell and half
    time step (vdf_dsmc_conv/A, A12, A25; four runs each) in place of the baseline solutions, in all combinations of
    refined or baseline sources and refined or baseline target. predictor_direct.run is used unchanged."""
    om = 0.5; nu = lambda F: F["rho"] * F["T"] ** (1 - om); name = "DSMC_w0p5_M{m}_reduced_vdf.npz"
    base = {m: sl.load_case(os.path.join(root, "vdf_dsmc", name.format(m=m))) for m in (8, 12, 25)}
    fine = {m: sl.load_case(os.path.join(root, "vdf_dsmc_conv", d, name.format(m=m))) for m, d in ((8, "A"), (12, "A12"), (25, "A25"))}
    res = {}
    print("\nMethod D, Mach 25 from Mach 8 and 12, hard spheres: baseline or refined (half cell and half time step) solutions")
    for sname, s in (("baseline", base), ("refined", fine)):
        for tname, t in (("baseline", base), ("refined", fine)):
            r = PD.run(t[25], s[8], s[12], nu, "DSMC, hard sphere")
            res[f"sources {sname}, target {tname}"] = {a: r[a] for a in ("q_L2", "sigma_L2", "g_max", "T_max", "width_ratio", "xspace_nstar_rms", "xspace_q_rms", "xspace_sigma_rms")}
            print(f"  sources {sname:8s} target {tname:8s}: equal n*  q {r['q_L2']:.2f}  sigma {r['sigma_L2']:.2f}   in x (rms/peak)  q {r['xspace_q_rms']:.2f}  sigma {r['xspace_sigma_rms']:.2f}  n* {r['xspace_nstar_rms']:.2f}   width ratio {r['width_ratio']:.3f}", flush=True)
    return res


def main(root):
    out = dict(definitions=dict(
        equal_nstar="q_L2, sigma_L2: 100 |f - f_ref| / |f_ref| over the 46 levels 0.03 <= n* <= 0.93 (manuscript tables predictor, predictor2, methods); *_rms_peak: rms difference in percent of the peak of the reference",
        x="profiles aligned at n* = 0.5, 300 points over the common x-range of 0.03 <= n* <= 0.93; *_rms_peak: rms in percent of the peak (manuscript tables predictor3, methods); *_L2: relative L2 norm; nstar_rms in percent of the density jump",
        resolution_json="dsmc_resolution_m12.py: relative L2 norm over -12 <= x <= 8 (every cell, directly summed moments), aligned at n* = 0.5; g at n* = 0.5 as weighted relative L2 norm; g_noise: Poisson scatter expected in the difference of the two runs",
        scatter="standard error of the mean of the four time blocks, same norms; stress of a block from rho, q, T_par of the block and the mass and energy fluxes of the full sample",
        g_n0p5_prediction="relative L2 residual of g, mean of the levels n* = 0.49 and 0.51 of the prediction grid"), cases={})
    for case in CASES:
        om, folder, pat, _ = PM.SETS[case["label"]]; tag = pat.format(m=case["mach"])
        base_path = os.path.join(root, folder, tag + "_reduced_vdf.npz")
        rec = dict(prediction={}, reference={})
        pr0 = None
        for lo, hi in case["sources"]:
            pr = MP.predict(root, case["label"], case["mach"], lo, hi); e = MP.errors(pr); pr0 = pr0 or pr
            rec["prediction"][f"from Mach {lo} and {hi}"] = {m: dict(equal_nstar=e[m]["equal_nstar"], x=e[m]["x"], g_n0p5=float(np.mean(e[m]["g_n049_n051"])),
                                                                        g_max=e[m]["g_max"], width_ratio=e[m]["width_ratio"]) for m in "PDH"}
        c = pr0["c"]; ref, xd, k = grid_profiles(c)
        assert np.allclose(ref["q"], pr0["ref"]["q"]) and np.allclose(xd, pr0["xd"])
        pb = prep(base_path)
        sc = block_scatter(base_path, c, k); sc["g_n0.5_poisson_pct"] = poisson_g(pb)
        rec["reference"]["scatter"] = sc
        rec["reference"]["variants"] = {}
        resj = json.load(open(os.path.join(root, "analysis", "out", case["res"]))) if case["res"] else {}
        for vname, what in case["variants"]:
            vpath = os.path.join(root, "vdf_dsmc_conv", vname, tag + "_reduced_vdf.npz")
            cv = sl.load_case(vpath); rv, xv, kv = grid_profiles(cv); pv = prep(vpath)
            sv = block_scatter(vpath, cv, kv)
            j = resj[vname]
            redo = dict(q_L2_pct=float(100 * np.linalg.norm(pv["prof"]["q"] - pb["prof"]["q"]) / np.linalg.norm(pb["prof"]["q"])),
                        sigma_L2_pct=float(100 * np.linalg.norm(pv["prof"]["sigma"] - pb["prof"]["sigma"]) / np.linalg.norm(pb["prof"]["sigma"])))
            quad = lambda a, b: float(np.hypot(a, b))
            rec["reference"]["variants"][vname] = dict(
                what=what, equal_nstar=MP.equal_nstar(ref, rv), x=MP.xspace(xd, xv, ref, rv),
                resolution_json={a: j[a] for a in ("q_L2_pct", "sigma_L2_pct", "g_L2_pct_n0.5", "g_noise_pct_n0.5", "width_change_pct", "delta_change_pct")},
                resolution_json_recomputed=redo, scatter_of_variant=sv,
                scatter_expected_in_difference=dict(
                    equal_nstar={key: quad(sc["equal_nstar"][key]["L2"], sv["equal_nstar"][key]["L2"]) for key in ("q", "sigma")},
                    x_rms_peak={key: quad(sc["x"][key]["rms_peak"], sv["x"][key]["rms_peak"]) for key in ("q", "sigma")},
                    x_resolution_measure={key: quad(sc["x_resolution_measure"][key], sv["x_resolution_measure"][key]) for key in ("q", "sigma")}))
        out["cases"][case["name"]] = rec
        # ---- printed table
        print(f"\n{case['name']}   (percent)")
        print(f"  {'':46s} | equal n*, L2    | equal n*, rms/peak | in x, rms/peak  | in x, L2        | n*(x) |  g(n*=0.5)")
        print(f"  {'':46s} |    q    sigma   |    q    sigma      |    q    sigma   |    q    sigma   |       |")
        row = lambda lab, a, b, g, nx: print(f"  {lab:46s} | {a['q_L2']:6.2f} {a['sigma_L2']:6.2f}   | {a['q_rms_peak']:6.2f} {a['sigma_rms_peak']:6.2f}      | {b['q_rms_peak']:6.2f} {b['sigma_rms_peak']:6.2f}   | {b['q_L2']:6.2f} {b['sigma_L2']:6.2f}   | {nx:5.2f} | {g}")
        for src, pm in rec["prediction"].items():
            for m in "HDP":
                row(f"prediction {m} - DSMC ({src})", pm[m]["equal_nstar"], pm[m]["x"], f"{pm[m]['g_n0p5']:.2f}", pm[m]["x"]["nstar_rms"])
        a = {f"{key}_{t}": sc["equal_nstar"][key][s] for key in ("q", "sigma") for t, s in (("L2", "L2"), ("rms_peak", "rms_peak"))}
        b = {f"{key}_{t}": sc["x"][key][s] for key in ("q", "sigma") for t, s in (("L2", "L2"), ("rms_peak", "rms_peak"))}
        row(f"scatter of the reference ({sc['blocks']} blocks)", a, b, f"{sc['g_n0.5_poisson_pct']:.2f} (Poisson)", sc["x"]["nstar_rms"])
        for vname, v in rec["reference"]["variants"].items():
            j = v["resolution_json"]
            row(f"reference, {v['what']} ({vname})", v["equal_nstar"], v["x"], f"{j['g_L2_pct_n0.5']:.2f} (scatter {j['g_noise_pct_n0.5']:.2f})", v["x"]["nstar_rms"])
            se = v["scatter_expected_in_difference"]
            print(f"  {'':10s}resolution measure (-12<=x<=8): q {j['q_L2_pct']:.2f} sigma {j['sigma_L2_pct']:.2f} (recomputed {v['resolution_json_recomputed']['q_L2_pct']:.2f}, {v['resolution_json_recomputed']['sigma_L2_pct']:.2f}); "
                  f"scatter expected in this difference: equal n* q {se['equal_nstar']['q']:.2f} sigma {se['equal_nstar']['sigma']:.2f}; resolution measure q {se['x_resolution_measure']['q']:.2f} sigma {se['x_resolution_measure']['sigma']:.2f}")
        print(f"  scatter of the reference in the resolution measure: q {sc['x_resolution_measure']['q']:.3f} sigma {sc['x_resolution_measure']['sigma']:.3f};  block-stress check {sc['block_stress_check_L2_pct']:.3f}", flush=True)
    out["direct_method_refined_solutions"] = refined_direct(root)
    json.dump(out, open(os.path.join(root, "analysis", "out", "prediction_vs_reference_accuracy.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
