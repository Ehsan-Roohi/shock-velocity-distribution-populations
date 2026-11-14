"""Sensitivity of the three-population prediction (method P, predictor.py / predictor_models.py) to the lower
bound on the extrapolated population temperatures, and two positive forms of the beam-temperature extrapolation.

In predictor.unscale the extrapolated scaled temperatures T_par/T2 and T_perp/T2 of the three populations are
bounded below by 1e-4 (and the mass fractions by 1e-9, then renormalised). Variants computed here:

  baseline      linear in 1/M^2 of T/T2, bound 1e-4                     (the implementation of the paper)
  bound_x0.1 .. bound_x10    the same with the bound 1e-5, 5e-5, 2e-4, 1e-3
  no_bound      linear in 1/M^2 of T/T2, no bound on the temperatures   (moments are algebraic in T_perp, so they
                                                                         can be evaluated with a negative value)
  beam_T1       beam T_par and T_perp combined linearly in 1/M^2 in upstream units, T/T1, no bound
  beam_logT2    ln(T_beam/T2) combined linearly in 1/M^2, no bound
  beam_logT1    ln(T_beam/T1) combined linearly in 1/M^2, no bound
  perp_T1, perp_logT1    as beam_T1 and beam_logT1 for the transverse temperature of the beam only; its parallel
                temperature stays linear in T/T2, so that the reduced distribution g is that of the baseline

The beam is the population with the largest scaled mean velocity (as in predictor_direct.stream). Only its two
temperatures are treated differently; everything else (mass fractions, mean velocities, temperatures of the other
two populations, the position step) is as in predictor.py. Error definitions are copied from predictor.py (BGK)
and predictor_models.py (DSMC). The hybrid method H of predictor_direct.py is run with the same variants of the
beam temperatures through a replacement of predictor_direct.stream_gh (the module on disk is not changed).

Usage: python positivity_sensitivity.py <shock_data dir>     -> analysis/out/positivity_sensitivity.json
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl
import predictor as P
import predictor_models as PM
import predictor_direct as PD

GRID = P.GRID
BOUND0 = 1e-4                       # predictor.py, line 47
VARIANTS = {"baseline": ("linT2", BOUND0), "bound_x0.1": ("linT2", 0.1 * BOUND0), "bound_x0.5": ("linT2", 0.5 * BOUND0),
            "bound_x2": ("linT2", 2 * BOUND0), "bound_x10": ("linT2", 10 * BOUND0), "no_bound": ("linT2", None),
            "beam_T1": ("linT1", None), "beam_logT2": ("logT2", None), "beam_logT1": ("logT1", None),
            "perp_T1": ("linT1p", None), "perp_logT1": ("logT1p", None)}
H_VARIANTS = {"baseline": ("linT2", 1e-6), "bound_1e-4": ("linT2", 1e-4), "beam_T1": ("linT1", None),
              "beam_logT2": ("logT2", None), "beam_logT1": ("logT1", None), "perp_T1": ("linT1p", None), "perp_logT1": ("logT1p", None)}
s = lambda M: 1.0 / M ** 2


def beam_temps(tl, th, wH, stl, sth, st, mode):
    """Beam temperature in units of T2 of the target from the source values tl, th (units of their own T2)."""
    if mode == "linT2":
        return (1 - wH) * tl + wH * th
    if mode == "linT1":
        return ((1 - wH) * tl * stl["T2"] / stl["T1"] + wH * th * sth["T2"] / sth["T1"]) * st["T1"] / st["T2"]
    if mode == "logT2":
        return np.exp((1 - wH) * np.log(tl) + wH * np.log(th))
    if mode == "logT1":
        return np.exp((1 - wH) * np.log(tl * stl["T2"] / stl["T1"]) + wH * np.log(th * sth["T2"] / sth["T1"])) * st["T1"] / st["T2"]
    raise ValueError(mode)


def combine(Ql, Qh, wH, stl, sth, st, mode):
    Qp = (1 - wH) * Ql + wH * Qh
    i = np.arange(len(GRID)); kl, kh = np.argmax(Ql[:, :, 1], 1), np.argmax(Qh[:, :, 1], 1); same = kl == kh
    if mode != "linT2":
        for j in ((3,) if mode.endswith("p") else (2, 3)):
            val = beam_temps(Ql[i, kl, j], Qh[i, kh, j], wH, stl, sth, st, mode.rstrip("p"))
            Qp[i[same], kl[same], j] = val[same]
    kb = np.where(same, kl, np.argmax(Qp[:, :, 1], 1))
    return Qp, kb, int((~same).sum())


def unscale(Q, st, tb):
    """predictor.unscale with the temperature bound tb (None: no bound)."""
    rho = 1 + GRID * (st["rho2"] - 1)
    frac = np.clip(Q[:, :, 0], 1e-9, None); frac = frac / frac.sum(1, keepdims=True)
    a, b = (Q[:, :, 2], Q[:, :, 3]) if tb is None else (np.clip(Q[:, :, 2], tb, None), np.clip(Q[:, :, 3], tb, None))
    return frac * rho[:, None], st["u2"] + Q[:, :, 1] * (st["u1"] - st["u2"]), a * st["T2"], b * st["T2"]


def reference(c, kind):
    v = c["vx"]
    if kind == "dsmc":                                           # predictor_models.py, lines 50-57
        core = np.where((c["nstar"] > 0.005) & (c["nstar"] < 0.97))[0]; k = np.arange(core.min() - 2, core.max() + 3)
        ns_all = np.maximum.accumulate(c["nstar"][k])
        ref = {key: np.interp(GRID, ns_all, c[key][k]) for key in ("q", "sigma", "T", "rho", "x")}
        Gref = np.array([np.interp(GRID, ns_all, c["G"][k, i]) for i in range(len(v))]).T
    else:                                                        # predictor.py, lines 114-117 and 131
        o = np.argsort(c["nstar"])
        ref = {key: np.interp(GRID, c["nstar"][o], c[key][o]) for key in ("q", "sigma", "T", "x")}
        Gref = np.array([c["G"][int(np.argmin(np.abs(c["nstar"] - ns)))] for ns in GRID])
    xd = ref["x"] - np.interp(0.5, GRID, ref["x"])
    return ref, Gref, xd, 1 / np.max(1 / np.gradient(xd, GRID))


def evaluate(c, refs, Q, om, tb, kind):
    st = c["st"]; v, w = c["vx"], c["wx"]; sw = np.sqrt(w); ref, Gref, xd, delta = refs
    n, u, a, b = unscale(Q, st, tb)
    F = P.mixture_fields(n, u, a, b)
    xp, dxdn = P.x_of_nstar(F, 1.0 if om is None else F["rho"] * F["T"] ** (1 - om))
    width = lambda xx: float(np.interp(0.8, GRID, xx) - np.interp(0.2, GRID, xx))
    eg = []
    for i in range(len(GRID)):
        g = Gref[i]
        with np.errstate(invalid="ignore"):
            p = np.stack([np.log(n[i]), u[i], np.log(a[i]), np.zeros(3)], 1).ravel()      # g does not depend on T_perp
        gp = sl.pop_model(p, v)[0].sum(0)
        if kind == "dsmc":
            gp = gp * (g @ w) / n[i].sum()
        eg.append(np.linalg.norm((gp - g) * sw) / np.linalg.norm(g * sw))
    res = dict(g_max=float(100 * np.max(eg)), g_mean=float(100 * np.mean(eg)),
               q_L2=float(100 * np.linalg.norm(F["q"] - ref["q"]) / np.linalg.norm(ref["q"])),
               sigma_L2=float(100 * np.linalg.norm(F["sigma"] - ref["sigma"]) / np.linalg.norm(ref["sigma"])),
               T_max=float(100 * np.max(np.abs(F["T"] - ref["T"]) / ref["T"])), monotone=bool(np.all(dxdn > 0)),
               width_ratio=width(xp) / width(xd), x_rms=float(np.sqrt(np.mean((xp - xd) ** 2)) / delta))
    res.update({k_[:-2]: v_ for k_, v_ in P.extra_checks(F, n, u, a, b, st, ref, xd, xp, dxdn, "X").items()})
    return res


def positivity(Q, kb, st, tb):
    """Smallest extrapolated values before any bound, and where the temperature bound acts."""
    i = np.arange(len(GRID)); beam = Q[i, kb]; r = st["T2"] / st["T1"]
    out = dict(mass_fraction_min=float(Q[:, :, 0].min()), Tpar_min_T2=float(Q[:, :, 2].min()), Tperp_min_T2=float(Q[:, :, 3].min()),
               beam_Tpar_T1=[float(beam[:, 2].min() * r), float(beam[:, 2].max() * r)],
               beam_Tperp_T1=[float(beam[:, 3].min() * r), float(beam[:, 3].max() * r)],
               beam_Tpar_min_T2=float(beam[:, 2].min()), beam_Tperp_min_T2=float(beam[:, 3].min()),
               all_positive=bool((Q[:, :, [0, 2, 3]] > 0).all()))
    lim = 0.0 if tb is None else tb
    act = (Q[:, :, 2:] < lim).any(axis=(1, 2))
    out["bound_active"] = bool(act.any()) if tb is not None else False
    out["nonpositive_or_below_bound_points"] = int(act.sum())
    if act.any():
        out["nstar_range"] = [float(GRID[act].min()), float(GRID[act].max())]
        out["beam_mass_fraction_there"] = [float(beam[act, 0].min()), float(beam[act, 0].max())]
        out["which"] = sorted({("Tpar", "Tperp")[j] + "_pop%d" % k for _, k, j in zip(*np.where(Q[:, :, 2:] < lim))})
    return out


def run_P(c, cl, ch, Ql, Qh, refs, om, kind):
    st = c["st"]; wH = (s(st["M1"]) - s(cl["st"]["M1"])) / (s(ch["st"]["M1"]) - s(cl["st"]["M1"]))
    out = dict(mach=st["M1"], sources=[cl["st"]["M1"], ch["st"]["M1"]], weight_high=float(wH), variants={})
    for name, (mode, tb) in VARIANTS.items():
        Qp, kb, ndiff = combine(Ql, Qh, wH, cl["st"], ch["st"], st, mode)
        r = evaluate(c, refs, Qp, om, tb, kind); r["positivity"] = positivity(Qp, kb, st, tb); r["beam_index_differs_points"] = ndiff
        act = (Qp[:, :, 2:] < (0.0 if tb is None else tb)).any(axis=(1, 2))
        if act.any():                                            # share of sum_k n_k T_perp,k held by the beam in the two sources at these positions
            i = np.arange(len(GRID)); sh = lambda Q: (Q[i, kb, 0] * Q[i, kb, 3] / (Q[:, :, 0] * Q[:, :, 3]).sum(1))[act]
            r["positivity"]["beam_share_of_transverse_energy_in_sources_there"] = [float(min(sh(Ql).min(), sh(Qh).min())), float(max(sh(Ql).max(), sh(Qh).max()))]
        out["variants"][name] = r
    return out


def run_H(c, cl, ch, nu_of, fl, fh):
    """predictor_direct.run with the beam temperatures of the target formed by each variant."""
    out = {}; orig = PD.stream_gh
    for name, (mode, tb) in H_VARIANTS.items():
        calls = []
        def stream_gh(q, st, mode=mode, tb=tb, calls=calls):
            calls.append((q.copy(), st)); q = q.copy()
            if len(calls) == 3:                                  # third call: the beam of the target
                (ql, stl), (qh, sth) = calls[0], calls[1]
                wH = (s(st["M1"]) - s(stl["M1"])) / (s(sth["M1"]) - s(stl["M1"]))
                for j in ((3,) if mode.endswith("p") else (2, 3)):
                    q[:, j] = beam_temps(ql[:, j], qh[:, j], wH, stl, sth, st, mode.rstrip("p"))
                calls.append((q.copy(), st))
            a, b = (q[:, 2], q[:, 3:4]) if tb is None else (np.clip(q[:, 2], tb, None), np.clip(q[:, 3:4], tb, None))
            s2 = a * st["T2"] / (st["u1"] - st["u2"]) ** 2
            g = q[:, 0:1] / np.sqrt(2 * np.pi * s2[:, None]) * np.exp(-(PD.XIG[None, :] - q[:, 1:2]) ** 2 / (2 * s2[:, None]))
            return g, 2 * b * g
        PD.stream_gh = stream_gh
        try:
            r = PD.run(c, cl, ch, nu_of, "", fl, fh)
        finally:
            PD.stream_gh = orig
        qt = calls[3][0]; rr = c["st"]["T2"] / c["st"]["T1"]
        keep = {k_: r[k_] for k_ in ("g_max", "q_L2", "sigma_L2", "T_max", "width_ratio", "monotone", "flux_mass", "flux_mom", "flux_energy",
                                     "xspace_nstar_rms", "xspace_q_rms", "xspace_sigma_rms", "xspace_T_rms") if k_ in r}
        keep.update(beam_Tpar_min_T2=float(qt[:, 2].min()), beam_Tperp_min_T2=float(qt[:, 3].min()),
                    beam_Tpar_min_T1=float(qt[:, 2].min() * rr), beam_Tperp_min_T1=float(qt[:, 3].min() * rr),
                    bound_active=bool(tb is not None and (qt[:, 2:] < tb).any()),
                    g_negmass=r["positivity"]["g_negmass"], h_negmass=r["positivity"]["h_negmass"])
        out[name] = keep
    return out


def main(root):
    fits = os.path.join(root, "analysis", "out", "popfits"); od = os.path.join(root, "analysis", "out")
    out = dict(bound=dict(file="analysis/predictor.py", function="unscale", line=47, temperatures_T2=BOUND0, mass_fraction=1e-9),
               P={}, H={}, source_beam_temperatures={})
    # ---------------- DSMC
    for label, (om, folder, pat, tests) in PM.SETS.items():
        if not label.startswith("DSMC"):
            continue
        cache = {}
        def get(m):
            if m not in cache:
                c = sl.load_case(os.path.join(root, folder, pat.format(m=m) + "_reduced_vdf.npz"))
                f = np.load(os.path.join(fits, c["name"] + "_branch3.npz"))
                cache[m] = (c, P.scaled(f, c["st"]), reference(c, "dsmc"), f)
            return cache[m]
        for tgt, lo, hi in tests:
            try:
                (c, Qf, refs, _), (cl, Ql, _, fl), (ch, Qh, _, fh) = get(tgt), get(lo), get(hi)
            except (FileNotFoundError, OSError):
                continue
            key = f"{label}|M{tgt} from {lo},{hi}"
            out["P"][key] = run_P(c, cl, ch, Ql, Qh, refs, om, "dsmc")
            out["P"][key]["fit"] = evaluate(c, refs, Qf, om, BOUND0, "dsmc")
            out["P"][key]["fit"]["positivity"] = positivity(Qf, np.argmax(Qf[:, :, 1], 1), c["st"], BOUND0)
            if (tgt, lo, hi) in ((25, 8, 12), (12, 6, 8), (12, 8, 10)):
                out["H"][key] = run_H(c, cl, ch, lambda F: F["rho"] * F["T"] ** (1 - om), fl, fh)
            print("done", key, flush=True)
        i = np.arange(len(GRID))
        for m in (5, 6, 8, 10, 12, 25):                          # beam temperatures of the fits, in upstream units
            try:
                c, Q = get(m)[:2]
            except (FileNotFoundError, OSError):
                continue
            kb = np.argmax(Q[:, :, 1], 1); r = c["st"]["T2"] / c["st"]["T1"]
            out["source_beam_temperatures"][f"{label}|M{m}"] = dict(
                nstar=[float(GRID[j]) for j in (0, 6, 16, 26, 36, 45)],
                Tpar_T1=[float(Q[j, kb[j], 2] * r) for j in (0, 6, 16, 26, 36, 45)], Tperp_T1=[float(Q[j, kb[j], 3] * r) for j in (0, 6, 16, 26, 36, 45)],
                Tpar_T1_range=[float((Q[i, kb, 2] * r).min()), float((Q[i, kb, 2] * r).max())],
                Tperp_T1_range=[float((Q[i, kb, 3] * r).min()), float((Q[i, kb, 3] * r).max())],
                beam_fraction=[float(Q[j, kb[j], 0]) for j in (0, 6, 16, 26, 36, 45)])
    # ---------------- BGK
    red = os.path.join(root, "vdf_reduced"); cache = {}
    def getb(t):
        if t not in cache:
            c = sl.load_case(os.path.join(red, f"{t}_BGK_reduced_vdf.npz")); f = np.load(os.path.join(fits, f"{t}_BGK_branch3.npz"))
            cache[t] = (c, P.scaled(f, c["st"]), reference(c, "bgk"), f)
        return cache[t]
    for tgt, lo, hi, kind in P.TESTS:
        (c, Qf, refs, _), (cl, Ql, _, fl), (ch, Qh, _, fh) = getb(tgt), getb(lo), getb(hi)
        key = f"BGK|{tgt}_from_{lo}_{hi}"
        out["P"][key] = run_P(c, cl, ch, Ql, Qh, refs, None, "bgk")
        out["P"][key]["fit"] = evaluate(c, refs, Qf, None, BOUND0, "bgk")
        out["P"][key]["fit"]["positivity"] = positivity(Qf, np.argmax(Qf[:, :, 1], 1), c["st"], BOUND0)
        if (tgt, lo, hi) == ("M12_gl3medium", "M6_gl3fine", "M8"):
            out["H"][key] = run_H(c, cl, ch, lambda F: 1.0, fl, fh)
        print("done", key, flush=True)
    # ---------------- comparison of the baseline with the stored results of predictor.py / predictor_models.py / predictor_direct.py
    chk = {}
    pm = json.load(open(os.path.join(od, "predictor_models_summary.json"))); ps = json.load(open(os.path.join(od, "predictor_summary.json")))
    pdj = json.load(open(os.path.join(od, "predictor_direct.json")))
    names = ("g_max", "q_L2", "sigma_L2", "T_max", "width_ratio", "xspace_nstar_rms", "xspace_q_rms", "xspace_sigma_rms")
    for key, r in out["P"].items():
        old = pm.get(key) if not key.startswith("BGK|") else ps.get(key[4:])
        if old is None:
            continue
        b = r["variants"]["baseline"]
        chk[key] = {nm: [old.get(nm + "_pred"), b.get(nm)] for nm in names}
        chk[key]["max_abs_diff"] = float(max(abs(old[nm + "_pred"] - b[nm]) for nm in names if nm + "_pred" in old and nm in b))
        chk[key]["max_abs_diff_fit"] = float(max(abs(old[nm + "_fit"] - r["fit"][nm]) for nm in ("g_max", "q_L2", "sigma_L2")))
    for key, r in out["H"].items():
        old = pdj.get(key + "|hybrid")
        if old is not None:
            chk["H|" + key] = dict(max_abs_diff=float(max(abs(old[nm] - r["baseline"][nm]) for nm in ("g_max", "q_L2", "sigma_L2", "T_max", "width_ratio"))))
    out["baseline_vs_stored"] = chk
    json.dump(out, open(os.path.join(od, "positivity_sensitivity.json"), "w"), indent=1)

    # ---------------- printed summary
    print("\nbaseline against stored json: largest difference over all tests and columns:",
          max(v["max_abs_diff"] for v in chk.values()), "(fit:", max(v.get("max_abs_diff_fit", 0) for v in chk.values()), ")")
    fmt = lambda r: (f"g {r['g_max']:6.3f} q {r['q_L2']:6.3f} sig {r['sigma_L2']:6.3f} T {r['T_max']:6.3f} width {r['width_ratio']:.4f} mono {int(r['monotone'])} | x: n* "
                     f"{r.get('xspace_nstar_rms', float('nan')):5.3f} q {r.get('xspace_q_rms', float('nan')):5.3f} sig {r.get('xspace_sigma_rms', float('nan')):5.3f} | flux m/p/e "
                     f"{r['flux_mass']:.2f}/{r['flux_mom']:.2f}/{r['flux_energy']:.2f}")
    show = [k for k in out["P"] if ("M25 from 8,12" in k or "M25 from 10,12" in k or "M12 from 6,8" in k or "M12 from 5,8" in k or "M12 from 8,10" in k
                                    or k.startswith("BGK|M12") or k.startswith("BGK|M8_from_M5"))]
    for key in show:
        r = out["P"][key]; print(f"\nP  {key}   (fit: g {r['fit']['g_max']:.3f} q {r['fit']['q_L2']:.3f} sig {r['fit']['sigma_L2']:.3f})")
        for name, v in r["variants"].items():
            p = v["positivity"]
            print(f"  {name:11s} {fmt(v)} || min frac {p['mass_fraction_min']:.4f} Tpar {p['Tpar_min_T2']:.5f} Tperp {p['Tperp_min_T2']:.5f} T2; beam Tpar/T1 "
                  f"{p['beam_Tpar_T1'][0]:.3f}-{p['beam_Tpar_T1'][1]:.3f} Tperp/T1 {p['beam_Tperp_T1'][0]:.3f}-{p['beam_Tperp_T1'][1]:.3f} active {int(p['bound_active'])} "
                  f"pts {p['nonpositive_or_below_bound_points']} {p.get('nstar_range', '')} {p.get('which', '')} beamfrac {p.get('beam_mass_fraction_there', '')}")
    for key, r in out["H"].items():
        print(f"\nH  {key}")
        for name, v in r.items():
            print(f"  {name:11s} g {v['g_max']:6.3f} q {v['q_L2']:6.3f} sig {v['sigma_L2']:6.3f} T {v['T_max']:6.3f} width {v['width_ratio']:.4f} | x: n* {v.get('xspace_nstar_rms', float('nan')):5.3f} "
                  f"q {v.get('xspace_q_rms', float('nan')):5.3f} sig {v.get('xspace_sigma_rms', float('nan')):5.3f} || beam min Tpar {v['beam_Tpar_min_T2']:.5f} Tperp {v['beam_Tperp_min_T2']:.5f} T2 "
                  f"({v['beam_Tpar_min_T1']:.3f}, {v['beam_Tperp_min_T1']:.3f} T1) active {int(v['bound_active'])} g- {v['g_negmass']:.2e} h- {v['h_negmass']:.2e}")
    # largest change of any error over all tests when the beam temperatures are treated in upstream units or in the logarithm
    for var in ("beam_T1", "beam_logT2", "beam_logT1", "perp_T1", "perp_logT1"):
        worst = {}
        for key, r in out["P"].items():
            if "M25" in key:
                continue
            b, v = r["variants"]["baseline"], r["variants"][var]
            for nm in ("g_max", "q_L2", "sigma_L2", "T_max"):
                d = v[nm] - b[nm]
                if nm not in worst or abs(d) > abs(worst[nm][0]):
                    worst[nm] = (round(d, 3), key)
        print(var, "largest change over the tests below Mach 25:", worst)
    print("\nbeam temperatures of the direct fits in upstream units (n* = 0.03, 0.15, 0.35, 0.55, 0.75, 0.93):")
    for key, v in out["source_beam_temperatures"].items():
        print(f"  {key:28s} Tpar/T1 {np.round(v['Tpar_T1'], 3)} Tperp/T1 {np.round(v['Tperp_T1'], 3)} frac {np.round(v['beam_fraction'], 3)}")


if __name__ == "__main__":
    main(sys.argv[1])
