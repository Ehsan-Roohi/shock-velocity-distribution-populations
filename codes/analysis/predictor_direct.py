"""Prediction by direct combination of scaled distributions, with the same position step as the three-population
predictor.

The scaled reduced distributions g (u1-u2)/rho and h (u1-u2)/(rho T2), on a common grid of xi_hat and at fixed n*, of
two source Mach numbers are combined linearly in 1/M^2 and evaluated with the Rankine-Hugoniot states of the target.
No Gaussian fit is involved. The moments of the combined distribution give sigma_xx and int xi_x^3 g, and
dx/dn* = d(int xi_x^3 g)/dn* / (-nu sigma_xx) gives the positions, exactly as in predictor.py. The same errors are
reported as for the three-population predictor (analysis/out/predictor_summary.json, predictor_models_summary.json),
so that the two can be compared test by test.

Usage: python predictor_direct.py <shock_data dir>     -> analysis/out/predictor_direct.json
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shocklib as sl
import predictor as P
import predictor_models as PM

GRID = P.GRID
XIG = np.linspace(-7, 5, 3841)
KEEP = None            # set to a dict to receive the station-by-station arrays of the last call of run()
BGK_TESTS = [("M3", "M2p5", "M4"), ("M4", "M3", "M5"), ("M5", "M4", "M6_gl3fine"), ("M6_gl3fine", "M5", "M8"), ("M8", "M6_gl3fine", "M12_gl3medium"),
             ("M8", "M5", "M6_gl3fine"), ("M12_gl3medium", "M6_gl3fine", "M8"), ("M12_gl3medium", "M5", "M8"), ("M12_gl3medium", "M5", "M6_gl3fine")]


def on_grid(c):
    """Contiguous cells across the shock and their monotone n*; scaled g and h interpolated to GRID."""
    st = c["st"]; du = st["u1"] - st["u2"]; xi = sl.xi_hat(c["vx"], st)
    core = np.where((c["nstar"] > 0.005) & (c["nstar"] < 0.97))[0]; k = np.arange(core.min() - 2, core.max() + 3)
    ns = np.maximum.accumulate(c["nstar"][k])
    Gs = np.array([np.interp(XIG, xi, c["G"][j] * du / c["rho"][j], left=0, right=0) for j in k])
    Hs = np.array([np.interp(XIG, xi, c["H"][j] * du / (c["rho"][j] * st["T2"]), left=0, right=0) for j in k])
    g = np.array([[np.interp(a, ns, Gs[:, i]) for i in range(len(XIG))] for a in GRID]) if False else np.stack([_interp_rows(a, ns, Gs) for a in GRID])
    h = np.stack([_interp_rows(a, ns, Hs) for a in GRID])
    return k, ns, g, h


def _interp_rows(a, ns, M):
    j = int(np.argmax(ns >= a)); j = max(j, 1); t = (a - ns[j - 1]) / (ns[j] - ns[j - 1])
    return (1 - t) * M[j - 1] + t * M[j]


def stream(fit, st):
    """Scaled reduced distributions of the fast population of the three-population fit on XIG, at the levels GRID,
    and its scaled parameters (mass fraction, xi_hat, T_par/T2, T_perp/T2)."""
    Q = P.scaled(fit, st)                                   # [grid, 3, 4]
    k = np.argmax(Q[:, :, 1], 1); q = Q[np.arange(len(GRID)), k]
    return q


def stream_gh(q, st):
    s2 = np.clip(q[:, 2], 1e-6, None) * st["T2"] / (st["u1"] - st["u2"]) ** 2
    g = q[:, 0:1] / np.sqrt(2 * np.pi * s2[:, None]) * np.exp(-(XIG[None, :] - q[:, 1:2]) ** 2 / (2 * s2[:, None]))
    return g, 2 * np.clip(q[:, 3:4], 1e-6, None) * g


def run(c, cl, ch, nu_of, om_label, fl=None, fh=None):
    st = c["st"]; s = lambda M: 1 / M ** 2
    wH = (s(st["M1"]) - s(cl["st"]["M1"])) / (s(ch["st"]["M1"]) - s(cl["st"]["M1"]))
    k, ns, _, _ = on_grid(c)
    _, _, gl, hl = on_grid(cl); _, _, gh, hh = on_grid(ch)
    gd, hd = (1 - wH) * gl + wH * gh, (1 - wH) * hl + wH * hh
    pos = {}
    negmass = lambda a, ref: float(100 * np.max(np.clip(-a, 0, None).sum(1) / ref.sum(1)))      # percent of the mass, worst station
    relmin = lambda a, ref: float(100 * np.min(a.min(1) / ref.max(1)))                           # percent of the peak, worst station
    if fl is not None:                                      # stream separately, remainder directly
        ql, qh = stream(fl, cl["st"]), stream(fh, ch["st"])
        (bl, bhl), (bh, bhh) = stream_gh(ql, cl["st"]), stream_gh(qh, ch["st"])
        qt = (1 - wH) * ql + wH * qh
        pos.update(beam_fraction_range=[float(qt[:, 0].min()), float(qt[:, 0].max())], beam_Tpar_min=float(qt[:, 2].min()), beam_Tperp_min=float(qt[:, 3].min()),
                   beam_clipped=bool((qt[:, 0] < 0).any() or (qt[:, 0] > 1).any() or (qt[:, 2:] < 1e-6).any()),
                   remainder_g_negmass=max(negmass(gl - bl, gl), negmass(gh - bh, gh)), remainder_g_min=min(relmin(gl - bl, gl), relmin(gh - bh, gh)),
                   remainder_h_negmass=max(negmass(hl - bhl, hl), negmass(hh - bhh, hh)), remainder_h_min=min(relmin(hl - bhl, hl), relmin(hh - bhh, hh)))
        qt[:, 0] = np.clip(qt[:, 0], 0, 1)
        bt, bht = stream_gh(qt, st)
        gd = (1 - wH) * (gl - bl) + wH * (gh - bh) + bt
        hd = (1 - wH) * (hl - bhl) + wH * (hh - bhh) + bht
    # positivity of the combined distributions: nothing is clipped or renormalised
    pos.update(weight_low=float(1 - wH), weight_high=float(wH), g_negmass=negmass(gd, gd), g_min=relmin(gd, gd), h_negmass=negmass(hd, hd), h_min=relmin(hd, hd))
    du = st["u1"] - st["u2"]; v = st["u2"] + XIG * du; w = np.full(len(v), v[1] - v[0])
    rho = 1 + GRID * (st["rho2"] - 1)
    G = gd * rho[:, None] / du; H = hd * rho[:, None] * st["T2"] / du
    m = sl.moments(v, w, G, H)
    F = dict(rho=m["rho"], u=m["u"], T=m["T"], q=m["q"], sigma=m["sigma"], Phi2=G @ (v ** 3 * w), mass=G @ (v * w), mom=G @ (v ** 2 * w))
    energy = 0.5 * (G @ (v ** 3 * w) + H @ (v * w))
    ref = {key: np.interp(GRID, ns, c[key][k]) for key in ("q", "sigma", "T", "rho", "x")}
    xd = ref["x"] - np.interp(0.5, GRID, ref["x"])
    xp, dxdn = P.x_of_nstar(F, nu_of(F))
    width = lambda xx: float(np.interp(0.8, GRID, xx) - np.interp(0.2, GRID, xx))
    vt, wt = c["vx"], c["wx"]; sw = np.sqrt(wt); xi_t = sl.xi_hat(vt, st); eg = []
    for i in range(len(GRID)):
        gref = _interp_rows(GRID[i], ns, c["G"][k])
        gp = np.interp(xi_t, XIG, gd[i]) * (gref @ wt) / du
        eg.append(np.linalg.norm((gp - gref) * sw) / np.linalg.norm(gref * sw))
    e1 = st["u1"] * (0.5 * st["u1"] ** 2 + 2.5)
    res = dict(mach=st["M1"], sources=[cl["st"]["M1"], ch["st"]["M1"]], kind="interp" if 0 <= wH <= 1 else "extrap",
               g_max=float(100 * np.max(eg)), q_L2=float(100 * np.linalg.norm(F["q"] - ref["q"]) / np.linalg.norm(ref["q"])),
               sigma_L2=float(100 * np.linalg.norm(F["sigma"] - ref["sigma"]) / np.linalg.norm(ref["sigma"])),
               T_max=float(100 * np.max(np.abs(F["T"] - ref["T"]) / ref["T"])), monotone=bool(np.all(dxdn > 0)),
               width_ratio=width(xp) / width(xd),
               flux_mass=float(100 * np.max(np.abs(F["mass"] / st["u1"] - 1))), flux_mom=float(100 * np.max(np.abs(F["mom"] / (st["u1"] ** 2 + 1) - 1))),
               flux_energy=float(100 * np.max(np.abs(energy / e1 - 1))), positivity=pos)
    if KEEP is not None:
        KEEP.update(eg=np.array(eg), F=F, ref=ref, xd=xd, xp=xp, dxdn=dxdn, energy=energy / e1, k=k, ns=ns)
    if fl is not None:                                      # the three-population prediction of the same test: are its parameters admissible?
        Ql, Qh = P.scaled(fl, cl["st"]), P.scaled(fh, ch["st"]); Qp = (1 - wH) * Ql + wH * Qh
        res["populations_min"] = dict(mass_fraction=float(Qp[:, :, 0].min()), Tpar=float(Qp[:, :, 2].min()), Tperp=float(Qp[:, :, 3].min()),
                                      mass_fraction_sum_range=[float(Qp[:, :, 0].sum(1).min()), float(Qp[:, :, 0].sum(1).max())])
    if res["monotone"]:
        lo, hi = max(xd[0], xp[0]), min(xd[-1], xp[-1]); X = np.linspace(lo, hi, 300)
        res["xspace_nstar_rms"] = float(100 * np.sqrt(np.mean((np.interp(X, xp, GRID) - np.interp(X, xd, GRID)) ** 2)))
        for key in ("q", "sigma"):
            res[f"xspace_{key}_rms"] = float(100 * np.sqrt(np.mean((np.interp(X, xp, F[key]) - np.interp(X, xd, ref[key])) ** 2)) / np.abs(ref[key]).max())
        res["xspace_T_rms"] = float(100 * np.sqrt(np.mean((np.interp(X, xp, F["T"]) - np.interp(X, xd, ref["T"])) ** 2)) / (st["T2"] - st["T1"]))
    else:
        bad = dxdn <= 0; res["nonmonotone_nstar"] = [float(GRID[bad].min()), float(GRID[bad].max())]
    return res


def main(root):
    out = {}
    red = os.path.join(root, "vdf_reduced")
    cache = {}
    def bgk(t):
        if t not in cache:
            cache[t] = sl.load_case(os.path.join(red, f"{t}_BGK_reduced_vdf.npz"))
        return cache[t]
    fits = os.path.join(root, "analysis", "out", "popfits")
    for tgt, lo, hi in BGK_TESTS:
        r = run(bgk(tgt), bgk(lo), bgk(hi), lambda F: 1.0, "BGK")
        r2 = run(bgk(tgt), bgk(lo), bgk(hi), lambda F: 1.0, "BGK", np.load(os.path.join(fits, f"{lo}_BGK_branch3.npz")), np.load(os.path.join(fits, f"{hi}_BGK_branch3.npz")))
        out[f"BGK|{tgt}_from_{lo}_{hi}"] = r; out[f"BGK|{tgt}_from_{lo}_{hi}|hybrid"] = r2
        print("BGK", tgt, lo, hi, "direct", {a: (round(b, 2) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
        print("BGK", tgt, lo, hi, "hybrid", {a: (round(b, 2) if isinstance(b, float) else b) for a, b in r2.items()}, flush=True)
    for label, (om, folder, pat, tests) in PM.SETS.items():
        if not label.startswith("DSMC"):
            continue
        cache2 = {}
        def get(mm):
            if mm not in cache2:
                cache2[mm] = sl.load_case(os.path.join(root, folder, pat.format(m=mm) + "_reduced_vdf.npz"))
            return cache2[mm]
        for tgt, lo, hi in tests:
            if 25 in (tgt, lo, hi) and (tgt, lo, hi) not in ((25, 8, 12),):
                continue
            try:
                r = run(get(tgt), get(lo), get(hi), lambda F: F["rho"] * F["T"] ** (1 - om), label)
            except (FileNotFoundError, OSError):
                continue
            fbr = lambda mm: np.load(os.path.join(fits, get(mm)["name"] + "_branch3.npz"))
            r2 = run(get(tgt), get(lo), get(hi), lambda F: F["rho"] * F["T"] ** (1 - om), label, fbr(lo), fbr(hi))
            out[f"{label}|M{tgt} from {lo},{hi}"] = r; out[f"{label}|M{tgt} from {lo},{hi}|hybrid"] = r2
            print(label, tgt, lo, hi, "direct", {a: (round(b, 2) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
            print(label, tgt, lo, hi, "hybrid", {a: (round(b, 2) if isinstance(b, float) else b) for a, b in r2.items()}, flush=True)
    json.dump(out, open(os.path.join(root, "analysis", "out", "predictor_direct.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
