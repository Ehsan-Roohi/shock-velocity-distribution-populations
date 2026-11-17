"""Excitation function of one collision: cross-section for leaving a cold stream molecule with rotational energy
above E, from the classical trajectories of ctc_kernel.py (all collisions, not only those that send the molecule
upstream), next to the Larsen-Borgnakke rule on the same pairs with constant probability and with the
energy-dependent probability of Boyd (1993), both with the VHS collision rate.

Usage: python ctc_excite.py <out.npz> <raw npz> [...]      [zinf=10] [tstar=91.5]"""
import sys, math
import numpy as np

files = [a for a in sys.argv[2:] if "=" not in a]; opt = dict(a.split("=") for a in sys.argv[2:] if "=" in a)
zinf = float(opt.get("zinf", 10.0)); tstar = float(opt.get("tstar", 91.5))
D = [np.load(f) for f in files]; P = {k[4:]: float(D[0][k]) for k in D[0].files if k.startswith("par_")}
R = {k: np.concatenate([d[k] for d in D]) for k in ("v1x", "e1", "e1_0", "e2_0", "w", "drift", "cchi", "lb_e1", "lb_v1x", "wv")}
T1, T2, om = P["T1"], P["T2"], P["omega"]; eta = 2.5 - om; MU = 14.0; sigmax = math.pi * P["bmax"] ** 2
ok = np.abs(R["drift"]) < 1e-3; w = R["w"] * ok; g = R["w"]; et = 0.5 * MU * g ** 2; n = len(g)
sv = math.pi * 4.17 ** 2 * (273.0 / et) ** (om - 0.5) / math.gamma(2.5 - om); svm = float(np.average(sv, weights=g))     # rate-weighted VHS cross-section
E = np.concatenate([np.arange(0, 0.5, 0.01), np.arange(0.5, 8.001, 0.05)])                                              # in units of T2
rng = np.random.default_rng(11)
# energy-dependent probability (Boyd 1993, written for one molecule), molecule 1 first then molecule 2
c1 = math.gamma(3.5 - om) / math.gamma(3 - om) * math.pi ** 1.5 / 2; c2 = math.gamma(3.5 - om) / math.gamma(2.5 - om) * (math.pi ** 2 / 4 + 2)
ec = et + R["e1_0"]; pb = np.minimum(1.0, (1 + c1 * np.sqrt(tstar / ec) + c2 * tstar / ec) / zinf)
ch = rng.random(n) < pb; bo_e1 = np.where(ch, ec * (1 - rng.random(n) ** (1 / eta)), R["e1_0"])


def exc(e, wt, scale):
    """scale * (weighted share with e > E T2) for every E"""
    k = np.argsort(e); cw = np.cumsum(wt[k]); tot = cw[-1]; idx = np.searchsorted(e[k], E * T2, side="right")
    return scale * (1 - np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0) / tot)


out = dict(E=E, sig_vhs=svm, sigmax=sigmax, T1=T1, T2=T2, n=n, pb_mean=float(np.average(pb, weights=R["wv"])), et_mean=float(np.average(et, weights=g)))
out["ctc"] = exc(R["e1"], w, sigmax)                                   # A^2
out["lb"] = exc(R["lb_e1"], R["wv"], svm); out["boyd"] = exc(bo_e1, R["wv"], svm); out["before"] = exc(R["e1_0"], w, sigmax)
rec = R["v1x"] < 0; out["ctc_rec"] = exc(R["e1"][rec], w[rec], 1.0); recl = R["lb_v1x"] < 0; out["lb_rec"] = exc(R["lb_e1"][recl], R["wv"][recl], 1.0)
# excitation by the size of the deflection: hard (cos chi < 0.9) and grazing collisions
hard = R["cchi"] < 0.9; out["ctc_hard"] = exc(R["e1"][hard], w[hard], sigmax * w[hard].sum() / w.sum()); out["sig_hard"] = float(sigmax * w[hard].sum() / w.sum())
de = R["e1"] - R["e1_0"]; out["gain_ctc"] = float(sigmax * (w * de).sum() / w.sum() / svm)                      # mean gain per VHS collision, K
out["gain_lb"] = float(np.average(R["lb_e1"] - R["e1_0"], weights=R["wv"])); out["gain_boyd"] = float(np.average(bo_e1 - R["e1_0"], weights=R["wv"]))
np.savez(sys.argv[1], **out)
i01 = int(np.argmin(np.abs(E - 0.1))); i03 = int(np.argmin(np.abs(E - 0.3))); i1 = int(np.argmin(np.abs(E - 1.0)))
print(f"{len(files)} files, {n} trajectories; T1 {T1} K, T2 {T2} K; mean relative energy {out['et_mean'] / T2:.2f} T2; VHS cross-section {svm:.1f} A^2; sigma_max {sigmax:.1f} A^2")
for name in ("ctc", "lb", "boyd"):
    print(f"   {name:5s}: cross-section for e1' above 0.1, 0.3, 1 T2: {out[name][i01]:.1f}, {out[name][i03]:.1f}, {out[name][i1]:.1f} A^2 "
          f"({out[name][i01] / svm:.2f}, {out[name][i03] / svm:.2f}, {out[name][i1] / svm:.2f} of the VHS cross-section)")
print(f"   hard collisions (cos chi < 0.9): {out['sig_hard']:.1f} A^2, of which above 0.1 T2: {out['ctc_hard'][i01]:.1f} A^2")
print(f"   mean rotational energy gained per VHS collision / T2: trajectories {out['gain_ctc'] / T2:.3f}, Larsen-Borgnakke (p = 0.2) {out['gain_lb'] / T2:.3f}, energy-dependent {out['gain_boyd'] / T2:.3f} (mean probability {out['pb_mean']:.3f})")
print(f"   recoiled stream molecule above 0.1, 0.3, 1 T2: trajectories {out['ctc_rec'][i01]:.3f}, {out['ctc_rec'][i03]:.3f}, {out['ctc_rec'][i1]:.3f}; Larsen-Borgnakke {out['lb_rec'][i01]:.3f}, {out['lb_rec'][i03]:.3f}, {out['lb_rec'][i1]:.3f}")
