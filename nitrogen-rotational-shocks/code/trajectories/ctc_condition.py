"""Single collision of a stream molecule, resolved by what went in (classical trajectories of ctc_kernel.py):

  * cross-section for a GAIN of rotational energy above E (e' - e > E), for crossing the level E from below
    (e <= E < e'), and for being above E before the collision, all in units of the rate-averaged VHS cross-section;
  * the same excitation function for collisions in narrow bins of the relative translational energy Et before the
    collision, in units of the VHS cross-section of the bin: against E/kT2 and against the share E/Etot of the total
    energy of the pair; mean gain per VHS collision and local slope in each bin; the distribution of Et over the VHS
    collisions of the ensemble;
  * the same split by the rotational state of the partner (cold or hot).

Usage: python ctc_condition.py <out.npz> <raw npz> [...]"""
import sys, math
import numpy as np

files = sys.argv[2:]; D = [np.load(f) for f in files]; P = {k[4:]: float(D[0][k]) for k in D[0].files if k.startswith("par_")}
R = {k: np.concatenate([d[k] for d in D]) for k in ("e1", "e1_0", "e2_0", "w", "drift")}
T2, om = P["T2"], P["omega"]; MU = 14.0; sigmax = math.pi * P["bmax"] ** 2
g = R["w"]; ok = np.abs(R["drift"]) < 1e-3; w = g * ok; et = 0.5 * MU * g ** 2
sv = math.pi * 4.17 ** 2 * (273.0 / et) ** (om - 0.5) / math.gamma(2.5 - om); svm = float(np.average(sv, weights=g))
E = np.concatenate([np.arange(0, 0.5, 0.01), np.arange(0.5, 8.001, 0.05)]); S = np.linspace(0, 1, 201)
e1, e0, e2 = R["e1"], R["e1_0"], R["e2_0"]; etot = et + e0 + e2; res = dict(E=E, S=S, T2=T2, sig_vhs=svm, sigmax=sigmax)


def above(x, wt, grid):
    """weighted share of the trajectories with x above each grid value"""
    k = np.argsort(x); cw = np.cumsum(wt[k]); idx = np.searchsorted(x[k], grid, side="right")
    return 1 - np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0) / cw[-1]


i = lambda x: int(np.argmin(np.abs(E - x)))
res["after"] = sigmax * above(e1, w, E * T2) / svm; res["gain"] = sigmax * above(e1 - e0, w, E * T2) / svm; res["before"] = sigmax * above(e0, w, E * T2) / svm
cross = np.array([np.sum(w[(e1 > x * T2) & (e0 <= x * T2)]) for x in E]) / w.sum(); res["cross"] = sigmax * cross / svm
print(f"{len(g)} trajectories, T2 = {T2:g} K, rate-averaged VHS cross-section {svm:.2f} A^2, sampled {sigmax:.1f} A^2; mean Et/kT2 over VHS collisions {np.average(et, weights=g * sv) / T2:.2f}")
print("cross-section / VHS at E = 0.1, 0.3, 1, 2, 3 kT2:")
for nm in ("after", "before", "cross", "gain"):
    print(f"   {nm:7s}" + " ".join(f"{res[nm][i(x)]:8.3f}" for x in (0.1, 0.3, 1, 2, 3)) + (f"   slope 1-3: {2 / math.log(res[nm][i(1)] / res[nm][i(3)]):.3f} T2" if nm != "before" else ""))
print(f"mean gain per VHS collision {sigmax * np.average(e1 - e0, weights=w) / svm / T2:.3f} kT2; share of the sampled trajectories with a gain above 0.1 kT2: {above(e1 - e0, w, np.array([0.1 * T2]))[0]:.3f}")

edges = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 20, 1e9]) * T2; res["et_edges"] = edges / T2
rate = g * sv; print("bins of Et/kT2: share of the VHS collisions | mean Et | sigma(e'>E)/sigma_VHS(bin) at 0.3, 1, 2, 3 kT2 | slope 1-3 | mean gain per VHS collision /kT2 and /Et | share s = e'/Etot above 0.1, 0.2, 0.3, 0.5")
B = []
for a, b in zip(edges[:-1], edges[1:]):
    m = (et >= a) & (et < b)
    if m.sum() < 2000:
        B.append(None); continue
    svb = float(np.average(sv[m], weights=g[m])); fE = sigmax * above(e1[m], w[m], E * T2) / svb; fS = sigmax * above(e1[m] / etot[m], w[m], S) / svb
    gn = sigmax * np.average(e1[m] - e0[m], weights=w[m]) / svb; me = float(np.average(et[m], weights=rate[m]))
    sl = 2 / math.log(fE[i(1)] / fE[i(3)]) if fE[i(3)] > 0 else float("nan")
    B.append(dict(share=float(rate[m].sum() / rate.sum()), et=me / T2, fE=fE, fS=fS, gain=gn / T2, slope=sl))
    print(f"   {a / T2:4.0f}-{min(b / T2, 99):4.0f}: {B[-1]['share']:.3f} | {me / T2:5.2f} | " + " ".join(f"{fE[i(x)]:.3f}" for x in (0.3, 1, 2, 3)) + f" | {sl:.3f} | {gn / T2:.3f} {gn / me:.3f} | "
          + " ".join(f"{fS[int(round(x * 200))]:.3f}" for x in (0.1, 0.2, 0.3, 0.5)))
res["bin_share"] = np.array([b["share"] if b else np.nan for b in B]); res["bin_et"] = np.array([b["et"] if b else np.nan for b in B])
res["bin_gain"] = np.array([b["gain"] if b else np.nan for b in B]); res["bin_slope"] = np.array([b["slope"] if b else np.nan for b in B])
res["bin_fE"] = np.array([b["fE"] if b else np.full(len(E), np.nan) for b in B]); res["bin_fS"] = np.array([b["fS"] if b else np.full(len(S), np.nan) for b in B])
h, he = np.histogram(et / T2, bins=np.arange(0, 20.01, 0.25), weights=rate); res["et_hist"] = h / h.sum() / 0.25; res["et_hist_x"] = 0.5 * (he[1:] + he[:-1])
for nm, m in (("cold partner (e2 < 0.3 kT2)", e2 < 0.3 * T2), ("hot partner (e2 > 0.3 kT2)", e2 >= 0.3 * T2)):
    svb = float(np.average(sv[m], weights=g[m])); fE = sigmax * above(e1[m], w[m], E * T2) / svb; res["partner_" + nm.split()[0]] = fE
    print(f"{nm}: share of the VHS collisions {rate[m].sum() / rate.sum():.3f}; sigma(e'>E)/sigma_VHS at 0.3, 1, 2, 3 kT2 " + " ".join(f"{fE[i(x)]:.3f}" for x in (0.3, 1, 2, 3))
          + f"; slope 1-3 {2 / math.log(fE[i(1)] / fE[i(3)]):.3f} T2; mean gain per VHS collision {sigmax * np.average(e1[m] - e0[m], weights=w[m]) / svb / T2:.3f} kT2")
np.savez(sys.argv[1], **res)
