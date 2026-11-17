"""Analysis of the classical-trajectory collisions written by ctc_kernel.py (several seeds merged).

Prints, for the stream molecule that leaves the collision moving upstream, the distribution of its rotational energy
from the trajectories and from the Larsen-Borgnakke and Pullin rules on the same pairs; the cross-sections of the
potential next to those of the VHS model; and, against the deflection angle, the differential cross-section, the mean
rotational energy given to the stream molecule and the exchange probability that an isotropic-scattering DSMC needs
at that deflection to give the cold molecule the same rotational energy per unit time.

Usage: python ctc_analyse.py <out prefix> <npz> [<npz> ...]      (writes <prefix>.json and <prefix>_summary.npz)"""
import sys, math, json
import numpy as np

files = sys.argv[2:]; D = [np.load(f) for f in files]
keys = [k for k in D[0].files if not k.startswith("par_")]; R = {k: np.concatenate([d[k] for d in D]) for k in keys}; P = {k[4:]: float(D[0][k]) for k in D[0].files if k.startswith("par_")}
T1, T2, om = P["T1"], P["T2"], P["omega"]; eta = 2.5 - om; MU = 14.0; bmax = P["bmax"]; sigmax = math.pi * bmax ** 2
ok = np.abs(R["drift"]) < 1e-3; n = len(ok); print(f"{len(files)} files, {n} trajectories, {n - ok.sum()} left out (energy change above 1e-3); T1 {T1} K, T2 {T2} K, u1 {P['u1']} m/s")
w = R["w"] * ok; g = R["w"]; et = 0.5 * MU * g ** 2
out = dict(par=P, n=int(n))


def summary(name, vx, e, wt):
    m = vx < 0; ww = wt[m]; ee = e[m] / T2; q = lambda c: float((ww * (ee < c)).sum() / ww.sum()); neff = ww.sum() ** 2 / (ww ** 2).sum()
    mean = float((ww * ee).sum() / ww.sum()); sd = math.sqrt(float((ww * (ee - mean) ** 2).sum() / ww.sum()) / neff)
    print(f"{name:22s}: recoil share {(wt * m).sum() / wt.sum():.4f}; rotational energy of the recoiled stream molecule / T2: mean {mean:.3f} +- {sd:.3f}; "
          f"below 0.1: {q(0.1):.3f}, below 0.3: {q(0.3):.3f}, below 1: {q(1.0):.3f}, above 2: {1 - q(2.0):.3f}")
    return dict(share=float((wt * m).sum() / wt.sum()), mean=mean, se=sd, below01=q(0.1), below03=q(0.3), below1=q(1.0), above2=1 - q(2.0))


out["ctc"] = summary("classical trajectories", R["v1x"], R["e1"], w)
out["lb"] = summary("Larsen-Borgnakke", R["lb_v1x"], R["lb_e1"], R["wv"])
out["pullin"] = summary("Pullin", R["pu_v1x"], R["pu_e1"], R["wv"])
# cross-sections: VHS total cross-section, d_ref = 4.17 A at 273 K
sig_vhs = math.pi * 4.17 ** 2 * (273.0 / et) ** (om - 0.5) / math.gamma(2.5 - om)
print(f"mean relative energy {np.average(et, weights=w) / T2:.2f} T2 = {np.average(et, weights=w):.0f} K; VHS total cross-section (rate-weighted) {np.average(sig_vhs, weights=g):.1f} A^2")
rec = R["v1x"] < 0; print(f"cross-section for sending the stream molecule upstream: trajectories {sigmax * (w * rec).sum() / w.sum():.2f} A^2; "
                          f"VHS, isotropic {np.average(sig_vhs, weights=g) * out['lb']['share']:.2f} A^2")
out["sig_recoil_ctc"] = float(sigmax * (w * rec).sum() / w.sum()); out["sig_recoil_vhs"] = float(np.average(sig_vhs, weights=g) * out["lb"]["share"])
# the upstream-moving partner after the collision (molecule 2 stays upstream-moving)
if "cchi" in R:
    c = R["cchi"]; e2_0 = R["e2_0"]; e1_0 = R["e1_0"]
    smu = sigmax * (w * (1 - c ** 2)).sum() / w.sum(); smu_vhs = 2.0 / 3.0 * np.average(sig_vhs, weights=g)
    print(f"viscosity cross-section 2 pi int sin^2(chi) b db: trajectories {smu:.1f} A^2; VHS {smu_vhs:.1f} A^2")
    outer = R["b"] > 0.9 * bmax; print(f"share of the viscosity cross-section from b > 0.9 bmax: {(w * (1 - c ** 2) * outer).sum() / (w * (1 - c ** 2)).sum():.4f}; "
                                       f"share of the rotational energy given to the stream molecule from b > 0.9 bmax: {(w * (R['e1'] - e1_0) * outer).sum() / (w * (R['e1'] - e1_0)).sum():.4f}")
    out.update(sig_mu_ctc=float(smu), sig_mu_vhs=float(smu_vhs))
    edges = np.linspace(-1, 1, 21); cc = 0.5 * (edges[1:] + edges[:-1]); k = np.clip(np.digitize(c, edges) - 1, 0, 19)
    W = np.bincount(k, w, 20); dsig = sigmax * W / w.sum() / (2 * math.pi * (edges[1] - edges[0]))                 # A^2 / sr
    de = R["e1"] - e1_0; mde = np.bincount(k, w * de, 20) / W; me1 = np.bincount(k, w * R["e1"], 20) / W
    p03 = np.bincount(k, w * (R["e1"] > 0.3 * T2), 20) / W; p01 = np.bincount(k, w * (R["e1"] > 0.1 * T2), 20) / W
    ec1 = np.bincount(k, w * (et + e1_0), 20) / W                                                                   # energy available to the LB redistribution
    vhs_sr = np.average(sig_vhs, weights=g) / (4 * math.pi)
    # exchange probability for an isotropic-scattering model: same rotational energy gained per unit time and solid angle
    pneed = dsig * mde / (vhs_sr * (ec1 / (1 + eta) - np.bincount(k, w * e1_0, 20) / W))
    print("cos(chi)   dsigma/dOmega [A^2/sr] (VHS %.2f)   <e1'>/T2   P(e1' > 0.1 T2)   P(e1' > 0.3 T2)   exchange probability needed with isotropic scattering" % vhs_sr)
    for i in range(20):
        print(f"{cc[i]:6.2f} {dsig[i]:12.3f} {me1[i] / T2:22.3f} {p01[i]:12.3f} {p03[i]:16.3f} {pneed[i]:18.3f}")
    print(f"mean exchange probability needed (isotropic average): {pneed.mean():.3f}  (constant Zrot = 5 corresponds to 0.2)")
    # both molecules, by deflection: mean energy given to rotation of the two molecules as a fraction of the relative energy
    f2 = np.bincount(k, w * (R["e1"] + R["e2"] - e1_0 - e2_0) / et, 20) / W
    np.savez(sys.argv[1] + "_summary.npz", cos=cc, dsig=dsig, me1=me1 / T2, p01=p01, p03=p03, pneed=pneed, frac_rot=f2, vhs_sr=vhs_sr,
             hist_edges=np.linspace(0, 6, 121), hist_ctc=np.histogram(R["e1"][rec] / T2, np.linspace(0, 6, 121), weights=w[rec])[0] / w[rec].sum(),
             hist_lb=np.histogram(R["lb_e1"][R["lb_v1x"] < 0] / T2, np.linspace(0, 6, 121), weights=R["wv"][R["lb_v1x"] < 0])[0] / R["wv"][R["lb_v1x"] < 0].sum(),
             hist_pu=np.histogram(R["pu_e1"][R["pu_v1x"] < 0] / T2, np.linspace(0, 6, 121), weights=R["wv"][R["pu_v1x"] < 0])[0] / R["wv"][R["pu_v1x"] < 0].sum())
    out.update(cos=cc.tolist(), dsig=dsig.tolist(), me1=(me1 / T2).tolist(), pneed=pneed.tolist(), vhs_sr=float(vhs_sr))
    # the partner: molecule 2 was moving upstream; state of those that still move upstream after the collision
    v2x = None
json.dump(out, open(sys.argv[1] + ".json", "w"), indent=1)
