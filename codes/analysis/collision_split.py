"""Stress and heat flux of the DSMC shocks split by collision history.

The distribution is the sum of the part carried by the unscattered stream (generation 0; the ballistic part in the
sense of the decomposition of Vargas et al. 2014 and Tatsios et al. 2015) and the part carried by scattered molecules
(generations 1, 2, 3+; the collision part). Every moment taken about the mean velocity u of the whole gas is then a sum
over the parts:

    sigma = sum_k sigma_k,   sigma_k = (2/3) n_k [ <c_x^2>_k - <c_perp^2>_k / 2 ],     c = xi - u
    q     = sum_k q_k,       q_k     = (1/2) n_k [ <c_x^3>_k + <c_x c_perp^2>_k ]

The stress needs the class sums of 1, v, v^2, w^2 (version 2 of the DSMC code); the heat flux needs also v^3 and
v w^2 (version 6, array `mraw` in the class files).

Usage: python collision_split.py <shock_data dir> [class dirs, default: vdf_cls vdf_split; later ones take precedence]
Writes analysis/out/collision_split.json.
"""
import sys, os, json, glob, re
import numpy as np

LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


def split(f):
    d = np.load(f)
    st = json.loads(str(d["states"])); u1 = st["u1"]
    x, ns = d["x"], d["nstar"]
    if "mraw" in d.files:                                   # n <1, v, v^2, w^2, v^3, v w^2> of generations 0..3+
        m = d["mraw"][5:]
        m0, m1, m2, mw = m[:, :, 0], m[:, :, 1], m[:, :, 2], m[:, :, 3]
    else:
        mg = np.nan_to_num(d["mgen"]); m0 = mg[:, :, 0]; m1 = m0 * mg[:, :, 1]
        m2 = m0 * (mg[:, :, 2] + mg[:, :, 1] ** 2); mw = m0 * 2 * mg[:, :, 3]; m = None
    n = m0.sum(0); u = m1.sum(0) / n
    pxx = m2 - 2 * u * m1 + u ** 2 * m0                     # n_k <c_x^2>_k
    sig = (2.0 / 3.0) * (pxx - 0.5 * mw) / u1 ** 2          # in units of rho1 u1^2
    out = dict(x=x, nstar=ns, share=m0 / n, sigma=sig)
    if m is not None and m.shape[2] >= 6:
        m3, mvw = m[:, :, 4], m[:, :, 5]
        q = 0.5 * (m3 - 3 * u * m2 + 3 * u ** 2 * m1 - u ** 3 * m0 + mvw - u * mw)
        out["q"] = q / (0.5 * u1 ** 3)                      # in units of rho1 u1^3 / 2
    return out


def at(ns, y, lv):
    j = int(np.argmin(np.abs(ns - lv))); return float(y[j])


def main(root, subs=("vdf_cls", "vdf_split")):
    res = {}; files = {}
    for sub in subs:                                        # later directories take precedence
        for f in sorted(glob.glob(os.path.join(root, sub, "DSMC_w*_M*_classes.npz"))):
            files[re.sub(r"_classes\.npz$", "", os.path.basename(f))[5:]] = f
    for case, f in sorted(files.items()):
        if case.startswith("w0p66") or case.startswith("w0p7_") or "M7p18" in case:
            continue
        s = split(f); ns = s["nstar"]; sg = s["sigma"]; tot = sg.sum(0); js = int(np.argmax(tot)); dx = s["x"][1] - s["x"][0]
        r = dict(sigma_peak=float(tot[js]), nstar_sigma_peak=float(ns[js]),
                 sigma_share_by_generation_at_peak=[float(v / tot[js]) for v in sg[:, js]],
                 mass_share_by_generation_at_peak=[float(v) for v in s["share"][:, js]],
                 sigma_ballistic_share_integral=float(sg[0].sum() / tot.sum()),
                 sigma_ballistic_share={str(lv): at(ns, sg[0] / tot, lv) for lv in LEVELS},
                 mass_ballistic_share={str(lv): at(ns, s["share"][0], lv) for lv in LEVELS})
        line = f"{case:12s} sigma peak {r['sigma_peak']:.4f} at n*={r['nstar_sigma_peak']:.2f}; ballistic share of sigma there {r['sigma_share_by_generation_at_peak'][0]:.2f} (mass {r['mass_share_by_generation_at_peak'][0]:.2f}); gen 1: {r['sigma_share_by_generation_at_peak'][1]:+.2f} (mass {r['mass_share_by_generation_at_peak'][1]:.2f}); integral share {r['sigma_ballistic_share_integral']:.2f}"
        if "q" in s:
            q = s["q"]; qt = q.sum(0); jq = int(np.argmin(qt))
            r.update(q_peak=float(qt[jq]), nstar_q_peak=float(ns[jq]), q_ballistic_at_peak=float(q[0, jq]), q_collision_at_peak=float(q[1:, jq].sum()),
                     q_by_generation_at_peak=[float(v) for v in q[:, jq]],
                     q_ballistic_max=float(q[0].max()), nstar_q_ballistic_max=float(ns[int(np.argmax(q[0]))]),
                     q_ballistic_integral=float(q[0].sum() * dx), q_collision_integral=float(q[1:].sum() * dx),
                     q_ballistic={str(lv): at(ns, q[0], lv) for lv in LEVELS}, q_collision={str(lv): at(ns, q[1:].sum(0), lv) for lv in LEVELS})
            line += f" | q peak {r['q_peak']:+.4f} = ballistic {r['q_ballistic_at_peak']:+.4f} + collision {r['q_collision_at_peak']:+.4f}; max ballistic q {r['q_ballistic_max']:+.4f} at n*={r['nstar_q_ballistic_max']:.2f}"
        res[case] = r; print(line, flush=True)
    json.dump(res, open(os.path.join(root, "analysis", "out", "collision_split.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], tuple(sys.argv[2:]) or ("vdf_cls", "vdf_split"))
