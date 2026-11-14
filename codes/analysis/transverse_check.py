"""Test of the three-population description beyond the reduced distributions g and h.

The fit uses g = int f dxi_y dxi_z and h = int xi_perp^2 f. The DSMC runs of dsmc_shock_vdf4.cpp also sample
    h2(x, xi_x) = int xi_perp^4 f dxi_y dxi_z        (xi_perp^2 = xi_y^2 + xi_z^2)
and the moments <xi_x^2 xi_perp^2>, <xi_perp^4>. A mixture of bi-Maxwellian populations predicts
    h2 = sum_k 8 T_perp,k^2 g_k ,    int c^2 xi_perp^2 f = sum_k 2 n_k T_perp,k (T_par,k + d_k^2) ,
none of which entered the fit. The populations are those fitted earlier to the eight-seed data (popfits/), the
distributions come from independent seeds; stations are matched by the normalised density.

Usage: python transverse_check.py <runs_h2 dir> <popfit dir> <out json> [min_steps]
"""
import sys, os, glob, re, json
import numpy as np

GAM = 5.0 / 3.0


def rh(M):
    r = (GAM + 1) * M ** 2 / ((GAM - 1) * M ** 2 + 2); p = (2 * GAM * M ** 2 - (GAM - 1)) / (GAM + 1); u1 = M * np.sqrt(GAM)
    return dict(u1=u1, u2=u1 / r, T2=p / r, rho2=r)


def gk_of(p, v):
    q = p.reshape(-1, 4); n, u, a, b = np.exp(q[:, 0]), q[:, 1], np.exp(q[:, 2]), np.exp(q[:, 3])
    g = n[:, None] * np.exp(-(v[None, :] - u[:, None]) ** 2 / (2 * a[:, None])) / np.sqrt(2 * np.pi * a[:, None])
    return g, n, u, a, b


def main(runs, fitdir, outj, min_steps=215000):
    cases = sorted({re.sub(r"_s\d+\.txt$", "", os.path.basename(p)) for p in glob.glob(os.path.join(runs, "M*_w*_s*.txt"))})
    R = {}
    for case in cases:
        tot = None
        for stem in sorted(p[:-4] for p in glob.glob(os.path.join(runs, case + "_s*.txt"))):
            h = dict(l.split() for l in open(stem + ".txt") if l.strip())
            if int(h["steps_done"]) < min_steps or not os.path.exists(stem + ".h2.bin"):
                continue
            nc, nb, NB, NM = (int(h[k]) for k in ("nc", "nb", "NB", "NM"))
            a = np.fromfile(stem + ".bin"); cnt = a[:nc * nb].reshape(nc, nb); hs = a[nc * nb:2 * nc * nb].reshape(nc, nb)
            mom = a[2 * nc * nb:2 * nc * nb + NB * nc * NM].reshape(NB, nc, NM).sum(0); ns = a[-NB:].sum()
            h2 = np.fromfile(stem + ".h2.bin").reshape(nc, nb)
            cur = [cnt, hs, h2, mom, ns]
            tot = cur if tot is None else [x + y for x, y in zip(tot, cur)]
        if tot is None:
            continue
        cnt, hs, h2, mom, ns = tot
        M, om, dx, dv, FN = (float(h[k]) for k in ("M", "omega", "dx", "dv", "FN")); st = rh(M)
        v = float(h["vlo"]) + (np.arange(nb) + 0.5) * dv; sc = FN / (dx * dv * ns)
        G, H, H2 = cnt * sc, hs * sc, h2 * sc
        N = mom[:, 0]; rho = N * FN / (dx * ns); u = mom[:, 1] / N
        nstar = (rho - 1) / (st["rho2"] - 1)
        Tpar = mom[:, 2] / N - u ** 2; Tperp = 0.5 * mom[:, 5] / N; T = (Tpar + 2 * Tperp) / 3
        m4p = rho * mom[:, 8] / N                                             # int xi_perp^4 f
        mix = rho * (mom[:, 7] / N - 2 * u * mom[:, 6] / N + u ** 2 * mom[:, 5] / N)   # int c^2 xi_perp^2 f
        m4p_neq, mix_neq = m4p - 8 * rho * T ** 2, mix - 2 * rho * T ** 2
        core = np.where((nstar > 0.005) & (nstar < 0.995))[0]; kk = np.arange(core.min() - 1, core.max() + 2); nsm = np.maximum.accumulate(nstar[kk])

        def at(lv, arr):
            j = max(int(np.argmax(nsm >= lv)), 1); a_ = (lv - nsm[j - 1]) / (nsm[j] - nsm[j - 1])
            return (1 - a_) * arr[kk[j - 1]] + a_ * arr[kk[j]]

        tag = f"DSMC_w{om:g}_M{M:g}".replace(".", "p")
        rec = dict(mach=M, omega=om)
        for K, fname, key in ((3, tag + "_branch3.npz", "p3"), (2, tag + "_popfit.npz", "p2")):
            fp = os.path.join(fitdir, fname)
            if not os.path.exists(fp):
                continue
            f = np.load(fp); ok = np.all(np.isfinite(f[key]), 1) & (f["nstar"] > 0.02) & (f["nstar"] < 0.98)
            res = dict(g=[], h=[], h2=[], m4p=[], mix=[], lv=[])
            for p, lv in zip(f[key][ok], f["nstar"][ok]):
                gk, n, uk, a_, b_ = gk_of(p, v)
                g, hh, hh2 = at(lv, G), at(lv, H), at(lv, H2)
                nrm = lambda x: np.sqrt(np.sum(x ** 2))
                res["g"].append(nrm(gk.sum(0) - g) / nrm(g)); res["h"].append(nrm((2 * b_[:, None] * gk).sum(0) - hh) / nrm(hh))
                res["h2"].append(nrm((8 * b_[:, None] ** 2 * gk).sum(0) - hh2) / nrm(hh2))
                rr = n.sum(); U = (n * uk).sum() / rr; d = uk - U
                res["m4p"].append((8 * n * b_ ** 2).sum() - at(lv, m4p)); res["mix"].append((2 * n * b_ * (a_ + d ** 2)).sum() - at(lv, mix)); res["lv"].append(float(lv))
            e4, em = np.abs(res["m4p"]), np.abs(res["mix"]); c = core
            rec[f"K{K}"] = dict(g_max=float(100 * np.max(res["g"])), h_max=float(100 * np.max(res["h"])), h2_max=float(100 * np.max(res["h2"])),
                                h2_median=float(100 * np.median(res["h2"])), h2_worst_nstar=float(res["lv"][int(np.argmax(res["h2"]))]),
                                m4perp_err_over_peak=float(100 * e4.max() / np.abs(m4p[c]).max()), m4perp_err_over_neq_peak=float(100 * e4.max() / np.abs(m4p_neq[c]).max()),
                                mixed_err_over_peak=float(100 * em.max() / np.abs(mix[c]).max()), mixed_err_over_neq_peak=float(100 * em.max() / np.abs(mix_neq[c]).max()))
        rec["neq_over_total"] = dict(m4perp=float(np.abs(m4p_neq[core]).max() / np.abs(m4p[core]).max()), mixed=float(np.abs(mix_neq[core]).max() / np.abs(mix[core]).max()))
        R[tag] = rec
        print(tag, {k: ({a: round(b, 2) for a, b in x.items()} if isinstance(x, dict) else x) for k, x in rec.items()}, flush=True)
    json.dump(R, open(outj, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 215000)
