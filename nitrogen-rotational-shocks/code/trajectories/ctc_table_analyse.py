"""Reads the trajectory table of ctc_table.py and expresses the mean rotational energy given to each molecule as the
exchange probability that the Larsen-Borgnakke rule would need, per VHS collision, to give the same mean change:

    p1 = (sigma_max / sigma_VHS(Et)) <e1' - e1> / ((Et + e1)/(1 + eta) - e1),   eta = 5/2 - omega,

and the same for molecule 2. Also the probability that the rotational energy of a molecule changes by more than 10 %
of the relative energy, and the mean deflection-weighted cross-section (viscosity cross-section) next to VHS.

Usage: python ctc_table_analyse.py <out.json> <table npz> [...]"""
import sys, math, json
import numpy as np

om = 0.74; eta = 2.5 - om; MU = 14.0
states = {}
for f in sys.argv[2:]:
    d = np.load(f); grid = d["grid"]; bmax = float(d["bmax"])
    for k in d.files:
        if k[0] == "s" and k[1:].isdigit():
            states.setdefault(int(k[1:]), []).append(d[k])
rows = []
for k in sorted(states):
    a = np.concatenate(states[k], axis=1).astype(float); et, e1, e2 = grid[k]; c, e1f, e2f, etf, b, drift, rfin = a
    ok = (np.abs(drift) < 1e-3) & (rfin > 0); n = int(ok.sum()); sigmax = math.pi * bmax ** 2
    sv = math.pi * 4.17 ** 2 * (273.0 / et) ** (om - 0.5) / math.gamma(2.5 - om); r = sigmax / sv
    d1 = e1f[ok] - e1; d2 = e2f[ok] - e2
    lb1 = (et + e1) / (1 + eta) - e1; lb2 = (et + e2) / (1 + eta) - e2
    p1 = r * d1.mean() / lb1; p2 = r * d2.mean() / lb2; s1 = r * d1.std() / math.sqrt(n) / abs(lb1); s2 = r * d2.std() / math.sqrt(n) / abs(lb2)
    q1 = r * np.mean(np.abs(d1) > 0.1 * et); q2 = r * np.mean(np.abs(d2) > 0.1 * et)
    smu = sigmax * np.mean(1 - c[ok] ** 2 * etf[ok] / et)        # momentum-transfer weighting with the final speed
    rows.append(dict(k=k, Et=float(et), e1=float(e1), e2=float(e2), n=n, p1=float(p1), p2=float(p2), s1=float(s1), s2=float(s2), q1=float(q1), q2=float(q2),
                     d1=float(d1.mean()), d2=float(d2.mean()), sig_mu=float(smu), sig_mu_vhs=float(2 * sv / 3), unusable=int((~ok).sum())))
json.dump(rows, open(sys.argv[1], "w"), indent=1)
ER = sorted({r["e1"] for r in rows} | {r["e2"] for r in rows})
for et in sorted({r["Et"] for r in rows}):
    R = [r for r in rows if r["Et"] == et]; sm = np.mean([r["sig_mu"] for r in R])
    print(f"\nEt = {et:g} K: viscosity cross-section {sm:.1f} A^2 (VHS {R[0]['sig_mu_vhs']:.1f}); exchange probability needed for the molecule in the row, partner in the column (statistical error about {np.median([r['s1'] for r in R]):.3f})")
    print("   rotational energy  " + " ".join(f"{e:8g}" for e in ER) + "    | mean change of its rotational energy / Et")
    for ea in ER:
        line = f"   {ea:8g} K        "; line2 = ""
        for eb in ER:
            v = [(r["p1"], r["d1"]) for r in R if r["e1"] == ea and r["e2"] == eb] + [(r["p2"], r["d2"]) for r in R if r["e2"] == ea and r["e1"] == eb and ea != eb]
            line += f"{v[0][0]:8.3f} " if v else "       - "; line2 += f"{v[0][1] / et:7.3f} " if v else "      - "
        print(line + "   | " + line2)
