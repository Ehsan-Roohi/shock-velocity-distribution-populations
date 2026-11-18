"""Shock runs with the ab initio kernel (tag v19ai) against the main runs with the two-centre Lennard-Jones kernel
(tag rin): the numbers of the sentence on the ab initio shock simulations in the Supplemental Material.
  level populations: ratio of the two predictions over the populations of the Letter's comparison (resolved levels at
                     the stations with n* above 0.02 at Mach 7.0 and 0.13 at Mach 12.9), and the rms deviation of each
                     from the measurement, pooled and per seed;
  tail slopes:       station by station, slope temperatures (1 to 3 kT2) of the groups with one, two, three or more
                     collisions and of all molecules over the measured hot line (definitions of make_text_numbers.py).
The profile separation and the decay length of the upstream-moving fraction are printed by
code/paper/make_text_numbers.py (python make_text_numbers.py <data> rin v19ai).

Usage: python ai_vs_lj_shock.py <data folder> [lj tag, default rin]   (after rt_predict_rin.py and rt_predict_ai.py)"""
import sys, os, json, math
import numpy as np

N2 = sys.argv[1]; LJ = sys.argv[2] if len(sys.argv) > 2 else "rin"; sys.path.insert(0, N2)
import robben_talbot_1966 as rt
TH = rt.THETA; NMIN = {"7": 0.02, "12.9": 0.13}
L = lambda f: json.load(open(os.path.join(N2, f)))


def sel(recs, m): return [r for r in recs if r["kp"] > 0 and r["nstar"] > NMIN[m]]


def rms(recs, m):
    q = np.concatenate([np.array(r["pred"][:r["kp"]]) / np.array(r["meas"][:r["kp"]]) for r in sel(recs, m)])
    return 100 * np.sqrt(np.mean((q - 1) ** 2))


def stations(key):
    rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); en = (ex[:, 2] - ex[0, 2]) / (ex[-3:, 2].mean() - ex[0, 2]); return rows, ex, en, ex[-1, 1]


def slopes(tag, lab, key):
    V = np.load(os.path.join(N2, f"erh_{tag}.npz")); n = V[f"{lab}_n"]; ns = (n - 1) / (float(V[f"{lab}_n2"]) - 1); erh = V[f"{lab}_erh"]
    rows, ex, en, T2 = stations(key); out = {9: [], 10: [], 11: [], "all": []}; st = []
    for j, r_ in enumerate(rows):
        kp = len(r_[2])
        if not (NMIN[key] < en[j] < 0.97) or kp == 0: continue
        i = int(np.argmin(np.abs(ns - en[j]))); s = slice(max(i - 2, 0), i + 3); st.append(round(float(en[j]), 2))
        for g in (9, 10, 11):
            c = erh[g, s].sum(0)[::-1].cumsum()[::-1]; out[g].append(2 / math.log(c[20] / c[60]))
        e1 = TH * kp * (kp + 1) / T2; c = erh[0, s].sum(0)[::-1].cumsum()[::-1]; E0 = np.arange(160) * 0.05
        lc = lambda e: np.interp(e, E0, np.log(np.maximum(c, 1e-300))); out["all"].append(1.5 / (lc(e1) - lc(e1 + 1.5)))
    return st, {k: np.array(v) for k, v in out.items()}


lj, ai = L(f"rt_predict_{LJ}.json"), L("rt_predict_v19ai.json"); ljs, ais = L(f"rt_predict_{LJ}_seeds.json"), L("rt_predict_v19ai_seeds.json")
for m in ("7", "12.9"):
    a, b = sel(ai[f"M{m}k"], m), sel(lj[f"M{m}k"], m)
    q = 100 * np.concatenate([np.array(x["pred"][:x["kp"]]) / np.array(y["pred"][:y["kp"]]) - 1 for x, y in zip(a, b)])
    print(f"Mach {m}: ab initio / Lennard-Jones - 1 over {len(q)} populations: rms {np.sqrt(np.mean(q ** 2)):.1f}%, "
          f"largest {np.abs(q).max():.1f}%, mean {q.mean():+.1f}%")
    print(f"   rms from the measurement: Lennard-Jones {rms(lj[f'M{m}k'], m):.1f}% (seeds "
          + " ".join(f"{rms(ljs[f'M{m}s{s}'], m):.1f}" for s in (11, 12, 13, 14)) + f"), ab initio {rms(ai[f'M{m}k'], m):.1f}% (seeds "
          + " ".join(f"{rms(ais[f'M{m}s{s}'], m):.1f}" for s in (11, 12, 13, 14)) + ")")
    st, sa = slopes("v19ai", f"M{m}k", m); _, sb = slopes(LJ, f"M{m}k", m)
    print(f"   tail slopes at n* = {st}, change in %:")
    for g, nm in ((9, "one collision"), (10, "two"), (11, "three or more"), ("all", "all, hot line")):
        d = 100 * (sa[g] / sb[g] - 1); print(f"      {nm:14s} " + " ".join(f"{v:+.1f}" for v in d) + f"   largest {np.abs(d).max():.1f}")
