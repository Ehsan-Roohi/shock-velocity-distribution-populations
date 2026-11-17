"""Numbers quoted in the text of the letter and of the supplemental material that are not read off a table, computed
from the result files so that none is typed by hand.

Usage: python make_text_numbers.py <n2 folder> v17|rin [tag]
  v17: kernel runs of version 17 (Boltzmann stream; erh_v17.npz, first_v17.npz, rt_predict_ktab.json, ...)
  rin: runs whose stream enters with the measured free-stream distribution (erh_rin.npz, first_rin.npz, ...)"""
import sys, os, json, math
import numpy as np

N2, SRC = sys.argv[1], sys.argv[2]; sys.path.insert(0, N2)
TAG = sys.argv[3] if len(sys.argv) > 3 else "rin"   # files of another set with the measured stream, e.g. v19 (label-based freeze)
import robben_talbot_1966 as rt
RIN = SRC == "rin"; TH = rt.THETA
V = np.load(os.path.join(N2, f"erh_{TAG}.npz" if RIN else "erh_v17.npz")); FR = np.load(os.path.join(N2, f"first_{TAG}.npz" if RIN else "first_v17.npz"))
if RIN:
    PR = json.load(open(os.path.join(N2, f"rt_predict_{TAG}.json"))); PRL = PR; XK = XL = XC = json.load(open(os.path.join(N2, f"exch_{TAG}.json")))
else:
    PR = json.load(open(os.path.join(N2, "rt_predict_ktab.json"))); PRL = json.load(open(os.path.join(N2, "rt_predict_lb.json")))
    XK = json.load(open(os.path.join(N2, "exch_ktab.json"))); XL = json.load(open(os.path.join(N2, "exch_ref.json"))); XC = json.load(open(os.path.join(N2, "exch_lb12.json")))
PRE = np.load(os.path.join(N2, f"letter_precursor_{TAG}.npz" if RIN else "letter_precursor.npz"))
PREK = np.load(os.path.join(N2, f"letter_precursor_k_{TAG}.npz" if RIN else "letter_precursor_k.npz"))
EXP = {"7": dict(n=0.158, lag=1.9), "12.9": dict(n=0.167, lag=2.54)}
NMIN = {"7": 0.02, "12.9": 0.13}


def stations(key):
    rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); en = (ex[:, 2] - ex[0, 2]) / (ex[-3:, 2].mean() - ex[0, 2]); return rows, ex, en, ex[-1, 1]


def cells(lab, nstar):
    n = V[f"{lab}_n"]; ns = (n - 1) / (float(V[f"{lab}_n2"]) - 1); i = int(np.argmin(np.abs(ns - nstar))); return slice(max(i - 2, 0), i + 3)


print(f"================ source: {SRC}")
# ---------------------------------------------------------------- hot group against the counts (Fig. 2)
for lab, key in (("M7k", "7"), ("M12.9k", "12.9")):
    n, erh = V[f"{lab}_n"], V[f"{lab}_erh"]; ns = (n - 1) / (float(V[f"{lab}_n2"]) - 1); t = erh.sum(-1); tot = np.maximum(t[0], 1e-30)
    m = (ns > 0.005) & (ns < 0.99) & (np.arange(len(ns)) < np.argmax(ns > 0.99) + 1 if (ns > 0.99).any() else True); k = np.argsort(ns[m]); X = ns[m][k]
    s1 = (1 - t[2] / tot)[m][k]; s2 = ((t[10] + t[11]) / tot)[m][k]; s3 = (t[11] / tot)[m][k]
    rows, ex, en, T2 = stations(key); s = (en > 0.02) & (en < 0.97)
    r = [np.sqrt(np.mean((np.interp(en[s], X, v) - ex[s, 4]) ** 2)) for v in (s1, s2, s3)]; dmax = np.max(np.interp(en[s], X, s1) - ex[s, 4])
    print(f"Fig2 {lab}: rms from measured hot: >=1 {r[0]:.3f}, >=2 {r[1]:.3f}, >=3 {r[2]:.3f}; largest excess of >=1 over the data {dmax:.2f}")
# ---------------------------------------------------------------- groups at the stations (Fig. 3, Table counts)
for lab, key in (("M7k", "7"), ("M12.9k", "12.9")):
    rows, ex, en, T2 = stations(key); erh = V[f"{lab}_erh"]; erl = V[f"{lab}_erl"]; sl = {9: [], 10: [], 11: []}; h3 = []; band = []; s12 = []; hot = []; hotm = []; st = []
    for j, r_ in enumerate(rows):
        kp = len(r_[2])
        if not (NMIN[key] < en[j] < 0.97) or kp == 0: continue
        s = cells(lab, en[j]); tot = erh[0, s].sum(); st.append(en[j])
        for g in (9, 10, 11):
            c = erh[g, s].sum(0)[::-1].cumsum()[::-1]; sl[g].append(2 / math.log(c[20] / c[60]))
        ie = int(TH * kp * kp / T2 / 0.05); hh = [erh[g, s].sum(0)[ie:].sum() for g in (2, 9, 10, 11)]; h3.append(hh[3] / sum(hh))
        # band 0.1-0.8 kT2 from the fine bins (0.005 T2): share supplied by one or two collisions
        b = [erl[g, s].sum(0)[20:160].sum() for g in (0, 9, 10)]; band.append((b[1] + b[2]) / b[0]); s12.append((erh[9, s].sum() + erh[10, s].sum()) / tot)
        # slope of all molecules over the measured hot line: from the energy of level K' over 1.5 kT2
        e1 = TH * kp * (kp + 1) / T2; c = erh[0, s].sum(0)[::-1].cumsum()[::-1]; E0 = np.arange(160) * 0.05
        lc = lambda e: np.interp(e, E0, np.log(np.maximum(c, 1e-300))); hot.append(1.5 / (lc(e1) - lc(e1 + 1.5))); hotm.append(r_[1] / T2)
    f = lambda v: f"{min(v):.2f}-{max(v):.2f}"
    print(f"Groups {lab} at n* = " + " ".join(f"{v:.2f}" for v in st))
    print(f"   slopes (1-3 kT2): one collision {f(sl[9])}, two {f(sl[10])}, three or more {f(sl[11])}; 3+ share of the hot line {100 * min(h3):.0f}-{100 * max(h3):.0f}%")
    print(f"   one or two collisions: share of the gas " + " ".join(f"{v:.2f}" for v in s12) + "; share of the 0.1-0.8 kT2 band " + " ".join(f"{100 * v:.0f}%" for v in band))
    print(f"   slope over the hot line, all molecules: " + " ".join(f"{v:.2f}" for v in hot) + " | measured T': " + " ".join(f"{v:.2f}" for v in hotm))
    fr = [v for v, e in zip(hot, st) if e < 0.4]; fm = [v for v, e in zip(hotm, st) if e < 0.4]
    print(f"   where n* < 0.4: simulated {f(fr)}, measured {f(fm)}")
# ---------------------------------------------------------------- level populations
for lab, key in (("M7k", "7"), ("M12.9k", "12.9")):
    for nm, D, l_ in (("kernel", PR, lab), ("LB p(Ec)", PRL, lab.replace("k", "zi10"))):
        recs = [r for r in D[l_] if r["kp"] > 0 and r["nstar"] > NMIN[key]]; q = np.concatenate([np.array(r["pred"][:r["kp"]]) / np.array(r["meas"][:r["kp"]]) for r in recs])
        E = np.concatenate([TH * np.arange(r["kp"]) * (np.arange(r["kp"]) + 1) / (296 if key == "7" else 305) for r in recs])
        lo = q[E < 0.1]; mid = q[(E > 0.3) & (E < 1.5)]
        print(f"Levels {l_} ({nm}): {len(q)} populations, ratio {q.min():.2f}-{q.max():.2f}, rms {100 * np.sqrt(np.mean((q - 1) ** 2)):.1f}%; "
              f"mean ratio below 0.1 kT2 {lo.mean():.2f}, between 0.3 and 1.5 kT2 {mid.mean():.2f}")
# ---------------------------------------------------------------- profiles
for key, lk, ll, lc in (("7", "M7k", "M7zi10", "M7c"), ("12.9", "M12.9k", "M12.9zi10", "M12.9c")):
    e = EXP[key]; a, b, c = XK[lk], XL[ll], XC[lc]
    print(f"Profiles Mach {key}: lead p=0.2 {c['lag']:.2f}, p(Ec) {b['lag']:.2f}, kernel {a['lag']:.2f} ({100 * (a['lag'] / e['lag'] - 1):.0f}% over measured {e['lag']}); "
          f"L*/delta p=0.2 {c['dens']:.3f}, p(Ec) {b['dens']:.3f}, kernel {a['dens']:.3f} (measured {e['n']}; largest difference {100 * max(abs(q['dens'] / e['n'] - 1) for q in (a, b, c)):.1f}%)")
    print(f"   upstream-moving T_rot/T2 at beta 1e-2 and 1e-3: kernel {a['beta0.01']['trot']:.2f}, {a['beta0.001']['trot']:.2f}; p(Ec) {b['beta0.01']['trot']:.2f}; p=0.2 {c['beta0.01']['trot']:.2f}; "
          f"decay length kernel {a['decay_length']:.2f}, p(Ec) {b['decay_length']:.2f} l1")
# ---------------------------------------------------------------- first collision
for lab in ("M7k", "M12.9k", "M7lb", "M12.9lb"):
    g = lambda k: float(FR[f"{lab}_{k}"]); f = FR[f"{lab}_first"]; c = f[0][::-1].cumsum()[::-1] / f[2].sum(); EB = np.arange(200) * 0.05
    mm = (EB >= 1 - 1e-9) & (EB <= 4 + 1e-9) & (c > 0); p = np.polyfit(EB[mm], np.log(c[mm]), 1); dev = np.exp(np.abs(np.log(c[mm]) - np.polyval(p, EB[mm])).max()) - 1
    loc = [0.5 / math.log(c[int(round(x / 0.05))] / c[int(round((x + 0.5) / 0.05))]) for x in np.arange(1, 3.51, 0.5)]
    print(f"First {lab}: mean Et {g('et'):.2f}, mean e' {g('mean'):.3f}, P(>3) {g('above3'):.4f}, through 1 and 3: {g('amp'):.2f} exp(-E/{g('slope'):.3f}); "
          f"fit 1-4: {math.exp(p[1]):.2f} exp(-E/{-1 / p[0]:.3f}), largest deviation {100 * dev:.1f}%, local slope {min(loc):.2f}-{max(loc):.2f}")
r3 = [float(FR[f"M{m}lb_above3"]) / float(FR[f"M{m}k_above3"]) for m in ("7", "12.9")]
print(f"   LB/kernel above 3 kT2: {r3[0]:.2f} (Mach 7), {r3[1]:.2f} (Mach 12.9)")
# ---------------------------------------------------------------- precursor
for lab, cl in (("M7b", "M7c"), ("M12.9b", "M12.9c")):
    s, c = float(PRE[lab + "_slope"]), float(PRE[cl + "_slope"])
    b = PRE[f"{lab}_beta"]; cn = PRE[f"{lab}_cn"]; xx = PRE[f"{lab}_x"]; up = xx < 0; comp = []
    for lo, hi in ((10 ** -2.25, 10 ** -1.75), (10 ** -4.25, 10 ** -3.75)):
        w = up & (b >= lo) & (b < hi); cc = cn[:, w].sum(1); comp.append(np.array([cc[1], cc[4], cc[2] + cc[3]]) / cc.sum())
    print(f"Precursor {lab}: decay length {float(PRE[lab + '_length']):.2f} l1, cascade/DSMC rate {c / s:.4f} ({100 * abs(c / s - 1):.1f}%); "
          f"near beta 1e-2: once-collided stream {comp[0][0]:.2f}, scattered by stream {comp[0][1]:.2f}; near 1e-4 these two {comp[1][0] + comp[1][1]:.2f}")
for lab in ("M7k", "M12.9k"):
    print(f"Precursor {lab}: decay length {float(PREK[lab + '_length']):.2f} l1")
