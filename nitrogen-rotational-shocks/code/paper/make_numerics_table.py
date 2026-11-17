"""Table of the numerical tests of the kernel simulations (supplemental material): reference runs against cell size
and time step halved, a kernel from half of the trajectories, a kernel on every second tabulated energy, and the
measured free-stream distribution as inflow. Writes sm_tab_numerics.tex.

Inputs in the n2 folder: erh_v17.npz, first_v17.npz, rt_predict_ktab.json, exch_ktab.json (reference, labels M7k and
M12.9k) and erh_tests.npz, first_tests.npz, rt_predict_tests.json, exch_tests.json (labels M7<test>, M12.9<test>).
Usage: python make_numerics_table.py <n2 data folder> <output folder>"""
import sys, os, json, math
import numpy as np
N2, OUT = sys.argv[1], sys.argv[2]; sys.path.insert(0, N2)
import robben_talbot_1966 as rt

R = dict(erh=np.load(os.path.join(N2, "erh_v17.npz")), first=np.load(os.path.join(N2, "first_v17.npz")), rt=json.load(open(os.path.join(N2, "rt_predict_ktab.json"))),
         ex=json.load(open(os.path.join(N2, "exch_ktab.json"))))
T = dict(erh=np.load(os.path.join(N2, "erh_tests.npz")), first=np.load(os.path.join(N2, "first_tests.npz")), rt=json.load(open(os.path.join(N2, "rt_predict_tests.json"))),
         ex=json.load(open(os.path.join(N2, "exch_tests.json"))))
ROWS = [("reference (four runs)", R, "k"), ("cell size and time step halved", T, "grid"), ("kernel from half of the trajectories", T, "half"),
        ("kernel on every second energy", T, "egrid")]
INFLOW = ("measured free stream as inflow", T, "rin")             # reported in the text: populations without replacement


def hot_rms(d, lab, key):
    """root-mean-square difference between the share with 3+ collisions and the measured hot group over all stations (as Fig. 2)"""
    n, erh = d[f"{lab}_n"], d[f"{lab}_erh"]; ns = (n - 1) / (float(d[f"{lab}_n2"]) - 1); t = erh.sum(-1); tot = np.maximum(t[0], 1e-30)
    m = (ns > 0.005) & (ns < 0.99) & (np.arange(len(ns)) < np.argmax(ns > 0.99) + 1 if (ns > 0.99).any() else True); k = np.argsort(ns[m]); X = ns[m][k]; s3 = (t[11] / tot)[m][k]
    e = rt.reduce(rt.CASES[key]["rows"]); en = (e[:, 2] - e[0, 2]) / (e[-3:, 2].mean() - e[0, 2]); s = (en > 0.02) & (en < 0.97)
    return float(np.sqrt(np.mean((np.interp(en[s], X, s3) - e[s, 4]) ** 2)))


def levels(recs, nmin, key="pred"):
    q = np.concatenate([np.array(r[key][:r["kp"]]) / np.array(r["meas"][:r["kp"]]) for r in recs if r["kp"] > 0 and r["nstar"] > nmin])
    return 100 * float(np.sqrt(np.mean((q - 1) ** 2))), float(q.min()), float(q.max())


def change(recs, ref, nmin, key="pred"):
    """root-mean-square change (per cent) of the predicted populations against the reference runs, tabulated levels"""
    a = [r for r in ref if r["kp"] > 0 and r["nstar"] > nmin]; b = [r for r in recs if r["kp"] > 0 and r["nstar"] > nmin]
    q = np.concatenate([np.array(y[key][:y["kp"]]) / np.array(x["pred"][:x["kp"]]) for x, y in zip(a, b)])
    return 100 * float(np.sqrt(np.mean((q - 1) ** 2)))


BEG = r"\begin{ruledtabular}" + "\n"; END = r"\end{tabular}" + "\n" + r"\end{ruledtabular}" + "\n"; EOL = r" \\" + "\n"; HL = r"\hline" + "\n"
t = BEG + r"\begin{tabular}{llccccccc}" + "\n" + r"Mach & runs & rms (\%) & range & change (\%) & hot group & $T_s/T_2$ & $L^*/\delta$ & $\Delta/L^*$" + EOL + HL
for mach, key, nmin in (("7.0", "7", 0.02), ("12.9", "12.9", 0.13)):
    pre = "M7" if key == "7" else "M12.9"
    for q, (name, D, tag) in enumerate(ROWS):
        lab = pre + tag
        if f"{lab}_n" not in D["erh"].files:
            print("missing", lab); continue
        rms, lo, hi = levels(D["rt"][lab], nmin); h = hot_rms(D["erh"], lab, key); fs = float(D["first"][f"{lab}_slope"]); x = D["ex"][lab]
        ch = "--" if q == 0 else f"{change(D['rt'][lab], R['rt'][pre + 'k'], nmin):.1f}"
        t += f"{mach if q == 0 else ''} & {name} & {rms:.1f} & {lo:.2f}--{hi:.2f} & {ch} & {h:.3f} & {fs:.3f} & {x['dens']:.3f} & {x['lag']:.2f}" + EOL
        print(f"Mach {mach} {name:40s}: levels rms {rms:.1f}% ({lo:.2f}-{hi:.2f}), change {ch}%, 3+ vs hot {h:.3f}, first-collision T {fs:.3f}, L*/delta {x['dens']:.3f}, lag {x['lag']:.2f}")
    name, D, tag = INFLOW; lab = pre + tag
    rms, lo, hi = levels(D["rt"][lab], nmin, "pred_raw"); ch = change(D["rt"][lab], R["rt"][pre + "k"], nmin, "pred_raw"); h = hot_rms(D["erh"], lab, key); x = D["ex"][lab]
    print(f"Mach {mach} {name:40s}: levels rms {rms:.1f}% ({lo:.2f}-{hi:.2f}) with no replacement, change {ch:.1f}%, 3+ vs hot {h:.3f}, first-collision T {float(D['first'][lab + '_slope']):.3f}, "
          f"mean {float(D['first'][lab + '_mean']):.3f}, L*/delta {x['dens']:.3f}, lag {x['lag']:.2f}  [not in the table]")
    t += HL if key == "7" else ""
open(os.path.join(OUT, "sm_tab_numerics.tex"), "w", encoding="utf-8", newline="\n").write(t + END); print("written sm_tab_numerics.tex")
