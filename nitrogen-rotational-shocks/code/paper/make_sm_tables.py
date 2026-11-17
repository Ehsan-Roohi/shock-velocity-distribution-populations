"""Tables of the supplemental material, written as LaTeX files sm_tab_<name>.tex from the result files, so that no
number is typed by hand.   Usage: python make_sm_tables.py <n2 data folder> <output folder> [rin]
   rin: runs whose stream enters with the measured free-stream rotational distribution (the main comparison);
   the tests of the deflection threshold stay with the runs of version 17 (Boltzmann stream)."""
import sys, os, json, math
import numpy as np
from scipy.optimize import nnls

N2, OUT = sys.argv[1], sys.argv[2]; sys.path.insert(0, N2); RIN = "rin" in sys.argv[3:]
import robben_talbot_1966 as rt
CT = os.path.join(N2, "ctc_cluster")


def write(name, text):
    open(os.path.join(OUT, f"sm_tab_{name}.tex"), "w", encoding="utf-8", newline="\n").write(text); print("written sm_tab_" + name)


# ---------------------------------------------------------------- trajectories: one row per set
def row(s):
    d = np.load(os.path.join(CT, f"exc_{s}.npz")); E = d["E"]; sv = float(d["sig_vhs"]); f = d["ctc"] / sv; T2 = float(d["T2"])
    i = lambda x: int(np.argmin(np.abs(E - x))); th = 2.0 / math.log(f[i(1.0)] / f[i(3.0)]); A = f[i(1.0)] * math.exp(1.0 / th)
    a = json.load(open(os.path.join(CT, f"an_{s}.json"))); c = np.load(os.path.join(CT, f"cond_{s}.npz"))      # gain of rotational energy (ctc_condition.py)
    return f"{c['gain'][i(0.1)]:.2f} & {f[i(0.3)]:.3f} & {f[i(1.0)]:.3f} & {A:.2f} & {th:.3f} & {float(d['gain_ctc']) / T2:.2f} & {a['ctc']['mean']:.2f} & {a['sig_mu_ctc']:.1f}"


head = r"$\sigma^{\rm g}_{0.1}$ & $\sigma_{0.3}$ & $\sigma_{1}$ & $A$ & $T_s/T_2$ & $\langle\Delta e\rangle/kT_2$ & $e_{\rm up}/kT_2$ & $\sigma_\mu$ (\AA$^2$)"
sets = [("Mach 12.9 conditions, reference", "M12.9tm"), ("Mach 7.0 conditions, reference", "M7tm"), ("parameter set B, Mach 12.9", "M12.9"), ("parameter set B, Mach 7.0", "M7"),
        ("parameter set C, Mach 12.9", "M12.9b"), (r"largest impact parameter 7.0~\AA", "accB"), ("6000 integration steps", "accN"), (r"path half-length 20~\AA", "accD"),
        (r"all temperatures $\times2$ ($T_2=610$~K)", "simT2"), (r"all temperatures $\times4$ ($T_2=1220$~K)", "simT4")]
write("traj", "\\begin{ruledtabular}\n\\begin{tabular}{lcccccccc}\n & " + head + " \\\\\n\\hline\n" + "".join(f"{n} & {row(s)} \\\\\n" for n, s in sets) + "\\end{tabular}\n\\end{ruledtabular}\n")
elo = [("0.20", "0.634", "elo20"), ("0.345", "1.094", "M12.9tm"), ("0.40", "1.268", "elo40"), ("0.50", "1.585", "elo50"), ("0.65", "2.061", "elo65")]
write("elo", "\\begin{ruledtabular}\n\\begin{tabular}{cccccccccc}\nelongation & bond (\\AA) & " + head + " \\\\\n\\hline\n" + "".join(f"{e} & {b} & {row(s)} \\\\\n" for e, b, s in elo)
      + "\\end{tabular}\n\\end{ruledtabular}\n")

# ---------------------------------------------------------------- kernel tests
k = np.load(os.path.join(N2, "kcell", "kernel_tests.npz")); kg = np.load(os.path.join(N2, "kcell", "kernel_gain.npz"))
t = "\\begin{ruledtabular}\n\\begin{tabular}{lcccccc}\n & $\\sigma^{\\rm g}_{0.1}$ & $\\sigma^{\\rm g}_{0.3}$ & $\\sigma^{\\rm g}_{1}$ & $\\sigma^{\\rm g}_{2}$ & $\\sigma^{\\rm g}_{3}$ & $T_s/T_2$ \\\\\n\\hline\n"
kdiff = []
for nm, lab in (("M12p9", "Mach 12.9"), ("M7", "Mach 7.0")):
    E = kg[nm + "_E"]; i = lambda x: int(np.argmin(np.abs(E - x)))
    for key, who in (("gain_direct", "trajectories"), ("gain", "kernel")):
        f = kg[f"{nm}_{key}"]; t += f"{lab}, {who} & " + " & ".join(f"{f[i(x)]:.3f}" for x in (0.1, 0.3, 1, 2, 3)) + f" & {2 / math.log(f[i(1)] / f[i(3)]):.3f} \\\\\n"
    m = (E >= 0.1) & (E <= 3.0); kdiff.append(float(np.max(np.abs(kg[nm + "_gain"][m] / kg[nm + "_gain_direct"][m] - 1))))
write("kernel", t + "\\end{tabular}\n\\end{ruledtabular}\n"); print("   kernel against direct trajectories, largest difference of the gain cross-section between 0.1 and 3 kT2:", [round(v, 3) for v in kdiff])

# ---------------------------------------------------------------- one collision at fixed collision energy
t = "\\begin{ruledtabular}\n\\begin{tabular}{cccccccccc}\nMach & $E_t/kT_2$ & collisions & $\\langle E_t\\rangle/kT_2$ & $T_s/T_2$ & $kT_s/\\langle E_t\\rangle$ & $\\langle\\Delta e\\rangle/\\langle E_t\\rangle$ & $x_s$ & rms & $\\langle x\\rangle$ \\\\\n\\hline\n"
cnum = {}; xfit = []
for mach, s in (("12.9", "M12.9tm"), ("7.0", "M7tm")):
    c = np.load(os.path.join(CT, f"cond_{s}.npz")); ed = c["et_edges"]; first = True; E = c["E"]; i = lambda x: int(np.argmin(np.abs(E - x))); SS = c["S"]
    for j in range(len(c["bin_et"])):
        if not np.isfinite(c["bin_et"][j]) or ed[j] < 2 or ed[j] >= 14: continue
        f = c["bin_fS"][j]; mm = (SS >= 0.05) & (SS <= 0.4); pf = np.polyfit(SS[mm], np.log(f[mm]), 1); rr = np.log(f[mm]) - np.polyval(pf, SS[mm]); xm = float(np.trapezoid(f, SS))
        xfit.append((-1 / pf[0], math.exp(pf[1]), xm, np.sqrt(np.mean(rr ** 2)), np.abs(rr).max()))
        t += (f"{mach if first else ''} & {ed[j]:.0f}--{ed[j + 1]:.0f} & {c['bin_share'][j]:.3f} & {c['bin_et'][j]:.2f} & {c['bin_slope'][j]:.3f} & {c['bin_slope'][j] / c['bin_et'][j]:.3f} & "
              f"{c['bin_gain'][j] / c['bin_et'][j]:.3f} & {-1 / pf[0]:.3f} & {np.sqrt(np.mean(rr ** 2)):.3f} & {xm:.3f} \\\\\n"); first = False
    t += "\\hline\n" if mach == "12.9" else ""
    sl = lambda f: 2 / math.log(f[i(1)] / f[i(3)]); cnum[s] = dict(cold=sl(c["partner_cold"]), hot=sl(c["partner_hot"]), et=float(np.sum(c["et_hist_x"] * c["et_hist"]) * 0.25),
                                                                  g01=float(c["gain"][i(0.1)]), g03=float(c["gain"][i(0.3)]), g1=float(c["gain"][i(1.0)]), a01=float(c["after"][i(0.1)]), b01=float(c["before"][i(0.1)]), c01=float(c["cross"][i(0.1)]))
write("energy", t + "\\end{tabular}\n\\end{ruledtabular}\n"); print("   single collision:", {s: {a: round(b, 3) for a, b in v.items()} for s, v in cnum.items()})
xfit = np.array(xfit); print(f"   share fits: scale {xfit[:, 0].min():.3f}-{xfit[:, 0].max():.3f}, amplitude {xfit[:, 1].min():.2f}-{xfit[:, 1].max():.2f}, mean share {xfit[:, 2].min():.3f}-{xfit[:, 2].max():.3f}, "
                             f"rms of ln {xfit[:, 3].min():.3f}-{xfit[:, 3].max():.3f}, max {xfit[:, 4].max():.3f}")

# ---------------------------------------------------------------- scatter between the four independent runs (trajectory kernel)
sd = json.load(open(os.path.join(N2, "rt_predict_rin_seeds.json" if RIN else "rt_predict_ktab_seeds.json"))); seedn = {}
for key, nmin in (("M7", 0.02), ("M12.9", 0.13)):
    rr = []; PP = []
    for s in (11, 12, 13, 14):
        rec = [r for r in sd[f"{key}s{s}"] if r["kp"] > 0 and r["nstar"] > nmin]; PP.append(np.concatenate([np.array(r["pred"][:r["kp"]]) for r in rec]))
        q = np.concatenate([np.array(r["pred"][:r["kp"]]) / np.array(r["meas"][:r["kp"]]) for r in rec]); rr.append(np.sqrt(np.mean((q - 1) ** 2)))
    PP = np.array(PP); rel = PP.std(0, ddof=1) / PP.mean(0); seedn[key] = (100 * np.mean(rr), 100 * np.std(rr, ddof=1) / 2, 100 * rel.max() / 2)
    print(f"   {key}: rms of predicted/measured over the four runs {seedn[key][0]:.2f} +- {seedn[key][1]:.2f} %; largest standard error of a predicted population {seedn[key][2]:.2f} %")
open(os.path.join(OUT, "sm_numbers.tex"), "w", encoding="utf-8", newline="\n").write(
    f"\\newcommand{{\\SMeqratio}}{{{float(k['eq_ratio']):.3f}}}\n\\newcommand{{\\SMeqerr}}{{{float(k['eq_err']):.3f}}}\n\\newcommand{{\\SMeqabove}}{{{float(k['eq_above2']):.4f}}}\n"
    f"\\newcommand{{\\SMeqbelow}}{{{float(k['eq_below01']):.4f}}}\n\\newcommand{{\\SMrelp}}{{{float(k['rel_rate']) / 1.063:.2f}}}\n\\newcommand{{\\SMrelpearly}}{{{float(k['rel_rate_early']) / 1.063:.2f}}}\n"
    f"\\newcommand{{\\SMkdiff}}{{{100 * max(kdiff):.1f}}}\n\\newcommand{{\\SMcoldA}}{{{cnum['M12.9tm']['cold']:.2f}}}\n\\newcommand{{\\SMhotA}}{{{cnum['M12.9tm']['hot']:.2f}}}\n"
    f"\\newcommand{{\\SMcoldB}}{{{cnum['M7tm']['cold']:.2f}}}\n\\newcommand{{\\SMhotB}}{{{cnum['M7tm']['hot']:.2f}}}\n\\newcommand{{\\SMetA}}{{{cnum['M12.9tm']['et']:.1f}}}\n\\newcommand{{\\SMetB}}{{{cnum['M7tm']['et']:.1f}}}\n"
    f"\\newcommand{{\\SMafterA}}{{{cnum['M12.9tm']['a01']:.2f}}}\n\\newcommand{{\\SMafterB}}{{{cnum['M7tm']['a01']:.2f}}}\n\\newcommand{{\\SMstayA}}{{{cnum['M12.9tm']['a01'] - cnum['M12.9tm']['c01']:.2f}}}\n"
    f"\\newcommand{{\\SMstayB}}{{{cnum['M7tm']['a01'] - cnum['M7tm']['c01']:.2f}}}\n\\newcommand{{\\SMgainA}}{{{cnum['M12.9tm']['g01']:.2f}}}\n\\newcommand{{\\SMgainB}}{{{cnum['M7tm']['g01']:.2f}}}\n"
    f"\\newcommand{{\\SMseedA}}{{{seedn['M7'][0]:.2f}}}\n\\newcommand{{\\SMseedAerr}}{{{seedn['M7'][1]:.2f}}}\n\\newcommand{{\\SMseedAmax}}{{{seedn['M7'][2]:.1f}}}\n"
    f"\\newcommand{{\\SMseedB}}{{{seedn['M12.9'][0]:.2f}}}\n\\newcommand{{\\SMseedBerr}}{{{seedn['M12.9'][1]:.2f}}}\n\\newcommand{{\\SMseedBmax}}{{{seedn['M12.9'][2]:.1f}}}\n")

# ---------------------------------------------------------------- level populations with no fitted number
if RIN:
    _r = json.load(open(os.path.join(N2, "rt_predict_rin.json"))); P = {m: _r for m in ("const", "pec", "soft", "kern")}
else:
    P = {"const": json.load(open(os.path.join(N2, "rt_predict_lb12.json"))), "pec": json.load(open(os.path.join(N2, "rt_predict_lb.json"))),
         "soft": json.load(open(os.path.join(N2, "rt_predict_pullin5.json"))), "kern": json.load(open(os.path.join(N2, "rt_predict_ktab.json")))}
LAB = {"const": ("M7c", "M12.9c"), "pec": ("M7zi10", "M12.9zi10"), "soft": ("M7pu5", "M12.9pu5"), "kern": ("M7k", "M12.9k")}
t = "\\begin{ruledtabular}\n\\begin{tabular}{cccccccc}\nMach & $n^*$ & levels & LB, $p=0.2$ & LB, $p(E_c)$ & soft exchange & kernel & kernel, range \\\\\n\\hline\n"
for im, (mach, nmin) in enumerate((("7.0", 0.02), ("12.9", 0.13))):
    recs = {m: [r for r in P[m][LAB[m][im]] if r["kp"] > 0 and r["nstar"] > nmin] for m in P}; allr = []
    for j, r in enumerate(recs["kern"]):
        q = np.array(r["pred"][:r["kp"]]) / np.array(r["meas"][:r["kp"]]); allr += q.tolist()
        t += f"{mach if j == 0 else ''} & {r['nstar']:.2f} & {r['kp']} & " + " & ".join(f"{100 * recs[m][j]['res_stream']:.1f}" for m in ("const", "pec", "soft", "kern")) + f" & {q.min():.2f}--{q.max():.2f} \\\\\n"
    allr = np.array(allr); rms = {}
    for m in P:
        qq = np.concatenate([np.array(r["pred"][:r["kp"]]) / np.array(r["meas"][:r["kp"]]) for r in recs[m]]); rms[m] = (np.sqrt(np.mean((qq - 1) ** 2)), qq.min(), qq.max())
    t += " & mean & & " + " & ".join(f"{100 * np.mean([r['res_stream'] for r in recs[m]]):.1f}" for m in ("const", "pec", "soft", "kern")) + f" & {allr.min():.2f}--{allr.max():.2f} \\\\\n"
    t += " & rms of ratio & & " + " & ".join(f"{100 * rms[m][0]:.1f}" for m in ("const", "pec", "soft", "kern")) + " & \\\\\n" + ("\\hline\n" if im == 0 else "")
    print(f"   Mach {mach}: ratio ranges " + "; ".join(f"{m} {rms[m][1]:.2f}-{rms[m][2]:.2f}" for m in P))
write("levels", t + "\\end{tabular}\n\\end{ruledtabular}\n")

# ---------------------------------------------------------------- three fixed shapes
KMAX = 24; TH = rt.THETA; K = np.arange(KMAX + 1); G = 2 * K + 1; EK = TH * K * (K + 1)


def levels(r_):
    X, Tp, dk, last = r_; kp = len(dk); A = last * np.exp(TH * (kp - 0.5) * (kp + 0.5) / Tp)
    d = np.array([dk[j] if j < kp else A * np.exp(-EK[j] / Tp) for j in K]); N = G * d
    return N / (N.sum() + A * Tp / TH * np.exp(-TH * (KMAX + 0.5) * (KMAX + 1.5) / Tp)), kp


t = "\\begin{ruledtabular}\n\\begin{tabular}{cccccccc}\nMach & $n^*$ & free stream & one collision & thermal & sum & residual (\\%) & two shapes (\\%) \\\\\n\\hline\n"
for key, exc in (("7", "exc_M7tm.npz"), ("12.9", "exc_M12.9tm.npz")):
    rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); T2 = ex[-1, 1]; en = (ex[:, 2] - rho1) / (rho2 - rho1)
    d = np.load(os.path.join(CT, exc)); E = d["E"] * T2; sv = float(d["sig_vhs"]); sm = float(d["sigmax"]); edge = TH * np.arange(KMAX + 2) ** 2
    f1 = (d["ctc"] - d["before"] * (1 - sv / sm)) / sv; c = np.interp(edge, E, f1); S1 = np.maximum(c[:-1] - c[1:], 0) / c[0]
    S0, _ = levels(rows[0]); S2 = G * np.exp(-EK / T2); S2 = S2 / (S2.sum() + T2 / TH * np.exp(-TH * (KMAX + 0.5) * (KMAX + 1.5) / T2)); B = np.stack([S0, S1, S2], 1); first = True
    for j, r_ in enumerate(rows):
        p, kp = levels(r_)
        if not (0.02 < en[j] < 0.97) or kp == 0: continue
        w, _ = nnls(B, p); r3 = np.linalg.norm(p - B @ w) / np.linalg.norm(p); w2, _ = nnls(B[:, [0, 2]], p); r2 = np.linalg.norm(p - B[:, [0, 2]] @ w2) / np.linalg.norm(p)
        t += f"{rt.CASES[key]['mach'] if first else ''} & {en[j]:.2f} & {w[0]:.2f} & {w[1]:.2f} & {w[2]:.2f} & {w.sum():.3f} & {100 * r3:.1f} & {100 * r2:.1f} \\\\\n"; first = False
    t += "\\hline\n" if key == "7" else ""
write("shapes", t + "\\end{tabular}\n\\end{ruledtabular}\n")

# ---------------------------------------------------------------- profiles and precursor
if RIN:
    _x = json.load(open(os.path.join(N2, "exch_rin.json"))); X = {m: _x for m in ("const", "pec", "soft", "kern")}
else:
    X = {"const": json.load(open(os.path.join(N2, "exch_lb12.json"))), "pec": json.load(open(os.path.join(N2, "exch_ref.json"))),
         "soft": json.load(open(os.path.join(N2, "exch_pullin5.json"))), "kern": json.load(open(os.path.join(N2, "exch_ktab.json")))}
NAME = {"const": "LB rule, $p=0.2$", "pec": "LB rule, $p(E_c)$", "soft": "soft exchange", "kern": "trajectory kernel"}
t = "\\begin{ruledtabular}\n\\begin{tabular}{lcccccccc}\n & \\multicolumn{4}{c}{Mach 7.0} & \\multicolumn{4}{c}{Mach 12.9} \\\\\n & $L^*/\\delta$ & $\\Delta/L^*$ & $e^-_{\\rm rot}$ & $e^-_{\\rm tr}$ & $L^*/\\delta$ & $\\Delta/L^*$ & $e^-_{\\rm rot}$ & $e^-_{\\rm tr}$ \\\\\n\\hline\n"
t += "measured \\cite{RobbenTalbot1966a,RobbenTalbot1966b} & 0.158 & 1.9 & & & 0.167 & 2.54 & & \\\\\n"
for m in ("const", "pec", "soft", "kern"):
    a, b = X[m][LAB[m][0]], X[m][LAB[m][1]]
    t += f"{NAME[m]} & " + " & ".join(f"{q['dens']:.3f} & {q['lag']:.2f} & {q['beta0.01']['trot']:.2f} & {q['beta0.01']['ttr']:.2f}" for q in (a, b)) + " \\\\\n"
write("profiles", t + "\\end{tabular}\n\\end{ruledtabular}\n")
pr = np.load(os.path.join(N2, "letter_precursor_rin.npz" if RIN else "letter_precursor.npz")); pk = np.load(os.path.join(N2, "letter_precursor_k_rin.npz" if RIN else "letter_precursor_k.npz"))
t = "\\begin{ruledtabular}\n\\begin{tabular}{llcccc}\nMach & collisions & decay rate & cascade & ratio & decay length ($\\ell_1$) \\\\\n\\hline\n"
for mach, lab, cas, nm in (("7.0", "M7l", "M7c", "VHS, $p=0.2$"), ("8.0", "M8l", "M8c", "VHS, $p=0.2$"), ("12.0", "M12l", "M12c", "VHS, $p=0.2$"), ("12.9", "M12.9l", "M12.9c", "VHS, $p=0.2$"),
                           ("7.0", "M7b", "M7c", "VHS, $p(E_c)$"), ("12.9", "M12.9b", "M12.9c", "VHS, $p(E_c)$")):
    s, se, c = float(pr[lab + "_slope"]), float(pr[lab + "_slope_se"]), float(pr[cas + "_slope"])
    t += f"{mach} & {nm} & ${-s:.4f}\\pm{se:.4f}$ & {-c:.4f} & {c / s:.3f} & {float(pr[lab + '_length']):.2f} \\\\\n"
for mach, lab in (("7.0", "M7k"), ("12.9", "M12.9k")):
    t += f"{mach} & trajectory kernel & ${-float(pk[lab + '_slope']):.4f}\\pm{float(pk[lab + '_slope_se']):.4f}$ & & & {float(pk[lab + '_length']):.2f} \\\\\n"
write("precursor", t + "\\end{tabular}\n\\end{ruledtabular}\n")

# ---------------------------------------------------------------- counting collisions (dsmc_shock_vdf17)
V17 = np.load(os.path.join(N2, "erh_v17.npz")); FR17 = np.load(os.path.join(N2, "first_v17.npz"))
V = np.load(os.path.join(N2, "erh_rin.npz")) if RIN else V17; FR = np.load(os.path.join(N2, "first_rin.npz")) if RIN else FR17
THR = RIN and "M7chi0.8_n" in V.files              # threshold runs with the measured, frozen stream (version 19)
TV = V if THR else V17


def station_counts(lab, key, nmin, V=V):
    """per measured station: n*, shares of 0, 1, 2, 3+ collisions, slope temperatures (1-3 kT2) of 1, 2, 3+,
    share of the measured hot line (energies above level K') from 3+, measured hot fraction"""
    n, erh = V[f"{lab}_n"], V[f"{lab}_erh"]; ns = (n - 1) / (float(V[f"{lab}_n2"]) - 1)
    rows = rt.CASES[key]["rows"]; ex = rt.reduce(rows); en = (ex[:, 2] - ex[0, 2]) / (ex[-3:, 2].mean() - ex[0, 2]); T2 = ex[-1, 1]; out = []
    for j, r_ in enumerate(rows):
        kp = len(r_[2])
        if not (nmin < en[j] < 0.97) or kp == 0: continue
        i = int(np.argmin(np.abs(ns - en[j]))); s = slice(max(i - 2, 0), i + 3); tot = erh[0, s].sum(); sh = [erh[r, s].sum() / tot for r in (2, 9, 10, 11)]
        sl = []
        for r in (9, 10, 11):
            h = erh[r, s].sum(0); c = h[::-1].cumsum()[::-1]; sl.append(2 / math.log(c[20] / c[60]))
        ie = int(rt.THETA * kp * kp / T2 / 0.05); hot = [erh[r, s].sum(0)[ie:].sum() for r in (2, 9, 10, 11)]
        out.append((en[j], sh, sl, hot[3] / sum(hot), ex[j, 4]))
    return out


BEG = r"\begin{ruledtabular}" + "\n"; END = r"\end{tabular}" + "\n" + r"\end{ruledtabular}" + "\n"; EOL = r" \\" + "\n"; HL = r"\hline" + "\n"
t = BEG + r"\begin{tabular}{ccccccccccc}" + "\n" + r"Mach & $n^*$ & \multicolumn{4}{c}{share with 0, 1, 2, 3+ collisions} & \multicolumn{3}{c}{$T_s/T_2$ for 1, 2, 3+} & 3+ in $T'$ line & measured hot" + EOL + HL
for mach, lab, key, nmin in (("7.0", "M7k", "7", 0.02), ("12.9", "M12.9k", "12.9", 0.13)):
    for q, (e_, sh, sl, h3, mh) in enumerate(station_counts(lab, key, nmin)):
        t += f"{mach if q == 0 else ''} & {e_:.2f} & " + " & ".join(f"{v:.3f}" for v in sh) + " & " + " & ".join(f"{v:.2f}" for v in sl) + f" & {h3:.2f} & {mh:.3f}" + EOL
    t += HL if mach == "7.0" else ""
write("counts", t + END)

DEG = r"$^\circ$"
t = BEG + r"\begin{tabular}{llcccccc}" + "\n" + r"Mach & collisions & $\langle E_t\rangle/kT_2$ & $\langle e'\rangle/kT_2$ & $P(e'>0.3\,kT_2)$ & $P(e'>kT_2)$ & $P(e'>3\,kT_2)$ & tail" + EOL + HL
ROWS1 = [("7.0", "M7k", "kernel", FR), ("7.0", "M7lb", r"LB rule, $p(E_c)$", FR)]
ROWS2 = [("12.9", "M12.9k", "kernel", FR), ("12.9", "M12.9lb", r"LB rule, $p(E_c)$", FR)]
if THR:                                                # tests of the threshold with the measured stream
    ROWS1 += [("7.0", "M7chi0.8", "kernel, 37" + DEG, FR), ("7.0", "M7chi0.97", "kernel, 14" + DEG, FR)]
    ROWS2 += [("12.9", "M12.9chi0.8", "kernel, 37" + DEG, FR), ("12.9", "M12.9chi0.97", "kernel, 14" + DEG, FR)]
elif RIN:                                              # tests of the threshold: Boltzmann stream (version 17)
    ROWS1 += [("7.0", "M7k", "kernel, 26" + DEG + ", B", FR17), ("7.0", "M7chi0.8", "kernel, 37" + DEG + ", B", FR17), ("7.0", "M7chi0.97", "kernel, 14" + DEG + ", B", FR17)]
    ROWS2 += [("12.9", "M12.9k", "kernel, 26" + DEG + ", B", FR17), ("12.9", "M12.9chi0.8", "kernel, 37" + DEG + ", B", FR17), ("12.9", "M12.9chi0.97", "kernel, 14" + DEG + ", B", FR17)]
else:
    ROWS1 += [("7.0", "M7chi0.8", "kernel, 37" + DEG, FR), ("7.0", "M7chi0.97", "kernel, 14" + DEG, FR)]
    ROWS2 += [("12.9", "M12.9chi0.8", "kernel, 37" + DEG, FR), ("12.9", "M12.9chi0.97", "kernel, 14" + DEG, FR)]
for mach, lab, nm, SRC in ROWS1 + ROWS2:
    g = lambda k: float(SRC[f"{lab}_{k}"])
    t += f"{mach} & {nm} & {g('et'):.2f} & {g('mean'):.3f} & {g('above03'):.3f} & {g('above1'):.4f} & {g('above3'):.4f} & " + rf"${g('amp'):.2f}\,e^{{-E/{g('slope'):.2f}\,kT_2}}$" + EOL
write("first", t + END)

t = BEG + r"\begin{tabular}{llccccc}" + "\n" + r"Mach & threshold & 3+ share at the stations & $T_s/T_2$, 1 collision & 2 collisions & 3+ & rms from measured hot" + EOL + HL
for mach, key, nmin, labs in (("7.0", "7", 0.02, (("M7chi0.97", "14" + DEG), ("M7k", "26" + DEG), ("M7chi0.8", "37" + DEG))),
                              ("12.9", "12.9", 0.13, (("M12.9chi0.97", "14" + DEG), ("M12.9k", "26" + DEG), ("M12.9chi0.8", "37" + DEG)))):
    for q, (lab, nm) in enumerate(labs):
        o = station_counts(lab, key, nmin, TV); sl = np.array([x[2] for x in o]); s3 = np.array([x[1][3] for x in o]); mh = np.array([x[4] for x in o])
        t += f"{mach if q == 0 else ''} & {nm} & " + " ".join(f"{v:.2f}" for v in s3) + " & " + " & ".join(f"{sl[:, c].min():.2f}--{sl[:, c].max():.2f}" for c in range(3)) + f" & {np.sqrt(np.mean((s3 - mh) ** 2)):.3f}" + EOL
    t += HL if mach == "7.0" else ""
write("threshold", t + END)

# ---------------------------------------------------------------- tests of the exponential tails (least-squares fits)
EB = np.arange(200) * 0.05


def tailfit(c, lo, hi):
    """fit of ln P(e > E) between lo and hi (kT2): temperature, rms and largest deviation of ln P, range of the local slope over 0.5 kT2"""
    m = (EB >= lo - 1e-9) & (EB <= hi + 1e-9) & (c > 0); p = np.polyfit(EB[m], np.log(c[m]), 1); r = np.log(c[m]) - np.polyval(p, EB[m])
    loc = [0.5 / math.log(c[int(round(a / 0.05))] / c[int(round((a + 0.5) / 0.05))]) for a in np.arange(lo, hi - 0.49, 0.5)]
    return -1 / p[0], np.sqrt(np.mean(r ** 2)), np.abs(r).max(), min(loc), max(loc)


t = BEG + r"\begin{tabular}{llcccccc}" + "\n" + r"Mach & distribution & range ($kT_2$) & $T_s/T_2$ & rms & largest & local slope ($T_2$)" + EOL + HL
for mach, lab, lb, key in (("7.0", "M7k", "M7lb", "7"), ("12.9", "M12.9k", "M12.9lb", "12.9")):
    for l_, nm in ((lab, "first collision, kernel"), (lb, r"first collision, LB $p(E_c)$")):
        f = FR[f"{l_}_first"]; c = f[0][::-1].cumsum()[::-1] / f[2].sum()
        q = tailfit(c, 1.0, 4.0); t += f"{mach} & {nm} & 1--4 & {q[0]:.3f} & {q[1]:.3f} & {q[2]:.3f} & {q[3]:.2f}--{q[4]:.2f}" + EOL
    n, erh = V[f"{lab}_n"], V[f"{lab}_erh"]; ns = (n - 1) / (float(V[f"{lab}_n2"]) - 1)
    for x in (0.07, 0.24, 0.58):
        i = int(np.argmin(np.abs(ns - x))); s = slice(max(i - 2, 0), i + 3)
        for nm, r, hi in (("1 collision", 9, 3.5), ("2 collisions", 10, 3.5), ("3+ collisions", 11, 4.0)):
            h = erh[r, s].sum(0); c = np.zeros(200); c[:160] = h[::-1].cumsum()[::-1] / h.sum()
            q = tailfit(c, 1.0, hi); t += f"{mach} & {nm}, $n^*={x:.2f}$ & 1--{hi:g} & {q[0]:.3f} & {q[1]:.3f} & {q[2]:.3f} & {q[3]:.2f}--{q[4]:.2f}" + EOL
    t += HL if mach == "7.0" else ""
write("expfit", t + END)
