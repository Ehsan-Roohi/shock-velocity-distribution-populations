"""Why the distribution after the first collision is exponential although it is a sum over collision energies.

The share distributions of single collisions in bins of the relative energy Et (classical trajectories, cond_<set>.npz)
are summed with the distribution of Et of the first collisions in the shock (first_<...>.npz) and set against the
distribution after the first collision in DSMC. Prints and tabulates, at E = 1, 2, 3, 4 kT2: the local slope of the
DSMC distribution, of the sum of the trajectory share curves and of the sum of Eq. (1) extended beyond its range; the
mean pair energy and share of the collisions that put the molecule above E. Also the local scale of the share
distribution of each Et bin for shares 0.05-0.4 and 0.4-0.6. Writes sm_tab_mixture.tex and sm_tab_sharescale.tex.

Usage: python make_mixture_table.py <n2 folder> <first npz> <label M7> <label M12.9> <output folder>"""
import sys, os
import numpy as np

N2, FIRST, L7, L12, OUT = sys.argv[1:6]
F = np.load(os.path.join(N2, FIRST)); C = {"7.0": np.load(os.path.join(N2, "ctc_cluster/cond_M7tm.npz")), "12.9": np.load(os.path.join(N2, "ctc_cluster/cond_M12.9tm.npz"))}
BEG = r"\begin{ruledtabular}" + "\n"; END = r"\end{tabular}" + "\n" + r"\end{ruledtabular}" + "\n"; EOL = r" \\" + "\n"; HL = r"\hline" + "\n"
EQ = (1.0, 2.0, 3.0, 4.0); H = 0.25


def scale(S, y, x1, x2):
    i1, i2 = np.argmin(abs(S - x1)), np.argmin(abs(S - x2))
    return (S[i2] - S[i1]) / np.log(y[i1] / y[i2]) if y[i2] > 0 and y[i1] > 0 else np.nan


tm = BEG + r"\begin{tabular}{ccccccc}" + "\n" + r"Mach & $E/kT_2$ & DSMC & trajectories & Eq.~(1) extended & $\langle E_{\rm tot}\rangle/kT_2$ & $E/\langle E_{\rm tot}\rangle$" + EOL + HL
ts = BEG + r"\begin{tabular}{cccc}" + "\n" + r"Mach & $E_t/kT_2$ & scale, shares 0.05--0.4 & scale, shares 0.4--0.6" + EOL + HL
for q, (mach, lab) in enumerate((("7.0", L7), ("12.9", L12))):
    c = C[mach]; S = c["S"]; ed = c["et_edges"]; fS = np.nan_to_num(c["bin_fS"].copy()); fS[0] = fS[1]   # no trajectories below kT2: the 1-2 kT2 curve
    f = F[f"{lab}_first"]; et = (np.arange(200) + 0.5) * 0.1; g = f[2] / f[2].sum(); pm = float(F[f"{lab}_partner_mean"])
    cd = f[0][::-1].cumsum()[::-1] / f[2].sum(); E0 = np.arange(200) * 0.05
    Eg = np.array(sorted({e + d for e in EQ for d in (-H, 0.0, H)})); P = np.zeros_like(Eg); P1 = np.zeros_like(Eg); wE = np.zeros_like(Eg)
    for i, e in enumerate(et):
        if g[i] < 1e-12: continue
        b = min(np.searchsorted(ed, e, side="right") - 1, len(fS) - 1); Etot = e + pm
        y = np.interp(Eg / Etot, S, fS[b], right=0.0); P += g[i] * y; wE += g[i] * y * Etot
        P1 += g[i] * np.minimum(1.0, 0.98 * np.exp(-(Eg / Etot) / 0.15))
    D = np.interp(Eg, E0, cd); at = lambda Q, e: Q[np.argmin(abs(Eg - e))]
    sl = lambda Q, e: 2 * H / np.log(at(Q, e - H) / at(Q, e + H))
    for k, e in enumerate(EQ):
        Et_ = at(wE, e) / at(P, e)
        tm += f"{mach if k == 0 else ''} & {e:.0f} & {sl(D, e):.2f} & {sl(P, e):.2f} & {sl(P1, e):.2f} & {Et_:.1f} & {e / Et_:.2f}" + EOL
        print(f"Mach {mach} E = {e:.0f} kT2: local slope DSMC {sl(D, e):.3f}, trajectory share curves {sl(P, e):.3f}, Eq.(1) extended {sl(P1, e):.3f}; "
              f"mean pair energy of the contributing collisions {Et_:.2f} kT2, share {e / Et_:.2f}")
    tm += HL if q == 0 else ""
    for b in range(1, len(c["bin_et"])):
        if not np.isfinite(c["bin_et"][b]) or c["bin_et"][b] > 12: continue
        a1, a2 = scale(S, fS[b], 0.05, 0.4), scale(S, fS[b], 0.4, 0.6)
        ts += f"{mach if b == 1 else ''} & {ed[b]:g}--{ed[b + 1]:g} & {a1:.3f} & {a2:.3f}" + EOL
        print(f"   Mach {mach} Et {ed[b]:g}-{ed[b + 1]:g}: scale of the share distribution 0.05-0.4 {a1:.3f}, 0.4-0.6 {a2:.3f}")
    ts += HL if q == 0 else ""
open(os.path.join(OUT, "sm_tab_mixture.tex"), "w", encoding="utf-8", newline="\n").write(tm + END)
open(os.path.join(OUT, "sm_tab_sharescale.tex"), "w", encoding="utf-8", newline="\n").write(ts + END)
print("written sm_tab_mixture.tex, sm_tab_sharescale.tex")
