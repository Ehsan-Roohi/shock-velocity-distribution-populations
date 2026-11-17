"""One line per trajectory set: the single-collision excitation function (ctc_excite.py output) reduced to the numbers
quoted in the letter, so that accuracy tests, similarity tests and other molecules can be set side by side.

For each set: share of the VHS cross-section (nitrogen) with e' above 0.1, 0.3 and 1 kT2; amplitude and temperature
of the exponential between 1 and 3 kT2; the same excitation cross-sections in units of the set's own viscosity
cross-section times 3/2 (for molecules other than nitrogen); mean energy gained per VHS collision; rotational energy
of the molecule sent upstream (from the json of ctc_analyse.py when present).

Usage: python ctc_sets.py <folder with exc_<set>.npz and an_<set>.json> <set> [<set> ...]"""
import sys, os, json, math
import numpy as np

d0 = sys.argv[1]
print(f"{'set':10s} {'T2':>6s} | above 0.1  0.3   1 kT2 (of sigma_VHS) | amplitude  T/T2 | above 0.1, 0.3, 1 (of 1.5 sigma_mu) | gain/kT2 | recoil e/kT2 | sigma_mu A^2 (VHS) | trajectories")
for s in sys.argv[2:]:
    d = np.load(os.path.join(d0, f"exc_{s}.npz")); E = d["E"]; sv = float(d["sig_vhs"]); f = d["ctc"] / sv; T2 = float(d["T2"])
    i = lambda x: int(np.argmin(np.abs(E - x))); th = 2.0 / math.log(f[i(1.0)] / f[i(3.0)]); A = f[i(1.0)] * math.exp(1.0 / th)
    jn = os.path.join(d0, f"an_{s}.json"); a = json.load(open(jn)) if os.path.exists(jn) else {}
    smu = a.get("sig_mu_ctc", float("nan")); smv = a.get("sig_mu_vhs", float("nan")); rec = a.get("ctc", {}).get("mean", float("nan")); rse = a.get("ctc", {}).get("se", float("nan"))
    g = 1.5 * smu / sv
    print(f"{s:10s} {T2:6.0f} | {f[i(0.1)]:9.3f} {f[i(0.3)]:5.3f} {f[i(1.0)]:5.3f}              | {A:8.3f} {th:6.3f} | {f[i(0.1)] / g:8.3f} {f[i(0.3)] / g:6.3f} {f[i(1.0)] / g:6.3f}        "
          f"| {float(d['gain_ctc']) / T2:7.3f}  | {rec:6.3f} +- {rse:.3f} | {smu:6.1f} ({smv:.1f})      | {int(d['n'])}")
