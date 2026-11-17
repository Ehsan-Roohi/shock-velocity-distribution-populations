"""Record of the first collision of stream molecules (dsmc_shock_vdf17, <out>.first.bin = double [4][200]):
rotational energy of the stream molecule after the collision (bins of 0.05 T2 and of 0.005 T2), relative
translational energy before the collision (bins of 0.1 T2), rotational energy of the partner before (bins of 0.05 T2).
Sums the runs of each label and prints the numbers that are set against the classical trajectories.

Usage (in the DSMC folder): python n2_first_reduce.py <out.npz> <label>=<glob of .txt files> [...]"""
import sys, glob, math
import numpy as np

out = {}
for spec in sys.argv[2:]:
    lab, pat = spec.split("=", 1); stems = [f[:-4] for f in sorted(glob.glob(pat))]
    if not stems: print(lab, "no files"); continue
    per = np.array([np.fromfile(s + ".first.bin").reshape(4, 200) for s in stems]); F = per.sum(0); n = F[2].sum()

    def stats(f):
        ec = (np.arange(200) + 0.5) * 0.05; c = f[0][::-1].cumsum()[::-1] / f[2].sum()      # share with e' above the lower edge of each coarse bin
        at = lambda x: float(c[int(round(x / 0.05))]); sl = 2 / math.log(at(1) / at(3))
        fine = f[1].cumsum() / f[2].sum(); a01 = 1 - float(fine[19]); a03 = 1 - float(fine[59])                  # above 0.1 and 0.3 T2 from the fine bins
        return dict(above01=a01, above03=a03, above1=at(1), above2=at(2), above3=at(3), slope=sl, amp=at(1) * math.exp(1 / sl), mean=float((f[0] * ec).sum() / f[0].sum()),
                    et=float((f[2] * (np.arange(200) + 0.5) * 0.1).sum() / f[2].sum()), partner_cold=float(f[3][:6].sum() / f[3].sum()), partner_mean=float((f[3] * ec).sum() / f[3].sum()))

    s = stats(F); ps = [stats(p) for p in per]; se = {k: (float(np.std([q[k] for q in ps], ddof=1) / math.sqrt(len(ps))) if len(ps) > 1 else 0.0) for k in s}
    out.update({f"{lab}_first": F, f"{lab}_n": n}); out.update({f"{lab}_{k}": v for k, v in s.items()}); out.update({f"{lab}_{k}_se": v for k, v in se.items()})
    print(f"{lab}: {len(stems)} runs, {n:.3g} first collisions; outcome above 0.1, 0.3, 1, 2, 3 kT2: {s['above01']:.3f} {s['above03']:.3f} {s['above1']:.3f} {s['above2']:.3f} {s['above3']:.4f}; "
          f"slope 1-3: {s['slope']:.3f} +- {se['slope']:.3f} T2, amplitude {s['amp']:.3f}; mean outcome {s['mean']:.3f} kT2; mean Et before {s['et']:.2f} kT2; partner below 0.3 kT2: {s['partner_cold']:.3f}, partner mean {s['partner_mean']:.3f} kT2")
np.savez(sys.argv[1], **out)
