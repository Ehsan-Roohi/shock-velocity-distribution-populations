"""Hot rotational group measured by Robben & Talbot (1966) against the fraction of molecules whose rotational energy
has been exchanged, for several exchange rules, at the normalised densities of the measurement.

Usage: python hot_fraction_rules.py <out.json> <class npz> [<class npz> ...]      (labels M7<rule>, M12.9<rule>)"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robben_talbot_1966 as rt

D = {}
for f in sys.argv[2:]:
    d = np.load(f); D.update({k: d[k] for k in d.files})
labs = sorted({k.rsplit("_", 1)[0] for k in D if k.endswith("_cls")}); out = {}
for mach, key in (("M7", "7"), ("M12.9", "12.9")):
    ex = rt.reduce(rt.CASES[key]["rows"]); rho1 = ex[0, 2]; rho2 = ex[-3:, 2].mean(); en = (ex[:, 2] - rho1) / (rho2 - rho1); s = (en > 0.02) & (en < 0.97)
    sel = [l for l in labs if l.startswith(mach)]
    print(f"\nMach {key}: normalised density " + " ".join(f"{v:6.3f}" for v in en[s]))
    print(f"   {'measured hot group':34s} " + " ".join(f"{v:6.3f}" for v in ex[s, 4]))
    out[mach] = dict(nstar=en[s].tolist(), measured=ex[s, 4].tolist())
    for l in sel:
        n, cls, tag, cnt = (D[f"{l}_{k}"] for k in ("n", "cls", "tag", "cnt")); ns = (n - 1) / (float(D[f"{l}_n2"]) - 1)
        m = (ns > 0.005) & (ns < 0.99) & (np.arange(len(ns)) < np.argmax(ns > 0.99) + 1 if (ns > 0.99).any() else True); k = np.argsort(ns[m])
        c = cls[:5] / cls[:5].sum(0); exch = 1 - tag / cnt; coll = 1 - c[0]
        v = np.interp(en[s], ns[m][k], exch[m][k]); w = np.interp(en[s], ns[m][k], coll[m][k])
        print(f"   {l + ': exchanged':34s} " + " ".join(f"{a:6.3f}" for a in v) + f"   rms difference from measured {np.sqrt(np.mean((v - ex[s, 4]) ** 2)):.3f}")
        out[mach][l] = dict(exchanged=v.tolist(), collided=w.tolist())
    print(f"   {'at least one collision (' + sel[-1] + ')':34s} " + " ".join(f"{a:6.3f}" for a in w))
json.dump(out, open(sys.argv[1], "w"), indent=1)
