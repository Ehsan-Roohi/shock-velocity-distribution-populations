"""Exchange rules with exact detailed balance against the classical-trajectory table: which form reproduces the mean
change of the rotational energy of a molecule for every pre-collision state (Et, e, e of the partner)?

All rules propose a redistribution drawn from the equilibrium distribution at fixed collision energy and accept it with
a probability that is symmetric in the states before and after, so the equilibrium state is exact.

  seq-const   each molecule in turn, Ec = Et + e, constant acceptance p             (ordinary Larsen-Borgnakke)
  seq-min     each molecule in turn, a = ainf + (a0 - ainf) exp(-min(e, e')/(x0 Ec))
  seq-max     each molecule in turn, a = a0 exp(-max(e, e')/(x0 Ec))
  pair-const  both molecules together, Ec = Et + e1 + e2, constant acceptance p
  pair-max    both together, A = a0 exp(-max(R, R')/(x0 Ec)),  R = e1 + e2 before, R' after
  pair-maxe   both together, A = a0 exp(-max(e1, e2, e1', e2')/(x0 Ec))
  pair-min    both together, A = ainf + (a0 - ainf) exp(-min(R, R')/(x0 Ec))
  pair-tr     both together, A = ainf + (a0 - ainf) min(Et/Ec, Et'/Ec)^k

Usage: python sa_fit2.py <json of ctc_table_analyse.py>   or   python sa_fit2.py logs <log files of ctc_table.py>"""
import sys, math, json, re
import numpy as np
from scipy.optimize import least_squares

om = 0.74; eta = 2.5 - om; bmax = 5.5
sv = lambda et: math.pi * 4.17 ** 2 * (273.0 / et) ** (om - 0.5) / math.gamma(2.5 - om)
S = []          # (Et, e1, e2, mean change of e1, of e2, per VHS collision)
if sys.argv[1] == "logs":
    pat = re.compile(r"Et (\S+) K, e1 (\S+) K, e2 (\S+) K;.*mean change of e1 (\S+) K, of e2 (\S+) K")
    for f in sys.argv[2:]:
        for ln in open(f):
            m = pat.search(ln)
            if m:
                et, e1, e2, d1, d2 = (float(v) for v in m.groups()); r = math.pi * bmax ** 2 / sv(et); S.append((et, e1, e2, r * d1, r * d2))
else:
    for q in json.load(open(sys.argv[1])):
        r = math.pi * bmax ** 2 / sv(q["Et"]); S.append((q["Et"], q["e1"], q["e2"], r * q["d1"], r * q["d2"]))
S = np.array(S)
if "emax" in sys.argv[-1]: S = S[(S[:, 1] <= float(sys.argv[-1][5:])) & (S[:, 2] <= float(sys.argv[-1][5:]))]
print(f"{len(S)} states; relative energies {sorted(set(S[:, 0]))}")
rng = np.random.default_rng(5); NS = 20000
U1 = rng.random(NS); U2 = rng.random(NS); UA = rng.random(NS); UB = rng.random(NS)
# pair redistribution: (f1, f2, ft) ~ Dirichlet(1, 1, eta) = equilibrium shares of two rotors and the relative translation
G = rng.gamma([1.0, 1.0, eta], size=(NS, 3)); G /= G.sum(1, keepdims=True)


def seq(par, kind, et, e1, e2):
    """Mean changes of e1, e2 for the sequential rule (molecule 1 first, then 2, and the other order, averaged)."""
    out = np.zeros(2)
    for order in (0, 1):
        ea, eb = (e1, e2) if order == 0 else (e2, e1); etl = np.full(NS, et); res = []
        for e, u, ua in ((ea, U1, UA), (eb, U2, UB)):
            ec = etl + e; en = ec * (1 - u ** (1 / eta))
            if kind == "const": a = par[0]
            elif kind == "min": a = par[1] + (par[0] - par[1]) * np.exp(-np.minimum(e, en) / ec / par[2])
            else: a = par[0] * np.exp(-np.maximum(e, en) / ec / par[1])
            ok = ua < a; res.append(np.where(ok, en - e, 0.0).mean()); etl = np.where(ok, ec - en, etl)
        out += np.array(res if order == 0 else res[::-1]) / 2
    return out


def pair(par, kind, et, e1, e2):
    ec = et + e1 + e2; n1 = G[:, 0] * ec; n2 = G[:, 1] * ec
    if kind == "const": a = par[0]
    elif kind == "max": a = par[0] * np.exp(-np.maximum(e1 + e2, n1 + n2) / ec / par[1])
    elif kind == "maxe": a = par[0] * np.exp(-np.maximum(np.maximum(e1, e2), np.maximum(n1, n2)) / ec / par[1])
    elif kind == "min": a = par[1] + (par[0] - par[1]) * np.exp(-np.minimum(e1 + e2, n1 + n2) / ec / par[2])
    else: a = par[1] + (par[0] - par[1]) * np.minimum(et / ec, 1 - (n1 + n2) / ec) ** par[2]      # power of the smaller translational share
    ok = UA < a
    return np.array([np.where(ok, n1 - e1, 0.0).mean(), np.where(ok, n2 - e2, 0.0).mean()])


MODELS = {"seq-const": (lambda p, *s: seq(p, "const", *s), [0.2], [1.0]), "seq-min": (lambda p, *s: seq(p, "min", *s), [0.45, 0.1, 0.1], [1, 1, 5]),
          "seq-max": (lambda p, *s: seq(p, "max", *s), [0.6, 0.5], [1, 20]), "pair-const": (lambda p, *s: pair(p, "const", *s), [0.3], [1.0]),
          "pair-max": (lambda p, *s: pair(p, "max", *s), [0.8, 0.5], [1, 20]), "pair-maxe": (lambda p, *s: pair(p, "maxe", *s), [0.8, 0.5], [1, 20]),
          "pair-min": (lambda p, *s: pair(p, "min", *s), [0.8, 0.1, 0.2], [1, 1, 5]), "pair-tr": (lambda p, *s: pair(p, "tr", *s), [0.9, 0.05, 1.0], [1, 1, 8])}
best = {}
for name, (fn, p0, hi) in MODELS.items():
    res = lambda p: np.concatenate([(fn(p, et, e1, e2) - np.array([d1, d2])) / et for et, e1, e2, d1, d2 in S])
    f = least_squares(res, p0, bounds=([1e-3] * len(p0), hi), diff_step=0.02); best[name] = f.x
    per = {et: np.sqrt(np.mean(res(f.x).reshape(-1, 2)[S[:, 0] == et] ** 2)) for et in sorted(set(S[:, 0]))}
    print(f"{name:11s}: parameters " + " ".join(f"{v:.3f}" for v in f.x) + f"; rms residual {np.sqrt(np.mean(f.fun ** 2)):.4f} of Et; by Et: " + " ".join(f"{k:g}: {v:.3f}" for k, v in per.items()))
names = ["seq-const", "seq-min", "pair-min", "pair-tr"]
print("\n   Et     e1     e2 | trajectories: change of e1, e2 | " + " | ".join(f"{n:>17s}" for n in names))
for et, e1, e2, d1, d2 in S[np.lexsort((S[:, 2], S[:, 1], S[:, 0]))]:
    print(f"{et:6g} {e1:6g} {e2:6g} | {d1:13.1f} {d2:8.1f}        | " + " | ".join("{:8.1f} {:8.1f}".format(*MODELS[n][0](best[n], et, e1, e2)) for n in names))
