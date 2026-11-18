"""Single-collision results of the ab initio potential (Hellmann 2013, version B) against the two-centre Lennard-Jones
model: the quantities of Table S1 and the share-law fit of Eq. (1) of the Letter, for the ensembles M12.9 and M7.

Usage: python ai_vs_lj_single.py <data folder>   (reads ctc_cluster/{an,cond,exc}_M*tm.* and _M*ai.*)"""
import sys, os, json, math
import numpy as np

CT = os.path.join(sys.argv[1], "ctc_cluster")


def row(s):
    d = np.load(os.path.join(CT, f"exc_{s}.npz")); E = d["E"]; sv = float(d["sig_vhs"]); f = d["ctc"] / sv; T2 = float(d["T2"])
    i = lambda x: int(np.argmin(np.abs(E - x))); th = 2.0 / math.log(f[i(1.0)] / f[i(3.0)]); A = f[i(1.0)] * math.exp(1.0 / th)
    a = json.load(open(os.path.join(CT, f"an_{s}.json"))); c = np.load(os.path.join(CT, f"cond_{s}.npz"))
    out = dict(g01=float(c["gain"][i(0.1)]), s03=float(f[i(0.3)]), s1=float(f[i(1.0)]), A=A, Ts=th, gain=float(d["gain_ctc"]) / T2,
               eup=a["ctc"]["mean"], sigmu=a["sig_mu_ctc"])
    # share law, the form of Eq. (1): one fit of ln sigma(e' > x Etot) between x = 0.05 and 0.4 over the bins 2-10 kT2
    ed = c["et_edges"]; SS = c["S"]
    ff = [c["bin_fS"][j] for j in range(len(c["bin_et"])) if np.isfinite(c["bin_et"][j]) and 2 <= ed[j] < 10]
    fa = np.mean(ff, 0); mm = (SS >= 0.05) & (SS <= 0.4); pf = np.polyfit(SS[mm], np.log(fa[mm]), 1)
    out.update(eq1_amp=math.exp(pf[1]), eq1_scale=-1 / pf[0])
    return out


for m in ("M12.9", "M7"):
    lj, ai = row(m + "tm"), row(m + "ai")
    print(f"== {m}: quantity, Lennard-Jones, ab initio, relative difference")
    for k in lj:
        print(f"   {k:10s} {lj[k]:9.4f} {ai[k]:9.4f}  {100 * (ai[k] / lj[k] - 1):+6.1f}%")
