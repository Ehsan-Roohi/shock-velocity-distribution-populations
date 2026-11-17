"""Collision kernel with exact detailed balance from the trajectories of ctc_kcell.py.

At fixed total energy of the pair the state is the pair of rotor shares (x1, x2), binned into ordered cells (a, b).
From the trajectories of every source cell the flux between cells at equilibrium is estimated,
    F(c, c') = pi_c  n(c -> c') / n(c),      pi_c = integral over the cell of (1 - x1 - x2),
and symmetrised, F <- (F + F^T)/2. The kernel is: with probability F(c, c')/pi_c the pair moves to the cell c' and
takes shares drawn inside c' with the equilibrium density (proportional to 1 - x1 - x2); otherwise the shares stay
as they are. This Markov kernel satisfies detailed balance exactly with respect to the equilibrium distribution of
colliding pairs, whatever the statistical error of the trajectories; the trajectories set the rates. The deflection
angle is drawn from the trajectories of the transition.

Usage: python kcell.py pack <out.bin> <parts c_*.npz> [fake=<energy index>] [frac=<f>] [every=<m>]    binary kernel read by dsmc_shock_vdf16
       python kcell.py info <parts> [fake=<energy index>] [frac=<f>] [every=<m>]
fake=<k> (software test only) uses the trajectories of the energy ETOT[k] for every energy.
frac=<f> (resolution test) keeps a random fraction f of the trajectories of every source cell.
every=<m> (resolution test) keeps every m-th tabulated total energy, including the first and the last."""
import sys, math
import numpy as np

OM = 0.74; DREF = 4.17; CAP = 32


def sig_vhs(et):
    return math.pi * DREF ** 2 * (273.0 / np.maximum(et, 1e-12)) ** (OM - 0.5) / math.gamma(2.5 - OM)


class KCell:
    def __init__(self, files, fake=None, seed=0, frac=1.0, every=1):
        parts = [np.load(f) for f in files]; p0 = parts[0]; rng = np.random.default_rng(seed)
        self.E = p0["etot"].astype(float); self.B = p0["bmax"].astype(float); self.XE = p0["xe"].astype(float); self.src = p0["src"].astype(int)
        self.nE, self.NB, self.nS = len(self.E), len(self.XE) - 1, len(self.src); NB, nS = self.NB, self.nS; NC = NB * NB
        jobs = {}
        for p in parts:
            for k in p.files:
                if k[0] == "j" and k[1:].isdigit(): jobs[int(k[1:])] = p[k]
        if fake is not None: jobs = {e * nS + s: jobs[fake * nS + s] for e in range(self.nE) for s in range(nS)}
        if every > 1:                                                   # coarser grid of total energies (resolution test)
            keep = sorted(set(list(range(0, self.nE, every)) + [self.nE - 1]))
            jobs = {i * nS + s: jobs[k * nS + s] for i, k in enumerate(keep) for s in range(nS) if k * nS + s in jobs}
            self.E, self.B = self.E[keep], self.B[keep]; self.nE = len(keep)
        if frac < 1.0:                                                  # fewer trajectories per source cell (resolution test)
            sub = np.random.default_rng(seed + 7)
            for q in list(jobs):
                a = jobs[q]; n = a.shape[1]; jobs[q] = a[:, np.sort(sub.choice(n, max(1, int(round(frac * n))), replace=False))]
        miss = [q for q in range(self.nE * nS) if q not in jobs]
        if miss: raise SystemExit(f"{len(miss)} jobs missing, first {miss[:5]}")
        self.lookup = -np.ones((NB, NB), int)
        for s, (a, b) in enumerate(self.src): self.lookup[a, b] = s
        # equilibrium weight of every ordered cell: integral of max(1 - x1 - x2, 0)
        m = 240; t = (np.arange(m) + 0.5) / m; lo, w = self.XE[:-1], np.diff(self.XE); pts = lo[:, None] + w[:, None] * t[None, :]            # [NB][m]
        self.pi = np.array([[np.maximum(1 - pts[a][:, None] - pts[b][None, :], 0).mean() * w[a] * w[b] for b in range(NB)] for a in range(NB)])
        self.toff = [0]; dest, cum, coff, ccnt, chi = [], [], [], [], []; self.maxoff = 0.0; self.scaled = 0
        for k in range(self.nE):
            F = np.zeros((NC, NC)); D = {}                           # D[source s] = (destination cell, cos chi) of its trajectories
            for s, (a, b) in enumerate(self.src):
                x1f, x2f, cc = jobs[k * nS + s].astype(float); n = len(cc)
                ia = np.clip(np.searchsorted(self.XE, x1f, side="right") - 1, 0, NB - 1); ib = np.clip(np.searchsorted(self.XE, x2f, side="right") - 1, 0, NB - 1)
                d = ia * NB + ib; dm = ib * NB + ia; D[s] = (d, cc); c1 = np.bincount(d, minlength=NC) / n; c2 = np.bincount(dm, minlength=NC) / n
                if a != b: F[a * NB + b] += self.pi[a, b] * c1; F[b * NB + a] += self.pi[a, b] * c2
                else: F[a * NB + a] += self.pi[a, a] * 0.5 * (c1 + c2)
            Fs = 0.5 * (F + F.T); np.fill_diagonal(Fs, 0.0); pv = self.pi.ravel(); ok = pv > 0
            S = np.zeros(NC); S[ok] = Fs[ok].sum(1) / pv[ok]; self.maxoff = max(self.maxoff, S.max())
            f = np.where(S > 0.995, 0.995 / np.maximum(S, 1e-300), 1.0); self.scaled += int((f < 1).sum()); Fs *= np.minimum(f[:, None], f[None, :])
            for s, (a, b) in enumerate(self.src):
                c = a * NB + b; p = Fs[c] / pv[c]; d, cc = D[s]; lst = np.where(p > 0)[0]; lst = lst[np.argsort(-p[lst])]
                tr = [(c, 1 - p.sum())] + [(int(j), float(p[j])) for j in lst]; run = 0.0
                for j, pj in tr:
                    smp = cc[d == j]
                    if len(smp) == 0 and j != c:                    # transition seen only in the reverse direction: use its deflections
                        a2, b2 = divmod(j, NB); s2 = self.lookup[min(a2, b2), max(a2, b2)]
                        if s2 >= 0:
                            d2, cc2 = D[s2]; back = (b * NB + a) if a2 > b2 else c; smp = cc2[d2 == back]
                    if len(smp) == 0: smp = cc
                    if len(smp) > CAP: smp = rng.choice(smp, CAP, replace=False)
                    run += pj; dest.append(j); cum.append(run); coff.append(len(chi)); ccnt.append(len(smp)); chi.extend(smp.tolist())
                cum[-1] = 1.0; self.toff.append(len(dest))
        self.toff = np.array(self.toff); self.dest = np.array(dest); self.cum = np.array(cum); self.coff = np.array(coff); self.ccnt = np.array(ccnt); self.chi = np.array(chi, np.float32)
        row = np.repeat(np.arange(self.nE * nS), np.diff(self.toff)); self.gcum = row + np.minimum(self.cum, 1.0)      # for a vectorised look-up

    def pack(self, fn):
        with open(fn, "wb") as f:
            np.array([self.nE, self.NB, self.nS, len(self.dest), len(self.chi)], np.int32).tofile(f)
            self.E.tofile(f); self.B.tofile(f); self.XE.tofile(f); self.src.astype(np.int32).tofile(f); self.lookup.astype(np.int32).tofile(f)
            self.toff.astype(np.int32).tofile(f); self.dest.astype(np.int32).tofile(f); self.cum.astype(np.float32).tofile(f)
            self.coff.astype(np.int32).tofile(f); self.ccnt.astype(np.int32).tofile(f); self.chi.tofile(f)

    def energy_node(self, eK):
        kt = np.clip(np.searchsorted(self.E, eK, side="right") - 1, 0, self.nE - 1); up = np.minimum(kt + 1, self.nE - 1)
        wk = np.where(kt < self.nE - 1, np.log(eK / self.E[kt]) / np.log(np.where(up > kt, self.E[up] / self.E[kt], 2.0)), 0.0)
        return kt, wk, np.maximum(self.B[kt], self.B[up])

    def sigma_cr(self, cr, e1, e2):
        eK = 0.25 * cr * cr + e1 + e2; tab = eK >= self.E[0]; _, _, kbig = self.energy_node(np.maximum(eK, self.E[0]))
        return np.where(tab, math.pi * kbig ** 2, sig_vhs(0.25 * cr * cr)) * cr, tab

    def collide(self, cr, e1, e2, rng):
        """The rule of the DSMC code for accepted pairs of the table branch. Returns new e1, e2, relative translational
        energy, cos(chi), the pairs that do not collide (beyond the impact parameters of the chosen energy), and the
        pairs whose shares changed."""
        n = len(cr); NB = self.NB; etot = 0.25 * cr * cr + e1 + e2; kt, wk, kbig = self.energy_node(etot)
        kn = np.where((kt < self.nE - 1) & (rng.random(n) < wk), kt + 1, kt); null = (self.B[kn] < kbig) & (rng.random(n) * kbig ** 2 >= self.B[kn] ** 2)
        x1 = e1 / etot; x2 = e2 / etot; a = np.clip(np.searchsorted(self.XE, x1, side="right") - 1, 0, NB - 1); b = np.clip(np.searchsorted(self.XE, x2, side="right") - 1, 0, NB - 1)
        sw = a > b; a, b = np.where(sw, b, a), np.where(sw, a, b); x1, x2 = np.where(sw, x2, x1), np.where(sw, x1, x2)
        s = self.lookup[a, b]; out = s < 0; q = kn * self.nS + np.maximum(s, 0)
        t = np.minimum(np.searchsorted(self.gcum, q + rng.random(n), side="right"), self.toff[q + 1] - 1); t = np.maximum(t, self.toff[q])
        cc = self.chi[self.coff[t] + (rng.random(n) * self.ccnt[t]).astype(int)].astype(float); moved = (t > self.toff[q]) & ~out
        cc = np.where(out, 2 * rng.random(n) - 1, cc)               # states outside the tabulated cells: elastic, isotropic
        x1n = x1.copy(); x2n = x2.copy(); da, db = np.divmod(self.dest[t], NB); todo = moved.copy()
        while todo.any():
            w = np.where(todo)[0]; u1 = self.XE[da[w]] + (self.XE[da[w] + 1] - self.XE[da[w]]) * rng.random(len(w)); u2 = self.XE[db[w]] + (self.XE[db[w] + 1] - self.XE[db[w]]) * rng.random(len(w))
            xt = 1 - u1 - u2; ok = rng.random(len(w)) * (1 - self.XE[da[w]] - self.XE[db[w]]) < xt; x1n[w[ok]] = u1[ok]; x2n[w[ok]] = u2[ok]; todo[w[ok]] = False
        return np.where(sw, x2n, x1n) * etot, np.where(sw, x1n, x2n) * etot, (1 - x1n - x2n) * etot, cc, null, moved


if __name__ == "__main__":
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); args = [a for a in sys.argv[1:] if "=" not in a]
    fake = int(opt["fake"]) if "fake" in opt else None
    T = KCell(args[2:] if args[0] == "pack" else args[1:], fake=fake, frac=float(opt.get("frac", 1.0)), every=int(opt.get("every", 1)))
    print(f"{T.nE} energies {T.E.tolist()} K, impact-parameter limits {T.B.tolist()} A, {T.NB} bins of the share, {T.nS} source cells, {len(T.dest)} transitions, {len(T.chi)} deflection samples")
    print(f"   largest probability of leaving a cell in one collision before the limit of 0.995: {T.maxoff:.3f}; rows scaled: {T.scaled}")
    if args[0] == "pack":
        T.pack(args[1]); print("written", args[1])
