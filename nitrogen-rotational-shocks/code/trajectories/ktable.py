"""Collision kernel from the trajectory table of ctc_ktable.py: loading, packing for the DSMC code, and the same
collision rule as dsmc_shock_vdf16.cpp in vectorised form, for tests in a homogeneous gas.

Energies in K; relative speed c_r in units in which Et = c_r^2 / 4 (molecular mass 1, k = 1), as in the DSMC code.

Usage: python ktable.py pack <out.bin> <parts k_*.npz>          writes the binary table read by the DSMC code
       python ktable.py info <parts>                             number of outcomes per node, cross-sections"""
import sys, math
import numpy as np

OM = 0.74; DREF = 4.17


def sig_vhs(et):
    return math.pi * DREF ** 2 * (273.0 / np.maximum(et, 1e-12)) ** (OM - 0.5) / math.gamma(2.5 - OM)


class KTable:
    def __init__(self, files):
        parts = [np.load(f) for f in files]; p0 = parts[0]
        self.E = p0["etot"].astype(float); self.B = p0["bmax"].astype(float); self.X = p0["x"].astype(float); self.pairs = p0["pairs"].astype(int)
        self.nE, self.nX, self.nP = len(self.E), len(self.X), len(self.pairs)
        self.pid = -np.ones((self.nX, self.nX), int)
        for q, (i, j) in enumerate(self.pairs): self.pid[i, j] = q
        nodes = {}
        for p in parts:
            for k in p.files:
                if k[0] == "n" and k[1:].isdigit(): nodes[int(k[1:])] = p[k]
        miss = [q for q in range(self.nE * self.nP) if q not in nodes]
        if miss: raise SystemExit(f"{len(miss)} nodes missing, first {miss[:5]}")
        self.cnt = np.array([nodes[q].shape[1] for q in range(self.nE * self.nP)]); self.off = np.concatenate([[0], np.cumsum(self.cnt)[:-1]])
        self.d = np.concatenate([nodes[q] for q in range(self.nE * self.nP)], axis=1).astype(np.float32)      # rows: dx1, dx2, cos chi

    def pack(self, fn):
        with open(fn, "wb") as f:
            np.array([self.nE, self.nX, self.nP], np.int32).tofile(f); self.E.tofile(f); self.B.tofile(f); self.X.tofile(f); self.pairs.astype(np.int32).tofile(f)
            for q in range(self.nE * self.nP):
                np.array([self.cnt[q]], np.int32).tofile(f); np.ascontiguousarray(self.d[:, self.off[q]:self.off[q] + self.cnt[q]].T).tofile(f)

    def energy_node(self, eK):
        """Lower neighbour, weight of the upper one, and the larger of the two impact-parameter limits."""
        kt = np.clip(np.searchsorted(self.E, eK, side="right") - 1, 0, self.nE - 1); up = np.minimum(kt + 1, self.nE - 1)
        wk = np.where(kt < self.nE - 1, np.log(eK / self.E[kt]) / np.log(np.where(up > kt, self.E[up] / self.E[kt], 2.0)), 0.0)
        return kt, wk, np.maximum(self.B[kt], self.B[up])

    def sigma_cr(self, cr, e1, e2):
        """sigma * c_r (A^2 times speed) used for the selection of collisions, and the branch (True: table)."""
        eK = 0.25 * cr * cr + e1 + e2; tab = eK >= self.E[0]; _, _, kbig = self.energy_node(np.maximum(eK, self.E[0]))
        return np.where(tab, math.pi * kbig ** 2, sig_vhs(0.25 * cr * cr)) * cr, tab

    def share_node(self, x, rng):
        m = np.clip(np.searchsorted(self.X, x, side="right") - 1, 0, self.nX - 1); up = np.minimum(m + 1, self.nX - 1)
        w = np.where(up > m, (x - self.X[m]) / np.where(up > m, self.X[up] - self.X[m], 1.0), 0.0)
        return np.where(rng.random(len(x)) < w, up, m)

    def collide(self, cr, e1, e2, rng):
        """Outcome for accepted pairs of the table branch: new e1, e2, relative translational energy, cos(chi), and a
        mask of the pairs for which the collision does not take place (impact parameter beyond the node's range)."""
        n = len(cr); etot = 0.25 * cr * cr + e1 + e2; kt, wk, kbig = self.energy_node(etot)
        kn = np.where((kt < self.nE - 1) & (rng.random(n) < wk), kt + 1, kt)
        null = (self.B[kn] < kbig) & (rng.random(n) * kbig ** 2 >= self.B[kn] ** 2)
        x1 = e1 / etot; x2 = e2 / etot; sw = x1 > x2; x1, x2 = np.where(sw, x2, x1), np.where(sw, x1, x2)
        na = self.share_node(x1, rng); nb = self.share_node(x2, rng); fl = na > nb
        na, nb = np.where(fl, nb, na), np.where(fl, na, nb); x1, x2 = np.where(fl, x2, x1), np.where(fl, x1, x2); sw = sw ^ fl
        pq = self.pid[na, nb]
        for _ in range(self.nX):
            bad = (pq < 0) & (nb > na)
            if not bad.any(): break
            nb = np.where(bad, nb - 1, nb); pq = self.pid[na, nb]
        for _ in range(self.nX):
            bad = pq < 0
            if not bad.any(): break
            na = np.where(bad, na - 1, na); nb = np.where(bad, na, nb); pq = self.pid[na, nb]
        q0 = kn * self.nP + pq; x1n = x1.copy(); x2n = x2.copy(); cc = np.zeros(n); todo = np.ones(n, bool)
        for _ in range(20):
            if not todo.any(): break
            idx = self.off[q0[todo]] + (rng.random(todo.sum()) * self.cnt[q0[todo]]).astype(int); v = self.d[:, idx]
            a1 = x1[todo] + v[0]; a2 = x2[todo] + v[1]; ok = (a1 >= 0) & (a2 >= 0) & (a1 + a2 < 1 - 1e-9); w = np.where(todo)[0]
            cc[w] = v[2]; x1n[w[ok]] = a1[ok]; x2n[w[ok]] = a2[ok]; todo[w[ok]] = False
        e1n = np.where(sw, x2n, x1n) * etot; e2n = np.where(sw, x1n, x2n) * etot
        return e1n, e2n, (1 - x1n - x2n) * etot, cc, null, todo


if __name__ == "__main__":
    T = KTable(sys.argv[3:] if sys.argv[1] == "pack" else sys.argv[2:])
    print(f"{T.nE} energies {T.E.tolist()} K, impact-parameter limits {T.B.tolist()} A, {T.nP} pairs of shares, {T.d.shape[1]} outcomes ({T.cnt.min()} to {T.cnt.max()} per node)")
    for k in range(T.nE):
        s = slice(T.off[k * T.nP], T.off[k * T.nP] + T.cnt[k * T.nP]); c = T.d[2, s]
        print(f"   Etot {T.E[k]:g} K, both rotors at rest: cross-section with cos(chi) < 0.9: {math.pi * T.B[k] ** 2 * np.mean(c < 0.9):.1f} A^2; viscosity cross-section {math.pi * T.B[k] ** 2 * np.mean(1 - c ** 2):.1f} A^2 "
              f"(VHS {2 / 3 * sig_vhs(T.E[k]):.1f}); mean share given to each rotor {T.d[0, s].mean():.4f}")
    if sys.argv[1] == "pack":
        T.pack(sys.argv[2]); print("written", sys.argv[2])
