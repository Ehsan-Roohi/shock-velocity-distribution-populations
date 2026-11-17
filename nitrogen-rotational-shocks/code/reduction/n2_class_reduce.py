"""Seeds combined: number of samples in every cell by collision class (0 unscattered stream, 1 one collision, 2 two
collisions, 3 three or more or born downstream, 4 scattered molecule whose last partner was a stream molecule) and by
generation (0-3), with the density profile, for the nitrogen DSMC shocks; also the number of molecules that still carry
the rotational energy of the free stream (from the .erv.bin file) and the rotational temperature.
Usage (in the DSMC folder): python n2_class_reduce.py <out.npz> <label>=<glob of .txt files> [...]"""
import sys, glob
import numpy as np
from n2_check import read

out = {}
for spec in sys.argv[2:]:
    lab, pat = spec.split("=", 1); stems = sorted(s[:-4] for s in glob.glob(pat)); stems = [s for s in stems if "steps_done 215000" in open(s + ".txt").read()]
    cls = None; mom = None; ns = 0; tag = None; cnt = None; rot = None
    for s in stems:
        h, c, m, n_, r, fly = read(s); nc, nb = int(h["nc"]), int(h["nb"])
        a = np.fromfile(s + ".cls.bin", dtype=np.float64, offset=4 * 9 * nc * nb).reshape(9, nc, 4)[:, :, 0]
        t = np.fromfile(s + ".erv.bin").reshape(2, nc, nb)[1].sum(1)
        cls = a if cls is None else cls + a; mom = m if mom is None else mom + m; ns += n_; tag = t if tag is None else tag + t; cnt = c.sum(1) if cnt is None else cnt + c.sum(1); rot = r if rot is None else rot + r
    M, gam, dx = h["M"], h["gamma"], h["dx"]; n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); T2 = (2 * gam * M * M - (gam - 1)) / (gam + 1) / n2
    x = h["xmin"] + (np.arange(nc) + 0.5) * dx; N = mom[:, 0]; n = N * h["FN"] / (ns * dx); ic = int(np.argmin(np.abs((n - 1) / (n2 - 1) - 0.5))); m = (x - x[ic] > -30) & (x - x[ic] < 15)
    out.update({f"{lab}_x": x[m] - x[ic], f"{lab}_n": n[m], f"{lab}_cls": cls[:, m], f"{lab}_tag": tag[m], f"{lab}_cnt": cnt[m], f"{lab}_N": N[m], f"{lab}_Trot": (rot[0] / N)[m] / T2,
                f"{lab}_n2": n2, f"{lab}_T2": T2, f"{lab}_M": M, f"{lab}_seeds": len(stems)})
    i = int(np.argmin(np.abs((n - 1) / (n2 - 1) - 0.2))); print(lab, "seeds", len(stems), "class shares at n* = 0.2:", np.round(cls[:5, i] / cls[:5, i].sum(), 3), "generation shares:", np.round(cls[5:, i] / cls[5:, i].sum(), 3),
                                                                 "exchanged fraction", round(1 - tag[i] / cnt[i], 3))
np.savez_compressed(sys.argv[1], **out)
