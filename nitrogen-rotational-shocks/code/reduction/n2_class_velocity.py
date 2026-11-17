"""Velocity statistics of the collision classes and generations inside the diatomic shock, seeds combined:
share of the molecules, mean and width of the Rankine-Hugoniot-scaled streamwise velocity (xi - u2)/(u1 - u2).
Usage (in the DSMC folder): python n2_class_velocity.py <glob of .txt files> [n* n* ...]"""
import sys, glob, math
import numpy as np
from n2_check import read

stems = sorted(s[:-4] for s in glob.glob(sys.argv[1])); levels = [float(a) for a in sys.argv[2:]] or [0.1, 0.4, 0.7]
cc = None; mom = None; ns = 0
for s in stems:
    h, cnt, m, n_, rot, fly = read(s); nc, nb = int(h["nc"]), int(h["nb"])
    c = np.fromfile(s + ".cls.bin", dtype=np.float32, count=9 * nc * nb).reshape(9, nc, nb).astype(float)
    cc = c if cc is None else cc + c; mom = m if mom is None else mom + m; ns = ns + n_
M, gam, dx = h["M"], h["gamma"], h["dx"]; n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2); u1 = M * math.sqrt(gam); u2 = u1 / n2
v = h["vlo"] + (np.arange(nb) + 0.5) * h["dv"]; xi = (v - u2) / (u1 - u2)
n = mom[:, 0] * h["FN"] / (ns * dx); nst = (n - 1) / (n2 - 1)
names = ["class 0: unscattered stream", "class 1: stream molecule, one collision", "class 2: two collisions", "class 3: three or more / hot",
         "class 4: scattered, last partner a stream molecule", "generation 0", "generation 1", "generation 2", "generation 3+"]
print(f"Mach {M:g}, {len(stems)} seeds")
for t in levels:
    i = int(np.argmin(np.abs(nst - t))); f = cc[:, i - 2:i + 3].sum(1); tot = f[:5].sum()
    print(f"n* = {t}:")
    rows = [(names[k], f[k]) for k in range(9)] + [("classes 1 + 2 + 4 together", f[1] + f[2] + f[4]), ("generations 1 + 2 together", f[6] + f[7])]
    for name, w in rows:
        m1 = (w * xi).sum() / w.sum(); sd = math.sqrt((w * (xi - m1) ** 2).sum() / w.sum())
        print(f"   {name:52s} share {w.sum() / tot:.3f}  mean scaled velocity {m1:+.3f}  width {sd:.3f}")
