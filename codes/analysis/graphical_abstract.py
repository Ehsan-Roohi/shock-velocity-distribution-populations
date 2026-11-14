"""Graphical abstract for JFM: one panel, aspect ratio 1.2:1, JPEG at 300 dpi, almost no text.

Who carries the precursor of a Mach-8 hard-sphere shock.  Of the upstream-moving molecules found at a given distance
ahead of the plane where their mass fraction is 1e-2, the share that flew there freely from that plane (blue) and the
share that was produced on the way by collisions with the incoming stream (orange).  Boundary: linear cascade model
driven by the flux measured in DSMC (survival / total).  Circles: free-flight flag carried by the DSMC molecules.

Usage: python graphical_abstract.py <shock_data dir>   ->  paper/jfm_main/graphical_abstract.jpg
"""
import sys, os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans"})
BLUE, ORANGE, DARK, NAVY = "#1f6fb2", "#f28e2b", "#1a1a1a", "#0b3156"


def main(root):
    out = os.path.join(root, "analysis", "out")
    xref = json.load(open(os.path.join(out, "precursor_cascade.json")))["cascade_dsmc_source"]["w0p5_M8"]["x_ref"]
    FL = json.load(open(os.path.join(out, "free_flight.json")))["M8_w0.5"]
    a = np.mean([np.loadtxt(f) for f in sorted(glob.glob(os.path.join(root, "cascade", "tab_M8_w0.5_s*.txt")))], 0)
    ok = a[:, 1] > 0; dc = -a[ok, 0]; share = a[ok, 2] / a[ok, 1]
    dmax = 5.2; k = dc <= dmax
    d = np.concatenate(([0.0], dc[k], [dmax])); s = np.concatenate(([1.0], share[k], [np.interp(dmax, dc, share)]))
    lev = ["0.003", "0.001", "0.0003", "0.0001"]
    df = np.array([xref - FL["pooled"][l]["x"] for l in lev]); sf = np.array([FL["levels"][l]["mean"][0] for l in lev])
    print("free-flight share, model at DSMC stations:", np.round(np.interp(df, d, s), 3), "DSMC flag:", np.round(sf, 3))

    fig = plt.figure(figsize=(6.0, 5.0)); ax = fig.add_axes([0.215, 0.125, 0.745, 0.835])
    ax.fill_between(d, s, 1.0, color=ORANGE, lw=0)
    ax.fill_between(d, 0.0, s, color=BLUE, lw=0)
    ax.plot(d, s, color=DARK, lw=2.6, solid_capstyle="round")
    ax.plot(df, sf, "o", ms=12, mfc="white", mec=DARK, mew=2.4, zorder=5, clip_on=False)
    ax.set_xlim(0, dmax); ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0", "50%", "100%"]); ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=18, length=7, width=1.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(1.8)
    ax.plot(1.0, 0.0, ">", color="k", ms=13, transform=ax.transAxes, clip_on=False, zorder=6)
    ax.set_xlabel("distance ahead of the shock", fontsize=21, labelpad=12)
    ax.set_ylabel("upstream-moving molecules", fontsize=19.5, labelpad=6)
    ax.text(0.60, 0.70, "regenerated\nby collisions", transform=ax.transAxes, fontsize=30, fontweight="bold", color="white", ha="center", va="center", linespacing=1.0)
    ax.text(0.545, 0.445, "free flight", transform=ax.transAxes, fontsize=24, fontweight="bold", color=NAVY, ha="center", va="center")
    ax.annotate("", xy=(0.045, 0.40), xytext=(0.315, 0.445), xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", lw=2.4, color=NAVY, mutation_scale=24, shrinkA=0, shrinkB=0))
    tmp = os.path.join(root, "paper", "jfm_main", "graphical_abstract.png")
    fig.savefig(tmp, dpi=300, facecolor="white")
    im = Image.open(tmp).convert("RGB"); jpg = tmp[:-4] + ".jpg"
    im.save(jpg, "JPEG", quality=95, dpi=(300, 300), subsampling=0)
    im.resize((283, 236), Image.LANCZOS).save(tmp[:-4] + "_thumbnail_check.png")        # 2.4 cm x 2 cm at 300 dpi
    print(jpg, im.size, "aspect", round(im.size[0] / im.size[1], 3), round(os.path.getsize(jpg) / 1e3), "kB")


if __name__ == "__main__":
    main(sys.argv[1])
