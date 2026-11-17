"""Makes dsmc_shock_vdf20.cpp from version 19.

freeze=2: a collision exchanges no rotational energy (elastic, VHS cross-section, also with the trajectory kernel)
          when both molecules lie inside the velocity range of the cold stream, |v - (u1, 0, 0)| < wfrz (kT1/m)^1/2
          (default wfrz=4, which holds 99.9% of a Maxwellian at T1). The rule depends on the velocities of the pair
          only, not on the collision history, so the labels (collision count, class) do not act on the solution.
          It keeps the stream frozen, as the free stream of the experiment is, up to the shock.
freeze=1: as version 19 (collisions between two unscattered molecules are elastic; history-dependent).
The header records freeze and wfrz. With freeze=0 or 1 the random sequence and the solution are those of version 19.
Usage: python make_vdf20.py <folder with dsmc_shock_vdf19.cpp>"""
import sys, os
d = sys.argv[1]; t = open(os.path.join(d, "dsmc_shock_vdf19.cpp"), encoding="utf-8").read()


def rep(a, b, n=1):
    global t
    assert t.count(a) == n, (t.count(a), a[:70]); t = t.replace(a, b)


rep("    const bool FREEZE = arg(argc, argv, \"freeze\", 0) > 0;     // version 19: collisions between two unscattered molecules are elastic\n",
    "    const int FREEZE = (int)arg(argc, argv, \"freeze\", 0);      // 1 (version 19): collisions between two unscattered molecules are elastic;\n"
    "    const double wfrz = arg(argc, argv, \"wfrz\", 4), WF2 = wfrz * wfrz * T1;   // 2 (version 20): elastic when both lie in the stream's velocity range\n"
    "    auto inwin = [&](int m) { double a = p.vx[m] - u1; return a * a + p.vy[m] * p.vy[m] + p.vz[m] * p.vz[m] < WF2; };\n")
rep("const bool frz = FREEZE && p.k[i] == 0 && p.k[j] == 0;",
    "const bool frz = FREEZE == 1 ? (p.k[i] == 0 && p.k[j] == 0) : (FREEZE == 2 && inwin(i) && inwin(j));")
rep("rin_me %.10g\\nfreeze %d\\n\",", "rin_me %.10g\\nfreeze %d\\nwfrz %.10g\\n\",")
rep("RIN_ME, (int)FREEZE);", "RIN_ME, FREEZE, wfrz);")
rep("// stream). See make_vdf19.py.\n", "// stream). See make_vdf19.py.\n// Version 20: freeze=2 (no rotational exchange when both molecules lie in the velocity range of the cold stream). See make_vdf20.py.\n")
open(os.path.join(d, "dsmc_shock_vdf20.cpp"), "w", encoding="utf-8", newline="\n").write(t); print("written dsmc_shock_vdf20.cpp,", len(t.splitlines()), "lines")
