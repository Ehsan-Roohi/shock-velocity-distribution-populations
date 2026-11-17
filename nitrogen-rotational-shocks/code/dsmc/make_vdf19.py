"""Makes dsmc_shock_vdf19.cpp from version 18.

freeze=1: collisions between two unscattered molecules (collision count 0) are elastic with the VHS cross-section, in
          every exchange rule and with the trajectory kernel. The stream then keeps the distribution it enters with
          (rin=<file>: the measured rotational populations; translation at T1) until its first counted collision, as
          the frozen free stream of the experiment does. These collisions are not counted in any version.
rhrin=1:  with rin, the downstream reservoir is held at the Rankine-Hugoniot state of a stream that carries the mean
          rotational energy of the measured populations (frozen), instead of that of the Boltzmann stream at T1.
          The energy bins of the histograms keep the Boltzmann T2 as unit, so that the files compare with earlier
          versions. The header records n2res, u2res, T2res and rin_me.
Without freeze and rhrin the random sequence and the solution are those of version 18.
Usage: python make_vdf19.py <folder with dsmc_shock_vdf18.cpp>"""
import sys, os
d = sys.argv[1]; t = open(os.path.join(d, "dsmc_shock_vdf18.cpp"), encoding="utf-8").read()


def rep(a, b, n=1):
    global t
    assert t.count(a) == n, (t.count(a), a[:70]); t = t.replace(a, b)


rep("static std::vector<double> RIN_C, RIN_LO, RIN_HI;   // measured upstream rotational distribution (version 18)\n",
    "static std::vector<double> RIN_C, RIN_LO, RIN_HI;   // measured upstream rotational distribution (version 18)\n"
    "static double RIN_ME = 1.0;                          // its mean energy in units of k T1 (version 19)\n")
rep("me += w * 0.5 * (RIN_LO[i] + RIN_HI[i]); } me /= s;", "me += w * 0.5 * (RIN_LO[i] + RIN_HI[i]); } me /= s; RIN_ME = me;")
rep("    add_maxwell(p, xmin, 0, n1, u1, T1, FN, 0);\n    add_maxwell(p, 0, xmax, n2, u2, T2, FN, 3);\n",
    "    const bool FREEZE = arg(argc, argv, \"freeze\", 0) > 0;     // version 19: collisions between two unscattered molecules are elastic\n"
    "    double n2R = n2, u2R = u2, T2R = T2;                         // state of the downstream reservoir\n"
    "    if (arg(argc, argv, \"rhrin\", 0) > 0 && !RIN_C.empty()) {     // Rankine-Hugoniot state of a stream with frozen rotational energy RIN_ME\n"
    "        auto fr = [&](double r) { return 0.5 * u1 * u1 * (1 - 1 / (r * r)) + 2.5 + RIN_ME - 3.5 * (u1 * u1 + 1 - u1 * u1 / r) / r; };\n"
    "        double a = 1.5, b = 0.999 * (gam + 1) / (gam - 1);\n"
    "        for (int it = 0; it < 200; it++) { double c = 0.5 * (a + b); if (fr(a) * fr(c) <= 0) b = c; else a = c; }\n"
    "        n2R = 0.5 * (a + b); T2R = (u1 * u1 + 1 - u1 * u1 / n2R) / n2R; u2R = u1 / n2R;\n"
    "        N0 = (n1 * (0 - xmin) + n2R * (xmax - 0)) / FN;\n"
    "        printf(\"downstream reservoir for the measured stream: n2 %.5f (Boltzmann stream %.5f), T2 %.5f (%.5f), u2 %.5f\\n\", n2R, n2, T2R, T2, u2R);\n"
    "    }\n"
    "    add_maxwell(p, xmin, 0, n1, u1, T1, FN, 0);\n    add_maxwell(p, 0, xmax, n2R, u2R, T2R, FN, 3);\n")
rep("add_maxwell(p, xmax, xmax + W, n2, u2, T2, FN, 3);", "add_maxwell(p, xmax, xmax + W, n2R, u2R, T2R, FN, 3);")
rep("add_maxwell(p, xmax - dx, xmax, n2, u2, T2, FN, 3);", "add_maxwell(p, xmax - dx, xmax, n2R, u2R, T2R, FN, 3);")
rep("* FN / (n2 - n1);", "* FN / (n2R - n1);")
rep("int kt = -1; double wk = 0, kbig = 0;\n                if (KT.on) {",
    "int kt = -1; double wk = 0, kbig = 0; const bool frz = FREEZE && p.k[i] == 0 && p.k[j] == 0;\n                if (KT.on && !frz) {")
rep("if (ROT && !KT.on) {", "if (ROT && !KT.on && !frz) {")
rep("ktab %d\\nkt1 %.10g\\nkchi %.10g\\n\",", "ktab %d\\nkt1 %.10g\\nkchi %.10g\\nn2res %.10g\\nu2res %.10g\\nT2res %.10g\\nrin_me %.10g\\nfreeze %d\\n\",")
rep("(int)KT.on, kt1, kchi);", "(int)KT.on, kt1, kchi, n2R, u2R, T2R, RIN_ME, (int)FREEZE);")
rep("See make_vdf18.py.\n//\n// Build:", "See make_vdf18.py.\n// Version 19: freeze=1 (collisions between unscattered molecules elastic), rhrin=1 (downstream reservoir for the measured\n// stream). See make_vdf19.py.\n//\n// Build:")
open(os.path.join(d, "dsmc_shock_vdf19.cpp"), "w", encoding="utf-8", newline="\n").write(t); print("written dsmc_shock_vdf19.cpp,", len(t.splitlines()), "lines")
