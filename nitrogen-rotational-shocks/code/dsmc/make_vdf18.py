"""Makes dsmc_shock_vdf18.cpp from version 17: rin=<file> gives the molecules that enter from the upstream reservoir
(and the initial upstream gas) a measured rotational distribution instead of the Boltzmann distribution at T1.

The file has one line per level: lower and upper rotational energy of the level in units of k T1 and its population
(not normalised). A molecule takes a level with the given probability and an energy uniform inside it. The
downstream reservoir keeps the Boltzmann distribution at T2. Without rin the random sequence and the solution are
those of version 17.
Usage: python make_vdf18.py <folder with dsmc_shock_vdf17.cpp>"""
import sys, os
d = sys.argv[1]; t = open(os.path.join(d, "dsmc_shock_vdf17.cpp"), encoding="utf-8").read()


def rep(a, b):
    global t
    assert t.count(a) == 1, (t.count(a), a[:70]); t = t.replace(a, b)


rep("static bool ROT = false;      // diatomic gas: rotational energy with two degrees of freedom\n",
    "static bool ROT = false;      // diatomic gas: rotational energy with two degrees of freedom\n"
    "static std::vector<double> RIN_C, RIN_LO, RIN_HI;   // measured upstream rotational distribution (version 18)\n"
    "static double rot_sample(double T, unsigned char tag) {\n"
    "    if (tag != 0 || RIN_C.empty()) return -T * std::log(1.0 - urand());\n"
    "    double u = urand(); size_t i = (size_t)(std::upper_bound(RIN_C.begin(), RIN_C.end(), u) - RIN_C.begin()); if (i >= RIN_C.size()) i = RIN_C.size() - 1;\n"
    "    return RIN_LO[i] + (RIN_HI[i] - RIN_LO[i]) * urand();\n"
    "}\n")
rep("p.push(px, a, b, c, tag, ROT ? -T * std::log(1.0 - urand()) : 0.0); }", "p.push(px, a, b, c, tag, ROT ? rot_sample(T, tag) : 0.0); }")
rep("    add_maxwell(p, xmin, 0, n1, u1, T1, FN, 0);\n    add_maxwell(p, 0, xmax, n2, u2, T2, FN, 3);\n",
    "    { const std::string rin = sarg(argc, argv, \"rin\", \"\");\n"
    "      if (!rin.empty()) {\n"
    "          FILE* fr = fopen(rin.c_str(), \"r\"); if (!fr) { fprintf(stderr, \"cannot read %s\\n\", rin.c_str()); return 1; }\n"
    "          double a, b, c, s = 0; while (fscanf(fr, \"%lf %lf %lf\", &a, &b, &c) == 3) { RIN_LO.push_back(a); RIN_HI.push_back(b); s += c; RIN_C.push_back(s); } fclose(fr);\n"
    "          double me = 0, q = 0; for (size_t i = 0; i < RIN_C.size(); i++) { double w = RIN_C[i] - q; q = RIN_C[i]; me += w * 0.5 * (RIN_LO[i] + RIN_HI[i]); } me /= s;\n"
    "          for (auto& v : RIN_C) v /= s;\n"
    "          printf(\"upstream rotational distribution from %s: %zu levels, mean energy %.4f k T1\\n\", rin.c_str(), RIN_C.size(), me);\n"
    "      } }\n"
    "    add_maxwell(p, xmin, 0, n1, u1, T1, FN, 0);\n    add_maxwell(p, 0, xmax, n2, u2, T2, FN, 3);\n")
rep("See make_vdf17.py.\n//\n// Build:", "See make_vdf17.py.\n// Version 18: rin=<file> gives the upstream molecules a measured rotational distribution. See make_vdf18.py.\n//\n// Build:")
open(os.path.join(d, "dsmc_shock_vdf18.cpp"), "w", encoding="utf-8", newline="\n").write(t); print("written dsmc_shock_vdf18.cpp,", len(t.splitlines()), "lines")
