"""Makes cascade_linear_rot.cpp from cascade_linear.cpp: the linear recoil cascade for a diatomic gas.
With zrot > 0 every molecule carries a rotational energy (two degrees of freedom, gamma = 7/5); stream molecules
have rotational temperature T1, source molecules Trs*T2 (Trs = 1: downstream equilibrium); in every collision each
partner exchanges rotational and translational energy with probability 1/zrot by the Larsen-Borgnakke rule, exactly
as in dsmc_shock_vdf7.cpp. With zrot = 0 the program is the monatomic one, with the same random-number sequence.
Usage: python make_cascade_rot.py <folder with cascade_linear.cpp>"""
import sys, os
d = sys.argv[1]; t = open(os.path.join(d, "cascade_linear.cpp"), encoding="utf-8").read()


def rep(a, b):
    global t
    assert t.count(a) == 1, (t.count(a), a[:70]); t = t.replace(a, b)


rep("// Units as in dsmc_shock_vdf.cpp. Build: g++ -O3 -march=x86-64-v2 -o cascade_linear cascade_linear.cpp",
    "// Diatomic gas (zrot > 0): two rotational degrees of freedom, gamma = 7/5, Larsen-Borgnakke exchange with probability\n"
    "// 1/zrot for each partner in every collision (as in dsmc_shock_vdf7.cpp). Stream molecules have rotational\n"
    "// temperature T1; source molecules Trs*T2. A former stream molecule carries a tag until it exchanges rotational\n"
    "// energy. Extra output columns: rotational temperature and (2/3) translational energy of the upstream-moving\n"
    "// molecules, in units of T2, for all / primaries / scattered source / recoil, and the density of tagged molecules.\n"
    "//\n"
    "// Units as in dsmc_shock_vdf.cpp. Build: g++ -O3 -o cascade_linear_rot cascade_linear_rot.cpp")
rep("struct Mol { double x, vx, vy, vz; unsigned char orig, nc; };",
    "struct Mol { double x, vx, vy, vz, er; unsigned char orig, nc, tag; };")
rep("const double PI = 3.14159265358979323846, GAM = 5.0 / 3.0;", "const double PI = 3.14159265358979323846;")
rep("    double alpha = arg(argc, argv, \"alpha\", 1);       // VSS exponent; 1 = isotropic scattering\n",
    "    double alpha = arg(argc, argv, \"alpha\", 1);       // VSS exponent; 1 = isotropic scattering\n"
    "    double zrot = arg(argc, argv, \"zrot\", 0), Trs = arg(argc, argv, \"Trs\", 1);\n"
    "    const bool ROT = zrot > 0; const double GAM = ROT ? 7.0 / 5.0 : 5.0 / 3.0;\n")
rep("    double wkill = u1 - cut * s1;\n",
    "    double wkill = u1 - cut * s1;\n"
    "    const double lbx = 1.0 / (2.5 - omega), prot = ROT ? 1.0 / zrot : 0.0;\n")
rep("std::vector<double> tau((size_t)NT * nc, 0.0), tauv(nc, 0.0);",
    "std::vector<double> tau((size_t)NT * nc, 0.0), tauv(nc, 0.0), ter((size_t)NT * nc, 0.0), tek((size_t)NT * nc, 0.0), ttag(nc, 0.0);")
rep("        m.x = 0; m.orig = 0; m.nc = 0;\n",
    "        m.x = 0; m.orig = 0; m.nc = 0; m.tag = 0; m.er = ROT ? -Trs * T2 * std::log(urand()) : 0.0;\n")
rep("                if (wr < wkill) break;                                   // can no longer be turned back",
    "                // can no longer be turned back (with rotation: not even if all its rotational energy became translational)\n"
    "                if ((ROT ? 0.5 * (wr + std::sqrt(wr * wr + 4 * (q.er + 8 * T1))) : wr) < wkill) break;")
rep("if (len > 0) { double c = w * len / (-q.vx); tau[i] += c; tau[(size_t)cat * nc + i] += c; tauv[i] += c * (-q.vx); }",
    "if (len > 0) { double c = w * len / (-q.vx); tau[i] += c; tau[(size_t)cat * nc + i] += c; tauv[i] += c * (-q.vx);\n"
    "                                       if (ROT) { double ek = 0.5 * (q.vx * q.vx + q.vy * q.vy + q.vz * q.vz);\n"
    "                                                  ter[i] += c * q.er; ter[(size_t)cat * nc + i] += c * q.er; tek[i] += c * ek; tek[(size_t)cat * nc + i] += c * ek;\n"
    "                                                  if (q.tag) ttag[i] += c; } }")
rep("                if (urand() >= pr) continue;\n",
    "                if (urand() >= pr) continue;\n"
    "                double gs = 1.0, erc = 0.0; unsigned char btag = 1;\n"
    "                if (ROT) {\n"
    "                    erc = -T1 * std::log(urand()); double et = 0.25 * g * g; bool ch = false;\n"
    "                    if (urand() < prot) { double ec = et + q.er; q.er = ec * (1 - std::pow(urand(), lbx)); et = ec - q.er; ch = true; q.tag = 0; }\n"
    "                    if (urand() < prot) { double ec = et + erc; erc = ec * (1 - std::pow(urand(), lbx)); et = ec - erc; ch = true; btag = 0; }\n"
    "                    if (ch) gs = 2 * std::sqrt(et) / g;\n"
    "                }\n")
rep("                double mx = 0.5 * (q.vx + cx), my = 0.5 * (q.vy + cy), mz = 0.5 * (q.vz + cz);\n",
    "                if (ROT) { hx *= gs; hy *= gs; hz *= gs; }\n"
    "                double mx = 0.5 * (q.vx + cx), my = 0.5 * (q.vy + cy), mz = 0.5 * (q.vz + cz);\n")
rep("b.orig = 1; b.nc = 1;", "b.orig = 1; b.nc = 1; b.er = erc; b.tag = btag;")
rep('fprintf(f, "# M %.10g omega %.10g L %.10g dx %.10g bias %.10g cut %.10g np %ld seed %d source %s\\n", M, omega, L, dxc, bias, cut, np, seed, src.c_str());',
    'fprintf(f, "# M %.10g omega %.10g L %.10g dx %.10g bias %.10g cut %.10g np %ld seed %d source %s zrot %.10g gamma %.10g Trs %.10g\\n", M, omega, L, dxc, bias, cut, np, seed, src.c_str(), zrot, GAM, Trs);')
rep('        fprintf(f, "%.6g %.8e %.8e %.8e %.8e %.6g\\n", -(i + 0.5) * dxc, tau[i] * k, tau[nc + i] * k, tau[2 * (size_t)nc + i] * k, tau[3 * (size_t)nc + i] * k, tau[i] > 0 ? tauv[i] / tau[i] : 0.0);',
    '        fprintf(f, "%.6g %.8e %.8e %.8e %.8e %.6g", -(i + 0.5) * dxc, tau[i] * k, tau[nc + i] * k, tau[2 * (size_t)nc + i] * k, tau[3 * (size_t)nc + i] * k, tau[i] > 0 ? tauv[i] / tau[i] : 0.0);\n'
    '        if (ROT) {\n'
    '            for (int c = 0; c < NT; c++) { double tt = tau[(size_t)c * nc + i]; fprintf(f, " %.6g", tt > 0 ? ter[(size_t)c * nc + i] / tt / T2 : 0.0); }\n'
    '            for (int c = 0; c < NT; c++) { double tt = tau[(size_t)c * nc + i]; fprintf(f, " %.6g", tt > 0 ? tek[(size_t)c * nc + i] / tt * 2 / 3 / T2 : 0.0); }\n'
    '            fprintf(f, " %.8e", ttag[i] * k);\n'
    '        }\n'
    '        fprintf(f, "\\n");')
rep('fprintf(f, "# x  beta_all  beta_primary  beta_scattered_source  beta_recoil  mean_backward_speed   (beta = density of xi_x<0 molecules / n1)\\n");',
    'fprintf(f, "# x  beta_all  beta_primary  beta_scattered_source  beta_recoil  mean_backward_speed   (beta = density of xi_x<0 molecules / n1)%s\\n",\n'
    '            ROT ? "  Trot/T2: all primary scattered recoil  (2/3)Etr/T2: all primary scattered recoil  beta_tagged" : "");')
open(os.path.join(d, "cascade_linear_rot.cpp"), "w", encoding="utf-8", newline="\n").write(t)
print("written")
