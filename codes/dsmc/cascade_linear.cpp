// Linear recoil cascade ahead of a strong shock wave.
//
// Far ahead of the shock almost every molecule belongs to the incoming stream (density n1 = 1, velocity u1,
// temperature T1 = 1), and the few others collide only with stream molecules. Their transport is then a
// linear problem: a molecule that enters the region x < 0 through the plane x = 0 moves freely, collides
// with a stream molecule at the VHS rate  n1 A g^(2-2w), and BOTH partners of the collision are followed
// afterwards (the struck stream molecule recoils and is no longer part of the stream). Molecules are
// dropped when they return through x = 0, when they leave through x = -L, or when their speed relative to
// the stream has fallen below u1 - 4 sqrt(T1): such a molecule and its descendants can no longer move
// upstream. The code tallies the steady density of molecules with xi_x < 0 as a function of x, split into
//   primaries          molecules from the source that have not collided yet (pure survival),
//   scattered source   molecules from the source after one or more collisions,
//   recoil             former stream molecules.
//
// Source at x = 0 (flux through the plane, xi_x < 0):
//   source=maxwell     half-range Maxwellian of the downstream state (rho2, u2, T2); the x component can be
//                      sampled from a hotter Maxwellian (bias > 1) with weights, to reach large depths;
//   source=<file>      table  vx  flux  Tperp  (one row per velocity bin, vx < 0), e.g. taken from the
//                      nonlinear DSMC solution at a station in the precursor.
//
// Units as in dsmc_shock_vdf.cpp. Build: g++ -O3 -march=x86-64-v2 -o cascade_linear cascade_linear.cpp
// Run:   ./cascade_linear M=8 omega=0.5 source=maxwell bias=2 L=40 np=20000000 seed=1 out=casc_M8_w0.5_s1.txt
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cstdint>

static uint64_t s_[4];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t next() {
    uint64_t r = s_[0] + s_[3], t = s_[1] << 17;
    s_[2] ^= s_[0]; s_[3] ^= s_[1]; s_[1] ^= s_[2]; s_[0] ^= s_[3]; s_[2] ^= t; s_[3] = rotl(s_[3], 45);
    return r;
}
static inline double urand() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static void seed_rng(uint64_t z) {
    for (int i = 0; i < 4; i++) { z += 0x9e3779b97f4a7c15ULL; uint64_t x = z; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; s_[i] = x ^ (x >> 31); }
}
static inline double nrand() { return std::sqrt(-2 * std::log(urand())) * std::cos(6.283185307179586 * urand()); }

static const char* sarg(int argc, char** argv, const char* key, const char* def) {
    size_t n = strlen(key);
    for (int i = 1; i < argc; i++) if (!strncmp(argv[i], key, n) && argv[i][n] == '=') return argv[i] + n + 1;
    return def;
}
static double arg(int argc, char** argv, const char* key, double def) { const char* s = sarg(argc, argv, key, nullptr); return s ? atof(s) : def; }

struct Mol { double x, vx, vy, vz; unsigned char orig, nc; };   // orig: 0 from the source, 1 former stream molecule

int main(int argc, char** argv) {
    const double PI = 3.14159265358979323846, GAM = 5.0 / 3.0;
    double M = arg(argc, argv, "M", 8), omega = arg(argc, argv, "omega", 0.5), L = arg(argc, argv, "L", 40), dxc = arg(argc, argv, "dx", 0.25);
    double bias = arg(argc, argv, "bias", 1), cut = arg(argc, argv, "cut", 4), T1 = arg(argc, argv, "T1", 1);
    double alpha = arg(argc, argv, "alpha", 1);       // VSS exponent; 1 = isotropic scattering
    long np = (long)arg(argc, argv, "np", 1e6); int seed = (int)arg(argc, argv, "seed", 1);
    std::string src = sarg(argc, argv, "source", "maxwell"), out = sarg(argc, argv, "out", "cascade.txt");
    seed_rng(777 + 1000003ULL * seed);

    double r = (GAM + 1) * M * M / ((GAM - 1) * M * M + 2), p2 = (2 * GAM * M * M - (GAM - 1)) / (GAM + 1);
    double u1 = M * std::sqrt(GAM), u2 = u1 / r, T2 = p2 / r, s1 = std::sqrt(T1);
    double sref = 15 * std::sqrt(PI) / (2 * (5 - 2 * omega) * (7 - 2 * omega));
    double A = sref * std::pow(4.0, omega - 0.5) / std::tgamma(2.5 - omega), ex = 2 - 2 * omega;
    A *= (alpha + 1) * (alpha + 2) / (6 * alpha);
    double wkill = u1 - cut * s1;

    // source table
    std::vector<double> tv, tc, tT; double Jb = 0, dvt = 0;
    bool table = src != "maxwell";
    if (table) {
        FILE* f = fopen(src.c_str(), "r"); if (!f) { fprintf(stderr, "cannot open %s\n", src.c_str()); return 1; }
        double a, b, c; while (fscanf(f, "%lf %lf %lf", &a, &b, &c) == 3) { tv.push_back(a); Jb += b; tc.push_back(Jb); tT.push_back(c); }
        fclose(f); dvt = std::fabs(tv[1] - tv[0]);
    } else {
        Jb = r * (std::sqrt(T2 / (2 * PI)) * std::exp(-u2 * u2 / (2 * T2)) - 0.5 * u2 * std::erfc(u2 / std::sqrt(2 * T2)));
    }
    double Ts = bias * T2, sTs = std::sqrt(Ts), sT2 = std::sqrt(T2);

    int nc = (int)std::ceil(L / dxc);
    const int NT = 4;                                   // all, primaries, scattered source, recoil
    std::vector<double> tau((size_t)NT * nc, 0.0), tauv(nc, 0.0);
    double sw = 0, sw2 = 0; long ncol = 0, nmol = 0, nover = 0;
    std::vector<Mol> st; st.reserve(1 << 16);

    for (long ip = 0; ip < np; ip++) {
        Mol m; double w = 1;
        if (table) {
            double u = urand() * Jb; size_t lo = 0, hi = tc.size() - 1;
            while (lo < hi) { size_t mid = (lo + hi) / 2; if (tc[mid] < u) lo = mid + 1; else hi = mid; }
            m.vx = tv[lo] + (urand() - 0.5) * dvt; if (m.vx > -1e-6) m.vx = -1e-6;
            double sp = std::sqrt(tT[lo]); m.vy = sp * nrand(); m.vz = sp * nrand();
        } else {
            double s;
            do { s = sTs * std::sqrt(-2 * std::log(urand())); } while (urand() > std::exp(-s * u2 / Ts));
            m.vx = -s; m.vy = sT2 * nrand(); m.vz = sT2 * nrand();
            if (bias != 1) w = std::exp(-(s + u2) * (s + u2) * (0.5 / T2 - 0.5 / Ts));
        }
        m.x = 0; m.orig = 0; m.nc = 0;
        sw += w; sw2 += w * w;
        st.clear(); st.push_back(m);
        while (!st.empty()) {
            Mol q = st.back(); st.pop_back(); nmol++;
            if (st.size() > 4000000) { nover++; st.clear(); break; }
            for (;;) {
                double wx = q.vx - u1, wr = std::sqrt(wx * wx + q.vy * q.vy + q.vz * q.vz);
                if (wr < wkill) break;                                   // can no longer be turned back
                double gmax = wr + 6 * s1, numax = A * (ex == 0.0 ? 1.0 : (ex == 1.0 ? gmax : std::pow(gmax, ex)));
                double t = -std::log(urand()) / numax, xn = q.x + q.vx * t;
                if (q.vx < 0) {                                          // track-length tally of upstream-moving molecules
                    double xa = xn < -L ? -L : xn, xb = q.x;             // xa < xb <= 0
                    int ia = (int)((-xb) / dxc), ib = (int)((-xa) / dxc); if (ib >= nc) ib = nc - 1;
                    int cat = q.nc == 0 ? 1 : (q.orig ? 3 : 2);
                    for (int i = ia; i <= ib; i++) {
                        double lo = -(i + 1) * dxc, hi = -i * dxc, len = (hi < xb ? hi : xb) - (lo > xa ? lo : xa);
                        if (len > 0) { double c = w * len / (-q.vx); tau[i] += c; tau[(size_t)cat * nc + i] += c; tauv[i] += c * (-q.vx); }
                    }
                }
                q.x = xn;
                if (q.x >= 0 || q.x <= -L) break;
                double cx = u1 + s1 * nrand(), cy = s1 * nrand(), cz = s1 * nrand();
                double gx = q.vx - cx, gy = q.vy - cy, gz = q.vz - cz, g = std::sqrt(gx * gx + gy * gy + gz * gz);
                double pr = ex == 0.0 ? 1.0 : (ex == 1.0 ? g / gmax : std::pow(g / gmax, ex));
                if (urand() >= pr) continue;
                double hx, hy, hz;
                if (alpha == 1.0) {
                    double ct = 2 * urand() - 1, sn = std::sqrt(1 - ct * ct), ph = 2 * PI * urand();
                    hx = 0.5 * g * ct; hy = 0.5 * g * sn * std::cos(ph); hz = 0.5 * g * sn * std::sin(ph);
                } else {
                    double cc = 2 * std::pow(urand(), 1.0 / alpha) - 1, sc = std::sqrt(1 - cc * cc), ep = 2 * PI * urand();
                    double gp = std::sqrt(gy * gy + gz * gz), rx, ry, rz;
                    if (gp > 1e-12 * g) {
                        rx = cc * gx + sc * std::sin(ep) * gp;
                        ry = cc * gy + sc * (g * gz * std::cos(ep) - gx * gy * std::sin(ep)) / gp;
                        rz = cc * gz - sc * (g * gy * std::cos(ep) + gx * gz * std::sin(ep)) / gp;
                    } else { rx = cc * gx; ry = sc * std::cos(ep) * g; rz = sc * std::sin(ep) * g; }
                    hx = 0.5 * rx; hy = 0.5 * ry; hz = 0.5 * rz;
                }
                double mx = 0.5 * (q.vx + cx), my = 0.5 * (q.vy + cy), mz = 0.5 * (q.vz + cz);
                ncol++;
                Mol b; b.x = q.x; b.vx = mx - hx; b.vy = my - hy; b.vz = mz - hz; b.orig = 1; b.nc = 1;
                st.push_back(b);
                q.vx = mx + hx; q.vy = my + hy; q.vz = mz + hz; if (q.nc < 250) q.nc++;
            }
        }
    }
    FILE* f = fopen(out.c_str(), "w");
    fprintf(f, "# M %.10g omega %.10g L %.10g dx %.10g bias %.10g cut %.10g np %ld seed %d source %s\n", M, omega, L, dxc, bias, cut, np, seed, src.c_str());
    fprintf(f, "# u1 %.10g u2 %.10g T2 %.10g A %.10g Jb %.12g sum_w %.12g sum_w2 %.12g collisions %ld molecules %ld overflow %ld\n", u1, u2, T2, A, Jb, sw, sw2, ncol, nmol, nover);
    fprintf(f, "# x  beta_all  beta_primary  beta_scattered_source  beta_recoil  mean_backward_speed   (beta = density of xi_x<0 molecules / n1)\n");
    for (int i = 0; i < nc; i++) {
        double k = Jb / (sw * dxc);
        fprintf(f, "%.6g %.8e %.8e %.8e %.8e %.6g\n", -(i + 0.5) * dxc, tau[i] * k, tau[nc + i] * k, tau[2 * (size_t)nc + i] * k, tau[3 * (size_t)nc + i] * k, tau[i] > 0 ? tauv[i] / tau[i] : 0.0);
    }
    fclose(f);
    printf("done: %ld primaries, %ld collisions, %ld molecules followed, overflow %ld\n", np, ncol, nmol, nover);
    return 0;
}
