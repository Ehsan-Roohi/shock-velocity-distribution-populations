// DSMC of a stationary normal shock in a monatomic VHS gas, with sampling of the reduced
// velocity distributions g(x, vx) = int f dvy dvz and h(x, vx) = int (vy^2+vz^2) f dvy dvz.
//
// Units: m = k = 1, upstream n1 = T1 = 1, u1 = M sqrt(5/3). The reference cross-section is chosen
// so that the first-approximation VHS viscosity at T1 is mu1 = 1; lengths are then in units of
// l1 = sqrt(R T1) mu1 / p1, the same unit as the kinetic-model solutions.
//   sigma_ref = 15 sqrt(pi) / (2 (5-2w)(7-2w)),   sigma c_r = sigma_ref 4^(w-1/2) / Gamma(5/2-w) c_r^(2-2w)
// (Bird 1994, Eqs. 4.62-4.63). w = 0.5 is the hard sphere.
//
// Boundaries: equilibrium reservoirs (ghost cells refilled every step) at the Rankine-Hugoniot
// states. The shock position is estimated from the total particle number and the whole particle
// field is shifted by one cell whenever the shock has moved by more than half a cell; the emptied
// end cell is refilled from the corresponding reservoir.
//
// Version 2: every molecule also carries its collision history since it left the upstream beam, and the
// streamwise distribution and low moments are sampled separately for five classes:
//   0  unscattered beam: born upstream, has collided only with other unscattered molecules
//   1  one collision since leaving the beam          2  two collisions (last one not with a beam molecule)
//   3  three or more collisions (last one not with a beam molecule); molecules born downstream start here
//   4  scattered molecules whose most recent collision was with an unscattered beam molecule
// and for four generations (classes 5-8): generation 0 is the unscattered beam; after any other collision both
// partners take the generation min(g_i, g_j) + 1, capped at 3, so that the generation is the number of collisions
// that separates a molecule from the beam along its most recent lineage. Molecules born downstream start at 3.
// The tags use no random numbers, so trajectories are identical to version 1 for the same seed.
// Version 5: free-flight flag. Bit j of the flag is set when a molecule crosses the plane x = xref<j+1> moving
// upstream, and the whole flag is cleared at every collision of the molecule, of any kind. A flagged molecule has
// therefore flown freely from that plane to where it is. Extra output: <out>.fly.bin = double [4][nc]: number of
// samples with vx < 0 in each cell: all, and those with bit 0, 1, 2 set. Unused planes: leave xref at its default.
// Extra output: <out>.cls.bin = float32 counts [9][nc][nb], then double sums [9][nc][4] of 1, v, v^2, w^2.
// Version 7: rotational energy. With zrot > 0 the gas is diatomic (two rotational degrees of freedom, gamma = 7/5):
// every molecule carries a rotational energy er, the reservoirs are in equilibrium at the Rankine-Hugoniot states of
// gamma = 7/5, and in every collision each partner exchanges rotational and translational energy with probability
// 1/zrot by the Larsen-Borgnakke rule (Bird 1994, Sec. 5.5): the new er is Ec (1 - U^(1/(5/2-w))) with Ec the
// relative translational energy plus the er of that molecule. With zrot = 0 the code is version 5 (monatomic).
// Extra output: <out>.rot.bin = double [19][nc]: sum of er over all samples (row 0); for the generations 0..3,
// number of samples with vx < 0 (rows 1-4), their er (5-8), er of all samples of the generation (9-12), and the
// translational energy (v^2 + w^2)/2 of those with vx < 0 (15-18); number and er of the samples with vx < 0 that
// carry bit 0 of the free-flight flag (13, 14).
//
// Version 8: tag of the rotational energy. A molecule that enters from upstream keeps the tag until it exchanges
// rotational energy in a collision in which at least one partner has left the unscattered stream; molecules that
// enter from downstream never carry it. Rows 19-24 of
// <out>.rot.bin, for the samples with vx < 0 without the tag (first row of each pair) and with it (second row):
// number (19, 20), er (21, 22), translational energy (v^2 + w^2)/2 (23, 24).
//
// Version 10: histogram of the rotational energy. <out>.erh.bin = double [6][nc][160]: number of samples with
// er in bins of width 0.05 T2 (T2 = downstream temperature): all molecules (0), those with vx < 0 (1), and the
// collision generations 0, 1, 2, 3 (2-5).
//
// Version 11: <out>.erv.bin = double [2][nc][nb]: sum of the rotational energy of the samples in every velocity bin of
// every cell, and number of samples that carry the rotational tag (divide by the counts of <out>.bin).
//
// Version 12: pphi > 0 selects the exchange model of Pullin (1978) (every collision exchanges an active part pphi of the
// rotational and translational energies) instead of the Larsen-Borgnakke rule. Rows 25-29 of <out>.rot.bin: number of
// samples with vx < 0 in the collision classes 0-4; rows 30-34: their rotational energy. <out>.erh.bin and the new
// <out>.erl.bin (bins of 0.005 T2) = double [9][nc][160]: all, vx < 0, generations 0-3, and vx < 0 in classes 1, 4, 3.
//
// Build: g++ -O3 -march=x86-64-v2 -o dsmc_shock_vdf7 dsmc_shock_vdf7.cpp
// Run:   ./dsmc_shock_vdf M=5 omega=0.5 seed=1 out=run_M5_w05_s1 [xmin=-40 xmax=25 dx=0.1 dt=0.004
//                          ppc=200 steps=110000 sample_start=10000 sample_every=2 dv=0.2]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct Particles {
    std::vector<double> x, vx, vy, vz, er;
    std::vector<unsigned char> k, r, g, f, o;   // collisions since leaving the beam (capped at 3); 1 if last partner was a beam molecule
    size_t n() const { return x.size(); }
    void push(double a, double b, double c, double d, unsigned char kk, double e) { x.push_back(a); vx.push_back(b); vy.push_back(c); vz.push_back(d); er.push_back(e); k.push_back(kk); r.push_back(0); g.push_back(kk); f.push_back(0); o.push_back(kk == 0 ? 1 : 0); }
    void resize(size_t m) { x.resize(m); vx.resize(m); vy.resize(m); vz.resize(m); er.resize(m); k.resize(m); r.resize(m); g.resize(m); f.resize(m); o.resize(m); }
};

static std::mt19937_64 rng;
static inline double urand() { return (rng() >> 11) * (1.0 / 9007199254740992.0); }
static inline double rgamma(double a) { std::gamma_distribution<double> g(a, 1.0); return g(rng); }
static inline double rbeta(double a, double b) { double x = rgamma(a), y = rgamma(b); return x / (x + y); }
static inline double nrand() {
    static bool has = false; static double spare;
    if (has) { has = false; return spare; }
    double u, v, s;
    do { u = 2 * urand() - 1; v = 2 * urand() - 1; s = u * u + v * v; } while (s >= 1 || s == 0);
    double m = std::sqrt(-2 * std::log(s) / s);
    spare = v * m; has = true; return u * m;
}

static bool ROT = false;      // diatomic gas: rotational energy with two degrees of freedom
static void add_maxwell(Particles& p, double xa, double xb, double n, double u, double T, double FN, unsigned char tag) {
    double expect = n * (xb - xa) / FN;
    long k = (long)expect;
    if (urand() < expect - k) k++;
    double s = std::sqrt(T);
    for (long i = 0; i < k; i++) { double px = xa + (xb - xa) * urand(), a = u + s * nrand(), b = s * nrand(), c = s * nrand(); p.push(px, a, b, c, tag, ROT ? -T * std::log(1.0 - urand()) : 0.0); }
}

static void remove_outside(Particles& p, double xmin, double xmax) {
    size_t j = 0;
    for (size_t i = 0; i < p.n(); i++) {
        if (p.x[i] >= xmin && p.x[i] < xmax) {
            if (i != j) { p.x[j] = p.x[i]; p.vx[j] = p.vx[i]; p.vy[j] = p.vy[i]; p.vz[j] = p.vz[i]; p.er[j] = p.er[i]; p.k[j] = p.k[i]; p.r[j] = p.r[i]; p.g[j] = p.g[i]; p.f[j] = p.f[i]; p.o[j] = p.o[i]; }
            j++;
        }
    }
    p.resize(j);
}

static double arg(int argc, char** argv, const char* key, double def) {
    size_t L = strlen(key);
    for (int i = 1; i < argc; i++) if (!strncmp(argv[i], key, L) && argv[i][L] == '=') return atof(argv[i] + L + 1);
    return def;
}
static std::string sarg(int argc, char** argv, const char* key, const char* def) {
    size_t L = strlen(key);
    for (int i = 1; i < argc; i++) if (!strncmp(argv[i], key, L) && argv[i][L] == '=') return std::string(argv[i] + L + 1);
    return def;
}

int main(int argc, char** argv) {
    const double PI = 3.14159265358979323846;
    double zrot = arg(argc, argv, "zrot", 0); ROT = zrot > 0;
    const double pphi = arg(argc, argv, "pphi", 0);        // > 0: Pullin exchange in every collision, active fraction pphi
    const double gam = ROT ? 7.0 / 5.0 : 5.0 / 3.0;
    double M = arg(argc, argv, "M", 5), omega = arg(argc, argv, "omega", 0.5);
    double alpha = arg(argc, argv, "alpha", 1);       // VSS exponent; 1 = isotropic scattering (VHS)
    double xmin = arg(argc, argv, "xmin", -40), xmax = arg(argc, argv, "xmax", 25), dx = arg(argc, argv, "dx", 0.1);
    double dt = arg(argc, argv, "dt", 0.004), ppc = arg(argc, argv, "ppc", 200), dv = arg(argc, argv, "dv", 0.2);
    long steps = (long)arg(argc, argv, "steps", 110000), s0 = (long)arg(argc, argv, "sample_start", 10000);
    int se = (int)arg(argc, argv, "sample_every", 2), seed = (int)arg(argc, argv, "seed", 1);
    const int NB = 4, NM = 9;     // moments: 1, v, v^2, v^3, v^4, w2, v w2, v^2 w2, w2^2   (w2 = vy^2 + vz^2)
    std::string out = sarg(argc, argv, "out", "dsmc_out");
    const int NP = 3; double xr[NP] = {arg(argc, argv, "xref1", -1e30), arg(argc, argv, "xref2", -1e30), arg(argc, argv, "xref3", -1e30)};
    rng.seed(1234567ULL * (unsigned long long)seed + 987654321ULL);

    double n1 = 1, T1 = 1, u1 = M * std::sqrt(gam);
    double n2 = (gam + 1) * M * M / ((gam - 1) * M * M + 2), p2 = (2 * gam * M * M - (gam - 1)) / (gam + 1);
    double T2 = p2 / n2, u2 = u1 / n2;
    int nc = (int)std::lround((xmax - xmin) / dx);
    double FN = n1 * dx / ppc;
    double N0 = (n1 * (0 - xmin) + n2 * (xmax - 0)) / FN;
    double W = dx * std::ceil((u1 + 8 * std::sqrt(T2)) * dt / dx);
    double sref = 15 * std::sqrt(PI) / (2 * (5 - 2 * omega) * (7 - 2 * omega));
    double A = sref * std::pow(4.0, omega - 0.5) / std::tgamma(2.5 - omega), ex = 2 - 2 * omega;
    A *= (alpha + 1) * (alpha + 2) / (6 * alpha);          // same viscosity for the VSS law
    double vlo = u2 - 8 * std::sqrt(T2), vhi = u1 + 8 * std::sqrt(T2);
    int nb = (int)std::ceil((vhi - vlo) / dv);

    Particles p;
    p.f.reserve((size_t)(1.3 * N0)); p.er.reserve((size_t)(1.3 * N0));
    p.o.reserve((size_t)(1.3 * N0)); p.k.reserve((size_t)(1.3 * N0)); p.r.reserve((size_t)(1.3 * N0)); p.g.reserve((size_t)(1.3 * N0));
    p.x.reserve((size_t)(1.3 * N0)); p.vx.reserve((size_t)(1.3 * N0)); p.vy.reserve((size_t)(1.3 * N0)); p.vz.reserve((size_t)(1.3 * N0));
    add_maxwell(p, xmin, 0, n1, u1, T1, FN, 0);
    add_maxwell(p, 0, xmax, n2, u2, T2, FN, 3);

    std::vector<double> cnt((size_t)nc * nb, 0.0), hs((size_t)nc * nb, 0.0), hs2((size_t)nc * nb, 0.0), mom((size_t)NB * nc * NM, 0.0), nsamp(NB, 0.0);
    const int NCLS = 9, NCM = 4;
    std::vector<float> ccnt((size_t)NCLS * nc * nb, 0.0f);
    std::vector<double> cmom((size_t)NCLS * nc * NCM, 0.0);
    std::vector<double> fly((size_t)(NP + 1) * nc, 0.0);
    const int NR = 35; std::vector<double> rot((size_t)NR * nc, 0.0);
    const int NE = 160, NH = 9; const double de = 0.05 * T2, del = 0.005 * T2;
    std::vector<double> erh((size_t)NH * nc * NE, 0.0), erl((size_t)NH * nc * NE, 0.0);
    std::vector<double> erv((size_t)2 * nc * nb, 0.0);
    const double lbx = 1.0 / (2.5 - omega), prot = ROT ? 1.0 / zrot : 0.0, peta = 2.5 - omega;
    std::vector<double> crmax(nc, A * std::pow(6 * std::sqrt(T2), ex)), rem(nc, 0.0);
    std::vector<int> cell, start(nc + 1), order, fill(nc);
    long nshift = 0, ncoll = 0, nshift_abs = 0;

    auto write = [&](long step) {
        FILE* f = fopen((out + ".txt").c_str(), "w");
        fprintf(f, "M %.10g\nomega %.10g\nxmin %.10g\nxmax %.10g\ndx %.10g\ndt %.10g\nppc %.10g\ndv %.10g\nvlo %.10g\nnc %d\nnb %d\nNB %d\nNM %d\n"
                   "FN %.12g\nsteps_done %ld\nsample_start %ld\nsample_every %d\nseed %d\nnshift_net %ld\nnshift_abs %ld\nncoll %ld\nnpart %zu\nNCLS %d\nNCM %d\nzrot %.10g\ngamma %.10g\nNR %d\n",
                M, omega, xmin, xmax, dx, dt, ppc, dv, vlo, nc, nb, NB, NM, FN, step, s0, se, seed, nshift, nshift_abs, ncoll, p.n(), NCLS, NCM, zrot, gam, NR);
        fclose(f);
        f = fopen((out + ".bin").c_str(), "wb");
        fwrite(cnt.data(), 8, cnt.size(), f); fwrite(hs.data(), 8, hs.size(), f);
        fwrite(mom.data(), 8, mom.size(), f); fwrite(nsamp.data(), 8, nsamp.size(), f);
        fclose(f);
        f = fopen((out + ".cls.bin").c_str(), "wb");
        fwrite(ccnt.data(), 4, ccnt.size(), f); fwrite(cmom.data(), 8, cmom.size(), f);
        fclose(f);
        f = fopen((out + ".h2.bin").c_str(), "wb");       // reduced distribution of the transverse fourth moment: int (vy^2+vz^2)^2 f dvy dvz
        fwrite(hs2.data(), 8, hs2.size(), f); fclose(f);
        f = fopen((out + ".fly.bin").c_str(), "wb");
        fwrite(fly.data(), 8, fly.size(), f); fclose(f);
        f = fopen((out + ".rot.bin").c_str(), "wb");
        fwrite(rot.data(), 8, rot.size(), f); fclose(f);
        f = fopen((out + ".erh.bin").c_str(), "wb");
        fwrite(erh.data(), 8, erh.size(), f); fclose(f);
        f = fopen((out + ".erl.bin").c_str(), "wb");
        fwrite(erl.data(), 8, erl.size(), f); fclose(f);
        f = fopen((out + ".erv.bin").c_str(), "wb");
        fwrite(erv.data(), 8, erv.size(), f); fclose(f);
    };

    for (long step = 0; step < steps; ++step) {
        add_maxwell(p, xmin - W, xmin, n1, u1, T1, FN, 0);
        add_maxwell(p, xmax, xmax + W, n2, u2, T2, FN, 3);
        size_t N = p.n();
        for (size_t i = 0; i < N; i++) {
            double xo = p.x[i], xn = xo + p.vx[i] * dt;
            if (p.vx[i] < 0) for (int q = 0; q < NP; q++) if (xo >= xr[q] && xn < xr[q]) p.f[i] |= (unsigned char)(1 << q);
            p.x[i] = xn;
        }
        remove_outside(p, xmin, xmax);

        double xs = -((double)p.n() - N0) * FN / (n2 - n1);
        if (xs > 0.5 * dx) {
            for (size_t i = 0; i < p.n(); i++) p.x[i] -= dx;
            remove_outside(p, xmin, xmax);
            add_maxwell(p, xmax - dx, xmax, n2, u2, T2, FN, 3);
            nshift--; nshift_abs++;
        } else if (xs < -0.5 * dx) {
            for (size_t i = 0; i < p.n(); i++) p.x[i] += dx;
            remove_outside(p, xmin, xmax);
            add_maxwell(p, xmin, xmin + dx, n1, u1, T1, FN, 0);
            nshift++; nshift_abs++;
        }

        N = p.n();
        cell.resize(N); order.resize(N);
        std::fill(start.begin(), start.end(), 0);
        for (size_t i = 0; i < N; i++) {
            int c = (int)((p.x[i] - xmin) / dx);
            if (c < 0) c = 0; if (c >= nc) c = nc - 1;
            cell[i] = c; start[c + 1]++;
        }
        for (int c = 0; c < nc; c++) start[c + 1] += start[c];
        for (int c = 0; c < nc; c++) fill[c] = start[c];
        for (size_t i = 0; i < N; i++) order[fill[cell[i]]++] = (int)i;

        for (int c = 0; c < nc; c++) {
            int Nc = start[c + 1] - start[c];
            if (Nc < 2) continue;
            double cand = 0.5 * Nc * (Nc - 1.0) * FN * crmax[c] * dt / dx + rem[c];
            long k = (long)cand; rem[c] = cand - k;
            const int* o = &order[start[c]];
            for (long q = 0; q < k; q++) {
                int a = (int)(urand() * Nc), b = (int)(urand() * (Nc - 1));
                if (b >= a) b++;
                int i = o[a], j = o[b];
                double gx = p.vx[i] - p.vx[j], gy = p.vy[i] - p.vy[j], gz = p.vz[i] - p.vz[j];
                double cr = std::sqrt(gx * gx + gy * gy + gz * gz);
                double scr = (ex == 1.0) ? A * cr : ((ex == 0.0) ? A : A * std::pow(cr, ex));
                if (scr > crmax[c]) crmax[c] = scr;
                if (urand() * crmax[c] < scr) {
                    double hx, hy, hz;
                    if (ROT) {                                  // Larsen-Borgnakke exchange, each partner with probability 1/zrot
                        double et = 0.25 * cr * cr; bool ch = false;
                        if (pphi > 0) {                 // Pullin: an active part of each energy is pooled and redistributed
                            double e1 = p.er[i], e2 = p.er[j];
                            double s1 = rbeta(pphi, 1 - pphi), s2 = rbeta(pphi, 1 - pphi), s3 = rbeta(pphi * peta, (1 - pphi) * peta);
                            double pool = s1 * e1 + s2 * e2 + s3 * et, s5 = rbeta(2 * pphi, pphi * peta), s4 = rbeta(pphi, pphi);
                            p.er[i] = (1 - s1) * e1 + s4 * s5 * pool; p.er[j] = (1 - s2) * e2 + (1 - s4) * s5 * pool;
                            et = (1 - s3) * et + (1 - s5) * pool; ch = true;
                            if (p.g[i] | p.g[j]) { p.o[i] = 0; p.o[j] = 0; }
                        } else {
                        if (urand() < prot) { double ec = et + p.er[i]; p.er[i] = ec * (1 - std::pow(urand(), lbx)); et = ec - p.er[i]; ch = true; if (p.g[i] | p.g[j]) p.o[i] = 0; }
                        if (urand() < prot) { double ec = et + p.er[j]; p.er[j] = ec * (1 - std::pow(urand(), lbx)); et = ec - p.er[j]; ch = true; if (p.g[i] | p.g[j]) p.o[j] = 0; }
                        }
                        if (ch) { double cn = 2 * std::sqrt(et), s = cr > 0 ? cn / cr : 1.0; gx *= s; gy *= s; gz *= s; cr = cn; }
                    }
                    if (alpha == 1.0) {
                        double ct = 2 * urand() - 1, st = std::sqrt(1 - ct * ct), ph = 2 * PI * urand();
                        hx = 0.5 * cr * ct; hy = 0.5 * cr * st * std::cos(ph); hz = 0.5 * cr * st * std::sin(ph);
                    } else {                                    // deflection chi about the relative velocity: cos chi = 2 U^(1/alpha) - 1
                        double cc = 2 * std::pow(urand(), 1.0 / alpha) - 1, sc = std::sqrt(1 - cc * cc), ep = 2 * PI * urand();
                        double gp = std::sqrt(gy * gy + gz * gz), rx, ry, rz;
                        if (gp > 1e-12 * cr) {
                            rx = cc * gx + sc * std::sin(ep) * gp;
                            ry = cc * gy + sc * (cr * gz * std::cos(ep) - gx * gy * std::sin(ep)) / gp;
                            rz = cc * gz - sc * (cr * gy * std::cos(ep) + gx * gz * std::sin(ep)) / gp;
                        } else { rx = cc * gx; ry = sc * std::cos(ep) * cr; rz = sc * std::sin(ep) * cr; }
                        hx = 0.5 * rx; hy = 0.5 * ry; hz = 0.5 * rz;
                    }
                    double mx = 0.5 * (p.vx[i] + p.vx[j]), my = 0.5 * (p.vy[i] + p.vy[j]), mz = 0.5 * (p.vz[i] + p.vz[j]);
                    p.vx[i] = mx + hx; p.vy[i] = my + hy; p.vz[i] = mz + hz;
                    p.vx[j] = mx - hx; p.vy[j] = my - hy; p.vz[j] = mz - hz;
                    ncoll++; p.f[i] = 0; p.f[j] = 0;
                    unsigned char ki = p.k[i], kj = p.k[j];
                    if (ki | kj) {            // collisions between two unscattered beam molecules are not counted
                        p.k[i] = ki < 3 ? ki + 1 : 3; p.k[j] = kj < 3 ? kj + 1 : 3;
                        p.r[i] = (ki > 0 && kj == 0); p.r[j] = (kj > 0 && ki == 0);
                        unsigned char gm = std::min(p.g[i], p.g[j]); gm = gm < 2 ? gm + 1 : 3; p.g[i] = gm; p.g[j] = gm;
                    }
                }
            }
        }

        if (step >= s0 && (step - s0) % se == 0) {
            int blk = (int)(((step - s0) * NB) / (steps - s0));
            if (blk >= NB) blk = NB - 1;
            double* mb = &mom[(size_t)blk * nc * NM];
            for (size_t i = 0; i < N; i++) {
                int c = cell[i];
                double v = p.vx[i], w2 = p.vy[i] * p.vy[i] + p.vz[i] * p.vz[i];
                int b = (int)std::floor((v - vlo) / dv);
                int cls = p.r[i] ? 4 : p.k[i];
                if (b >= 0 && b < nb) { cnt[(size_t)c * nb + b] += 1.0; hs[(size_t)c * nb + b] += w2; hs2[(size_t)c * nb + b] += w2 * w2; ccnt[((size_t)cls * nc + c) * nb + b] += 1.0f;
                                       erv[(size_t)c * nb + b] += p.er[i]; if (p.o[i]) erv[((size_t)nc + c) * nb + b] += 1.0; }
                double* cm = &cmom[((size_t)cls * nc + c) * NCM]; cm[0] += 1; cm[1] += v; cm[2] += v * v; cm[3] += w2;
                int cg = 5 + p.g[i];
                if (b >= 0 && b < nb) ccnt[((size_t)cg * nc + c) * nb + b] += 1.0f;
                cm = &cmom[((size_t)cg * nc + c) * NCM]; cm[0] += 1; cm[1] += v; cm[2] += v * v; cm[3] += w2;
                { int gg = p.g[i]; double e = p.er[i]; rot[c] += e;
                  { int hs[4] = {0, v < 0 ? 1 : -1, 2 + gg, v < 0 ? (cls == 1 ? 6 : (cls == 4 ? 7 : (cls == 3 ? 8 : -1))) : -1};
                    int ie = (int)(e / de), il = (int)(e / del);
                    for (int q = 0; q < 4; q++) { if (hs[q] < 0) continue;
                        if (ie < NE) erh[((size_t)hs[q] * nc + c) * NE + ie] += 1.0;
                        if (il < NE) erl[((size_t)hs[q] * nc + c) * NE + il] += 1.0; } } rot[(size_t)(9 + gg) * nc + c] += e;
                  if (v < 0) { rot[(size_t)(25 + cls) * nc + c] += 1; rot[(size_t)(30 + cls) * nc + c] += e; rot[(size_t)(1 + gg) * nc + c] += 1; rot[(size_t)(5 + gg) * nc + c] += e; rot[(size_t)(15 + gg) * nc + c] += 0.5 * (v * v + w2);
                               if (p.f[i] & 1) { rot[(size_t)13 * nc + c] += 1; rot[(size_t)14 * nc + c] += e; }
                               { int oo = p.o[i]; rot[(size_t)(19 + oo) * nc + c] += 1; rot[(size_t)(21 + oo) * nc + c] += e; rot[(size_t)(23 + oo) * nc + c] += 0.5 * (v * v + w2); } } }
                if (v < 0) { fly[c] += 1; for (int q = 0; q < NP; q++) if ((p.f[i] >> q) & 1) fly[(size_t)(q + 1) * nc + c] += 1; }
                double* m = mb + (size_t)c * NM;
                m[0] += 1; m[1] += v; m[2] += v * v; m[3] += v * v * v; m[4] += v * v * v * v; m[5] += w2; m[6] += v * w2; m[7] += v * v * w2; m[8] += w2 * w2;
            }
            nsamp[blk] += 1;
        }
        if ((step + 1) % 10000 == 0) {
            printf("step %ld N %zu shifts net %ld abs %ld collisions %ld\n", step + 1, p.n(), nshift, nshift_abs, ncoll); fflush(stdout);
            if (step >= s0) write(step + 1);
        }
    }
    write(steps);
    return 0;
}
