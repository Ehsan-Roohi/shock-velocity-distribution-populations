# Rotational populations of a strong shock wave resolved by collision count

Codes and data for the Letter

> E. Roohi, *Rotational populations of a strong shock wave resolved by collision count* (2026), and its Supplemental
> Material.

The Letter computes the nitrogen shock waves measured by Robben and Talbot (Phys. Fluids 9, 653, 1966) at Mach 7.0 and
12.9 by direct simulation Monte Carlo (DSMC). Rotational energy is exchanged by the Larsen-Borgnakke rule, by a soft
exchange rule, or through a collision kernel built from classical trajectories of two rigid rotors with two
Lennard-Jones sites each. Every simulated molecule carries the number of its collisions in the shock. This folder holds
the codes, the measured free-stream populations used as inflow, and the tables of the Supplemental Material as CSV; the
reduced and raw simulation data are in the release
[`nitrogen-rotational-shocks-data`](https://github.com/Ehsan-Roohi/shock-velocity-distribution-populations/releases/tag/nitrogen-rotational-shocks-data)
of this repository.

## Units

DSMC code units: upstream number density, temperature and molecular mass are 1, `u1 = M sqrt(7/5)`, lengths are in the
upstream mean free path `l1` of the variable-hard-sphere gas with viscosity exponent 0.74, rotational energies in
`kT1`. The analysis scripts give rotational energies in `kT2`, with `T2 = 296` K (Mach 7.0) and 305 K (Mach 12.9),
`T1 = 28.3` and 9.16 K. `n* = (n - n1)/(n2 - n1)` is the normalised density.

## Contents

| folder | content |
|---|---|
| `code/dsmc/` | `dsmc_shock_vdf19.cpp` (main runs), `dsmc_shock_vdf20.cpp` (control runs; identical to version 19 for `freeze=0` and `freeze=1`), `dsmc_shock_vdf17.cpp` (reference runs and numerical tests), `dsmc_shock_vdf12.cpp` (runs at Mach 8 and 12 of the precursor table), `make_vdf18/19/20.py` (the changes from version 17 to 20, documented as patches), `cascade_linear_rot.cpp` (linear recoil cascade of the precursor) |
| `code/trajectories/` | classical trajectories (`ctc_kernel.py`: ensembles at the conditions of the shock front; `ctc_kcell.py`: trajectories for the kernel), the kernel builder `kcell.py` (writes `kernel.bin`), its tests (`kcell_test.py`, `kcell_gain.py`) and the analyses of the ensembles (`ctc_analyse.py`, `ctc_condition.py`, `ctc_excite.py`); `n2_hellmann.py` is the ab initio N2-N2 potential of Hellmann (Mol. Phys. 111, 387, 2013; version B, parameters from its supplementary material), used with `pot=1 bond=1.1014` in place of the two-centre Lennard-Jones model |
| `code/reduction/` | reduction of the DSMC output: histograms by number of collisions (`n2_erh_reduce.py`), first collisions (`n2_first_reduce.py`), profiles (`n2_exchange_compare.py`), precursor (`n2_letter_precursor.py`), state of the stream and conserved fluxes (`n2_stream_state.py`); `n2_check.py` reads one run |
| `code/analysis/` | the measurements of Robben and Talbot transcribed from their Tables I and II (`robben_talbot_1966.py`), the measured free stream (`rt_inflow.py`), the level populations predicted from DSMC (`rt_predict.py`, `rt_predict_rin.py`), and further analyses used while preparing the Letter |
| `code/paper/` | the scripts that make the figures (`make_schematic.py`: the overview figure), the tables and every number quoted in the text from the reduced data |
| `code/rykov/` | discrete-velocity solver of the Rykov model in the form of Wu *et al.* (J. Fluid Mech. 763, 24, 2015) used in the Supplemental Material |
| `data/inflow/` | rotational populations measured at the first station (`rin_M7.txt`, `rin_M12.9.txt`), with which the simulated stream enters |
| `data/tables/` | the 14 tables of the Supplemental Material as CSV, with their captions in `README.md` |

## Data in the release

| asset | content |
|---|---|
| `reduced_data.tar.gz` | the reduced files read by `code/paper/` and `code/analysis/rt_predict*.py`. Unpack into a folder and pass it as the data folder. `*_rin.*`: main runs (measured free stream, collisions between two unscattered molecules exchange no rotational energy, version 19); `*_v20.*`: control runs (no rotational exchange when both velocities lie within `4 (kT1/m)^1/2` of the stream velocity); `erh_v17.npz`, `first_v17.npz`, `*_ktab*`: reference runs with the Boltzmann stream; `*_tests*`: numerical tests; `ctc_cluster/`: analyses of the trajectory ensembles; `kcell/`: kernel tests |
| `dsmc_main_v19.tar.gz` | raw output of the main runs (`v19/k`, `lb`, `p02`, `soft`: trajectory kernel, LB rule with `p(Ec)`, LB rule with `p = 0.2`, soft exchange; seeds 11-14) and of the threshold tests (`v19/k37`, `v19/k14`) |
| `dsmc_control_v20.tar.gz` | raw output of the control runs |
| `dsmc_reference.tar.gz` | reference runs with the Boltzmann stream (`v17k`, `v17lb`, `v17chi`), numerical tests (`v17ref/grid`, `half`, `egrid`), LB runs with `p = 0.2` at Mach 7, 8, 12 and 12.9 (`lb12`), linear cascade (`casc_n2`) |
| `trajectories.tar.gz` | trajectory ensembles (`ctc/`) and the trajectories of the kernel with the kernels themselves (`kcell/kernel.bin`, `kernel_half.bin`, `kernel_egrid.bin`) |
| `rykov_model.tar.gz` | solutions of the Rykov model (profiles, convergence histories) |

Each DSMC run writes `<out>.txt` (all parameters, one per line) and binary arrays (`.bin`, `.cls.bin`, `.erh.bin`,
`.erl.bin`, `.erv.bin`, `.first.bin`, `.fly.bin`, `.h2.bin`, `.rot.bin`) whose layout is described in the header
comment of `dsmc_shock_vdf19.cpp`; `code/reduction/n2_check.py` reads them.

## Reproducing

Build: `g++ -O3 -march=x86-64-v2 -o dsmc_shock_vdf19 dsmc_shock_vdf19.cpp` (the same for the other versions).

Main runs, trajectory kernel, Mach 7.0, seed 11 (seeds 11-14 were run):

```
./dsmc_shock_vdf19 M=7 omega=0.74 zrot=5 ktab=kcell/kernel.bin kt1=28.3 rin=rin_M7.txt freeze=1 rhrin=1 seed=11 \
  dt=0.0035 xmin=-60 xmax=30 dx=0.1 ppc=200 dv=0.2 steps=215000 sample_start=15000 sample_every=2 \
  xref1=-6.65 xref2=0 xref3=-3.325 out=runs_n2/v19/k/M7_s11
```

Mach 12.9: `M=12.9 kt1=9.16 rin=rin_M12.9.txt dt=0.002 xmin=-90 xmax=40 xref1=-8.55 xref3=-4.275`. Other rules, in
place of `ktab=... kt1=...`: LB rule with `p(Ec)`: `zinf=10 tstar=3.233` (Mach 7.0) or `tstar=9.99` (Mach 12.9); LB
rule with `p = 0.2`: nothing; soft exchange: `pphi=0.5`. Deflection threshold that counts a collision in the kernel
runs: `kchi=0.9` (26 degrees, default), `0.8` (37 degrees), `0.97` (14 degrees). Control runs: `dsmc_shock_vdf20` with
`freeze=2` (window `wfrz=4`). Reference runs: `dsmc_shock_vdf17` without `rin`, `freeze` and `rhrin`.

Trajectories: `python ctc_kernel.py ctc/M12.9tm_s300.npz N=60000 seed=300 eps=47.2 sig=3.17 bond=1.094` (seeds
300-315; Mach 7.0 with `T1=28.3 T2=296 u1=759 Th=414 Trh=379`, seeds 316-331). Kernel trajectories:
`python ctc_kcell.py kcell/c_<i>.npz <7 i> 7 N=2500 seed=<9000 + i> eps=47.2 sig=3.17 bond=1.094` for `i = 0..206`,
then `python kcell.py pack kcell/kernel.bin kcell/c_*.npz`.

Reduction (in the folder of the runs), for example
`python n2_erh_reduce.py erh_rin.npz M7k='runs_n2/v19/k/M7_s*.txt' M12.9k='runs_n2/v19/k/M12.9_s*.txt'`; the labels
used by the paper scripts are `k` (kernel), `zi10` (LB `p(Ec)`; `lb` in the first-collision files), `c` (LB
`p = 0.2`), `pu5` (soft exchange), `chi0.8`, `chi0.97` (thresholds) and `s11`-`s14` (single runs).

Paper: unpack `reduced_data.tar.gz` into a folder `<data>` and copy `code/analysis/*.py` into it (the scripts import
`robben_talbot_1966.py` from the data folder). In `<data>`: `python rt_predict_rin.py`; then, from `code/paper/`,
`python make_sm_tables.py <data> <out> rin`, `python make_figures.py <data> figs ctc=M12.9tm ctc7=M7tm sm=1 src=rin`,
`python make_mixture_table.py <data> first_rin.npz M7k M12.9k <out>` and `python make_text_numbers.py <data> rin`
(`python make_text_numbers.py <data> rin v20` for the control runs).

## Licence

MIT; see `LICENSE` at the top of the repository.
