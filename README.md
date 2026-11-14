# Population structure of the velocity distribution in normal shock waves

Codes and processed data for the paper

> E. Roohi, *Population structure of the velocity distribution in normal shock waves: Gaussian representations,
> Mach-number similarity, and prediction*, submitted to the Journal of Fluid Mechanics (2026).

The paper studies the molecular velocity distribution inside a steady, plane normal shock in a monatomic gas, using
discrete-velocity solutions of the BGK, Shakhov and ES-BGK models up to Mach 12 and direct simulation Monte Carlo
(DSMC) of hard-sphere and variable-hard-sphere gases up to Mach 25. This repository holds what is needed to reproduce
its results on

1. the three Gaussian populations (beam, intermediate, hot) of the reduced distributions and what they are, checked
   with a collision record carried by every DSMC molecule;
2. the similarity of the distribution in variables scaled with the Rankine-Hugoniot states, and the prediction of a
   shock from two others by three methods: P (three populations), D (direct combination of the scaled distributions),
   H (D with the beam treated separately);
3. the upstream precursor: the mass fraction of upstream-moving molecules ahead of the shock, its decay, the
   free-flight flag, and the linear cascade model.

The many-kernel (512-Gaussian) phase-space representation that the paper summarises in its Sec. 10 has its own
repository: <https://github.com/Ehsan-Roohi/KineticGaussian>.

## Units and symbols

`rho1 = T1 = R = 1`, `u1 = M sqrt(5/3)`; lengths in `l1 = sqrt(R T1) mu1 / p1`.
`n* = (rho - rho1)/(rho2 - rho1)`; `xi_hat = (xi_x - u2)/(u1 - u2)`; `beta` = mass fraction of molecules with
`xi_x < 0`; `omega` = viscosity exponent (0.5 hard sphere, 1 Maxwell-type); `N` = column density ahead of the shock
centre.

## Contents

### `codes/dsmc/`

| file | purpose |
|---|---|
| `dsmc_shock_vdf.cpp` | DSMC of the shock; samples the reduced distributions `g` and `h` in every cell |
| `dsmc_shock_vdf2.cpp` | adds the collision record (generation of each molecule) |
| `dsmc_shock_vdf3.cpp` | adds variable-soft-sphere scattering |
| `dsmc_shock_vdf4.cpp` | adds the transverse fourth moment and the mixed moment |
| `dsmc_shock_vdf5.cpp` | adds the free-flight flag relative to reference planes |
| `cascade_linear.cpp` | linear cascade model of the precursor (constant of the stopping rule: argument `cut`, default 4) |
| `dsmc_to_reduced.py`, `dsmc_classes_to_npz.py` | conversion of the DSMC output |
| `precursor_seeds.py`, `rate_seeds.py` | per-run fits of the precursor and its decay rates |
| `fly_analysis.py`, `domain_compare.py` | free-flight shares; test of the domain length |

Build with any C++11 compiler, for example `g++ -O3 -o dsmc_shock_vdf dsmc_shock_vdf.cpp`. A production run of the
paper is

```
./dsmc_shock_vdf M=8 omega=0.5 seed=1 dt=0.0035 xmin=-60 xmax=30 dx=0.1 ppc=200 dv=0.2 \
                 steps=215000 sample_start=15000 sample_every=2 out=runs/M8_w0.5_s1
```

with eight seeds per case, time steps of 0.002-0.01, and the domain `-90 <= x <= 40` at Mach 10 and 12. The Mach-25
runs use `dsmc_shock_vdf2`, `dt=0.001`, `steps=245000 sample_start=30000`, and the domains `[-60, 30]`, `[-160, 60]`
and `[-300, 100]` for `omega` = 0.5, 0.81 and 1.

### `codes/analysis/`

| file | purpose |
|---|---|
| `shocklib.py` | Rankine-Hugoniot states, moments, population fits |
| `run_population_fits.py`, `branch_fits.py`, `fit_stability.py`, `robustness.py` | fits with one to four populations and their stability |
| `bgk_reduced_shock.py`, `bgk_resolution.py` | reduced BGK solver (recomputed Mach 6, 8, 12) and its resolution tests |
| `integral_check.py`, `ohwada_compare.py`, `takata_compare.py` | verification of the reference solutions |
| `dsmc_analysis.py`, `dsmc_table.py`, `dsmc_resolution.py`, `dsmc_resolution_m12.py` | DSMC populations; cell, time-step and particle-number tests at Mach 8 and 12 |
| `similarity_lomo.py` | similarity test with one Mach number withheld |
| `predictor.py`, `predictor_models.py` | prediction with three populations (P), for BGK, DSMC and Shakhov solutions |
| `predictor_direct.py` | direct (D) and separate-beam (H) predictions, with positivity checks |
| `predictor_clean.py` | archived against recomputed BGK solutions |
| `transverse_check.py` | transverse fourth moment and mixed moment |
| `generations.py`, `precursor_cascade.py`, `cascade_sources.py` | collision record and cascade model of the precursor |
| `paper_figures.py`, `methods_figure.py`, `mechanism_figure.py`, `decomposition_figure.py`, `graphical_abstract.py` | figures |

Python 3 with `numpy`, `scipy` and `matplotlib`.

### `results/`

JSON summaries written by the codes above; the numbers of the paper are taken from them. Examples:
`predictor_summary.json`, `predictor_models_summary.json`, `predictor_direct.json`, `predictor_clean.json`
(predictions), `generation_shares_precursor.json`, `free_flight.json`, `cascade_table_source_composition.json`,
`rate_seeds_*.json`, `precursor_seeds*.json` (precursor), `bgk_resolution.json`, `dsmc_resolution_summary.json`,
`dsmc_resolution_m12.json` (resolution tests), `fit_stability.json`, `transverse_check.json` (population fits).

### `data/`

| file | content |
|---|---|
| `M8r_BGK_reduced_vdf.npz` | recomputed Mach-8 BGK shock: `x_mfp`, `vx`, `wx`, `G`, `H`, states, convergence history |
| `cascade/*.txt` | cascade model: `x`, `beta` (all, uncollided source, scattered source, recoil), mean speed. `casc_*`: constant 4; `cut3_*`, `cut5_*`: constants 3 and 5; `tab_*`: driven by the flux measured in DSMC; `vss_*`: variable-soft-sphere scattering |

The raw DSMC output and the three-dimensional discrete-velocity arrays are several gigabytes and are available from
the author on request.

## Licence

MIT; see `LICENSE`.
