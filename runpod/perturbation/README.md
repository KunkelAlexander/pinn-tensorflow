# ELBDM/Perturbation: density power spectrum mismatch and FD4 gravity (session 2, 2026-11-16)

Goal: reproduce the hybrid-vs-reference density power spectrum mismatch from the GAMER wiki ([ELBDM] Hybrid Scheme,
section Perturbation) and test whether `POT_STENCIL_ORDER 4` (fork KunkelAlexander/gamer, branch `fd4-poisson`) helps.

## Setup
- Pod `xvrtb9qs67yv2i` (`pinn-perturb-4090`, RTX 4090, 16 vCPUs, EUR-IS-1, $0.89/h), image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
  Build: `setup_pert.sh` (hybrid GPU binary and a non-GPU base-spectral wave binary, both `--comoving=false`; single precision, `FLOAT8 OFF`).
- Problem: shipped `example/test_problem/ELBDM/Perturbation` (2D, 64x64x16 base, `MAX_LEVEL 3`, interference + spectral refinement,
  `ELBDM_FIRST_WAVE_LEVEL 3`, END_T 2.0 default). Spectral interpolation tables came from `LSS_Hybrid/download_spectral_interpolation_tables.sh`.
  Only changes: `OUTPUT_DT 0.5`, `OPT__OUTPUT_POT 0`, `OMP_NTHREAD` 8 (4 for the FD4 run).
- Runs (all on the pod, `/workspace`): `pert_ref512` (wave build, `ELBDM_BASE_SPECTRAL 1`, 512^2, `MAX_LEVEL 0`; 2323 steps),
  `pert_hyb` (default gravity, 94 steps, 2836 s while sharing the CPUs), `pert_hyb_fd4` (`POT_STENCIL_ORDER 4`, 91 steps, 3686 s).
  A 1024^2 reference was started and killed at step 70 (about 9000 steps needed, several seconds each); 512^2 overlays 1024^2 in the wiki plot.
- Analysis: `example/yt/extract_power_spectrum.py -s 4 -e 4 -lv <0|3> -L 1 -ndim 2` (copy with `np.asarray(data[field])`, the original fails with a unyt dtype error). `-L 1` reproduces the wiki k axis (k = 2 pi * index).
  Hybrid sampled on the native 512^2 grid (`-lv 3`), so k <= 1608.
- The wiki image (identical to the screenshot) is in `wiki_reference/`; `digitize.py` extracted the curves (about 5% accuracy in P).

## Results at t = 2.0 (Data_000004), ratio to the 512^2 spectral reference, geometric mean per k band
| k band | wiki hybrid / ref (digitized) | ours, default gravity | ours, FD4 |
|---|---|---|---|
| 6-12 | 1.84 | 1.52 | 0.97 |
| 12-25 | 2.39 | 2.28 | 1.28 |
| 25-50 | 2.24 | 2.18 | 0.92 |
| 50-100 | 3.36 | 3.14 | 0.54 |
| 100-200 | 5.46 | 4.67 | 0.59 |
| 200-400 | 5.32 | 3.95 | 0.75 |
| 400-800 | 2.37 | 2.20 | 0.66 |
| 800-1200 | 0.74 | 1.52 | 0.80 |
| 1200-1609 | 0.55 | 2.23 | 1.11 |

- Our 512^2 reference matches the digitized wiki 1024^2 curve (e.g. P(k~130) 1.0e-3 vs 0.98e-3, P(k~380) 2.2e-4 vs 2.3e-4).
- Default gravity reproduces the mismatch for k < 800 (factor 1.5-4.7, wiki 1.8-5.5). Above 800 the wiki hybrid curve is below the reference, ours is above (probably a different sampling level in the wiki).
- FD4 reduces the RMS log-ratio over the bands from a factor 2.6 to 1.4, but overcorrects at k = 50-800 (power 25-45% too low).
- Caveats: one run each; FD4 changes both the periodic base-level FFT (FD4 symbol) and the SOR solver on refined levels, so which part matters is not separated;
  the analysis is a single snapshot and a 2D radial P(k) of z-index 0.
- Data in the repo: `data/spectra_t2.csv` (k and P(k) at t = 2.0, 256 bins up to k = 1608.5, 4 significant digits; columns `ref512_spectral`,
  `hybrid_default`, `hybrid_fd4`; md5 `236c208fe4fb856ce7a2b26c58cf5ef1`, copied from `/workspace/analysis/spectra_t2.csv` on the pod),
  `wiki_reference/` (wiki image and the digitized curves). `plot_results.py` regenerates `perturbation_spectra_t2.png` and the band table
  from these files (`C:/Users/TE/anaconda3/envs/pinn_env/python.exe plot_results.py`).
- Point-wise (not band-averaged) ratios to the reference, from the figure: default gravity peaks at about 6 near k = 150-200 (digitized wiki hybrid: about 7.5);
  FD4 stays between about 0.3 and 1.5.
- Not preserved: the run directories, logs, HDF5 snapshots and the t = 1.0 spectra were only on the pod. The pod was stopped and terminated on
  2026-11-16 at 14:25 UTC (account audit log; total billed $1.40), which deleted `/workspace`. Only the files in this folder survive.
  Reproducing a run needs a new pod and `setup_pert.sh` (about 6 min build; hybrid default run about 50 min, FD4 about 60 min, 512^2 reference about 80 min while sharing 16 vCPUs).
- `wiki_reference/runs.txt` is a paste of the transfer command that contained the CSV; the CSV in `data/` is the verified copy.

## Session 3 (2026-11-17): GPU utilisation and better agreement (L40S pod, 16 vCPUs, $1.09/h; the RTX 4090 was out of stock)
Full table of runs: `data/runs_session3.md`; spectra `data/*_t2.txt`; `compare.py` (band table), `plot_session3.py` -> `perturbation_spectra_t2_session3.png`.

**Why the GPU is idle.** Not the GPU architecture settings (`GPU_ARCH` ADA_LOVELACE is a copy of Ampere; no gain expected, none tested). The hybrid run is CPU bound:
ghost-zone interpolation with the spectral scheme on the fluid levels takes about 90% of the fluid step; the GPU solvers take a few seconds. The wave level (3) appears only after t ~ 1.65.
`SPEC_INT_GHOST_BOUNDARY` 2/3 gave -15%/0% time. `FLU_GPU_NPGROUP`/`GPU_NSTREAM` were not swept (auto values 568/4).

**Speed levers (all from t = 0 to 2, same machine):**
- `OPT__FLU_INT_SCHEME 4` (CQuad) for the ghost zones with `OPT__REF_FLU_INT_SCHEME 8` (spectral, for newly refined patches): 1.85x faster (2292 -> 1235 s) with no loss of accuracy (spectral ghosts w2spec vs CQuad ghosts w2: identical RMS). Using CQuad also for refinement (session 2 test) is what made the spectrum much worse (RMS 3.2).
- `ELBDM_FIRST_WAVE_LEVEL 2` instead of 3: another 3.4x (1263 -> 370-380 s); level 1 is slower to converge and worse (RMS 2.0).

**Accuracy levers (RMS log factor of P/P_ref at t = 2, lower is better; default 2.49):**
- New option `POT_FFT_SYMBOL` (fork commit afa7e9630): base-level FFT Green's function evaluated at the finest-level spacing (2) or exact -k^2 (1) -> 1.64/1.62. The default base-level symbol (64^2 grid) over-estimates the potential by 20-150% at the base-grid mid/high k, and the refined levels inherit this through their boundary conditions.
- Plus `POT_STENCIL_ORDER 4` (SOR on refined levels) with `POT_FFT_SYMBOL 2` -> 1.31 (bands 0.8-1.2 except low k 1.2). FD4 symbol at the finest level (3) -> 1.35 (undershoots).
- Best accuracy: FD4 SOR + `POT_FFT_SYMBOL 2` + CQuad ghosts + spectral refinement: RMS 1.27 in 1263 s (default 2.49 in ~2300 s).
- Fast: same with `POT_STENCIL_ORDER 2` and `ELBDM_FIRST_WAVE_LEVEL 2`: RMS 1.56 in 380 s (6x faster than default).
- No effect: `OPT__INT_PHASE 1` (RMS 1.63 vs 1.63), spectral vs CQuad ghost interpolation. `ELBDM_MATCH_PHASE 0` aborts at start.
- Moving the wave solver down to level 1 does not improve agreement, so the fluid scheme (MUSCL) is not the dominant cause of the mismatch.

**Caveats.** Single runs; configurations that differ only slightly (e.g. s4f2 vs s4f2cq, sym1 vs sym2) differ by up to 15-25% in individual bands, so differences below about 20% are not significant.
The reference is the 2nd-order-symbol 512^2 run, so 'agreement' is not the same as accuracy; a FD4-accurate operator landing below the reference suggests compensating errors may be involved.
The run directories were only on the pod.

## Session 4 (2026-11-17): fluid-scheme and restriction tests, mass loss, wave-solver floor

Figure: `perturbation_spectra_t2_session4.png` (`plot_session4.py`). Run table, per-band values and mass histories: `data/runs_session4.md`.
Scripts (pod setup, runs, wave-only run, snapshot comparison, restriction operator test): `session4/`.
Pod: L40S, secure cloud, $1.09/h, 16 vCPUs (community 4090 / 5090 / L40S were refused for capacity at every retry); terminated at the end of the session,
so run directories, snapshots and logs are gone. Kept: the spectra listed below and the tables in `data/runs_session4.md`.

### Code (fork `KunkelAlexander/gamer`, branch `fd4-poisson`, pushed up to `bb90fd39e`; defaults unchanged)
- `--hybrid_scheme=THIRD`: third-order (kappa = 1/3) upwind-biased density flux in the HJ fluid solver, same stencil as MUSCL, no Lax-Wendroff time factor;
  falls back to the limited MUSCL flux where the face value is not bounded by the two adjacent cells.
- `--hybrid_qp_order=4`: five-point gradient/Laplacian of ln(rho) in the quantum pressure (fits the existing 2 ghost cells per RK stage).
- `OPT__RESTRICT_ORDER` (runtime): 2 = box average (default), 4 = cubic restriction of all fields on the 16^3 block of the eight son patches
  (one-sided cubic stencils at the block edges, continuously unwrapped phase, box-average fallback for non-positive density),
  3 = fourth order for phase / wave function only, 5 = density only.
- Not compiled on a pod before this session; all builds completed without errors (no warnings from `Flu_FixUp_Restrict.cpp`; warnings in the other new code were not inspected).

### Findings
1. **Runs are bitwise deterministic**, independent of thread count and build: `s4_rep` (new binary, base configuration) reproduces the session 3 `s4f2cq` run in every
   band (RMS 1.27). The "scatter" of 10-20% per band between near-identical configurations is therefore not run-to-run noise but the deterministic, chaotic response
   of the solution to any small change. At t = 1.0 the third-order fluid scheme already changes the density by 14% (relative L2) compared with the default flux,
   i.e. the fluid-level solution is not converged. There is no noise floor to measure; differences must be judged against this sensitivity.
2. **Floor set by the wave solver:** a uniform 512x512x16 run of GAMER's GramFE solver (no AMR, no fluid) reproduces the spectral reference to 1-4% in every band
   (RMS 1.09). The best hybrid configuration (RMS 1.27) is within ~20% of this floor; the shipped defaults are at 3.03.
3. **Mass loss is not a hybrid bug and not caused by the session 3 settings.** Total mass error at t = 2: shipped defaults -3.5%, best configuration -3.7%,
   uniform GramFE wave run -5.1%. In all three it starts around t ~ 1.7 and grows quickly (uniform run: -0.2% at 1.7, -1.2% at 1.8, -3.8% at 1.9). Earlier I had only
   looked at the best configuration and had not checked the shipped defaults. What is established: it occurs in the pure wave solver; level 3 covers the whole volume from
   t = 1.84, so it is not a fluid/wave-interface effect; level-3 time steps (~6e-4) are below the CFL limit (8.7e-4); the reference solver conserves mass exactly.
   What is inferred, not measured: the GramFE evolution operator applies a very sharp k-space filter every step (`exp(-16 ln10 |k/kmax|^100)`, see
   `ELBDM_GramFE_EvolutionMatrix.cpp`) that only acts within ~10% of the extended grid's Nyquist, so the lost mass sat at the grid scale (structure forming after t ~ 1.7).
   The spectrum is insensitive to it (uniform GramFE: RMS 1.09 with -5% mass). Not tested: whether the onset moves with resolution (1024^2 GramFE), or the exact content removed.
4. **Fourth-order restriction as implemented is harmful** and should not be used. The operator itself is correct (numpy check `session4/restriction_operator_test.py`:
   interior error 3e-2 -> 3e-3 -> 1.7e-4 for h -> h/2 -> h/4, mean preserved on smooth data, gain on grid-scale modes only moderately above the box average),
   but in the hybrid both halves hurt: phase / wave function only (`res3`) RMS 3.01, mass -10.6%; density only (`res5`) RMS 1.71 but mass -40.9%; all fields (`res4`) RMS 7.54, mass -15.6%;
   with the third-order fluid scheme (`res4_third_qp4`) RMS 3.30, mass -6.6%. At t = 1.0 `res3` differs from the baseline by only 2% in density, so the damage develops later
   (wave-dominated phase). Likely cause (untested): stencils with negative weights overshoot on oscillatory fields at wave/fluid boundaries, where the positive-weight box average is
   robust; density restriction additionally changes the coarse-level Poisson sources. A wiring error in the GAMER integration (e.g. phase branch consistency across son patches in the
   fluid-fluid path, where `PHAS` is restricted without unwrapping) is **not excluded**; both variants being bad is what a wiring problem would look like.
5. **Third-order flux + fourth-order quantum pressure (`third_qp4`): mixed.** Bands 50-400 move to within 11% of the reference (1.06, 0.89, 1.10 against 0.68, 0.76, 0.87),
   but the top band rises to 1.93 (1.06 before) and bands 6-50 to 1.3-1.4, so RMS is 1.85 against 1.27 (RMS k < 400: 1.27 against 1.38). Mass error -4.5%.
   The two changes (flux order, quantum-pressure order) were not separated.
6. **Correction of an earlier statement:** the wave level (level 3) does not appear only after t ~ 1.65; wave patches exist from t ~ 0.7 and cover 35% of the volume at t = 0.93,
   58% at 1.06 and 100% from t = 1.84 (`Record__Hybrid`).

### Observations not yet tested
- The default MUSCL flux contains a Lax-Wendroff factor (1 - CFL) and `DT__HYBRID_VELOCITY` defaults to 1.0, so the second-order correction can vanish at high local CFL.
  The THIRD flux removes the factor; a half time step (`dt05`) would test the temporal side.
- The reference problem is a thin slab (z = 16 base cells); at level 3 the hybrid carries 128 z cells for z-independent physics. `NX0_TOT_Z 8` should give about 2x speed for free.
  Untested whether results are z-independent.
- The uniform GramFE run took 2100 s with 6 threads on a shared pod, so the hybrid's speed advantage over a uniform wave solver is modest for this 2D test (timings are not clean).
- Our 512^2 reference is 1.13x the digitized wiki 1024^2 curve in the top band (k 1200-1609), 0.92-1.04 elsewhere; comparisons above k ~ 1200 are limited by the reference itself.

### Not completed
`flagfine` (finer refinement thresholds; stopped at t = 1.3), `snap025` (snapshots every 0.25 for cheap restarts from t = 1.75; stopped at t = 1.04), `resphase`
(`OPT__RES_PHASE 1`) and `dt05` (half time step) were queued but not run to the end. `res3` / `res4` / `res5` / `res4_third_qp4` spectra were not saved, only band values.

### Suggested next steps
1. Finish `flagfine` and `dt05` (cheapest tests of fluid-level convergence), then `resphase`.
2. Separate `HYBRID_SCHEME=THIRD` and `HYBRID_QP_ORDER=4` and find out why the top band worsens.
3. Test `NX0_TOT_Z 8` for speed.
4. Only if higher-order restriction is still wanted: add a diagnostic that compares box and cubic restriction cell by cell during a run and check phase branch consistency across son patches.
5. Optional: 1024^2 uniform GramFE run to see whether the mass-loss onset moves.

Spectra saved: `data/wave_gramfe_t2.txt`, `data/third_qp4_t2.txt`, `data/default_ship_t2.txt` (256 values, 4 significant digits, same k bins as `spectra_t2.csv`; `compare.py` picks them up).
