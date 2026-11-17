# Session 4 runs (2026-11-17)

Pod: L40S, secure cloud ($1.09/h), US-NC-1, 16 vCPUs, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. Fork `KunkelAlexander/gamer`, branch `fd4-poisson`
at `bb90fd39e` (parent commits `cc4950314`, `1a3b60d17`). Single precision, 64x64x16 base, `MAX_LEVEL 3`, t = 0 -> 2.
Reference: `ref512_spectral` of `spectra_t2.csv` (512^2 spectral base-level run). Metric: geometric-mean P/P_ref per band, "RMS log" over k > 6.

Base configuration ("best accuracy" of session 3, `s4f2cq`): `POT_FFT_SYMBOL 2`, `POT_STENCIL_ORDER 4`, `OPT__FLU_INT_SCHEME 4`, `OPT__REF_FLU_INT_SCHEME 8`,
`ELBDM_FIRST_WAVE_LEVEL 3`. Binaries: `base` (default MUSCL flux, `HYBRID_QP_ORDER 2`) and `third_qp4` (`--hybrid_scheme=THIRD --hybrid_qp_order=4`).

| tag | what | RMS log | RMS k<400 | mass error at t = 2 | spectrum file |
|---|---|---|---|---|---|
| s4_rep | repeat of session 3 `s4f2cq` with the new binary | 1.27 | 1.38 | -3.72% | identical to session 3 `s4f2cq_t2.txt` (all bands equal) |
| default_ship | shipped defaults (`POT_FFT_SYMBOL 0`, SOR order 2, spectral ghost + refinement interpolation) | 3.03 | 5.09 | -3.54% | `default_ship_t2.txt` |
| wave_gramfe | uniform 512x512x16 GramFE wave run (`--elbdm_scheme=WAVE`, no AMR, no fluid, base-level FFT gravity), 6 threads | 1.09 | 1.08 | -5.10% | `wave_gramfe_t2.txt` |
| third_qp4 | `HYBRID_SCHEME=THIRD` + `HYBRID_QP_ORDER=4` on the base configuration | 1.85 | 1.27 | -4.50% | `third_qp4_t2.txt` |
| res3 | `OPT__RESTRICT_ORDER 3`: 4th-order restriction of phase / wave function only (density stays box average) | 3.01 | 2.72 | -10.55% | band values only |
| res4 | `OPT__RESTRICT_ORDER 4`: 4th-order restriction of everything | 7.54 | 7.03 | -15.64% | band values only |
| res5 | `OPT__RESTRICT_ORDER 5`: 4th-order restriction of density only | 1.71 | 1.78 | -40.88% | band values only |
| res4_third_qp4 | `OPT__RESTRICT_ORDER 4` with the `third_qp4` binary | 3.30 | 4.01 | -6.55% | band values only |
| flagfine | finer refinement thresholds (interference QP 0.03 -> 0.01, spectral 1.0 -> 0.3) | not finished (stopped at t = 1.3) | | | |
| snap025, resphase, dt05 | snapshots every 0.25; `OPT__RES_PHASE 1`; half time step | not run / not finished | | | |

Wall times (1480-3840 s) are not comparable: runs shared the 16 vCPUs with up to four other jobs and builds, with 6 or 8 threads each.
Results do not depend on the thread count (see README, determinism).

## Per-band geometric mean P/P_ref

| band | default_ship | s4_rep | wave_gramfe | third_qp4 | res3 | res4 | res5 | res4_third_qp4 |
|---|---|---|---|---|---|---|---|---|
| 6-25 | 2.09 | 1.20 | 1.01 | 1.36 | 1.88 | 2.58 | 1.34 | 1.95 |
| 25-50 | 1.64 | 1.04 | 1.01 | 1.32 | 3.54 | 6.35 | 1.59 | 3.61 |
| 50-100 | 3.41 | 0.68 | 1.02 | 1.06 | 4.54 | 5.45 | 0.74 | 4.26 |
| 100-200 | 6.45 | 0.76 | 1.02 | 0.89 | 2.32 | 8.08 | 1.94 | 3.55 |
| 200-400 | 5.48 | 0.87 | 1.04 | 1.10 | 2.25 | 7.33 | 1.71 | 4.36 |
| 400-800 | 2.82 | 0.81 | 0.96 | 0.84 | 2.41 | 5.15 | 0.77 | 2.34 |
| 800-1700 | 1.86 | 1.06 | 1.02 | 1.93 | 3.29 | 8.44 | 1.39 | 3.16 |
| RMS log | 3.03 | 1.27 | 1.09 | 1.85 | 3.01 | 7.54 | 1.71 | 3.30 |

## Total mass error (Mass_Psi relative error) over time

| run | t = 1.5 | 1.7 | 1.8 | 1.9 | 2.0 |
|---|---|---|---|---|---|
| s4_rep | +0.11% | +0.06% | -0.08% | -0.73% | -3.72% |
| default_ship | +0.11% | +0.06% | -0.14% | -0.97% | -3.54% |
| wave_gramfe | -0.00% | -0.18% | -1.23% | -3.77% | -5.10% |
| third_qp4 | +0.05% | -0.02% | -0.19% | -1.22% | -4.50% |
| res3 | +0.10% | +0.07% | -0.02% | -0.95% | -10.55% |
| res4 | +0.10% | +0.02% | -0.16% | -2.99% | -15.64% |
| res4_third_qp4 | +0.05% | -0.03% | -0.26% | -1.50% | -6.55% |
| res5 | +0.10% | -0.42% | -6.18% | -23.50% | -40.88% |

Wave-level coverage in `s4_rep` (`Record__Hybrid`): first wave patches at t ~ 0.7 (14% of patches, 1.6% of the volume), 35% of the volume at t = 0.93,
58% at t = 1.06, 93% at t = 1.63, 100% from t = 1.84 (74880 patches, 87.5% of them on level 3).

## Density at t = 1.0 relative to `s4_rep` (level-1 covering grid, relative L2 difference)

res3 2.0%, res4 11.8%, res5 12.2%, third_qp4 14.0% (max local difference up to 20 x the mean density).
