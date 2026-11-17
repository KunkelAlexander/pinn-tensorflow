# Session 3 runs (L40S pod, 16 vCPUs, single precision, hybrid GramFE MATMUL, t = 0 -> 2, 64x64x16 base, MAX_LEVEL 3)
Spectra: `<tag>_t2.txt` = 256 P(k) values (4 significant digits) at t = 2.0, same k bins as `spectra_t2.csv`. Reference = `ref512_spectral` column there.
`compare.py` prints the band table. Binary: fork KunkelAlexander/gamer, branch fd4-poisson at afa7e9630 (POT_FFT_SYMBOL).

| tag | settings changed vs default | wall time (s) | RMS log factor |
|---|---|---|---|
| default (session 2) | - | not timed with this binary (about the same as sym2) | 2.49 |
| cquad | OPT__FLU_INT_SCHEME 4, OPT__REF_FLU_INT_SCHEME 4 | not recorded (about 2.5x faster than spectral in the restart sweep) | 3.22 |
| sym2 | POT_FFT_SYMBOL 2 | 2292 | 1.64 |
| sym1 | POT_FFT_SYMBOL 1 | 2368 | 1.62 |
| s4f2 | POT_FFT_SYMBOL 2, POT_STENCIL_ORDER 4 | 2279 | 1.31 |
| s4f3 | POT_FFT_SYMBOL 3, POT_STENCIL_ORDER 4 | 2757 | 1.35 |
| cqghost | POT_FFT_SYMBOL 2, OPT__FLU_INT_SCHEME 4 (ghost zones), OPT__REF_FLU_INT_SCHEME 8 (refinement stays spectral) | 1235 | 1.52 |
| s4f2cq | POT_FFT_SYMBOL 2, POT_STENCIL_ORDER 4, OPT__FLU_INT_SCHEME 4, OPT__REF_FLU_INT_SCHEME 8 | 1263 | 1.27 |
| w2 | s4f2cq + ELBDM_FIRST_WAVE_LEVEL 2 | 370 | 1.63 |
| w1 | s4f2cq + ELBDM_FIRST_WAVE_LEVEL 1 | 423 | 2.02 |
| w2fd2 | w2 with POT_STENCIL_ORDER 2 (FD2 SOR, POT_FFT_SYMBOL 2) | 380 | 1.56 |
| w2ip1 | w2 + OPT__INT_PHASE 1 | 440 | 1.63 |
| w2spec | w2 with spectral ghost interpolation (OPT__FLU_INT_SCHEME 8) | 1150 | 1.63 |
| w2mp0 | w2 + ELBDM_MATCH_PHASE 0 | crashed at start ("MaxPot == 0.0 at lv 2") | - |

Restart timing sweep (t = 1.0 -> 1.05, same binary): spectral default 114 s, CQuad ghost+refine 44 s, SPEC_INT_GHOST_BOUNDARY 2: 97 s, 3: 113 s (no useful gain).
Notes: level 3 (wave level) exists only after t ~ 1.65; before that all levels are fluid levels. Run-to-run scatter between near-identical settings is about 10-20% per band.
