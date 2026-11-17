# Research directions for a next-generation Schrödinger–Poisson (SP) solver

Written 2026-11-04 after a review of everything in `toy_models/`. It is a synthesis of existing
write-ups, not new experiments. Nothing was re-run for it. Numbers are quoted from the files named in
each section, so check the source before relying on one.

**Project goal (from the user):** a high-performance SP solver for cosmological fuzzy/wave dark matter
(GAMER-like 3D AMR). In their 3D runs the gravity solver gives a mismatch in the **mid to outer halo**
(plot relative error `rho/rho_ref - 1` against log r). Solitons are easy. Halos are the problem. The
toy models here exist to reproduce that mismatch cheaply (1D/2D), narrow down the cause, and test fixes.
The user wants autonomous, independent research ("no need to check in").

**Scope of all evidence below:** dimensionless 1D/2D periodic systems, fixed scale factor, uniform fine
wave grid (except where stated), no compiled GAMER run. Nothing here measures a 3D cosmological halo or
a production speedup.

---

## 1. Established findings (with evidence)

| # | Finding | Where |
|---|---|---|
| 1 | **Gravity discretisation dominates the error budget, not the wave solver.** With FD2 gravity, Gram-FE and the FD (Laplacian-4) wave solver give nearly identical profiles. Spectral gravity exposes the wave-solver differences. Six cells/wave + spectral gravity beats twelve cells + FD2. FD4 is about 60–80x better than FD2 on the same grid, the cheapest lever. | `fd_wave_crosscheck.html`, `gravity_accuracy.html`, `zoom2d_results/SUMMARY.md` (Follow-up 1 and 2) |
| 2 | **Inherited-boundary Poisson hierarchies have an error floor.** The trace-free (tidal) error is about 4e-4 at level 6 even when the fine solve is exact (exact boundaries: 1e-6). The ~1e-3 error for a core crossing a patch edge does not improve with N or dt. Method of local corrections (MLC) helps only modestly. | `gamer_diagnosis.html` (sec. 3), `frozen_gravity.html`, SUMMARY.md |
| 3 | **Late-time mid/outer-halo error is mostly spurious self-force, i.e. loss of translation covariance.** A fixed coarse grid cannot conserve momentum. Granule-scale (speckle) density aliases through cell-mean restriction + Lagrange interpolation. A net self-force appears, abs(p) saturates in about 1 time unit, and the halo drifts. Measured about each run's own centroid, the outer/far errors collapse (coarse r8: +165%/+480% -> -1%/+12%). | SUMMARY.md Follow-up 3, `analyze_centred.py`, `static_force.py` |
| 4 | **Control parameter is sigma/H** (granule or soliton width over Poisson cell). Net self-force switches on steeply below sigma ~ H (static 1D: 6e-9 at 1.6H, 1e-4 at 1.0H, 5.6e-2 at 0.48H). Same mechanism in 1D and 2D. | SUMMARY.md Follow-up 4, `minimal_1d.py`, `minimal_selfforce.py` |
| 5 | **A second, opposite mechanism is force softening.** The smooth TSC restriction/prolongation pair removes the drift but low-pass filters the density: core -13 to -22%, far -13 to -32%, unchanged when measured about the centroid. | SUMMARY.md Follow-up 3, "Mechanism 2" |
| 6 | **Symmetrising the operator does not remove the drift.** CIC/TSC transpose pairs, or a coarse-only Galerkin solve (abs(p) 7.9), still drift. | SUMMARY.md Follow-up 3 |
| 7 | **Composite-grid Galerkin patch is the only treatment that fixed both mechanisms**, but only when the fine region covers the granular halo (half-width 32-48 fine cells, about r90-r99). Net self-force 5e-4-9e-4 vs 0.06-0.6 (inherited patch), 0.03 (MLC), 0.2-3 (coarse only). It is a global sparse LU, so not scalable as written. | `soliton_gravity.py` (`kind='galerkin'`), SUMMARY.md |
| 8 | **1D stationary halo:** coarse H >= 0.25 makes the whole halo wander or drift. At L=128, H = 0.125 is within 1-2x the noise floor, so there is no clear systematic density bias without drift. H = 1.0 also builds a spurious far halo. The profile is Gaussian-like, not power-law. | `halo_1d.py`, `soliton_gravity_results/halo_1d_{M64,L128}_table.txt` |
| 9 | **Wave-solver interface issues are unresolved.** (a) Time interpolation of complex psi across coarse/fine levels gives about 2% density error (0.021 at phase 0.25/step) that depends on the phase gauge and survives resolution doubling; the constant-potential phase changes the error while the continuum density is unchanged. (b) Gram-FE grows weak short-wave modes under a prescribed potential, with no inaccurate gravity needed. Neither is established as the cause of the user's halo mismatch. | `gamer_diagnosis.html` sec. 4, `frozen_gravity.html`, `gramfe_boundary_time.py` |
| 10 | **A global adaptive SIPG operator removes interface errors** (mass 1e-12, energy 1e-6) but showed **no dof saving in 2D** (5632 dofs for 7.7e-4; uniform FFT reaches ~1e-3 at N~80, about 6400 dofs) and took about 350 s in Python. The mesh must follow the moving core. | SUMMARY.md (first section), `adaptive_sipg.py`, `zoom2d.py` |
| 11 | **Dead ends / downgraded ideas:** fixed reference-Hamiltonian splitting (failed moving-core counterexample, see `review.html`); fixed mapped grids (worse for expanding SP packets); narrow Fourier bands (mass/energy about 1e-12, density wrong); nonuniform-Fourier conditioning (an artefact of the Fourier space, not an intrinsic limit). | `report.html`, `review.html`, `representations.py` |
| 12 | **Neural approaches:** the literature search found none that beats or has been benchmarked against GAMER for SP. They are emulators or proofs of concept. Every failure above has a structural (non-learned) fix. The 2026 arXiv IDs in `FUTURE_RESEARCH.md` were not independently verified. | `FUTURE_RESEARCH.md` |

---

## 2. Pitfalls that cost wrong conclusions (read before any new halo test)

Also recorded in the agent memory file `feedback-halo-test-pitfalls`.

1. A random-phase halo carries net momentum and drifts through the periodic box. Remove it with a
   Galilean boost and centre it (`stationary_halo.centre_and_rest`). An earlier "virial" halo drifted
   to (-4.5, 4.6) and confounded the patch-size results (corrected in SUMMARY.md, Follow-up 3).
2. Confirm stationarity under **spectral** gravity (rms, core density, profile, virial ratio, energy,
   centroid) before comparing methods. A single wave function is only statistically stationary. Build it
   from eigenstates of its own mean-field potential. Halos A/B are stationary for T <~ 25 and relax
   slowly after that. Always compare to the spectral run, never to the initial state.
3. Symmetric (self-adjoint) rho->phi operators do **not** conserve momentum. That needs translation
   invariance. The self-force is huge on speckled density and tiny on the smooth mean profile, so test
   on real snapshots.
4. Pointwise density error is a poor metric because granule phases dephase. Use profile/statistical
   metrics, relative error vs log r, centroid, net momentum, and compare to the noise floor. The
   realisation scatter of the T=16 mean profile is 8-37%, mostly in the core. The floor is the spectral
   run with a 1e-6 perturbation.
5. Soliton tests are not sensitive probes of the profile error. They show drift only.

---

## 3. Recommended research directions (priority order)

Status tags: **[ready]** = existing harness can run it now; **[new]** = needs building; **[untested]** =
hypothesis with no evidence yet.

### 1. Translation-covariance defect as a diagnostic, run on the 3D runs first [ready for toy, new for 3D]
Translate the density by sub-cell offsets, re-solve, and measure (a) the net self-force
`F_net = -int rho grad(phi)` and (b) how the force varies with offset, level by level. It costs seconds
per snapshot and needs no evolution (`static_force.py`, `minimal_1d.py` show the recipe). Apply it to
snapshots of the user's 3D AMR runs. It would tell whether the 3D mismatch is the drift type, the
softening type, or neither. SUMMARY.md explicitly says this is not yet established. Everything below
depends on the answer.

### 2. Force-symmetrisation fix before any new solver [untested]
Subtract the mass-weighted mean force per level (Newton's third law projection). It can only cure
bulk drift, not per-granule heating or softening. Test in the 1D halo harness (`halo_1d.py`, about
3 min for 4 seeds). If drift is the dominant 3D symptom this is the cheapest possible remedy. If it
fails, it shows that granule-level self-force matters.

**Result (2026-11-04, `halo_1d_meanforce.py`, 1D L=32 N=1024, M=32 T=4 rc=8, 3 seeds, T_end=60, avg over last 20):**
mean-force subtraction removes the bulk drift (max centroid 14.6 -> 0.007 at H=0.5; 4.8 -> 0.036 at H=1.0;
0.064 -> 0 at H=0.25). Profile error vs spectral (zone-summed density ratio, mean over seeds):

| H | core | mid | outer | drift |
|---|---|---|---|---|
| 0.25 | +0.2% | +0.9% | +6.8% | 0.064 |
| 0.25 + mf | +0.7% | +0.1% | +7.5% | 0 |
| 0.5 | -0.9% | +2.3% | +458% | 14.6 |
| 0.5 + mf | +0.2% | -2.1% | +259% | 0.007 |
| 1.0 | +2.8% | -22% | +25600% | 4.8 |
| 1.0 + mf | -1.3% | -9.7% | +2300% | 0.036 |

Reading: the fix cures drift and roughly halves the mid-halo error at H=1.0, but the **outer error (spurious far
halo / evaporated granules) is barely changed**, so drift is not the whole story. The remaining error is
granule-level self-force (heating), as predicted if it fails. Caveats: 3 seeds only, outer bins hold tiny
density so the huge percentages are ratios of near-zero mass, the perturbed-spectral floor is ~0.0% in this
metric (too clean to judge the 0.1-1% core/mid numbers), and 1D only. Next: test a per-granule-aware remedy
(directions 3/4) against this baseline.

### 3. Translation-covariant, deconvolvable transfer operators [new, untested]
Use a band-limited high-order B-spline quasi-interpolant for restriction and prolongation. The
operator is drift-free by construction, and its softening is a known linear filter that can be
pre-compensated. A small data-driven variant: optimise the stencil weights to minimise net self-force
plus force error over random-phase speckle fields at many offsets (a handful of parameters; the
`pinn_env` TensorFlow install is available).

### 4. sigma/H-aware refinement or switching criterion [new, untested]
Derive the local de Broglie wavelength from the Madelung velocity and require H_level <~ sigma_granule.
Where that cannot be met, either coarse-grain the density for gravity with a covariant filter, or
hand the region to a fluid (HJM) solver (hybrid wave/fluid schemes are in the literature list in
`FUTURE_RESEARCH.md`). Test with a spatially varying H in the 1D halo.

### 5. Scalable composite Galerkin patch [new]
Add an FD4 or compact fine operator (FD2 on the fine grid has its own few-percent profile error;
FD4 is <0.1%). Replace the sparse LU with matrix-free CG preconditioned by multigrid/FAC. The composite
system is symmetric (semi-)definite. Demonstrate in 2D, then in 3D at toy size. The scaling is the
research question. Known cost of the current version: build about 17 s, solve about 65 ms at N=256.

### 6. Fix wave-solver interface errors [ready for 1D]
Interpolate in time with Hermite, using psi_t = -iH psi from the equation, or in a gauge-removed
variable (psi * exp(i int Phi dt)) instead of real/imaginary parts. `gramfe_boundary_time.py` has the
1D setup that reproduced the 2% error.

### 7. Compact core in a large empty domain [new]
The one test still not run. It decides whether adaptivity saves anything. The 2D result showed no dof
saving, so non-uniform wave discretisations may not be worth building. If they are not, the answer is
"keep AMR and fix gravity", and most exotic representations can be dropped.

### 8. Close the toy's realism gaps before generalising [new]
- A halo with a power-law mid/outer envelope. The lowered-isothermal halos have Gaussian-like tails,
  so "mid halo" is poorly represented. Options: other f(E), or a soliton + NFW-like initial state.
- Ensembles over seeds, with the perturbed-spectral noise floor (the 1D runs used only 4-6 seeds).
- Comoving expansion (scale factor a(t)).
- 3D. The granule self-force scaling with sigma/H may differ in 3D.

---

## 4. Deprioritise

PINN and Neural Galerkin solvers, Fourier-band and mapped-grid representations, fixed
reference-Hamiltonian splitting, learned emulators in the solver's critical path (only the stencil
optimisation in direction 3 looks useful), quantum algorithms (the one relevant hit, arXiv 2307.06032,
is unread).

---

## 5. File map

- `zoom2d_results/SUMMARY.md`: the main running log (Follow-ups 1-5b). Read this first.
- `README.txt`: index of the 1D rounds and reproduction commands. It is out of date for the 2D work.
- 1D reports (open `review.html` first, then `report.html`): `review.html`, `report.html`,
  `scale_communication.html`, `gamer_diagnosis.html`, `frozen_gravity.html`, `gravity_accuracy.html`,
  `potential_replay.html`, `fd_wave_crosscheck.html`.
- 2D/halo code: `soliton_gravity.py` (gravity kinds: spectral, fd2/fd4, coarse rN, patch, mlc, galerkin),
  `stationary_halo.py`, `halo_gravity.py`, `halo_long.py`, `static_force.py`, `analyze_*.py`,
  `plot_halo_*.py`.
- Minimal tests: `minimal_1d.py`, `minimal_selfforce.py`, `minimal_soliton_drift.py`, `halo_1d.py`.
- Results and figures: `soliton_gravity_results/`.
- Literature notes (a pasted search log, not an authored review): `FUTURE_RESEARCH.md`.

**Environment:** run everything with `C:\Users\TE\anaconda3\envs\pinn_env\python.exe` (see the repo's
`CLAUDE.md`). Jupyter kernels die in the sandbox, so run plain `.py` scripts.

## Result: stage decomposition (2026-11-04, halo_1d_stages.py; 1D, L=32, N=1024, 2 seeds)
Zone-summed profile error vs spectral. Restriction/parent/prolongation variants (mf = mean-force subtracted):

| H | variant | mid | outer | drift |
|---|---|---|---|---|
| 0.5 | mean/fd2/lag4 (current) | +2.9% | +385% | 14.7 |
| 0.5 | mean/spec/lag4 (+mf) | +6.7% (+1.2%) | +84% (-30%) | 2.7 (0.001) |
| 0.5 | fourier/fd2/fourier | -2.8% | +22% | 0.000 |
| 0.5 | fourier/spec/fourier | +1.7% | +17% | 0.000 |
| 1.0 | mean/fd2/lag4 (+mf) | -17.8% (-9.4%) | +30082% (+2083%) | 6.8 (0.04) |
| 1.0 | fourier/fd2/fourier | -6.1% | +5.2% | 0.000 |
| 1.0 | fourier/spec/fourier | +2.2% | +34.5% | 0.000 |

Interpretation: the culprit is the **cell-mean restriction** (aliasing of sub-cell granule structure into the coarse grid breaks
translation covariance), not the fd2 stencil: swapping fd2 for the exact parent solve barely helps; a band-limited (Fourier)
restriction+prolongation removes drift entirely and cuts the outer error by 1-4 orders of magnitude, with no mean-force fix needed.
Remaining 5-35% outer error is the low-pass softening of gravity (fine-scale force removed) - small vs the spurious far halo.
Caveat: Fourier restriction is global; the AMR-feasible analogue is a wide local band-limiting filter (direction 3: B-spline/sinc-windowed
quasi-interpolant, anti-aliased before decimation). Next: test a compact local anti-alias filter (e.g. cell-mean applied twice / cubic
B-spline prefilter) in this harness and see how much of the Fourier benefit survives; then 2D/3D.

## Result: local anti-alias restriction (2026-11-04, halo_1d_filters.py; 1D L=32 N=1024, 2 seeds)
Restriction = B-spline kernel of degree d (width (d+1)H) per coarse cell, fd2 parent, quartic Lagrange prolongation. Outer-zone error / drift:

| variant | H=0.5 | H=1.0 |
|---|---|---|
| B0 (cell mean, current) | +385% / 14.7 | +28214% / 6.8 |
| B1 (triangle) | +55% / 0.9 | +7188% / 3.0 |
| B2 | +24% / 2.5 | +15710% / 14.3 |
| B3 (cubic) | +17% / 2.4 | +24953% / 9.2 |
| B3 + coarse deconvolution | +70350% / 12.9 | +3622% / 4.8 |
| B3 + dec + mean force | +54% / 0.012 | +700% / 0.03 |
| Fourier/fd2/Fourier (global) | +22% / 0 | +5% / 0 |

Interpretation: a compact B-spline anti-alias prefilter recovers most of the benefit at H=0.5 (outer 385% -> 17-55%) but leaves residual
drift (aliasing not fully removed) and fails at H=1.0 (error stays 10^3-10^4 %): a local filter is not sharp enough. Coarse deconvolution
amplifies aliased high-k power and hurts. Mean-force subtraction on top of B3+dec helps (drift ~0) but still leaves +700% at H=1.0.
=> when H >~ granule scale (sigma/H small) no local transfer fix suffices; need direction 4 (refine so H <~ sigma, or switch to a
band-limited/spectral patch there) plus mean-force subtraction for the residual drift. Next: quantify the sigma/H threshold per variant
(sweep H at fixed halo) and test a spatially varying H (fine near-core, coarse only where local sigma/H is safe).

## Result: H sweep and composite (spatially varying) resolution (halo_1d_varH.py, varH.log)

Outer-zone error vs spectral (drift in brackets), 2 seeds, L=32, N=1024:

| H | mean | mean+mf | B3 | B3+mf | Fourier |
|---|---|---|---|---|---|
| 0.25 | -2% (0.02) | -2% (0) | -29% (0.01) | -29% (0) | +33% (0) |
| 0.50 | +385% (14.7) | +189% (0.01) | +17% (2.4) | +20% (0.001) | +22% (0) |
| 1.00 | +30082% (6.8) | +2083% (0.04) | +24953% (9.2) | +14% (0.002) | +5% (0) |
| 2.00 | +23536% (1.6) | +26917% (1.6) | +2543% (3.6) | +164% (0.002) | -4% (0) |

- Threshold: plain cell-mean is fine at H<=0.25 and breaks at H=0.5. B3 prefilter + mean force is the best local scheme: drift-free, outer error +20% / +14% at H=0.5 / 1.0, +164% at H=2. (An earlier table's B3+dec+mf at H=1 gave +700%: without deconvolution it is far better, so deconvolution should be dropped.)
- Composite (fine spectral solve for |x|<rw, coarse cell-mean for the rest): error 0.0% everywhere. This is VACUOUS: this halo is so compact that 99.8% of the mass is inside |x|<2 (far mass fraction 2e-3 at rw=2, 3e-8 at rw=6), so the coarse part carries nothing. Direction 4 cannot be judged until the halo is extended (L=128 box, power-law envelope; directions 7/8).
- Corollary: the spurious outer halo is generated by aliasing of the compact core's granules, so resolving the core finely removes it. In a real extended halo the far granules need their own sigma/H check.
Next: extended halo with mass in the envelope, then re-run composite and B3+mf.

## Result: extended halos (halo_1d_varH2.py; varH2_T12.log, varH2_T24.log)

Halos with real envelope mass (L=32, N=1024, M=32): T=12 rc=10 (20% of mass at |x|>2, 3% at |x|>4) and T=24 rc=12 (49% / 19%). All runs with mean-force subtraction, 2 seeds. Zone errors vs spectral (outer / mid):

| H | T=12 mean+mf | T=12 B3+mf | T=24 mean+mf | T=24 B3+mf | best composite (rw=4) |
|---|---|---|---|---|---|
| 0.5 | +1.6 / -6% | +11 / -2% | +10 / -5% | +5 / -6% | +0.6..+4 / <3% |
| 1.0 | +23 / -1% | +8 / -1% | +6 / -4% | +3 / -2% | -2..+5 / <4% |
| 2.0 | +91 / -27% (drift 1.0) | +7 / +2% | -7 / -2% | +11 / -5% | +1..+3 / <4% |

Findings
- With envelope mass the catastrophic outer errors of the compact halo (+10^2..10^4 %) disappear: nearly everything is within ~10% even at H=2 (cell-mean+mf fails only at T=12, H=2: drift ~1, +91%). Core granules there are ~0.6-1.0 wide, i.e. H >> granule, so sigma/H alone is NOT the controlling parameter.
- Reinterpretation: the huge relative errors of the compact halo come from aliased/spurious far-field mass landing in a region whose true density is ~0. The relevant ratio is (aliased spurious density) / (true density at that r). In a halo with a real envelope the true density swamps it. So relative-error-vs-log r blows up only where the reference density is tiny (steep truncation / low-density outskirts).
- B3+mean force is the most uniform fix (<=11% outer in all six cases); the composite (fine core |x|<4, coarse far part) is slightly better still (<=5% typical) but costs a fine core solve.
- Caveat: zone errors of ~5-10% are comparable to realisation scatter at T=24 (the pert run's drift is 0.04), seeds=2 only; differences between schemes below ~10% are not resolved.
- Implication for the 3D problem: a mid/outer mismatch of order tens of percent at moderate H points to the aliasing/low-density-region mechanism or to the (unsubtracted) net self-force, not to a global stencil error; diagnostics on the user's 3D data should compare rho_spurious = (coarse - ref) against rho_ref per radial bin and measure net force/centroid drift.
Next: more seeds on T=24 for a noise floor; probe the transition (T between 4 and 12) where relative outer error becomes large; then 2D check of B3+mf.

## Result: transition from compact to extended halo (halo_1d_transition.py, transition.log; 4 seeds, mean±scatter)

Outer-zone error vs spectral (mean+mf | B3+mf); the perturbed-spectral floor is ~0.1%:

| halo (mass at |x|>2 / >4) | H=0.5 | H=1.0 | H=2.0 |
|---|---|---|---|
| T=6 (1.8% / 0.03%) | +51±45 \| +18±24 % | +402±165 \| +24±22 % | +1396±525 \| +69±37 % |
| T=8 (6% / 0.3%) | +15±12 \| -9±11 % | +93±58 \| -2±16 % | +285±131 \| +39±61 % |
| T=10 (12% / 1.1%) | +4±11 \| -8±6 % | +15±9 \| -9±17 % | +116±61 \| -10±14 % |

- The outer-zone error of the cell-mean scheme falls monotonically as envelope mass grows and rises steeply with H; at H=2 it is >100% until the halo has ~12% of its mass beyond |x|>2.
- B3+mean force is consistently 5-20x better than cell mean+mf in the outer zone and its mid-zone error stays ~10-25% at H=2. Its outer error is within the per-seed scatter for H<=1, so it is statistically indistinguishable from spectral there.
- Mid-zone: cell-mean+mf is biased low (-5..-27%) at H>=1, B3+mf biased high (+3..+24%): the two filters err in opposite directions, consistent with aliasing vs low-pass softening of the force.
- Practical rule from the 1D data: cell-mean with mean-force subtraction needs H <~ 0.5 (about the core granule scale 0.6-1.0 or less) for a compact halo; B3+mf extends this to H ~ 1-2 granule widths.
Next: 2D check of B3+mf and cell-mean+mf on a 2D halo (the user's target dimensionality is 3D).

## Result: 2D check (halo_2d_b3.py, halo2d_A.log; stationary halo A: T=5 rc=8 M=80, N=256, h=0.125, 2 seeds, tend 40, avg last 15)

Zone errors vs spectral (core r<1, mid 2<r<5, outer 5<r<12), all with mean-force subtraction; noise floor ~0.0%:

| H | scheme | core | mid | outer | drift |
|---|---|---|---|---|---|
| 0.5 | cell mean | +6.6% | -2.0% | +4.9% | 0.001 |
| 0.5 | B3 | -3.8% | +5.2% | -3.4% | 0.001 |
| 1.0 | cell mean | +5.6% | +2.7% | +19.0% | 0.005 |
| 1.0 | B3 | -15.1% | +11.8% | -8.2% | 0.000 |
| 2.0 | cell mean | +3.8% | +4.3% | +31.2% | 0.021 |
| 2.0 | B3 | -15.5% | +20.9% | +6.2% | 0.000 |

- Same qualitative picture as 1D: with mean force no drift; cell-mean's outer error grows with H (+5 -> +19 -> +31%); B3 keeps the outer zone within ~8% but trades it for a ~15% core deficit and a +12..21% mid excess at H>=1, i.e. low-pass softening of gravity (core less bound, halo puffs outwards).
- So there is no free lunch: B3 removes aliasing but replaces it with a smoothing bias that grows with H. Deconvolution was supposed to fix this but amplified aliasing in 1D; a mild/partial deconvolution (gain cap ~1.5) or a lower-degree B-spline is the obvious middle ground.
- Halo A is not very compact in 2D, so the 1D blow-ups (+10^3 %) are not reproduced here. A compact 2D halo (T=2, rc=6) is being run as halo2d_C.log.

### 2D compact halo C (T=2 rc=6 M=80; halo2d_C.log), all +mean force

| H | scheme | core | mid | outer | drift |
|---|---|---|---|---|---|
| 0.5 | cell mean | -1.6% | +6% | +188% | 0.000 |
| 0.5 | B3 | -7.3% | +29% | +247% | 0.000 |
| 1.0 | cell mean | -18.5% | +237% | +552% | 0.002 |
| 1.0 | B3 | -25.3% | +607% | +1013% | 0.002 |
| 2.0 | cell mean | -42.4% | +2166% | +47146% | 0.883 |
| 2.0 | B3 | -52.2% | +5587% | +8195% | 0.011 |

- This reproduces the 3D symptom qualitatively: core loses density while mid/outer gain a spurious halo, growing steeply with H, with no bulk drift (mean force on).
- B3 is WORSE than cell mean here (opposite of 1D, and of halo A). In a compact 2D halo the dominant error is softening of the core's gravity (core unbinds, mass leaks outward), which the wider B3 kernel aggravates; aliasing is secondary. The 1D B3 benefit is therefore not robust. The mean force fix is necessary but not sufficient.
- Implication: local low-pass restriction cannot fix an under-resolved compact core; the core needs the fine Poisson solve (the composite / refinement route, direction 4/5), or a higher-order parent+transfer scheme.
Next: 2D composite (fine spectral core patch + coarse far) on halo C to see whether refinement of the core removes the mid/outer excess.

## GAMER's gravity path (read from the public master, commit e367975; src/SelfGravity, src/Init, src/Interpolation)

What was read (not run; the user's own branch, including their GramFE interpolation and refinement criteria, was not visible):

- **Density source (`Poi_Prepare_Rho`)**: each level's Poisson solve uses that level's cell data. Where children exist the parent cell holds the cell mean of its children (the cell-mean restriction of the toy "coarse rR"). Ghost zones come from `OPT__RHO_INT_SCHEME` (default `INT_CQUAD`, conservative quadratic, allowed 1-7). The mean density is subtracted for periodic boundaries.
- **Potential boundary (`Poi_Prepare_Pot`)**: the parent-level potential is interpolated to give the Dirichlet boundary of each patch-group solve, with one interpolation ghost cell, `OPT__POT_INT_SCHEME` default `INT_CQUAD`, restricted to {CQUAD, QUAD}. Temporal interpolation of the parent potential is optional (`OPT__INT_TIME`). Newly refined patches get their potential from `OPT__REF_POT_INT_SCHEME` (default CQUAD).
- **Solvers**: FFT on the base level (a Green's function in k space; for isolated BC a zero-padded real-space 1/r kernel with the self-cell term set by `GFUNC_COEFF0`), SOR or multigrid on refined patches.
- **Wave step**: the ELBDM gravity step multiplies psi by exp(-i Eta (Phi + Lambda rho) dt) and needs no potential or fluid ghost zones.
- **Spectral/GramFE interpolation** (`INT_SPECTRAL` = 8, tables from the GramFE Python generator) is only reachable for the wave function (`OPT__FLU_INT_SCHEME`, `OPT__REF_FLU_INT_SCHEME`, set in `Init_ResetParameter` for the wave scheme). It is outside the allowed range of the potential, density and gravity interpolation options in this version.
- **No momentum / mean-force correction** exists (confirmed by the user).

Why it matters for the toy-model findings
- Every level solves Poisson from its own cell-mean density, so a core or inner envelope that lives on a level whose cell is larger than the granule/core scale sees a softened force: the same under-binding of the core and spurious mid/outer halo as the toy "coarse rR" (2D halo C, force error -74% at r<0.5 for H=1).
- The toy used quartic Lagrange prolongation; GAMER defaults to quadratic (CQUAD), so the toy likely understates the interpolation error. To test: add CQUAD prolongation and CQUAD ghost-zone density interpolation to the toy.
- Mean-force subtraction needs one global sum over the leaf cells, F_bar = -sum(rho grad Phi dV)/M, applied as a uniform force; it removes bulk drift (toy: centroid drift 14.6 -> 0.007) but not the softening.
- The composite result (fine solve for the core, coarse outside) corresponds to making sure the level resolving the core also carries a Poisson solve of matching resolution, i.e. a refinement criterion tied to the local granule width / core scale rather than to density alone.

## 2D compact halo C: stage decomposition and composite (2 seeds, mean force on)

Script: halo_2d_stages.py, summary: summarize_2d_stages.py. Error vs spectral (core / mid / outer):

| H | variant | core | mid | outer | drift |
|---|---|---|---|---|---|
| 0.5 | mean/fd2/lag4 | -1.6% | +6% | +188% | 0 |
| 0.5 | mean/spec/lag4 | -4% | +22% | +128% | 0 |
| 0.5 | fourier/spec/fourier | -1% | -4% | +38% | 0 |
| 1.0 | mean/fd2/lag4 | -18% | +237% | +552% | 0.002 |
| 1.0 | mean/spec/lag4 | -23% | +299% | +993% | 0.002 |
| 1.0 | fourier/spec/fourier | -9% | +156% | +466% | 0 |
| 2.0 | mean/fd2/lag4 | -42% | +2166% | +47146% | 0.88 |
| 2.0 | mean/spec/lag4 | -53% | +3525% | +12984% | 0.37 |
| 2.0 | fourier/spec/fourier | -33% | +1118% | +1647% | 0.002 |

Static radial-force error (r<0.5): H=0.5/1/2 -> -17%/-72%/-93% (mean/fd2/lag4).

Conclusions
- Cause of the mass leak is RESOLUTION: the coarse grid cannot represent the core, so the core force is too weak
  (underbound core -> mass leaks outward -> spurious mid/outer halo where true density ~0).
- Not the cause: parent stencil (spectral parent no better), prolongation, ideal band-limited restriction (Fourier still +466% at H=1).
- Cell-mean restriction aliasing DOES cause bulk drift at H=2 (0.88 vs 0.002); mean-force subtraction removes drift, not the weak core force.
- Composite rows (0.0%) are VACUOUS for halo C: all mass lies inside the taper radius, so composite == spectral.
  Real test needs a halo with envelope mass (1D extended halos: composite within ~5%).
- Static radial-force error tracks the evolved error ordering across H (not yet checked across all solvers).
- Practical implication for 3D/GAMER: refine on core/granule scale for GRAVITY so Poisson cell size <~ 0.5 core size.

Next steps (to do another time)
1. 2D composite test with a halo that has envelope mass.
2. Add CQUAD prolongation + ghost-zone density interpolation to the toy (GAMER defaults).
3. Check static-force error as predictor of evolved error across solvers; make it a standalone function for 3D snapshots.
4. Leaf-cell cross-level mean-force sum in GAMER.
5. Compare user's refined-level cell size vs core radius (needs their data).

## Same-grid resolution + discretisation sensitivity (2D halo C, 2 seeds; halo_2d_ppw.py, halo_2d_sens.py, summarize_2d_sens.py)

Wave and gravity on the SAME grid (n=256/192/128/96 -> 8.8/6.6/4.4/3.3 points per wavelength at lambda=1.1), error vs n=256 exact-wave/spectral-gravity.
lambda=1.1: 99% of psi power lies at wavelength > 1.1 (median ~3). Count points per wavelength on the SHORTEST power-carrying wavelength.
Wave symbols: exact (proxy for high-order GRAM-FE), FD2/FD4/FD6; gravity: spectral, FD2/FD4/FD6 Laplacian (converged relaxation on a periodic box).

Outer / mid error (%):
- exact wave + spectral gravity: ok down to 4.4 ppw (mid +2, outer -1); breaks at 3.3 ppw (+48 / +204). Below ~4 ppw nothing works.
- gravity FD2 stencil (exact wave): +11 mid / +28 outer even at 8.8 ppw; +125 / +24 at 4.4. FD4 gravity: +4 / 0 at 4.4. FD6: +1 / -4.
- wave FD2: outer +70 at 6.6 ppw, +405 at 4.4. FD4 wave: +22 at 4.4. FD6 wave: -3 at 4.4 (like exact).
- a 2nd-order piece in either wave or gravity dominates; FD2 gravity + FD6 wave at 4.4 ppw: mid +116 / outer +44.
- time step: insensitive up to dt=0.02 (outer +0.4%), dt=0.04 -> +13%.
Caveats: 2D periodic toy, no AMR/parent BC, SOR tolerance not modelled, GRAM-FE only proxied, 2 seeds with large scatter (non-monotonic entries,
e.g. n=96). 'drift' is identical (0.265) for all dt -> set by initial state, not dynamics; not investigated.
Interpretation: user's 3D need for 10-15 ppw (with GRAM-FE wave + standard SOR gravity) is consistent with the 2nd-order Poisson stencil being the
limiting piece, not the wave solver. Untested: n=384 convergence of FD2 gravity; AMR patch-edge BC; real SOR tolerance.
Suggested tests: (1) FD4 (or compact 4th-order) relaxation stencil on the toy with Dirichlet parent BC; (2) static-force check with the FD2 stencil
on the user's snapshots; (3) two-level patch test with varying patch size.

### GAMER Poisson stencils (read from src/SelfGravity/CPU_Poisson, public master e367975)
- SOR (`CPU_PoissonSolver_SOR.cpp`): the residual is the standard 7-point, 2nd-order Laplacian, sum of 6 neighbours - 6*phi - dh^2*Poi_Coeff*rho, update phi += Omega/6*residual. Iteration stops at Max_Iter, or when the 1-norm residual grows after Min_Iter (a relative tolerance is not used there).
- The initial guess inside the patch is the parent potential interpolated with slope-based quadratic stencils (CQUAD-like).
- Base-level FFT, periodic BC (`FFT_Periodic`): the denominator is -4(sin^2(kx/2)+sin^2(ky/2)+sin^2(kz/2)), the symbol of the same 2nd-order stencil; the spectral -k^2 line is commented out. Isolated BC instead uses a real-space continuum 1/r kernel (`Init_GreenFuncK`), with the self-cell value set by GFUNC_COEFF0.
=> For periodic BC every level, including the base level, is effectively an FD2 gravity solver: the "FD2 gravity" rows of the sensitivity study are the relevant model.

## Two-level patch test: FD2 vs FD4 relaxation with parent Dirichlet BC (2D halo C, 2 seeds; halo_2d_patch.py, summarize_2d_patch.py)
Parent = cell-mean density, periodic FFT with FD2/FD4 symbol; patch around the core solved directly (= converged relaxation), BC from quadratically
interpolated parent potential; wave exact on the fine grid. Error vs n=256 exact/spectral, mid / outer:
- n=192 (6.6 ppw): FD2/FD2 +17..18% / +10%; FD2/FD4 +1% / -2%; FD4/FD4 +1% / -2%.
- n=128 (4.4 ppw): FD2/FD2 +123..125% / +24%; FD2/FD4 +3..4% / -1..0%; FD4/FD4 +4..5% / 0..+1%.
- Patch half-width 4 vs 8: no difference. Parent stencil (FD2 vs FD4): no difference once the patch is FD4.
- FD2/FD2 patch numbers equal the whole-box FD2 numbers of the sensitivity study -> parent Dirichlet BC is not the limiting piece in this toy.
=> Fine-level stencil order is what matters; a 4th-order fine-patch relaxation removes the extra resolution requirement in the toy.
Caveats: 2D, one 2:1 level, patch centred on the core (hw>=4), direct solve (no SOR tolerance), exact wave solver, 2 seeds; FD4 relaxation not tried in a real SOR.
Implementation hazards (untested): FD4 needs 2 ghost layers; red-black SOR is invalid for the +-2 neighbours (same colour as the centre); options are
defect correction (phi += FD2-solve(rho - L4 phi), keeps the existing SOR kernel) or a compact (Mehrstellen, rhs = rho + h^2/12 lap(rho)) stencil.
