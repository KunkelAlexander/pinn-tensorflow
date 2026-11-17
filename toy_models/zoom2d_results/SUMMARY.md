# zoom2d: moving core across a refinement boundary (periodic 2D SP, L=32, T=0.8)

Reference: Fourier Yoshida4 N=256 (checked against N=512 and dt/2, <1e-6). Metric: density L2 error at T unless noted.

## Global adaptive SIPG (one symmetric operator, no separate coarse wave; self-consistent global Poisson)
Finest cell 1.0 (degree p -> p+1 nodes), i.e. finest spacing equivalent to a uniform 256^2 grid at p=7 (65,536 dofs).

| mesh | p | dofs | coupled rho err | profile err | mass drift | energy drift |
|---|---|---|---|---|---|---|
| fine region does NOT cover packet path | 5 | 2304 | 1.3e-1 | 1.2e-1 | 2e-13 | 7e-6 |
| | 7 | 4096 | 1.4e-2 | 1.2e-2 | 6e-12 | 5e-5 |
| fine region covers path | 5 | 3168 | 1.0e-2 | 1.1e-3 | 5e-13 | 1e-6 |
| | 7 | 5632 | 7.7e-4 | 3.3e-5 | 4e-12 | 2e-6 |

- Uncovered error equals the projection error of the coarse mesh where the packet goes (projection 4.9e-2 / 7e-3): the operator is not the bottleneck, the mesh is. Moving the core through a coarse region needs the mesh to follow it (time-adaptive remeshing), which is not tested here.
- Timestep is not the limit: dt = .005/.01/.02 give identical errors (p=5: .1306/.1306/.1307; p=7: .01407/.01406/.01413). Replay with the exact FFT potential changes the error by <4%, so the self-consistent adaptive Poisson is not the limit either.

## Uniform controls (same problem)
- Global FFT: N=48: 2.3e-1, 64: 1.6e-2, 96: 1.2e-4, 128: 1.1e-6. Gram-FE + spectral gravity: N=128 6e-5, N=256 1e-6.
- Gram-FE + local-hierarchy gravity, fine patch covering the core: 128 -> 1.8e-4, 256 -> 1.3e-5 (quartic); crossing the boundary: 3.7e-3 / 1.1e-3 (quartic), 5.0e-3 / 1.5e-3 (quadratic). Halving dt changes nothing (3.7e-3 -> 3.7e-3), so the crossing error is a spatial gravity-boundary error that does not converge with N.
- fd2 gravity (no hierarchy) gives the same error as hierarchy2 covered (4.7e-3 / 1.2e-3): the gravity discretization, not the interface, dominates there.

## Cost vs uniform
- Covered p=7: 5632 dofs, 7.7e-4. Cheapest uniform FFT reaching ~1e-3 is N~80 (~6400 dofs). No dof saving on this 2D problem.
- Versus the uniform grid at the same finest spacing (256^2 = 65,536): 11.6x fewer dofs, but that grid is far more accurate than needed (1e-6), so it is the wrong comparison.
- SIPG wall time (Python, Chebyshev exponential, ~26k kinetic matvecs at p=7) was ~350 s vs seconds for FFT.

## Takeaways
1. A single globally coupled self-adjoint operator removes the interface errors: crossing the refinement boundary costs nothing once the mesh covers the packet (mass and energy conserved to 1e-12 / 1e-6).
2. The problematic error in the Gram-FE hierarchy test is the inherited-boundary gravity: it saturates at ~1e-3 independent of N and dt, while global spectral gravity gives 1e-6 with the same wave solver. A global gravity treatment is the highest-yield next step.
3. No resource saving demonstrated in 2D: the packet fills its refined region, and the FFT reaches 1e-3 on a small grid. A saving needs a problem with a large dynamic range (compact core in a large empty domain); this test does not have one. That test, plus a global adaptive Poisson that scales (FMM or multigrid), is what decides the 32k^3 question.

## Corrections and open questions (after review discussion)
- The stand-in `gravity_hierarchy` (zoom2d.py) already restricts fine mass to the parent source (cell means), as GAMER's restriction does. So the ~1e-3 crossing error is NOT explained by missing upward mass flow. Candidate causes: coarse-grid representation of the compact core near the patch edge, interpolation of the coarse potential to the patch boundary, and no fine-to-coarse potential correction. Not yet separated.
- GAMER's actual Poisson path was not checked or reproduced; only this stand-in was tested.
- Globally coupled gravity does not need all-to-all data communication. Fine levels need halo exchange, and global coupling lives in small coarse levels (multigrid/FAC, latency-bound) or a tree (FMM). Uniform FFT at 32k^3 is the true all-to-all case. Not measured here.
- Candidate next test (not run): method of local corrections (Balls & Colella): local fine solve with infinite-domain BC, coarse solve with a correction charge, fine boundary from the corrected coarse potential. Goal: bring the 1e-3 crossing error towards the 1e-6 of spectral gravity with only local communication plus one coarse global solve.
- Also not run: compact core in a large empty domain (the only case where adaptivity can save resources).

## Follow-up: gravity-only error for bound localised objects (soliton_gravity.py, halo_gravity.py)
Setup: periodic 2D, L=32, G=1, N=256 (h=0.125). The wave solver is exact Fourier Strang on the full uniform grid for every run, so only the potential routine differs. Gravity variants: spectral (reference), fd2, fd4, "coarse rN" (whole domain solved on a grid N times coarser, cell-mean restriction, quartic interpolation back), and "patch rN halfW" (fine Dirichlet FD2 patch of half-width W fine cells, boundary from the coarse FD2 solve interpolated quartically, i.e. the inherited-boundary treatment). Results in soliton_gravity_results/. These are errors vs the same solver with spectral gravity; floors (half dt, 1e-8 perturbed start) are listed.

**Stationary soliton (mass 10, rms radius 1.13, 9 cells): not a sensitive probe.** The exact-gravity control already drifts 4e-4 because my ground state has a residual of 4e-4 (independent of dt: .01/.005/.002 give 4.2/4.1/4.05e-4). fd2, coarse r2/r4 and every patch variant give drifts of the same order (T=1 run with a less polished state: floor 5e-3, all variants 1e-3 to 5e-3; coarse r8 1e-2). No clean separation, consistent with solitons being easy. The initial state was not polished further.

**Bound, contracting halo (mass 80, Gaussian envelope R=2.5 with random smooth phase, theta_rms 1.5, T=8): reproduces gravity-induced density errors.** The halo collapses (peak density 2 -> 21, rms radius ~4.4 at the end). Density L2 error vs spectral gravity, max over t in [0,8] / mean over t / error of the time-averaged radial profile / mean relative error of the central density (r<1):

| gravity | max | mean | mean profile | core |
|---|---|---|---|---|
| floor (half dt) | 2.7e-5 | 1.6e-5 | 1.4e-6 | 3e-6 |
| fd4 on wave grid | 2.0e-4 | 9e-5 | 1.9e-5 | 3e-5 |
| fd2 on wave grid | 1.3e-2 | 6.0e-3 | 6.0e-4 | 1.5e-3 |
| gravity 2x coarser than wave | 2.7e-2 | 1.2e-2 | 1.3e-3 | 3.1e-3 |
| gravity 4x coarser | 1.5e-1 | 6.3e-2 | 1.2e-2 | 1.9e-2 |
| gravity 8x coarser | 1.1 | 5.4e-1 | 7.6e-2 | 1.5e-1 |
| patch r2, half 16/32/48/64 | 2.2e-2/1.4e-2/1.3e-2/1.3e-2 | | 5.6e-4..5.9e-4 | 1.5e-3..1.9e-3 |
| patch r4, half 16/32/48/64 | 5.7e-2/2.3e-2/1.4e-2/1.3e-2 | | 3.2e-3/7.9e-4/5.2e-4/5.5e-4 | 5.4e-3/1.8e-3/1.6e-3/1.5e-3 |
| patch r8, half 16/32/48/64 | 3.9e-1/1.2e-1/4.6e-2/2.1e-2 | | 3.0e-2/4.6e-3/1.2e-3/4.3e-4 | 5.1e-2/7.8e-3/3.3e-3/2.0e-3 |

Readings:
1. The gravity solver alone changes the density of a bound collapsing object by 1e-2 (fd2 on the same grid as the wave) up to order 1 (potential solved 8x coarser than the wave). The wave solver and mass are identical in all runs.
2. Part of this is the Laplacian order, not the hierarchy: fd4 on the same grid is ~60x better than fd2 (2e-4 vs 1.3e-2).
3. An inherited-boundary patch that covers the object well (half 64 = 8 length units vs rms ~4.4) saturates at the fd2 floor. A tight patch (half 16 = 2 length units, inside the halo) with a large parent ratio is much worse (r8: 39% max, 5% core), and the error falls monotonically as the patch grows. So the error depends on where the boundary sits relative to the mass and on how coarse the parent that supplies the boundary potential is.
4. Time-averaged profiles are much more forgiving than instantaneous density: worst non-extreme cases are 1e-3 to 3e-2.
Caveats: this is a transient collapse, not a virialised equilibrium halo; wave grid is uniform and not adaptive; one patch level, parent solved with FD2, quartic interpolation; mass-conserving but not checked for energy. The hot first halo (mass 30, theta 3) was unbound (rms radius 3 -> 11.8) and showed all variants under 1e-3, so unbound/diffuse cases do not discriminate; the result is halo_n256.json. The method-of-local-corrections and large-empty-domain tests are still not run.

## Follow-up 2: virial-tuned halo, method of local corrections, stored profiles
Files: halo_gravity.py (`--suite mlc`), soliton_gravity.py (`kind='mlc'`), plot_halo_profiles.py. Results in soliton_gravity_results/: halo_n256_virial_mlc.json (per-snapshot azimuthal profiles for the reference and every variant, 48 bins of 0.25 to r=12, every 0.1 time units), halo_n256_virial_mlc_final_density.npz (2D density at t=0 and t=16 for every variant), and three figures `halo_n256_virial_mlc_profiles_{grid,patch_r4,patch_r8}.png` (profiles overlaid at t=2/6/10/16 on top, |rho-rho_ref|/peak below).

**Why the first halo was not the right test.** In 2D the virial condition is 2K = G M^2/(4 pi), independent of size. The "bound" halo above had 2K/|W| = 0.63 (sub-virial, hence a collapse). Phase amplitude theta_rms = 1.9 gives ratio 1.0 (mass 80, R=2.5, ell 1.2). T=16.

**Caveat on this halo.** Even at virial ratio 1 it is not stationary: the reference rms radius grows from 3.5 to 7.8 over T=16 (peak density 2 -> 9.9). It is a relaxing, expanding granular halo, not an equilibrium one. So a patch that covers it at t=0 does not at t=16 (half 48 = 6 length units < rms 7.8 at the end), which confounds "patch size" with "time". A truly stationary halo (e.g. a time-averaged or relaxed state, or a soliton+halo ground state) is not yet built.

**Local-corrections variant (`mlc`).** FD2 throughout: fine free-space potential of tapered patch density (FFT convolution with the FD2 lattice Green function, verified: its discrete Laplacian is delta to 1e-14) plus a smooth correction = coarse periodic solve minus coarse free-space solve of the same patch density. Two things were needed to make it work: (i) the free-space coarse term must be the lattice Green function of the same discrete operator (continuum log kernel gave 1e-3 errors); (ii) a smooth taper chi (half/2 wide) splitting the density, otherwise the density jump at the patch edge leaves an error the coarse grid cannot resolve; plus evaluating the coarse free term on a grid extended 2 cells beyond the periodic seam (its kink at the seam otherwise broke the periodic interpolation, error 0.08 at the box edge). Static potential error vs fd2 on this halo falls monotonically with patch size (r4: 3.9e-4 / 2.3e-4 / 1.5e-4 for half 16/32/48; r8: 2.0e-3 / 1.0e-3 / 6e-4), coarse-only r8 is 1.9e-3.

**Dynamic results (virial halo, density L2 error vs spectral; max over t / mean-profile error / core error):**

| gravity | max | mean profile | core |
|---|---|---|---|
| floor (half dt) | 3.2e-5 | 1.4e-6 | 3.6e-6 |
| fd4 | 3.0e-4 | 8.9e-6 | 5.0e-5 |
| fd2 | 2.5e-2 | 5.2e-4 | 2.5e-3 |
| coarse r4 | 3.3e-1 | 1.3e-2 | 4.0e-2 |
| coarse r8 | 1.1 | 5.8e-2 | 1.8e-1 |
| patch r4 half 16/32/48 | 2.8e-1 / 9.1e-2 / 4.0e-2 | 9.8e-3 / 2.0e-3 / 9.1e-4 | 4.3e-2 / 6.4e-3 / 4.2e-3 |
| mlc r4 half 16/32/48 | 3.0e-1 / 1.1e-1 / 5.6e-2 | 1.1e-2 / 2.6e-3 / 9.9e-4 | 4.1e-2 / 1.1e-2 / 2.5e-3 |
| patch r8 half 16/32/48 | 1.0 / 6.5e-1 / 2.2e-1 | 5.3e-2 / 2.0e-2 / 5.9e-3 | 1.7e-1 / 5.1e-2 / 1.8e-2 |
| mlc r8 half 16/32/48 | 1.1 / 5.2e-1 / 1.1e-1 | 3.7e-2 / 9.9e-3 / 3.9e-3 | 1.5e-1 / 6.2e-2 / 2.0e-2 |

Readings:
1. The halo again shows large gravity-induced profile errors, and they are larger here (virial, relaxing) than for the collapse: coarse r4 gives 33% max density error and r8 order 1, over 16 dynamical times, with identical wave solver. fd4 stays at 3e-4. In the profile plots the coarse r8 profile has a visibly wrong core and shoulder by t=10-16.
2. Local corrections give a modest, not dramatic, gain over the inherited boundary: mean-profile error improves 1.4-2x at r8 (e.g. half 32: 9.9e-3 vs 2.0e-2) and max error at r8 half 48 halves (0.11 vs 0.22), but it is no better, sometimes slightly worse, at r4 and for core error. A boundary scheme cannot fix mass that lies outside the patch and is only resolved at the coarse level; the error is dominated by how much of the halo sits on the coarse grid, and with the halo expanding to rms 7.8 that is a growing share of it. Accuracy approaches the fd2 floor only when the fine region covers the halo.
3. Implication for the design: the gravity error is controlled by where the mass is resolved, not by boundary-condition cleverness. A hierarchy has to keep the whole bound object (the halo, not just the core) at the fine level, or have intermediate levels. This is cheap for compact cores but not for extended halos.
4. fd4 beats fd2 by ~80x at the same grid, so the Laplacian order is the cheapest lever and it is separate from the hierarchy. The local-corrections implementation is FD2 only; a fd4 lattice version was not built.

Still not done: stationary halo; fd4 local corrections; multi-level patches with intermediate refinement; adaptive wave grid; energy conservation; compact object in a large empty domain.

## Follow-up 3: a verified stationary halo, long-time tests, and the cause of the gravity error (stationary_halo.py, halo_long.py, static_force.py)

**Correction to Follow-up 2.** The "virial-tuned" halo was not just non-stationary, it was *drifting*: its random phases carried a net velocity (-0.29, +0.30), so the halo walked to (-4.5, +4.6) by T=16 while the fine patch stayed at the origin. Its rms about its own centroid grew only 3.5 -> 4.4 (not 3.5 -> 7.8). The patch-size dependence reported there is therefore largely "the halo leaves the patch", and the statement that the error is controlled by mass outside the fine region was over-stated. The fd2/fd4/coarse numbers remain valid measurements on that (drifting) object, but it should not be used as the halo test.

**What is compared (all runs):** the wave solver is identical everywhere (exact Fourier Strang kick-drift-kick on the uniform N=256 grid); only the Poisson routine differs (spectral reference; fd2/fd4 on the wave grid; coarse-grid solves with different restriction/interpolation; inherited-boundary patch; local corrections; composite-grid Galerkin). So this isolates *gravity discretisation*. It does not test an adaptive wave grid; in an AMR code the coarse regions also carry the wave on the coarse grid, so sub-cell granule aliasing is smaller there than in this toy (see caveats).

### Stationary halo (stationary_halo.py, analyze_stationary.py)
A single wave function has no stationary |psi|^2 except the soliton, so a stationary halo is a random-phase superposition of eigenstates of its own mean-field potential, with a lowered-isothermal occupation f(E) = c[exp(-(E-E0)/T) - exp(-(Ec-E0)/T)], Ec = Phi(r_c), iterated to self-consistency with the same periodic spectral potential the wave runs use (radial eigenproblem per angular momentum m). Net momentum is removed with a Galilean boost and the centroid shifted to the origin (random-phase halos otherwise translate).
- Checks of the solution: mean-field 2K/(M^2/4pi) = 0.95 (halo A), 0.92 (B); the deficit from 1 is the periodic neutralising background, verified numerically; residual 1e-7; the eigenstates stay stationary in a fixed potential (rms 1.8 +- 0.1).
- Halo A (T=5, r_c=8, M=80): 84 states, rms 2.76, r50 2.2, r90 4.3, peak 4.2; profile falls sharply beyond r~8 (truncated).
- Halo B (T=4, r_c=12): 167 states, rms 3.45, r50 2.5, r90 5.5, rho from 3.3 to 7e-5 at r=12 (extended envelope).
- Stationarity under spectral gravity (T=20, three seeds for A): mass drift 1e-12, energy drift 1e-5, momentum 1e-9, rms drifts +2-4% (fluctuation 1.5-2%), 2K/(M^2/4pi) oscillates 0.84-1.08 about its mean-field value, bulk profile (rho>1e-2) within the granule noise, core density +-20-30% (granules). Over T=60 there is a slow *real* relaxation (A: rms 2.80 -> 3.00, core +40%); halo B: rms +6% over T=40, tail (r>6) grows 10-40%. Treat the halos as stationary for T <~ 25 and secularly relaxing beyond; compare methods to the spectral run, never to the initial state. A T=4, r_c=6 variant builds a growing soliton core (+26%) and was rejected; T<=3 sits near a core-collapse bifurcation.
- Realisation scatter: the T=16 mean profile differs by 8-37% (mostly the core) between random-phase seeds of the same halo, so differences below that in a single halo are not statistically meaningful in the core.

### T=16 suite (halo A: 2 floors + 22 methods, halo_n256_stationary.json; halo B at T=24: halo_n256_stationaryB.json)
Pointwise density error is a poor metric here (granule phases dephase); the time-averaged azimuthal profile is the right one. Relative error of the time-averaged profile (t>=4), halo A: fd2 <=0.3%, fd4 <0.1%, coarse r2/r4 <=1.6%, coarse r8 core -29%, mid +5%, outer +7%; patch r4 (any size) <=0.5%; patch r8 half16 outer +2-4%, half>=32 <=0.6%; mlc similar. At T=16 the toy does **not** show a mid/outer-halo mismatch for anything except coarse r8: the errors need longer times to accumulate.

### Long-time tests (T=48, halo A; T=40, halo B): the 3D-like symptom appears
Late-time (t>=24) relative error of <rho>(r) versus spectral, profile about the grid origin:

| method (halo A) | centroid drift | net abs(p) | core <1 | mid 1-3 | outer 3-6 | far 6-10 |
|---|---|---|---|---|---|---|
| fd2 / fd4 | 0.001 | 2e-7 | -6% / 0 | +1% / 0 | -1% / 0 | +2% / 0 |
| coarse r2 (cell mean + Lagrange) | 0.001 | 1e-4 | -3% | +1% | -0.5% | +2% |
| coarse r4 (cell mean + Lagrange) | 1.15 | 2.1 | -16% | +3% | +8% | +27% |
| coarse r8 (cell mean + Lagrange) | 3.25 | 21 | -90% | -49% | +165% | +480% |
| coarse r4 / r8, TSC pair + FD2 | 0.10 / 0.09 | 0.07 / 1.2 | -13% / -22% | +8% / +6% | +0.5% / +9% | -13% / -31% |
| patch r8 half32 / half48 (inherited) | 0.90 / 0.54 | 2.4 / 1.7 | -16% / -13% | +5% / +3% | +4% / +0.4% | +10% / +3% |
| mlc r8 half32 / half48 | 0.22 / 0.03 | 0.8 / 0.2 | -7% / -4% | 0 / +1% | +1% / -1% | 0 / +3% |
| Galerkin r8 half32 / r8 half48 / r4 half32 | 0.05 / 0.000 / 0.001 | 0.14 / 2e-3 / 3e-3 | -5% / -6% / -6% | +1% | -1% | +1-2% |

Halo B shows the same ordering with smaller amplitude (coarse r8: centroid 1.05, outer +6%, far +7%; patch r8 half32: abs(p) 1.6; Galerkin half32/half48: abs(p) 0.03/0.02, profile identical to fd2 within noise).

**Mechanism 1: spurious self-force.** The wave kick conserves momentum only if the rho -> phi map commutes with translations, which no fixed coarse grid does. The net self-force F_net = -int rho grad(phi) is exactly zero for spectral/fd2/fd4 and a grid-phase-dependent non-zero value otherwise. Static test (static_force.py, seconds per method, no wave evolution): on the *smooth* mean halo the net force is tiny for plain coarse grids (2e-5 at r4) but large for patches (r8 half32: 6e-2, which times T=48 reproduces the measured abs(p)=2.4); on actual *speckled* snapshots the plain coarse grid is far worse (r4: 0.5-1.0, r8: 1.7-3.2) because sub-cell granule structure aliases through the cell-mean restriction and Lagrange interpolation. It acts immediately (abs(p) saturates within ~1 time unit) and the halo then drifts (centroid up to 3-9). **Most of the late-time mid/outer-halo "error" of these schemes is that drift**: measured about each run's own centroid (analyze_centred.py), the outer/far errors fall from +165%/+480% to -1%/+12% (coarse r8), from +8%/+27% to -1%/+8% (r4), and for patch r8 half32 from +4%/+10% to -1%/+1%. Symmetrising the operator (CIC/TSC restriction = transpose of the interpolation, or a Galerkin solve on the coarse span alone) does *not* remove the drift (Galerkin r8 coarse-only: abs(p) 7.9).

**Mechanism 2: force softening.** Replacing cell-mean + Lagrange by the smooth quadratic B-spline (TSC) pair removes the drift (centroid 0.1) but low-pass filters the density, softening gravity: the profile is distorted *without* drift (core -13 to -22%, mid +6-8%, far -13 to -32%, unchanged when measured about the centroid). This is a genuine mid/outer-halo profile bias, of opposite sign to the drift error.

**Composite-grid Galerkin patch (new, Gravity kind 'galerkin').** phi lies in span{coarse B-splines not wholly inside the patch} + {fine nodal functions in the patch}; solve (B^T L_fine B) u = B^T rho with a sparse LU (bordered for the constant mode). There is no interpolated boundary. With a fine patch large enough to hold the granular halo core (here half 32-48 fine cells = 4-6 length units, r ~ r90-r99) the net self-force on speckled density drops to 5e-4-9e-4 (inherited patch 0.06-0.6, mlc 0.03, coarse-only 0.2-3), the halo stays centred in the dynamics, and the profile matches the fd2 run, i.e. it reaches the floor of the second-order fine operator (fd2 itself is -6%/+1%/-1%/+2%; fd4 would remove that). It is the only treatment tested that fixes both mechanisms. The coarse-only Galerkin (no patch) still drifts, so the benefit comes from resolving the granular region at the fine level and coupling it variationally, not from symmetry alone.

### Caveats and open points
- The wave grid is uniform and fine everywhere; in AMR the coarse region's wave grid also cannot hold sub-cell granules, so the aliasing self-force found here is probably an upper bound for that region; conversely wave-solver errors at the interface (not tested here) are an additional possible source of 3D mismatch. Whether the 3D mismatch is the drift type or the softening type is not established by this toy; the relative-error-vs-log-r figures can be compared with the 3D plots directly.
- 2D; one realisation per halo for the long runs (seed scatter 8-37% in the core), one mass, L=32 periodic box.
- The Galerkin solver is a global sparse direct solve (build ~17 s, solve ~65 ms at N=256): it demonstrates the target discretisation, not a scalable implementation. The composite system is symmetric (semi-)definite, so CG with a multigrid/FAC preconditioner would be the route to scale.
- fd2 on the fine grid has its own few-percent profile error; the fine-level Laplacian order is a separate cheap lever (fd4: <0.1%) not yet built into the Galerkin patch.
- Halo A's tail is truncated at r~8 and halo B's tail slowly grows, so far-halo numbers (r>6, rho<1e-2) are the least certain.

### Files
stationary_halo.py (construction + spectral check), analyze_stationary.py, halo_gravity.py (--suite stationary --psi0), halo_long.py (T=48 recorder: momentum, centroid, energy, rms, profiles about origin and centroid, contrast), analyze_long.py, analyze_centred.py, static_force.py, plot_halo_relative.py (log-r relative error), plot_halo_profiles.py. Data and figures are in soliton_gravity_results/: stationary_halo_n256_*.json/png, halo_n256_stationary*.json, halo_n256_long*.json, halo_n256_long_all_summary.png, halo_n256_long_all_relative_logr.png, halo_n256_longC_centred.png, static_force_A.json/png. New options in soliton_gravity.py: `kind='galerkin'` and `adjoint=1|2` (CIC|TSC).

## Follow-up 4: minimal test problem (minimal_selfforce.py, minimal_soliton_drift.py, minimal_1d.py)

Goal: distil the mechanism of Follow-up 3 (spurious self-force of a fixed coarse Poisson grid on granule-scale density) into the smallest problem. Only spectral gravity vs coarse fd2 (cell mean + quartic Lagrange) is compared; the wave solver is the exact Fourier Strang everywhere.

**Control parameter: sigma/H**, the width of a lone granule/soliton over the coarse cell. A lone soliton at rest is stationary under exact gravity (momentum conserved to 1e-11); under a coarse grid it has a net self-force that depends on the sub-cell offset.
- Static (seconds, no evolution), 2D Gaussian blob, |F_net| / (M^2 / 2 pi sigma): coarse r8 1.7e-1 (sigma=H/4), 3e-2 (H/2), 1.6e-5 (H), <1e-9 (1.5 H); TSC pair about 5-10x smaller but non-zero; pair force: TSC softens by 40% at d=H/2, 10% at 2H, cell-mean+Lagrange is within 3% for d >= H.
- 2D dynamic (T=4, N=256, max over 4 sub-cell offsets, spectral: displacement 0, |p| 1e-11): coarse r8: mass 20 (sigma 0.57 H) displacement 0.55, |p| 1.0; mass 40 (0.40 H) 0.33, 8; mass 80 (0.28 H) 0.31, 60 (peak -26%); mass 160 (0.20 H) 0.92, 160 (peak -49%); mass 8 (0.9 H) 0.007. Coarse r4: mass 40 (0.8 H4) 0.18, mass 160 (0.4 H4) 0.26 with |p| 118. Each mass needs ~10 min in 2D (ground state + 9 evolutions).
- **1D works** (1D SP has solitons; the mechanism is dimension independent; N=2048, L=32, seconds per case): coarse r32 (H=0.5): static |F_net|/(M^2/2) = 6e-9 (sigma 1.6 H), 2e-6 (1.2 H), 1e-4 (1.0 H), 2e-3 (0.77 H), 1.4e-2 (0.61 H), 5.6e-2 (0.48 H); dynamic displacement after T=4 up to 0.23 (non-monotonic: the soliton oscillates in the spurious potential); coarse r16 stays clean for all sigma tested (sigma >= 1 H16). Threshold: the self-force turns on steeply below sigma ~ H.

**Takeaway.** Net self-force (hence momentum non-conservation and halo drift) is controlled by sigma/H and offset, not by halo shape; the static test is the cheap screen (milliseconds in 1D). The lone soliton reproduces mechanism 1 (drift) but not the profile of a halo; a 1D stationary halo (random-phase eigenstates, as in stationary_halo.py) is the next step to get a mid/outer-halo profile error.
Files: minimal_selfforce.py, minimal_soliton_drift.py (+ _20/_40/_80/_160 json/log), minimal_1d.py, soliton_gravity_results/minimal_selfforce.png, minimal_1d.png.

## Follow-up 5: 1D stationary halo (spectral vs coarse fd2), 2026-11-04

Files: `halo_1d.py` (build + run), `analyze_halo_1d.py` (plot/table), `soliton_gravity_results/halo_1d_M64{.json,.log,.png,_table.txt}`.

Setup: L=32, N=1024 (h=0.031), M=64, lowered-isothermal T=32, r_c=10; random-phase superposition of the 78 bound eigenstates of the
self-consistent mean-field potential (exact Fourier kinetic operator). Same exact Fourier Strang wave solver for all runs; only the
Poisson routine differs (spectral vs coarse cell-mean restriction + quartic Lagrange + fd2 parent solve, H = 0.125/0.25/0.5/1.0).
t=0..60, profile = centroid-following, +/-x symmetrised, averaged over t in [40,60], 4 seeds. Noise floor = spectral run with a 1e-6
perturbation of psi0. Whole study (4 seeds x 6 runs) takes ~3 min.

Median |<rho>/<rho>_spec - 1| over seeds (core |x|<1 / mid 1-4 / outer 4-10) and max |centroid| per seed:

| run | core | mid | outer | max abs centroid |
|---|---|---|---|---|
| spectral perturbed (floor) | 0.026 | 0.030 | 0.007 | 0 |
| coarse H=0.125 | 0.100 | 0.087 | 0.169 | 0.02-0.03 |
| coarse H=0.25 | 0.121 | 0.088 | 0.107 | 4.7, 4.2, 0.18, 1.5 |
| coarse H=0.5 | 0.229 | 0.158 | 0.616 | 6.5-11.7 |
| coarse H=1.0 | 0.194 | 0.132 | 0.981 | 11.1-11.7 |

Takeaways: (1) the mismatch reproduces in 1D at ~3 min cost; errors are 3-30x the noise floor. (2) Coarse H>=0.25 makes the whole halo
wander (centroid sloshing/drifting by many core radii), the bulk-drift mechanism of Follow-up 4 acting on many granules; H=0.125 does not
drift but still gives ~10% core and ~17% outer-profile error. (3) Caveats: only 4 seeds; this halo is compact (rho falls from ~10 to 1e-3 over
|x|=0..8, so the 'outer' bins sit in a steep truncated tail where relative errors are inflated); error sign is not monotonic in H;
wave grid fine everywhere; no fix (patch/Galerkin) has been tested in 1D yet.

### Follow-up 5b: larger box (L=128, N=2048, h=0.0625), same halo family

`halo_1d.py` now takes `HALO1D_L`/`HALO1D_N` env vars; the self-consistent solve runs on a 1024-pt internal grid and Fourier-interpolates eigenstates
(valid while E_c < ~300, i.e. k_max of that grid). M=64, T=32, r_c=40: 528 states, tail to |x|~40. 6 seeds, t=0..80, average over [50,80], dt=.004,
coarse H = 0.125/0.25/0.5/1.0. Files: `soliton_gravity_results/halo_1d_L128{.json,.log,.png,_table.txt}`. ~25 min total.

Median |<rho>/<rho>_spec - 1| (core / mid 1-4 / outer 4-10 / far 10-40) and max |centroid| over seeds:

| run | core | mid | outer | far | max abs centroid |
|---|---|---|---|---|---|
| spectral perturbed (floor) | .053 | .051 | .029 | .008 | 0 |
| coarse H=0.125 | .067 | .100 | .048 | .016 | 0.02-0.18 |
| coarse H=0.25 | .038 | .072 | .079 | .024 | 2-12 |
| coarse H=0.5 | .096 | .126 | .174 | .124 | 6-58 |
| coarse H=1.0 | .217 | .234 | 1.18 | 3.86 | 20-58 |

Reading: with the larger box the non-drifting case (H=0.125) is within ~1-2x the noise floor, i.e. NO clear systematic density bias for a stationary halo;
the dominant failure is bulk drift (H>=0.25; H>=0.5 flies across the box, H=1.0 also builds a spurious far halo, +390%). Caveat: the density profile
is still Gaussian-like (flat core to |x|~1 then ~6 decades fall by |x|~25), not a mid-halo power law, because lowered-isothermal occupations give exp(-Phi/T) tails.
