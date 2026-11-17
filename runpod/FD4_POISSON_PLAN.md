# Plan: 4th-order Poisson solver in GAMER

Status (2026-11-15): implemented in the fork (option A + FFT symbol), built on the pod for GPU and CPU, unit-tested; production
LSS_Hybrid test with `POT_STENCIL_ORDER 4` not yet run (the pod is busy with the timed light-halo run). Results in section 6.
Source read: gamer-project/gamer commit e36797538 (the commit on the pod).
Motivation: `toy_models/RESEARCH_DIRECTIONS.md`, sections "Same-grid resolution + discretisation sensitivity" and
"Two-level patch test": in the 2D toy, a 2nd-order gravity stencil dominates the mid/outer-halo error (+11/+28 % at 8.8 ppw),
and a 4th-order fine-patch stencil removes it. That is 2D evidence only. Whether it carries over to 3D GAMER is the thing to test.

## 1. How GAMER solves Poisson today (verified in the source)

| Level | Solver | File | Discretisation |
|---|---|---|---|
| 0 (base grid, 64^3 in LSS_Hybrid) | FFT, periodic | `src/SelfGravity/CPU_Poisson/CPU_PoissonSolver_FFT.cpp`, `FFT_Periodic` | divides by `Deno = -4(sin^2(kx/2)+sin^2(ky/2)+sin^2(kz/2))`, the symbol of the 7-point FD2 Laplacian. The spectral `-k^2` line is commented out. |
| >= 1 | SOR, one 8^3 patch at a time | GPU: `src/SelfGravity/GPU_Poisson/CUPOT_PoissonSolver_SOR.cu`; CPU twin: `CPU_PoissonSolver_SOR.cpp` | 7-point FD2 residual (line 379), red-black SOR |

Details of the SOR path that matter for the change:

- Each patch works on an `18^3` array (`POT_NXT_F = PS1 + 2*POT_GHOST_SIZE`, `POT_GHOST_SIZE = 5`, hard-wired to 5 for the GPU solver by `CUAPI_SetDevice.cu:188`).
- Initial guess and Dirichlet boundary condition both come from the parent-level potential, interpolated with `INT_CQUAD` (`OPT__POT_INT_SCHEME 4` in the LSS_Hybrid input). The interpolation fills the whole 18^3 array (step b in the kernel).
- The density array is `RHO_NXT = PS1 + 2*RHO_GHOST_SIZE = 16^3` (`RHO_GHOST_SIZE = POT_GHOST_SIZE - 1`). So SOR updates only the inner 16^3 cells and the outermost one-cell layer is frozen as the boundary.
- Output is `GRA_NXT^3`: 8^3 for ELBDM unless `STORE_POT_GHOST`, then 12^3. ELBDM uses the potential directly (`GRA_GHOST_SIZE = 0`), so there is no force-gradient stencil to upgrade. The Poisson stencil is the only gravity discretisation in the loop.
- SOR parameters are set in `Init_Set_Default_SOR_Parameter.cpp`: `Omega = 2/(1+sin(pi/NCell))`, `Max_Iter = 40+3*NCell` (float), `Min_Iter = Max_Iter/5`. The loop stops when the 1-norm residual *increases* (kernel line 418).
- A multigrid solver also exists (`configure.py --poisson=MG`, `CPU_PoissonSolver_MG.cpp`, `CUPOT_PoissonSolver_MG.cu`). It is not used by default. Leave it unsupported (abort if the new option is on).
- Plumbing for a new run-time parameter follows `SOR_OMEGA`: `Global.h`, `Init_Load_Parameter.cpp`, `Main.cpp`, `InvokeSolver.cpp`, `Aux_Check_Parameter.cpp`, `Aux_TakeNote.cpp`, `Output_DumpData_Total_HDF5.cpp`, `Init_ByRestart_v2.cpp`.

## 2. Options for the FD4 patch solver

Stencils (1D symbol, theta = k*h, s = sin^2(theta/2)): FD2 `-4s`; FD4 13-point `-4s - (4/3)s^2`; both agree with `-theta^2` to the stated order.

| Option | What changes | Pros | Cons |
|---|---|---|---|
| A. Deferred correction on the existing kernel (recommended first) | Solve `L2 phi = f + (1/12) delta4(phi_old)`, with `delta4` the 4th difference summed over the three axes. Keep the red-black FD2 sweep unchanged; refresh the correction term every few sweeps. | Small kernel change, red-black stays valid (only +-1 neighbours in the sweep), converges to the FD4 solution. The correction operator needs +-2 neighbours, which exist in the 18^3 array. | Needs 2 frozen boundary layers (update region 14^3 instead of 16^3; the 8^3 output is still covered, even the 12^3). Extra cost of the correction passes. The residual-growth stop must be reset when the correction is refreshed. |
| B. Mehrstellen 19-point compact stencil | `L19 phi = rho + (h^2/12) lap(rho)` | 3x3x3 footprint, needs only 1 boundary layer for phi. | Edge neighbours share the centre's colour, so plain red-black is invalid; needs an 8-colour sweep, which means rewriting the tuned kernel (padding, shuffle reduction). The `lap(rho)` term needs one more density layer than we have, so the update region shrinks to 14^3 anyway. Smoothing of granular rho by the `h^2/12 lap(rho)` term has to be checked. |
| C. Direct 13-point SOR | replace the residual line | simplest to write | The +-2 neighbours have the same red-black colour as the centre, so parallel updates race (non-deterministic). Not recommended. |

Contraction of option A per correction refresh: the ratio of symbols `(s^2/3)/s = s/3 <= 1/3` in the worst (Nyquist) mode and ~`theta^2/12` for resolved modes. 2-3 refreshes are enough for resolved physics.

Base level: replace the FFT denominator by `-4*sum_i (s_i + s_i^2/3)`. Single-line change, FD4 symbol (not spectral, so that base and patch levels use the same operator). The isolated-BC path (`Init_GreenFuncK`) uses a continuum kernel and is not touched; LSS_Hybrid is periodic.

New input parameter `POT_STENCIL_ORDER` (2 = current behaviour, default; 4 = new), passed to the GPU kernel as an argument, so one binary can A/B the two without rebuilding.

## 3. Validation, in order

1. **No GAMER change needed: size of the effect on real 3D data.** On the light-halo snapshot at a=1 (the profile run is writing it), cut out a sub-box around the halo at the finest level from yt, take its `Dens`, and solve Laplace-Poisson with the FD2 operator (must reproduce the snapshot `Pote`, which validates the setup), the FD4 operator and a spectral operator, with Dirichlet data from the snapshot `Pote` on the boundary. The FD2-FD4 difference tells whether the stencil matters at the resolution GAMER used.
2. **Unit test of the patch solver.** A manufactured density (Gaussian, analytic potential) on a uniform grid, one refinement level, compare the error of FD2 vs FD4 against h. Expect slope 2 vs 4. Use the CPU build as an oracle for the GPU kernel (same inputs, results agree to roundoff).
3. **Convergence of SOR with the correction**: iterations to reach the same residual vs the FD2 case; check that the stopping rule does not exit early.
4. **Production test**: LSS_Hybrid light halo, same ICs and refinement, `POT_STENCIL_ORDER 2` vs `4` at the same resolution; then the wave-vs-gravity resolution scan (vary `MAX_LEVEL`/wave resolution and gravity stencil independently). This is the experiment that answers the open question.

## 4. Known risks

- The 3D benefit is unproven; the toy was 2D, periodic, single refinement level, direct solve, exact wave solver, 2 seeds.
- Interface errors are not changed by FD4 alone: the boundary comes from quadratic interpolation of the parent potential (`INT_CQUAD`). The toy said this was not limiting, GAMER has not been tested.
- Higher-order stencil + the density that is only piecewise smooth at refinement boundaries: the fine-grid density in the ghost zone comes from interpolation.
- Runtime: extra correction passes in shared memory, probably 1.3-2x in the Poisson step. The Poisson fraction of the total run time on the 4090 has not been measured.
- Single vs double precision: the correction term subtracts nearly equal numbers; check with `FLOAT8` if results look noisy.

## 5. Where the code lives (decision)

Fork https://github.com/KunkelAlexander/gamer, branch `fd4-poisson` (base e36797538), local clone `C:/Users/TE/Desktop/Privat/Projects/gamer`, the pod
pulls from the fork. Details in `runpod/README.md` ("GAMER source and the 4th-order Poisson solver").

## 6. What was implemented and what the unit tests show (2026-11-15)

Commits on `fd4-poisson`: `POT_STENCIL_ORDER` parameter; FD4 symbol in `FFT_Periodic`; SOR deferred correction in
`CUPOT_PoissonSolver_SOR.cu` and `CPU_PoissonSolver_SOR.cpp` (3 refreshes of the correction after the first FD2 solve; outermost layer of the
density array frozen as extra boundary); CPU residual diagnostic (`GAMER_POT_DEBUG=1`); Plummer and periodic-sine profiles for the Hydro/Gravity test
(`Gra_DensProf 3` and `4`). Order 2 is unchanged.

Checks that passed:
- **GPU == CPU**: the same Plummer case (`Gra_DensProf 3`, r0=1, Dens0=3e4, `OPT__POT_INT_SCHEME 5`) gives potentials that agree to 1.5e-4 of the maximum error for
  both orders (max |GPU-CPU| 0.05 of 316), so the GPU kernel (padding, shared-memory correction) matches the plain CPU reference.
- **The solver solves the operator it should**: CPU `GAMER_POT_DEBUG` residuals, relative to max|f|: order-2 solution has |L4 phi - f| 8e-3 (level 4) and |L2 phi - f| 1e-4;
  order-4 solution has |L4 phi - f| 1.5e-4 and |L2 phi - f| 8e-3. (The FD4 residual is 5e-4 at worst, not roundoff, so 3 refreshes are not fully converged; 400 SOR
  iterations change nothing, so the limit is the number of refreshes, not SOR convergence. Untested: more refreshes.)
- **Periodic sine wave** (`Gra_DensProf 4`, `OPT__BC_POT 1`, 64^3 base, lambda=50 = 32 cells, levels L0-L1, default `POT/RHO_INT_SCHEME 4`): max potential error / amplitude
  drops from 2.0e-3 (FD2) to 2.3e-4 (FD4) on L0 and from 2.2e-3 to 2.8e-4 on L1, about 9x. For lambda=25 (16 cells) L0: 2.8e-3 to 1.1e-3.

Caveats found (these decide whether FD4 helps the real problem):
- **Isolated-BC base level is not an FD operator.** In the isolated Plummer/Hernquist tests the L0 potential (FFT with the continuum kernel) has an FD2-like error
  `-(h0^2/12)*4*pi*G*rho` (measured -2240 at the centre vs -2380 predicted). Fine patches take their boundary from the parent, so this smooth error is inherited and
  FD4 on the refined levels gives no gain (even slightly worse, e.g. finest-level error 19 -> 316 for r0=1). `POT_STENCIL_ORDER` does not change L0 in that case.
  LSS_Hybrid is periodic, so its L0 does get the FD4 symbol.
- **General mechanism**: the error of any coarse level in a smooth region reaches the finer levels through the Dirichlet boundary of every 8^3 patch (there is no
  fine-to-coarse correction of the potential). Resolution/stencil errors of the coarse levels in the halo are therefore inherited.
- **Interpolation schemes matter.** With `OPT__POT_INT_SCHEME`/`OPT__RHO_INT_SCHEME 5` (point-value quadratic) FD4 was worse than FD2 on the sine test (3.3e-3 vs 9e-4 on L1),
  with 4 (conservative, cell-average semantics) it was 9x better. A plausible reason (not isolated): restricted coarse densities are cell averages while scheme 5 treats them
  as point values, an O((kh)^2/32) mismatch per axis. Keep 4 for FD4 until this is tested.
- Plummer isolated tests are dominated by the above; they are not a clean convergence test. Hernquist cusps are under-resolved at every level.
- Not tested: LSS_Hybrid itself, FLOAT8, MPI, runtime cost of the correction passes.

Reproduce (pod): `bash setup_gamer.sh gravity` (GPU) or `gravity_cpu`; copy `/workspace/gamer/example/test_problem/Hydro/Gravity`, edit `Input__Parameter`
(`OMP_NTHREAD`, `OPT__BC_POT 1` and `OPT__BC_FLU_* 1` for the sine test, `MAX_LEVEL 2`, `POT_STENCIL_ORDER 2|4` after `SOR_MIN_ITER`) and `Input__TestProb`
(`Gra_DensProf 4`, `Gra_Radius0 50`, `Gra_Dens0 0.04`; the density thresholds in `Input__Flag_Rho` are x4 per level, so a large `Gra_Dens0` exceeds MAX_PATCH). The run takes
seconds; errors are in `PotError.txt` (column 8 = numerical - analytic potential along the diagonal, one block per level). `analyze_plummer.py` here summarises the Plummer runs.
