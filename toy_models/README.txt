Exploratory one dimensional Schrodinger Poisson experiments
=========================================================

Newest round: frozen_gravity.html tests five versus ten cells per wave with an
exact frozen coherent 2D density and six nested Poisson levels. It also provides
a small long-run kinetic failure with an exact prescribed potential, verified
by repeated FFT/MATMUL runtime updates. New outputs: frozen_gravity_results/.
The report lists commands and limitations; previous outputs are preserved.

Previous round: gamer_diagnosis.html reconstructs the pinned actual Gram-FE tables,
tests same-level propagation and long-run mode growth, couples them to SP,
isolates temporal ghost interpolation, and adds a cheap 2D six-level Poisson test.
The report lists all reproduction commands. New outputs use gramfe_results/;
fetch_gramfe_sources.py obtains missing pinned tables, while original outputs stay
unchanged. This is a controlled audit, not a compiled GAMER halo reproduction.

Latest round: scale_communication.html tests translated wave/interface problems,
uniform coarse/fine controls, genuine monochromatic interface scattering,
full-density versus coarse-wave gravity sources, and fully coupled SP with a
moving core and incoming coherent wave. All new outputs use communication_results/.

  & 'C:\Users\TE\anaconda3\envs\pinn_env\python.exe' toy_models/scale_communication.py
  & 'C:\Users\TE\anaconda3\envs\pinn_env\python.exe' toy_models/build_communication_report.py

Use --only linear, source, coupled, controls, analytic, scattering, time_audit, or plots for
individual controls. The coupled dense accuracy oracle takes several minutes.
No new experiment is a claim of 3D performance or a diagnosis of GAMER.

Start with review.html for the critical follow-up, then report.html for the first
round. Both embed their plots and can be opened directly in a browser. The review
qualifies the earlier Fourier-conditioning claim and downgrades fixed reference
Hamiltonians after a moving-core counterexample.
All scripts and outputs are isolated in this directory. No original benchmark or
figure is imported or overwritten. No training data or learned model is required.

Run from the repository root in PowerShell:

  $py = 'C:\Users\TE\anaconda3\envs\pinn_env\python.exe'
  & $py toy_models/experiments.py
  & $py toy_models/representations.py
  & $py toy_models/frozen_hamiltonian.py
  & $py toy_models/verify.py
  & $py toy_models/build_report.py

Follow-up experiments (separate outputs; do not regenerate the first report):

  & $py toy_models/conditioning_review.py
  & $py toy_models/moving_core_review.py
  & $py toy_models/shared_basis_review.py
  & $py toy_models/build_review.py

conditioning_review.py compares weighted global/local sampling spaces, constructs
an unresolved wave pair with different future halo density, and evaluates exact
Gaussian-prior optimal predictors. It does not train a neural network.
moving_core_review.py checks fixed-reference splitting against independently
translated and Galilean-boosted stationary SP states.
shared_basis_review.py uses one conforming high-order mass/stiffness pair for
manufactured gravity and a moving free wave on a graded mesh, with a uniform
Fourier baseline. It does not yet couple them into nonlinear SP.
review_results/ stores the new JSON, three figures, and primary-source provenance.
The follow-up contains 22 numerical assertions, including parameterized checks.

Dependencies: NumPy, SciPy, Matplotlib; already in pinn_env. No extra installation.
The complete suite usually takes a few minutes on this computer. Timings vary.
Regeneration replaces outputs only under toy_models, including report.html.
Use --only to select experiments in experiments.py and representations.py.

experiments.py:
  gravity       Modal accuracy, density aliasing, and padded source quadrature.
  restriction   Reduced reproduction of a conditional GAMER restriction edge case.
  amr           Fine Poisson solve with independently contaminated boundary values.
  nufft         Exact dense nonuniform Fourier transform and inverse conditioning.
  linear        Strang, Yoshida fourth order, CN, and full exponential propagation.
  sp            Self-gravitating colliding packets; gravity and time ablations.
  mapped        Hermitian mapped spectral operators; a localized linear bound state.

representations.py:
  mapped_sp     Self-gravitating expanding/breathing packet on fixed mapped grids.
  bands         Few-carrier Fourier Galerkin SP; exact cross-carrier gravity.

frozen_hamiltonian.py:
  Stationary and perturbed nonlinear cores; frozen full-H exponentials,
  self-consistent exponential midpoint, and reference-Hamiltonian splitting.

verify.py:
  Independent Fourier identities, convolution and energy-gradient checks,
  temporal convergence, reference refinement, and physical mass checks.

results/ contains machine-readable JSON, PNG figures, and two small NPZ references.
Each JSON reports its normalization, resolution, tolerances, and/or work counts
where relevant. Fixed random seeds are used for the nonuniform-transform diagnostic.

GAMER source audit:
  Public main was pinned at e36797538a1a5eaa456bc51ad44faf2b201a4aef.
  This may differ from the version used for the published simulations.
  sources/gamer_provenance.json lists the files inspected.
  & $py toy_models/fetch_sources.py
  downloads that pinned source subset and its license into sources/gamer/.
  The downloaded source snapshot and repository tree are locally ignored.
  No GAMER compilation or end-to-end GAMER run was performed.

Scope:
  Dimensionless periodic 1D systems with fixed scale factor. A 1D Poisson kernel
  is not the spherical 3D gravitational kernel. None of these experiments measures
  a cosmological halo profile or establishes a production speedup. Dense matrices
  provide diagnostic oracles only. A 3D implementation would need matrix-free
  actions, global elliptic accuracy, parallel communication, and convergence of
  the actual halo observables. The report distinguishes results from proposals.
