# Solve the linear Schrödinger equation in 1D using a Physics Informed Neural Network

This code heavily draws on the implementation of the PINN approach published by Jan Blechschmidt under https://github.com/janblechschmidt/PDEsByNNs/ (MIT license) and was further developed using Claude. Please see [my blog post](https://kunkelalexander.github.io/blog/computers-learning-pdes/) for a motivation. 

![Spectral bias](figures/spectral_bias.gif)

## Explanation
In the following, we will solve the dimensionless 1D Schrödinger equation with inhomogeneous Dirichlet boundary conditions using a Physics Informed Neural Network (PINN).
This boundary values problem can be stated as

$$i \hbar \partial_t \psi(x, t) = \left(-\frac{\hbar^2}{2m}\partial_x^2 + V(x, t)\right) \psi(x, t)$$
with
$$\psi(x, t = t_0) \equiv \psi_0(x) \qquad \psi(x=x_l, t) \equiv \psi_L(t) \qquad \psi(x=x_r, t) \equiv \psi_R(t)$$
for $t \in [t_0, t_1]$ and $x \in [x_l, x_r]$ and $\psi \in C_2([t_0, t_1] \times [x_l, x_r], \mathbb{C})$.

In the first approach considered here - the continuous time approach - we approximate $\psi(x, t)$ via a neural network $\psi_{\theta}(x, t)$ that takes two input parameters and outputs two ouput parameters - the real and imaginary part of the wave function. Note that we approximate the wave function in both spatial and temporal dimensions. In order for the NN to satisfy the boundary value problem, the residual of the PDE

$$r_{\theta}(x, t) \equiv \left(i \hbar \partial_t + \frac{\hbar^2}{2m} \partial_x^2 - V(x, t)\right)\psi_{\theta}(x, t)$$
is included in the loss functional. In total, the loss functional then contains three terms:
 - The mean squared residual
 - The mean squared misfit w.r.t. initial conditions
 - The mean squared misfit w.r.t. boundary conditions

It is minimised over a number of collocation points that are randomly sampled from $[t_0, t_1] \times [x_l, x_r]$.



![PINN prediction of the real part of the wave function](figures/continuous_16_pred_surface_real.png)

## Files

| File | Description |
|---|---|
| `1_linear_schroedinger_1d.ipynb` | Original TensorFlow implementation (continuous time) |
| `2_linear_schroedinger_1d_discrete_time.ipynb` | Original TensorFlow implementation (discrete time) |
| `3_benchmark_resolution.py` | Resolution and step-size benchmarks of both PINNs against classical solvers (PyTorch, standalone) |
| `5_benchmark_strategies.py` | Periodic features and the stage-derivative (K) form for the discrete time PINN, with a regression diagnostic of what the networks can represent, on the benchmarks of script 3; builds on `3_benchmark_resolution.py` |

## Performance comparison

Both TensorFlow notebooks solve the same plane wave test problem on $x \in [0, 1]$, $t \in [0, \pi]$ with 5000 Adam iterations and the same learning rate schedule, run on CPU on the same machine.

| | Continuous time | Discrete time |
|---|---|---|
| Training points per iteration | 10000 collocation + 50 initial + 50 boundary | 100 initial + 2 boundary |
| Network outputs | 2 | 2(q+1) = 514 (q = 256 IRK stages) |
| Training time | 398 s | 37 s |
| Final loss | 1.2e-5 | 1.5e-6 |
| L1 error | 0.0006 (random test points in $x$–$t$) | 0.0015 (at $t_1$) |
| Mean density $\lvert\psi\rvert^2$ (exact: 1) | 0.9999 | 0.996 |

The discrete time approach trains about 11 times faster because the IRK scheme replaces the collocation points in time. Note that the L1 errors are evaluated on different points: over the whole domain for the continuous time approach and at the final time $t_1$ for the discrete time approach.

## Resolution benchmark

`3_benchmark_resolution.py` compares how the error of both PINNs depends on resolution with that of classical solvers.

![L1 error vs. resolution](figures/benchmark_1_error_vs_resolution.png)

![L1 error vs. degrees of freedom and wall-clock time](figures/benchmark_2_error_vs_cost.png)

### Setup

- **Test problem:** plane wave $\psi = e^{i(kx - \omega t)}$ with $\omega = k^2/2$ on $x \in [0, 1]$, for 16 wavelengths (as in the paper, where $N = 2^5$ is two points per wavelength) and for 2 wavelengths. The wave travels a third of the domain until $t_1$. This avoids $\omega t_1$ being a multiple of $2\pi$, which would make the aliased solution at $N = 16$ exact by accident.
- **PINNs:** both notebook methods, ported to PyTorch in float64. Each is trained with 2000 Adam steps followed by 1500 L-BFGS steps. The networks are the notebook defaults: 4×20 for continuous time (1,362 parameters) and 4×50 for discrete time with $q = 256$ (33,964 parameters). The PDE residual of the continuous time PINN is divided by $\omega$ so that it is of the same order as the initial and boundary terms.
- **References:** 4th- and 6th-order central finite differences with RK4 at half the stability limit, and an FFT solver with exact time integration, all on a periodic grid.
- **Resolution $N$:** grid points for FD and FFT; collocation points per dimension for the PINNs ($N \times N$ in space-time for continuous time, $N$ initial points for discrete time). The networks stay the same for all $N$. The continuous time PINN is capped at $N = 2^7$ for runtime reasons.
- **Error:** mean $L_1$ error of real and imaginary part at $t_1$, $\frac{1}{2}\,\mathrm{mean}(|\Delta \mathrm{Re}\,\psi| + |\Delta \mathrm{Im}\,\psi|)$.

### Results

| | 16 wavelengths | 2 wavelengths |
|---|---|---|
| FD 4th / 6th order | Clean $N^{-4}$ / $N^{-6}$ | Same slopes; 6th order reaches round-off (about $10^{-12}$) from $N = 2^9$ |
| FFT | About $10^{-15}$ from two points per wavelength | About $10^{-16}$ |
| Continuous time PINN | Fails, error 0.7–2 at every $N$ | Flat at about $1.2 \times 10^{-4}$ from $N = 2^4$ |
| Discrete time PINN | Fails, error about 0.63 at every $N$ | Flat at about $5 \times 10^{-3}$ from $N = 2^5$ |

- **16 wavelengths:** neither PINN learns the solution at any resolution, not even where 4th-order FD already reaches $10^{-6}$. This is spectral bias: plain tanh networks learn high frequencies very slowly ([Rahaman et al. 2019](https://arxiv.org/abs/1806.08734); [Wang, Yu & Perdikaris, JCP 2022](https://arxiv.org/abs/2007.14527)).
- **2 wavelengths:** the PINN errors do not decrease with $N$. The final loss is the same for every $N$ (about $6 \times 10^{-6}$ for discrete time, about $5 \times 10^{-7}$ for continuous time), so the optimiser limits the accuracy, not the number of collocation points.
- **Network size barely matters:** across widths from 382 to 82k parameters at $N = 64$, the continuous time PINN stays at $1.3$–$1.8 \times 10^{-4}$, and the discrete time PINN gets worse from width 50 to width 100. At equal degrees of freedom, FD is 5–10 orders of magnitude more accurate and about 1000 times faster.
- **Continuous vs. discrete time:** the continuous time PINN is about 40 times more accurate with about 25 times fewer parameters. Most parameters of the discrete time PINN sit in its 514-output final layer.
- **Caveat:** one seed per point, so expect PINN values to scatter by roughly a factor of 2 between seeds.

### Network size vs. grid points

There is no established mapping like "number of network parameters = number of grid points". A PINN has two separate resolutions:

- **Collocation points** play the role of quadrature nodes. The generalisation error is bounded by the training error plus a quadrature term that decays as $N_\mathrm{colloc}^{-\alpha}$, with $\alpha = 1/2$ for random sampling and higher for grid or quasi-Monte Carlo points ([Mishra & Molinaro, IMA J. Numer. Anal. 2023](https://arxiv.org/abs/2006.16144)).
- **Network width and depth** set how rich the approximation is, similar to the polynomial order or number of basis functions, not the grid ([De Ryck, Lanthaler & Mishra, Neural Networks 2021](https://arxiv.org/abs/2104.08938)).

In practice, neither usually limits accuracy: optimisation does ([Krishnapriyan et al., NeurIPS 2021](https://arxiv.org/abs/2109.01050)). This is why the benchmark also plots the error against degrees of freedom and wall-clock time, following the recommendation to compare at equal accuracy or equal runtime by [McGreivy & Hakim (Nat. Mach. Intell. 2024)](https://arxiv.org/abs/2407.07218) and [Grossmann et al. (IMA J. Appl. Math. 2024)](https://arxiv.org/abs/2302.04107).

Periodic features against spectral bias (a periodic variant of the Fourier features of [Tancik et al. 2020](https://arxiv.org/abs/2006.10739) and [Wang, Wang & Perdikaris 2021](https://arxiv.org/abs/2012.10047)) and a reformulation of the stage equations that needs no network derivative at evaluation are tested in the [strategy benchmark](#strategy-benchmark) below.

## Step-size benchmark

The discrete time approach promises large time steps, since the implicit Gauss-Legendre scheme has no stability limit. The step-size benchmark tests whether this carries over to the PINN.

![L1 error and cost vs. step size](figures/benchmark_3_error_vs_step_size.png)

### Setup

- **Test problem:** plane wave with 2 wavelengths in the periodic potential $V(x) = 20\cos(2\pi x)$, advanced by a single step of size $\omega\Delta t = 1, \dots, 128$ with $\omega = k^2/2$. Without a potential, the Fourier propagator would be exact for any step size; the potential is what makes the classical solvers depend on the step size, as in Schrödinger-Poisson. The exact solution is computed from the eigen-decomposition of the Hamiltonian in a truncated Fourier basis.
- **PINNs** (same training and networks as above, $N = 64$ collocation points per dimension):
  - discrete time as in the notebook ($q = 256$, $\psi^{n+1}$ as a network output)
  - discrete time with $q$ adapted to the step (smallest Gauss-Legendre scheme with a temporal error below $10^{-8}$) and $\psi^{n+1} = \psi_0 + \Delta t \sum_k b_k \mathcal{N}[\psi_k]$ computed from the stages
  - continuous time on $[0, \Delta t]$ with 16 collocation points per period in $t$
- **Classical single steps:** split-step Fourier (Strang), Crank-Nicolson, and the Gauss-Legendre scheme with the same $q$ solved exactly by linear algebra, i.e. what a perfectly trained discrete time PINN would achieve.

### Results

| $\omega\Delta t$ | 1 | 4 | 16 | 32 | 128 |
|---|---|---|---|---|---|
| Required stages $q$ | 8 | 24 | 96 | 192 | > 256 |
| Gauss IRK, exact solve | $2 \times 10^{-10}$ | $1 \times 10^{-9}$ | $2 \times 10^{-9}$ | $2 \times 10^{-9}$ | $2 \times 10^{-3}$ ($q$ capped at 256) |
| PINN discrete time (notebook) | $5 \times 10^{-3}$ | $3 \times 10^{-2}$ | $8 \times 10^{-2}$ | 0.45 | 0.41 |
| PINN discrete time ($q$ adapted) | $6 \times 10^{-3}$ | $4 \times 10^{-2}$ | 0.14 | 0.45 | 0.40 |
| PINN continuous time | $5 \times 10^{-4}$ | $4 \times 10^{-3}$ | $6 \times 10^{-2}$ | 0.19 | 0.63 |
| Split-step Fourier (Strang) | $1 \times 10^{-2}$ | 0.35 | 1.1 | 0.78 | 0.81 |

- **The time scheme allows large steps, the PINN does not deliver them.** Solved exactly, the Gauss-Legendre scheme stays at about $10^{-9}$ up to $\omega\Delta t = 32$. The discrete time PINNs solving the same equations are 7–8 orders of magnitude less accurate, and their error grows with the step size until they fail completely at $\omega\Delta t \approx 32$.
- **No free lunch in time.** The number of stages needed grows linearly with the step, $q \approx 3\,\omega\Delta t$, because every Fourier mode excited by the potential has to be resolved in phase. A large step does not remove temporal degrees of freedom; it moves them into the stages.
- **Training gets harder with the step size.** The final loss grows from about $10^{-5}$ at $\omega\Delta t = 1$ to about $5 \times 10^{-4}$ at $\omega\Delta t \geq 32$. Errors in the curvature of the stages enter the IRK update multiplied by $\omega\Delta t$.
- **Adapting $q$ and computing $\psi^{n+1}$ from the stages does not help.** Both discrete time variants perform the same, so the number of outputs and the loss weighting of $\psi^{n+1}$ are not what limits accuracy. The continuous time PINN is about 10 times more accurate at $\omega\Delta t = 1$, but its advantage shrinks with the step size and is gone by $\omega\Delta t = 16$.
- **Single-step comparisons with second-order schemes are misleading.** The discrete time PINN beats one Strang or Crank-Nicolson step for $\omega\Delta t \geq 2$, but those schemes are meant for small steps: they need milliseconds per step, whereas training one PINN step takes 1–10 minutes.

For Schrödinger-Poisson this means that large time steps are better obtained with classical implicit or exponential integrators (or the phase-free Madelung formulation) than with a discrete time PINN.

## Strategy benchmark

`5_benchmark_strategies.py` repeats the resolution and step-size benchmarks above (same problems, same $N$ and $\omega\Delta t$, same network, optimiser and error) for changes of the discrete time PINN that need neither a uniform grid nor the solution, so that they could carry over to a 3D cosmological box. It tests whether they fix the three failures found above: spectral bias (16 wavelengths), the error plateau (2 wavelengths) and the failure at large steps.

![L1 error vs. resolution for each strategy](figures/benchmark_4_strategies_vs_resolution.png)

![L1 error and training loss vs. step size for each strategy](figures/benchmark_5_strategies_vs_step_size.png)

![Ablation of all variants](figures/benchmark_6_strategy_ablation.png)

### Setup

- **Baseline:** the network of script 3 with $\psi^{n+1} = \psi_0 + \Delta t\sum_k b_k \mathcal{N}[\psi_k]$ computed from the stages and a *periodicity loss* for $\psi$ and $\psi_x$ at $x = 0, 1$ instead of exact boundary data, which a simulation would not have. The same network with exact boundary data shows what that costs; the notebook version of script 3 is the grey reference.
- **Periodic features (`per`):** the input is embedded in $[\sin(2\pi m x), \cos(2\pi m x)]$ for $m = 1 \dots 16$. The network is exactly periodic, and the boundary term is dropped. The features are the same for all problems, but contain their wave numbers (2 and 16, the latter the highest feature), so for these the benchmark tests the optimisation and not the discovery of the frequency. $m$ up to 32 does not train (2 waves, $N = 256$: loss of the regression $5 \times 10^{-2}$ instead of $9 \times 10^{-5}$ for $m \leq 16$), since the second derivative of a feature grows as $m^2$.
- **K form (`k`):** the network outputs $K_i = \mathcal{N}[\psi_i]/\omega$, the stages are $\psi_i = \psi_0 + \Delta t\,\omega \sum_j a_{ij} K_j$, the residual is $K_i - \mathcal{N}[\psi_i]/\omega$, and $\psi^{n+1} = \psi_0 + \Delta t\,\omega\sum_i b_i K_i$ needs no derivative of the network. The initial data is exact. Both changes are also run together.
- **Regression floor:** for every network, the exact IRK stages (values and second derivatives, $K$ and its second derivative for the K form) are fitted by regression with the same network, optimiser and budget, and $\psi^{n+1}$ is formed from the fit. This is what the network can represent. The difference to the PINN is the optimisation error. The exact stages follow from the linear stage equations, decoupled in the eigenbasis of the Hamiltonian.
- **Resolution benchmark:** $q = 256$, $N = 2^4 \dots 2^{10}$, 2 and 16 wavelengths. **Step-size benchmark:** $q$ adapted to the step, $\omega\Delta t = 1 \dots 128$, at $N = 256$ instead of 64: a grid with $N = 64$ has only 4 points per period of the highest feature, and the network can then satisfy the equations between the collocation points in many ways.
- Everything else as in script 3: 4×50 network, 2000 Adam + 1500 L-BFGS steps, one seed in the sweeps. In the ablation figure (2 and 16 waves at $N = 256$, one step of $\omega\Delta t = 16$), every variant and floor is shown with the seed-0 result of the sweeps (`--seeds 5` adds four more seeds, shown as median and range).
- **Not included**, since they would not scale to a cosmological box: Levenberg-Marquardt (a dense Jacobian with $2qN^d$ rows and one column per parameter, full batch: about 140 GB at $q = 256$, $N = 1024$), and the interaction picture of an earlier version of this work (a spectral collocation loss on a uniform grid, where a PINN is not needed). Fourier features with random frequencies, adaptive loss weights and random weight factorisation were tested in an earlier version of this script (git history): only features matched to the wave number of the solution helped.

### Results

$L_1$ error at $t_1$, one seed per point (the regression floor of the same network is in brackets):

| | 16 waves, $N = 256$ | 16 waves, $N = 1024$ | 2 waves, $N = 256$ | 2 waves, $N = 1024$ | $\omega\Delta t = 1$ | $\omega\Delta t = 4$ | $\omega\Delta t = 16$ | $\omega\Delta t = 32$ |
|---|---|---|---|---|---|---|---|---|
| Baseline | 1.1 (0.61) | 1.1 (0.35) | $6.4 \times 10^{-3}$ ($9 \times 10^{-4}$) | $1.4 \times 10^{-2}$ ($7 \times 10^{-4}$) | $1.9 \times 10^{-3}$ | $2.2 \times 10^{-2}$ | 0.63 ($2 \times 10^{-3}$) | 0.62 |
| Periodic features | $5.1 \times 10^{-3}$ ($5 \times 10^{-4}$) | $3.9 \times 10^{-3}$ ($5 \times 10^{-4}$) | $4.5 \times 10^{-3}$ ($7 \times 10^{-4}$) | $5.8 \times 10^{-3}$ ($7 \times 10^{-4}$) | $6.7 \times 10^{-3}$ | $9.1 \times 10^{-2}$ | 0.41 (0.09) | 0.62 |
| K form | 1.1 (3.0) | 1.1 (2.5) | $5.7 \times 10^{-3}$ ($2 \times 10^{-4}$) | 0.16 ($1 \times 10^{-4}$) | $1.6 \times 10^{-3}$ | $6.7 \times 10^{-2}$ | 0.63 (0.01) | 0.62 |
| K form + periodic features | $6.0 \times 10^{-3}$ ($3 \times 10^{-3}$) | $5.6 \times 10^{-3}$ ($3 \times 10^{-3}$) | $1.7 \times 10^{-3}$ ($6 \times 10^{-4}$) | $3.0 \times 10^{-3}$ ($5 \times 10^{-4}$) | $2.3 \times 10^{-4}$ | $6.3 \times 10^{-2}$ | 0.41 (0.13) | 0.61 |

- **Spectral bias (16 wavelengths): fixed by periodic features, and only by them.** The baseline and the K form do not learn the solution at any $N$ (error about 1, which is no better than predicting zero). With periodic features the error drops to $4$–$6 \times 10^{-3}$ at $N \geq 256$ and no longer depends on $N$. The features contain the wave number, so this says that the network trains once it can represent the frequency, not that it finds the frequency. The remaining gap to the regression floor ($5 \times 10^{-4}$) is optimisation error. From $N \approx 2^9$ on, fourth-order finite differences are more accurate.
- **The error plateau (2 wavelengths) is not fixed.** Every variant stays between $2 \times 10^{-3}$ and $10^{-2}$ from $N = 256$ to $1024$. The regression floors lie at $10^{-4}$–$7 \times 10^{-4}$, so the networks could do 3–10 times better, but the best variant (K form + periodic features) is only about a factor 3 better than the baseline at $N = 256$ and the K form alone becomes unstable at $N = 1024$ (0.16). With one seed this difference is not established.
- **Large steps are not fixed.** At $\omega\Delta t = 16$ all variants are at 0.4–0.6 and at $\omega\Delta t = 32$ all fail ($\approx 0.6$). Only at $\omega\Delta t = 1$ does the K form with periodic features help (a factor 8 below the baseline). At $\omega\Delta t = 16$ the baseline and the K form could represent the solution (floors $2 \times 10^{-3}$ and $10^{-2}$) but do not find it, whereas for the periodic features the floor itself rises to 0.09–0.13: the features cannot represent the stage derivatives at large steps. The final training losses grow with the step in the same way for all variants.
- **Exact boundary data matters little, except at $\omega\Delta t = 16$.** Replacing the periodicity loss by exact boundary data changes the error by a factor of at most 4 in the 2-wave resolution benchmark (in favour of the periodicity loss at $N = 256$: $6.4 \times 10^{-3}$ vs. $2.3 \times 10^{-2}$) and at $\omega\Delta t \leq 4$, and both fail with 16 waves. Only at $\omega\Delta t = 16$ is it 7 times better ($8.9 \times 10^{-2}$ vs. 0.63), and at $\omega\Delta t = 32$ both fail.
- **Caveats:** one seed per point, and one network size and optimiser budget. The ablation figure shows the single seed-0 values of the sweeps; `--seeds 5` would add the spread.

For a 3D cosmological box this means that only periodic features that match the frequencies of the solution address spectral bias, and neither they nor the K form change the accuracy plateau or the step size at which the discrete time PINN fails.

## Running the benchmarks

```bash
python 3_benchmark_resolution.py                      # both benchmarks, about 2.5 h on a 4-core CPU
python 3_benchmark_resolution.py --benchmarks step    # only the step-size benchmark, about 1 h
python 3_benchmark_resolution.py --quick              # smoke test, a few minutes
python 3_benchmark_resolution.py --plot-only          # replot from cached results

python 5_benchmark_strategies.py                      # strategy benchmark, takes hours, reads the baselines of script 3
python 5_benchmark_strategies.py --waves 16 --benchmarks resolution   # one process per cache file,
python 5_benchmark_strategies.py --waves 2 --benchmarks resolution    # so these three can run in parallel
python 5_benchmark_strategies.py --benchmarks step
python 5_benchmark_strategies.py --seeds 5            # extra seeds for the ablation (about 40 % more runtime)
python 5_benchmark_strategies.py --quick              # smoke test
python 5_benchmark_strategies.py --plot-only          # replot from cached results
```

Results are cached in `results/benchmark_plane_wave_<waves>waves.json` and `results/benchmark_step_size.json` after every run, so an interrupted benchmark resumes where it left off. The strategy benchmark adds its variants to the same files. See `python 3_benchmark_resolution.py --help` and `python 5_benchmark_strategies.py --help` for all options.

## Setup

### Create the conda environment

```bash
conda env create -f environment.yml
conda activate pinn_env
```

This installs a CPU-only PyTorch build. For GPU support, open `environment.yml`, remove the `cpuonly` line, and add `pytorch-cuda=12.1` (adjust the version to match your CUDA installation), then re-run the command above.
