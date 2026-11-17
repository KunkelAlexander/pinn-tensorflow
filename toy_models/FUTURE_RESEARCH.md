

The search found no neural method that beats GAMER, or has even been benchmarked against it, for Schrödinger-Poisson. The abstracts and summaries I saw were thin, so I haven't verified the details inside the papers.

What exists

SPINN (arXiv 2506.02957, ApJ) is a PINN that solves the SP equations for FDM collapse in 1D and 3D.
The authors report accurate mass conservation, density profiles and structure suppression.
The search results suggest it is validated only on limited time frames, not full cosmological runs.
A variant that predicts density, velocity and potential directly failed to reproduce the wave features.
I found no speed or accuracy comparison against GAMER.
Cosmo-SPINN (arXiv 2607.28604) is a generative U-Net with a physics-informed loss.
It covers evolution from initial conditions and super-resolution of FDM simulations.
Its training data comes from a second-order split-step pseudo-spectral solver. It is therefore an emulator that depends on a conventional code.
It names autoregressive neural operators as future work, so none is implemented there.
Quantum variational time evolution (arXiv 2307.06032) was in the results. I didn't look into it.
The conventional side

GAMER-2 (arXiv 1712.07070) is GPU-accelerated AMR. It scales to 4096 GPUs and uniform resolutions up to 10240³ cells, and it has been used extensively for wave dark matter.
Newer work, such as a hybrid scheme, couples a wave solver on small scales with a fluid (Hamilton-Jacobi-Madelung) solver on large scales. That is a different way to cut cost.
There is also a deep zoom-in FDM galactic halo simulation, which is directly relevant to your zoom-in case.
What this means for you

My earlier answer holds. The nearest neural work is a proof-of-concept PINN and a generative emulator trained on conventional-solver output. Neither is a drop-in alternative to GAMER. The most plausible near-term role for a network is the Cosmo-SPINN one: super-resolution or emulation on top of a conventional solver. It is not replacing the solver.

Sources:

SPINN (arXiv)
SPINN (ApJ)
Cosmo-SPINN
Hybrid Schrödinger / Hamilton-Jacobi-Madelung scheme
Quantum variational time evolution
Deep zoom-in FDM halo
GAMER-2


All of these papers come from my research group - i want to go beyond. please do a broader literature review
Web SearchNeural Galerkin schemes sequential-in-time training nonlinear Schrödinger Dirac-Frenkel variational principle neural network time evolution
OUT
Web search results for query: "Neural Galerkin schemes sequential-in-time training nonlinear Schrödinger Dirac-Frenkel variational principle neural network time evolution"

Links: [{"title":"Randomized Sparse Neural Galerkin Schemes for Solving Evolution Equations with Deep Networks","url":"https://arxiv.org/html/2310.04867"},{"title":"Randomized Sparse Neural Galerkin Schemes for Solving ...","url":"https://arxiv.org/pdf/2310.04867"},{"title":"Neural Galerkin schemes for sequential-in-time solving of partial differential equations with deep networks - ScienceDirect","url":"https://www.sciencedirect.com/science/article/abs/pii/S1570865924000061"},{"title":"Neural Galerkin Schemes with Active Learning for High- ...","url":"https://arxiv.org/pdf/2203.01360"},{"title":"Sequential-in-time training of nonlinear parametrizations for solving time-dependent partial differential equations","url":"https://arxiv.org/pdf/2404.01145"},{"title":"Journal of Computational Physics 496 (2024) 112588","url":"https://par.nsf.gov/servlets/purl/10537598"},{"title":"Neural Galerkin schemes for sequential-in-time solving of partial differential equations with deep networks - NYU Scholars","url":"https://nyuscholars.nyu.edu/en/publications/neural-galerkin-schemes-for-sequential-in-time-solving-of-partial-2"},{"title":"Coupling parameter and particle dynamics for adaptive sampling in Neural Galerkin schemes - ScienceDirect","url":"https://www.sciencedirect.com/science/article/abs/pii/S0167278924000800"},{"title":"Randomized Sparse Neural Galerkin Schemes for Solving ...","url":"https://proceedings.neurips.cc/paper_files/paper/2023/file/0cb310ed8121549488fea8e8c2056096-Paper-Conference.pdf"}]

The search turned up a consistent picture of Neural Galerkin schemes and their link to the Dirac-Frenkel principle, though nothing specific to the nonlinear Schrödinger equation.

- **Core idea:** Neural Galerkin methods train a network sequentially in time instead of fitting the whole space-time solution at once. The approach was developed by Berman, Schwerdtner, and Peherstorfer, and their survey in the ScienceDirect chapter describes it as a variational treatment of time-dependent PDEs with deep networks ([3-1]; [3-3]).
- **Dirac-Frenkel connection:** At each time step, the parameter update is the orthogonal projection of the dynamics onto the tangent space of the manifold the network parameterization defines ([1-5]). Because the approximation can depend nonlinearly on its parameters, it can lie on a smooth manifold rather than in a fixed linear space ([1-4]).
- **Quantum dynamics lineage:** The principle has a long history in quantum dynamics. Computational chemists have used it to solve the Schrödinger equation with Gaussian wave packets ([5-2]), which is the closest match to your nonlinear Schrödinger query, though I found no source applying Neural Galerkin schemes specifically to the nonlinear Schrödinger equation.
- **Practical refinements:** The randomized sparse variant updates random subsets of network parameters each step, which the authors report reduces overfitting locally in time and lowers training cost ([2-8]; [2-9]). A separate line of work adds adaptive sampling of training points over time ([4-2]).

Sources: arXiv 2310.04867 (randomized sparse variant), the ScienceDirect survey chapter, arXiv 2203.01360 (active learning variant), and arXiv 2404.01145 (sequential-in-time review).


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
Web SearchFourier neural operator nonlinear Schrödinger equation long-time rollout unitarity conservation structure-preserving neural operator
OUT
Web search results for query: "Fourier neural operator nonlinear Schrödinger equation long-time rollout unitarity conservation structure-preserving neural operator"

Links: [{"title":"Conservation-preserved Fourier Neural Operator through ...","url":"https://arxiv.org/pdf/2505.24579"},{"title":"Geometry-Conditioned Fourier Neural Operators for Cubic Nonlinear Schrödinger Dynamics on Periodic Domains","url":"https://arxiv.org/html/2606.27459v2"},{"title":"Fourier Neural Operators Explained: A Practical Perspective","url":"https://arxiv.org/pdf/2512.01421"},{"title":"An Inverse Scattering Inspired Fourier Neural Operator for Time-Dependent PDE Learning","url":"https://arxiv.org/pdf/2512.19439"},{"title":"An Inverse Scattering Inspired Fourier Neural Operator for Time-Dependent PDE Learning","url":"https://arxiv.org/html/2512.19439"},{"title":"[2512.19439] An Inverse Scattering Inspired Fourier Neural Operator for Time-Dependent PDE Learning","url":"https://arxiv.org/abs/2512.19439"},{"title":"Structure-Informed Neural Operators for Long-Time Prediction of Parametric Hamiltonian PDEs","url":"https://arxiv.org/html/2606.14913"},{"title":"Numerical long-time energy conservation for the nonlinear Schrödinger equation","url":"https://academic.oup.com/imajna/article-pdf/37/4/2067/20667282/drw057.pdf"},{"title":"Approximating Numerical Fluxes Using Fourier Neural Operators for Hyperbolic Conservation Laws","url":"https://arxiv.org/html/2401.01783v3"}]

The search turned up several relevant papers, though most are snippets rather than full texts, so treat the details as provisional.

- **Conservation-preserved FNO (arXiv 2505.24579):** This paper directly addresses the Schrödinger case. It reports that standard FNOs often miss conservation laws such as norm conservation, and it tests linear and nonlinear Schrödinger equations, with norm-conservation error reported as near zero for the nonlinear case. It uses an adaptive correction during training rather than only a loss penalty. The snippets also note that large penalty weights can make training unstable.
- **Structure-Informed FNO / EP-FNO (arXiv 2606.14913):** This work targets long-time prediction for parametric Hamiltonian PDEs. It adds an invariant projection step after each FNO prediction to restore quantities such as mass or Hamiltonian energy, and it applies that projection at every rollout step. The authors also argue that architectures without explicit conservation or Hamiltonian structure can drift during long autoregressive rollouts.
- **Geometry-conditioned FNO for cubic NLS (arXiv 2606.27459):** This study covers the cubic NLS on 2D tori. Its ablations report that including an extra frequency term improves long-time accuracy, and it cites structure-preserving learning as a way to reduce drift.
- **Classical split-step background (Oxford IMA J. Numer. Anal.):** Work on the split-step Fourier method shows that energy is nearly preserved over long, polynomially scaled time intervals under a time-step nonresonance condition. This is useful as a numerical-analysis baseline for what a learned operator should aim to match.
- **Other context:** A survey notes that conservation or geometric constraints can extend stable autoregressive rollout horizons, and the IS-FNO paper (arXiv 2512.19439) reports that an unconstrained latent design can lose stability on stiff cases.

The most directly relevant items are 2505.24579 and 2606.14913. I did not find a dedicated paper on unitarity-preserving neural operators for the NLS, so that angle may need a separate search.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
Web SearchGauss-Newton OR natural gradient PINN high accuracy machine precision Levenberg-Marquardt energy natural gradient 2024 2025
OUT
Web search results for query: "Gauss-Newton OR natural gradient PINN high accuracy machine precision Levenberg-Marquardt energy natural gradient 2024 2025"

Links: [{"title":"Gauss-Newton Natural Gradient Descent for Physics-Informed Computational Fluid Dynamics by Anas Jnini, Flavio Vella, Marius Zeinhofer :: SSRN","url":"https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5201329"},{"title":"SS-ESOAP: Self-Scaled Adaptive Preconditioning for Physics-Informed Learning","url":"https://arxiv.org/pdf/2608.29448"},{"title":"On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime","url":"https://arxiv.org/pdf/2601.03162"},{"title":"Gauss-Newton Natural Gradient Descent for Shape Learning","url":"https://arxiv.org/pdf/2602.00099"},{"title":"Achieving High Accuracy with PINNs via Energy Natural Gradient Descent","url":"https://arxiv.org/pdf/2302.13163"},{"title":"Dual Natural Gradient Descent for Scalable Training of ...","url":"https://arxiv.org/pdf/2505.21404"},{"title":"From Non-Convex Self-Concordant Regularization to Scalable Quasi-Newton Training of PINNs","url":"https://arxiv.org/pdf/2608.04206"},{"title":"Beyond PINNs: A Unified Gauss--Newton and Petrov--Galerkin Framework for Neural and Hybrid PDE Solvers","url":"https://arxiv.org/pdf/2609.20641"},{"title":"Accelerating Natural Gradient Descent for PINNs with","url":"https://arxiv.org/pdf/2505.11638"},{"title":"Curvature-Aware Optimization for High-Accuracy Physics-Informed Neural Networks","url":"https://arxiv.org/html/2604.05230"}]

The search turned up a cluster of 2023–2025 papers on natural-gradient and Gauss-Newton training for PINNs aimed at high accuracy. I ran one search, so this is a sample rather than a full survey.

- **Energy natural gradient (Müller & Zeinhofer, ICML 2023):** This is the foundational paper. It uses a Hessian-induced metric, and the energy natural gradient update direction corresponds to the Newton direction in function space, modulo an orthogonal projection onto the tangent space of the model. It is also the paper the others build on.
- **Gauss-Newton natural gradient for CFD (Jnini, Vella, Zeinhofer, 2024/2025):** This applies Gauss-Newton in function space to Navier-Stokes. Its reported result is close to single-precision accuracy measured in relative $L^2$ norm on a number of benchmark problems.
- **Dual natural gradient (Jnini & Vella, arXiv 2025):** This addresses cost. The Gauss-Newton step, which is normally solved in parameter space with a prohibitive O(n^3) time complexity, is reformulated in a smaller residual space.
- **Curvature-aware comparison (arXiv 2604.05230, 2026):** This paper compares quasi-Newton methods with Gauss-Newton and natural-gradient methods across elliptic, parabolic, and hyperbolic PDEs, and reports that all of the main optimizers considered were able to produce high-accuracy solutions.
- **Other related work:** ANaGRAM (ICLR 2025), a natural gradient relative to an adapted model, and a Kronecker-factored curvature approach for PINNs appear in reference lists. Quasi-Newton and Nyström-based variants also show up in the results.

The search did not establish the "machine precision" claim directly. The strongest reported results are near single precision, and the results I saw come from the papers' own benchmarks. I did not check the Levenberg-Marquardt connection, so treat that as unverified. For details, check the arXiv and SSRN entries for the original papers.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
Web Searchneural network Poisson solver OR learned multigrid preconditioner cosmological N-body gravity AMR deep learning
OUT
Web search results for query: "neural network Poisson solver OR learned multigrid preconditioner cosmological N-body gravity AMR deep learning"

Links: [{"title":"Solving the cosmological Vlasov–Poisson equations with physics-informed Kolmogorov–Arnold networks","url":"https://academic.oup.com/mnras/article/545/4/staf2241/8404154"},{"title":"Field-Level Comparison and Robustness Analysis of Cosmological N-Body Simulations","url":"https://arxiv.org/html/2505.13620v1"},{"title":"Field-Level Comparison and Robustness Analysis of Cosmological N-body Simulations","url":"https://arxiv.org/pdf/2505.13620"},{"title":"[2512.11795] Solving the Cosmological Vlasov-Poisson Equations with Physics-Informed Kolmogorov-Arnold Networks","url":"https://arxiv.org/abs/2512.11795"},{"title":"DEUS Full Observable {\\Lambda}CDM Universe Simulation: the numerical challenge","url":"https://arxiv.org/pdf/1206.2838"},{"title":"(PDF) Field Level Neural Network Emulator for Cosmological N-body Simulations","url":"https://www.researchgate.net/publication/361206478_Field_Level_Neural_Network_Emulator_for_Cosmological_N-body_Simulations"},{"title":"Cosmological Hydrodynamics with Adaptive Mesh Refinement: a new high resolution code called RAMSES","url":"https://arxiv.org/pdf/astro-ph/0111367"},{"title":"Hybrid Physical-Neural ODEs for Fast N-body Simulations","url":"https://arxiv.org/pdf/2207.05509"},{"title":"Field Level Neural Network Emulator for Cosmological N-body Simulations","url":"https://arxiv.org/pdf/2206.04594"}]

The search didn't turn up a learned multigrid preconditioner or a neural Poisson solver inside a cosmological AMR code. It did surface related work on the classical solvers and on neural approaches to N-body gravity.

- **Classical AMR baseline:** Enzo's AMR solver uses a Fourier method on the root grid and multigrid on the refined subgrids, solving Poisson's equation level by level with Dirichlet boundary conditions from the coarser level. RAMSES evolves particles with a particle-mesh solver while solving the Poisson equation with a multigrid method. These are the classical components a learned preconditioner would have to replace or accelerate.
- **Physics-informed networks:** A 2025–2026 paper uses a Kolmogorov–Arnold network with physics-informed loss to evolve cold dark matter in one dimension. It achieves sub-percent residual errors even after seven shell crossings, but current runtimes exceed those of N-body methods. It does not address a grid-based Poisson solve.
- **Neural force corrections:** A hybrid approach replaces the particle-mesh correction with a neural network implemented as a Fourier-space filter. This is the closest match to a learned gravity-solver component, though it is not multigrid and not AMR.
- **Emulators:** Field-level CNN emulators learn N-body evolution directly. One is fully differentiable, but it is an emulator rather than a solver inside a simulation code.

For a learned multigrid preconditioner specifically, a more targeted search (for example, on arXiv with terms like "learned multigrid" and "Poisson" and "AMR") would be needed.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
Web Searchmachine learning emulator N-body simulation zoom-in super-resolution map2map style generative dark matter cosmology review
OUT
Web search results for query: "machine learning emulator N-body simulation zoom-in super-resolution map2map style generative dark matter cosmology review"

Links: [{"title":"AI-assisted super-resolution cosmological simulations III: Time evolution","url":"https://arxiv.org/pdf/2305.12222"},{"title":"AI-assisted super-resolution cosmological simulations IV: An emulator for deterministic realizations","url":"https://arxiv.org/pdf/2408.09051"},{"title":"AI-assisted superresolution cosmological simulations","url":"https://www.pnas.org/doi/10.1073/pnas.2022038118"},{"title":"Separate Universe Super-Resolution Emulator","url":"https://arxiv.org/html/2605.09004v1"},{"title":"*Corresponding author: prithwineel.paul@iem.edu.in Deep Space Insights:","url":"https://www.epj-conferences.org/articles/epjconf/pdf/2025/10/epjconf_iemphys2025_01012.pdf"},{"title":"Masked Autoencoder Pretraining on Strong-Lensing Images for Joint Dark-Matter Model Classification and Super-Resolution","url":"https://arxiv.org/pdf/2512.06642"},{"title":"SuperBench: A Super-Resolution Benchmark Dataset for Scientific Machine Learning","url":"https://arxiv.org/pdf/2306.14070"},{"title":"[2010.06608] AI-assisted super-resolution cosmological simulations","url":"https://arxiv.org/abs/2010.06608"},{"title":"AI-assisted super-resolution cosmological simulations III: time evolution","url":"https://academic.oup.com/mnras/article/528/1/281/7492793"}]

The search didn't turn up a dedicated review of this exact topic, and nothing in the results mentions map2map by name. The results do cover the main threads of your query.

- **AI-assisted super-resolution series (Kodi Ramanah et al.):** The PNAS paper reports a GAN-based model trained on only 16 pairs of small-volume LR-HR simulations that reproduces high-resolution power spectra to percent level up to 16 h⁻¹Mpc. It was then deployed in a box 1,000 times larger than the training simulation box. Later papers in the series extend this to time evolution (Paper III, which notes that generating SR fields at different redshifts requires retraining) and to an emulator for deterministic realizations (Paper IV).
- **Separate-universe emulator (2026 preprint):** This work trains a generative adversarial network, allowing injected noise to be interpreted as stochastic structure, conditioned on a low-resolution field and cosmological parameters including Ω_k, Ω_m, σ_8, h, and redshift.
- **Broader context:** A 2025 conference paper summarizes ML upscaling of N-body simulations and notes that machine learning algorithms have reduced the computation resources. Its sourcing is thin, so treat it as a secondary overview. A general review of cosmological simulations (Vogelsberger et al. 2020) is cited in the Paper III introduction, but I didn't retrieve its content.

I did not find sources on map2map specifically, on diffusion-based or flow-matching approaches for dark matter fields, or on a single comprehensive review covering all of these. A follow-up search on "map2map" or "deep learning cosmological field mapping" would be the next step if you need those.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
Web Searchneural network solver Gross-Pitaevskii OR time-dependent Schrödinger equation PINN failure modes causal training discrete-time Runge-Kutta PINN
OUT
Web search results for query: "neural network solver Gross-Pitaevskii OR time-dependent Schrödinger equation PINN failure modes causal training discrete-time Runge-Kutta PINN"

Links: [{"title":"PINN-Based Framework for Soliton Solutions of Gross Pitaevskii and Nonlinear Schrodinger Equations","url":"https://arxiv.org/pdf/2607.22685"},{"title":"Causality-enhanced Discreted Physics-informed Neural Networks for Predicting","url":"https://www.ijcai.org/proceedings/2024/0497.pdf"},{"title":"Mitigating Propagation Failures in Physics-informed Neural Networks using Retain-Resample-Release (R3) Sampling","url":"https://arxiv.org/pdf/2207.02338"},{"title":"STVS-PINN: Segmented Training and Variable Scaling for Physics-Informed Neural Networks in Stiff PDEs","url":"https://link.springer.com/chapter/10.1007/978-981-92-3520-9_21"},{"title":"Neural Spectral Element Methods for stiff multiphysics PDEs with electrochemical transport benchmarks","url":"https://arxiv.org/pdf/2606.02335"},{"title":"A unified scalable framework for causal sweeping strategies for Physics-Informed Neural Networks (PINNs) and their temporal decompositions","url":"https://arxiv.org/pdf/2302.14227"},{"title":"Sub-Sequential Physics-Informed Learning with State Space Model","url":"https://arxiv.org/pdf/2502.00318"},{"title":"(PDF) Physics-Informed Neural Networks as Solvers for the Time-Dependent Schr\\\"odinger Equation","url":"https://www.researchgate.net/publication/364689503_Physics-Informed_Neural_Networks_as_Solvers_for_the_Time-Dependent_Schrodinger_Equation"},{"title":"RESPECTING CAUSALITY IS ALL YOU NEED FOR TRAINING","url":"https://www.osti.gov/pages/servlets/purl/2339526"},{"title":"Physics-Informed Neural Networks (PINNs). PDE Constraints in Loss Functions","url":"https://theorempath.com/topics/physics-informed-neural-networks"}]

The search turned up good coverage of PINN failure modes and causal training, but nothing that directly combines Gross-Pitaevskii or Schrödinger solvers with discrete-time Runge-Kutta schemes.

- **Failure modes:** Several sources point to gradient imbalance across loss terms and NTK-based convergence differences as causes of training instability (imbalance in the gradients of multiple loss terms could lead to poor convergence of PINNs). Krishnapriyan et al. give a failure-mode taxonomy, and one summary notes that vanilla training mixes early- and late-time residuals as if they were independent.
- **Causal training:** Wang et al. found that continuous-time PINNs trained by gradient descent are implicitly biased towards first approximating PDE solutions at later times. Their fix is a causal weighting scheme, which one paper writes as w_i = exp(-ε Σ_{j<i} L_j).
- **Schrödinger-specific work:** One paper on PINNs for the time-dependent Schrödinger equation reports that FCN-PINNs are not sufficient to reproduce complex dynamics, but this can be alleviated with techniques such as causal training. A 2025 arXiv paper applies PINNs to a Gross-Pitaevskii equation with time-dependent scattering length, though its details were not verified.
- **Discrete-time alternative:** The IJCAI 2024 paper proposes implicit time differencing schemes to enforce temporal causality, and use transfer learning to sequentially update the PINNs in space. This is the closest match to the discrete-time idea in your query, but the results don't show a Runge-Kutta variant specifically.

Gap: I found no result that tests discrete-time Runge-Kutta PINNs on GPE or TDSE problems. Searching for "discretized PINN Runge-Kutta" or "Schrödinger PINN Runge-Kutta" would be the next step.


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
Web Searchdifferentiable cosmological simulation JAX neural network hybrid solver learned correction coarse grid fuzzy dark matter OR wave dark matter
OUT
Web search results for query: "differentiable cosmological simulation JAX neural network hybrid solver learned correction coarse grid fuzzy dark matter OR wave dark matter"

Links: [{"title":"jaxion — jaxion","url":"https://jaxion.readthedocs.io/en/latest/"},{"title":"Jaxion: differentiable astrophysics simulations","url":"https://philip-mocz.medium.com/jaxion-differentiable-astrophysics-simulations-2ebc42be988d"},{"title":"Jaxion: A JAX package for Fuzzy Dark Matter Philip Mocz 1¶","url":"https://www.theoj.org/joss-papers/joss.09542/10.21105.joss.09542.pdf"},{"title":"DISCO-DJ II: a differentiable particle-mesh code for cosmology","url":"https://arxiv.org/pdf/2510.05206"},{"title":"pmwd: A Differentiable Cosmological Particle-Mesh $N$-body Library","url":"https://arxiv.org/pdf/2211.09958"},{"title":"Testing interacting dark energy with Stage IV cosmic shear surveys through differentiable neural emulators","url":"https://arxiv.org/pdf/2410.10603"},{"title":"Uncertainty-Aware Neural Networks for Fuzzy Dark Matter Model Selection from \\texorpdfstring{$x_{\\rm HI}$}{x_HI} Measurements","url":"https://arxiv.org/pdf/2509.05551"},{"title":"HIcosmo: a differentiable JAX-based framework for cosmology inference","url":"https://arxiv.org/html/2606.28175"},{"title":"Automatically Differentiable Weak Lensing Simulations for Full-Field Cosmological Inference","url":"https://dlanzieri.github.io/talks/Munich2023/"},{"title":"(PDF) JAX-COSMO: An End-to-End Differentiable and GPU Accelerated Cosmology Library","url":"https://www.researchgate.net/publication/370388352_JAX-COSMO_An_End-to-End_Differentiable_and_GPU_Accelerated_Cosmology_Library"}]

The search didn't turn up a paper that combines all of the requested pieces (a JAX-differentiable solver, a learned correction on a coarse grid, and fuzzy or wave dark matter). The closest matches are these:

- **Jaxion** is the strongest FDM match. It is a JAX-based research code for fuzzy dark matter simulations that uses automatic differentiation and multi-GPU acceleration. Its authors describe it as combining spectral, particle-mesh, and finite volume solvers, coupled through gravity. They also say it supports hybrid physics-ML modeling. This is a differentiable solver, but I didn't see a learned coarse-grid correction in the results.
- **Neural corrections on coarse particle-mesh solvers** appear in a few cosmology-only papers. One summary describes an additional effective force within the differential equations, parameterized by a Fourier-space neural network acting on the PM-estimated gravitational potential. A separate talk describes a hybrid physical-neural ODE trained on a CAMELS simulation to correct small-scale approximations. Neither addresses FDM.
- **pmwd** is a differentiable cosmological particle-mesh N-body library. It is a CDM-focused, JAX-based code that is relevant background, not an FDM solution.

For the specific combination, the next step would be searching arXiv for terms like "learned correction particle-mesh Schrödinger-Poisson" or "neural correction coarse grid wave dark matter."


REMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.
10:01 PM
This covers seven searches outside your group's SPINN and Cosmo-SPINN papers. I only saw abstracts and snippets, not full texts, so treat the details as leads to verify. The main result is that nobody I found has combined neural solvers with Schrödinger-Poisson at production scale. The useful ideas come from adjacent fields.

1. Time-evolution formulations that don't need space-time PINNs
Neural Galerkin schemes (survey, active learning, randomized sparse, sequential-in-time review). The network is trained sequentially in time. Each step projects the dynamics onto the tangent space of the network manifold (the Dirac-Frenkel principle).
This is the closest neural analogue to your discrete-time PINN, and it is a time integrator on a parametrised manifold.
It comes from quantum dynamics, so Schrödinger-type equations are a natural fit. I found no application to the Schrödinger-Poisson system.
The randomized-sparse variant and adaptive sampling address cost and non-uniform points.
Causal and discrete-time PINNs (causal weighting, causal sweeping, IJCAI 2024 implicit-scheme PINN). These address the late-time bias of continuous PINNs. I found no Runge-Kutta PINN tested on Gross-Pitaevskii or Schrödinger problems, so your scripts 6 and 7 may be filling a gap.
2. Optimisers for high accuracy (relevant to your lm result)
Energy natural gradient (Müller & Zeinhofer), Gauss-Newton natural gradient (Jnini et al.) and dual natural gradient.
These report near single-precision accuracy.
The dual form solves in residual space, which is what train_lm does when residuals are fewer than parameters.
Accelerated natural gradient, curvature-aware comparison and Gauss-Newton + Petrov-Galerkin. These address scaling, which is your bottleneck for 3D.
3. Neural operators and conservation
Conservation-preserved FNO tests linear and nonlinear Schrödinger. It reports near-zero norm error using an adaptive correction.
Structure-informed FNO for Hamiltonian PDEs projects onto the invariants after each rollout step.
Geometry-conditioned FNO for cubic NLS works on 2D periodic domains.
Inverse-scattering-inspired FNO covers time-dependent PDEs.
I found no dedicated unitary neural operator for NLS, and none for the coupled Poisson problem. That looks like an open niche.
4. Cosmology-side ML
Super-resolution and emulators.
AI-assisted super-resolution (PNAS), Paper III, time evolution, Paper IV, deterministic emulator and separate-universe emulator.
These are CDM-only. A wave-DM or FDM version would be new, but Cosmo-SPINN is in your group.
Hybrid physical-neural ODEs. Hybrid physical-neural ODEs for N-body and the field-level emulator learn a Fourier-space correction to a coarse particle-mesh solver. This is the clearest template for a learned correction in a wave solver.
Vlasov-Poisson with networks. Physics-informed KANs for cosmological Vlasov-Poisson reach sub-percent residuals in 1D after seven shell crossings, but they are slower than N-body.
Learned gravity solvers. I found no learned multigrid or neural Poisson solver inside an AMR code. The classical baselines are FFT on the root grid and multigrid on refined levels (Enzo and RAMSES).
5. Differentiable solvers
Jaxion is a JAX FDM code with spectral, particle-mesh and finite-volume solvers. It claims support for hybrid physics-ML models.
pmwd and DISCO-DJ II are differentiable particle-mesh codes for CDM.
Differentiable solvers are the natural base for learned corrections and for training through the simulation.
Where I'd look for a gap to go beyond
Neural Galerkin or manifold time stepping for Schrödinger-Poisson. I found none, and the network-as-integrator idea is where your discrete-time work sits.
A conservation-preserving learned correction on a coarse spectral or AMR wave solver. The ingredients exist separately (Jaxion, hybrid neural ODEs, conservation-projected FNOs), but I found no combination.
Natural-gradient training with matrix-free or structured curvature. This is needed to scale beyond 1D, and the dual form already fits your setup.
Limits of this search
Several results were abstract-level only. The "near-zero norm error" and "single-precision" claims are the papers' own and are unchecked.
The searches were US-only, and arXiv ids from 2026 were not independently verified.
I did not search for these topics: quantum-algorithm approaches to SP (one hit, arXiv 2307.06032, is unread), diffusion or flow-matching models for dark-matter fields, tensor-network or low-rank methods, or the standard AMR FDM codes beyond GAMER (SCALAR and others).
I can run a second round on any of those, or read one or two of these papers in full.