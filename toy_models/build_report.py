"""Build a standalone HTML research report from the saved toy experiments."""
import base64
from html import escape
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'
SHA='e36797538a1a5eaa456bc51ad44faf2b201a4aef'
GAMER=f'https://github.com/gamer-project/gamer/blob/{SHA}/'


def read(name):
    d=json.loads((OUT/f'{name}.json').read_text())
    return d.get('results',d)


def number(value):
    if isinstance(value,float): return f'{value:.3g}'
    return str(value)


def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join(f'<th>{escape(h)}</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{escape(number(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def figure(name,caption):
    data=base64.b64encode((OUT/name).read_bytes()).decode('ascii')
    return f'<figure><img src="data:image/png;base64,{data}" alt="{escape(caption)}"><figcaption>{caption}</figcaption></figure>'


def link(url,title): return f'<a href="{url}">{title}</a>'


def source(path,title,lines=''):
    return link(GAMER+path+lines,title)


def main():
    gravity=read('gravity'); sp=read('sp'); mapped=read('mapped'); msp=read('mapped_sp')
    bands=read('bands'); frozen=read('frozen'); amr=read('amr'); nu=read('nufft')
    linear=read('linear'); restriction=read('restriction')
    sections=[]
    sections.append(f'''
<header><p class="eyebrow">Exploratory numerical research · November 2026</p>
<h1>Accuracy and time integration for a cosmological wave solver</h1>
<p class="lead">The strongest direction from these experiments is to combine globally accurate gravity with propagation around a frozen reference Hamiltonian. The tests support that direction near equilibrium, while showing clear limits of fixed nonuniform grids and fixed carrier bands.</p>
<p>Ten small experiment groups isolate spatial operators, gravity boundary errors, time integration, and wave representation. All calculations are dimensionless and one dimensional; they are diagnostics for a future three dimensional solver. They do not establish a cosmological halo result or a production speedup.</p>
<nav><a href="#gravity">Gravity</a><a href="#audit">GAMER audit</a><a href="#time">Time steps</a><a href="#reference">Reference Hamiltonian</a><a href="#mesh">Nonuniform grids</a><a href="#bands">Fourier bands</a><a href="#research">Research direction</a><a href="#reproduce">Reproduce</a></nav></header>
<section><h2>What the experiments establish</h2>
{table(['Diagnostic','Measured outcome','Implication'],[
('Coupled SP at 128 points','FD2 gravity: 1.09e-3 density error; spectral: 1.41e-11','An accurate wave solver can be limited entirely by gravity.'),
('Refined interval with inherited potential','Fine residual below 3e-12; potential error up to 4.10e-3','Converging the local Poisson solve cannot repair its boundary data.'),
('Stationary nonlinear core, one step to T=2','Ordinary Strang: 0.673; reference splitting: about 1e-14','Preserving cancellation in the full Hamiltonian can remove an artificial time scale.'),
('Perturbed core, 32 steps','Ordinary fourth order: 4.87e-6; reference fourth order: 6.12e-7','Reference splitting improves this case, without proving a cost advantage.'),
('Fixed mapped grid','Large gain for a bound linear state; worse for expanding SP packet','Spatial concentration alone is insufficient.'),
('Narrow Fourier bands','Mass and energy accurate to about 1e-12, density substantially wrong','Invariants cannot replace solution convergence.')])}
</section>''')

    sections.append(f'''
<section id="gravity"><h2>Gravity can erase the benefit of accurate wave propagation</h2>
<p>The modal test uses ψ = exp(8ix) + 0.8 exp(−8ix) on [0, 2π). Both waves fit exactly in the wave basis. Their density contains a mode at 16, so the gravitational source requires twice the wave bandwidth. At three points per wave wavelength, a direct density sampled on that same grid aliases into the wrong mode. Poisson smoothing then magnifies the error because the aliased wave number is smaller.</p>
<p>With ten points per wave wavelength there are only five points per density fringe. For the second order Poisson symbol, the potential amplitude bias is analytically</p>
<pre>Φ_h / Φ = (k h / 2)² / sin²(k h / 2)
relative bias at five points per density wavelength = 0.142674...</pre>
{table(['Poisson operator','Relative potential error at ten wave points per wavelength'],[(r['operator'],r['potential_error']) for r in gravity if r['n']==80 and r['operator']!='spectral_padded_source'])}
<p>Zero-padding the represented wave coefficients before forming the density removes the initial two-wave aliasing even on the smaller grid. This adds source quadrature nodes, not independent wave information. In three dimensions, doubling nodes in all directions costs eight times the temporary grid volume. It also does not make a subsequent projected nonlinear kick exact or automatically unitary: the kick can create additional wave modes.</p>
{figure('gravity_transfer.png','Initial potential error, evaluated on an independent fine grid. Padded source evaluation separates nonlinear quadrature from wave degrees of freedom.')}
<p>The evolving test uses two colliding Gaussian packets with carriers ±8 × 2π/20, total mass 4, coupling 1, box length 20, and T = 2. The kinetic step is Fourier exact for every gravity variant. Fourth order splitting with Δt = 0.002 makes temporal error small. Only the Poisson symbol changes.</p>
{table(['Grid points','Spectral gravity','Second order gravity','Fourth order gravity','Sixth order gravity'],[(n,*[next(r['density_error'] for r in sp['resolution'] if r['n']==n and r['gravity']==g) for g in ('spectral','fd2','fd4','fd6')]) for n in (64,128,256)])}
<p>Errors are relative L2 density errors on the 1024 point reference grid. A reference with 1536 points and half the step differs by {sp['reference_density_disagreement']:.3g}. Importantly, the second order gravity case still conserves its own discrete energy at about 10⁻¹³. It accurately evolves a different spatially discretized system.</p>
{figure('sp_coupling.png','Left: gravity order limits the density accuracy. Right: fourth order time splitting gives better error per Poisson solve once the step is sufficiently small; very large steps can still resonate or fail to be accurate.')}
<p><strong>Interpretation:</strong> this reproduces a mechanism that could contribute to the resolution puzzle. It does not identify the cause of the published halo discrepancy. Interference spectra, orientation, density restriction, boundary coupling, and wave interpolation remain separate error sources.</p>
</section>''')

    sections.append(f'''
<section id="audit"><h2>What the public GAMER implementation shows</h2>
<p>The audit uses public main pinned at <code>{SHA}</code>. Your simulation branch, build options, and input files may differ. Only a source subset was read; GAMER was not compiled or executed.</p>
<ol>
<li><strong>Gravity has second order spatial accuracy in the inspected paths.</strong> The periodic FFT root solve explicitly divides by −4 Σ sin²(kᵢh/2), with a comment choosing consistency with the discrete Laplacian. The SOR path uses the seven point stencil and potential boundary interpolation from the parent. This is deliberate, not an implementation bug. See {source('src/SelfGravity/CPU_Poisson/CPU_PoissonSolver_FFT.cpp','root FFT solver','#L92-L97')} and {source('src/SelfGravity/CPU_Poisson/CPU_PoissonSolver_SOR.cpp','SOR residual','#L233-L242')}.</li>
<li><strong>A conditional density-loss edge case is reproducible.</strong> With standard complex restriction, eight child values [1, −1, 1, −1, 1, −1, 1, −1] have average density 1 and average complex amplitude 0. The rescaling guard sets the parent density to 0 whenever the averaged amplitude has zero norm, even if the conserved averaged density is positive. The reduced Python reproduction returns density {restriction['returned_coarse_density']:.0f} instead of {restriction['expected_coarse_density']:.0f}. See {source('src/Fluid/Flu_FixUp_Restrict.cpp','restriction guard','#L453-L475')}. This is a conservation defect in that branch; its frequency and practical effect on your runs are unknown. Wave-to-wave restriction with <code>OPT__RES_PHASE=1</code> follows another path. The safest investigation is to count positive-density entries into this guard and record their mass before choosing a fallback phase.</li>
<li><strong>Interface phase mismatch is already documented upstream.</strong> The {source('example/test_problem/ELBDM/RestrictionMismatch/README.md','RestrictionMismatch test')} reports that phase restriction remedies the immediate mismatch but does not remove later interface artifacts. This supports testing coupling independently of the interior wave propagator.</li>
<li><strong>The gravity timestep uses the absolute potential.</strong> The inspected {source('src/Model_ELBDM/ELBDM_GetTimeStep_Gravity.cpp','gravity step criterion','#L15-L38')} bounds the local rotation through max |Φ|. A spatially constant potential changes only a common phase on a fixed domain, so this criterion can be conservative. Any gauge change must be global and consistent across levels and subcycling; independent patch gauges can themselves corrupt phase matching. Removing this criterion alone does not remove splitting or interface errors.</li>
<li><strong>Relaxation convergence and spatial accuracy are different.</strong> The {source('src/SelfGravity/CPU_Poisson/CPU_PoissonSolver_SOR.cpp','CPU stopping rule','#L258-L267')} stops after the residual starts to increase, subject to a minimum iteration count. This does not certify an error tolerance. It is a diagnostic target, not evidence of premature stopping in your setup. More iterations also cannot correct the stencil or fixed boundary errors.</li>
</ol>
<p>The manufactured refined-interval test directly solves the fine finite-difference linear system. All variants use the exact fine source. Parent potentials are reconstructed with a periodic cubic spline for this diagnostic, rather than GAMER’s interpolation. Even an exact coarse spectral solve leaves interpolation error when used as boundary data.</p>
{table(['Boundary source','Fine potential error, 256 interior points','Max discrete residual'],[(r['boundary'],r['relative_potential_error'],r['max_discrete_residual']) for r in amr if r['fine_interior_n']==256])}
{figure('inherited_boundary.png','A tiny fine-grid residual can coexist with a boundary-induced accuracy floor. Composite gravity must correct coupling across levels, not only local residuals.')}
<p>No upstream implementation was modified. The cancellation reproduction does not establish that this edge case explains any halo result.</p>
</section>''')

    sections.append(f'''
<section id="time"><h2>Which timestep restrictions can actually be removed</h2>
<p>For one mode of a time independent Hermitian Hamiltonian, Crank–Nicolson is norm preserving for every step, but advances the phase with</p>
<pre>exact phase per step: −E Δt
CN phase per step:    −2 arctan(E Δt / 2).</pre>
<p>As E Δt grows, these disagree. Unconditional linear stability is therefore compatible with a very inaccurate density after modes interfere. Adding the same constant to every energy also distorts their relative numerical phases under unmodified CN, although physically it only adds a global phase.</p>
<p>The linear test uses a 96 point periodic Fourier Hamiltonian with V(x) = −15 cos x + 3 cos 2x and a superposition of three eigenstates. At 128 steps to T = 1, CN has density error 0.00121 without a common offset and 0.806 with an offset of 500. Analytically factoring out that known offset restores the original error. Strang is unchanged by the offset, but still has its ordinary splitting error.</p>
{figure('linear_time.png','Norm preservation does not imply accurate relative phases. A common energy shift exposes CN’s unnecessary sensitivity to fast global rotation.')}
<p><strong>Arbitrarily large exact steps are possible for a static linear Hamiltonian:</strong> exp(−iΔt H) gives the exact semi-discrete solution. The action test achieves density error 2.74 × 10⁻¹³ in one step, but takes 3444 counted Hamiltonian vector applications, including adjoint and norm-estimation work. Large step count reduction is not free work reduction. This uses SciPy’s exponential action algorithm, not a special low-rank halo model.</p>
<p>For time dependent or self-consistent H, a frozen exponential has temporal error. Rapid global phase can be removed; evolving interference, density redistribution, tidal changes, and nonlinear feedback still need an accuracy-controlled treatment. Existing {link('https://pmc.ncbi.nlm.nih.gov/articles/PMC7749873/','comparisons of adaptive splitting and commutator-free Magnus propagation')} provide relevant integrator machinery.</p>
</section>''')

    sections.append(f'''
<section id="reference"><h2>Propagate the equilibrium Hamiltonian together</h2>
<p>A stationary core satisfies (T + Φ[ψ₀])ψ₀ = μψ₀. Its density is constant even though its phase rotates. Splitting T and Φ separately destroys that exact cancellation at finite step size.</p>
<p>The tested alternative freezes a reference potential for the entire interval and splits the original equation as</p>
<pre>H_ref = T + diag(Φ_ref)
δΦ[ψ] = Φ[ψ] − Φ_ref
H[ψ] = H_ref + diag(δΦ[ψ]).

half kick with δΦ → full exp(−iΔt H_ref) → half kick with updated δΦ</pre>
<p>This keeps the original nonlinear equations. The reference only changes the integration split. The stationary state is preserved because its defect potential stays zero. A fourth order composition uses the same three Yoshida factors as the conventional method, so the two have equal stage counts. The cost of each reference exponential is different and must be measured separately.</p>
<p>The core is found by imaginary-time initialization followed by a nonlinear eigenproblem solve on 97 points, length 20, mass 4. Its stationary residual is {frozen['stationary_residual']:.3g}; comparison to a 193 point core gives relative density disagreement {frozen['ground_spatial_density_disagreement']:.3g}. For the perturbed case the initial state is multiplied by exp(0.2i cos(2πx/4)); the two time references disagree by {frozen['perturbed_reference_disagreement']:.3g}.</p>
{table(['Core','Method','Steps to T = 2','Relative density error'],[(c,m,s,next(r['density_error'] for r in frozen['rows'] if r['case']==c and r['method']==m and r['steps']==s)) for c,m,s in [('stationary','strang',1),('stationary','reference_strang',1),('perturbed','strang',32),('perturbed','reference_strang',32),('perturbed','yoshida4',32),('perturbed','reference_yoshida4',32),('perturbed','exponential_midpoint',32)]])}
{figure('frozen_hamiltonian.png','Reference splitting preserves the discrete equilibrium and improves the perturbed-core error in this test. Freezing the full potential at the start of every step is only first order for general evolution; the self-consistent midpoint variant recovers second order.')}
<p><strong>Why this is worth pursuing:</strong> a slowly changing full reference potential can absorb the large balancing terms inside a halo. Updating only its defect may allow accurate steps determined by departure from the reference rather than the largest local phase rotation.</p>
<p><strong>What is unproved:</strong> the reference is dense-diagonalized here. A cosmological box cannot do that. Matrix-free polynomial or rational exponential actions, recycled Krylov spaces, high-order local bases, and globally coupled boundaries would be needed. Their work can still grow with spectral width and step size. Violent mergers or rapidly changing granules may erase the gain. Reference refresh, cosmological coefficients, and moving core frames remain unimplemented. This is a promising adaptation of established interaction-picture ideas, not a claim of a new numerical principle.</p>
</section>''')

    sections.append(f'''
<section id="mesh"><h2>A nonuniform Fourier transform is not an adaptive discretization by itself</h2>
<p>The transform diagnostic uses the exact dense matrix Aⱼₖ = exp(ikxⱼ), with 41 modes and 100 samples. This removes NUFFT approximation tolerance as a possible explanation. A* is an adjoint, not an inverse. Quadrature weighting can improve it, but a genuine inverse needs a solve. When points leave large holes outside the refined region, even noiseless fitting becomes ill-conditioned.</p>
{table(['Point distribution','Condition number','Unweighted adjoint error','Least-squares coefficient error'],[(r['distribution'],r['condition'],r['adjoint_error'],r['least_squares_error']) for r in nu])}
<p>This matches the issue explained in {link('https://finufft.readthedocs.io/en/latest/tutorial/inv1d2.html','FINUFFT’s inverse-transform tutorial')}: sample gaps control conditioning. A global Cartesian Fourier mode set still has global bandwidth and, in three dimensions, a global mode count. NUFFT accelerates evaluations; it does not automatically turn that basis into an octree basis. A learned inverse cannot recover an arbitrary missing signal without additional assumptions.</p>
{figure('nonuniform_fourier.png','Exact nonuniform Fourier diagnostic. Errors in the strongly concentrated distributions are sampling/basis failures, not NUFFT library errors.')}
<p>The mapped alternative uses x(q) = L(q − α sin q)/(2π), with periodic q and Jacobian J. With u = √J ψ and Fourier differentiation D, the discrete operators are</p>
<pre>T_u = −(1 / (2√J)) D (1/J) D (1/√J)
A_P = −D (1/J) D
A_P Φ = −J (ρ − weighted_mean(ρ)).</pre>
<p>The kinetic matrix is Hermitian in u, so kinetic exponential actions preserve physical mass. The Poisson discretization is symmetric and includes its correct constant nullspace. Variable metric coefficients mean the operators are no longer diagonal in Fourier space. An odd grid avoids an extra Nyquist nullspace from squaring a first-derivative matrix.</p>
{table(['Test','Points','Uniform relative density error','Mapped α = 0.8 error'],[
('Localized linear bound state',65,*[next(r['density_error'] for r in mapped['rows'] if r['n']==65 and r['alpha']==a) for a in (0.,.8)]),
('Expanding self-gravitating packet at T=1',65,*[next(r['density_error'] for r in msp['resolution'] if r['n']==65 and r['alpha']==a) for a in (0.,.8)]),
('Expanding self-gravitating packet at T=1',193,*[next(r['density_error'] for r in msp['resolution'] if r['n']==193 and r['alpha']==a) for a in (0.,.8)])])}
{figure('mapped_spectral.png','A static localized bound state benefits strongly from concentrating the grid around it. The potential is −25 exp(−x²/(2 × 0.45²)); the reference uses 1025 uniform points.')}
{figure('mapped_sp.png','The expanding self-gravitating packet reverses the advantage. The fixed map overconcentrates points near the center and loses accuracy elsewhere. Physical mass is still conserved.')}
<p>A CDM-derived hierarchy can initialize a zoom region, but these results argue for error-based adjustment using the evolving wave and gravity. A moving map must include its mesh-velocity terms and remap errors. In three dimensions, simple separable maps also refine long coordinate slabs, which may waste work relative to local patches.</p>
<p>Coordinate-adaptive SP already has direct precedent: {link('https://journals.aps.org/prd/abstract/10.1103/PhysRevD.105.083521','Munive-Villa et al. (2022)')} apply moving meshes to stationary configurations and subhalo toy problems. {link('https://www.global-sci.com/cicp/article/view/6339','Kormann (2016)')} combines high-order elements, Magnus–Lanczos integration, and separate spatial/temporal error estimates. These establish related ingredients, not their performance for your cosmological zoom target.</p>
</section>''')

    sections.append(f'''
<section id="bands"><h2>Separated Fourier bands retain interference but need adaptive enrichment</h2>
<p>The speculative test represents ψ = Σⱼ aⱼ exp(ikⱼx) with bands around ±24, rather than every mode between them. Initial envelopes are exp(2 cos x), whose coefficients are known Bessel functions. The coherent wave is normalized to mean density 1; coupling is 20, box length 2π, and final time 0.5.</p>
<p>All represented cross-carrier terms are retained exactly:</p>
<pre>ρ_q = Σ_(k_i−k_j=q) a_i conjugate(a_j)
Φ_q = −g ρ_q / q²,  q ≠ 0
(V a)_i = Σ_j Φ_(k_i−k_j) a_j.</pre>
<p>This is a Galerkin truncation of a coherent wave, not an incoherent stream mixture or learned superresolution. A common carrier energy is factored out analytically. DOP853 integrates the coefficient ODE at relative tolerance 2 × 10⁻¹¹; it still needs many substeps. A 512 point fourth order reference and a 768 point half-step reference differ by {bands['reference_disagreement']:.3g}.</p>
{table(['Half bandwidth per carrier','Complex coefficients','Relative density error','Mass drift','Energy drift'],[(r['half_bandwidth'],r['complex_dofs'],r['density_error'],r['mass_error'],r['relative_energy_drift']) for r in bands['rows'] if r['half_bandwidth'] in (4,8,12,16,20)])}
{figure('fourier_bands.png','Strong self-gravity broadens the envelopes. Increasing the bands recovers the reference, but a representation that is initially almost exact can become inaccurate.')}
<p>Adding satellite bands at ±72 with the same narrow half bandwidth barely improves the dominant error and increases time-integration work. For example, half bandwidth 8 with four bands has error about 0.0831, essentially the same as the two-band case. Widening the primary bands is more useful here than adding distant carriers.</p>
<p>The promising version would select both local spatial support and momentum bands from a residual, retaining phase-coherent cross terms and allowing bands to merge into a full wave representation. In a virialized halo the local momentum distribution may fill the available volume, eliminating compression. This toy does not establish compression at fixed high accuracy, and its O(M²) dense convolution is not a proposed production algorithm.</p>
<p>Related representation work includes {link('https://doi.org/10.1063/1.5127856','time-reversible adaptive phase-space grids')} and {link('https://doi.org/10.1016/j.jcp.2024.113029','spawning Hagedorn wavepackets')}. Both show that adapting phase-space representations is an established area. This search does not certify novelty of the proposed coherent, gravity-coupled adaptation.</p>
</section>''')

    sections.append(f'''
<section id="research"><h2>The research direction I would prioritize</h2>
<p><strong>First, make the gravitational source and global elliptic solve independently trustworthy.</strong> On an adaptive hierarchy, form gravity sources from the best available leaf densities with conservative restriction of density or moments. Avoid making those sources depend on a coarse complex wavefunction that cannot represent the fine interference. Resolve both global low-frequency gravity and fine near-field corrections in one consistent operator.</p>
<p>A high-order composite spectral-element Poisson solve with matching fluxes is one concrete option. Another is periodic coarse FFT gravity plus a controlled fine-density correction through multipoles, FMM, or multiresolution Green-function convolution. The fine correction must contribute back to the coarse/global field; merely tightening independent patch solves is insufficient. Boundary conditions and the density-to-potential operator need their own convergence tests.</p>
<p><strong>Then test propagation around a reference full Hamiltonian on that same spatial representation.</strong> Use high-order elements or another local orthogonal basis, global weighted unitarity, and matrix-free reference exponential actions. Refresh the reference on a time slab when its defect grows. Control error with the evolving defect and step doubling, rather than an absolute phase bound alone. Keep a single consistent global phase convention across the hierarchy.</p>
<p>A research contribution could lie in this combination: accurate composite gravity sourced directly from leaf information, equilibrium-preserving reference propagation, and adaptivity driven by errors relevant to halo profiles. The individual components are established; novelty would need a more focused literature and implementation comparison.</p>
<p><strong>Keep phase-space compression as the higher-risk branch.</strong> It may help smooth inflows and a few coherent streams, while dense virialized regions use a full wave basis. The band experiments currently show a failure mode rather than a convincing speed/accuracy win. Fixed mapped Fourier grids also need a wave-based refinement criterion; CDM alone cannot guarantee coverage of all relevant wave structure.</p>
<p>Useful adjacent work is {link('https://arxiv.org/abs/1507.01888','MADNESS')}, which provides adaptive multiwavelet representations, separated operator kernels, and a parallel runtime. It is evidence that nonuniform accuracy-controlled operator application can be practical, not a ready-made cosmological SP solution. {link('https://arxiv.org/html/2405.04924v2','Schwersenz et al. (2024)')} already study adaptive fourth order splitting for 1D cosmological SP; simply replacing Strang with higher order would therefore be a baseline improvement, not a new method.</p>
<p>The halo validation target should remain the full profile and its convergence. The {link('https://arxiv.org/html/2412.09908v3','supplied halo study')} requires resolution across the surrounding halo and compares to matched N-body results outside the core. Its supplement quotes 6–12 cells per granule with granule size about 0.35 λ_dB: roughly 17–34 cells per that definition of λ_dB. A plane-wave points-per-wavelength number must be compared using consistent velocity and length definitions.</p>
<p>Next decisive tests would be a periodic 3D manufactured gravity field on a two-level mesh; a moving nonlinear core crossing a refinement interface; a core interacting with a resolved wave bath; and finally matched-CDM halo-profile convergence. Before a large implementation, measure reference exponential work and gravity error on those tests. More available processors do not remove global communication costs or a representation’s intrinsic complexity.</p>
</section>''')

    sections.append('''
<section id="reproduce"><h2>Reproduce and inspect</h2>
<p>All files are under <code>toy_models/</code>. No original outputs were changed. The suite uses the existing NumPy, SciPy, and Matplotlib environment, with no extra dependencies. Downloaded GAMER sources are ignored locally; their pinned provenance is saved.</p>
<pre>$py = 'C:\\Users\\TE\\anaconda3\\envs\\pinn_env\\python.exe'
&amp; $py toy_models/experiments.py
&amp; $py toy_models/representations.py
&amp; $py toy_models/frozen_hamiltonian.py
&amp; $py toy_models/verify.py
&amp; $py toy_models/build_report.py</pre>
<p>The full suite takes roughly a few minutes on this machine. The report’s figures are embedded and need no network. See <a href="README.txt">README.txt</a> for individual experiments and the source-fetch command. Raw files: <a href="results/gravity.json">gravity</a>, <a href="results/amr.json">boundary test</a>, <a href="results/restriction.json">restriction</a>, <a href="results/nufft.json">nonuniform transform</a>, <a href="results/linear.json">linear time integration</a>, <a href="results/sp.json">coupled SP</a>, <a href="results/mapped.json">mapped bound state</a>, <a href="results/mapped_sp.json">mapped SP</a>, <a href="results/bands.json">carrier bands</a>, <a href="results/frozen.json">reference Hamiltonian</a>.</p>
<p>The independent checks verify the uniform limit of the mapped operator, its nullspaces and Hermiticity, the sparse nonlinear convolution against overintegrated FFT evaluation, the Hamiltonian energy gradient, temporal convergence rates, reference refinement, and weighted mass. A small residual or conserved invariant is never used alone to claim physical accuracy.</p>
<p>Limitations: these are fixed-scale-factor periodic 1D systems. Their Poisson kernel is not spherical 3D gravity, and the stationary state is a 1D analogue rather than a cosmological soliton. Dense eigenproblems and exact transforms are diagnostic oracles. No distributed implementation, learned operator, full GAMER execution, halo profile, or 3D cost scaling was measured.</p>
</section><footer>Generated from local experiment results. Research proposals are distinguished from measured outcomes; no claim of production superiority or certified novelty is made.</footer>''')
    css='''body{font:17px/1.65 system-ui,-apple-system,Segoe UI,sans-serif;color:#183044;background:#f3f6f8;margin:0}main{max-width:1120px;margin:auto;background:white;padding:50px 60px}h1{font-size:2.5rem;line-height:1.15;color:#102b40}h2{font-size:1.6rem;line-height:1.3;color:#102b40;margin-top:0}p{max-width:92ch}.lead{font-size:1.2rem}.eyebrow{color:#526b7b;font-size:.9rem}section{padding:35px 0;border-top:1px solid #dbe5eb}nav{display:flex;flex-wrap:wrap;gap:10px 24px;margin:30px 0}a{color:#086f8a;text-underline-offset:3px}table{width:100%;border-collapse:collapse;font-size:.92rem}th,td{text-align:left;padding:12px 14px;border-bottom:1px solid #dbe5eb;vertical-align:top}th{background:#edf4f7}td:first-child{font-weight:500}.table{overflow-x:auto;margin:25px 0}pre{background:#edf4f7;padding:20px;overflow:auto;font-size:.94rem}code{font-size:.9em;word-break:break-word}figure{margin:30px 0}img{width:100%;height:auto}figcaption{font-size:.9rem;color:#526b7b}li{margin:16px 0}footer{font-size:.85rem;color:#526b7b;border-top:1px solid #dbe5eb;padding-top:20px}@media(max-width:760px){main{padding:24px 18px}h1{font-size:2rem}body{font-size:16px}}@media print{body{background:white}main{padding:0}section{break-before:auto}figure,table{break-inside:avoid}}'''
    document='<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Accuracy and time integration for a cosmological wave solver</title><style>'+css+'</style></head><body><main>'+''.join(sections)+'</main></body></html>'
    (ROOT/'report.html').write_text(document,encoding='utf-8')
    print(f'Wrote {ROOT/"report.html"} ({len(document):,} characters)')


if __name__=='__main__': main()
