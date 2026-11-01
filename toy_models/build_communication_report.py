"""Build a standalone report from the scale communication experiments."""
import base64
import hashlib
import json
from html import escape
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'communication_results'


def read(name):return json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))
def fmt(x):return f'{x:.4g}' if isinstance(x,float) else escape(str(x))
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+fmt(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
def figure(name,caption):
    data=base64.b64encode((OUT/name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{escape(caption)}"><figcaption>{escape(caption)}</figcaption></figure>'


def main():
    lin=read('linear');src=read('source');sp=read('coupled');ctl=read('interface_controls');analytic=read('analytic_core');scattering=read('monochromatic_interface');audit=read('time_audit')
    def extent(rows,key):return f"{min(r[key] for r in rows):.3g} to {max(r[key] for r in rows):.3g}"
    linear_rows=[(p,extent([r for r in lin['rows'] if r['degree']==p],'initial_projection_error'),extent([r for r in lin['rows'] if r['degree']==p],'wave_error'),extent([r for r in lin['rows'] if r['degree']==p],'transmitted_phase')) for p in (4,6,8,10)]
    sp_rows=[(p,extent([r for r in sp['rows'] if r['degree']==p],'wave_error'),extent([r for r in sp['rows'] if r['degree']==p],'density_error'),extent([r for r in sp['rows'] if r['degree']==p],'potential_error')) for p in (4,6,8,10)]
    provenance=dict(experiment='Scale communication: translation, source, coupled SP and scattering controls',
                    files={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['scale_communication.py','shared_basis_review.py','frozen_hamiltonian.py','experiments.py']},
                    sources=[
                        dict(url='https://arxiv.org/abs/1605.00437',scope='Existing finite-element SP convergence analysis; different boundary conditions/time discretization.'),
                        dict(url='https://arxiv.org/abs/2103.00400',scope='Existing spectral-element accuracy analysis; not a prediction of the measured errors here.'),
                        dict(url='https://arxiv.org/abs/2007.01471',scope='Prior adaptive multiresolution DG nonlinear Schrodinger work; not a cosmological SP performance demonstration.')])
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>How scales communicate: controlled wave and gravity experiments</title>
<style>body{{font:17px/1.6 system-ui,sans-serif;color:#182634;background:#f5f7fa;margin:0}}main{{max-width:1100px;margin:auto;padding:35px;background:white}}h1,h2{{line-height:1.2}}h2{{margin-top:2em}}table{{border-collapse:collapse;width:100%;font-size:15px;margin:22px 0}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #d5dce3}}th{{background:#eaf1f8}}img{{width:100%;height:auto}}figure{{margin:25px 0}}figcaption,.small{{font-size:14px;color:#415568}}.box{{background:#edf5fa;padding:18px;border-left:5px solid #2b6b9c}}code,pre{{background:#f0f3f6}}pre{{padding:14px;overflow:auto}}a{{color:#155e91}}</style>
<main><h1>How scales communicate</h1>
<p>Third experimental round. Fixed nonuniform meshes, reversible coherent wave coordinates, full-density gravity, and independent controls for propagation and interfaces.</p>
<div class="box"><strong>Finding:</strong> a single global, conservative Galerkin system still has substantial propagation error at modest order. The uniform-mesh controls show that this cannot all be blamed on the refinement boundary. Accurate gravity can be communicated without a coarse wavefunction, but this does not repair kinetic dispersion. The coupled solver converges on this benchmark; no 3D efficiency advantage has been demonstrated.</div>

<h2>What was implemented</h2>
<p>The equations are i psi_t = -psi_xx/2 + Phi psi and Phi_xx = |psi|² - mean(|psi|²), on a periodic interval of length 20. Elements have width 0.5 on [-5,0] and width 1 elsewhere. Translating the entire initial condition by -2, 0, or +2 leaves the continuum problem unchanged but changes its placement relative to the mesh. The 25-element graded mesh has 25p degrees of freedom at polynomial degree p.</p>
<p>A continuous polynomial space supplies an exactly integrated mass matrix M and stiffness matrix K. Generalized eigenvectors of K/2 against M give orthonormal coordinates a and kinetic eigenvalues E. All physical wave details represented in this space are retained. There are no separately evolved coarse waves, patch boundary predictions, mesh changes, or level subcycling.</p>
<pre>psi at quadrature = Q a
b = Qᵀ W |Q a|²
Phi coefficients = G b,     G_j = -1/(2 E_j), G_0 = 0
i da/dt = E a + Qᵀ W [(Q G b) (Q a)]
energy = sum_j E_j |a_j|² + (1/2) bᵀ G b</pre>
<p>The source and potential force use the same quadrature and energy functional. Gauss quadrature with 2p+4 points per element integrates the polynomial nonlinear loads exactly, up to roundoff. Linear propagation is exact in time for the discrete operator. Nonlinear propagation uses DOP853 in the kinetic interaction picture; it integrates the projected nonlinear equation, avoiding an unjustified pointwise phase kick for a consistent finite-element mass matrix. Dense matrices and eigensolves make this an accuracy oracle, not a scalable solver.</p>

<h2>1. A well-projected wave can propagate badly</h2>
<p>A periodized Gaussian of width 0.7, centered at x=-2 and carrying phase exp(4i(x+2)), is evolved freely to T=1.25. Errors are physical L2 norms on independent 80-point Gauss quadrature. The reference is exact Fourier evolution; 2048 versus 4096 modes agree to {lin['reference_refinement']:.3g}. Each table interval spans the three translations.</p>
{table(['Degree','Initial projection error','Final wave error','Transmitted-window phase error (rad)'],linear_rows)}
<p>At degree 4 and shift zero, initial projection error is 0.000849, while final wave error is 0.1505. Discrete mass is conserved to roundoff. At degree 10, final errors fall below 2e-6 for all translations. Thus neither a good initial fit nor conservation certifies propagation accuracy. These measurements do not claim that low-order runs already meet an adequate resolution criterion.</p>
{figure('translation.png','Left: linear wave error under translations. Middle: negative-frequency error mass, a counterpropagation diagnostic rather than a pure reflection coefficient. Right: density error for the fully coupled core plus incoming wave.')}

<h2>2. The uniform controls prevent blaming everything on the interface</h2>
<p>The uniform coarse mesh uses width 1 everywhere; the uniform fine mesh uses width 0.5 everywhere. These match the two local resolutions of the graded mesh. Their DOF counts differ deliberately. The following comparison uses the unshifted packet.</p>
{table(['Degree','Mesh','DOFs','Final wave error','Evolution error with projected initial state'],[(r['degree'],r['mesh'],r['dofs'],r['wave_error'],r['propagation_error']) for r in ctl['rows'] if r['degree'] in (4,8)])}
<p>In the last column, exact Fourier evolution starts from the same projected initial field as the element method, isolating subsequent propagation error. The graded method is more accurate than the uniform coarse method here, but much less accurate than the uniform fine method. A large part of the discrepancy is consistent with insufficient propagation accuracy in coarse elements. This comparison does not uniquely partition the error into boundary scattering and bulk dispersion.</p>
<p>The negative-mode diagnostic projects the wave error onto physical Fourier modes k&lt;0. It is not reported as a pure reflection probability: finite-element bulk dispersion and projection can also populate this error channel. Doubling the diagnostic sampling changes the degree-6 generated negative-mode error by a relative {ctl['negative_mode_sampling_relative_difference']:.3g}.</p>
<p>The equal-DOF uniform Fourier solver is also retained in the linear JSON. At 100 DOFs its error is approximately roundoff. That is an accuracy baseline for a small smooth 1D problem, not a proposed affordable replacement for a 3D zoom simulation. The present 2:1 mesh and limited localized complexity do not demonstrate the memory savings sought in a cosmological calculation.</p>

<h2>3. A direct interface scattering control</h2>
<p>To measure actual interface reflection separately, a monochromatic wave scatters between two semi-infinite uniform element chains. Condensing the interior of each element in (K/2 - E M) gives a two-face dynamic stiffness with diagonal a and off-diagonal b. The discrete propagation angle satisfies cos(theta)=-a/b, and the outgoing flux factor is Z=-b sin(theta). Then r=(Z_left-Z_right)/(Z_left+Z_right), R=|r|² and T=(Z_right/Z_left)|1+r|².</p>
<p>All probes use k h_coarse &lt; pi on the first propagation band. This is an algebraic scattering calculation, with no packet projection or time error. The table fixes physical k=2.8 and a 2:1 refinement ratio. Ratios 4:1 and 8:1 are also included in the data.</p>
{table(['Degree','Reflection probability R','Coarse relative wavenumber error','Fine relative wavenumber error'],[(r['degree'],r['reflection_probability'],r['coarse_relative_wavenumber_error'],r['fine_relative_wavenumber_error']) for r in scattering['rows'] if r['physical_k']==2.8 and r['refinement_ratio']==2.])}
<p>Flux balance R+T=1 holds to {scattering['flux_balance_error']:.3g}. Conserving flux allows reflection; matching the discrete response suppresses it as order increases. The stationary k=2.8 probe is separate from the k=4 broadband packet, so its reflection coefficient cannot be substituted for the packet diagnostic. A high-order globally coupled interface need not have an irreducible reflection floor, but this does not remove bulk phase error or prove robustness for arbitrary 3D interfaces.</p>
{figure('scattering_and_controls.png','Left: genuine reflection coefficients at 2:1, 4:1 and 8:1 stationary interfaces. Values below 1e-16 are shown at the display floor. Right: propagation errors on graded, uniform coarse and uniform fine meshes, starting from the same projected field within each run.')}

<h2>4. Gravity needs the density information, not a coarse copy of the wave</h2>
<p>A Gaussian envelope multiplies two high-frequency carriers with a small wavenumber difference. Their density contains the envelope and long beat, while the carrier cancels. The full oscillatory wave is evaluated with 64-point Gauss quadrature per element. The same coarse Poisson operator receives either the correctly integrated full density or the density of a projected coarse wave.</p>
{table(['Degree','DOFs','Potential error: full-density source','Potential error: coarse-wave source','Mass retained in coarse wave'],[(r['degree'],r['dofs'],r['density_source_potential_error'],r['coarse_wave_source_potential_error'],r['represented_mass_fraction']) for r in src['rows']])}
<p>At degree 4, the full-density load gives potential error 2.49e-5, whereas forming the source from the coarse wave gives 0.153. The full fine wave was never assumed to be recoverable from that deliberately inadequate coarse projection. This control tests what an auxiliary gravity representation must receive; it does not propose evolving the physical wave in the coarse space. GAMER can separately restrict density, so this comparison is not evidence that it always forms the source from a squared restricted wave.</p>
{figure('source_and_profile.png','Left: source formation at fixed gravity discretization. Middle: the coupled core plus incoming packet at T=2. Right: density errors normalized by the reference peak; nearly identical-looking profiles can conceal different errors.')}

<h2>5. Fully coupled self-gravity</h2>
<p>A mass-4 numerical stationary core, shifted to x=-2 and boosted at velocity 4 pi/10, meets a coherent incoming Gaussian wave centered at x=2 with amplitude 0.3 and carrier -3. The complete superposition, including interference, is normalized to mass 4. The problem is evolved to T=2 for all three spatial translations. Unlike the earlier shared-basis test, gravity and wave evolution are now fully coupled.</p>
{table(['Degree','Wave error across translations','Density error across translations','Potential error across translations'],sp_rows)}
<p>The reference uses a fourth-order split Fourier solver. At fixed 1024 modes, halving dt from 0.002 to 0.001 changes the wave by {sp['reference']['time_wave']:.3g}. At fixed dt=0.002, increasing 1024 to 1536 modes changes it by {sp['reference']['space_wave']:.3g}. Space and time were refined separately. The finite-element method is a different spatial and temporal implementation.</p>
<p>An additional independent control evolves an isolated boosted stationary core and compares against its analytically translated and phase-rotated state. The degree-10 coupled solver gives wave error {analytic['wave_error']:.3g} and density error {analytic['density_error']:.3g} at T=1. This tests the nonlinear implementation without relying on another time integrator as the reference.</p>
<p>Across the coupled runs, maximum mass drift is {max(r['mass_drift'] for r in sp['rows']):.3g}, and maximum energy drift relative to max(1,|initial energy|) is {max(r['energy_drift'] for r in sp['rows']):.3g}. The directional derivative of the discrete energy agrees with the implemented Hamiltonian action to {max(r['energy_derivative_error'] for r in sp['invariants']):.3g}. These checks establish implementation consistency, not continuum accuracy.</p>
{table(['Degree','Wave change after tightening time tolerance'],[(r['degree'],r['time_refinement_wave_difference']) for r in sp['rows'] if 'time_refinement_wave_difference' in r])}
<p><strong>The smallest tabulated errors mix spatial and temporal effects.</strong> At degree 10, shift zero, tightening relative time tolerance from 2e-9 to 2e-11 changes the wave by 5.75e-8, comparable to its original 7.24e-8 error. The tighter run has wave error {audit['wave_error']:.3g} and density error {audit['density_error']:.3g}. Its initial projection error is {audit['initial_projection_error']:.3g}. Thus the coarse-order conclusion is robust, while the finest rows must not be interpreted as a clean measurement of spatial error alone. Only the unshifted degree-10 case received this extra tight audit; the translated rows retain the original tolerance.</p>

<h2>What this changes</h2>
<p><strong>Keep the two communication problems separate:</strong> gravity can use accurate density loads or source moments; coherent propagation requires accurate reciprocal coupling and dispersion. A single global discretization removes certain inconsistent transfers but does not, by itself, deliver the accuracy gain sought in GAMER. The tests support the user's concern that Galerkin methods retain important difficulties.</p>
<p><strong>Use the frequency response as an interface criterion:</strong> test reflected amplitude and transmitted phase at the same physical frequency on both sides. A conservative transfer can still couple two inaccurate propagation laws. The condensed interface response used above is a useful diagnostic, not a proposed production patch using a frequency-dependent solve for every mode.</p>
<p><strong>No method winner yet:</strong> the coherent high-order system converges rapidly, but uniform Fourier is much more efficient in spatial DOFs on the linear toy. The experiments do not measure 3D work, communication, cosmological expansion, long-time relaxation, or halo density profiles. They do not reproduce GAMER's Gram-FE operator or diagnose its halo discrepancy. A 2:1 static mesh is far milder than a deep zoom hierarchy. Time-dependent mesh transfers and different level timesteps remain untested.</p>
<p>The next useful comparison would apply the same dispersion and interface-response measurements to the actual Gram-FE patch operator, while keeping a separate full-density Poisson reference. That would test whether its advertised local accuracy survives coupling between patches, without replacing the whole solver on the strength of a toy result.</p>

<h2>Reproduce and inspect</h2>
<pre>&amp; 'C:\\Users\\TE\\anaconda3\\envs\\pinn_env\\python.exe' toy_models/scale_communication.py
&amp; 'C:\\Users\\TE\\anaconda3\\envs\\pinn_env\\python.exe' toy_models/build_communication_report.py</pre>
<p>Individual experiments can be selected with --only linear, source, coupled, controls, analytic, scattering, time_audit, or plots. New outputs are confined to communication_results and this page. Previous results and reports were not regenerated.</p>
<p><a href="scale_communication.py">Experiment source</a> · <a href="communication_results/linear.json">Linear data</a> · <a href="communication_results/interface_controls.json">Uniform controls</a> · <a href="communication_results/monochromatic_interface.json">Scattering data</a> · <a href="communication_results/source.json">Source data</a> · <a href="communication_results/coupled.json">Coupled data</a> · <a href="communication_results/analytic_core.json">Analytic core check</a> · <a href="communication_results/time_audit.json">Tight time audit</a> · <a href="communication_results/provenance.json">Code hashes and literature</a></p>
<p>Prior art establishes these method families, not a performance prediction for the present tests: <a href="https://arxiv.org/abs/1605.00437">finite-element Schrödinger–Poisson convergence</a>, <a href="https://arxiv.org/abs/2103.00400">spectral-element wave accuracy</a>, and <a href="https://arxiv.org/abs/2007.01471">adaptive multiresolution DG for nonlinear Schrödinger equations</a>.</p>
</main></html>'''
    (ROOT/'scale_communication.html').write_text(text,encoding='utf-8')
    print('Wrote scale_communication.html',len(text),'characters')


if __name__=='__main__':main()
