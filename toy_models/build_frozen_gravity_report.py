"""Standalone report for the frozen-source and long-run follow-up."""
import base64
import json
from html import escape
from pathlib import Path
from experiments import np,plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'frozen_gravity_results'


def load(name):return json.loads((OUT/(name+'.json')).read_text())
def table(headers,rows):
    def cell(x):return f'{x:.6g}' if isinstance(x,float) else escape(str(x))
    return '<table><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+cell(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def pic(name,caption):
    return '<figure><img alt="'+escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'"><figcaption>'+escape(caption)+'</figcaption></figure>'


def main():
    g=load('frozen_gravity');k=load('frozen_potential');r=load('runtime_long');p=load('phase_control')
    fig,ax=plt.subplots(figsize=(8,4))
    for kind in ('stored_matmul','fft_runtime'):
        rr=[x for x in r['rows'] if x['kind']==kind]
        ax.semilogy([x['time'] for x in rr],[x['mass_ratio'] for x in rr],'o-',label=kind)
    ax.axhline(1,color='black',ls='--',label='Exact norm / periodic FFT')
    ax.set(xlabel='Dimensionless physical time',ylabel='Mass / initial mass',title='Repeated runtime updates, 10 cells per initial wave')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(OUT/'runtime_mass.png',dpi=160);plt.close(fig)
    uni=[]
    for n in (50,100,200):
        rr=[x for x in g['uniform'] if x['n']==n]
        uni.append([n/10]+[f"{100*min(x[key] for x in rr):.3g}–{100*max(x[key] for x in rr):.3g}%" for key in ('potential','force','tidal')])
    finest=[x for x in g['nested'] if x['seed']==7 and x['level']==6]
    norm=[x for x in r['rows'] if 20<x['time']<60]
    rates=[x for x in k['rows'] if x['n']==160 and x['dt_factor']==1 and x['kind']!='periodic_fft']
    control=[x for x in p['rows'] if x['n']==160 and x['dt_factor']==1 and x['potential_ratio']==.2]
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gravity resolution and long-run Gram-FE stability</title>
<style>body{{font:17px/1.6 system-ui;background:#edf3f7;color:#183248;margin:0}}main{{max-width:1100px;margin:auto;padding:36px;background:white}}h1,h2{{line-height:1.2}}h2{{margin-top:2em}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d8e1e8;text-align:left}}th{{background:#e8f1f7}}img{{width:100%}}figure{{margin:25px 0}}figcaption{{font-size:14px;color:#476075}}.box{{background:#e9f4f9;padding:18px;border-left:5px solid #307da9}}a{{color:#155e91}}pre{{overflow:auto;background:#eef3f7;padding:15px}}</style><main>
<h1>Can gravity explain five versus ten cells per wave?</h1>
<p>Follow-up to <a href="gamer_diagnosis.html">the pinned GAMER audit</a>. New results are isolated in <code>frozen_gravity_results/</code>. No existing results were regenerated.</p>
<div class="box"><strong>Yes, it is quantitatively plausible, but not established as the cause of the halo failure.</strong> Fully resolved coherent fields have 18–20% RMS Hessian error with FD2 gravity at five cells per wave, versus about 4% at ten. Separately, an exactly prescribed potential causes much stronger long-run growth in both reconstructed Gram-FE variants than the free-wave test revealed. This latter failure needs no inaccurate Poisson solve or AMR interface.</div>

<h2>1. Wave accuracy is not gravity accuracy</h2>
<p>For counterpropagating waves, ψ = exp(ikx) + exp(−ikx), density contains a mode q=2k. A wave resolved by p cells produces a density fringe resolved by p/2 cells. For an axis-aligned Fourier mode, the FD2 potential relative to the continuum solution is [qh/(2 sin(qh/2))]². This is a discretization error even if the Poisson linear system is solved to roundoff.</p>
{table(['Cells per wave','Cells per density fringe','FD2 potential amplitude bias'],[(x['wave_ppw'],x['density_ppw'],f"{100*x['fd2_potential_bias']:.2f}%") for x in g['modal']])}
<p>These are individual fringe-mode errors, not errors in a halo profile. The inverse Laplacian attenuates high-frequency potential modes. A single boosted smooth core is also a poor test of this mechanism: boosting changes the carrier wavelength without changing density or its gravitational field. Multistream interference is needed to relate carrier resolution to gravity resolution.</p>

<h2>2. A frozen multidimensional interference field</h2>
<p>The test uses eight coherent plane waves with three deterministic sets of phases, multiplied by a smooth envelope on a periodic 2D box of side 2π. Mean density is one. The maximum component wave number is K=10, the density bandwidth is 2K=20, and the smallest grid is 50². Thus even at five cells per shortest wave, every density mode is resolved. Finite Fourier convolution gives the exact density and continuum potential, without a large reference simulation. This is a smooth enveloped wave field, not an equilibrated halo.</p>
{table(['Cells per wave','Potential RMS error','Force RMS error','Full Hessian RMS error'],uni)}
<p>Derivatives in this uniform comparison are analytic Fourier derivatives of the discrete potential's Fourier interpolant. They diagnose the gravity field used by wave evolution, without adding a force-difference stencil. The full Hessian includes trace and shear; it is not only the tracefree tidal tensor. Potential errors below 0.2% coexist with Hessian errors near 20%. These data make a higher refinement requirement plausible, but do not supply a dynamical halo-stability threshold.</p>
{pic('frozen_gravity.png','Three frozen coherent fields on uniform grids, then a six-level exact-source hierarchy. The right panel shows inherited tracefree Hessian error for one field.')}

<h2>3. Six levels with exact fine density</h2>
<p>Every child receives the exact analytic density and a direct FD2 Dirichlet solve. Only the boundary potential comes from the parent. Root FD2 and spectral solves, quadratic and quartic boundary transfer, and exact boundary data form separate controls. This is a point-centered hierarchy model, not GAMER's cell-centered interpolation implementation. There is no wave interpolation, no temporal interpolation, and no relaxation stopping error.</p>
{table(['Root cells/wave','Root solver','Boundary degree','Level-6 Hessian error','Exact-boundary error','Inherited shear'],[(x['root_wave_ppw'],x['root_symbol'],x['degree'],x['tidal_error'],x['exact_boundary_tidal_error'],x['inherited_shear']) for x in finest])}
<p>For the first phase seed, a five-cell root with FD2/quadratic transfer leaves 0.135% fine Hessian error, versus 0.00167% with exact boundaries. Doubling the whole hierarchy reduces the former to 0.0345%. The inherited difference is discrete harmonic: a small residual cannot identify it. Increasing boundary interpolation order helps here but does not remove fine FD2 error.</p>
<p>Those numbers average over the patch interior. To check that boundary-adjacent cells are not the whole effect, the center Hessian error is also recorded: 0.109% versus 0.000266% with exact boundaries for the five-cell root; doubling the hierarchy lowers the former to 0.0219%.</p>
<p><strong>Negative result:</strong> using a spectral root does not consistently improve the deepest patch in this test; local FD2 solves and inherited errors can partly cancel. Improving one piece need not monotonically improve a selected observable. The six-level error is far smaller than the root-grid fringe error, and this test does not reproduce a large halo-profile discrepancy or failure under whole-hierarchy refinement.</p>

<h2>4. Nonlinear bandwidth is a distinct issue</h2>
<p>Even when ψ and |ψ|² are resolved, the instantaneous nonlinear term Φψ can extend to 3K. At five cells per wave, its extreme modes exceed the grid Nyquist frequency. Exact coefficient convolution measures an aliased contribution of 0.069–0.074% relative to the full nonlinear RHS in these fields. At ten cells it is absent for this instantaneous cubic product. This does not guarantee absence of aliasing after evolution: the exponential potential kick and subsequent dynamics can create additional modes. The small measured alias is a possible seed, not an explanation of a large density error by itself.</p>

<h2>5. A small long-run failure without a Poisson error</h2>
<p>A separate 1D test evolves a smooth standing wave with carrier k=16 in V(x)=A cos(32x+0.37). The carrier energy is ω=128. A/ω is varied over 0, 0.05 and 0.2. The periodic box has length 2π; grids 80 and 160 give five and ten cells per initial wave. The largest potential phase per step in this scan is 0.0403 radians, so these results do not require a large potential kick. The time step is the audited default λh² or half that step.</p>
<p>Both real potential kicks and the periodic FFT kinetic update are unitary on the same grid, so periodic FFT Strang conserves norm regardless of splitting error. The patched Gram-FE kinetic map is not unitary in the standard grid norm. Potential phases can couple its modes and change long-run amplification. The potential is exact and prescribed: this experiment isolates the composition of wave propagation and a nonuniform potential, not gravity-solver inaccuracy.</p>
{table(['Variant','A/ω','Largest amplitude growth rate per physical time','Amplitude e-fold time'],[(x['kind'],x['potential_over_carrier_energy'],x['amplitude_growth_rate_per_physical_time'],x['efold_time']) for x in rates])}
{pic('frozen_potential.png','Growth rates use physical time, not only growth per step. Halving the time step or doubling resolution does not uniformly reduce the largest growth rate.')}
<p>The spectrum alone would be insufficient: weakly excited modes can remain invisible. I therefore repeated 254,648 actual updates from the ordinary smooth initial state at ten cells per wave, A/ω=0.2. The MATMUL column operation and the factorized FFT projection/extension/FFT sequence both eventually grow. The exact mass ratio is one.</p>
{table(['Runtime variant','Time','Updates','Mass ratio'],[(x['kind'],x['time'],x['steps'],x['mass_ratio']) for x in norm])}
{pic('runtime_mass.png','Actual repeated updates, not eigenvalue extrapolation. At time 25 both variants still look almost mass-conserving; much later both fail. The large final numbers demonstrate instability and are not physically meaningful predictions.')}
<p>At T=100 the measured mass ratios are about 8.73×10¹⁹ (MATMUL) and 6.92×10¹² (FFT); exact values depend on seeding and arithmetic. Matrix-power predictions agree closely with MATMUL and within about 3% in FFT mass at this late time. A deliberately initialized growing eigenmode also agrees with the predicted 3,000-step growth to better than 10⁻⁶ relatively in FFT. The periodic FFT controls conserve mass to better than 10⁻⁸ over the tested interval.</p>
<p>The initial wave has ten cells per wavelength, but the potential generates shorter sidebands. We have not demonstrated that all appreciably populated modes remain ten-cell resolved. The model also lacks GAMER mass corrections, floors, AMR, HJM transitions, and 3D sweep ordering. A=25.6 is a chosen stress test, not a measured halo potential. Nevertheless, mere under-resolution or aliasing cannot make the unitary same-grid FFT map gain norm: nonunitary amplification is independently present in the patched map.</p>
<p>Inspection of the fastest-growing eigenmode finds dominant wave number 44, or 3.64 cells per wavelength, in both variants; more than 99.97% of its Fourier power lies at wavelengths shorter than five cells. This qualifies the result: a well-resolved initial carrier is not itself the unstable mode. The failure involves growth of shorter components that the full discrete evolution can seed.</p>

<h2>6. Replacing the Taylor phase does not fix this example</h2>
{table(['FFT patch control','Amplitude growth rate, n=160, A/ω=0.2'],[(x['phase'],x['growth_rate']) for x in control])}
<p>The control replaces the truncated kinetic Taylor phase with exp(−ik²dt/2) while retaining extension, crop and filtering. It leaves the growth essentially unchanged. Double-precision basis reconstruction differs from the high-precision fused map in operator norm by about 3×10⁻⁶; the same-basis Taylor and exact-phase controls make the comparison fair. Free-wave growth at very small rates is more sensitive to this reconstruction. This is evidence against the Taylor truncation being the main cause of the much larger frozen-potential growth; it is not a proof assigning blame to one particular table or boundary operation.</p>

<h2>7. Interpretation and next discriminating experiment</h2>
<p>Gravity can plausibly demand ten rather than five cells per wave because its source has shorter fringes and the FD2 Green function is inaccurate there. However, the stronger new result is that accurate prescribed gravity can still destabilize the patched kinetic evolution. A Poisson-only diagnosis is therefore premature. The next useful production comparison is this small prescribed-potential test in compiled GAMER, with actual mass corrections and precision, followed by a self-consistent variant and a scan of potential amplitude and spatial scale. Check wave spectra as well as mass: rescaling mass could hide amplification while the profile changes.</p>
<p>The previous interpolation result concerned linear interpolation in time between coarse complex states, not Gram-FE spatial interpolation. The user's observation with density/phase interpolation rules it out as a universal explanation. Neither new experiment uses temporal interpolation. No density/phase interpolation cure is recommended for vortices.</p>

<h2>Reproduce and inspect</h2>
<pre>python toy_models/frozen_interference_gravity.py
python toy_models/gramfe_frozen_potential.py
python toy_models/gramfe_frozen_potential.py --runtime-long
python toy_models/gramfe_phase_control.py
python toy_models/build_frozen_gravity_report.py</pre>
<p>Run <code>python toy_models/verify_frozen_gravity.py</code> for artifact checks, norm-control verification and growing-mode bandwidth diagnostics.</p>
<p>Use the repository's pinn_env Python. Dependencies are already installed. The 2D frozen-source sweep took {g['seconds']:.1f} seconds; repeated runtime updates take longer but remain a small CPU experiment. Raw data: <a href="frozen_gravity_results/frozen_gravity.json">gravity</a>, <a href="frozen_gravity_results/frozen_potential.json">fixed-map stability</a>, <a href="frozen_gravity_results/runtime_long.json">runtime verification</a>, <a href="frozen_gravity_results/phase_control.json">phase control</a>. Analytic source/product identities and an independent FFT Poisson calculation agree within {max(g['checks'].values()):.3g}. Direct residuals and discrete harmonic checks are asserted in the script.</p>
<p>Implementation provenance remains pinned to <a href="https://github.com/gamer-project/gamer/tree/e36797538a1a5eaa456bc51ad44faf2b201a4aef">GAMER e367975</a>, using the <a href="https://github.com/gamer-project/gamer/blob/e36797538a1a5eaa456bc51ad44faf2b201a4aef/src/Model_ELBDM/CPU_ELBDM/CPU_ELBDMSolver_GramFE_FFT.cpp">FFT implementation</a> and <a href="https://github.com/gamer-project/gamer/blob/e36797538a1a5eaa456bc51ad44faf2b201a4aef/src/Model_ELBDM/ELBDM_GramFE_EvolutionMatrix.cpp">MATMUL builder</a>. The <a href="https://arxiv.org/html/2411.17288v2">hybrid-scheme paper</a> supplies the method context; all numerical claims above are from these new toy experiments.</p>
</main></html>'''
    (ROOT/'frozen_gravity.html').write_text(body,encoding='utf-8')
    print('Wrote toy_models/frozen_gravity.html')


if __name__=='__main__':main()
