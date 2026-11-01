"""Report the pinned Gram-FE reconstruction and gravity/coupling controls."""
import base64
import json
from html import escape
from pathlib import Path
from experiments import np,plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'gramfe_results'
SHA='e36797538a1a5eaa456bc51ad44faf2b201a4aef'
def load(name):return json.loads((OUT/(name+'.json')).read_text())
def fmt(v):return f'{v:.4g}' if isinstance(v,float) else escape(str(v))
def table(headers,rows):return '<table><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+fmt(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
def pic(name,caption):return '<figure><img alt="'+escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'"><figcaption>'+escape(caption)+'</figcaption></figure>'
def source(path,label,line=None):return '<a href="https://github.com/gamer-project/gamer/blob/'+SHA+'/'+path+('' if line is None else '#L'+str(line))+'">'+escape(label)+'</a>'


def main():
    d=load('dispersion');g=load('gravity_hierarchy');s=load('selfgravity');b=load('boundary_time');r=load('boundary_refinement');p=load('precision_audit');long=load('long_run_spectrum');runtime=load('runtime_fft_check')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for kind in ('stored_matmul','fft_fused'):
        rr=[v for v in d['one_step'] if v['kind']==kind]
        axs[0].loglog([v['ppw'] for v in rr],[v['relative_error'] for v in rr],'o-',label=kind)
        rr=[v for v in d['bloch'] if v['kind']==kind]
        axs[1].loglog([v['ppw'] for v in rr],[v['wave_error'] for v in rr],'o-',label=kind)
    for ax,title in zip(axs,('One patch update','10,000 same-level updates')):
        ax.set(xlabel='Points per wavelength',ylabel='Relative wave error',title=title);ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'dispersion.png',dpi=160);plt.close(fig)
    sample=[v for v in s['rows'] if v['n']==128 and v['boost']==12 and v['dt_factor']==.5]
    hierarchy=[v for v in g['rows'] if v['root_n']==32 and v['root_symbol']=='fd2' and v['interpolation_degree']==2]
    linear=[v for v in b['rows'] if v['coarse_ppw']==10 and v['boundary']=='complex_linear']
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GAMER Gram-FE and gravity: diagnostic toy models</title>
<style>body{{font:17px/1.6 system-ui,sans-serif;background:#f4f7fa;color:#172e42;margin:0}}main{{max-width:1120px;margin:auto;padding:35px;background:white}}h1,h2{{line-height:1.2}}h2{{margin-top:2em}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}td,th{{padding:9px;border-bottom:1px solid #d2dde5;text-align:left}}th{{background:#e9f1f7}}img{{width:100%}}figure{{margin:25px 0}}figcaption,.small{{font-size:14px;color:#476075}}.box{{background:#eaf4fa;border-left:5px solid #286f9f;padding:18px}}pre{{background:#f0f3f6;padding:15px;overflow:auto}}a{{color:#155e91}}</style><main>
<h1>GAMER: separate the patch operator, gravity, and level coupling</h1>
<p>Pinned public commit <code>{SHA}</code>. Python reconstructions and controlled problems; no compiled GAMER run and no reproduction of a cosmological halo.</p>
<div class="box"><strong>Result:</strong> the actual-table Gram-FE kinetic update is accurate for well-resolved smooth waves, while second-order gravity dominates a moving-core error budget. A six-level Poisson toy develops inherited tidal error that a 1D Poisson test cannot represent. A separate 1D boundary-time test produces percent-level error in both variants that survives doubling the fine resolution, but only for complex temporal interpolation. Weakly growing short-wave patch modes and a MATMUL table-precision floor remain additional findings, not established causes of the user's halo discrepancy.</div>

<h2>What was reconstructed, and what was not</h2>
<p>The kinetic column update uses GAMER's actual patch-size-8 tables: 16 physical cells, 8 ghost cells on each side, 14 boundary samples and 32 extension cells. The default source sets DT__FLUID=0.2, which produces dt/(eta dx²)=4(0.2)/pi={d['lambda_dt_over_dx2']:.6g}. The implemented Taylor polynomial has powers 0 through 6. MATMUL uses exp[-16 log(10)(k/kmax)^100]; FFT uses exp[-32 log(10)(k/kmax)^200]. These are reconstructed as written, rather than assumed identical. See {source('include/Macro.h','mesh constants',715)}, {source('src/Init/Init_ResetParameter.cpp','default timestep',117)}, {source('src/Model_ELBDM/ELBDM_GramFE_EvolutionMatrix.cpp','MATMUL builder',91)} and {source('src/Model_ELBDM/CPU_ELBDM/CPU_ELBDMSolver_GramFE_FFT.cpp','FFT update',325)}.</p>
<p>MATMUL table contractions use 55 decimal digits before the resulting update is cast to float64. This prevents a naive double-precision port from introducing artificial cancellation error; it does not reproduce binary128 rounding exactly. FFT is tested both through a high-precision fused linear map and through its actual factorized projection, extension, FFT and inverse-FFT sequence. The coupled FFT runtime check disagrees with its fused control by at most {max(v['runtime_vs_fused_wave'] for v in runtime['rows']):.3g} in wave norm.</p>
<p>Input/output fields are double precision unless explicitly labeled otherwise. Production single precision, GPU execution, 3D sweep ordering, density floors, mass corrections, moving refinement, and HJM-wave transitions are outside this reconstruction. The user has observed the discrepancy with both FFT and MATMUL; therefore a difference between their filters or arithmetic is not treated as the main explanation.</p>

<h2>1. Well-resolved same-level propagation is not the leading error here</h2>
{table(['Points/wavelength','Variant','Wave error after 10,000 updates','Mass ratio'],[(v['ppw'],v['kind'],v['wave_error'],v['mass_ratio']) for v in d['bloch'] if v['ppw'] in (4.,6.,10.,16.)])}
<p>The long-step result uses the 16-by-16 Bloch matrix of the periodic patch arrangement, retaining coupling to patch harmonics. It is not just the mean one-step amplification factor raised to a power. At ten points per wavelength both variants stay below 6e-6 wave error after 10,000 updates. This does not rule out weak instabilities over much larger step counts, nor contamination by shorter waves.</p>
{pic('dispersion.png','Plane-wave errors from the actual table-based column updates. FFT_fused is checked against the runtime factorization; MATMUL includes the supplied stored-table accuracy floor.')}

<h2>2. Gravity dominates the 1D moving-core experiment</h2>
<p>A spatially converged periodic stationary SP core is boosted and translated analytically to provide the reference. All methods use the same Strang splitting; only the kinetic update and Poisson Fourier symbol change. At N=128, boost index 12 gives 10.67 cells per carrier wavelength. The following uses half the default kinetic timestep and evolves to T=2.</p>
{table(['Kinetic update','Gravity','Density error','Peak-density bias','RMS-radius bias'],[(v['kinetic'],v['gravity'],v['density_error'],v['relative_peak_density_error'],v['relative_rms_radius_error']) for v in sample])}
<p>Second-order gravity gives approximately 0.00131 relative density error for all three kinetic solvers, versus about 2e-6 with spectral gravity. It makes this core slightly denser and narrower. The correct interpretation is that gravity can hide the gain from an accurate wave propagator. A 0.13% toy error does not establish the magnitude or origin of the user's halo discrepancy.</p>
<p>The full grid/timestep sweep includes N=64,128,256, stationary and boosted cores, and the default and halved kinetic timestep. With spectral gravity, timestep refinement matters at adequate spatial resolution. With FD2 gravity, halving the timestep barely changes the dominant error. Doubling the entire uniform grid does reduce the FD2 error by approximately four: this particular test does not reproduce a failure to converge under global refinement.</p>
{pic('selfgravity.png','Actual Gram-FE patch propagation combined with spectral or FD2 gravity. Dashed curves use FD2 gravity; solid curves use spectral gravity. All plotted runs use half the default kinetic timestep.')}

<h2>3. Why a 1D Poisson toy can miss a 3D problem</h2>
<p>If two fine-grid Poisson solutions have the same source and differ only in boundary values, their difference satisfies Laplace's equation. In 1D, delta Phi = a + bx. The constant changes global phase and the slope produces uniform acceleration. After removing translation, neither directly changes the internal structure of an isolated core in a linear external field. This statement assumes correct fine sources, boundaries outside the core, and a converged solve; source errors or interfaces through the core can still matter in 1D.</p>
<p>In two or three dimensions, harmonic errors can contain tides. For example delta Phi=epsilon(x²-y²) has zero Laplacian and nonzero Hessian. A small residual therefore does not certify the correct tidal field. This provides a concrete reason to supplement 1D tests with a cheap multidimensional elliptic control.</p>
<p>The new 2D problem contains two positive Gaussian density clumps, periodic mean subtraction, and six nested central patches. Every child uses the exact analytic density and a direct FD2 solve, so there is no wave-restriction error or relaxation stopping error. Boundary values come from the parent using quadratic or quartic tensor-product interpolation. This is a controlled nested Poisson model, not a port of GAMER's cell-centered interpolation or boundary stencil.</p>
{table(['Level','Relative center-Hessian error','Same fine solve with exact boundaries','Inherited tidal error'],[(v['level'],v['center_hessian_error'],v['oracle_hessian_error'],v['inherited_tidal_error']) for v in hierarchy])}
<p>At level 6, inherited boundaries leave error 3.77e-4, while exact boundaries give 1.19e-6. The approximately 300-fold difference remains despite relative linear-system residuals around 1e-15. The inherited difference is discrete-harmonic; its Hessian trace is approximately zero but its trace-free tidal component is not. Refining only deeper levels approaches a floor. Doubling root and patch resolution together reduces this floor by about four.</p>
{pic('gravity_hierarchy.png','Left: the Hessian error plateaus under deeper refinement with inherited boundaries. Right: the level-6 inherited error after removing its constant and linear parts. A 1D harmonic error would vanish under the same subtraction.')}
<p>Switching only the root to a spectral solve improves this toy but does not remove the floor from the remaining levels. Increasing boundary interpolation order alone has a smaller effect. The measured error is modest; no experiment here demonstrates that it accumulates into the observed halo discrepancy.</p>
<p>A separate exact 3D Gaussian-in-a-trap calculation verifies that a trace-free quadratic potential error can change central density and shape despite zero Poisson residual. Its imposed amplitudes are illustrative, larger than the small hierarchy error measured above, and are not inferred GAMER errors. The response is in the JSON; it is not a soliton or halo calculation.</p>

<h2>4. A 1D failure shared by FFT and MATMUL: time interpolation of phase</h2>
<p>This control prescribes exact parent wave values at both parent times and exact spatial ghost values. The fine region contains four patch groups. Four fine steps fit inside a parent interval. Only the interpolation between parent times is changed.</p>
<p>For a plane wave rotating at omega, interpolating its real and imaginary parts gives a chord through the complex plane. At the midpoint its amplitude is multiplied by cos(omega DeltaT/2). A constant potential increases omega while leaving the continuum density unchanged. Thus this interpolation introduces a gauge-dependent amplitude error even with perfectly accurate gravity and spatial resolution.</p>
<p>At ten coarse / twenty fine points per wavelength, after 2048 fine updates:</p>
{table(['Constant-potential phase per parent step','Variant','Density error with complex interpolation','Mass ratio'],[(v['potential_phase_per_parent_step'],v['kind'],v['density_error'],v['mass_ratio']) for v in linear])}
<p>All imposed potential rotations are below the default ELBDM gravity cap pi/4. Exact temporal ghost values and density/phase temporal interpolation remove this error for the single-wave control: the latter is exact here because density is constant and unwrapped phase is linear in time. This is not a universal cure near interference nodes.</p>
{pic('boundary_time.png','The continuum density is unchanged by the constant potential. Complex temporal interpolation creates percent-level density error in both variants; exact and density/phase temporal controls remain accurate.')}
<p>The fixed-physical-problem refinement control is especially diagnostic:</p>
{table(['Fine points/wavelength','Parent timestep','Fine timestep','Subcycles','Density error'],[(v['fine_ppw'],v['parent_dt'],v['fine_dt'],v['subcycles'],v['density_error']) for v in r['rows']])}
<p>Doubling only fine spatial resolution changes error from 6.94% to 7.28%. Halving both level timesteps instead reduces it to 1.80%, and quartering them gives 0.456%. This reproduces a specific failure of fine-grid refinement in a cheap 1D model, not the user's halo failure as a whole.</p>
<p><strong>Configuration caveat:</strong> this mechanism corresponds to the wave-level real/imaginary temporal-interpolation branch. The general documented default OPT__INT_PHASE is 1; the pinned LSS_Hybrid_Zoomin example explicitly sets it to 0. The user's exact setting remains unconfirmed. If density/phase interpolation was active throughout the affected interfaces, this pure-wave mechanism is not an explanation. See {source('src/Main/InterpolateGhostZone.cpp','complex interpolation',345)}, {source('src/Main/InterpolateGhostZone.cpp','density/phase temporal branch',1735)}, and {source('example/test_problem/ELBDM/LSS_Hybrid_Zoomin/Input__Parameter','zoom example settings',166)}.</p>

<h2>5. Weakly growing modes require longer tests than a smooth packet</h2>
<p>A scan over Bloch phase finds spectral radii slightly greater than one in both variants. The most amplified eigenvectors are dominated by wavelengths of roughly 3.4–4.6 cells. They are not the well-resolved ten-cell plane waves from the first test. The table reports amplitude growth, not mass growth.</p>
{table(['Variant','dt/default','Maximum spectral radius','Dominant cells/wavelength','Predicted amplitude after 10,000 steps','Direct runtime amplitude'],[(v['kind'],v['dt_factor'],format(v['max_spectral_radius'],'.10f'),v['unstable_mode_dominant_ppw'],v['predicted_10000_step_amplitude'],v['runtime_10000_step_amplitude']) for v in long['rows']])}
<p>Direct iteration uses MATMUL column actions or actual factored FFT updates with quasiperiodic ghost data. It confirms the sign and scale of growth. The direct FFT and fused predictions differ by at most {max(v['runtime_prediction_difference'] for v in long['rows'] if v['kind']=='fft_fused'):.3g} in amplitude after 10,000 steps.</p>
<p>The corresponding amplitude e-folding counts range from approximately 1.3e5 to 1.1e6 steps in these probes. Amplifying a 1e-7 seed to order unity would take approximately 2e6 to 1.7e7 fixed-operator steps. These are extrapolations, not cosmological forecasts: mode seeding and nonlinear evolution are unknown. Smaller timesteps do not monotonically reduce the per-step growth, and taking more steps changes cumulative amplification. Recording actual fine-level step counts and short-wave power is therefore worthwhile before declaring the kinetic operator irrelevant.</p>

<h2>6. A narrower MATMUL precision defect, not the shared halo explanation</h2>
<p>The stored MATMUL extension coefficients reach 6.86e9. Although declared as float128 constants, the DFT entries differ from high-precision analytic values at roughly double-precision scale ({p['dft_max_entry_error']:.3g} maximum entry error). With filtering disabled and dt=0, the reconstructed stored-table operator differs from the exact crop by {d['stored_zero_time_no_filter_matrix_error']:.3g} in operator norm; analytic DFT tables reduce that control to {d['analytic_dft_zero_time_no_filter_matrix_error']:.3g}.</p>
<p>However, correcting only DFT tables does <em>not</em> improve the observed smooth-wave floor in the actual filtered update. Reconstructing the extension from the FFT branch's factored tables at high precision, retaining the MATMUL filter, does improve it. This negative control prevents attributing the whole issue to DFT rounding alone.</p>
{table(['Points/wavelength','Stored MATMUL error','DFT-only correction','Factored-extension control','Float32 runtime control'],[(v['ppw'],v['stored_error'],v['corrected_dft_error'],v['factorized_extension_error'],v['stored_float32_error']) for v in p['rows']])}
<p>This is a reproducible accuracy-floor issue in the table-based reconstruction and merits checking the table-generation precision. It is much smaller than the gravity/interface errors demonstrated here, and the user also observes the halo discrepancy with FFT. No GAMER source was patched. The direct 60-digit builder agrees with the cached polynomial construction to {p['direct_builder_vs_polynomial_cache']:.3g} in operator norm.</p>

<h2>What to investigate next in GAMER</h2>
<ol><li><strong>Separate inherited boundary error from relaxation error.</strong> On a small multidimensional frozen-density hierarchy, compare potential Hessians from the current solve with a global accurate reference. Inspect the trace-free part after removing constant potential and uniform acceleration. Tightening a local residual alone does not test this.</li>
<li><strong>Check actual temporal-interpolation branches and parent phase advance.</strong> If OPT__INT_PHASE=0 applies, the constant-potential plane wave is a cheap regression test. Compare exact or synchronized parent data before changing spatial interpolation methods. If phase interpolation is active, use interfering waves and phase nodes rather than claiming the single-wave result applies.</li>
<li><strong>Log cumulative fine-level updates and short-wave contamination.</strong> The weakly growing modes provide a concrete long-run kinetic hypothesis. Their excitation must be measured, not assumed from the eigenvalues.</li>
<li><strong>Use resolution changes that isolate the hierarchy.</strong> Refining the whole hierarchy, refining only the finest region, and shortening parent time intervals are different experiments. The user reports that doubling resolution did not resolve the investigation; which of these was changed remains important.</li></ol>
<p><strong>Assessment:</strong> gravity remains the strongest demonstrated error source for a well-resolved moving core in this round. The multidimensional boundary channel explains why some 1D tests can miss it. The time-interpolation test supplies a common FFT/MATMUL failure with a clear intervention, but only under its stated configuration. None yet proves the cause of the user's halo-profile discrepancy.</p>

<h2>Reproduce</h2><pre>python toy_models/fetch_gramfe_sources.py
python toy_models/gramfe_probe.py
python toy_models/gramfe_probe.py --long-run
python toy_models/gramfe_precision_audit.py
python toy_models/gramfe_selfgravity.py
python toy_models/gramfe_selfgravity.py --runtime-check
python toy_models/gramfe_boundary_time.py
python toy_models/gramfe_boundary_time.py --refinement
python toy_models/gravity_hierarchy_probe.py
python toy_models/build_gamer_diagnosis.py</pre>
<p>Use the project's pinn_env Python. The source fetch is the only network-dependent step; downloaded GAMER files remain under the ignored sources/gamer directory. High-precision operator construction is cached with the header hash. All new outputs are under gramfe_results; earlier reports and numerical outputs are preserved.</p>
<p><a href="gramfe_probe.py">Gram-FE reconstruction</a> · <a href="gramfe_selfgravity.py">Coupled SP test</a> · <a href="gramfe_boundary_time.py">Boundary-time tests</a> · <a href="gravity_hierarchy_probe.py">Nested gravity test</a> · <a href="gramfe_precision_audit.py">Precision audit</a></p>
<p>{' · '.join('<a href="gramfe_results/'+name+'.json">'+escape(name.replace('_',' '))+'</a>' for name in ['source_manifest','operator_provenance','dispersion','selfgravity','runtime_fft_check','gravity_hierarchy','boundary_time','boundary_refinement','long_run_spectrum','precision_audit'])}</p>
</main></html>'''
    (ROOT/'gamer_diagnosis.html').write_text(text,encoding='utf-8')
    print('Wrote gamer_diagnosis.html',len(text),'characters')


if __name__=='__main__':main()
