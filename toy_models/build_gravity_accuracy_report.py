"""Report gravity-order, equilibrium, bandwidth and interface experiments."""
import base64
import json
from html import escape
from pathlib import Path
from experiments import np,plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'gravity_accuracy_results'


def load(name):return json.loads((OUT/(name+'.json')).read_text())
def table(headers,rows):
    def fmt(x):return f'{x:.5g}' if isinstance(x,float) else escape(str(x))
    return '<table><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+fmt(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def pic(name,caption):return '<figure><img alt="'+escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'"><figcaption>'+escape(caption)+'</figcaption></figure>'
def percent(x):return f'{100*x:.4g}%'


def main():
    e=load('evolution');b=load('bandwidth');q=load('equilibrium');i=load('interface');a=load('amr_evolution');c=load('amr_controls');w=load('work')
    fig,axs=plt.subplots(1,2,figsize=(11,4));labels=['uniform2','uniform4','hierarchy2_2','hierarchy4_2','hierarchy4_4','oracle4']
    for ax,shift in zip(axs,(0,16)):
        rr=[next(r for r in a['rows'] if r['label']==label and r['shift_cells']==shift) for label in labels]
        ax.bar(np.arange(len(rr)),[r['density_error'] for r in rr]);ax.set_yscale('log')
        ax.set_xticks(np.arange(len(rr)),['Uniform\nFD2','Uniform\nFD4','FD2 +\nquadratic','FD4 +\nquadratic','FD4 +\nquartic','FD4 +\nreference BC'],fontsize=7)
        ax.set(title='Centered patch' if shift==0 else 'Displaced patch',ylabel='Final relative density error');ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'amr_evolution.png',dpi=160);plt.close(fig)
    standing=[r for r in e['rows'] if r['case']=='standing' and r['n'] in (96,192) and r['kinetic']=='fft_fused']
    weak=[r for r in b['weak_gravity'] if r['kinetic']=='fft_fused']
    envelope=[r for r in e['rows'] if r['case']=='enveloped' and r['n'] in (96,192) and r['kinetic'] in ('global_fft','fft_fused')]
    eq=[r for r in q['equilibria'] if r['n'] in (64,128)]
    frozen=[r for r in i['rows'] if r['root_n']==50 and r['shift']==0]
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Does better gravity reduce the required wave resolution?</title>
<style>body{{font:17px/1.6 system-ui;background:#edf3f7;color:#183248;margin:0}}main{{max-width:1120px;margin:auto;padding:36px;background:white}}h1,h2{{line-height:1.2}}h2{{margin-top:2em}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}}td,th{{padding:9px;border-bottom:1px solid #d8e1e8;text-align:left}}th{{background:#e8f1f7}}img{{width:100%}}figure{{margin:25px 0}}figcaption,.small{{font-size:14px;color:#476075}}.box{{background:#e9f4f9;padding:18px;border-left:5px solid #307da9}}a{{color:#155e91}}pre{{overflow:auto;background:#eef3f7;padding:15px}}</style><main>
<h1>Does better gravity reduce the required wave resolution?</h1>
<div class="box"><strong>Correction to the research premise:</strong> the user confirms that doubling resolution has always helped. The question is why faithful halo reconstruction needs more than the initially expected five–six cells per wavelength. Earlier statements treating failure under resolution doubling as an observed fact were a misunderstanding. This report supersedes that interpretation; previous numerical outputs remain preserved.</div>
<p><strong>Result:</strong> better gravity can replace a factor-two refinement in selected coupled tests, and removes a measurable equilibrium-profile bias. It does not universally make five–six cells sufficient. Density bandwidth, wave propagation and gravity transfer across refinement boundaries can each become the next limitation. These are small controlled systems, not a cosmological halo validation.</p>

<h2>1. Coupled interference: hold the wave evolution fixed</h2>
<p>Dimensionless periodic 1D SP uses L=2π, iψₜ=−ψₓₓ/2+Φψ, Φₓₓ=g(|ψ|²−mean|ψ|²). The standing case superposes carriers ±16; the enveloped case adds a smooth amplitude envelope and a smaller carrier at 7. The same initial data, final time T=0.5 and 2,500 Strang steps are used for all compared gravity operators. Kinetic controls are global FFT and the previously reconstructed actual-table FFT/MATMUL Gram-FE maps. No AMR, mass correction, or cosmological expansion is present.</p>
<p>FD2, FD4 and spectral Poisson are all solved by FFT on uniform grids, so this comparison measures the discrete Laplacian rather than relaxation tolerance or solver speed. With s=sin²(kh/2), their positive Laplacian symbols are 4s/h², 4s(1+s/3)/h² and k². Dense reference operations and uniform spectral gravity are diagnostic oracles, not proposed production zoom-in algorithms.</p>
<p>The following table uses Gram-FE FFT propagation and g=100 for the standing wave. Six and twelve cells refer to the initial carrier wavelength.</p>
{table(['Cells/carrier','Gravity','Final density error'],[(r['n']/16,r['gravity'],percent(r['density_error'])) for r in standing])}
<p><strong>Positive result:</strong> six cells with spectral gravity outperform twelve with FD2 by about a factor of six. <strong>Limit:</strong> FD4 at six cells does not quite replace twelve-cell FD2 in this standing-wave case. Merely increasing the order to four is not a universal factor-two saving.</p>
<p>A less strongly gravitating enveloped field, g=10, gives a useful intermediate result:</p>
{table(['Cells/carrier','Gravity','Final density error, Gram-FE FFT'],[(r['n']/16,r['gravity'],percent(r['density_error'])) for r in weak])}
<p>Here six-cell FD4 and twelve-cell FD2 are nearly equal, about 0.0137% density error. This is direct coupled evidence that a modest gravity improvement can substitute for refinement in a suitable regime. The envelope increases the maximum initial wave number from 16 to 17, so six cells per carrier correspond to 5.65 cells per shortest initial wave.</p>
{pic('evolution.png','Identical wave stepping with three gravity operators. Both easy and difficult initial conditions are retained. Errors compare pointwise density against a time- and space-refined spectral reference.')}

<h2>2. A counterexample: improving gravity alone does not always suffice</h2>
<p>The strongly gravitating enveloped case, g=100, develops new wave and density modes. Its results are:</p>
{table(['Cells/carrier','Kinetic method','Gravity','Density error'],[(r['n']/16,r['kinetic'],r['gravity'],percent(r['density_error'])) for r in envelope])}
<p>At six cells per initial carrier, Gram-FE remains at roughly 4–5% error even with spectral gravity. In the reference solution, about 25.4% of wave power has moved to wavelengths shorter than five cells on that grid, and the unresolved density spectrum has 2.66% relative RMS amplitude. Almost all wave power is still below the wave Nyquist limit; that weaker statement does not guarantee accurate Gram-FE propagation or resolution of |ψ|². At twelve cells the density bandwidth problem is far smaller.</p>
{pic('bandwidth.png','The density spectrum evolves beyond its initial bandwidth. Dashed lines mark the Nyquist limits for N=80, 96, 160 and 192. A criterion based only on the initial carrier misses this change.')}
<p>This is not a counterexample to resolving all waves accurately: the coarse grid no longer meets that premise during the evolution. It shows why a fixed initial points-per-wavelength number is insufficient. The standing-wave result also contains tiny generated sidebands; it establishes a final-time accuracy improvement in this test, not an all-times error bound.</p>

<h2>3. Does gravity shift the equilibrium profile?</h2>
<p>A periodic 1D self-gravitating core with mass 4 and box length 20 is solved to a nonlinear residual below 10⁻⁹. The kinetic operator is spectral for every case; only Poisson changes. The discrete stationary solutions are compared to a separately refined continuum-reference core. This tests a systematic bias without needing an instability or long time integration.</p>
{table(['Grid cells','Gravity','Peak-density bias','RMS-radius bias'],[(r['n'],r['gravity'],percent(r['peak_bias']),percent(r['radius_bias'])) for r in eq])}
<p>At N=64, FD2 makes the core about 0.476% denser at its peak and 0.403% narrower in RMS radius. FD4 reduces those biases to 0.0266% and 0.0187%. The FD4 radius bias at 64 cells is smaller than the FD2 bias at 128 cells. A stationary real core has no unique carrier wavelength, so these cell counts must not be translated into a de Broglie points-per-wavelength claim.</p>
<p>Starting every method from the same continuum core produces breathing around the altered equilibrium. A second case adds a weak phase perturbation and a small coherent incoming packet. At T=12, the perturbed-core density error is 0.00844% for N=64 FD4 versus 0.0518% for N=128 FD2. The temporal control changes these errors little. Radius oscillations have periods near three in these units, but the period estimates come from only a few peaks and a 0.03 sampling interval; the measured profile bias is firmer evidence than small apparent frequency differences.</p>
{pic('equilibrium.png','Gravity changes the discrete equilibrium itself. The right panel starts from the continuum core and shows artificial breathing caused by the gravity mismatch.')}

<h2>4. Frozen interface ablations: what can an accurate interior solver fix?</h2>
<p>The exact finite-Fourier 2D interference field from the previous round supplies a source and potential oracle. A fine patch is displaced relative to the fixed density field. FD2/FD4 direct solves are combined with exact or interpolated boundary potentials, and exact density or density reconstructed from a deliberately low-accuracy coarse-wave interpolation. That latter control uses quadratic interpolation, not GAMER Gram-FE interpolation; it isolates a source error and is not a quantitative accusation against GAMER.</p>
{table(['Fine/root order','Fine source','Boundary data','Patch Hessian error'],[(r['order'],r['source'],r['boundary'],percent(r['hessian_error'])) for r in frozen])}
<p>With exact source and boundaries, FD4 lowers the patch Hessian error from 4.59% to 0.526%. With quadratic parent potential transfer, FD4 instead gives 1.70%. With an inaccurate source but exact boundaries, its error is 5.76%. Thus neither exact boundaries nor high-order interior gravity repairs a wrong source. The comparison uses identical discrete Hessian diagnostics on the numerical and analytic sampled potentials; it does not mix in different force stencils.</p>
{pic('interface.png','Move the gravity interface while holding the physical density fixed. The near-origin error distinguishes boundary placement from a change in the underlying field.')}

<h2>5. Evolving 2D SP with a gravity hierarchy</h2>
<p>To go beyond frozen fields, a 96² uniform wave grid evolves self-consistently with either uniform gravity or a 48² gravity parent plus one refined gravity patch. This keeps wave propagation identical. The fine density is used directly inside the child. All child solves are direct; there is no relaxation error, wave interpolation or subcycling. The initial shortest wave has 9.6 fine cells and 4.8 parent cells. Evolution lasts 200 steps to T=0.1 at coupling 20.</p>
{table(['Gravity treatment','Centered patch density error','Displaced patch density error'],[(label,percent(next(r['density_error'] for r in a['rows'] if r['label']==label and r['shift_cells']==0)),percent(next(r['density_error'] for r in a['rows'] if r['label']==label and r['shift_cells']==16))) for label in labels])}
{pic('amr_evolution.png','The wave method and grid are fixed. Better gravity helps, but the remaining error depends on the interface position and potential transfer. The oracle also supplies the exterior potential, so it does not isolate boundary values alone.')}
<p>For the displaced patch, FD2/quadratic transfer gives 0.301% density error; FD4/quadratic gives 0.125%; FD4/quartic gives 0.0635%. The reference-boundary/exterior control reaches about 0.00695%. The central density errors show the same broad pattern. Mass stays conserved to about 10⁻¹³ because the wave update remains unitary; mass conservation alone would miss these density errors.</p>
<p>Parent density is point-sampled in this toy, not restricted as cell averages as in an actual AMR implementation. A separate parent-source control projects the fine density onto root-resolved Fourier modes to prevent aliasing and preserve its mean. Half-timestep results are:</p>
{table(['Patch shift (cells)','Parent treatment','Density error'],[(r['shift_cells'],r['label'],percent(r['density_error'])) for r in c['rows']])}
<p>This control distinguishes coarse-source aliasing from other interface errors. It is an oracle operation, not a suggested replacement for production restriction. The interface experiment has only one gravity refinement level and uniform wave propagation, so it does not reproduce the full GAMER hierarchy.</p>

<h2>6. Energy transfer and numerical checks</h2>
<p>For the strong envelope case, kinetic-energy changes are accumulated separately across every gravity kick and kinetic step. This avoids inferring gravitational work from mass conservation or under-sampled force histories.</p>
{table(['Grid','Kinetic method','Gravity','Gravity work / reference − 1','Spurious kinetic-step work'],[(r['n'],r['kinetic'],r['gravity'],percent(r['relative_gravity_work_bias']),r['numerical_kinetic_work']) for r in w['rows']])}
<p>The refined reference gravity transfers {w['reference_gravity_work']:.6g} units of kinetic energy. Global Fourier kinetic steps conserve kinetic energy individually. Gram-FE can add or remove kinetic energy in the difficult coarse case; gravity improvements cannot repair that independent contribution. The split work budget closes within 10⁻⁹. The sampled <code>integrated_gravity_work</code> diagnostic in the general evolution JSON is under-resolved for rapid interference oscillations and is not used for conclusions; <code>work.json</code> contains the validated substep budget.</p>
<p>Reference checks include N=768 versus 1536 and halved time steps for 1D interference; independently solved N=257/385 stationary cores; N=512/1024 and halved time steps for core dynamics; N=192²/384² and halved steps for 2D evolution. The largest 1D interference reference time discrepancy is below 2×10⁻⁹. The 2D reference time discrepancy is 6.1×10⁻⁷, much smaller than the hierarchy errors. Actual comparisons also include timestep halving at coarse resolutions. For Gram-FE, halving the time step changes the number of per-step filters, so it need not monotonically improve spatial errors.</p>
<p>Reference density is evaluated at coarse sample locations using all reference Fourier coefficients, rather than low-pass resampling. A provisional version used the latter and was corrected before this report; all affected results were rerun. An independent analytic oscillatory-density identity checks the sampling. A quartic manufactured potential verifies FD4 boundary handling. No full GAMER binary was compiled or run.</p>

<h2>Interpretation</h2>
<p><strong>A more accurate gravity treatment is justified as a concrete next improvement, not as a complete explanation or universal cure.</strong> There is now coupled evidence of a factor-two resolution saving in a weak interference case, stronger saving with spectral-reference gravity in a standing-wave case, and a reduced equilibrium bias. There is also a retained counterexample where newly generated bandwidth and the wave propagator dominate.</p>
<p>The simplest next production experiment would retain Gram-FE, improve Poisson order, and make the source and potential transfer consistent with that accuracy. Judge it by density-profile convergence and by the spectra of both ψ and |ψ|² throughout the evolution. Increasing only relaxation iterations will not correct the errors isolated here. The present evidence does not justify replacing the complete solver with a neural operator or assuming that any particular number of cells per wavelength will work universally.</p>

<h2>Reproduce</h2>
<pre>python toy_models/gravity_accuracy_evolution.py
python toy_models/gravity_accuracy_equilibrium.py
python toy_models/gravity_accuracy_interface.py
python toy_models/gravity_accuracy_amr_evolution.py
python toy_models/gravity_accuracy_bandwidth.py
python toy_models/gravity_accuracy_work.py
python toy_models/verify_gravity_accuracy.py --extra-controls
python toy_models/build_gravity_accuracy_report.py
python toy_models/verify_gravity_accuracy.py</pre>
<p>Use the installed pinn_env Python. All new artifacts are under <code>toy_models/gravity_accuracy_results/</code>. Raw results: <a href="gravity_accuracy_results/evolution.json">coupled evolution</a>, <a href="gravity_accuracy_results/equilibrium.json">equilibrium</a>, <a href="gravity_accuracy_results/interface.json">frozen interfaces</a>, <a href="gravity_accuracy_results/amr_evolution.json">evolving hierarchy</a>, <a href="gravity_accuracy_results/amr_controls.json">hierarchy controls</a>, <a href="gravity_accuracy_results/bandwidth.json">bandwidth</a>, <a href="gravity_accuracy_results/work.json">discrete work</a>. Previous reports: <a href="frozen_gravity.html">frozen-gravity follow-up</a> and <a href="gamer_diagnosis.html">Gram-FE source audit</a>.</p>
</main></html>'''
    (ROOT/'gravity_accuracy.html').write_text(text,encoding='utf-8');print('Wrote toy_models/gravity_accuracy.html')


if __name__=='__main__':main()
