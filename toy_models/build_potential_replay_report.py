"""Standalone report and plots for prescribed-gravity ablations."""
import base64
import json
from html import escape
from pathlib import Path
from experiments import np,plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'potential_replay_results'


def table(headers,rows):
    def fmt(v):return f'{v:.6g}' if isinstance(v,float) else escape(str(v))
    return '<table><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+fmt(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def pic(name,caption):
    return '<figure><img alt="'+escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'"><figcaption>'+escape(caption)+'</figcaption></figure>'


def main():
    a=json.loads((OUT/'results.json').read_text());v=json.loads((OUT/'validation.json').read_text())
    mass=json.loads((OUT/'mass_control.json').read_text())
    rows=a['rows'];cases=['standing','weak_envelope','strong_envelope','perturbed_core']
    titles=['Standing interference, g=100','Weak envelope, g=10','Strong envelope, g=100','Perturbed localized core']
    def row(case,n,gravity='spectral',branch='prescribed',kinetic='fft_fused'):
        return next(r for r in rows if all(r[k]==val for k,val in [('case',case),('n',n),('gravity',gravity),('branch',branch),('kinetic',kinetic)]))
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    choices=[('spectral','prescribed','Accurate potential replay'),('spectral','self_consistent','Self-consistent spectral'),
             ('fd2','prescribed','FD2 potential replay'),('fd2','self_consistent','Self-consistent FD2')]
    for ax,case,title in zip(axs.ravel(),cases,titles):
        n=64 if case=='perturbed_core' else 96
        for grav,branch,label in choices:
            rr=row(case,n,grav,branch);h=rr['history'][1:]
            ax.semilogy([r['time'] for r in h],[max(r['density_l2'],1e-12) for r in h],
                        '--' if branch=='self_consistent' else '-',label=label)
        ax.set(title=f'{title}, N={n}',xlabel='Time',ylabel='Relative density L2 error');ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'density_history.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,label in zip(axs,['wave_l1','density_l2'],['Mean component wave L1 error','Relative density L2 error']):
        for grav,branch,legend in choices:
            vals=[row(c,64 if c=='perturbed_core' else 96,grav,branch)[metric] for c in cases]
            ax.semilogy(np.arange(4),vals,'o--' if branch=='self_consistent' else 'o-',label=legend)
        ax.set_xticks(np.arange(4),['Standing','Weak envelope','Strong envelope','Core'],fontsize=8)
        ax.set(ylabel=label);ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'final_errors.png',dpi=170);plt.close(fig)
    s=row('standing',96);ss=row('standing',96,branch='self_consistent');fd=row('standing',96,'fd2');fds=row('standing',96,'fd2','self_consistent')
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Prescribed gravity versus gravity feedback</title>
<style>body{{font:17px/1.6 system-ui;background:#eef3f6;color:#203547;margin:0}}main{{max-width:1120px;margin:auto;background:white;padding:32px}}h1,h2{{line-height:1.25}}h2{{margin-top:2em}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}}td,th{{padding:8px;border-bottom:1px solid #d4e0e7;text-align:left}}th{{background:#eaf2f7}}figure{{margin:25px 0}}img{{width:100%}}figcaption,.small{{font-size:14px;color:#526575}}.box{{padding:16px;background:#e7f3f7;border-left:5px solid #327e9e}}pre{{background:#eef3f6;padding:15px;overflow:auto}}a{{color:#145c89}}</style><main>
<h1>Prescribed gravity versus self-consistent gravity</h1>
<div class="box">The user confirms that doubling resolution helps. This test asks why five-six cells per wave can still be insufficient for coupled gravity, and whether replacing numerical gravity by a recorded accurate potential removes the error. All previous numerical outputs remain untouched.</div>
<h2>Experiment and interpretation</h2>
<p>Every branch starts from identical point samples of the high-resolution reference wave and uses the same uniform wave grid and timestep. The spectral reference supplies its time-dependent potential at every kick endpoint. It is not frozen in time. The reference uses fourth-order splitting with two internal substeps, and independent finer-space and smaller-step references are checked.</p>
<p><strong>Accurate potential replay</strong> uses the fine reference potential evaluated at target nodes. <strong>Coarse spectral replay</strong> instead computes a spectral potential from the reference density sampled on the target grid: its difference from accurate replay detects source sampling and coarse representation effects without feedback. <strong>FD2/FD4/hierarchy replay</strong> applies the approximate gravity map to that same reference density. <strong>Self-consistent</strong> runs recompute the selected potential from their own evolving wave. All replay runs are linear equations with a prescribed time-dependent potential. Approximate replay has no density-error feedback, but still has gravity approximation error.</p>
<p>Prescribed and self-consistent branches are not an additive error decomposition. Their errors can cancel, and self-consistency can restore an equilibrium. A ratio of final errors is not a universal feedback amplification factor. Global-FFT kinetics controls splitting and grid-representation error. Actual-table FFT and MATMUL Gram-FE are both included. The model excludes GAMER's flux correction, density rescaling, wave AMR, asynchronous ghost updates, and cosmological expansion.</p>
<h2>Standing wave: gravity error exists without feedback</h2>
<p>At six cells per carrier, FFT Gram-FE density errors are {s['density_l2']:.6g} with accurate replay and {ss['density_l2']:.6g} with self-consistent spectral gravity. FD2 replay already gives {fd['density_l2']:.6g}, essentially the self-consistent FD2 error {fds['density_l2']:.6g}. The approximate gravitational forcing, rather than strong feedback amplification, accounts for most of the difference in this example. At all sampled reference times, the wave power shorter than six cells is below {a['bandwidth']['standing']['96']['max_power_below_six_cells']:.4g}.</p>
{pic('density_history.png','FFT Gram-FE with fixed wave grids. Curves compare replay and self-consistent runs over time; the exactly zero initial error is omitted on logarithmic axes. Density errors alone do not measure gravitational work or halo-profile bias.')}
<h2>All four cases, both resolutions</h2>
<p>All errors in the tables are fractions, not percentages. Mean component L1 follows the paper-style real/imaginary wave metric; relative L2 and phase-aligned L2 are separate diagnostics. Global phase changes wave error but not density. Interference cases use T=0.5, dt=0.0002 and k=16; N=96 and 192 therefore correspond to 6 and 12 cells per initial carrier. Envelope sidebands initially reach k=17. The core uses L=20, mass=4, T=6, dt=0.002 and a weak phase/packet perturbation; it has no single carrier wavelength.</p>
'''
    for case,title in zip(cases,titles):
        selected=[r for r in rows if r['case']==case and r['kinetic']=='fft_fused']
        text+='<h3>'+escape(title)+'</h3>'+table(['N','Gravity','Branch','Wave mean L1','Wave relative L2','Aligned wave L2','Density relative L2'],
            [[r[k] for k in ('n','gravity','branch','wave_l1','wave_l2','aligned_wave_l2','density_l2')] for r in selected])
    text+=pic('final_errors.png','Mean-component wave errors and relative density errors are different observables; both are reported rather than converting between their norms.')
    text+='''<h2>Evolving bandwidth and propagation controls</h2><p>Each bandwidth maximum is measured on the independent reference trajectory at every target step, not inferred from a potentially damped numerical solution. Strictly shorter than six cells excludes a mode at exactly six cells. This threshold is a diagnostic, not a mathematical error bound. Unresolved density RMS is normalized by density fluctuations with the mean removed. A strong-envelope failure with substantial short-wave power does not disprove six-cell accuracy for a wave that actually remains six-cell resolved.</p>'''
    text+=table(['Case','N','Max wave power <6 cells','Max wave power <5 cells','Max wave power beyond Nyquist','Max unresolved density RMS'],
        [[case,n,b['max_power_below_six_cells'],b['max_power_below_five_cells'],b['max_power_beyond_nyquist'],b['max_unresolved_density_rms']]
         for case,bb in a['bandwidth'].items() for n,b in bb.items()])
    text+='<details><summary>Global FFT and MATMUL controls</summary>'+table(['Case','N','Kinetic','Gravity','Branch','Wave L1','Density L2','Mass change'],
        [[r[k] for k in ('case','n','kinetic','gravity','branch','wave_l1','density_l2','mass_change')] for r in rows if r['kinetic']!='fft_fused'])+'</details>'
    text+='''<h2>Assessment of the difficult case and feedback</h2><p>At N=96, accurate potential replay reduces FFT Gram-FE's strong-envelope density error from 4.62% (self-consistent) to 1.36%. Global FFT propagation with the same replay reaches 0.111%, so the remaining error is not all due to self-consistent gravity. However, as much as 37.6% of reference wave power lies below five cells per wavelength during this run, and unresolved density fluctuations reach 24.5% RMS at an intermediate time. The previous final-only bandwidth diagnostic understated this transient excursion. This is a propagation/representation failure in an evolving under-resolved regime, not evidence against the paper's six-cell free-wave convergence.</p>
<p>At N=192, accurate replay reaches 0.000691% density error, while self-consistent spectral gravity gives 0.0113%. Feedback can still matter when direct propagation errors are small; it is not an error source attributable solely to an inaccurate Poisson stencil. The coarse spectral replay control differs little from fine-potential replay in these final density metrics. This does not prove that all source aliasing is harmless.</p>
<p>For the localized core at N=64, FD2 replay has 0.268% density error and self-consistent FD2 has 0.200%. Self-consistency slightly reduces the final density error here. Unaligned wave errors of order 8-9% largely reflect an accumulated global phase: aligned wave errors are only about 0.28% and 0.19%. A large unaligned wave error must not be equated with a large halo-profile error.</p>
<h2>Mass-conservation caveat and additional control</h2><p>Strong-envelope FFT Gram-FE loses about 0.57% of total mass at N=96. The kinetic reconstruction does not include GAMER's separate conservation treatment. As a diagnostic only, globally normalizing the wave after every kinetic step reduces the self-consistent spectral-gravity density error from 4.62% to 3.19%, while prescribed-potential error changes from 1.36% to 1.34%. Mass drift contributes but is not the whole effect in this toy. Global normalization preserves no local flux and is neither a reconstruction of GAMER's correction nor a proposed production remedy.</p>'''
    text+=table(['N','Kinetics','Branch','Density without normalization','Density with normalization','Mass change with normalization'],
        [[r['n'],r['kinetic'],r['branch'],row('strong_envelope',r['n'],branch=r['branch'],kinetic=r['kinetic'])['density_l2'],r['density_l2'],r['mass_change']] for r in mass['rows']])
    text+='''<h2>Gravity refinement boundary in the core test</h2><p>The core-only hierarchy uses a parent of spacing 2h, a centered child covering half the box, quadratic potential interpolation and a direct FD2 child solve. All fine wave data remain available. Root sources are point samples, not GAMER's cell averages. The potential is assembled from the parent outside the child and the child inside. One-dimensional harmonic boundary errors are affine; this test cannot reproduce the trace-free tidal errors possible in 2D/3D. It diagnoses a hierarchy mechanism, not the quantitative error of GAMER's six-level halo calculation.</p>
<p><strong>Negative result:</strong> this centered 1D hierarchy adds almost no density error beyond uniform FD2 for the core. Self-consistent errors at N=64 are 0.19983% (hierarchy) and 0.19992% (uniform FD2). We have not reproduced the user's suspected AMR boundary failure in this experiment. The earlier 2D boundary tests are more informative about tidal communication.</p>
<h2>Reconciliation with the paper's linear convergence test</h2><p>The free plane wave calibration uses k=32 pi, L=1, hbar/m=1, T=0.015 and dt at or below the reconstructed recommended Gram-FE CFL, with periodic same-level patch exchange. This reproduces the analytic problem and parameter choices, not a compiled GAMER run or its complete conservation corrections. The envelope convergence test is not rerun here. See <a href="https://arxiv.org/html/2411.17288v2#S4.SS1.SSS1">the paper, section IV.1.1</a>. The test's mean component L1 is an accumulated evolution error, not a per-step error.</p>'''
    text+=table(['Cells/wave','Kinetics','Steps','Mean component wave L1','Relative wave L2','Relative density L2'],
        [[r[k] for k in ('ppw','kinetic','steps','wave_l1','wave_l2','density_l2')] for r in a['plane_calibration']])
    text+='''<p>For psi=exp(ikx)+a exp(-ikx), density contains a 2k fringe. Six cells per wave give three per fringe. FD2 overestimates that mode's potential amplitude by about 46%, independently of relaxation residual. This mode-specific result is not a predicted halo-profile error. It explains why high-order free wave convergence does not guarantee equally accurate self-gravitating dynamics.</p>
<h2>Validation and limits</h2>'''
    text+=table(['Check','Value'],[[k,val] for k,val in v.items() if k!='timestep_comparison' and not isinstance(val,list)])
    text+='<details><summary>Half-timestep comparisons (all lower-resolution controls)</summary>'+table(['Case','Kinetic','Gravity','Branch','Density baseline','Density half dt','Wave baseline','Wave half dt'],
        [[r[k] for k in ('case','kinetic','gravity','branch','baseline_density','half_dt_density','baseline_wave','half_dt_wave')] for r in v['timestep_comparison']])+'</details>'
    text+='''<p>The independent temporal test uses a plane wave and the commuting potential Phi(t)=sin(t), for which endpoint quadrature and the exact solution are known. It checks the replay time indexing and second-order temporal error. A manufactured source checks the direct child solve. Kinetic energy changes are accounted for at every substep; the sum closes. Halving dt changes both splitting error and the number of Gram-FE filter applications, so monotonic improvement is not assumed for that method. In particular, the strong-envelope prescribed density error increases from 1.36% to 1.47%, and self-consistent error from 4.62% to 5.43%. The qualitative attribution survives this control, but these numbers are not claimed to be a time-converged Gram-FE limit.</p>
<p>These experiments do not identify the dominant error in a cosmological halo, establish a universal cells-per-wavelength threshold, or show that a radial halo profile tracks pointwise density error. They do distinguish prescribed-force error, source-grid representation, self-consistent response, and wave propagation in cheap controlled cases.</p>
<h2>Reproduce and inspect</h2><pre>python toy_models/potential_replay.py
python toy_models/potential_replay.py --controls
python toy_models/potential_replay_mass_control.py
python toy_models/verify_potential_replay.py
python toy_models/build_potential_replay_report.py</pre>
<p>Use the project's pinn_env Python. Raw data: <a href="potential_replay_results/results.json">main results</a>, <a href="potential_replay_results/results_controls.json">half-step controls</a>, <a href="potential_replay_results/mass_control.json">mass control</a>, <a href="potential_replay_results/validation.json">validation</a>. Final fields are saved in fields.npz and fields_controls.npz in that directory. Figures are standalone PNGs and embedded here. <a href="gravity_accuracy.html">Previous gravity report</a>.</p></main></html>'''
    (ROOT/'potential_replay.html').write_text(text,encoding='utf-8')
    print('Wrote potential_replay.html')


if __name__=='__main__':main()
