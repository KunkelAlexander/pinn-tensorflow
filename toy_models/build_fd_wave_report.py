"""Standalone scientific report of the wave-stencil/conservation cross-check."""
import base64
import json
from html import escape
from pathlib import Path
from experiments import np,plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'fd_wave_results'
LABELS={'global_fft':'Global FFT','gram_fft':'Gram-FE FFT','gram_matmul':'Gram-FE MATMUL',
        'fd_lap2':'FD Laplacian-2, correction off','fd_lap2_mass':'FD Laplacian-2, correction on',
        'fd_lap4':'FD Laplacian-4, correction off','fd_lap4_mass':'FD Laplacian-4, correction on',
        'gram_fft_fd4flux':'Gram-FE + FD4 flux (experimental)'}


def table(headers,rows):
    def fmt(v):return f'{v:.6g}' if isinstance(v,float) else escape(str(v))
    return '<table><tr>'+''.join('<th>'+escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+fmt(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'


def pic(name,caption):
    return '<figure><img alt="'+escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'"><figcaption>'+escape(caption)+'</figcaption></figure>'


def main():
    a=json.loads((OUT/'results.json').read_text());f=json.loads((OUT/'results_fine.json').read_text());v=json.loads((OUT/'validation.json').read_text())
    rows=a['rows']+f['rows'];methods=['gram_fft','fd_lap4','fd_lap4_mass','gram_fft_fd4flux']
    def get(case,n,method,gravity='fd2',branch='self_consistent'):
        return next(r for r in rows if all(r[k]==val for k,val in [('case',case),('n',n),('method',method),('gravity',gravity),('branch',branch)]))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,gravity in zip(axs,['fd2','spectral']):
        for method in methods:
            rr=[get('perturbed_core',n,method,gravity) for n in (64,128,256)]
            ax.loglog([r['n'] for r in rr],[r['late_folded_profile_error'] for r in rr],'o-',label=LABELS[method])
        ax.set(title='FD2 gravity' if gravity=='fd2' else 'Spectral gravity',xlabel='Grid cells',ylabel='Relative averaged core-profile error');ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'core_convergence.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4));radius=(np.arange(16)+.5)*20/32
    baseline=get('perturbed_core',128,'gram_fft');reference=np.array(baseline['reference_late_folded_profile'])
    axs[0].plot(radius,reference,'k-',lw=3,alpha=.4,label='Reference')
    for method in methods[:3]:
        p=np.array(get('perturbed_core',128,method)['late_folded_profile'])
        axs[0].plot(radius,p,'o--',ms=3,label=LABELS[method])
        axs[1].plot(radius,(p-reference)/max(reference),'o-',ms=3,label=LABELS[method])
    axs[0].set(yscale='log',xlim=(0,6),ylim=(1e-7,3),xlabel='Radius',ylabel='Time-averaged folded density',title='Core profile, N=128, FD2 gravity')
    axs[1].set(xlim=(0,6),xlabel='Radius',ylabel='Profile residual / peak reference density',title='Small differences revealed by residuals')
    for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'core_profiles.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for method in methods:
        rr=[r for r in a['free_wave'] if r['method']==method]
        axs[0].loglog([r['n']/16 for r in rr],[r['density_l2'] for r in rr],'o-',label=LABELS[method])
    for order in (2,4):
        rr=[r for r in v['flux_spatial_order'] if r['laplacian_flag_order']==order]
        axs[1].loglog([r['n'] for r in rr],[r['continuity_rhs_relative_error'] for r in rr],'o-',label=f'GAMER flux, Laplacian-{order} branch')
    axs[0].set(xlabel='Cells per initial carrier wavelength',ylabel='Relative density error',title='Free multi-mode wave, T=0.5')
    axs[1].set(xlabel='Grid cells',ylabel='Relative continuity RHS error',title='Spatial order of the local mass correction')
    for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'correction_order.png',dpi=170);plt.close(fig)
    sha=a['gamer_commit'];source=f'https://github.com/gamer-project/gamer/blob/{sha}/src/Model_ELBDM/CPU_ELBDM/CPU_ELBDMSolver_FD.cpp'
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FD wave solver versus Gram-FE</title>
<style>body{{font:17px/1.6 system-ui;background:#edf3f7;color:#203648;margin:0}}main{{max-width:1160px;background:white;margin:auto;padding:32px}}h1,h2{{line-height:1.2}}h2{{margin-top:2em}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #d6e1e8}}th{{background:#edf3f7}}img{{width:100%}}figure{{margin:25px 0}}figcaption{{font-size:14px;color:#50677a}}.box{{background:#e7f3f8;border-left:5px solid #367eaa;padding:16px}}pre{{background:#eef3f7;padding:16px;overflow:auto}}a{{color:#155f90}}</style><main>
<h1>Does a resolved density profile hide the benefit of Gram-FE?</h1>
<div class="box"><strong>Yes, in the localized-core test with FD2 gravity.</strong> Once the wave error is below the common gravity error, the FD Laplacian-4 and Gram-FE profiles become very similar. Spectral gravity reveals the remaining wave-solver difference. Traveling interference remains a counterexample: its wave/density fields can differ substantially even when coarse averaged profiles appear close.</div>
<h2>Exactly which algorithms were compared?</h2>
<p>The cached <a href="{source}">GAMER CPU wave source</a>, revision <code>{sha}</code>, exposes <code>LAPLACIAN_4TH</code> off/on: the actual Laplacians are second-order (three points) and fourth-order (five points). We label these Laplacian-2 and Laplacian-4, avoiding an unsupported identification with a separate sixth-order Laplacian. This revision contains no sixth-order Laplacian branch in that solver. The Taylor update uses the default coefficient 1/6, with automatic optimization off as in the cached zoom example.</p>
<p>The FD half/full-step update and the optional face-flux amplitude correction are transcribed from the source. On the periodic grid, FFTs evaluate the finite-difference symbol exactly; this does not turn the derivative into a spectral derivative. An independent scalar real/imaginary implementation verifies the vectorized port. Both conservation settings are tested. No global normalization is applied.</p>
<p><strong>Correction to the previous discussion:</strong> Gram-FE does not implement this local continuity correction. Its absence is intentional, not a missing part of our Gram-FE reconstruction. Unchanged FFT and MATMUL Gram-FE are the production-method controls here. A separately labeled experimental Gram-FE+FD4flux applies the same low-order flux/amplitude formula using a Gram-FE half-step predictor. It is not a GAMER option and is not proposed as a solver improvement.</p>
<p>All methods share initial samples, gravity, timestep and second-order Strang splitting at each grid. The comparison includes recorded accurate potential, self-consistent spectral gravity, and self-consistent FD2 gravity. Wave grids are uniform: no AMR wave transfer, refluxing, density floors, directional splitting, or cosmological expansion. The gravity Poisson solve is exact for its selected discrete symbol, excluding relaxation error.</p>
<h2>Localized core: the user's observed pattern</h2>
<p>The core has mass 4 in a periodic box of length 20, with the same weak phase/packet perturbation as the replay experiment. It evolves to T=6. The profile is density averaged over the last third of the run, folded about the box center into 16 radial bins. It is a 1D proxy for a coarse density profile, not a spherical halo or a core-halo relation. Every profile is compared to a converged reference sampled and binned identically; mass is not normalized before comparison.</p>
<p>Profile errors below are percentages. At N=128, FD2 gravity masks much of Gram-FE's propagation advantage. Replacing gravity alone by spectral gravity exposes it.</p>
'''
    for n in (64,128,256):
        text+=f'<h3>Core, N={n}</h3>'+table(['Wave method','FD2 gravity: profile error (%)','Spectral gravity: profile error (%)','FD2 gravity: final density error (%)'],
            [[LABELS[m],100*get('perturbed_core',n,m)['late_folded_profile_error'],100*get('perturbed_core',n,m,'spectral')['late_folded_profile_error'],100*get('perturbed_core',n,m)['density_l2']] for m in methods])
    text+=pic('core_convergence.png','Same core and wave grid: a common FD2 gravity error masks the wave-solver advantage. Spectral gravity separates the curves. The experimental correction is not a production Gram-FE method.')
    text+=pic('core_profiles.png','Profiles can visually overlap even when their smaller residuals differ. The left panel shows absolute averaged density; the right panel exposes the bias relative to the reference.')
    text+='''<p>Agreement between two methods is not proof of convergence: both can inherit the same gravity bias. Here an independent reference establishes the bias. Conversely, tiny differences in an averaged profile do not imply equivalent phase, interference, force fluctuations, or time-dependent dynamics. The perturbed core has no single wavelength, so its grid sizes cannot be converted into a universal points-per-wave threshold.</p>
<h2>Conservation versus convergence</h2><p>The independent instantaneous-current test uses a known smooth finite-Fourier wave and its exact continuity RHS. Both GAMER flux branches approach second-order spatial convergence, including the branch paired with the fourth-order Laplacian. The free multi-mode test separately measures the accumulated evolution error. It confirms that appending the FD correction can remove much of Gram-FE's accuracy advantage.</p>'''
    text+=pic('correction_order.png','Left: evolving a free multi-mode wave, using the same timestep for each method on a given grid. Right: a direct spatial consistency test isolates the order of the continuity flux from time integration.')
    text+=table(['Cells/carrier','Method','Relative density error','Mean component wave L1'],
        [[r['n']/16,LABELS[r['method']],r['density_l2'],r['wave_l1']] for r in a['free_wave'] if r['method'] in methods])
    text+='''<p>The free test uses the enveloped interference initial condition with carrier k=16, maximum initial wave number 17, no gravity, T=0.5, and dt no larger than 0.25465 h^2. Thus nominal 6, 12 and 24 carrier cells correspond to 5.65, 11.29 and 22.59 cells at the shortest initial wavelength. This is not a rerun of the paper's Gaussian test. A constant-amplitude plane wave would hide the spatial error in the correction because its current is spatially constant.</p>
<p>With the correction enabled, total mass is conserved to roundoff when every proposed corrected density is positive. GAMER skips the correction when the proposed density or raw updated amplitude is nonpositive; this exact guard is retained and counted. Some interference and near-vacuum runs trigger it, so enabling the flag alone does not guarantee exact mass conservation in those runs.</p>
<h2>Interference tests: where the methods still differ</h2><p>These tables give final relative density errors in percent. Results remain separate from the averaged profile metric. Both old resolutions and an additional fine grid are shown where available.</p>'''
    for case in ('standing','weak_envelope','strong_envelope'):
        ns=sorted(set(r['n'] for r in rows if r['case']==case))
        text+='<h3>'+escape(case)+'</h3>'+table(['Cells/carrier','Wave method','Replay accurate potential (%)','Self spectral gravity (%)','Self FD2 gravity (%)','FD2 averaged profile (%)'],
            [[n/16,LABELS[m],100*get(case,n,m,'spectral','prescribed')['density_l2'],100*get(case,n,m,'spectral')['density_l2'],100*get(case,n,m)['density_l2'],100*get(case,n,m)['late_folded_profile_error']] for n in ns for m in methods])
    text+='''<p><strong>Standing-wave limitation:</strong> the two dominant carriers have opposite k but equal kinetic frequency. A symmetric dispersion error shifts their common phase and largely cancels from their leading interference density. This makes that case useful for isolating gravity but a weak test of generic wave-solver equivalence. The moving/enveloped waves break this degeneracy.</p>
<p><strong>Interpretation:</strong> the core result supports a shared gravity limitation as one explanation for similar high-resolution profiles. It does not establish that this is the dominant mechanism in GAMER halos. Spatial/time averaging, conservation corrections and refinement communication can also hide or replace the benefit of a more accurate wave solver.</p>
<h2>Validation, time controls and complete data</h2>'''
    text+=table(['Check','Value'],[['Max difference from previous FFT/Gram-FE replay results',v['max_previous_result_difference']],['Max corrected mass error when no guard fires',v['max_mass_error_when_no_guard']],['Runs with skipped correction cells',len(v['guarded_runs'])],['Max scalar-vs-vector FD port difference',max(r['max_difference'] for r in v['stencil_port'])]])
    text+='<details><summary>Half-timestep comparisons</summary>'+table(['Case','Method','Gravity','Density error','Half-dt density error','Profile error','Half-dt profile error'],
        [[r['case'],LABELS[r['method']],r['gravity'],r['density_base'],r['density_half'],r['profile_base'],r['profile_half']] for r in v['time_controls']])+'</details>'
    text+='<details><summary>All coupled results, including Laplacian-2, MATMUL, mass and positivity guards</summary>'+table(['Case','N','Method','Branch','Gravity','Density error','Wave L2','Mass change','Skipped cell-steps'],
        [[r['case'],r['n'],LABELS[r['method']],r['branch'],r['gravity'],r['density_l2'],r['wave_l2'],r['mass_change'],r['skipped_correction_cellsteps']] for r in rows])+'</details>'
    text+='''<p>The existing replay reference has independently verified spatial/time wave errors below 1e-10 at the old grids; the new runs reuse that reference algorithm and initial data. Fine runs use smaller reference and evolution steps. Half-step controls test the FD/correction variants at lower grids. Previous Gram-FE and global FFT results are reproduced independently. The small-grid FD port is checked against literal physical-space stencils and scalar C++-style flux formulas, including the optimized Taylor coefficient as a unit check. Production comparisons use automatic Taylor optimization off. No compiled GAMER binary was run.</p>
<h2>Reproduce</h2><pre>python toy_models/fd_wave_crosscheck.py
python toy_models/fd_wave_crosscheck.py --fine
python toy_models/fd_wave_crosscheck.py --controls
python toy_models/verify_fd_wave.py
python toy_models/build_fd_wave_report.py</pre>
<p>Use pinn_env Python. New data: <a href="fd_wave_results/results.json">main results</a>, <a href="fd_wave_results/results_fine.json">fine grids</a>, <a href="fd_wave_results/results_controls.json">time controls</a>, <a href="fd_wave_results/validation.json">validation and profile differences</a>. Final wave fields are saved in NPZ files alongside the JSON, and plots are standalone PNGs embedded here. <a href="potential_replay.html">Previous potential replay report</a>. All prior numerical outputs are preserved.</p></main></html>'''
    (ROOT/'fd_wave_crosscheck.html').write_text(text,encoding='utf-8')
    print('Wrote fd_wave_crosscheck.html')


if __name__=='__main__':main()
