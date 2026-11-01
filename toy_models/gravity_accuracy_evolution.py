"""Self-consistent 1D SP: can better gravity replace a factor-two refinement?

Periodic Fourier and actual-table Gram-FE propagation, identical Strang steps.
New outputs only. No claims of spherical 3D halo dynamics.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from experiments import np,plt,inverse_laplacian,modes
from scipy.signal import resample
from gramfe_probe import operator,periodic_update

OUT=Path(__file__).resolve().parent/'gravity_accuracy_results';OUT.mkdir(exist_ok=True)


def initial_wave(n,case):
    x=np.arange(n)*2*np.pi/n
    p=np.exp(16j*x)+.7*np.exp(-16j*x+.4j)
    if case=='enveloped':p=(1+.25*np.cos(x))*p+.2*np.exp(7j*x+.7j)
    return p/np.sqrt(np.mean(abs(p)**2))


def diagnostics(p,length,green,coupling):
    n=len(p);h=length/n;k=modes(n,length);rho=abs(p)**2
    ph=np.fft.ifft(green*np.fft.fft(rho)).real*coupling
    pk=np.fft.fft(p);kinetic=float(.5*h/n*np.sum(k*k*abs(pk)**2))
    grav=float(.5*h*np.dot(rho,ph))
    derivative=np.fft.ifft(1j*k*pk);force=-np.fft.ifft(1j*k*np.fft.fft(ph)).real
    work=float(h*np.dot(np.imag(p.conj()*derivative),force))
    return dict(kinetic=kinetic,gravitational=grav,energy=kinetic+grav,
                mass=float(h*np.sum(rho)),work_rate=work,
                power_below_five_cells=float(np.sum(abs(pk[abs(k*h)>2*np.pi/5])**2)/np.sum(abs(pk)**2)))


def evolve(p,length,coupling,gravity,kinetic,final,dt,order=2,samples=80,observe=None):
    n=len(p);h=length/n;k=modes(n,length);steps=int(np.ceil(final/dt));dt=final/steps
    green=inverse_laplacian(n,length,gravity);truegreen=inverse_laplacian(n,length,'spectral')
    a=1/(2-2**(1/3));factors=[1.] if order==2 else [a,1-2*a,a]
    stages=[(dt*f,np.exp(-.5j*k*k*dt*f),None if kinetic=='global_fft' else operator(dt*f/h**2,kinetic)) for f in factors]
    p=p.copy();history=[]
    for step in range(steps+1):
        if step%max(1,steps//samples)==0 or step==steps:
            own=diagnostics(p,length,green,coupling);physical=diagnostics(p,length,truegreen,coupling)
            history.append(dict(time=step*dt,**own,continuum_energy=physical['energy'],**({} if observe is None else observe(p))))
        if step==steps:break
        for stepdt,drift,op in stages:
            phi=np.fft.ifft(green*np.fft.fft(abs(p)**2)).real*coupling
            p*=np.exp(-.5j*stepdt*phi)
            p=np.fft.ifft(np.fft.fft(p)*drift) if op is None else periodic_update(p,op)
            phi=np.fft.ifft(green*np.fft.fft(abs(p)**2)).real*coupling
            p*=np.exp(-.5j*stepdt*phi)
    work=float(np.trapezoid([r['work_rate'] for r in history],[r['time'] for r in history]))
    return p,dict(steps=steps,dt=dt,history=history,integrated_gravity_work=work,
                  kinetic_change=history[-1]['kinetic']-history[0]['kinetic'])


def compare(p,reference):
    rho=abs(p)**2;ref=abs(reference)**2
    # Evaluate reference density itself, rather than square a truncated wave.
    target=resample(ref,len(p)).real
    low=np.fft.fft(rho)/len(p);lowref=np.fft.fft(target)/len(p)
    ids=np.r_[np.arange(1,5),np.arange(len(p)-4,len(p))]
    return dict(density_error=float(np.linalg.norm(rho-target)/np.linalg.norm(target)),
                low_density_absolute_rms=float(np.linalg.norm((low-lowref)[ids])),
                peak_density_bias=float(np.max(rho)/np.max(target)-1))


def main():
    start=perf_counter();rows=[];checks=[];references={};final=.5;dt=.0002;coupling=100.
    for case in ('standing','enveloped'):
        ref,stats=evolve(initial_wave(768,case),2*np.pi,coupling,'spectral','global_fft',final,dt,order=4,samples=400)
        temporal,_=evolve(initial_wave(768,case),2*np.pi,coupling,'spectral','global_fft',final,dt/2,order=4)
        spatial,_=evolve(initial_wave(1536,case),2*np.pi,coupling,'spectral','global_fft',final,dt,order=4)
        checks.append(dict(case=case,time_reference_error=compare(ref,temporal)['density_error'],
                           space_reference_error=compare(ref,spatial)['density_error']))
        references[case]=stats
        for n in (80,96,160,192):
            for kinetic in ('global_fft','fft_fused','stored_matmul'):
                for gravity in ('fd2','fd4','spectral'):
                    p,stats=evolve(initial_wave(n,case),2*np.pi,coupling,gravity,kinetic,final,dt)
                    own=stats['history'];energy_scale=abs(own[0]['kinetic'])+abs(own[0]['gravitational'])
                    rows.append(dict(case=case,n=n,carrier_ppw=n/16,shortest_initial_ppw=n/(17 if case=='enveloped' else 16),
                        kinetic=kinetic,gravity=gravity,**compare(p,ref),**stats,
                        own_energy_drift=(own[-1]['energy']-own[0]['energy'])/energy_scale,
                        continuum_energy_drift=(own[-1]['continuum_energy']-own[0]['continuum_energy'])/energy_scale))
            print('coupled accuracy',case,n,flush=True)
        # Time convergence at the lower resolutions is essential before attribution.
        for n in (80,96):
            for kinetic in ('global_fft','fft_fused'):
                for gravity in ('fd2','fd4','spectral'):
                    p,stats=evolve(initial_wave(n,case),2*np.pi,coupling,gravity,kinetic,final,dt/2)
                    checks.append(dict(case=case,n=n,kinetic=kinetic,gravity=gravity,half_dt_density_error=compare(p,ref)['density_error']))
    assert max(c['time_reference_error'] for c in checks if 'time_reference_error' in c)<1e-6
    assert max(c['space_reference_error'] for c in checks if 'space_reference_error' in c)<1e-7
    result=dict(rows=rows,checks=checks,references=references,final_time=final,coupling=coupling,
                seconds=perf_counter()-start,clarification='User confirms doubling resolution has always helped. The target is the higher-than-expected cells-per-wavelength requirement, not nonconvergence.',
                scope='Periodic 1D self-consistent SP; two coherent initial conditions; uniform point samples, no AMR or cosmology. Fixed identical Strang steps for compared runs. Spectral high-resolution fourth-order reference, independent time/space refinement. Gravity work is sampled j dot (-grad Phi) quadrature, not an exact discrete energy identity.')
    (OUT/'evolution.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,case in zip(axs,('standing','enveloped')):
        for kinetic in ('global_fft','fft_fused'):
            for gravity in ('fd2','fd4','spectral'):
                rr=[r for r in rows if r['case']==case and r['kinetic']==kinetic and r['gravity']==gravity]
                ax.loglog([r['carrier_ppw'] for r in rr],[r['density_error'] for r in rr],
                          'o-' if kinetic=='global_fft' else 's--',label=kinetic+' / '+gravity)
        ax.set(xlabel='Cells per carrier wavelength',ylabel='Final relative density error',title=case);ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'evolution.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(checks=checks,summary=[{k:v for k,v in r.items() if k in ('case','n','kinetic','gravity','density_error','own_energy_drift')} for r in rows if r['kinetic']=='global_fft']),indent=2))


if __name__=='__main__':main()
