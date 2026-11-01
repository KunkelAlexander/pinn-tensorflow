"""Recorded-potential ablation of Gram-FE and self-consistent gravity.

Only writes potential_replay_results/. All wave grids are uniform; hierarchy
branches alter gravity alone. No claim to reproduce GAMER's AMR evolution.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
from experiments import np, plt, inverse_laplacian, modes
from gramfe_probe import operator, periodic_update, LAMBDA
from gravity_accuracy_evolution import initial_wave, sample_periodic
from gravity_hierarchy_probe import interpolation
from frozen_hamiltonian import setup
from scipy.signal import resample
from scipy.linalg import solve_banded

OUT = Path(__file__).resolve().parent / 'potential_replay_results'


def errors(p, ref):
    d = p-ref
    aligned = p*np.exp(-1j*np.angle(np.vdot(ref,p)))
    rho, target = abs(p)**2, abs(ref)**2
    return dict(wave_l1=float((np.mean(abs(d.real))+np.mean(abs(d.imag)))/2),
                wave_l2=float(np.linalg.norm(d)/np.linalg.norm(ref)),
                aligned_wave_l2=float(np.linalg.norm(aligned-ref)/np.linalg.norm(ref)),
                density_l2=float(np.linalg.norm(rho-target)/np.linalg.norm(target)),
                mass_bias=float(np.sum(rho)/np.sum(target)-1))


def gravity_map(n, length, coupling, label):
    """Map point density to zero-mean potential, keeping gauge identical."""
    green = coupling*inverse_laplacian(n,length,label if label!='hierarchy2' else 'fd2')
    if label != 'hierarchy2':
        return lambda rho: np.fft.ifft(green*np.fft.fft(rho)).real
    # One parent grid (2h), one centered child spanning half the periodic box.
    # Root receives point density samples, child receives all fine point values.
    # Direct tridiagonal inversion excludes relaxation convergence as a cause.
    h=length/n; x=np.arange(n)*h; coarse=n//2
    rootgreen=coupling*inverse_laplacian(coarse,length,'fd2')
    transfer=interpolation(x[::2],x,2,periodic=True)
    lo,hi=n//4,3*n//4; count=hi-lo-1
    band=np.zeros((3,count));band[0,1:]=1;band[1]=-2;band[2,:-1]=1
    def potential(rho):
        phi=transfer@np.fft.ifft(rootgreen*np.fft.fft(rho[::2])).real
        rhs=coupling*h*h*(rho[lo+1:hi]-rho.mean())
        rhs[0]-=phi[lo];rhs[-1]-=phi[hi]
        phi[lo+1:hi]=solve_banded((1,1),band,rhs,check_finite=False)
        return phi-phi.mean()
    return potential


def reference(p0,length,coupling,dt,steps,targets,substeps=2):
    """Yoshida-4 reference; store wave and true potential at every kick time.

    No temporal interpolation or low-pass density downsampling is used.
    The prescribed run is endpoint Strang for a nonautonomous linear equation.
    """
    n=len(p0);k=modes(n,length);g=gravity_map(n,length,coupling,'spectral')
    a=1/(2-2**(1/3));factors=[a,1-2*a,a]
    stages=[(dt*f/substeps,np.exp(-.5j*k*k*dt*f/substeps)) for f in factors]
    waves={m:np.empty((steps+1,m),complex) for m in targets}
    pots={m:np.empty((steps+1,m),float) for m in targets}
    band={m:dict(max_power_below_six_cells=0.,max_power_below_five_cells=0.,
                 max_power_beyond_nyquist=0.,max_unresolved_density_rms=0.) for m in targets}
    p=p0.copy();stride=max(1,steps//100);history=[]
    freq=abs(np.fft.fftfreq(n)*n)
    for j in range(steps+1):
        rho=abs(p)**2;phi=g(rho);wp=abs(np.fft.fft(p))**2
        rp=abs(np.fft.fft(rho))**2;rp[0]=0
        for m in targets:
            waves[m][j]=sample_periodic(p,m)
            pots[m][j]=sample_periodic(phi,m).real
            b=band[m]
            for key,mask in [('max_power_below_six_cells',freq>m/6+1e-10),
                             ('max_power_below_five_cells',freq>m/5+1e-10),
                             ('max_power_beyond_nyquist',freq>=m/2)]:
                b[key]=max(b[key],float(wp[mask].sum()/wp.sum()))
            b['max_unresolved_density_rms']=max(b['max_unresolved_density_rms'],
                float(np.sqrt(rp[freq>=m/2].sum()/max(rp.sum(),1e-100))))
        if j%stride==0 or j==steps:
            history.append(dict(time=j*dt,bandwidth={m:dict(band[m]) for m in targets}))
        if j==steps:break
        for _ in range(substeps):
            for d,drift in stages:
                p*=np.exp(-.5j*d*g(abs(p)**2))
                p=np.fft.ifft(np.fft.fft(p)*drift)
                p*=np.exp(-.5j*d*g(abs(p)**2))
    # Keep the continuum zero-mean gauge (do not independently rephase runs).
    return dict(waves=waves,pots=pots,band=band,history=history,final=p)


def rollout(ref,length,coupling,n,dt,kinetic,branch,gravity,conserve_mass=False):
    target=ref['waves'][n];steps=len(target)-1;p=target[0].copy();h=length/n
    k=modes(n,length);drift=np.exp(-.5j*k*k*dt)
    op=None if kinetic=='global_fft' else operator(dt/h**2,kinetic)
    g=gravity_map(n,length,coupling,gravity)
    # Potential produced by the approximate map from the REFERENCE density.
    # For the spectral oracle use the fine potential, not a coarse re-solve.
    replay=ref['pots'][n] if gravity=='spectral' and branch=='prescribed' else np.array([g(abs(v)**2) for v in target])
    initial_mass=np.sum(abs(p)**2);history=[];kick_work=0.;drift_work=0.
    stride=max(1,steps//100);correction_work=0.
    def ke(v):return float(.5*h/n*np.sum(k*k*abs(np.fft.fft(v))**2))
    initial_ke=ke(p)
    for j in range(steps+1):
        if j%stride==0 or j==steps:
            history.append(dict(time=j*dt,**errors(p,target[j])))
        if j==steps:break
        phi=g(abs(p)**2) if branch=='self_consistent' else replay[j]
        before=ke(p);p*=np.exp(-.5j*dt*phi);after=ke(p);kick_work+=after-before
        p=np.fft.ifft(np.fft.fft(p)*drift) if op is None else periodic_update(p,op)
        now=ke(p);drift_work+=now-after
        if conserve_mass:
            p*=np.sqrt(initial_mass/np.sum(abs(p)**2))
            corrected=ke(p);correction_work+=corrected-now;now=corrected
        phi=g(abs(p)**2) if branch=='self_consistent' else replay[j+1]
        p*=np.exp(-.5j*dt*phi);kick_work+=ke(p)-now
    row=dict(n=n,dt=dt,steps=steps,kinetic=kinetic,branch=branch,gravity=gravity,
             **errors(p,target[-1]),mass_change=float(np.sum(abs(p)**2)/initial_mass-1),
             kinetic_kick_work=kick_work,numerical_drift_work=drift_work,
             work_closure=ke(p)-initial_ke-kick_work-drift_work-correction_work,history=history)
    if conserve_mass:row.update(global_mass_normalization=True,mass_normalization_work=correction_work)
    return p,row


def cases():
    return dict(standing=dict(length=2*np.pi,g=100.,dt=.0002,final=.5,ns=[96,192],nr=768),
                weak_envelope=dict(length=2*np.pi,g=10.,dt=.0002,final=.5,ns=[96,192],nr=768),
                strong_envelope=dict(length=2*np.pi,g=100.,dt=.0002,final=.5,ns=[96,192],nr=768),
                perturbed_core=dict(length=20.,g=1.,dt=.002,final=6.,ns=[64,128],nr=512))


def initial(case,n,ground=None):
    if case!='perturbed_core':return initial_wave(n,'standing' if case=='standing' else 'enveloped')
    x=np.arange(n)*20/n-10;p=resample(ground,n)*np.exp(.03j*np.cos(2*np.pi*x/5))
    for shift in (-20,0,20):
        p+=.02*np.exp(-(x-3+shift)**2/(2*1.2**2))*np.exp(1j*np.pi*(x+shift))
    return p*np.sqrt(4/(20/n*np.sum(abs(p)**2)))


def plane_check():
    rows=[]
    for n in (64,80,96,128):
        x=np.arange(n)/n;p0=np.exp(32j*np.pi*x);steps=int(np.ceil(.015/(LAMBDA/n**2)));dt=.015/steps
        ref=p0*np.exp(-.5j*(32*np.pi)**2*.015)
        for kinetic in ('global_fft','fft_fused','stored_matmul'):
            p=p0.copy();op=None if kinetic=='global_fft' else operator(dt*n*n,kinetic)
            drift=np.exp(-.5j*modes(n,1)**2*dt)
            for _ in range(steps):p=np.fft.ifft(np.fft.fft(p)*drift) if op is None else periodic_update(p,op)
            rows.append(dict(n=n,ppw=n/16,kinetic=kinetic,steps=steps,dt=dt,**errors(p,ref)))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--controls',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);start=perf_counter();allrows=[];checks={};bands={};histories={};fields={}
    _,_,_,ground,_,ground_residual=setup(257)
    for case,c in cases().items():
        length,g,dt=c['length'],c['g'],c['dt'];ns=c['ns'];nr=c['nr'];steps=round(c['final']/dt)
        if args.controls:dt/=2;steps*=2;ns=ns[:1]
        ref=reference(initial(case,nr,ground),length,g,dt,steps,ns)
        print('reference',case,'controls',args.controls,flush=True)
        if not args.controls:
            finer=reference(initial(case,nr*2,ground),length,g,dt,steps,[ns[0]])
            smaller=reference(initial(case,nr,ground),length,g,dt,steps,[ns[0]],substeps=4)
            checks[case]=dict(space=errors(ref['waves'][ns[0]][-1],finer['waves'][ns[0]][-1]),
                              time=errors(ref['waves'][ns[0]][-1],smaller['waves'][ns[0]][-1]))
            del finer,smaller
        bands[case]=ref['band'];histories[case]=ref['history']
        for n in ns:
            fields[f'{case}_{n}_reference']=ref['waves'][n][-1]
            gravities=['spectral','fd2','fd4']+(['hierarchy2'] if case=='perturbed_core' else [])
            kinetics=['global_fft','fft_fused'] if args.controls else ['global_fft','fft_fused','stored_matmul']
            if args.controls:gravities=['spectral','fd2']+(['hierarchy2'] if case=='perturbed_core' else [])
            for kinetic in kinetics:
                for gravity in gravities:
                    for branch in (('prescribed','prescribed_coarse','self_consistent') if gravity=='spectral' else ('prescribed','self_consistent')):
                        p,row=rollout(ref,length,g,n,dt,kinetic,branch,gravity);row['case']=case;allrows.append(row)
                        fields[f'{case}_{n}_{kinetic}_{gravity}_{branch}']=p
            print(case,n,'runs complete',round(perf_counter()-start,1),'seconds',flush=True)
        # Checkpoint new results after each case, so long runs are reviewable.
        result=dict(rows=allrows,checks=checks,bandwidth=bands,reference_history=histories,
                    ground_residual=ground_residual,seconds=perf_counter()-start,
                    scope='1D periodic SP. Reference: spectral Yoshida4, two substeps per target step, time/space controls. Recorded continuum-zero-mean potential at both kick endpoints, no temporal interpolation. Same initial point samples for every branch. Approximate prescribed potential solves on sampled reference density. Self branches solve on evolving density. Uniform wave grids, same-level actual-table Gram-FE, no wave AMR/subcycling. Core-only one-child FD2 gravity hierarchy uses point restriction, quadratic parent-potential interpolation and direct child solve; not GAMER. 1D harmonic boundary errors cannot reproduce 3D tidal errors. Bandwidth maxima sampled at every target step; strict six-cell cutoff excludes exactly-six-cell modes. Wave L1 is mean of absolute real and imaginary component errors; L2 relative; aligned L2 removes one global phase. Density errors are pointwise against full reference wave evaluated at target nodes.')
        suffix='_controls' if args.controls else ''
        (OUT/f'results{suffix}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(OUT/f'fields{suffix}.npz',**fields)
    if not args.controls:
        result['plane_calibration']=plane_check()
        (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    assert max(abs(r['work_closure']) for r in allrows)<1e-8
    if not args.controls:
        assert max(v['time']['wave_l2'] for v in checks.values())<1e-7,checks
        assert max(v['space']['wave_l2'] for v in checks.values())<1e-7,checks
    print('done',round(perf_counter()-start,1),'seconds',flush=True)


if __name__=='__main__':main()
