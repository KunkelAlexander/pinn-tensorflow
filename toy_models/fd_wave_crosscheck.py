"""GAMER FD wave steps versus Gram-FE, with local conservation on/off.

The pinned source has second/fourth-order Laplacians, explicitly labeled here.
Gram-FE+FD4flux is an experimental ablation, not a GAMER implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from experiments import np,modes
from gramfe_probe import operator,periodic_update,LAMBDA
from potential_replay import reference,initial,cases,errors,gravity_map
from frozen_hamiltonian import setup

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'fd_wave_results'
METHODS=['global_fft','gram_fft','gram_matmul','fd_lap2','fd_lap2_mass','fd_lap4','fd_lap4_mass','gram_fft_fd4flux']


def lap(u,order):
    if order==2:return np.roll(u,1)-2*u+np.roll(u,-1)
    return (-np.roll(u,2)+16*np.roll(u,1)-30*u+16*np.roll(u,-1)-np.roll(u,-2))/12


def coefficient(alpha,order,auto=False):
    if not auto:return 1/6
    # Algebraically rationalized form of ELBDM_SetTaylor3Coeff.cpp.
    if order==4:
        disc=81-576*alpha**2
        return 1.000157*2.25/(9+np.sqrt(disc)) if disc>=0 else 1/6
    disc=1-4*alpha**2
    return 1.000157/(4*(1+np.sqrt(disc))) if disc>=0 else 1/6


def continuity(old,raw,half,alpha,order):
    """Literal complex equivalent of GAMER's face flux/amplitude correction."""
    plus=np.roll(half,-1)
    if order==2:
        face=(half+plus)/2;derivative=plus-half
    else:
        minus=np.roll(half,1);plus2=np.roll(half,-2)
        face=(-minus+15*half+15*plus-plus2)/28
        derivative=(minus-15*half+15*plus-plus2)/12
    flux=2*np.imag(face.conj()*derivative)
    density=abs(old)**2-alpha*(flux-np.roll(flux,1))
    newdensity=abs(raw)**2;valid=(density>0)&(newdensity>0)
    out=raw.copy();out[valid]*=np.sqrt(density[valid]/newdensity[valid])
    return out,int(np.sum(~valid)),float(density.min())


def make_step(n,length,dt,method,auto=False):
    h=length/n;alpha=dt/(2*h*h);theta=2*np.pi*np.fft.fftfreq(n)
    if method=='global_fft':
        multiplier=np.exp(-.5j*modes(n,length)**2*dt)
        return lambda p:(np.fft.ifft(np.fft.fft(p)*multiplier),0,0.)
    if method.startswith('gram'):
        kind='stored_matmul' if method=='gram_matmul' else 'fft_fused'
        op=operator(dt/h**2,kind)
        if method!='gram_fft_fd4flux':return lambda p:(periodic_update(p,op),0,0.)
        halfop=operator(dt/(2*h*h),kind)
        def corrected(p):return continuity(p,periodic_update(p,op),periodic_update(p,halfop),alpha,4)
        return corrected
    order=4 if 'lap4' in method else 2;c=coefficient(alpha,order,auto)
    s=np.sin(theta/2)**2;ell=-4*s*(1+s/3 if order==4 else 1)
    # These FFTs evaluate precisely the periodic FD stencil, not spectral d2/dx2.
    halfmult=1+.5j*alpha*ell-c*alpha**2*ell**2
    fullmult=1+1j*alpha*ell-.5*alpha**2*ell**2-1j*c*alpha**3*ell**3
    def advance(p):
        coeff=np.fft.fft(p);raw=np.fft.ifft(fullmult*coeff)
        if not method.endswith('_mass'):return raw,0,0.
        return continuity(p,raw,np.fft.ifft(halfmult*coeff),alpha,order)
    return advance


def profile(rho,length):
    n=len(rho);radius=abs(np.arange(n)*length/n-length/2)
    bins=np.minimum((radius/(length/32)).astype(int),15)
    count=np.bincount(bins,minlength=16)
    return np.bincount(bins,weights=rho,minlength=16)/np.maximum(count,1)


def run(ref,n,length,coupling,dt,method,branch,gravity,auto=False):
    target=ref['waves'][n];p=target[0].copy();steps=len(target)-1
    advance=make_step(n,length,dt,method,auto);g=gravity_map(n,length,coupling,gravity)
    oldmass=float(np.sum(abs(p)**2));skip=0;mindens=0.;history=[]
    sumrho=np.zeros(n);sumref=np.zeros(n);samples=0
    stride=max(1,steps//100);radius=abs(np.arange(n)*length/n-length/2)
    for j in range(steps+1):
        if j%stride==0 or j==steps:
            rho=abs(p)**2;rr=abs(target[j])**2
            history.append(dict(time=j*dt,**errors(p,target[j]),
                radius=float(np.sqrt(np.sum(radius**2*rho)/rho.sum())),
                reference_radius=float(np.sqrt(np.sum(radius**2*rr)/rr.sum()))))
            if j>=2*steps//3:sumrho+=rho;sumref+=rr;samples+=1
        if j==steps:break
        phi=ref['pots'][n][j] if branch=='prescribed' else g(abs(p)**2)
        p*=np.exp(-.5j*dt*phi)
        p,bad,minimum=advance(p);skip+=bad;mindens=min(mindens,minimum)
        phi=ref['pots'][n][j+1] if branch=='prescribed' else g(abs(p)**2)
        p*=np.exp(-.5j*dt*phi)
        if not np.isfinite(p).all():raise RuntimeError((method,n,branch,gravity,j,'nonfinite'))
    avg=sumrho/samples;avgref=sumref/samples;prof=profile(avg,length);profref=profile(avgref,length)
    return p,dict(n=n,method=method,branch=branch,gravity=gravity,dt=dt,steps=steps,taylor_auto=auto,
        **errors(p,target[-1]),mass_change=float(np.sum(abs(p)**2)/oldmass-1),
        skipped_correction_cellsteps=skip,min_proposed_density=mindens,
        final_peak_bias=float(max(abs(p)**2)/max(abs(target[-1])**2)-1),
        final_radius_bias=history[-1]['radius']/history[-1]['reference_radius']-1,
        late_density_error=float(np.linalg.norm(avg-avgref)/np.linalg.norm(avgref)),
        late_folded_profile_error=float(np.linalg.norm(prof-profref)/np.linalg.norm(profref)),
        late_folded_profile=prof.tolist(),reference_late_folded_profile=profref.tolist(),history=history)


def independent_checks():
    rng=np.random.default_rng(781);p=rng.normal(size=64)+1j*rng.normal(size=64)
    checks=[];dt=.00017;length=2*np.pi;alpha=dt/(2*(length/64)**2)
    for order in (2,4):
        for auto in (False,True):
            c=coefficient(alpha,order,auto);half=p+.5j*alpha*lap(p,order)-c*alpha**2*lap(lap(p,order),order)
            raw=p+1j*alpha*lap(half,order)
            for conserve in (False,True):
                expected=raw.copy();bad=0
                if conserve:
                    # Independent scalar transcription of C++ real/imag flux.
                    flux=[]
                    for i in range(64):
                        if order==4:
                            v=[half[(i+j)%64] for j in (-1,0,1,2)]
                            r=(-v[0].real+15*v[1].real+15*v[2].real-v[3].real)/28
                            im=(-v[0].imag+15*v[1].imag+15*v[2].imag-v[3].imag)/28
                            dr=(v[0].real-15*v[1].real+15*v[2].real-v[3].real)/12
                            di=(v[0].imag-15*v[1].imag+15*v[2].imag-v[3].imag)/12
                        else:
                            r=(half[i].real+half[(i+1)%64].real)/2;im=(half[i].imag+half[(i+1)%64].imag)/2
                            dr=half[(i+1)%64].real-half[i].real;di=half[(i+1)%64].imag-half[i].imag
                        flux.append(2*(r*di-im*dr))
                    for i in range(64):
                        den=abs(p[i])**2-alpha*(flux[i]-flux[i-1])
                        if den>0 and abs(raw[i])**2>0:expected[i]*=np.sqrt(den/abs(raw[i])**2)
                        else:bad+=1
                actual,skipped,_=make_step(64,length,dt,f'fd_lap{order}'+('_mass' if conserve else ''),auto)(p)
                err=float(max(abs(actual-expected)));assert err<1e-12 and skipped==bad
                checks.append(dict(order=order,auto=auto,conserve=conserve,max_difference=err))
    return checks


def free_wave_tests():
    rows=[];length=2*np.pi;final=.5
    for n in (96,192,384):
        dtmax=LAMBDA*(length/n)**2;steps=int(np.ceil(final/dtmax));dt=final/steps
        p0=initial('weak_envelope',n);exact=np.fft.ifft(np.fft.fft(p0)*np.exp(-.5j*modes(n,length)**2*final))
        for method in METHODS:
            advance=make_step(n,length,dt,method);p=p0.copy();skip=0
            for _ in range(steps):p,bad,_=advance(p);skip+=bad
            rows.append(dict(n=n,carrier_ppw=n/16,method=method,dt=dt,steps=steps,**errors(p,exact),skipped_correction_cellsteps=skip))
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--controls',action='store_true');parser.add_argument('--fine',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);start=perf_counter();rows=[];bands={};fields={};checks=independent_checks()
    _,_,_,ground,_,residual=setup(257)
    spec=cases()
    if args.fine:spec={k:v for k,v in spec.items() if k in ('weak_envelope','strong_envelope','perturbed_core')}
    for case,c in spec.items():
        ns=c['ns'];dt=c['dt'];length=c['length'];methods=METHODS
        if args.controls:
            ns=ns[:1];dt/=2;methods=['fd_lap2','fd_lap2_mass','fd_lap4','fd_lap4_mass','gram_fft_fd4flux']
        if args.fine:
            ns=[256 if case=='perturbed_core' else 384];dt=.001 if case=='perturbed_core' else .00005
            methods=['global_fft','fd_lap4','fd_lap4_mass','gram_fft','gram_fft_fd4flux']
        steps=round(c['final']/dt)
        ref=reference(initial(case,c['nr'],ground),length,c['g'],dt,steps,ns)
        bands[case]=ref['band'];print('reference',case,flush=True)
        for n in ns:
            fields[f'{case}_{n}_reference']=ref['waves'][n][-1]
            for method in methods:
                combos=[('prescribed','spectral'),('self_consistent','spectral'),('self_consistent','fd2')]
                if args.controls:combos=[('self_consistent','spectral'),('self_consistent','fd2')]
                for branch,gravity in combos:
                    p,r=run(ref,n,length,c['g'],dt,method,branch,gravity);r['case']=case;rows.append(r)
                    fields[f'{case}_{n}_{method}_{branch}_{gravity}']=p
            print(case,n,'complete',round(perf_counter()-start,1),'s',flush=True)
        suffix='_controls' if args.controls else '_fine' if args.fine else ''
        result=dict(rows=rows,bandwidth=bands,stencil_checks=checks,ground_residual=residual,seconds=perf_counter()-start,
                    scope='Uniform periodic 1D wave grids. Same Strang splitting, initial samples, gravity, time steps for all wave methods at each grid. GAMER FD CPU half/full Taylor update and face-flux amplitude correction, default Taylor3=1/6, Laplacian2 or4. FFT evaluates FD symbols exactly, not spectral differentiation. No wave AMR, refluxing, density floors or cosmology. Gram-FE uses actual cached tables and no conservation by default. Gram-FE+FD4flux is experimental: same FD4 face flux/correction, with a separate Gram-FE half-step predictor. No global mass normalization. Profile is last-third time-average density folded about box center into 16 radial bins; not a 3D halo profile. Corrections skipped at nonpositive proposed/raw densities exactly as in source; skipped counts retained.')
        (OUT/f'results{suffix}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(OUT/f'fields{suffix}.npz',**fields)
    if not args.controls and not args.fine:
        result['free_wave']=free_wave_tests()
        sourcepaths=['src/Model_ELBDM/CPU_ELBDM/CPU_ELBDMSolver_FD.cpp','src/Model_ELBDM/ELBDM_SetTaylor3Coeff.cpp','src/Model_ELBDM/CPU_ELBDM/CPU_ELBDMSolver_GramFE_FFT.cpp']
        result['source_sha256']={p:hashlib.sha256((ROOT/'sources/gamer'/p).read_bytes()).hexdigest() for p in sourcepaths}
        result['gamer_commit']=json.loads((ROOT/'sources/gamer_provenance.json').read_text())['commit']
        (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('done',round(perf_counter()-start,1),'s',flush=True)


if __name__=='__main__':main()
