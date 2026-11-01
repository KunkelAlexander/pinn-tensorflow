"""Nonuniform mapped SP and separated Fourier-band SP experiments.

These small dense diagnostics study representations, not parallel performance.
"""
from __future__ import annotations
import argparse
import json
import platform
from time import perf_counter

from experiments import (OUT, eigh, inverse_laplacian, mapped_operators, modes,
                         np, phase_error, plt, poisson, relative, resample, savefig,
                         scipy)
from scipy.integrate import solve_ivp
from scipy.special import iv


def breathing_initial(x):
    return np.exp(-x*x/(2*.55**2)) * np.exp(.15j*np.cos(2*x))


def uniform_breathing(n=1024, dt=.001, final=1.):
    length=20.; x=np.arange(n)*length/n-length/2
    psi=breathing_initial(x)*np.sqrt(4/(np.sqrt(np.pi)*.55))
    k=modes(n,length); inv=inverse_laplacian(n,length)
    a=1/(2-2**(1/3)); steps=round(final/dt); dt=final/steps
    stages=[(dt*f,np.exp(-.5j*dt*f*k*k)) for f in (a,1-2*a,a)]
    def potential(p): return np.fft.ifft(np.fft.fft(abs(p)**2)*inv).real
    for _ in range(steps):
        for h,drift in stages:
            psi*=np.exp(-.5j*h*potential(psi))
            psi=np.fft.ifft(drift*np.fft.fft(psi))
            psi*=np.exp(-.5j*h*potential(psi))
    return x,psi


def mapped_breathing(n,alpha,dt=.002,final=1.,method='yoshida4'):
    length=20.
    q,x,jac,h,a=mapped_operators(n,alpha,length)
    dq=2*np.pi/n
    state=np.sqrt(jac)*breathing_initial(x)*np.sqrt(4/(np.sqrt(np.pi)*.55))
    initial_mass=np.vdot(state,state).real*dq
    ke,kv=eigh(h); ae,av=eigh(a)
    # Symmetric pseudoinverse with the physical constant mode removed.
    pinv=(av[:,1:]/ae[None,1:])@av[:,1:].T
    def potential(u):
        rho=abs(u)**2/jac
        mean=np.sum(jac*rho)/np.sum(jac)
        p=pinv@(-jac*(rho-mean))
        return p-np.sum(jac*p)/np.sum(jac)
    def energy(u):
        return float((np.vdot(u,h@u).real+.5*np.sum(abs(u)**2*potential(u)))*dq)
    initial_energy=energy(state)
    c=1/(2-2**(1/3)); steps=round(final/dt); dt=final/steps
    factors=(1.,) if method=='strang' else (c,1-2*c,c)
    drifts=[(dt*f,(kv*np.exp(-1j*dt*f*ke)[None,:])@kv.T) for f in factors]
    start=perf_counter()
    for _ in range(steps):
        for step,drift in drifts:
            state*=np.exp(-.5j*step*potential(state))
            state=drift@state
            state*=np.exp(-.5j*step*potential(state))
    stats=dict(n=n,alpha=alpha,method=method,dt=dt,steps=steps,poisson_calls=2*len(factors)*steps,
               seconds=perf_counter()-start,
               initial_mass_error=float(abs(initial_mass-4)/4),
               mass_error=float(abs(np.vdot(state,state).real*dq-initial_mass)/initial_mass),
               relative_energy_drift=float(abs(energy(state)-initial_energy)/abs(initial_energy)),
               kinetic_radius=float(ke[-1]))
    return state,stats


def evaluate_mapped(state,alpha,xfine,length=20.):
    n=len(state); q=xfine*(2*np.pi/length)
    for _ in range(20):
        q-=(q-alpha*np.sin(q)-xfine*2*np.pi/length)/(1-alpha*np.cos(q))
    if np.max(abs(q-alpha*np.sin(q)-xfine*2*np.pi/length))>1e-12:
        raise RuntimeError('Coordinate inversion failed')
    coeff=np.fft.fft(state)/n
    u=np.exp(1j*np.outer(q+np.pi,modes(n,2*np.pi)))@coeff
    return u/np.sqrt(length/(2*np.pi)*(1-alpha*np.cos(q)))


def mapped_sp():
    xf,ref=uniform_breathing()
    _,check=uniform_breathing(1536,.0005)
    consistency=relative(resample(abs(ref)**2,len(check)),abs(check)**2)
    rows=[]; profiles={}
    for alpha in (0.,.4,.8):
        for n in (33,49,65,97,129,193):
            u,stats=mapped_breathing(n,alpha)
            pf=evaluate_mapped(u,alpha,xf)
            stats.update(density_error=relative(abs(pf)**2,abs(ref)**2),wave_error=phase_error(pf,ref))
            rows.append(stats)
            if n==49: profiles[str(alpha)]=abs(pf)**2
    times=[]
    for method in ('strang','yoshida4'):
        for dt in (.2,.1,.05,.025,.0125):
            u,stats=mapped_breathing(65,.8,dt=dt,method=method)
            pf=evaluate_mapped(u,.8,xf)
            stats.update(density_error=relative(abs(pf)**2,abs(ref)**2))
            times.append(stats)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for alpha in (0.,.4,.8):
        d=[r for r in rows if r['alpha']==alpha]
        axes[0].semilogy([r['n'] for r in d],[r['density_error'] for r in d],'o-',label=f'alpha={alpha}')
        axes[1].plot(xf,profiles[str(alpha)],label=f'49 points, alpha={alpha}')
    axes[1].plot(xf,abs(ref)**2,'k--',label='Reference'); axes[1].set_xlim(-3,3)
    axes[0].set_xlabel('Total points'); axes[0].set_ylabel('Relative density error at T=1')
    axes[1].set_xlabel('Physical coordinate'); axes[1].set_ylabel('Density')
    for ax in axes: ax.grid(alpha=.25); ax.legend(fontsize=8)
    savefig('mapped_sp.png')
    return dict(reference_disagreement=consistency,resolution=rows,time=times)


def band_coefficients(k,carrier=24):
    # exp(2 cos(x)) has Fourier coefficients I_m(2), known analytically.
    c=iv(np.abs(k-carrier),2)+.8j*iv(np.abs(k+carrier),2)
    # Normalize by the analytically known mean density, with cross overlap.
    norm=np.sqrt((1+.8**2)*iv(0,4))
    return np.asarray(c/norm,dtype=complex)


def band_rhs(k,coupling=20.,carrier=24):
    """Exact Fourier-Galerkin density convolution on an arbitrary mode set.

    Every represented cross-carrier interference term enters gravity. No
    coarse density grid and no local smoothing are used. O(M^2) here: an
    adaptive block implementation would need FFT convolutions and mode control.
    """
    differences=k[:,None]-k[None,:]
    offset=int(np.max(abs(differences))); index=(differences+offset).ravel()
    dk=np.arange(-offset,offset+1)
    inv=np.zeros(len(dk)); mask=dk!=0; inv[mask]=-coupling/dk[mask]**2
    kinetic=(k*k-carrier*carrier)/2
    def rhs(t,c):
        product=(c[:,None]*c[None,:].conj()).ravel()
        rhok=np.bincount(index,weights=product.real,minlength=len(dk))
        rhok=rhok+1j*np.bincount(index,weights=product.imag,minlength=len(dk))
        vk=rhok*inv
        return -1j*(kinetic*c+vk[differences+offset]@c)
    def energy(c):
        rhs_h=1j*rhs(0,c)
        # Restore the removed common carrier energy; half the nonlinear energy.
        return float(np.sum(k*k/2*abs(c)**2)+.5*np.vdot(c,rhs_h-kinetic*c).real)
    return rhs,energy


def uniform_band_reference(n=512,dt=.00025,final=.5,carrier=24,coupling=20.):
    k=modes(n,2*np.pi).round().astype(int)
    psi=np.fft.ifft(band_coefficients(k,carrier)*n)
    inv=inverse_laplacian(n,2*np.pi)*coupling
    a=1/(2-2**(1/3)); steps=round(final/dt); dt=final/steps
    stages=[(dt*f,np.exp(-.5j*dt*f*k*k)) for f in (a,1-2*a,a)]
    def potential(p): return np.fft.ifft(np.fft.fft(abs(p)**2)*inv).real
    for _ in range(steps):
        for h,drift in stages:
            psi*=np.exp(-.5j*h*potential(psi))
            psi=np.fft.ifft(drift*np.fft.fft(psi))
            psi*=np.exp(-.5j*h*potential(psi))
    return psi


def fourier_bands():
    carrier=24; final=.5
    ref=uniform_band_reference()
    check=uniform_band_reference(n=768,dt=.000125)
    consistency=relative(resample(abs(ref)**2,len(check)),abs(check)**2)
    xf=np.arange(len(ref))*2*np.pi/len(ref)
    rows=[]
    for width in (2,4,6,8,10,12,16,20):
        k=np.unique(np.r_[np.arange(-carrier-width,-carrier+width+1),np.arange(carrier-width,carrier+width+1)])
        initial=band_coefficients(k,carrier)
        rhs,energy=band_rhs(k,carrier=carrier)
        start=perf_counter()
        solution=solve_ivp(rhs,(0,final),initial,method='DOP853',rtol=2e-11,atol=2e-13)
        if not solution.success: raise RuntimeError(solution.message)
        coeff=solution.y[:,-1]
        # Restore the global phase removed by the carrier gauge.
        p=np.exp(1j*np.outer(xf,k))@coeff*np.exp(-.5j*carrier*carrier*final)
        rows.append(dict(half_bandwidth=width,carriers=[-carrier,carrier],complex_dofs=len(k),max_mode=int(max(abs(k))),
                         density_bandwidth=int(max(k)-min(k)),
                         density_error=relative(abs(p)**2,abs(ref)**2),
                         wave_error=phase_error(p,ref),
                         initial_missing_mass=float(abs(1-np.sum(abs(initial)**2))),
                         mass_error=float(abs(np.sum(abs(coeff)**2)-np.sum(abs(initial)**2))),
                         relative_energy_drift=float(abs(energy(coeff)-energy(initial))/abs(energy(initial))),
                         rhs_evaluations=solution.nfev,accepted_steps=len(solution.t)-1,
                         seconds=perf_counter()-start))
    # Test whether adding the first nonlinear satellite bands improves the
    # two-carrier truncation floor. These carry no appreciable initial mass.
    spawned=[]
    for halfwidth in (4,8,12):
        k=np.unique(np.concatenate([np.arange(c-halfwidth,c+halfwidth+1)
                     for c in (-3*carrier,-carrier,carrier,3*carrier)]))
        initial=band_coefficients(k,carrier)
        rhs,energy=band_rhs(k,carrier=carrier)
        start=perf_counter()
        solution=solve_ivp(rhs,(0,final),initial,method='DOP853',rtol=2e-11,atol=2e-13)
        if not solution.success: raise RuntimeError(solution.message)
        coeff=solution.y[:,-1]
        p=np.exp(1j*np.outer(xf,k))@coeff*np.exp(-.5j*carrier*carrier*final)
        spawned.append(dict(half_bandwidth=halfwidth,carriers=[-3*carrier,-carrier,carrier,3*carrier],
                            complex_dofs=len(k),density_error=relative(abs(p)**2,abs(ref)**2),
                            wave_error=phase_error(p,ref),rhs_evaluations=solution.nfev,
                            mass_error=float(abs(np.sum(abs(coeff)**2)-np.sum(abs(initial)**2))),
                            relative_energy_drift=float(abs(energy(coeff)-energy(initial))/abs(energy(initial))),
                            seconds=perf_counter()-start))
    plt.figure(figsize=(7,4))
    plt.semilogy([r['complex_dofs'] for r in rows],[r['density_error'] for r in rows],'o-')
    plt.xlabel('Complex coefficients in two separated Fourier bands')
    plt.ylabel('Relative density error at T=0.5');plt.grid(alpha=.25)
    savefig('fourier_bands.png')
    np.savez_compressed(OUT/'bands_reference.npz',x=xf,psi=ref)
    return dict(carrier=carrier,coupling=20.,final_time=final,reference_n=512,
                reference_disagreement=consistency,rows=rows,spawned_bands=spawned,
                limitations='Prescribed separated bands; not an adaptive algorithm; dense O(M^2) convolution; few-stream test only')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',choices=['mapped_sp','bands'],nargs='+')
    args=parser.parse_args()
    for name,job in [('mapped_sp',mapped_sp),('bands',fourier_bands)]:
        if args.only and name not in args.only: continue
        start=perf_counter(); result=job()
        payload=dict(experiment=name,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                     elapsed_seconds=perf_counter()-start,results=result)
        (OUT/f'{name}.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(f'{name}: {payload["elapsed_seconds"]:.2f}s',flush=True)


if __name__=='__main__': main()
