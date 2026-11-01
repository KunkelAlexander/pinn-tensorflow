"""Frozen 2D coherent source: exact Fourier oracle versus FD2 and nested patches.

All wave and density modes are resolved even on the smallest root grid.
This is a controlled point-centered model, not a compiled GAMER reproduction.
"""
import json
from pathlib import Path
from time import perf_counter
from experiments import np, plt
from scipy.signal import convolve2d
from gravity_hierarchy_probe import solver, interpolation

OUT = Path(__file__).resolve().parent / 'frozen_gravity_results'
OUT.mkdir(exist_ok=True)
L = 2*np.pi


def coefficients(seed):
    rng = np.random.default_rng(seed)
    waves = np.zeros((19,19),complex)
    for mx,my in [(9,0),(-9,0),(0,9),(0,-9),(6,6),(-6,6),(6,-6),(-6,-6)]:
        waves[mx+9,my+9] = np.exp(1j*rng.uniform(0,2*np.pi))/np.sqrt(8)
    envelope = np.outer([.325,1,.325],[.325,1,.325])
    psi = convolve2d(waves,envelope)
    psi /= np.linalg.norm(psi)
    rho = convolve2d(psi,psi[::-1,::-1].conj())
    rho[20,20] = 0  # periodic mean subtraction
    k = np.arange(-20,21)
    q2 = k[:,None]**2+k[None,:]**2
    phi = np.zeros_like(rho); mask = q2>0
    phi[mask] = -rho[mask]/q2[mask]
    return psi,rho,phi


def evaluate(c,x,y):
    k = np.arange(-(len(c)//2),len(c)//2+1)
    return np.exp(1j*np.outer(x,k))@c@np.exp(1j*np.outer(k,y))


def coefficients_on_grid(c,n):
    k = np.arange(-(len(c)//2),len(c)//2+1)
    a = np.zeros((n,n),complex)
    # Accumulate aliases, including both signs of Nyquist if present.
    np.add.at(a,(np.broadcast_to(k[:,None]%n,c.shape),np.broadcast_to(k[None,:]%n,c.shape)),c)
    return a


def continuum_norms(delta,reference):
    k = np.arange(-20,21);q2 = k[:,None]**2+k[None,:]**2
    return {name:float(np.sqrt(np.sum(abs(delta)**2*q2**p)/np.sum(abs(reference)**2*q2**p)))
            for name,p in [('potential',0),('force',1),('tidal',2)]}


def discrete_hessian(a,h):
    xx=(a[2:,1:-1]-2*a[1:-1,1:-1]+a[:-2,1:-1])/h**2
    yy=(a[1:-1,2:]-2*a[1:-1,1:-1]+a[1:-1,:-2])/h**2
    xy=(a[2:,2:]-a[2:,:-2]-a[:-2,2:]+a[:-2,:-2])/(4*h*h)
    return np.stack([xx,yy,np.sqrt(2)*xy])


def patch_solve(x,boundary,rho):
    n=len(x)-1;h=x[1]-x[0];a,solve=solver(n)
    rhs=h*h*evaluate(rho,x[1:-1],x[1:-1]).real
    rhs[0]-=boundary[0,1:-1];rhs[-1]-=boundary[-1,1:-1]
    rhs[:,0]-=boundary[1:-1,0];rhs[:,-1]-=boundary[1:-1,-1]
    result=boundary.copy();result[1:-1,1:-1]=solve(rhs.ravel()).reshape(n-1,n-1)
    residual=np.linalg.norm(a@result[1:-1,1:-1].ravel()-rhs.ravel())/max(np.linalg.norm(rhs),1e-30)
    return result,float(residual)


def main():
    start=perf_counter();uniform=[];nested=[];alias=[];profiles=[];checks={}
    for seed in (7,19,41):
        psi,rho,phi=coefficients(seed)
        for n in (50,100,200):
            h=L/n;k=np.arange(-20,21)
            x=np.arange(n)*h
            sampled=evaluate(psi,x,x)
            density=evaluate(rho,x,x).real
            checks[f'density_identity_{seed}_{n}']=float(np.max(abs(density-(abs(sampled)**2-1))))
            q2=k[:,None]**2+k[None,:]**2
            d=(2*np.sin(k*h/2)/h)**2;denom=d[:,None]+d[None,:]
            fd=np.zeros_like(phi);mask=q2>0;fd[mask]=-rho[mask]/denom[mask]
            # Independent FFT solution checks the modal calculation.
            modes=2*np.pi*np.fft.fftfreq(n,d=h);symbol=(2*np.sin(modes*h/2)/h)**2
            denominator=symbol[:,None]+symbol[None,:];green=np.zeros_like(denominator)
            green[denominator>0]=-1/denominator[denominator>0]
            numerical=np.fft.ifft2(np.fft.fft2(density)*green).real
            checks[f'fd_fft_identity_{seed}_{n}']=float(np.max(abs(numerical-evaluate(fd,x,x).real)))
            metrics=continuum_norms(fd-phi,phi)
            # Tracefree continuum Hessian is the tidal shear; full Hessian also reported.
            uniform.append(dict(seed=seed,n=n,wave_ppw=n/10,density_ppw=n/20,**metrics))
            # The cubic instantaneous SP RHS has bandwidth 3K, although rho has 2K.
            rhs=convolve2d(phi,psi);folded=coefficients_on_grid(rhs,n)
            band=np.arange(-30,31);keep=(band>=-n//2)&(band<(n+1)//2)
            projected=coefficients_on_grid(rhs*keep[:,None]*keep[None,:],n)
            direct=np.fft.fft2(evaluate(phi,x,x)*sampled)/n**2
            checks[f'rhs_alias_identity_{seed}_{n}']=float(np.max(abs(folded-direct)))
            alias.append(dict(seed=seed,n=n,wave_ppw=n/10,
                aliased_rhs_relative=float(np.linalg.norm(folded-projected)/np.linalg.norm(rhs)),
                rhs_outside_grid_relative=float(np.linalg.norm(rhs*(1-keep[:,None]*keep[None,:]))/np.linalg.norm(rhs))))
            if n==200:continue
            for root_symbol,degree in [('fd2',2),('spectral',2),('fd2',4),('spectral',4)]:
                xp=np.arange(n)*h-L/2
                parent=evaluate(fd if root_symbol=='fd2' else phi,xp,xp).real
                for level in range(1,7):
                    width=L/2**level;xc=np.linspace(-width/2,width/2,n+1);hc=xc[1]-xc[0]
                    transfer=interpolation(xp,xc,degree,periodic=level==1)
                    boundary=transfer@parent@transfer.T
                    actual,residual=patch_solve(xc,boundary,rho)
                    truth=evaluate(phi,xc,xc).real
                    oracle,_=patch_solve(xc,truth,rho)
                    ha=discrete_hessian(actual,hc);ht=discrete_hessian(truth,hc);ho=discrete_hessian(oracle,hc)
                    # Omit only edge derivatives; compare identical diagnostics on oracle and numerical potential.
                    scale=np.linalg.norm(ht)
                    inherited=ha-ho
                    trace=inherited[0]+inherited[1]
                    shear=inherited.copy();shear[0]-=trace/2;shear[1]-=trace/2
                    center=(n-2)//2
                    center_scale=np.linalg.norm(ht[:,center,center])
                    nested.append(dict(seed=seed,n=n,root_wave_ppw=n/10,level=level,root_symbol=root_symbol,degree=degree,
                        tidal_error=float(np.linalg.norm(ha-ht)/scale),
                        exact_boundary_tidal_error=float(np.linalg.norm(ho-ht)/scale),
                        inherited_shear=float(np.linalg.norm(shear)/scale),
                        center_hessian_error=float(np.linalg.norm((ha-ht)[:,center,center])/center_scale),
                        center_exact_boundary_error=float(np.linalg.norm((ho-ht)[:,center,center])/center_scale),
                        inherited_trace_max=float(np.max(abs(trace))),residual=residual))
                    if seed==7 and n==50 and root_symbol=='fd2' and degree==2 and level in (1,6):
                        profiles.append(dict(level=level,x=xc[1:-1].tolist(),rho=evaluate(rho,xc[1:-1],xc[1:-1]).real.tolist(),
                            shear_error=np.sqrt(np.sum(shear**2,axis=0)).tolist()))
                    xp=xc;parent=actual
            print(f'frozen source seed={seed}, root={n} complete',flush=True)
    modal=[dict(wave_ppw=p,density_ppw=p/2,
                fd2_potential_bias=float((2*np.pi/p/np.sin(2*np.pi/p))**2-1)) for p in (5,10,20)]
    assert max(checks.values())<1e-11,checks
    assert max(r['residual'] for r in nested)<1e-10
    assert max(r['inherited_trace_max'] for r in nested)<2e-6
    assert max(r['aliased_rhs_relative'] for r in alias if r['n']>=100)<1e-15
    result=dict(scope='Frozen periodic 2D interference with eight coherent carriers and a smooth envelope; exact finite Fourier density and potential. No time integration, cosmological halo, or GAMER binary. Nested model uses exact fine sources, direct point-centered FD2 solves and polynomial boundary transfer. Hessian errors use identical finite-difference diagnostics on numerical and reference potentials, not an additional force stencil comparison.',
        normalization='L=2*pi; mean density=1; Delta Phi=rho-1. Axis bandwidth K=10 in psi, 20 in density, 30 in instantaneous Phi*psi. The diagonal carriers have smaller Euclidean wave number than K. All source modes resolved at n=50.',
        uniform=uniform,nested=nested,alias=alias,modal=modal,checks=checks,seconds=perf_counter()-start)
    (OUT/'frozen_gravity.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for name in ('potential','force','tidal'):
        for seed in (7,19,41):
            rr=[r for r in uniform if r['seed']==seed]
            axs[0].loglog([r['wave_ppw'] for r in rr],[r[name] for r in rr],'o-',label=name if seed==7 else None,alpha=.8)
    axs[0].set(xlabel='Cells per shortest wave wavelength',ylabel='Relative RMS error',title='Uniform FD2 gravity');axs[0].legend()
    for n in (50,100):
        for sym in ('fd2','spectral'):
            rr=[r for r in nested if r['seed']==7 and r['n']==n and r['root_symbol']==sym and r['degree']==2]
            axs[1].semilogy([r['level'] for r in rr],[r['tidal_error'] for r in rr],'o-',label=f'{sym}, root {n/10:g} cells/wave')
    rr=[r for r in nested if r['seed']==7 and r['n']==50 and r['root_symbol']=='fd2' and r['degree']==2]
    axs[1].semilogy([r['level'] for r in rr],[r['exact_boundary_tidal_error'] for r in rr],'k--',label='Exact boundaries, root 50')
    axs[1].set(xlabel='Refinement level',ylabel='Relative Hessian error',title='Fixed density, nested gravity');axs[1].legend(fontsize=7)
    p=profiles[-1];im=axs[2].imshow(np.array(p['shear_error']).T,origin='lower',extent=[p['x'][0],p['x'][-1]]*2)
    fig.colorbar(im,ax=axs[2]);axs[2].set(title='Inherited shear error at level 6',xlabel='x',ylabel='y')
    for ax in axs[:2]:ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'frozen_gravity.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(uniform=uniform,modal=modal,alias=alias,seconds=result['seconds']),indent=2))


if __name__=='__main__':main()
