"""Galilean-boost stress test of the previously favored fixed reference H.

Periodic integer boosts of a converged stationary SP state give an independent
continuum reference by translation. No timestep-refined reference integrator is
needed. Dense exponentials are only accuracy oracles; no speedup is claimed.
"""
import json
from conditioning_review import OUT, savefig
from experiments import np, plt, modes, relative, phase_error
from frozen_hamiltonian import setup, propagate
from scipy.signal import resample


def exact_moving(ground,mu,x,boost,time,length=20.):
    velocity=2*np.pi*boost/length
    shifted=np.fft.ifft(np.fft.fft(ground)*np.exp(-1j*modes(len(x),length)*velocity*time))
    return shifted*np.exp(1j*velocity*x-1j*(mu+.5*velocity**2)*time)


def main():
    x,kinetic,green,ground,mu,residual=setup(129)
    xf,_,_,groundf,muf,residualf=setup(257)
    final=2.; rows=[]; spatial=[]
    for boost in (0,2,6,12):
        initial=ground*np.exp(1j*2*np.pi*boost*x/20)
        ref=exact_moving(ground,mu,x,boost,final)
        fine=exact_moving(groundf,muf,xf,boost,final)
        spatial.append(dict(boost=boost,density_error=relative(resample(abs(ref)**2,len(xf)),abs(fine)**2),
                            wave_error=phase_error(resample(ref,len(xf)),fine)))
        for method in ('strang','yoshida4','reference_strang','reference_yoshida4'):
            for steps in (4,8,16,32,64):
                psi,stats=propagate(initial,kinetic,green,final/steps,steps,method)
                rows.append(dict(boost=boost,velocity=2*np.pi*boost/20,method=method,steps=steps,
                                 density_error=relative(abs(psi)**2,abs(ref)**2),
                                 wave_error=phase_error(psi,ref),**stats))
        print(f'Completed boost {boost}',flush=True)
    # An independent small-step convergence check against the boosted exact state.
    boost=12; initial=ground*np.exp(1j*2*np.pi*boost*x/20)
    ref=exact_moving(ground,mu,x,boost,final)
    accurate,_=propagate(initial,kinetic,green,.002,1000,'yoshida4')
    check=relative(abs(accurate)**2,abs(ref)**2)
    assert max(r['density_error'] for r in spatial)<1e-9
    assert max(r['wave_error'] for r in spatial)<1e-8
    assert check<1e-8
    assert max(r['mass_error'] for r in rows)<1e-11
    # Halving dt on the genuinely moving case must recover expected orders.
    for method,lo,hi in [('strang',3.5,4.5),('yoshida4',12.,20.),
                         ('reference_strang',3.,5.),('reference_yoshida4',10.,22.)]:
        es=[next(r['density_error'] for r in rows if r['boost']==6 and r['method']==method and r['steps']==s) for s in (32,64)]
        assert lo<es[0]/es[1]<hi,(method,es)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for method in ('strang','yoshida4','reference_strang','reference_yoshida4'):
        subset=[r for r in rows if r['method']==method and r['steps']==32]
        axes[0].semilogy([r['velocity'] for r in subset],[max(r['density_error'],1e-15) for r in subset],'o-',label=method)
        subset=[r for r in rows if r['method']==method and r['boost']==6]
        axes[1].loglog([r['steps'] for r in subset],[r['density_error'] for r in subset],'o-',label=method)
    axes[0].set(xlabel='Core velocity',ylabel='Density error at T=2, 32 steps')
    axes[1].set(xlabel='Steps to T=2',ylabel='Density error, velocity = 6 pi / 10')
    for ax in axes: ax.grid(alpha=.25);ax.legend(fontsize=8)
    savefig('moving_core.png')
    result=dict(n=129,reference_n=257,final_time=final,stationary_residual=residual,
                fine_stationary_residual=residualf,spatial_reference=spatial,
                small_step_boost12_density_error=check,rows=rows,
                limitation='One periodic 1D core; analytic continuum boost of a spatially converged numerical stationary state; dense exponential actions')
    (OUT/'moving_core.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(spatial_reference=spatial,small_step_check=check,
                         selected=[r for r in rows if r['steps']==32]),indent=2))


if __name__=='__main__': main()
