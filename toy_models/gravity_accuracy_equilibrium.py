"""Discrete equilibrium bias and short-time core response to gravity order."""
import json
from experiments import np,plt,inverse_laplacian,modes
from frozen_hamiltonian import setup
from gravity_accuracy_evolution import OUT,evolve,compare
from scipy.signal import resample,find_peaks
from scipy.optimize import root


def shape(p):
    x=np.arange(len(p))*20/len(p)-10;rho=abs(p)**2
    return dict(peak_density=float(max(rho)),rms_radius=float(np.sqrt(np.dot(x*x,rho)/sum(rho))))


def equilibrium(n,gravity,guess):
    h=20/n;k=modes(n,20);eye=np.fft.fft(np.eye(n),axis=0)
    kinetic=np.fft.ifft(.5*k[:,None]**2*eye,axis=0).real
    green=np.fft.ifft(inverse_laplacian(n,20,gravity)[:,None]*eye,axis=0).real
    p=guess.real;mu=float(np.dot(p,kinetic@p+(green@(p*p))*p)/np.dot(p,p))
    def residual(y):
        p=y[:-1];return np.r_[kinetic@p+(green@(p*p))*p-y[-1]*p,h*np.dot(p,p)-4]
    def jac(y):
        p=y[:-1];a=np.zeros((n+1,n+1));a[:-1,:-1]=kinetic+np.diag(green@(p*p)-y[-1])+2*p[:,None]*green*p[None,:]
        a[:-1,-1]=-p;a[-1,:-1]=2*h*p;return a
    answer=root(residual,np.r_[p,mu],jac=jac,tol=1e-11)
    error=float(np.linalg.norm(residual(answer.x)));assert error<1e-9
    return answer.x[:-1],float(answer.x[-1]),error


def initial(ground,n,case):
    p=resample(ground,n);x=np.arange(n)*20/n-10
    if case=='perturbed':
        p=p*np.exp(.03j*np.cos(2*np.pi*x/5))+.02*np.exp(-(x-3)**2/(2*1.2**2))*np.exp(3j*x)
        p*=np.sqrt(4/(20/n*np.sum(abs(p)**2)))
    return p


def main():
    _,_,_,ground,mu,residual=setup(257);_,_,_,checkground,_,_=setup(385)
    checks=dict(ground_residual=residual,ground_refinement=compare(ground,checkground)['density_error'])
    equilibria=[];rows=[];reference_stats={};final=12.;dt=.002
    for n in (32,48,64,96,128):
        target=resample(ground,n);baseline=shape(target)
        for gravity in ('fd2','fd4','spectral'):
            p,chemical,res=equilibrium(n,gravity,target)
            equilibria.append(dict(n=n,gravity=gravity,**compare(p,ground),residual=res,mu=chemical,
                peak_bias=shape(p)['peak_density']/baseline['peak_density']-1,
                radius_bias=shape(p)['rms_radius']/baseline['rms_radius']-1))
    for case in ('stationary','perturbed'):
        p0=initial(ground,512,case)
        ref,stats=evolve(p0,20,1,'spectral','global_fft',final,dt,order=4,samples=400,observe=shape)
        finer,_=evolve(initial(ground,1024,case),20,1,'spectral','global_fft',final,dt,order=4)
        half,_=evolve(p0,20,1,'spectral','global_fft',final,dt/2,order=4)
        checks[case+'_space']=compare(ref,finer)['density_error'];checks[case+'_time']=compare(ref,half)['density_error']
        reference_stats[case]=stats
        if case=='stationary':checks['stationary_analytic']=compare(ref,p0)['density_error']
        for n in (32,64,128):
            for gravity in ('fd2','fd4','spectral'):
                p,stats=evolve(initial(ground,n,case),20,1,gravity,'global_fft',final,dt,samples=400,observe=shape)
                history=stats['history'];radii=np.array([r['rms_radius'] for r in history]);times=np.array([r['time'] for r in history])
                peaks,_=find_peaks(radii,prominence=max(1e-8,float(np.ptp(radii))*.1),distance=20)
                period=float(np.mean(np.diff(times[peaks]))) if len(peaks)>=3 else None
                rows.append(dict(case=case,n=n,gravity=gravity,**compare(p,ref),**stats,
                    final_peak_bias=shape(p)['peak_density']/shape(resample(ref,n))['peak_density']-1,
                    final_radius_bias=shape(p)['rms_radius']/shape(resample(ref,n))['rms_radius']-1,
                    radius_peak_count=len(peaks),radius_period_if_three_peaks=period))
            print('core dynamics',case,n,flush=True)
        for gravity in ('fd2','fd4','spectral'):
            p,_=evolve(initial(ground,64,case),20,1,gravity,'global_fft',final,dt/2,observe=shape)
            checks[case+'_half_dt_'+gravity]=compare(p,ref)['density_error']
    assert checks['ground_refinement']<1e-8
    assert max(checks[c+'_space'] for c in ('stationary','perturbed'))<1e-7
    assert max(checks[c+'_time'] for c in ('stationary','perturbed'))<1e-7
    result=dict(equilibria=equilibria,rows=rows,checks=checks,references=reference_stats,
        scope='1D periodic SP mass=4, L=20. Stationary solutions use spectral kinetic energy and each discrete Poisson operator; same sampled continuum ground state initializes all dynamics. Perturbation adds a weak phase and an incoming coherent Gaussian packet. No 3D virial identity is assumed. Radius periods reported only if at least three sufficiently prominent peaks occur.')
    (OUT/'equilibrium.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for gravity in ('fd2','fd4','spectral'):
        rr=[r for r in equilibria if r['gravity']==gravity]
        axs[0].loglog([r['n'] for r in rr],[max(abs(r['radius_bias']),1e-15) for r in rr],'o-',label=gravity)
        row=next(r for r in rows if r['n']==64 and r['gravity']==gravity and r['case']=='stationary')
        hist=row['history'];base=hist[0]['rms_radius']
        axs[1].plot([r['time'] for r in hist],[r['rms_radius']/base-1 for r in hist],label=gravity)
    axs[0].set(xlabel='Grid cells',ylabel='Absolute relative equilibrium radius bias',title='Discrete equilibrium');axs[1].set(xlabel='Time',ylabel='Relative radius change',title='Start from the continuum core, N=64')
    for ax in axs:ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'equilibrium.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(checks=checks,equilibria=equilibria),indent=2))


if __name__=='__main__':main()
