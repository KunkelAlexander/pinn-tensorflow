"""Compare splitting and self-consistent full-H exponentials near equilibrium.

Dense eigensolves expose time-discretization error only. Their timings do not
predict a 3D matrix-free implementation. Failed nonlinear iterations are recorded.
"""
from __future__ import annotations
import json
from time import perf_counter

from experiments import OUT, eigh, inverse_laplacian, modes, np, phase_error, plt, relative, savefig
from scipy.signal import resample
from scipy.optimize import root


def setup(n=97,length=20.,mass=4.):
    x=np.arange(n)*length/n-length/2; dx=length/n; k=modes(n,length)
    fft_eye=np.fft.fft(np.eye(n),axis=0)
    kinetic=np.fft.ifft((k*k/2)[:,None]*fft_eye,axis=0).real
    green=np.fft.ifft(inverse_laplacian(n,length)[:,None]*fft_eye,axis=0).real
    psi=np.exp(-x*x/2)
    psi*=np.sqrt(mass/(np.dot(psi,psi)*dx))
    for _ in range(1800):
        phi=green@(psi*psi)
        psi*=np.exp(-.01*phi)
        psi=np.fft.ifft(np.exp(-.01*k*k)*np.fft.fft(psi)).real
        psi*=np.exp(-.01*(green@(psi*psi)))
        psi*=np.sqrt(mass/(np.dot(psi,psi)*dx))
    mu=float(np.dot(psi,kinetic@psi+(green@(psi*psi))*psi)/np.dot(psi,psi))
    def equations(y):
        p=y[:-1]; mu=y[-1]
        return np.r_[kinetic@p+(green@(p*p))*p-mu*p, dx*np.dot(p,p)-mass]
    def jacobian(y):
        p=y[:-1]; mu=y[-1]
        j=np.zeros((n+1,n+1))
        j[:-1,:-1]=kinetic+np.diag(green@(p*p)-mu)+2*p[:,None]*green*p[None,:]
        j[:-1,-1]=-p; j[-1,:-1]=2*dx*p
        return j
    solution=root(equations,np.r_[psi,mu],jac=jacobian,tol=1e-11)
    residual=np.linalg.norm(equations(solution.x))
    if residual>1e-9: raise RuntimeError(f'Stationary solution residual {residual}; {solution.message}')
    return x,kinetic,green,solution.x[:-1].astype(complex),float(solution.x[-1]),float(residual)


def propagate(psi,kinetic,green,dt,steps,method):
    psi=psi.copy(); ke,kv=eigh(kinetic); old=psi.copy()
    iterations=0; exponentials=0; max_fixed_point_residual=0.; failures=0
    a=1/(2-2**(1/3))
    factors=(1.,) if method in ('strang','reference_strang') else (a,1-2*a,a)
    split=method in ('strang','yoshida4','reference_strang','reference_yoshida4')
    # Frozen reference contains the large cancelling kinetic and gravity terms.
    # Only the potential defect is kicked. This remains the original nonlinear
    # equation: H = H_ref + diag(Phi[psi] - Phi_ref).
    phi_ref=green@abs(psi)**2 if method.startswith('reference_') else np.zeros(len(psi))
    if method.startswith('reference_'):
        ke,kv=eigh(kinetic+np.diag(phi_ref))
    if split:
        stages=[(dt*f,(kv*np.exp(-1j*dt*f*ke)[None,:])@kv.T) for f in factors]
    def action(p,phi,h):
        nonlocal exponentials
        e,v=eigh(kinetic+np.diag(phi)); exponentials+=1
        return v@(np.exp(-1j*h*e)*(v.T@p))
    start=perf_counter()
    for _ in range(steps):
        if split:
            for h,drift in stages:
                psi*=np.exp(-.5j*h*(green@abs(psi)**2-phi_ref))
                psi=drift@psi
                psi*=np.exp(-.5j*h*(green@abs(psi)**2-phi_ref))
        else:
            phi=green@abs(psi)**2
            if method=='frozen_start':
                psi=action(psi,phi,dt)
                continue
            limit=1 if method=='frozen_start' else 50
            for iteration in range(limit):
                midpoint=action(psi,phi,dt/2)
                updated=green@abs(midpoint)**2
                error=np.linalg.norm(updated-phi)/max(1.,np.linalg.norm(updated))
                iterations+=1
                if method=='frozen_start': break
                phi=updated
                if error<1e-11: break
            if method!='frozen_start':
                max_fixed_point_residual=max(max_fixed_point_residual,float(error))
                if error>=1e-11: failures+=1
            psi=action(psi,phi,dt)
    return psi,dict(seconds=perf_counter()-start,exponentials=exponentials,
                    nonlinear_iterations=iterations,unconverged_steps=failures,
                    max_fixed_point_residual=max_fixed_point_residual,
                    mass_error=float(abs(np.vdot(psi,psi).real-np.vdot(old,old).real)/np.vdot(old,old).real))


def main():
    x,kinetic,green,ground,mu,residual=setup()
    _,_,_,fine_ground,_,_=setup(n=193)
    spatial_disagreement=relative(resample(abs(ground)**2,len(fine_ground)),abs(fine_ground)**2)
    final=2.; cases=[]
    for case in ('stationary','perturbed'):
        initial=ground.copy()
        if case=='perturbed': initial*=np.exp(.2j*np.cos(2*np.pi*x/4))
        if case=='stationary': reference=ground*np.exp(-1j*mu*final)
        else:
            reference,_=propagate(initial,kinetic,green,.001,2000,'yoshida4')
            check,_=propagate(initial,kinetic,green,.0005,4000,'yoshida4')
            reference_disagreement=relative(abs(reference)**2,abs(check)**2)
        for method in ('strang','yoshida4','reference_strang','reference_yoshida4','frozen_start','exponential_midpoint'):
            for steps in (1,2,4,8,16,32):
                psi,stats=propagate(initial,kinetic,green,final/steps,steps,method)
                cases.append(dict(case=case,method=method,steps=steps,dt=final/steps,
                                  density_error=relative(abs(psi)**2,abs(reference)**2),
                                  phase_aligned_error=phase_error(psi,reference),**stats))
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,case in zip(axes,('stationary','perturbed')):
        for method in ('strang','yoshida4','reference_strang','reference_yoshida4','frozen_start','exponential_midpoint'):
            d=[r for r in cases if r['case']==case and r['method']==method]
            ax.loglog([r['steps'] for r in d],[max(r['density_error'],1e-15) for r in d],'o-',label=method)
        ax.set_title(case);ax.set_xlabel('Steps to T=2');ax.grid(alpha=.25)
    axes[0].set_ylabel('Relative density error');axes[0].legend(fontsize=8)
    savefig('frozen_hamiltonian.png')
    result=dict(n=len(x),length=20.,mass=4.,ground_mu=mu,stationary_residual=residual,
                ground_spatial_density_disagreement=spatial_disagreement,
                perturbed_reference_disagreement=reference_disagreement,rows=cases,
                scope='Same spatial operator for all methods; dense exponential oracle, not a scalable timing comparison')
    (OUT/'frozen.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('frozen Hamiltonian experiments saved',flush=True)


if __name__=='__main__': main()
